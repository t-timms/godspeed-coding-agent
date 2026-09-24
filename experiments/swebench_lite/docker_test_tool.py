"""Agent-in-loop tool: run the SWE-Bench Docker harness from inside the agent session.

Scoped to one SWE-Bench instance via constructor closure — the agent sees a
no-argument `swebench_verify_patch` tool; `instance_id`, `model_name`, and
`workdir` are baked in at registration time.

Usage (from `run_in_loop.py` per instance):

    from experiments.swebench_lite.docker_test_tool import SWEBenchVerifyTool

    registry.register(
        SWEBenchVerifyTool(
            instance_id="sqlfluff__sqlfluff-2419",
            model_name="nvidia_nim/moonshotai/kimi-k2.5",
            workdir=Path("experiments/swebench_lite"),
            split="dev",
        )
    )

Each call to the tool captures the current working-tree diff, hashes it,
and either:
  - returns a cached verdict if the hash matches the previous call
    (no-edit short-circuit — keeps a stuck agent from burning budget);
  - invokes `verify_patch.verify_patch()` and returns
    `resolved=<bool>\\n\\n<test_output_tail>`.

Budget: `MAX_VERIFY_CALLS` per instance (hard ceiling `HARD_VERIFY_CAP`).
Once exhausted the tool returns a failure result without invoking the
harness.

Each harness call spins a fresh container - ~60-90 seconds per
invocation. Agents should be prompted to call it sparingly, typically
only after they believe the fix is complete.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

# verify_patch lives in the same directory; make it importable whether
# this tool is loaded as a script or via the experiments package.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from verify_patch import verify_patch  # noqa: E402

logger = logging.getLogger(__name__)

MAX_OUTPUT_TAIL_CHARS = 2000
MAX_VERIFY_CALLS = 5
HARD_VERIFY_CAP = 8


class SWEBenchVerifyTool(Tool):
    """Run the SWE-Bench test harness on the agent's current edits.

    Bound to a single instance at construction. The agent only sees
    the tool's name/description/schema — the instance id and model name
    are closed over at registration time so the agent cannot spoof them.

    Tracks per-instance call count and the last diff SHA. When called
    with an unchanged working tree, returns the cached verdict instead
    of re-running the harness. When the call count exceeds
    `HARD_VERIFY_CAP`, further calls return failure without any harness
    invocation.
    """

    def __init__(
        self,
        instance_id: str,
        model_name: str,
        workdir: Path,
        split: str = "dev",
        timeout_s: int = 900,
        max_calls: int = MAX_VERIFY_CALLS,
        hard_cap: int = HARD_VERIFY_CAP,
    ) -> None:
        self.instance_id = instance_id
        self.model_name = model_name
        self.workdir = workdir
        self.split = split
        self.timeout_s = timeout_s
        self.max_calls = max_calls
        self.hard_cap = hard_cap
        self._call_count = 0
        self._last_diff_sha: str | None = None
        self._last_result: tuple[bool, str] | None = None

    @property
    def name(self) -> str:
        return "swebench_verify_patch"

    @property
    def description(self) -> str:
        return (
            "Run the SWE-Bench test harness on your current edits. "
            "This runs the failing test (and related tests) inside a Docker "
            "container configured for this instance. Returns whether the "
            "instance is now resolved plus a tail of the test output. "
            f"Each call takes ~60-90 seconds and you have a budget of "
            f"{self.max_calls} calls per instance - use sparingly, typically only "
            "after you believe your fix is complete. If your working tree "
            "is unchanged since the last call, the cached verdict is returned."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        diff = _capture_diff(context.cwd)
        if not diff.strip():
            return ToolResult.success(
                "resolved=False\n\n"
                "(working tree has no changes - nothing to verify. "
                "Make your edits first, then call this tool.)"
            )

        diff_sha = hashlib.sha1(diff.encode("utf-8"), usedforsecurity=False).hexdigest()

        if self._last_diff_sha == diff_sha and self._last_result is not None:
            resolved, test_output = self._last_result
            logger.info(
                "swebench_verify_patch: instance=%s cached (no edits since last call)",
                self.instance_id,
            )
            tail = test_output[-MAX_OUTPUT_TAIL_CHARS:]
            return ToolResult.success(
                f"resolved={resolved}\n\n"
                "(cached verdict - working tree is unchanged since the "
                "previous verify call. Make edits before calling again.)\n\n"
                f"{tail}"
            )

        if self._call_count >= self.hard_cap:
            return ToolResult.failure(
                f"verify budget exhausted ({self._call_count}/{self.hard_cap} calls). "
                "Make your best final edit and stop; no more harness runs."
            )

        if self._call_count >= self.max_calls:
            logger.warning(
                "swebench_verify_patch: instance=%s soft cap exceeded (%d/%d)",
                self.instance_id,
                self._call_count + 1,
                self.max_calls,
            )

        self._call_count += 1
        logger.info(
            "swebench_verify_patch: instance=%s call=%d/%d diff_lines=%d",
            self.instance_id,
            self._call_count,
            self.max_calls,
            diff.count("\n"),
        )
        try:
            resolved, test_output = verify_patch(
                instance_id=self.instance_id,
                model_name=self.model_name,
                model_patch=diff,
                workdir=self.workdir,
                timeout_s=self.timeout_s,
                split=self.split,
            )
        except subprocess.TimeoutExpired as e:
            return ToolResult.failure(
                f"harness timed out after {e.timeout}s - the container may be "
                f"slow or the test may hang. Consider narrowing your edit."
            )
        except Exception as exc:
            logger.exception("verify_patch raised")
            return ToolResult.failure(f"harness invocation failed: {exc}")

        self._last_diff_sha = diff_sha
        self._last_result = (resolved, test_output)

        tail = test_output[-MAX_OUTPUT_TAIL_CHARS:]
        return ToolResult.success(f"resolved={resolved}\n\n{tail}")


def _capture_diff(cwd: Path) -> str:
    """Capture the current working-tree diff.

    Matches ``run.py``'s ``_capture_patch`` retry logic: on Windows the
    default git config may produce empty diffs for CRLF-normalized files,
    so we retry with ``core.autocrlf=false --text`` if the first attempt
    is empty but ``status --porcelain`` shows staged / unstaged changes.
    """
    result = subprocess.run(
        ["git", "diff"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diff = result.stdout

    if diff.strip():
        return diff

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if not status.stdout.strip():
        return ""

    # Dirty tree but empty diff — retry with CRLF normalization disabled
    retry = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "diff", "--text"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return retry.stdout
