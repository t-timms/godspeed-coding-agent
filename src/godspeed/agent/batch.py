"""Parallel worktree-based task decomposition and dispatch (/batch).

A Godspeed-scale take on Claude Code's ``/batch``: decompose a goal into
independent units, run each unit as a sub-agent inside its own isolated git
worktree, and collect results.

Architecture
------------
- ``decompose_task`` tries a structure-prompted single LLM call (JSON schema
  validated, one retry); on any failure it falls back to the deterministic
  ``chunk_plan`` (explicit numbered list in the goal text, else one unit).
- ``WorktreeBatchRunner`` creates one ``git worktree`` per unit under a
  managed root (a sibling of the working directory — git refuses worktrees
  inside the main working tree), dispatches each unit through
  ``AgentCoordinator.spawn`` with a worktree-scoped ``ToolContext``, waits
  with a per-unit timeout, captures a patch, and removes the worktree on
  success (or leaves it and reports the path on failure).

PR automation
-------------
- ``create_pr`` invokes the ``gh`` CLI (``gh pr create --head <branch>
  --title <unit-derived> --body <summary+patchdiff info>``) guarded by gh
  presence (``shutil.which``), the ``open_pr`` flag, and a branch existing
  in the worktree. Failures are per-unit and never abort the batch — each
  unit records a clear note in its result summary and ``pr_messages``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from godspeed.agent.coordinator import AgentCoordinator, SubAgentConfig
from godspeed.llm.client import LLMClient

logger = logging.getLogger(__name__)

# -- Module constants ------------------------------------------------------

MAX_BATCH_UNITS = 30
MIN_BATCH_UNITS = 1
DEFAULT_BATCH_UNITS = 5
DEFAULT_PARALLELISM = 5
UNIT_TIMEOUT_SECONDS = 600.0
GIT_TIMEOUT_SECONDS = 30
DECOMPOSE_RETRIES = 1  # one retry on schema validation failure
WORKTREE_ROOT_SUFFIX = ".godspeed-batch"

#: Cap on gh stderr lines captured into a per-unit PR note (gh is verbose).
GH_STDERR_CAP_LINES = 5

#: Matches explicit numbered list items: "1. foo", "2) bar".
_NUMBERED_ITEM_RE = re.compile(r"^\s*(?:\d+)[.)]\s+(.+)$", re.MULTILINE)

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a task decomposition engine. Break the user's goal into independent,
parallelizable units of work. Each unit runs in its own isolated git worktree,
so it must be fully self-contained and must not depend on other units.

Respond with ONLY a JSON object in this exact schema:
{
  "units": [
    {
      "id": "u1",
      "title": "short title",
      "instructions": "detailed, self-contained instructions",
      "depends_on": []
    }
  ]
}

Rules:
- Between 1 and 30 units.
- ids must be unique, non-empty strings like "u1", "u2".
- depends_on must reference other unit ids; leave it empty for independent units.
- instructions must be self-contained (the unit runs in an isolated worktree).
"""


class BatchGitError(Exception):
    """Raised when the working directory is not safe for batch execution."""


@dataclass(frozen=True)
class BatchUnit:
    """A single independent unit of work in a batch plan."""

    id: str
    title: str
    instructions: str
    depends_on: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchPlan:
    """A decomposed plan: a goal plus its independent units."""

    goal: str
    units: list[BatchUnit]


@dataclass
class UnitResult:
    """Outcome of a single batch unit."""

    id: str
    ok: bool
    summary: str
    patch_available: bool = False
    patch_path: Path | None = None
    worktree_path: Path | None = None
    timed_out: bool = False
    pr_url: str | None = None


@dataclass(frozen=True)
class PrResult:
    """Outcome of a PR creation attempt for one unit.

    ``ok`` is True when ``gh pr create`` succeeded and returned a URL;
    ``url_or_error`` holds the PR URL on success or a capped error note.
    """

    ok: bool
    url_or_error: str


class GhRunner(Protocol):
    """Injected runner for the gh CLI (testable without a real gh binary)."""

    def run(self, args: list[str], cwd: Path) -> tuple[int, str, str]:
        """Run gh with *args* in *cwd*; return (returncode, stdout, stderr)."""
        ...


class UnitExecutor(Protocol):
    """Executes one batch unit inside its worktree."""

    async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
        """Run the unit; return a summary string."""
        ...


def validate_plan(plan: BatchPlan) -> None:
    """Validate plan invariants; raises ValueError on violation.

    Enforces: units non-empty, non-empty unique ids, titles/instructions
    present, dependencies resolvable, and the 30-unit cap.
    """
    if not plan.units:
        raise ValueError("Batch plan must contain at least one unit")
    if len(plan.units) > MAX_BATCH_UNITS:
        raise ValueError(
            f"Batch plan exceeds cap of {MAX_BATCH_UNITS} units (got {len(plan.units)})"
        )
    ids = [unit.id for unit in plan.units]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate unit ids: {duplicates}")
    for unit in plan.units:
        if not unit.id:
            raise ValueError("Unit id must be non-empty")
        if not unit.title:
            raise ValueError(f"Unit {unit.id!r} must have a title")
        if not unit.instructions:
            raise ValueError(f"Unit {unit.id!r} must have instructions")
        for dep in unit.depends_on:
            if dep not in ids:
                raise ValueError(f"Unit {unit.id!r} depends on unknown unit {dep!r}")


def _short_title(text: str, limit: int = 80) -> str:
    """First line of *text*, truncated to *limit* characters."""
    stripped = text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    if len(first_line) > limit:
        return first_line[: limit - 3] + "..."
    return first_line


def _parse_numbered_items(goal: str) -> list[str]:
    """Extract explicit numbered list items from the goal text."""
    items: list[str] = []
    for match in _NUMBERED_ITEM_RE.finditer(goal):
        text = match.group(1).strip()
        if text:
            items.append(text)
    return items


def chunk_plan(goal: str, num_units_hint: int = DEFAULT_BATCH_UNITS) -> BatchPlan:
    """Deterministic LLM-free decomposition.

    Splits by explicit numbered list in the goal text; falls back to a
    single unit covering the whole goal. Raises ValueError when the goal is
    empty, the hint exceeds the cap, or the parsed items exceed the cap.
    """
    if not goal.strip():
        raise ValueError("Goal must be non-empty")
    if num_units_hint > MAX_BATCH_UNITS:
        raise ValueError(f"num_units_hint exceeds cap of {MAX_BATCH_UNITS} (got {num_units_hint})")

    items = _parse_numbered_items(goal)
    if items:
        units = [
            BatchUnit(id=f"u{idx}", title=_short_title(item), instructions=item)
            for idx, item in enumerate(items, start=1)
        ]
    else:
        units = [BatchUnit(id="u1", title=_short_title(goal), instructions=goal)]

    plan = BatchPlan(goal=goal, units=units)
    validate_plan(plan)
    return plan


def _extract_json_object(content: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from an LLM response."""
    if not content:
        return None
    for block in re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL):
        try:
            data = json.loads(block.strip())
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(content[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            return None
        if isinstance(data, dict):
            return data
    return None


def _parse_decompose_response(content: str, goal: str) -> BatchPlan | None:
    """Parse and schema-validate an LLM decomposition response.

    Returns None when the response is not valid JSON or fails schema
    validation — callers fall back to the deterministic chunk plan.
    """
    data = _extract_json_object(content)
    if data is None:
        return None
    raw_units = data.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        return None

    units: list[BatchUnit] = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            return None
        unit_id = raw.get("id")
        title = raw.get("title")
        instructions = raw.get("instructions")
        depends_on = raw.get("depends_on", [])
        if not isinstance(unit_id, str) or not unit_id:
            return None
        if not isinstance(title, str) or not title:
            return None
        if not isinstance(instructions, str) or not instructions:
            return None
        if not isinstance(depends_on, list) or not all(isinstance(dep, str) for dep in depends_on):
            return None
        units.append(
            BatchUnit(
                id=unit_id,
                title=title,
                instructions=instructions,
                depends_on=list(depends_on),
            )
        )

    try:
        plan = BatchPlan(goal=goal, units=units)
        validate_plan(plan)
        return plan
    except ValueError:
        return None


async def _llm_decompose(
    goal: str,
    num_units_hint: int,
    llm_client: LLMClient,
) -> BatchPlan | None:
    """Model-driven decomposition with schema validation and one retry.

    Returns None when the LLM is unavailable or the response fails schema
    validation after the retry — callers fall back to ``chunk_plan``.
    """
    prompt = (
        f"Goal: {goal}\n\n"
        f"Decompose into approximately {num_units_hint} independent units "
        f"(between {MIN_BATCH_UNITS} and {MAX_BATCH_UNITS})."
    )
    for attempt in range(DECOMPOSE_RETRIES + 1):
        try:
            response = await llm_client.chat(
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                task_type="plan",
            )
            plan = _parse_decompose_response(response.content, goal)
            if plan is not None:
                return plan
            logger.warning("batch.decompose invalid_schema attempt=%d", attempt)
        except Exception as exc:
            logger.warning("batch.decompose llm_error attempt=%d error=%s", attempt, exc)
    return None


async def decompose_task(
    goal: str,
    num_units_hint: int = DEFAULT_BATCH_UNITS,
    llm_client: LLMClient | None = None,
) -> BatchPlan:
    """Decompose a goal into a validated batch plan.

    Uses the model-driven decomposition when an LLM client is available;
    falls back to the deterministic chunk plan on any failure. Raises
    ValueError for an empty goal or a hint above the 30-unit cap.
    """
    if not goal.strip():
        raise ValueError("Goal must be non-empty")
    if num_units_hint > MAX_BATCH_UNITS:
        raise ValueError(f"num_units_hint exceeds cap of {MAX_BATCH_UNITS} (got {num_units_hint})")
    if llm_client is not None:
        plan = await _llm_decompose(goal, num_units_hint, llm_client)
        if plan is not None:
            return plan
        logger.info("batch.decompose falling_back_to_chunk_plan")
    return chunk_plan(goal, num_units_hint)


def default_worktree_root(working_dir: Path) -> Path:
    """Managed worktree root: a sibling of the working directory.

    git refuses to add worktrees inside the main working tree, so the
    managed root lives next to the repo (e.g. ``myrepo.godspeed-batch/``).
    """
    return working_dir.parent / f"{working_dir.name}{WORKTREE_ROOT_SUFFIX}"


def _safe_unit_path(unit_id: str) -> str:
    """Sanitize a unit id for use as a directory/branch name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit_id)
    return safe or "unit"


def _git_output(cwd: Path, *args: str) -> str:
    """Run a git command and return stdout; raises BatchGitError on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        raise BatchGitError(f"git {' '.join(args)} failed: {exc}") from exc
    if result.returncode != 0:
        raise BatchGitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _git_ok(cwd: Path, *args: str) -> bool:
    """Run a git command; return True on success, False on any failure."""
    try:
        _git_output(cwd, *args)
        return True
    except BatchGitError:
        return False


def _run_gh(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run the gh CLI via subprocess; returns (returncode, stdout, stderr).

    Never raises: subprocess failures (missing binary, timeout, OS error)
    are folded into a nonzero returncode with the error text on stderr.
    """
    try:
        result = subprocess.run(
            ["gh", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 1, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def _cap_stderr(stderr: str, limit: int = GH_STDERR_CAP_LINES) -> str:
    """Cap gh stderr at *limit* non-empty lines for per-unit notes."""
    lines = [line for line in stderr.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[:limit]) + f"\n... ({len(lines) - limit} more lines)"


def _current_branch(worktree_path: Path) -> str | None:
    """Current branch of the worktree, or None when detached/unavailable."""
    try:
        branch = _git_output(worktree_path, "branch", "--show-current").strip()
    except BatchGitError:
        return None
    return branch or None


def _pr_title(unit: BatchUnit) -> str:
    """PR title derived from the unit (falls back to the unit id)."""
    return _short_title(unit.title, limit=72) or f"batch: {unit.id}"


def _pr_body(unit: BatchUnit, result: UnitResult) -> str:
    """PR body: unit summary plus patchdiff location info."""
    body = f"Unit: {unit.id}\n\n{result.summary}"
    if result.patch_path is not None:
        body += f"\n\nPatch file: {result.patch_path}"
    return body


def create_pr(
    unit_id: str,
    worktree_path: Path,
    *,
    branch: str | None = None,
    title: str | None = None,
    body: str | None = None,
    gh_runner: GhRunner | None = None,
) -> PrResult:
    """Create a GitHub PR for a unit's branch via the gh CLI.

    Pure helper: no runner state, no side effects beyond the gh subprocess.
    Guards, in order: gh present (``shutil.which``), a branch resolvable in
    the worktree. Returns ``PrResult(ok=True, url_or_error=<pr url>)`` on
    success; on any failure returns ``PrResult(ok=False, url_or_error=<reason>)``
    — never raises, so a PR failure can never abort a batch.

    Args:
        unit_id: The batch unit id (used for the default title/body).
        worktree_path: The unit's worktree; gh runs with this as cwd.
        branch: Branch to open the PR from. Defaults to the worktree's
            current branch.
        title: PR title. Defaults to a unit-derived title.
        body: PR body. Defaults to a unit summary + patchdiff note.
        gh_runner: Injected gh runner (testable). Defaults to ``_run_gh``.
    """
    if gh_runner is None:
        gh_runner = _run_gh
    if shutil.which("gh") is None:
        return PrResult(ok=False, url_or_error="gh CLI not available")
    if branch is None:
        branch = _current_branch(worktree_path)
    if not branch:
        return PrResult(ok=False, url_or_error="no branch found in worktree")
    if not _git_ok(worktree_path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"):
        return PrResult(ok=False, url_or_error=f"branch {branch!r} not found in worktree")
    returncode, stdout, stderr = gh_runner.run(
        [
            "pr",
            "create",
            "--head",
            branch,
            "--title",
            title or _pr_title_default(unit_id),
            "--body",
            body or _pr_body_default(unit_id),
        ],
        cwd=worktree_path,
    )
    if returncode != 0:
        detail = _cap_stderr(stderr) or f"gh pr create exited with code {returncode}"
        return PrResult(ok=False, url_or_error=detail)
    url = stdout.strip()
    if not url:
        return PrResult(ok=False, url_or_error="gh pr create returned no URL")
    return PrResult(ok=True, url_or_error=url)


def _pr_title_default(unit_id: str) -> str:
    """Fallback PR title when the caller does not supply one."""
    return f"batch: {unit_id}"


def _pr_body_default(unit_id: str) -> str:
    """Fallback PR body when the caller does not supply one."""
    return f"Unit: {unit_id}\n\nPatch available in worktree."


class CoordinatorUnitExecutor:
    """Default executor: spawns a sub-agent scoped to the unit's worktree.

    Reuses ``AgentCoordinator.spawn``'s execution path (Conversation +
    agent_loop) with a per-unit ``ToolContext`` whose cwd is the worktree.
    """

    def __init__(
        self,
        coordinator: AgentCoordinator,
        config: SubAgentConfig | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._config = config

    async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
        base_ctx = self._coordinator._tool_context
        unit_ctx = base_ctx.model_copy(update={"cwd": worktree_path})
        return await self._coordinator.spawn(
            unit.instructions,
            depth=0,
            config=self._config,
            tool_context=unit_ctx,
        )


class WorktreeBatchRunner:
    """Runs a BatchPlan across isolated git worktrees with a parallelism cap.

    Git-safety: refuses to run when the working directory is not a git repo
    or has uncommitted changes (unless ``allow_dirty``). Never touches the
    index or existing branches — only per-unit worktrees and their branches.
    """

    def __init__(
        self,
        working_dir: Path,
        coordinator: AgentCoordinator,
        parallelism: int = DEFAULT_PARALLELISM,
        unit_timeout: float = UNIT_TIMEOUT_SECONDS,
        allow_dirty: bool = False,
        keep_worktrees: bool = False,
        worktree_root: Path | None = None,
        executor: UnitExecutor | None = None,
        *,
        open_pr: bool = False,
    ) -> None:
        self._working_dir = Path(working_dir).resolve()
        self._coordinator = coordinator
        self._parallelism = max(1, parallelism)
        self._unit_timeout = unit_timeout
        self._allow_dirty = allow_dirty
        self._keep_worktrees = keep_worktrees
        self._worktree_root = (worktree_root or default_worktree_root(self._working_dir)).resolve()
        self._executor = executor or CoordinatorUnitExecutor(coordinator)
        self._open_pr = open_pr
        self._gh_runner: GhRunner | None = None
        self._run_id = uuid.uuid4().hex[:8]
        self._run_dir = self._worktree_root / self._run_id
        self._created_worktrees: list[Path] = []
        self._created_branches: list[str] = []
        self.pr_messages: list[str] = []

    @property
    def run_dir(self) -> Path:
        """Managed directory holding this run's worktrees and patches."""
        return self._run_dir

    def check_repo_state(self) -> None:
        """Refuse to run when cwd is not a git repo or the tree is dirty."""
        if not _git_ok(self._working_dir, "rev-parse", "--is-inside-work-tree"):
            raise BatchGitError(
                f"Not a git repository: {self._working_dir}. /batch requires a git repo."
            )
        if not self._allow_dirty:
            status = _git_output(self._working_dir, "status", "--porcelain")
            if status.strip():
                raise BatchGitError(
                    "Working tree has uncommitted changes. Commit or stash them, "
                    "or pass allow_dirty=True."
                )

    async def run(self, plan: BatchPlan, open_prs: bool = False) -> list[UnitResult]:
        """Execute all units in isolated worktrees and collect results.

        Args:
            plan: The validated batch plan to execute.
            open_prs: When True, attempt best-effort PR creation for units
                with a captured patch. ORs with the constructor's ``open_pr``.

        Returns:
            One UnitResult per unit, in plan order.
        """
        validate_plan(plan)
        self.check_repo_state()

        # Fresh run identity per call so re-runs never collide.
        self._run_id = uuid.uuid4().hex[:8]
        self._run_dir = self._worktree_root / self._run_id
        self._created_worktrees = []
        self._created_branches = []
        self.pr_messages = []

        self._run_dir.mkdir(parents=True, exist_ok=True)
        patches_dir = self._run_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        effective_open_pr = open_prs or self._open_pr
        semaphore = asyncio.Semaphore(self._parallelism)

        async def _run_unit(unit: BatchUnit) -> UnitResult:
            async with semaphore:
                return await self._run_one(unit, patches_dir, effective_open_pr)

        return list(await asyncio.gather(*(_run_unit(u) for u in plan.units)))

    async def _run_one(
        self,
        unit: BatchUnit,
        patches_dir: Path,
        open_prs: bool,
    ) -> UnitResult:
        safe_id = _safe_unit_path(unit.id)
        worktree_path = self._run_dir / safe_id
        branch = f"batch/{self._run_id}/{safe_id}"
        logger.info("batch.unit_start id=%s worktree=%s", unit.id, worktree_path)

        try:
            self._add_worktree(worktree_path, branch)
        except BatchGitError as exc:
            logger.error("batch.worktree_add_failed id=%s error=%s", unit.id, exc)
            return UnitResult(id=unit.id, ok=False, summary=f"worktree add failed: {exc}")

        self._created_worktrees.append(worktree_path)
        self._created_branches.append(branch)

        try:
            summary = await asyncio.wait_for(
                self._executor.execute(unit, worktree_path),
                timeout=self._unit_timeout,
            )
        except TimeoutError:
            logger.warning("batch.unit_timeout id=%s timeout=%s", unit.id, self._unit_timeout)
            return UnitResult(
                id=unit.id,
                ok=False,
                summary=f"Unit timed out after {self._unit_timeout:g}s",
                worktree_path=worktree_path,
                timed_out=True,
            )
        except Exception as exc:
            logger.error("batch.unit_error id=%s error=%s", unit.id, exc, exc_info=True)
            return UnitResult(
                id=unit.id,
                ok=False,
                summary=f"Unit failed: {exc}",
                worktree_path=worktree_path,
            )

        patch_path = self._capture_patch(worktree_path, patches_dir / f"{safe_id}.patch")
        patch_available = patch_path is not None
        result = UnitResult(
            id=unit.id,
            ok=True,
            summary=summary,
            patch_available=patch_available,
            patch_path=patch_path,
            worktree_path=worktree_path,
        )
        # PR creation runs while the worktree and its branch still exist;
        # a missing pushed branch is a per-unit skip, never a batch abort.
        if open_prs:
            self._maybe_open_pr(unit, result, branch)
        leftover = None if self._keep_worktrees else self._remove_worktree(worktree_path, branch)
        if not self._keep_worktrees:
            result.worktree_path = leftover
        logger.info("batch.unit_done id=%s ok=True", unit.id)
        return result

    def _add_worktree(self, worktree_path: Path, branch: str) -> None:
        """Create a per-unit worktree on a fresh branch at HEAD."""
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        _git_output(
            self._working_dir,
            "worktree",
            "add",
            "-b",
            branch,
            str(worktree_path),
            "HEAD",
        )

    def _capture_patch(self, worktree_path: Path, patch_path: Path) -> Path | None:
        """Capture the unit's diff (tracked + untracked) into a patch file.

        Uses ``git add -N`` (intent-to-add) so untracked files appear in the
        diff, then resets the index so the worktree is left untouched. Returns
        the patch path when there are changes, else None.
        """
        try:
            _git_output(worktree_path, "add", "-N", ".")
            diff = _git_output(worktree_path, "diff", "HEAD")
            _git_output(worktree_path, "reset", "-q")
        except BatchGitError:
            return None
        if not diff.strip():
            return None
        patch_path.write_text(diff, encoding="utf-8")
        return patch_path

    def _remove_worktree(self, worktree_path: Path, branch: str) -> Path | None:
        """Remove a worktree and its branch; returns the path if removal failed."""
        try:
            _git_output(self._working_dir, "worktree", "remove", "--force", str(worktree_path))
        except BatchGitError as exc:
            logger.warning("batch.worktree_remove_failed path=%s error=%s", worktree_path, exc)
            return worktree_path
        try:
            _git_output(self._working_dir, "branch", "-D", branch)
        except BatchGitError as exc:
            logger.warning("batch.branch_delete_failed branch=%s error=%s", branch, exc)
        return None

    def _maybe_open_pr(self, unit: BatchUnit, result: UnitResult, branch: str) -> None:
        """Best-effort PR creation for a unit with a captured patch.

        Guards: patch available, gh present, branch existing in the worktree.
        Never raises — failures are recorded as per-unit notes in the result
        summary and in ``pr_messages``, and the batch continues.
        """
        if not result.patch_available or result.patch_path is None:
            return
        worktree_path = result.worktree_path
        if worktree_path is None:
            message = (
                f"PR skipped for unit {unit.id}: worktree unavailable; "
                f"patch saved at {result.patch_path}"
            )
            self.pr_messages.append(message)
            result.summary = f"{result.summary}\n[pr] {message}"
            logger.info("batch.pr_skipped unit=%s reason=worktree_unavailable", unit.id)
            return
        pr = create_pr(
            unit.id,
            worktree_path,
            branch=branch,
            title=_pr_title(unit),
            body=_pr_body(unit, result),
            gh_runner=self._gh_runner,
        )
        if pr.ok:
            result.pr_url = pr.url_or_error
            message = f"PR opened for unit {unit.id}: {pr.url_or_error}"
            self.pr_messages.append(message)
            logger.info("batch.pr_opened unit=%s url=%s", unit.id, pr.url_or_error)
        else:
            message = f"PR skipped for unit {unit.id}: {pr.url_or_error}"
            self.pr_messages.append(message)
            result.summary = f"{result.summary}\n[pr] {message}"
            logger.info("batch.pr_skipped unit=%s reason=%s", unit.id, pr.url_or_error)

    def cleanup(self) -> list[Path]:
        """Force-remove all worktrees created by this runner and their branches.

        Returns a list of worktree paths that could not be removed.
        """
        leftover: list[Path] = []
        for worktree_path, branch in zip(
            self._created_worktrees, self._created_branches, strict=True
        ):
            if not worktree_path.exists():
                continue
            try:
                _git_output(self._working_dir, "worktree", "remove", "--force", str(worktree_path))
            except BatchGitError as exc:
                logger.warning("batch.cleanup_worktree_failed path=%s error=%s", worktree_path, exc)
                leftover.append(worktree_path)
                continue
            try:
                _git_output(self._working_dir, "branch", "-D", branch)
            except BatchGitError as exc:
                logger.warning("batch.cleanup_branch_failed branch=%s error=%s", branch, exc)
        with contextlib.suppress(BatchGitError):
            _git_output(self._working_dir, "worktree", "prune")
        shutil.rmtree(self._run_dir, ignore_errors=True)
        return leftover
