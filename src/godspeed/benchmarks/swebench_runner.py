"""Godspeed SWE-bench runner — Verified + Lite with NIM key rotation.

Integrates the existing experiments/swebench_lite/run.py with the NIM key
rotation manager for rate-limited free-tier API access. Supports:
- Parallel execution across multiple API keys
- Instance cooldown management
- Resume from checkpoint
- Metrics tracking
- sb-cli compatible prediction output
- Heartbeat logging for long runs (every 10 instances)
- Per-instance timeout enforcement (prevents agent hangs)
- Crash recovery via per-instance append-only predictions

Usage:
    # Via Python module
    python -m godspeed.benchmarks.swebench_runner \\
        --model nvidia_nim/deepseek-ai/deepseek-v4-pro \\
        --split test \\
        --instances 300 \\
        --parallel 4

    # Offline dry-run (no network, no LLM) with a local instances file:
    python -m godspeed.benchmarks.swebench_runner \\
        --model nvidia_nim/deepseek-ai/deepseek-v4-pro \\
        --split test \\
        --dry-run \\
        --instances-file path/to/instances.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from godspeed.benchmarks.nim_key_rotation import NIMKeyManager

logger = logging.getLogger(__name__)

PER_TASK_TIMEOUT_S = 600
INSTANCE_COOLDOWN_S = 5
MAX_ITERATIONS = 40
DEFAULT_TOOL_SET = "local"
HEARTBEAT_INTERVAL = 10  # log heartbeat every N instances

InstanceLoader = Callable[[str], list[dict]]


def load_instances_from_file(path: str | Path) -> InstanceLoader:
    """Return an instance loader that reads SWE-bench instances from a local JSONL file.

    Each line must be a JSON object with at least ``instance_id``, ``repo``,
    ``base_commit`` and ``problem_statement``.  The returned loader ignores
    the ``split`` argument (the file already selects the instances).
    """

    def _loader(split: str) -> list[dict]:  # noqa: ARG001 - interface contract
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Instances file not found: {p}")
        instances: list[dict] = []
        # utf-8-sig tolerates a UTF-8 BOM (PowerShell 5.1 Set-Content writes one)
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                inst = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in instances file {p}: {e}") from e
            if not isinstance(inst, dict):
                raise ValueError(
                    f"Instances file {p} must contain JSON objects, got {type(inst).__name__}"
                )
            instances.append(inst)
        return instances

    return _loader


# ---------------------------------------------------------------------------
# Structured per-run logging
# ---------------------------------------------------------------------------


class RunLogger:
    """Writes structured per-run logs: main log, failures log, summary JSON."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.main_log = self.log_dir / "main.log"
        self.failures_log = self.log_dir / "failures.log"
        self.summary_path = self.log_dir / "summary.json"
        self._t_start = time.monotonic()
        self._instance_times: list[float] = []

    def log_instance_start(self, idx: int, total: int, instance_id: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        msg = f"[{ts}] START [{idx}/{total}] {instance_id}"
        with open(self.main_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if idx > 1 and idx % HEARTBEAT_INTERVAL == 0:
            elapsed = time.monotonic() - self._t_start
            rate = elapsed / idx
            remaining = rate * (total - idx)
            with open(self.main_log, "a", encoding="utf-8") as f:
                f.write(
                    f"[{ts}] HEARTBEAT [{idx}/{total}] elapsed={elapsed:.0f}s "
                    f"rate={rate:.1f}s/inst remaining={remaining:.0f}s\n"
                )

    def log_instance_result(
        self,
        instance_id: str,
        status: str,
        patch_lines: int,
        metrics: dict,
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        msg = (
            f"[{ts}] DONE  {instance_id} status={status} "
            f"patch={patch_lines} lines cost=${metrics.get('cost_usd', 0):.4f}"
        )
        with open(self.main_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if status != "ok":
            with open(self.failures_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

    def write_summary(self, summary: dict) -> None:
        elapsed = time.monotonic() - self._t_start
        summary["wall_s"] = round(elapsed, 1)
        summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def _run_one_instance(
    inst: dict,
    model: str,
    split: str,
    out_dir: Path,
    per_task_timeout: int,
    max_iterations: int,
    agent_in_loop: bool,
    include_hints: bool,
    architect: bool,
    allow_web_search: bool,
    tool_set: str,
    nim_key_manager: NIMKeyManager | None,
    run_log: RunLogger,
    idx: int,
    total: int,
) -> tuple[dict, str]:
    """Run a single SWE-bench instance. Returns (metrics_dict, model_patch)."""
    from experiments.swebench_lite.run import (
        ARCHITECT_BLOCK,
        DEFAULT_PROMPT_TEMPLATE,
        HINTS_BLOCK_TEMPLATE,
        IN_LOOP_BLOCK,
        _capture_patch,
        _prepare_repo,
    )

    iid = inst["instance_id"]
    repo = inst["repo"]
    base = inst["base_commit"]

    run_log.log_instance_start(idx, total, iid)

    metrics: dict = {
        "instance_id": iid,
        "repo": repo,
        "base_commit": base,
        "model": model,
        "status": "ok",
        "cost_usd": 0.0,
        "patch_lines": 0,
        "patch_nonempty": False,
    }

    workspace = Path(tempfile.mkdtemp(prefix=f"swebench-{iid}-"))
    t_instance = time.monotonic()

    try:
        if nim_key_manager:
            api_key = await nim_key_manager.get_key()
            os.environ["NVIDIA_NIM_API_KEY"] = api_key

        try:
            _prepare_repo(repo, base, workspace)
        except Exception as e:  # noqa: BLE001
            metrics["status"] = "clone_error"
            metrics["error"] = str(e)[:400]
            run_log.log_instance_result(iid, metrics["status"], 0, metrics)
            return metrics, ""

        hints_block = ""
        if include_hints and inst.get("hints_text"):
            hints_block = HINTS_BLOCK_TEMPLATE.format(hints_text=inst["hints_text"].strip())
        architect_block = ARCHITECT_BLOCK if architect else ""
        in_loop_block = IN_LOOP_BLOCK if agent_in_loop else ""

        prompt = DEFAULT_PROMPT_TEMPLATE.format(
            problem_statement=inst["problem_statement"],
            hints_block=hints_block,
            architect_block=architect_block,
            in_loop_block=in_loop_block,
        )

        patch = ""
        try:
            if agent_in_loop:
                import sys as _sys

                _sys.path.insert(
                    0,
                    str(
                        Path(__file__).resolve().parent.parent.parent
                        / "experiments"
                        / "swebench_lite"
                    ),
                )
                from run_in_loop import run_one as _run_one_in_loop

                godspeed_payload: dict = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: _run_one_in_loop(
                            instance_id=iid,
                            model=model,
                            prompt=prompt,
                            project_dir=workspace,
                            split=split,
                            timeout_s=per_task_timeout,
                            verify_workdir=out_dir.resolve(),
                            max_iterations=max_iterations,
                            tool_set="full" if allow_web_search else tool_set,
                        ),
                    ),
                    timeout=per_task_timeout + 60,
                )
            else:
                import sys as _sys2

                _sys2.path.insert(
                    0,
                    str(
                        Path(__file__).resolve().parent.parent.parent
                        / "experiments"
                        / "swebench_lite"
                    ),
                )
                from run import _run_godspeed

                godspeed_payload = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: _run_godspeed(model, prompt, workspace, per_task_timeout),
                    ),
                    timeout=per_task_timeout + 60,
                )

            patch = _capture_patch(base, workspace)
            metrics["agent_in_loop"] = agent_in_loop
            metrics["verify_call_count"] = godspeed_payload.get("verify_call_count", 0)
            metrics["iterations_used"] = godspeed_payload.get("iterations_used")
            metrics["exit_reason"] = godspeed_payload.get("exit_reason")
            metrics["cost_usd"] = godspeed_payload.get("cost_usd", 0.0)

            if nim_key_manager and api_key:
                await nim_key_manager.report_success(api_key)

        except TimeoutError:
            metrics["status"] = "timeout"
            metrics["error"] = f"Instance timed out after {per_task_timeout + 60}s"
            logger.warning("[%d/%d] %s TIMEOUT", idx, total, iid)
        except Exception as e:  # noqa: BLE001
            metrics["status"] = "agent_error"
            metrics["error"] = str(e)[:400]
            logger.warning("[%d/%d] %s ERROR: %s", idx, total, iid, str(e)[:120])

        metrics["patch_lines"] = len(patch.splitlines()) if patch else 0
        metrics["patch_nonempty"] = bool(patch.strip() if patch else "")
        metrics["wall_s"] = round(time.monotonic() - t_instance, 1)

    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        run_log.log_instance_result(iid, metrics["status"], metrics["patch_lines"], metrics)

    patch_out = patch if patch else ""
    return metrics, patch_out


async def run_swebench(
    *,
    model: str,
    split: str = "test",
    instances: int | None = None,
    instance_ids: list[str] | None = None,
    out: Path | None = None,
    metrics_path: Path | None = None,
    parallel: int = 1,
    agent_in_loop: bool = True,
    per_task_timeout: int = PER_TASK_TIMEOUT_S,
    instance_cooldown: int = INSTANCE_COOLDOWN_S,
    max_iterations: int = MAX_ITERATIONS,
    include_hints: bool = False,
    architect: bool = False,
    resume: bool = True,
    allow_web_search: bool = False,
    tool_set: str = DEFAULT_TOOL_SET,
    nim_key_manager: NIMKeyManager | None = None,
    log_dir: Path | None = None,
    dry_run: bool = False,
    instance_loader: InstanceLoader | None = None,
) -> dict:
    """Run Godspeed against SWE-bench instances with NIM key rotation.

    Returns a summary dict with {total, resolved, errors, cost_usd, wall_s}.

    Parameters
    ----------
    dry_run:
        When ``True``, validate configuration and compute the instance plan
        without invoking any LLM, cloning any repo, or downloading any
        dataset.  The dataset loader is still called unless ``instance_loader``
        is provided; pass a stub to keep the dry-run fully offline.
    instance_loader:
        Injectable ``(split) -> list[dict]`` dataset loader.  ``None`` uses
        the real ``experiments.swebench_lite.run._load_instances`` (network).
    """
    from experiments.swebench_lite.run import _already_predicted, _filter

    if instance_loader is None:
        from experiments.swebench_lite.run import _load_instances

        instance_loader = _load_instances

    out_dir = Path("benchmarks/results") if out is None else out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out or out_dir / f"predictions_{split}.jsonl"
    metrics_path = metrics_path or out_dir / f"metrics_{split}.jsonl"

    run_log = RunLogger(log_dir or out_dir / "logs" / time.strftime("run_%Y-%m-%d_%H"))

    instances_list = instance_loader(split)
    instances_list = _filter(instances_list, instance_ids, instances)
    already = _already_predicted(predictions_path) if resume else set()
    to_run = [i for i in instances_list if i["instance_id"] not in already]

    logger.info(
        "split=%s total=%d to_run=%d already=%d parallel=%d agent_in_loop=%s",
        split,
        len(instances_list),
        len(to_run),
        len(already),
        parallel,
        agent_in_loop,
    )

    if nim_key_manager is None:
        try:
            nim_key_manager = NIMKeyManager.from_env()
        except ValueError:
            nim_key_manager = None
    if nim_key_manager:
        logger.info("NIM key rotation: %d keys active", nim_key_manager.key_count)
    else:
        logger.info("NIM key rotation: not configured (using single key from env)")

    summary: dict = {
        "total": len(to_run),
        "resolved": 0,
        "errors": 0,
        "timeouts": 0,
        "cost_usd": 0.0,
        "wall_s": 0.0,
        "model": model,
        "split": split,
    }

    if dry_run:
        summary["dry_run"] = True
        summary["instance_ids"] = [i["instance_id"] for i in to_run]
        summary["nim_keys_configured"] = nim_key_manager.key_count if nim_key_manager else 0
        summary["predictions_path"] = str(predictions_path)
        summary["metrics_path"] = str(metrics_path)
        summary["log_dir"] = str(run_log.log_dir)
        logger.info(
            "DRY-RUN: would run %d instances (model=%s split=%s parallel=%d)",
            len(to_run),
            model,
            split,
            parallel,
        )
        return summary

    t_start = time.monotonic()

    for idx, inst in enumerate(to_run, 1):
        if idx > 1 and parallel == 1 and instance_cooldown > 0:
            await asyncio.sleep(instance_cooldown)

        metrics, patch = await _run_one_instance(
            inst=inst,
            model=model,
            split=split,
            out_dir=out_dir,
            per_task_timeout=per_task_timeout,
            max_iterations=max_iterations,
            agent_in_loop=agent_in_loop,
            include_hints=include_hints,
            architect=architect,
            allow_web_search=allow_web_search,
            tool_set=tool_set,
            nim_key_manager=nim_key_manager,
            run_log=run_log,
            idx=idx,
            total=len(to_run),
        )

        prediction = {
            "instance_id": metrics["instance_id"],
            "model_name_or_path": model,
            "model_patch": patch,
        }

        with open(predictions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")

        summary["cost_usd"] += metrics.get("cost_usd", 0.0)
        if metrics["status"] != "ok":
            summary["errors"] += 1
            if metrics["status"] == "timeout":
                summary["timeouts"] += 1

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    summary["wall_s"] = round(time.monotonic() - t_start, 1)
    run_log.write_summary(summary)

    logger.info(
        "done. total=%d errors=%d timeouts=%d cost=$%.4f wall=%ss",
        summary["total"],
        summary["errors"],
        summary.get("timeouts", 0),
        summary["cost_usd"],
        summary["wall_s"],
    )
    logger.info(
        "predictions=%s metrics=%s log_dir=%s", predictions_path, metrics_path, run_log.log_dir
    )
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LiteLLM model id")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--instances", type=int, default=None, help="Cap on number of instances")
    parser.add_argument("--instance-ids", nargs="*", default=None, help="Specific instance IDs")
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=None,
        help="Local JSONL file of SWE-bench instances. When set, the dataset "
        "loader is replaced with this file (fully offline).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Predictions output path")
    parser.add_argument("--metrics", type=Path, default=None, help="Metrics output path")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel instances")
    parser.add_argument("--agent-in-loop", action="store_true", default=True)
    parser.add_argument("--no-agent-in-loop", dest="agent_in_loop", action="store_false")
    parser.add_argument("--per-task-timeout", type=int, default=PER_TASK_TIMEOUT_S)
    parser.add_argument("--instance-cooldown", type=int, default=INSTANCE_COOLDOWN_S)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--include-hints", action="store_true")
    parser.add_argument("--architect", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--allow-web-search", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the instance plan without running any "
        "instance, cloning repos, or calling an LLM. Requires a dataset loader; "
        "pass --instance-ids with a local loader to stay fully offline.",
    )
    args = parser.parse_args()

    instance_loader = None
    if args.instances_file is not None:
        instance_loader = load_instances_from_file(args.instances_file)

    summary = asyncio.run(
        run_swebench(
            model=args.model,
            split=args.split,
            instances=args.instances,
            instance_ids=args.instance_ids,
            out=args.out,
            metrics_path=args.metrics,
            parallel=args.parallel,
            agent_in_loop=args.agent_in_loop,
            per_task_timeout=args.per_task_timeout,
            instance_cooldown=args.instance_cooldown,
            max_iterations=args.max_iterations,
            include_hints=args.include_hints,
            architect=args.architect,
            resume=args.resume,
            allow_web_search=args.allow_web_search,
            dry_run=args.dry_run,
            instance_loader=instance_loader,
        )
    )
    logger.info("\nSummary: %s", json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
