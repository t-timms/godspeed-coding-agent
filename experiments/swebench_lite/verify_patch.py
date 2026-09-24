"""Run a single SWE-Bench patch through the local Docker harness.

Returns (resolved: bool, test_output: str). Used as the oracle signal
for the verify-then-retry loop in run.py and the agent-in-loop
`swebench_verify_patch` tool.

Two execution paths:

1. **WSL path** (Windows) - wraps `python3 -m swebench.harness.run_evaluation`
   inside `wsl -d Ubuntu -- bash -lc '...'`. Windows paths are translated
   to `/mnt/<drive>/...` for the WSL command. Swebench is installed inside
   the Ubuntu distro via `pip3 install --break-system-packages --user swebench`.

2. **Native path** (Linux, CI) - calls `python3 -m swebench.harness.run_evaluation`
   directly via `subprocess.run`. Swebench must be installed in the calling
   env (`pip install swebench`). No path translation.

Autodetection: native if `sys.platform != "win32"`. Override with
`GODSPEED_SWEBENCH_WSL=0` (force native) or `=1` (force WSL, e.g. for
debugging the WSL path from a Linux container).

Typical use from run.py (imported directly):

    from experiments.swebench_lite.verify_patch import verify_patch
    resolved, test_output = verify_patch(
        instance_id="sqlfluff__sqlfluff-2419",
        model_name="nvidia_nim/moonshotai/kimi-k2.5",
        model_patch=patch_str,
        workdir=Path("experiments/swebench_lite"),
    )

Standalone:

    python experiments/swebench_lite/verify_patch.py \
        --instance sqlfluff__sqlfluff-2419 \
        --model nvidia_nim/moonshotai/kimi-k2.5 \
        --patch-from experiments/swebench_lite/predictions_e1_kimi.jsonl
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# WSL command prefix. Uses the default Ubuntu distro.
WSL_CMD = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc"]


def _use_wsl() -> bool:
    """Decide whether to run the harness via WSL or natively.

    Env override: ``GODSPEED_SWEBENCH_WSL`` in ``{"0","false","no"}`` forces
    native; anything truthy forces WSL. Unset/empty autodetects on platform.
    """
    override = os.environ.get("GODSPEED_SWEBENCH_WSL", "").strip().lower()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return True
    return sys.platform == "win32"


def _wsl_run(bash_cmd: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    """Run a bash command inside WSL Ubuntu with sensible defaults."""
    return subprocess.run(
        [*WSL_CMD, bash_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def _native_run(bash_cmd: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    """Run a bash command natively (Linux/CI path)."""
    return subprocess.run(
        ["bash", "-lc", bash_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _windows_to_wsl(p: Path) -> str:
    """Convert a Windows path like C:\\Users\\x to /mnt/c/Users/x for WSL."""
    parts = list(p.resolve().parts)
    if len(parts) == 0:
        return str(p)
    drive = parts[0].rstrip(":\\").lower()
    return "/mnt/" + drive + "/" + "/".join(parts[1:]).replace("\\", "/")


PYTHON_ENV = "GODSPEED_SWEBENCH_PYTHON"
LEGACY_WSL_PYTHON = "/home/swebench_venv/bin/python3"
# swebench 5.x needs the image/eval_script/log_parser fields; the old princeton-nlp/ snapshot lacks them.
LITE_DATASET = "SWE-bench/SWE-bench_Lite"


def _swebench_python(use_wsl: bool) -> str:
    """Interpreter that has ``swebench`` installed.

    ``GODSPEED_SWEBENCH_PYTHON`` wins. Otherwise the historical WSL venv
    (``/home/swebench_venv``) is used when it exists, and the current
    interpreter (native Linux/CI, where swebench is installed in the same
    env) when it does not.
    """
    override = os.environ.get(PYTHON_ENV, "").strip()
    if override:
        return override
    if use_wsl or Path(LEGACY_WSL_PYTHON).exists():
        return LEGACY_WSL_PYTHON
    return sys.executable


@functools.lru_cache(maxsize=8)
def _supports_cache_level(python_path: str, use_wsl: bool) -> bool:
    """True if this swebench still accepts ``--cache_level`` (4.x does, 5.x removed it)."""
    runner = _wsl_run if use_wsl else _native_run
    try:
        result = runner(f"{python_path} -m swebench.harness.run_evaluation --help", timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "--cache_level" in (result.stdout + result.stderr)


def _harness_cmd(
    workdir_str: str,
    preds_str: str,
    instance_id: str,
    run_id: str,
    dataset_path: str | None = None,
    python_path: str = LEGACY_WSL_PYTHON,
    cache_level_flag: bool = True,
    split: str = "dev",
) -> str:
    """Compose the bash command to run the swebench harness.

    ``workdir_str`` and ``preds_str`` are already in the format the shell
    will consume (POSIX for WSL/native, Windows-translated for WSL).

    If ``dataset_path`` is provided (local .jsonl file), use that instead
    of a HuggingFace dataset name — avoids split/ID mismatch issues.
    """
    cache = " --cache_level instance" if cache_level_flag else ""

    if dataset_path:
        # Use local dataset file - no split needed, instance_ids filters it
        return (
            f"cd '{workdir_str}' && "
            f"{python_path} -m swebench.harness.run_evaluation "
            f"--predictions_path '{preds_str}' "
            f"--dataset_name '{dataset_path}' "
            f"--instance_ids {instance_id} "
            f"--max_workers 1 "
            f"--run_id {run_id}{cache}"
        )
    else:
        # Fallback to HuggingFace dataset (original behavior)
        return (
            f"cd '{workdir_str}' && "
            f"{python_path} -m swebench.harness.run_evaluation "
            f"--predictions_path '{preds_str}' "
            f"--dataset_name {LITE_DATASET} "
            f"--split {split} "
            f"--instance_ids {instance_id} "
            f"--max_workers 1 "
            f"--run_id {run_id}{cache}"
        )


def _find_instance_row(instance_id: str, split: str, project_root: Path) -> dict | None:
    """Return the dataset row for ``instance_id``.

    Looks in ``benchmarks/swebench_lite_test.jsonl`` first (offline copy, optional), then in the
    HuggingFace SWE-bench Lite splits (``split`` first, then the other one).
    """
    local = project_root / "benchmarks" / "swebench_lite_test.jsonl"
    if local.is_file():
        for line in local.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("instance_id") == instance_id:
                    return row
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed; cannot look up %s", instance_id)
        return None
    for name in [split, *[x for x in ("dev", "test") if x != split]]:
        try:
            for row in load_dataset(LITE_DATASET, split=name):
                if row["instance_id"] == instance_id:
                    return dict(row)
        except Exception as exc:  # noqa: BLE001 - network/dataset errors are non-fatal here
            logger.warning("could not load %s[%s]: %s", LITE_DATASET, name, exc)
    return None


def verify_patch(
    instance_id: str,
    model_name: str,
    model_patch: str,
    workdir: Path,
    timeout_s: int = 900,
    split: str = "dev",
) -> tuple[bool, str]:
    """Run the swebench harness on a single patch via local Docker.

    Returns ``(resolved, test_output)``. ``resolved`` is ``True`` iff the
    harness reports the instance as resolved. ``test_output`` is the raw
    test_output.txt contents if the harness produced one, or a short error
    summary if the harness itself failed.
    """
    if not model_patch.strip():
        return False, "(empty patch - nothing to verify)"

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Unique run id — hash of (instance + patch) so repeated calls on the
    # same content reuse harness artifacts.
    digest = hashlib.sha1(
        (instance_id + "::" + model_patch).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    run_id = f"verify_{instance_id.replace('/', '_')}_{digest}"

    # Write single-instance predictions file
    preds_path = workdir / f".verify_{digest}.jsonl"
    preds_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": model_patch,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Use a one-row local dataset file to avoid split/ID mismatch.
    dataset_path = workdir / f".dataset_{digest}.jsonl"
    # An empty file left by an older run must not be trusted.
    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        row = _find_instance_row(instance_id, split, workdir.parent.parent)
        if row is None:
            return False, (
                f"(instance {instance_id} not found in the local benchmarks/ jsonl or in "
                f"{LITE_DATASET} splits; cannot run the harness)"
            )
        dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    wsl = _use_wsl()
    python_path = _swebench_python(wsl)
    cache_flag = _supports_cache_level(python_path, wsl)
    if wsl:
        workdir_str = _windows_to_wsl(workdir)
        preds_str = _windows_to_wsl(preds_path)
        dataset_str = _windows_to_wsl(dataset_path)
        bash_cmd = _harness_cmd(
            workdir_str, preds_str, instance_id, run_id, dataset_str, python_path, cache_flag, split
        )
        logger.info("verify harness (wsl): %s (timeout %ds)", instance_id, timeout_s)
        result = _wsl_run(bash_cmd, timeout=timeout_s)
    else:
        bash_cmd = _harness_cmd(
            str(workdir),
            str(preds_path),
            instance_id,
            run_id,
            str(dataset_path),
            python_path,
            cache_flag,
            split,
        )
        logger.info("verify harness (native): %s (timeout %ds)", instance_id, timeout_s)
        result = _native_run(bash_cmd, timeout=timeout_s)

    # Expected report path (written to cwd by the harness).
    # Normalize the model name the same way the harness does: "/" -> "__"
    model_norm = model_name.replace("/", "__")
    report_path = workdir / f"{model_norm}.{run_id}.json"
    if not report_path.is_file():
        logger.warning(
            "verify: report not found at %s - harness likely failed. stderr tail:\n%s",
            report_path,
            result.stderr[-500:],
        )
        return False, f"(harness failed)\n{result.stderr[-1000:]}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = instance_id in report.get("resolved_ids", [])

    # Fetch test output for the agent's retry prompt context.
    # swebench writes logs/run_evaluation/<run_id>/<model>/<instance>/test_output.txt
    log_rel = Path("logs/run_evaluation") / run_id / model_norm / instance_id / "test_output.txt"
    log_path = workdir / log_rel
    if log_path.is_file():
        test_output = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        # Fallback: look in the whole logs tree (swebench's exact layout
        # has varied across versions).
        matches = (
            list((workdir / "logs").rglob("test_output.txt")) if (workdir / "logs").is_dir() else []
        )
        test_output = (
            matches[-1].read_text(encoding="utf-8", errors="replace")
            if matches
            else "(no test_output.txt found)"
        )

    return resolved, test_output


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--patch-from",
        type=Path,
        required=True,
        help="A predictions.jsonl file - we pick the row whose instance_id matches --instance",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("experiments/swebench_lite"),
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    for line in args.patch_from.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["instance_id"] == args.instance:
            patch = row["model_patch"]
            break
    else:
        raise SystemExit(f"instance {args.instance} not in {args.patch_from}")

    resolved, test_output = verify_patch(
        instance_id=args.instance,
        model_name=args.model,
        model_patch=patch,
        workdir=args.workdir,
        timeout_s=args.timeout,
    )
    print(f"resolved: {resolved}")
    print()
    print("--- test_output (tail) ---")
    print(test_output[-2000:])
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(_main())
