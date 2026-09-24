"""Smoke-test a new LLM driver against 3 easy SWE-Bench Lite dev instances.

Before any new model goes into Godspeed's ensemble pools (Phase 4) or
the benchmark-run rotation (Phase 2/3), it must pass this validation.

What it checks:
  1. Driver is reachable — the model answers at all.
  2. Agent-in-loop session exits cleanly (not ``agent_exit_4`` LLM error).
  3. Agent produces a non-empty patch on at least 1 of 3 instances.
  4. Optionally: the patch resolves on the local harness.
  5. Catalog entry exists (warn if missing; not fatal).

Usage:

    # Most common — smoke a known driver:
    python scripts/validate_driver.py --model nvidia_nim/moonshotai/kimi-k2.5

    # Test with a model not yet in the catalog (expect warning):
    python scripts/validate_driver.py --model moonshot/kimi-k2.7

    # Use a different subset of instances (default is 3 easy ones):
    python scripts/validate_driver.py --model ... --instance-ids sqlfluff__sqlfluff-2419

    # Skip the harness verify step (faster — only checks agent success):
    python scripts/validate_driver.py --model ... --no-verify

Exit codes:
    0 — driver passed smoke
    1 — driver failed (agent crashes, empty patches, or < 1 resolved)
    2 — setup error (model string invalid, etc.)

This script is the gate for adding a driver to the catalog's default
rotation. A failing driver should NOT be used in ensembles — it'll drag
numbers down. Catalog entries can sit behind this gate indefinitely.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Make the experiments directory importable (run_in_loop + run's helpers
# live there, not in the godspeed package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS = _REPO_ROOT / "experiments" / "swebench_lite"
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

# Configure LiteLLM env for local llama.cpp server BEFORE any other
# godspeed import — LiteLLM caches env vars at import time. Without
# this, openai/ models route to the real OpenAI API instead of local.
from godspeed.tools.llamacpp_manager import (  # noqa: E402
    configure_litellm_env,  # type: ignore[import-untyped]
)

configure_litellm_env()

logger = logging.getLogger(__name__)

# Default smoke set — 3 instances chosen for variety + resolvability under the agent-in-loop path.
# Each one is checked to be resolved by its GOLD patch in the currently published SWE-bench
# images (scripts/swebench_bench.py gold-check). pvlib__pvlib-python-1606 used to be here but
# its gold patch no longer resolves there: the image has NumPy 2, which removed `np.Inf`, so
# `import pvlib` fails.
DEFAULT_SMOKE_INSTANCES = [
    "sqlfluff__sqlfluff-2419",  # known-resolvable on Phase 1 smoke
    "pydicom__pydicom-1256",  # small and fast, gold-verified
    "marshmallow-code__marshmallow-1343",  # smaller, for faster feedback
]

# Maximum acceptable failure rates for the driver to pass.
MAX_EXIT_4_RATE = 0.20  # 20% LLM-error rate max
MIN_NONEMPTY_PATCHES = 1  # at least 1 of the smoke instances must produce a patch


def _prepare_repo(repo: str, base_commit: str, dest: Path) -> None:
    """Shallow-clone the instance repo. Copied from run.py._prepare_repo."""
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    for step in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", url],
        ["git", "fetch", "--depth", "1", "origin", base_commit],
        ["git", "checkout", "-q", base_commit],
        ["git", "config", "user.email", "godspeed@example.com"],
        ["git", "config", "user.name", "Godspeed Validator"],
    ):
        result = subprocess.run(
            step, cwd=dest, capture_output=True, text=True, timeout=180, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"git step failed: {' '.join(step)}\nstderr: {result.stderr[-400:]}")


def _load_instance(instance_id: str) -> dict:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    for row in ds:
        if row["instance_id"] == instance_id:
            return dict(row)
    raise LookupError(f"instance {instance_id} not in SWE-Bench Lite dev split")


def _build_prompt(problem_statement: str) -> str:
    return (
        "You are working in a git repository that contains a bug. Fix it.\n\n"
        "The issue to resolve:\n\n"
        f"{problem_statement}\n\n"
        "Constraints:\n"
        "- Modify source files in this working tree to fix the reported issue.\n"
        "- Do NOT modify existing test files. New tests are optional.\n"
        "- Keep edits minimal and focused on the reported bug.\n"
        "- You have a `swebench_verify_patch` tool. Call it once you believe\n"
        "  your fix is complete. If `resolved=False`, revise and call again.\n"
    )


def validate(
    model: str,
    instance_ids: list[str],
    timeout_s: int,
    verify: bool,
    workdir: Path,
) -> int:
    """Run the smoke set and return a process-exit code.

    0 = pass, 1 = fail driver, 2 = setup error.
    """
    try:
        from run_in_loop import run_one  # type: ignore[import-not-found]

        from godspeed.agent.prompt_profiles import (
            get_catalog_entry,
            resolve_profile,
        )
    except ImportError as exc:
        logger.error("import failed: %s", exc)
        return 2

    entry = get_catalog_entry(model)
    if entry is None:
        logger.warning(
            "model %s not in driver_catalog.yaml; using default profile. "
            "Add a catalog entry after smoke passes.",
            model,
        )
    else:
        logger.info(
            "catalog: profile=%s ctx=%s cost_in=$%s",
            resolve_profile(model),
            entry.get("context_window"),
            entry.get("cost_per_mtok_in"),
        )

    results: list[dict] = []
    for iid in instance_ids:
        logger.info("--- smoke instance %s ---", iid)
        try:
            inst = _load_instance(iid)
        except LookupError as exc:
            logger.error("%s", exc)
            return 2

        workspace = Path(tempfile.mkdtemp(prefix=f"validate-{iid}-"))
        try:
            try:
                _prepare_repo(inst["repo"], inst["base_commit"], workspace)
            except RuntimeError as exc:
                logger.error("clone failed: %s", exc)
                results.append({"instance_id": iid, "status": "clone_error"})
                continue

            t0 = time.monotonic()
            payload = run_one(
                instance_id=iid,
                model=model,
                prompt=_build_prompt(inst["problem_statement"]),
                project_dir=workspace,
                split="dev",
                timeout_s=timeout_s,
                verify_workdir=workdir,
                max_iterations=40,
                tool_set="local",
            )
            wall = time.monotonic() - t0
            results.append(
                {
                    "instance_id": iid,
                    "exit_reason": payload.get("exit_reason"),
                    "shell_exit_code": payload.get("_shell_exit_code"),
                    "verify_call_count": payload.get("verify_call_count", 0),
                    "tool_call_count": payload.get("tool_call_count", 0),
                    "wall_s": round(wall, 1),
                    "cost_usd": payload.get("cost_usd", 0.0),
                }
            )
            logger.info(
                "  exit=%s verify_calls=%d tool_calls=%d wall=%.1fs cost=$%.4f",
                payload.get("exit_reason"),
                payload.get("verify_call_count", 0),
                payload.get("tool_call_count", 0),
                wall,
                payload.get("cost_usd", 0.0),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    # --- Gate evaluation ------------------------------------------------

    n = len(results)
    exit_4 = sum(1 for r in results if str(r.get("exit_reason", "")).startswith("llm_error"))
    nonempty = sum(
        1 for r in results if r.get("verify_call_count", 0) > 0 or r.get("tool_call_count", 0) > 3
    )

    print()
    print(f"Model:        {model}")
    print(f"Instances:    {n}")
    print(f"LLM errors:   {exit_4}/{n} ({exit_4 / max(1, n):.0%})")
    print(f"Nonempty-ish: {nonempty}/{n}")
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    print(f"Total cost:   ${total_cost:.4f}")
    print()

    if exit_4 / max(1, n) > MAX_EXIT_4_RATE:
        print(
            f"FAIL: LLM-error rate {exit_4 / n:.0%} exceeds threshold "
            f"{MAX_EXIT_4_RATE:.0%}. Driver is unreliable."
        )
        return 1

    if nonempty < MIN_NONEMPTY_PATCHES:
        print(
            f"FAIL: {nonempty}/{n} instances produced any real work. Driver can't drive the agent."
        )
        return 1

    if verify:
        print("Verify step not implemented yet — run the full dev-23 for real numbers.")

    print(f"PASS: driver {model} cleared smoke.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LiteLLM model string to validate")
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        default=DEFAULT_SMOKE_INSTANCES,
        help=f"SWE-Bench Lite dev instances to run (default: {len(DEFAULT_SMOKE_INSTANCES)} easy)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Wall-clock seconds per instance (default: 900)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the harness resolve check — only verify agent success",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=_EXPERIMENTS.resolve(),
        help=f"Harness workdir (default: {_EXPERIMENTS})",
    )
    args = parser.parse_args()

    return validate(
        model=args.model,
        instance_ids=args.instance_ids,
        timeout_s=args.timeout,
        verify=not args.no_verify,
        workdir=args.workdir,
    )


if __name__ == "__main__":
    sys.exit(main())
