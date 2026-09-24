"""Run Godspeed against SWE-Bench Lite instances and produce predictions.jsonl.

For each instance in the selected subset:
  1. Clone the repo into a temp dir and checkout ``base_commit``
  2. Invoke ``godspeed run`` with ``problem_statement`` as the prompt
  3. Capture ``git diff <base_commit>`` as the model patch
  4. Append ``{instance_id, model_name_or_path, model_patch}`` to
     ``predictions.jsonl``

The resulting ``predictions.jsonl`` can be submitted to the official
SWE-Bench evaluation via ``sb-cli`` (free cloud) or the local Docker harness.

Usage:
    python experiments/swebench_lite/run.py \\
        --model nvidia_nim/qwen/qwen3.5-397b-a17b \\
        --split dev \\
        --limit 1 \\
        --out experiments/swebench_lite/predictions.jsonl

    # Then evaluate:
    sb-cli submit swe-bench_lite dev \\
        --predictions_path experiments/swebench_lite/predictions.jsonl \\
        --run_id godspeed-smoke --gen_report
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """You are working in a git repository that contains a bug. Fix it.

The issue to resolve:

{problem_statement}
{hints_block}{architect_block}{in_loop_block}
Constraints:
- Modify source files in this working tree to fix the reported issue.
- Do NOT modify existing test files. New tests are optional.
- Keep edits minimal and focused on the reported bug.
- When you believe the fix is correct, stop. A separate test harness will
  verify your patch.
"""

IN_LOOP_BLOCK = """
You have a `swebench_verify_patch` tool. Call it once you believe your
fix is complete to check whether the instance resolves. If it returns
`resolved=False`, read the test output tail and revise. Budget: 5 verify
calls per instance; the tool refuses duplicate calls on an unchanged
working tree, so you must edit between calls.

Guidance:
- Prefer a minimal targeted fix. A correct patch is usually under 40
  lines; a 100+ line rewrite is often a wrong-direction signal.
- If a revise makes test failures worse, prefer to revert that edit
  before piling on more changes.
- If you're stuck after 2-3 verify attempts, commit what you have and
  stop rather than burning iterations.
"""

HINTS_BLOCK_TEMPLATE = """
Maintainer / community hints from the GitHub thread that may help locate
the fix (do not trust blindly — verify against the code):

{hints_text}
"""

ARCHITECT_BLOCK = """
Work in two explicit phases:

PHASE 1 — PLAN (use read-only tools: file_read, grep_search, glob_search,
repo_map). Before making any edits, state in 3-6 bullets:
  - What the bug is (mechanically, not just symptomatically)
  - Which specific function/class is wrong
  - What the minimal fix is
  - Why that fix addresses the root cause (not just the symptom)

PHASE 2 — EXECUTE. Apply the planned fix using file_edit or file_write.
Only edit the files named in your plan; if Phase 2 surfaces new
information, update the plan before changing additional files.
"""

REFLECTION_PROMPT_TEMPLATE = """You previously produced the following patch
to fix a bug in this repository. Before the test harness runs, critically
review your own patch.

Bug description:

{problem_statement}

Your patch so far:

```diff
{current_patch}
```

Review questions to answer to yourself (silently, in your reasoning):
- Does the patch address the ROOT CAUSE or just the reported symptom?
- Are there edge cases (None, empty, nested, tz-aware, unicode, etc.) the
  patch doesn't handle?
- Does the patch change behavior for cases that were already working?
- If you see a clearly better fix, revise. Otherwise leave the patch
  alone.

If you decide to revise: use file_edit / file_write to update the patch
in place. If you think the current patch is correct, simply stop without
making any further edits.
"""

RETRY_PROMPT_TEMPLATE = """You previously produced a patch to fix a bug in
this repository. A test harness ran your patch and it FAILED. Use the
failing test output below to diagnose the real root cause and revise your
fix.

Bug description:

{problem_statement}

Your previous patch:

```diff
{previous_patch}
```

Relevant test output (tail — the key failure signal):

```
{test_output_tail}
```

Instructions:
- Read the FAILING TEST NAME(S) carefully — they name the specific case
  your previous fix missed.
- If the failure is "X was expected but Y was returned", your fix needs
  to produce X for that input. Re-check your edit — maybe you treated the
  wrong variable, wrong branch, or wrong class.
- Use file_read to reload the file you edited and file_edit to apply the
  revised fix. You may need to revert your previous edit and redo it
  differently.
- Do NOT modify the failing test — only source files.
- Keep edits minimal and targeted at making the failing test pass
  without breaking passing tests.
"""

PER_TASK_TIMEOUT_S = 900  # 15 min agent wall-clock per task


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def _prepare_repo(repo: str, base_commit: str, dest: Path) -> None:
    """Init a repo and fetch only the needed commit to keep disk usage low."""
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    steps = [
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", url],
        ["git", "fetch", "--depth", "1", "origin", base_commit],
        ["git", "checkout", "-q", base_commit],
        ["git", "config", "user.email", "godspeed@example.com"],
        ["git", "config", "user.name", "Godspeed Benchmark"],
    ]
    for step in steps:
        result = _run(step, cwd=dest, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"git step failed: {' '.join(step)}\nstderr: {result.stderr[-400:]}")


def _run_godspeed(model: str, prompt: str, project_dir: Path, timeout: int) -> dict:
    """Invoke ``godspeed run`` and return the parsed JSON payload."""
    cmd = [
        "godspeed",
        "run",
        prompt,
        "-m",
        model,
        "-d",
        str(project_dir),
        "--json-output",
        "--auto-approve",
        "all",
        "--max-iterations",
        "40",
        "--timeout",
        str(timeout),
    ]
    t0 = time.monotonic()
    proc = _run(cmd, timeout=timeout + 60)
    elapsed = time.monotonic() - t0
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {"_parse_error": True, "_stdout_tail": proc.stdout[-400:]}
    payload["_wall_s"] = round(elapsed, 1)
    payload["_shell_exit_code"] = proc.returncode
    return payload


def _capture_patch(base_commit: str, project_dir: Path) -> str:
    """Return the unified diff of the current working tree vs base_commit.

    Diagnostic: log ``git status --porcelain`` and ``git diff --stat`` so
    we can tell whether an empty patch means "no modifications happened"
    vs "modifications happened but git diff can't see them". The latter
    has bitten us when FileEditTool writes CRLF on Windows on files
    normalized LF in the index.
    """
    status = _run(["git", "status", "--porcelain"], cwd=project_dir, timeout=30)
    if status.stdout.strip():
        logger.info("git status (porcelain):\n%s", status.stdout[:2000])
    else:
        logger.info("git status: clean")

    stat = _run(["git", "diff", "--stat", base_commit], cwd=project_dir, timeout=30)
    if stat.stdout.strip():
        logger.info("git diff --stat:\n%s", stat.stdout[:1000])

    result = _run(["git", "diff", base_commit], cwd=project_dir, timeout=60)
    if result.returncode != 0:
        logger.warning("git diff failed rc=%d: %s", result.returncode, result.stderr[-200:])
        return ""
    if not result.stdout.strip() and status.stdout.strip():
        # Modifications exist per status but diff is empty — usually a CRLF
        # vs LF mismatch. Retry with --text to force text diff regardless
        # of git's autocrlf view.
        logger.warning("git diff empty despite dirty working tree — retrying with --text")
        result = _run(
            ["git", "-c", "core.autocrlf=false", "diff", "--text", base_commit],
            cwd=project_dir,
            timeout=60,
        )
    return result.stdout


def _load_instances(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return [dict(row) for row in ds]


def _filter(instances: list[dict], ids: list[str] | None, limit: int | None) -> list[dict]:
    if ids:
        id_set = set(ids)
        instances = [i for i in instances if i["instance_id"] in id_set]
    if limit is not None:
        instances = instances[:limit]
    return instances


def _already_predicted(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    done = set()
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LiteLLM model id for Godspeed")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of instances")
    parser.add_argument(
        "--instance-ids", nargs="*", default=None, help="Specific instance ids to run"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/swebench_lite/predictions.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("experiments/swebench_lite/run_metrics.jsonl"),
    )
    parser.add_argument(
        "--per-task-timeout",
        type=int,
        default=PER_TASK_TIMEOUT_S,
        help="Wall-clock timeout for each godspeed invocation (seconds)",
    )
    parser.add_argument(
        "--instance-cooldown",
        type=int,
        default=0,
        help="Seconds to sleep between instances. Useful with --agent-in-loop on "
        "rate-limited free-tier providers (NVIDIA NIM R&D free tier is 40 RPM "
        "shared; without cooldown, back-to-back instances sustain saturation and "
        "most crash with llm_error). 60-90s recommended for agent-in-loop runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already present in --out",
    )
    parser.add_argument(
        "--include-hints",
        action="store_true",
        help="Include SWE-Bench `hints_text` (GitHub thread comments) in the prompt. "
        "Standard practice for Aider / mini-swe-agent benchmark runs.",
    )
    parser.add_argument(
        "--architect",
        action="store_true",
        help="Invoke godspeed with /architect enabled (two-phase plan->execute).",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="After initial patch, run a reflection pass that shows the patch back "
        "to the agent with 'critique and revise if wrong.' Captures the final "
        "patch post-revision.",
    )
    parser.add_argument(
        "--verify-retry",
        action="store_true",
        help="DEPRECATED in favor of --agent-in-loop. After initial patch, run the local "
        "SWE-Bench harness (via WSL+Docker) to check whether the patch resolves the "
        "instance. If not, re-invoke Godspeed with the failing test output as context "
        "and capture the revised patch. Requires swebench installed in WSL Ubuntu. "
        "Ignored when --agent-in-loop is set.",
    )
    parser.add_argument(
        "--agent-in-loop",
        action="store_true",
        help="Drive the agent in-process via godspeed.agent.loop.agent_loop() with a "
        "per-instance swebench_verify_patch tool registered. The agent can call the "
        "tool mid-session to run the test harness and iterate until resolved. "
        "Replaces --verify-retry (post-hoc single-shot retry). Requires swebench "
        "installed in WSL Ubuntu (Windows) or natively (Linux).",
    )
    parser.add_argument(
        "--allow-web-search",
        action="store_true",
        help="Enable web_search / web_fetch tools during agent sessions. OFF by default "
        "for benchmark integrity — otherwise the agent could search GitHub for the "
        "ground-truth fix by instance id. Only set for real-world (non-benchmark) runs.",
    )
    parser.add_argument(
        "--isolate-agent-shell",
        action="store_true",
        help="Run every agent shell command in a private mount+PID namespace where $HOME and /mnt are "
        "empty and /tmp is per-task (scripts/agent_shell_isolate.sh). Agents have been seen searching "
        "the host for hidden tests and gold data. Linux only; fails closed. See docs/benchmark_hygiene.md.",
    )
    parser.add_argument(
        "--task-python",
        default=None,
        metavar="VERSION",
        help="Give each task its own throwaway venv on this Python (e.g. 3.9, the version of the "
        "SWE-bench images) so the agent's pip installs cannot touch Godspeed's venv or leak between "
        "tasks. Needs uv.",
    )
    args = parser.parse_args()

    # isolation is a sibling script module; import by path since run.py is run as a script.
    sys.path.insert(0, str(Path(__file__).parent))
    import isolation as _iso_module

    if args.isolate_agent_shell:
        try:
            _iso_module.check_isolation_available()
        except _iso_module.IsolationUnavailableError as exc:
            logger.error("--isolate-agent-shell requested but unavailable: %s", exc)
            return 2

    if args.agent_in_loop and args.verify_retry:
        logger.warning(
            "--verify-retry is deprecated and ignored when --agent-in-loop is set; "
            "the in-loop oracle replaces post-hoc retry."
        )
        args.verify_retry = False
    elif args.verify_retry:
        logger.warning(
            "--verify-retry is deprecated; switch to --agent-in-loop for "
            "mid-session oracle verification. This flag will be removed in v3.1."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    instances = _filter(_load_instances(args.split), args.instance_ids, args.limit)
    already = _already_predicted(args.out) if args.resume else set()
    to_run = [i for i in instances if i["instance_id"] not in already]
    logger.info(
        "split=%s total=%d filtered=%d already=%d to_run=%d",
        args.split,
        len(instances),
        len(instances),
        len(already),
        len(to_run),
    )

    for idx, inst in enumerate(to_run, 1):
        if idx > 1 and args.instance_cooldown > 0:
            logger.info(
                "cooldown: sleeping %ds before next instance to let NIM RPM window reset",
                args.instance_cooldown,
            )
            time.sleep(args.instance_cooldown)

        iid = inst["instance_id"]
        repo = inst["repo"]
        base = inst["base_commit"]
        logger.info("[%d/%d] %s (%s @ %s)", idx, len(to_run), iid, repo, base[:10])

        metrics = {
            "instance_id": iid,
            "repo": repo,
            "base_commit": base,
            "model": args.model,
            "status": "ok",
        }

        workspace = Path(tempfile.mkdtemp(prefix=f"swebench-{iid}-"))
        try:
            try:
                _prepare_repo(repo, base, workspace)
            except Exception as e:
                metrics["status"] = "clone_error"
                metrics["error"] = str(e)[:400]
                patch = ""
                godspeed_payload: dict = {}
            else:
                hints_block = ""
                if args.include_hints and inst.get("hints_text"):
                    hints_block = HINTS_BLOCK_TEMPLATE.format(hints_text=inst["hints_text"].strip())
                architect_block = ARCHITECT_BLOCK if args.architect else ""
                in_loop_block = IN_LOOP_BLOCK if args.agent_in_loop else ""
                prompt = DEFAULT_PROMPT_TEMPLATE.format(
                    problem_statement=inst["problem_statement"],
                    hints_block=hints_block,
                    architect_block=architect_block,
                    in_loop_block=in_loop_block,
                )
                iso_ctx = (
                    _iso_module.task_isolation(
                        tag=args.out.stem,
                        instance_id=iid,
                        python=args.task_python,
                        isolate_shell=args.isolate_agent_shell,
                    )
                    if (args.task_python or args.isolate_agent_shell)
                    else contextlib.nullcontext()
                )
                try:
                    with iso_ctx:
                        if args.agent_in_loop:
                            # run_in_loop is a sibling script module; import by
                            # path since run.py is typically invoked as a script.
                            import sys as _sys

                            _sys.path.insert(0, str(Path(__file__).parent))
                            from run_in_loop import run_one as _run_one_in_loop

                            godspeed_payload = _run_one_in_loop(
                                instance_id=iid,
                                model=args.model,
                                prompt=prompt,
                                project_dir=workspace,
                                split=args.split,
                                timeout_s=args.per_task_timeout,
                                verify_workdir=args.out.parent.resolve(),
                                max_iterations=40,
                                tool_set="full" if args.allow_web_search else "local",
                            )
                        else:
                            godspeed_payload = _run_godspeed(
                                args.model, prompt, workspace, args.per_task_timeout
                            )
                    if godspeed_payload.get("_shell_exit_code") != 0:
                        metrics["status"] = f"agent_exit_{godspeed_payload.get('_shell_exit_code')}"
                    patch = _capture_patch(base, workspace)
                    metrics["agent_in_loop"] = bool(args.agent_in_loop)
                    metrics["verify_call_count"] = godspeed_payload.get("verify_call_count", 0)
                    metrics["iterations_used"] = godspeed_payload.get("iterations_used")
                    metrics["exit_reason"] = godspeed_payload.get("exit_reason")
                    metrics["verify_retried"] = False
                    if args.verify_retry and patch.strip():
                        # verify_patch is a sibling module; this runner is
                        # invoked as a script so the parent package isn't
                        # on sys.path — import by path.
                        import sys as _sys

                        _sys.path.insert(0, str(Path(__file__).parent))
                        from verify_patch import verify_patch

                        try:
                            resolved_1, test_output = verify_patch(
                                instance_id=iid,
                                model_name=args.model,
                                model_patch=patch,
                                workdir=args.out.parent.resolve(),
                                timeout_s=args.per_task_timeout,
                            )
                        except Exception as e:
                            logger.warning("verify_patch failed for %s: %s", iid, e)
                            resolved_1, test_output = None, f"(verify error: {e})"

                        metrics["initial_patch_resolved"] = resolved_1
                        if resolved_1 is False:
                            logger.info(
                                "verify: %s UNRESOLVED on initial patch — retrying agent with test output",
                                iid,
                            )
                            retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                                problem_statement=inst["problem_statement"],
                                previous_patch=patch[:8000],
                                test_output_tail=test_output[-3000:],
                            )
                            try:
                                retry_payload = _run_godspeed(
                                    args.model,
                                    retry_prompt,
                                    workspace,
                                    args.per_task_timeout,
                                )
                                metrics["verify_retried"] = True
                                metrics["retry_tool_calls"] = retry_payload.get(
                                    "tool_call_count", 0
                                )
                                retry_patch = _capture_patch(base, workspace)
                                # Only keep the retry patch if it's non-empty;
                                # an empty retry shouldn't overwrite a non-empty first attempt.
                                if retry_patch.strip():
                                    patch = retry_patch
                                    logger.info(
                                        "retry produced %d-line patch for %s",
                                        len(patch.splitlines()),
                                        iid,
                                    )
                                else:
                                    logger.info(
                                        "retry produced empty patch for %s — keeping initial",
                                        iid,
                                    )
                            except subprocess.TimeoutExpired:
                                logger.warning("verify-retry godspeed timeout for %s", iid)
                        elif resolved_1 is True:
                            logger.info(
                                "verify: %s RESOLVED on initial patch — skipping retry", iid
                            )

                    metrics["reflected"] = False
                    if args.reflect and patch.strip():
                        logger.info(
                            "running reflection pass for %s (patch=%d lines)",
                            iid,
                            len(patch.splitlines()),
                        )
                        reflect_prompt = REFLECTION_PROMPT_TEMPLATE.format(
                            problem_statement=inst["problem_statement"],
                            current_patch=patch[:12000],  # cap to avoid huge prompts
                        )
                        try:
                            reflect_payload = _run_godspeed(
                                args.model,
                                reflect_prompt,
                                workspace,
                                args.per_task_timeout,
                            )
                            metrics["reflect_tool_calls"] = reflect_payload.get(
                                "tool_call_count", 0
                            )
                            metrics["reflect_wall_s"] = reflect_payload.get("_wall_s")
                            patch = _capture_patch(base, workspace)
                            metrics["reflected"] = True
                        except subprocess.TimeoutExpired:
                            logger.warning("reflection pass timed out for %s", iid)
                            metrics["reflected"] = False
                except subprocess.TimeoutExpired as e:
                    # godspeed's own --timeout should have fired first; if we get
                    # here godspeed hung past it. Don't kill the whole run.
                    logger.warning(
                        "godspeed subprocess timed out for %s after %ss — recording and moving on",
                        iid,
                        e.timeout,
                    )
                    metrics["status"] = "subprocess_timeout"
                    metrics["error"] = f"subprocess.TimeoutExpired after {e.timeout}s"
                    godspeed_payload = {"_wall_s": args.per_task_timeout + 60}
                    # Best-effort: capture whatever is on disk before the hung child was killed
                    try:
                        patch = _capture_patch(base, workspace)
                    except Exception:
                        patch = ""

            metrics["patch_lines"] = len(patch.splitlines())
            metrics["patch_nonempty"] = bool(patch.strip())
            metrics["wall_s"] = godspeed_payload.get("_wall_s")
            metrics["tool_call_count"] = godspeed_payload.get("tool_call_count")
            metrics["cost_usd"] = godspeed_payload.get("cost_usd", 0.0)
            metrics["output_tokens"] = godspeed_payload.get("output_tokens")

            prediction = {
                "instance_id": iid,
                "model_name_or_path": args.model,
                "model_patch": patch,
            }
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(prediction) + "\n")

            logger.info(
                "  -> patch_lines=%d status=%s wall=%ss",
                metrics["patch_lines"],
                metrics["status"],
                metrics["wall_s"],
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
            with args.metrics.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

    logger.info("done. predictions=%s metrics=%s", args.out, args.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
