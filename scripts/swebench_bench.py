#!/usr/bin/env python3
"""SWE-bench helpers: gold-check, pre-registered selection, post-hoc scoring, paired comparison.

    swebench_bench.py gold-check --split dev --sample 23 --seed 7 --select 8 --out gold.json
    swebench_bench.py score --tag A1 --predictions preds.jsonl --ids-file ids.txt --out a1.json
    swebench_bench.py compare --arm A a1.json a2.json --arm C c1.json c2.json

Why each step exists (all learned on a real run, see docs/benchmark_scoring.md):

* **gold-check**: run the official harness with the GOLD patch on every candidate task first.
  In the currently published images 8 of 23 SWE-bench Lite dev tasks are unsolvable even with
  the gold patch (NumPy 2 removed ``np.Inf``, a missing ``libGL.so.1``, a test-infrastructure
  error), so a score on them says nothing about the agent. The task set is then chosen by a
  rule fixed before any agent runs: a seeded random order, first N tasks whose gold patch
  resolves.
* **score**: score the final patches afterwards with the official harness, independently of any
  in-loop verify tool, and report Wilson 95% intervals.
* **compare**: paired, per-task comparison across draws with an exact sign test, because pooling
  draws of the same tasks as if they were independent samples overstates the evidence.

Needs the ``swebench`` package (``GODSPEED_SWEBENCH_PYTHON`` names the interpreter that has it) and
Docker for the harness. The parsing, statistics and selection logic is pure and unit-tested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"  # the princeton-nlp/ copy lacks the `image` column
PYTHON_ENV = "GODSPEED_SWEBENCH_PYTHON"


# --------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (fractions in [0, 1])."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def sign_test_two_sided(wins_a: int, wins_b: int) -> float:
    """Exact two-sided sign test on the tasks where the arms differ (ties are dropped)."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- selection


def preregistered_selection(
    all_ids: Iterable[str], valid: Iterable[str], seed: int, n: int
) -> tuple[list[str], list[str]]:
    """Seeded random order over ``all_ids``; the first ``n`` that are ``valid`` are selected.

    Fixed before any agent runs, so the task set cannot be tuned to a result. The order has the
    prefix property: sampling more ids later never reshuffles the ones already ordered.
    Returns ``(selected, full_order)``.
    """
    ids = sorted(set(all_ids))
    order = random.Random(seed).sample(ids, len(ids))  # noqa: S311 - seeded shuffle, not crypto
    valid_set = set(valid)
    return [i for i in order if i in valid_set][:n], order


# --------------------------------------------------------------------------- harness plumbing


def swebench_python() -> str:
    return os.environ.get(PYTHON_ENV, "").strip() or sys.executable


def harness_cmd(
    *,
    python: str,
    dataset: str,
    split: str,
    predictions: str,
    ids: Sequence[str],
    run_id: str,
    workers: int = 2,
    timeout: int = 1800,
) -> list[str]:
    """Command line of ``swebench.harness.run_evaluation`` (swebench 5.x flags)."""
    return [
        python,
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        dataset,
        "-s",
        split,
        "-p",
        predictions,
        "-i",
        *ids,
        "--max_workers",
        str(workers),
        "-t",
        str(timeout),
        "-id",
        run_id,
    ]


def read_report(workdir: Path, model_name: str, run_id: str) -> dict[str, Any]:
    """The harness writes ``<model>.<run_id>.json`` (``/`` in the model name becomes ``__``)."""
    path = workdir / f"{model_name.replace('/', '__')}.{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"harness produced no report at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"np\.Inf.*removed in the NumPy 2", "numpy2: np.Inf removed (image ships NumPy 2)"),
    (r"cannot open shared object file", "missing shared library in the image"),
    (r"ImportError while loading conftest", "import error while loading conftest"),
    (r"has no attribute 'tag'", "test-infrastructure AttributeError"),
    (r"ModuleNotFoundError", "missing module in the image"),
    (r"SyntaxError", "syntax error under the image's Python"),
)


def classify_failure(test_output: str) -> str:
    """One-line reason a gold patch did not resolve, from the harness's test_output.txt."""
    for pattern, reason in _FAILURE_PATTERNS:
        if re.search(pattern, test_output):
            return reason
    return "gold patch does not pass (no known infrastructure signature)"


def split_report(report: dict[str, Any], ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """(resolved, not resolved) among ``ids`` from a harness report."""
    resolved = set(report.get("resolved_ids", []))
    return [i for i in ids if i in resolved], [i for i in ids if i not in resolved]


# Report keys whose ids mean "the harness could not tell", not "the patch failed".
_PROBLEM_KEYS: dict[str, str] = {
    "error": "error_ids",
    "infra_failure": "infra_failure_ids",
    "incomplete": "incomplete_ids",
}
FATAL_PROBLEMS = ("error", "infra_failure", "incomplete", "unaccounted")


def harness_problems(report: dict[str, Any], ids: Sequence[str]) -> dict[str, list[str]]:
    """Tasks among ``ids`` the harness could not score (only non-empty categories are returned).

    ``error`` / ``infra_failure`` / ``incomplete`` (a failed image pull, a container that never
    ran) and ``unaccounted`` (in neither the resolved nor the unresolved list) are FATAL: counting
    them as "unresolved" would silently deflate the score. ``ambiguous_failure`` (tests ran, the
    log gave no clear pass/fail) is reported but does not block: those patches are still counted
    as unresolved.
    """
    wanted = set(ids)

    def listed(key: str) -> list[str]:
        value = report.get(key)
        return [i for i in value if i in wanted] if isinstance(value, list) else []

    problems = {name: listed(key) for name, key in _PROBLEM_KEYS.items()}
    problems["ambiguous_failure"] = listed("ambiguous_failure_ids")
    if isinstance(report.get("unresolved_ids"), list):  # this schema accounts for every task
        seen = set(report.get("resolved_ids", [])) | set(report["unresolved_ids"])
        seen |= {i for found in problems.values() for i in found}
        problems["unaccounted"] = [i for i in ids if i not in seen]
    return {name: found for name, found in problems.items() if found}


# --------------------------------------------------------------------------- commands


def _load_ids(args: argparse.Namespace) -> list[str]:
    if args.ids:
        return list(args.ids)
    if getattr(args, "ids_file", None):
        return Path(args.ids_file).read_text(encoding="utf-8").split()
    from datasets import load_dataset  # heavy import only when needed

    return sorted(r["instance_id"] for r in load_dataset(args.dataset, split=args.split))


def cmd_gold_check(args: argparse.Namespace) -> int:
    all_ids = _load_ids(args)
    unique = sorted(set(all_ids))
    order = random.Random(args.seed).sample(unique, len(unique))  # noqa: S311 - seeded shuffle
    candidates = order[: args.sample] if args.sample else order
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id
    cmd = harness_cmd(
        python=swebench_python(),
        dataset=args.dataset,
        split=args.split,
        predictions="gold",
        ids=candidates,
        run_id=run_id,
        workers=args.workers,
    )
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)
    try:
        report = read_report(workdir, "gold", run_id)
    except FileNotFoundError as exc:
        print(f"{exc}\n--- stderr tail ---\n{proc.stderr[-800:]}", file=sys.stderr)
        return 1
    valid, invalid = split_report(report, candidates)
    reasons: dict[str, str] = {}
    for iid in invalid:
        out = workdir / "logs" / "run_evaluation" / run_id / "gold" / iid / "test_output.txt"
        text = out.read_text(encoding="utf-8", errors="replace") if out.is_file() else ""
        reasons[iid] = classify_failure(text) if text else "no test output (harness error)"
    selected: list[str] = []
    if args.select:
        # The rule is "seeded order over ALL ids, first N gold-valid". Passing `candidates`
        # here re-shuffled just the sample into a different permutation, so with --sample M
        # smaller than the dataset the picks did not follow the documented rule. `valid` is a
        # subset of the first M of the full order, so this picks the same tasks a full-set
        # gold-check would have.
        selected, _ = preregistered_selection(unique, valid, args.seed, args.select)
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "checked": len(candidates),
        "valid": valid,
        "invalid": reasons,
        "selection_rule": f"seeded order (seed {args.seed}), first {args.select} gold-valid"
        if args.select
        else None,
        "selected": selected,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"gold-check: {len(valid)}/{len(candidates)} resolve with the gold patch")
    for iid, why in reasons.items():
        print(f"  INVALID {iid}: {why}")
    if selected:
        print("selected:", " ".join(selected))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def score_table(
    ids: Sequence[str], resolved: set[str], metrics: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for iid in ids:
        m = metrics.get(iid, {})
        rows.append(
            {
                "instance_id": iid,
                "resolved": iid in resolved,
                "patch_lines": m.get("patch_lines"),
                "wall_s": m.get("wall_s"),
                "iterations": m.get("iterations_used"),
                "verify_calls": m.get("verify_call_count"),
                "tool_calls": m.get("tool_call_count"),
                "output_tokens": m.get("output_tokens"),
                "exit_reason": m.get("exit_reason") or m.get("status"),
            }
        )
    return rows


def cmd_score(args: argparse.Namespace) -> int:
    predictions = {p["instance_id"]: p for p in load_jsonl(Path(args.predictions))}  # last wins
    metrics = {m["instance_id"]: m for m in load_jsonl(Path(args.metrics))} if args.metrics else {}
    ids = _load_ids(args) if (args.ids or args.ids_file) else sorted(predictions)
    with_patch = [i for i in ids if predictions.get(i, {}).get("model_patch", "").strip()]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    resolved: list[str] = []
    problems: dict[str, list[str]] = {}
    if with_patch:
        pfile = workdir / f"preds_{args.tag}.jsonl"
        pfile.write_text("".join(json.dumps(predictions[i]) + "\n" for i in with_patch), "utf-8")
        run_id = f"posthoc_{args.tag}"
        cmd = harness_cmd(
            python=swebench_python(),
            dataset=args.dataset,
            split=args.split,
            predictions=str(pfile),
            ids=with_patch,
            run_id=run_id,
            workers=args.workers,
        )
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)
        model = predictions[with_patch[0]].get("model_name_or_path", "unknown")
        try:
            report = read_report(workdir, model, run_id)
        except FileNotFoundError as exc:
            print(f"{exc}\n{proc.stderr[-800:]}", file=sys.stderr)
            return 1
        resolved, _ = split_report(report, with_patch)
        problems = harness_problems(report, with_patch)
    k, n = len(resolved), len(ids)
    lo, hi = wilson(k, n)
    walls = [metrics[i]["wall_s"] for i in ids if i in metrics and "wall_s" in metrics[i]]
    minutes = sum(walls) / 60 if walls else None
    result = {
        "tag": args.tag,
        "n": n,
        "resolved": sorted(resolved),
        "resolved_count": k,
        "empty_patches": [i for i in ids if i not in with_patch],
        "wilson95": [round(lo, 4), round(hi, 4)],
        "agent_minutes": round(minutes, 2) if minutes is not None else None,
        "solved_per_hour": round(k / (minutes / 60), 2) if minutes else None,
        "per_task": score_table(ids, set(resolved), metrics),
        "harness_problems": problems,
        "complete": not any(name in problems for name in FATAL_PROBLEMS),
    }
    out = Path(args.out)
    if not result["complete"] and not args.allow_harness_errors:
        # Fail closed: never leave a score file behind that a resumable driver would treat as final.
        held = out.with_name(out.stem + ".incomplete.json")
        held.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        fatal = {name: found for name, found in problems.items() if name in FATAL_PROBLEMS}
        print(
            f"SCORING INCOMPLETE: the harness could not score {sum(map(len, fatal.values()))} "
            f"task(s) {fatal}. They are NOT counted as unresolved; {out} was not written "
            f"(details: {held}). Fix the cause (often a rate-limited image pull) and re-run, "
            "or pass --allow-harness-errors.",
            file=sys.stderr,
        )
        return 3
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if problems:
        print(f"WARNING harness problems recorded in {out}: {problems}", file=sys.stderr)
    pct = 100 * k / max(n, 1)
    print(f"{args.tag}: resolved {k}/{n} = {pct:.1f}% (Wilson 95% {100 * lo:.0f}-{100 * hi:.0f}%)")
    return 0


def compare_arms(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Paired per-task comparison of two arms, each a list of score results (draws).

    A task counts for arm X as the number of X's draws that resolved it. Arm A "wins" a task when A
    resolved strictly more of its draws than B; ties are dropped from the sign test. Draws share the
    same tasks, so the pooled interval is descriptive only; the sign test over tasks is the
    conservative view.
    """
    names = list(arms)
    if len(names) != 2:
        raise ValueError("compare needs exactly two arms")
    a, b = names
    tasks = sorted(
        {t["instance_id"] for draws in arms.values() for d in draws for t in d["per_task"]}
    )
    counts = {
        name: {t: sum(t in d["resolved"] for d in arms[name]) for t in tasks} for name in names
    }
    wins_a = sum(counts[a][t] > counts[b][t] for t in tasks)
    wins_b = sum(counts[b][t] > counts[a][t] for t in tasks)
    pooled = {}
    for name in names:
        k = sum(len(d["resolved"]) for d in arms[name])
        n = sum(d["n"] for d in arms[name])
        lo, hi = wilson(k, n)
        pooled[name] = {"resolved": k, "n": n, "wilson95": [round(lo, 4), round(hi, 4)]}
    return {
        "arms": names,
        "draws": {n: len(arms[n]) for n in names},
        "per_task_solved_draws": counts,
        "strictly_more": {a: wins_a, b: wins_b},
        "ties": len(tasks) - wins_a - wins_b,
        "sign_test_two_sided_p": round(sign_test_two_sided(wins_a, wins_b), 4),
        "pooled": pooled,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    arms: dict[str, list[dict[str, Any]]] = {}
    for name, *files in args.arm:
        draws = []
        for f in files:
            draw = json.loads(Path(f).read_text(encoding="utf-8"))
            if draw.get("complete") is False:
                print(
                    f"WARNING {f}: scored with harness errors {draw.get('harness_problems')}; "
                    "those tasks are missing from the resolved set, not failures",
                    file=sys.stderr,
                )
            draws.append(draw)
        arms[name] = draws
    res = compare_arms(arms)
    print(json.dumps(res, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", default=DEFAULT_DATASET)
        p.add_argument("--split", default="dev", choices=["dev", "test"])
        p.add_argument("--workdir", default="bench_work")
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--ids", nargs="*", default=None)
        p.add_argument("--ids-file", default=None)

    g = sub.add_parser("gold-check", help="run the gold patch through the harness")
    common(g)
    g.add_argument(
        "--sample", type=int, default=0, help="check the first N of the seeded order (0=all)"
    )
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--select", type=int, default=0, help="pre-registered: first N gold-valid")
    g.add_argument("--run-id", default="gold_check")
    g.add_argument("--out", default="gold_check.json")
    g.set_defaults(func=cmd_gold_check)

    s = sub.add_parser("score", help="score predictions post-hoc with the official harness")
    common(s)
    s.add_argument("--tag", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--metrics", default=None)
    s.add_argument("--out", required=True)
    s.add_argument(
        "--allow-harness-errors",
        action="store_true",
        help="write the score even if the harness errored on some tasks (recorded, not counted)",
    )
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="paired comparison of two arms over several draws")
    c.add_argument(
        "--arm", nargs="+", action="append", required=True, metavar=("NAME", "SCORE.json")
    )
    c.set_defaults(func=cmd_compare)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
