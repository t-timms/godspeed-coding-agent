# Benchmark scoring: gold-check, pre-registered selection, post-hoc scoring, paired comparison

`scripts/swebench_bench.py` automates four steps of a SWE-bench evaluation. Each exists because skipping it
produced a wrong or unverifiable number on a real run (Qwen3.8-27B vs KAT-REAP50 on a 16 GB GPU).

Needs the `swebench` package (set `GODSPEED_SWEBENCH_PYTHON` to the interpreter that has it; 5.x flags are
used) and Docker. The dataset is `SWE-bench/SWE-bench_Lite`; the older `princeton-nlp/SWE-bench_Lite` copy
lacks the `image` column swebench 5.x reads.

## 1. Gold-check every candidate task

```bash
python scripts/swebench_bench.py gold-check --split dev --seed 20260924 --select 8 --out gold.json
```

Runs the official harness with the **gold** patch. In the images published in September 2026, 8 of the 23
SWE-bench Lite `dev` tasks did not resolve even with the gold patch, so any score on them is meaningless:

| tasks | why the gold patch fails |
|---|---|
| all 5 pvlib | `np.Inf` was removed in NumPy 2 and the image ships NumPy 2 |
| pyvista-4315 | `libGL.so.1: cannot open shared object file` |
| 2 pydicom | test-infrastructure `AttributeError` |

`gold.json` lists the valid tasks and, for each invalid one, the reason read from the harness log.
`scripts/validate_driver.py`'s default smoke set used to include a pvlib task.

## 2. Choose the tasks by a rule fixed in advance

`--select N --seed S` picks **the first N gold-valid tasks of a seeded random order**. Fix `S` and `N` before
any agent runs and write them down; the set cannot then be tuned to a result. The order has the prefix
property (asking for more tasks later keeps the earlier ones), and the rule is independent of input order.

## 3. Score post-hoc with the official harness

```bash
python scripts/swebench_bench.py score --tag A1 --predictions preds.jsonl --metrics metrics.jsonl \
    --ids-file ids.txt --out a1.json
```

Scores the **final patches** (the last prediction per instance), independently of any in-loop verify tool
the agent may have called. Empty patches count as unresolved and are never sent to the harness. Reports
`resolved/n`, the Wilson 95% interval, agent minutes and solved per hour (from the runner's metrics file).

**It fails closed.** A task the harness could not score (`error_ids`, `infra_failure_ids`,
`incomplete_ids`, or absent from the report) is not "unresolved": it is missing. Counting it as a failure
would deflate an arm's score whenever, say, Docker Hub rate-limits an image pull mid-run. In that case
`score` exits 3, writes the details to `<out>.incomplete.json` and does **not** write `<out>`, so a
resumable driver re-scores instead of treating the draw as finished. `--allow-harness-errors` writes the
score anyway, with `"complete": false` and `harness_problems` recorded. `ambiguous_failure` (the tests ran but
the log gave no clear verdict) is recorded but still counts as unresolved.

## 4. Compare arms as paired data

```bash
python scripts/swebench_bench.py compare --arm A a1.json a2.json a3.json --arm C c1.json c2.json c3.json
```

Draws of the same tasks are **not** independent samples, so pooling them into one big binomial overstates
the evidence. `compare` counts, per task, how many of each arm's draws resolved it, then reports on how
many tasks one arm is strictly ahead (ties dropped) with an **exact two-sided sign test**, next to the pooled
Wilson intervals (descriptive only). On the run above: 8 tasks x 3 draws per arm, 19/24 (60-91%) versus
12/24 (31-69%) pooled, the first arm strictly ahead on 4 tasks, tied on 4, behind on none, sign test
p = 0.125: suggestive, not established. Eight tasks cannot rank close arms.

## Also do

* Run the agent in an isolated shell and audit its commands (`docs/benchmark_hygiene.md`).
* State plainly when the agent had a verify tool: such scores are not comparable to leaderboards.
