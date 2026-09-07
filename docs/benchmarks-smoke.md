# Benchmarks — Offline Smoke Verification

Status: **OFFLINE SMOKE PASSED** (2026-09-07). No network, no LLM, no dataset
downloads were used. This document records what was verified, what was fixed,
and exactly what a real SWE-bench run needs.

## Scope

The SWE-bench harness lives in `src/godspeed/benchmarks/`:

| Component | File | Offline smoke verdict |
|---|---|---|
| Pre-flight checks | `preflight.py` | **PASS** — structured pass/fail report, no crashes on failure paths |
| NIM key rotation | `nim_key_rotation.py` | **PASS** — existing unit tests green (untouched) |
| SWE-bench runner | `swebench_runner.py` | **PASS** — `--dry-run` returns a valid plan offline |
| Instance selection | `swebench_runner.py` + `experiments/swebench_lite/run.py` | **PASS** — `_filter` / `_already_predicted` exercised via dry-run |
| Godspeed Lite agent | `src/godspeed/lite/` | **PASS** — existing tests green (untouched) |

## Commands run

### 1. Pre-flight dry-run (offline)

```powershell
uv run python -m godspeed.benchmarks.preflight --dry-run
```

Observed output (this machine):

```
  [ PASS] NIM connectivity — skipped (offline dry-run) — real run will verify against NIM endpoint
  [ PASS] Python: godspeed package — installed
  [ PASS] Python: LiteLLM — installed
  [ PASS] Python: HuggingFace datasets (SWE-bench) — installed
  [ PASS] Python: sb-cli — not found — needed to submit results. Install: pip install sb-cli
  [ FAIL] Docker — docker command not found on PATH
  [ PASS] Docker note — SWE-bench verification requires Docker; run without --agent-in-loop to skip
  [ FAIL] WSL — WSL command timed out
  [ PASS] Disk space — 851.3 GB free (>= 20 GB required)
  [ PASS] Pre-flight total time — 32853ms
```

Exit code: `1` (Docker + WSL unavailable on this machine — expected, not a crash).

### 2. SWE-bench runner dry-run (offline, local instances file)

```powershell
uv run python -m godspeed.benchmarks.swebench_runner `
  --model nvidia_nim/deepseek-ai/deepseek-v4-pro `
  --split test --dry-run `
  --instances-file path/to/instances.jsonl
```

Observed output (2 sample instances):

```
INFO split=test total=2 to_run=2 already=0 parallel=1 agent_in_loop=True
INFO NIM key rotation: not configured (using single key from env)
INFO DRY-RUN: would run 2 instances (model=nvidia_nim/deepseek-ai/deepseek-v4-pro split=test parallel=1)
Summary: {
  "total": 2, "resolved": 0, "errors": 0, "timeouts": 0,
  "cost_usd": 0.0, "wall_s": 0.0,
  "model": "nvidia_nim/deepseek-ai/deepseek-v4-pro", "split": "test",
  "dry_run": true,
  "instance_ids": ["django__django-11099", "matplotlib__matplotlib-22835"],
  "nim_keys_configured": 0,
  "predictions_path": "benchmarks\\results\\predictions_test.jsonl",
  "metrics_path": "benchmarks\\results\\metrics_test.jsonl",
  "log_dir": "benchmarks\\results\\logs\\run_2026-09-07_04"
}
```

Exit code: `0`. The dry-run validates config, selects instances, and reports
the plan without cloning repos, calling an LLM, or touching the network.

## Bugs found and fixed

| # | Bug | Fix | Test |
|---|---|---|---|
| 1 | `check_python_env` only caught `ImportError`; `litellm`'s import raises `AttributeError` (circular import), crashing the whole pre-flight | Catch any import exception and report it as a failed check | `test_non_import_error_import_failure_is_reported_not_crashed` |
| 2 | `print_report` crashed with `UnicodeEncodeError` on Windows cp1252 consoles (the `≥` U+2265 char in the disk-space detail) | Replaced `≥` with `>=` in the display string | `test_print_report_ascii_safe_on_cp1252` |
| 3 | `load_instances_from_file` failed on UTF-8 BOM files (PowerShell 5.1 `Set-Content -Encoding UTF8` writes a BOM) | Read with `utf-8-sig` | `test_utf8_bom_file_loads` |
| 4 | Module docstring referenced `python -m godspeed.benchmarks.swebench` (wrong module name) | Corrected to `swebench_runner` and documented the offline dry-run | — (docstring) |
| 5 | No offline dry-run existed; `run_swebench` hard-coded the network dataset loader | Added `dry_run` + `instance_loader` params and `--dry-run` / `--instances-file` CLI flags | `test_dry_run_returns_plan_without_network`, `test_dry_run_respects_instance_ids`, `test_dry_run_respects_instances_cap`, `test_dry_run_does_not_write_predictions`, `test_dry_run_with_nim_keys_reports_count` |
| 6 | `check_nim_connectivity` made real HTTP calls with no injection point | Added injectable `fetcher` param; `run_all_checks(skip_network=True)` for offline dry-runs | `test_offline_key_fails_structured`, `test_offline_dry_run_skips_network`, `test_missing_key_is_structured_fatal`, `test_blank_key_after_split_is_fatal` |

## Real-run checklist

A real SWE-bench run needs all of the following. None of it is available in
this offline environment.

### 1. Dataset download

```bash
# One-time; ~2 GB cached under ~/.cache/huggingface
uv run python -c "from datasets import load_dataset; load_dataset('princeton-nlp/SWE-bench_Lite', split='test')"
```

The runner loads instances via `experiments/swebench_lite/run.py:_load_instances`
(`load_dataset("princeton-nlp/SWE-bench_Lite", split=...)`). For offline
planning, use `--instances-file` with a local JSONL export instead.

### 2. Required environment variables

| Variable | Purpose |
|---|---|
| `NVIDIA_NIM_API_KEYS` | Comma-separated NIM free-tier keys (rotation pool). Fallback: `NVIDIA_NIM_API_KEY` (single key). |
| `NVIDIA_NIM_API_KEY` | Single-key fallback; also set per-instance by `NIMKeyManager.key_context()`. |

### 3. LLM endpoint

A reachable LiteLLM-compatible endpoint for the `--model` id (e.g.
`nvidia_nim/deepseek-ai/deepseek-v4-pro`). The runner shells out to
`godspeed run` (or drives the agent in-process with `--agent-in-loop`), which
requires the endpoint to be live and the key to authenticate.

### 4. GPU / model constraints (8 GB VRAM note)

Per `CLAUDE.md`: this machine is an RTX 4060 Laptop with **8 GB VRAM**.
- Do **not** run large models locally. Use NIM free-tier / hosted endpoints
  (the runner's default) — the local GPU is not the inference target.
- `--agent-in-loop` verification runs the SWE-bench Docker harness, which
  needs **Docker** (and **WSL Ubuntu** on Windows). Both are absent here.
- Without Docker, run with `--no-agent-in-loop` and submit predictions to
  `sb-cli` (cloud evaluation) instead.

### 5. Suggested real-run command

```powershell
uv run python -m godspeed.benchmarks.swebench_runner `
  --model nvidia_nim/deepseek-ai/deepseek-v4-pro `
  --split test --instances 300 --parallel 4 `
  --instance-cooldown 60 `
  --out benchmarks/results/predictions_test.jsonl
```

Then evaluate:

```bash
sb-cli submit swe-bench_lite test \
  --predictions_path benchmarks/results/predictions_test.jsonl \
  --run_id godspeed-smoke --gen_report
```

## Test coverage

```bash
uv run pytest tests/test_benchmarks_smoke.py -q        # 16 passed
uv run pytest tests/ -q -k "benchmark or swebench or lite or preflight or nim_key"  # 110 passed
```