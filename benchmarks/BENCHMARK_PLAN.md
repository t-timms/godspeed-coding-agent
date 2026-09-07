# Godspeed Benchmark Plan

## SOTA Targets & Execution Strategy

> **Driver:** `deepseek-v4-pro` via NVIDIA NIM — 4-key rotation, free tier, $0 cost.
> **Secondary:** `deepseek-v4-flash` for cheap oracle-merge ensembles.
> **Validated:** May 11, 2026. All SOTA numbers confirmed against live leaderboards.
>
> *Soli Deo Gloria.*

---

## Benchmarks

### Primary

| # | Benchmark | Instances | Category | SOTA (May 2026) | Godspeed Target |
|---|-----------|:---:|---|:---:|:---:|
| 1 | **SWE-bench Verified** | 500 | Agent (bug-fix) | 65%+ (mini-swe-agent v2) | 50%+ |
| 2 | **SWE-bench Lite** | 300 | Agent (bug-fix) | 62.7% (Claude Opus 4.6) | 50%+ |
| 3 | **Aider Polyglot** | 225 | Multi-lang edit | 88.0% (gpt-5 high) | 65%+ |

### Secondary

| # | Benchmark | Instances | Category | SOTA (May 2026) | Godspeed Target |
|---|-----------|:---:|---|:---:|:---:|
| 4 | **LiveCodeBench** | 300+ | Contamination-free | ~60% (Claude Opus 4) | 45%+ |
| 5 | **BigCodeBench** | 1140 | Practical tasks | ~48% (DeepSeek V3.2) | 40%+ |

### Future

| # | Benchmark | Instances | Category | Notes |
|---|-----------|:---:|---|---|
| 6 | **CodeClash** | TBD | Goal-oriented | New Nov 2025, agents as goal-oriented devs |

---

## Time & Resource Budget

### Per-instance estimates

| Metric | Single-shot | Agent-in-loop |
|--------|:---:|:---:|
| Wall time per instance | 45–90s | 90–180s |
| LLM calls per instance | 1–3 | 8–15 |
| Token budget per instance (input) | 15K–40K | 40K–80K |
| Token budget per instance (output) | 2K–8K | 8K–20K |

### Full-run estimates (parallel=4, agent-in-loop, NIM free tier)

| Benchmark | Instances | Time (est.) | LLM calls (est.) | RPM @4 workers |
|-----------|:---:|:---:|:---:|:---:|
| SWE-bench Lite | 300 | **1.5–2.5h** | 2.4K–4.5K | 20–40 |
| SWE-bench Verified | 500 | **2.5–4h** | 4K–7.5K | 20–40 |
| Aider Polyglot | 225 | **1–1.5h** | 1.8K–3.4K | 18–36 |
| LiveCodeBench | 300 | **1–2h** | 1.5K–3K | 12–25 |
| BigCodeBench | 400 | **1.5–2.5h** | 2K–4K | 12–25 |
| **Total** | **1,725** | **7.5–12.5h** | 11.7K–22.4K | |

RPM safety: 4 keys × 30 = 120 RPM capacity. Peak worker load ~40 RPM. **3× headroom.**

### API cost: $0

NVIDIA NIM free tier — all benchmarks run at zero cost with 4-key rotation.
If switching to DeepSeek direct: ~$165 for all benchmarks.

---

## Reliability Architecture

### Failure Modes & Mitigations

| Failure Mode | Impact | Mitigation |
|---|---|---|
| NIM HTTP 429 (rate limit) | Instance fails | 4-key rotation + exponential backoff + jitter |
| NIM HTTP 503 (overload) | Instance fails | Retry with next key, cooldown offending key |
| Agent hangs (infinite loop) | Instance blocks forever | Per-instance timeout (600s), budget prompt at N writes |
| Agent crash (SEGFAULT/OOM) | Instance lost | Process-level timeout, restart from checkpoint |
| Docker unavailable (WSL) | All instances fail | Pre-flight check, graceful skip with error log |
| Disk full (repo clones) | Run stops | Instance-level temp dirs deleted immediately after use |
| Network blip (transient) | Instance fails | LLMClient retry with backoff, key rotation retry |
| Power loss / system crash | Full run lost | Checkpoint JSONL, resume from last completed instance |

### Per-instance safety net

```
┌─────────────────────────────────────────────────────┐
│  FOR EACH INSTANCE                                   │
│                                                      │
│  ┌──────────┐   ┌───────────┐   ┌───────────────┐  │
│  │ Pre-check│ → │ Agent run │ → │ Patch capture │  │
│  │ (clone)  │   │ (600s cap)│   │ (git diff)    │  │
│  └──────────┘   └─────┬─────┘   └───────────────┘  │
│                       │                               │
│                  ┌─────▼─────┐                        │
│                  │ Timeout?  │───Yes──→ log + skip   │
│                  └─────┬─────┘                        │
│                        │No                            │
│                  ┌─────▼─────┐                        │
│                  │ Empty     │───Yes──→ log warning   │
│                  │ patch?    │                        │
│                  └─────┬─────┘                        │
│                        │                              │
│                  ┌─────▼─────┐                        │
│                  │ Write     │                        │
│                  │ prediction│                        │
│                  │ + metrics │                        │
│                  └───────────┘                        │
└─────────────────────────────────────────────────────┘
```

### Resume checkpointing

Predictions are written immediately after each instance completes (atomic append).
The runner reads the predictions file on startup and skips already-completed IDs.
A crash at instance 247 of 500 resumes at 248 — no work lost.

---

## Pre-Flight Checklist

Before starting any benchmark run:

```bash
# 1. Verify NIM keys
python -m godspeed.benchmarks.preflight --check-nim

# 2. Verify Docker (SWE-bench only)
python -m godspeed.benchmarks.preflight --check-docker

# 3. Verify disk space (>20 GB free)
python -m godspeed.benchmarks.preflight --check-disk

# 4. Run all checks
python -m godspeed.benchmarks.preflight --all
```

---

## Run Commands

```bash
# --- Setup ---
export NVIDIA_NIM_API_KEYS="nvapi-key1,nvapi-key2,nvapi-key3,nvapi-key4"
export GODSPEED_LOG_LEVEL=INFO

# --- Quick validation (23 dev instances, ~15 min) ---
python -m godspeed.benchmarks.swebench \
    --model nvidia_nim/deepseek-ai/deepseek-v4-pro \
    --split dev \
    --instances 23 \
    --agent-in-loop

# --- SWE-bench Lite (300 instances, ~2h) ---
python -m godspeed.benchmarks.swebench \
    --model nvidia_nim/deepseek-ai/deepseek-v4-pro \
    --split test \
    --instances 300 \
    --agent-in-loop \
    --parallel 4 \
    --resume

# --- SWE-bench Verified (500 instances, ~3.5h) ---
python -m godspeed.benchmarks.swebench \
    --model nvidia_nim/deepseek-ai/deepseek-v4-pro \
    --split test \
    --instances 500 \
    --agent-in-loop \
    --parallel 4 \
    --resume

# --- Submit to official leaderboard ---
sb-cli submit swebench_lite test \
    --predictions_path benchmarks/results/predictions_test.jsonl \
    --run_id godspeed-v4pro-v0.5 \
    --gen_report
```

---

## Output Structure

```
benchmarks/results/
├── predictions_dev.jsonl       # SWE-bench predictions (dev split)
├── predictions_test.jsonl      # SWE-bench predictions (test split)
├── metrics_dev.jsonl           # Per-instance metrics (dev)
├── metrics_test.jsonl          # Per-instance metrics (test)
├── reports/                    # sb-cli generated reports
│   └── godspeed-v4pro-v0.5/
│       ├── report.json
│       └── instances/
├── logs/                       # Per-run structured logs
│   └── run_2026-05-11_01/
│       ├── main.log
│       ├── failures.log
│       └── summary.json
└── checkpoint.json             # Last completed instance ID
```

---

## Research References

| Reference | Venue | Link |
|-----------|-------|------|
| SWE-bench | ICLR 2024 | [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770) |
| SWE-bench Verified | OpenAI Blog | [swebench.com/verified](https://www.swebench.com/verified.html) |
| ExpeRepair v1.0 | arXiv | [arxiv.org/abs/2503.08715](https://arxiv.org/abs/2503.08715) |
| LiveCodeBench | arXiv | [arxiv.org/abs/2403.07974](https://arxiv.org/abs/2403.07974) |
| BigCodeBench | ICLR 2025 Oral | [openreview.net/forum?id=YrycTjllL0](https://openreview.net/forum?id=YrycTjllL0) |
| RepoBench | arXiv | [arxiv.org/abs/2306.03091](https://arxiv.org/abs/2306.03091) |
| CrossCodeEval | Amazon Science | [github.com/amazon-science/cceval](https://github.com/amazon-science/cceval) |
| EvalPlus (HumanEval+/MBPP+) | NeurIPS 2023 | [evalplus.github.io](https://evalplus.github.io/leaderboard.html) |
| CodeClash | SWE-bench Family | [codeclash.ai](https://codeclash.ai) |
| EntroPO + R2E | arXiv | [arxiv.org/abs/2502.09496](https://arxiv.org/abs/2502.09496) |
| SWE-smith (training) | GitHub | [swesmith.com](https://swesmith.com) |
