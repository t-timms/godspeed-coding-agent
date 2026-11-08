# Adding a new LLM driver to Godspeed

Godspeed's driver layer is **config-driven**. You add a new model via
`src/godspeed/llm/driver_catalog.yaml` plus a smoke-test pass — no
Python code changes. This doc walks through the process end-to-end.

## The 3-step flow

1. **Add the catalog entry** — record the model's properties in
   `src/godspeed/llm/driver_catalog.yaml`.
2. **Smoke-test** — run `scripts/validate_driver.py --model <name>`.
3. **Use it** — pass `--model <name>` to any Godspeed command.

That's it. Steps 1 and 3 are config; step 2 gates the driver against
regressions.

---

## Step 1 — catalog entry

Open `src/godspeed/llm/driver_catalog.yaml` and append an entry. The
key is the **LiteLLM model string** (provider prefix + model name).

Minimum viable entry:

```yaml
  moonshot/kimi-k2.7:
    provider: moonshot
    context_window: 262144
    prompt_profile: default
    tool_call_format: openai
    requires_env: MOONSHOT_API_KEY
    cost_per_mtok_in: 1.10
    cost_per_mtok_out: 4.50
    notes: "Next Kimi release, hypothetical April 2026"
```

### Field reference

| Field | Required | Notes |
|---|---|---|
| `provider` | yes | LiteLLM provider prefix — `nvidia_nim`, `moonshot`, `anthropic`, `openai`, `ollama`, `azure`, etc. |
| `context_window` | yes | Hard token ceiling. Godspeed compacts at 0.8× this. |
| `prompt_profile` | yes | `default`, `thinking`, or `minimal`. See [profile guide](#prompt-profile-selection). |
| `tool_call_format` | yes | `openai` (standard tools array) or `xml` (inline, Anthropic-style). Most models: `openai`. |
| `requires_env` | if API key needed | Name of the env var. Keep the key in `~/.godspeed/.env.local`, **never** in source. |
| `cost_per_mtok_in` | yes | USD per 1M input tokens. `0.0` for free tiers. |
| `cost_per_mtok_out` | yes | USD per 1M output tokens. |
| `known_ceilings` | optional | Resolve-rate dict if published or measured. Useful for sanity-checking runs. |
| `notes` | optional | One-line freeform operator context. |

### Prompt profile selection

Pick the closest profile for the model family:

| Profile | Use for | Examples |
|---|---|---|
| `default` | General instruction-tuned chat models | Kimi K2.5/K2.6, Claude Sonnet, GPT-4o, Qwen-Coder |
| `thinking` | Extended-thinking / reasoning models (internal chain-of-thought) | Qwen3-Next Thinking, DeepSeek R1, Kimi-K2-thinking, o1/o3-style |
| `minimal` | Tiny or structure-rigid models | Ollama qwen3:4b, gemma3:1b |

If you're unsure, start with `default`. You can adjust later if
`validate_driver.py` shows a high `agent_exit_4` rate (indicates prompt
mismatch) or empty-patch rate >20%.

---

## Step 2 — smoke test

Run the validator:

```bash
python scripts/validate_driver.py --model moonshot/kimi-k2.7
```

What it does:
1. Loads the catalog entry. Warns if missing (not fatal — you can smoke
   before adding to the catalog).
2. Runs the driver on 3 easy SWE-Bench Lite dev instances
   (`sqlfluff__sqlfluff-2419`, `pydicom__pydicom-1256`,
   `marshmallow-code__marshmallow-1343`; each one resolves with its gold patch in the
   current SWE-bench images, see `docs/benchmark_scoring.md`).
3. Checks gate criteria:
   - **LLM-error rate ≤ 20%** (driver actually answers).
   - **At least 1/3 instances produce real work** (agent isn't stuck).
4. Exit 0 on pass, 1 on fail, 2 on setup error.

Expected wall-clock: ~5-15 minutes (3 instances × agent-in-loop cost).

### If it fails

- **High `llm_error` rate** → check the API key, check rate-limiting,
  check that the provider actually supports the model string.
- **Zero non-empty patches** → try a different `prompt_profile`
  (`thinking` ↔ `default` is the common flip).
- **Tool-calling schema mismatch** → your model may need
  `tool_call_format: xml` instead of `openai`.

---

## Step 3 — use it

Once smoke passes, the driver is a first-class citizen:

```bash
# CLI
godspeed -m moonshot/kimi-k2.7 "fix the bug in src/foo.py"

# SWE-Bench benchmarking
python experiments/swebench_lite/run.py \
  --model moonshot/kimi-k2.7 \
  --split dev --agent-in-loop
```

**To use in ensembles (Phase 4):** just include the model string in the
`--drivers` list. No further setup.

---

## Worked example: adding Kimi K2.6

K2.6 was released on Moonshot's direct API in April 2026 and isn't yet
on NVIDIA NIM's free tier. Here's the full process we followed:

### 1. Save the key

```bash
# ~/.godspeed/.env.local (gitignored, never committed)
echo "MOONSHOT_API_KEY=<YOUR_KEY_FROM_PLATFORM.MOONSHOT.AI>" >> ~/.godspeed/.env.local
```

### 2. Catalog entry

```yaml
  moonshot/kimi-k2.6:
    provider: moonshot
    context_window: 262144
    prompt_profile: default
    tool_call_format: openai
    requires_env: MOONSHOT_API_KEY
    cost_per_mtok_in: 0.95
    cost_per_mtok_out: 4.00
    known_ceilings:
      swe_bench_verified_claimed: 0.802  # per Moonshot, not reproduced
    notes: "Flagship; requires funded account. 80.2% Verified claimed."
```

### 3. Smoke

```bash
python scripts/validate_driver.py --model moonshot/kimi-k2.6
```

If the account has credit, this runs 3 instances and reports pass/fail.
If not, the smoke will fail with `quota_exceeded` errors — fund the
account and retry.

### 4. Use

```bash
python experiments/swebench_lite/run.py \
  --model moonshot/kimi-k2.6 \
  --split dev --agent-in-loop \
  --out experiments/swebench_lite/predictions_k2_6.jsonl
```

Done. New model is live, benchmark-able, ensemble-compatible. No code
changed.

---

## Design rationale (why this separation)

**The catalog is the single source of truth** for model-specific
behavior. The alternative — sprinkling `if model.startswith("moonshot")`
conditionals across the codebase — makes it hard to add a new provider
without touching many files. The catalog pattern keeps the code model-
agnostic and the config model-specific.

**`validate_driver.py` is the gate** because an untested driver in the
ensemble pool drags benchmark numbers down. The script is cheap (5-15
min) compared to a full dev-23 run (~1 hour), so it's affordable to run
before every driver swap.

**Prompt profiles are coarse on purpose.** Three buckets
(`default`/`thinking`/`minimal`) covers 95% of real models. Finer-grained
tuning lives in per-experiment prompt blocks
(e.g. `IN_LOOP_BLOCK` in `experiments/swebench_lite/run.py`).

---

## See also

- `src/godspeed/llm/driver_catalog.yaml` — the catalog itself
- `src/godspeed/agent/prompt_profiles.py` — the profile resolver
- `scripts/validate_driver.py` — the smoke-test runner
- `settings.yaml.example` — example user config with sample model blocks
- `experiments/swebench_lite/findings_2026_04_20.md` — benchmark results
  from the driver shootout that established these profiles
