# Local Qwen3.8-27B on a 16 GB GPU: what was measured

Measured on **one machine** on 2026-09-24: RTX 5070 Ti (16 GB, 16,303 MiB), Ryzen 5 7600, WSL2 Ubuntu,
llama.cpp `3173a56` (built 2026-08-29), model
[`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (10.44 GB, Apache-2.0). Treat every number as "what this box did",
not a promise for other hardware. VRAM figures include the ~1.3-1.9 GB the Windows desktop was using.

## Launch

`scripts/serve_qwen38_27b_llamacpp.sh` starts the exact configuration below. Godspeed on Windows cannot
launch a Linux binary, so start it in WSL and Godspeed attaches to `127.0.0.1:8080`; then use
`scripts/settings_local_llm_qwen38_27b.yaml`.

Key flags: `-c 32768 -fa on -np 1 --cache-type-k q8_0 --cache-type-v q8_0 -ub 512`
`--spec-type draft-mtp --spec-draft-n-max 4 --jinja`. MTP needs `-np 1` (one slot).

## Speed and VRAM (32K context, q8_0 KV)

| config | decode tok/s, real agent requests | decode tok/s, synthetic prompts | peak VRAM (MiB) |
|---|---|---|---|
| no speculation | 54.6 | 54.5 | 12,960 |
| MTP n=3 | 76.4 | 94.7 | 14,203 |
| **MTP n=4** | **78.2** | 101.0 | 14,128 |
| DFlash2 drafter, n=7 | 78.8 | 117.1 | 15,124 (`-ub 256`) |

* "Real agent requests" = 69 requests captured from an agent-in-loop run and replayed against each
  configuration (identical requests, at most 300 new tokens each). The live evaluations measured the same
  thing: 77.5 tok/s and 0.51 draft acceptance with MTP n=4 over 249 requests.
* "Synthetic prompts" (code, prose, tool-call and reasoning prompts written for the benchmark) overstate the
  speed-up: real output is mostly medium-effort reasoning prose, which the drafters accept poorly (DFlash2
  acceptance 0.36 on real requests vs 0.58 synthetic). Do not choose a speculation config from synthetic
  numbers alone.
* DFlash2 needs a separate drafter file and about 1 GB more VRAM for no measurable gain on real traffic, so
  MTP n=4 is the recommended setting.
* KV `q4_0` saves about 1 GiB with no speed change; its effect on quality was **not** measured.

## Prefix cache

The model is a Gated-DeltaNet hybrid, whose recurrent state cannot be rolled back. Measured on the real
Godspeed path (requests captured by a logging proxy): one system message per request, `reasoning_content` is
not sent back, and after the first request (about 7.9K tokens of system prompt + tool schemas in the full
harness) each agent step re-processes only 250-570 new tokens. Prefill is about 1.3K tok/s, so a full cache
miss at 24K context would cost about 17 s.

## Quality screen (small, read the caveats)

Godspeed's agent-in-loop SWE-bench runner (`experiments/swebench_lite`), 8 SWE-bench Lite **dev** instances,
`reasoning_effort: medium`, 40 iterations / 15 minutes per task, scored afterwards with the official swebench
harness. The 8 tasks were drawn in a seeded random order after excluding dev instances that the gold patch
itself cannot resolve in the published Docker images (8 of 23: all five pvlib ones - NumPy 2 removed
`np.Inf`; pyvista - missing `libGL.so.1`; two pydicom - test-infrastructure error).

| arm | draws | resolved |
|---|---|---|
| Qwen3.8-27B IQ3_XXS (this profile) | 3 (8 tasks each) | 7/8, 6/8, 6/8 = 19/24 (79%, Wilson 95% CI 60-91%) |
| KAT-Coder REAP-50 Q4_K_M, plain (32K, 32K, 64K context) | 3 (8 tasks each) | 4/8, 4/8, 4/8 = 12/24 (50%, CI 31-69%) |

* The agent can call the hidden-test harness as a tool (up to 5 times per task), so these scores are **not
  comparable to leaderboard numbers**. The draws share the same 8 tasks and are not independent. Per task,
  the 27B solved at least as many of its 3 draws as KAT on every task and strictly more on 5 of 8 (task-level
  sign test, two-sided p about 0.06): suggestive, not established. KAT's two draws with network access (32K
  and 64K) solved exactly the same four tasks, so its ceiling here is not the context window.
* Protocol differences between draws: the first 27B draw ran before the filesystem sandbox existed (its
  command log was audited: 1 of 165 shell commands touched anything outside the task workspace), and the
  first KAT draw ran in a sandbox that accidentally had no DNS. The other draws (27B: two, KAT: two) share
  one protocol: sandboxed, with network.
* `sqlfluff-1733` was unsolved by every draw of both arms; two other tasks flip between draws (about one
  task of noise per draw).
* Mean wall time per 8-task pass: 26 min (27B, three draws) vs 22 min (KAT, three draws), i.e. about 14.5
  vs 11.0 tasks solved per hour of agent time. KAT's plain decode measured 3.4x the 27B's plain and about
  2.4x its MTP n=4 speed on synthetic prompts (its real-workload speed was not measured separately), but its
  trajectories are longer and it solves fewer tasks: at 32K with network access 5 of its 8 tasks ended at the
  40-iteration cap (2 of 8 at 64K).

## Known issues

* **Context estimate.** Godspeed's token estimate (tiktoken cl100k) ran 1.29x below the server's true count on
  real Qwen3.8 requests (median), mostly because assistant `tool_calls` were not counted; with a hard 32K
  window compaction at 0.8 (0.8 x 1.29 is about 1.03 of the window) fired only when the real context was
  already at or past the limit (reply truncated mid tool call, llama-server HTTP 500, session lost). Fixed by counting `tool_calls` (median 1.11x after);
  until that is in your build, consider `compaction_threshold: 0.7`.
* **Benchmark hygiene.** The agent shell shares the host filesystem: during one run a model searched the host
  for hidden tests and gold data. Run benchmark agents in an isolated mount/PID namespace (or container) and
  give each task its own throwaway virtualenv; do not let the agent `pip install` into the harness venv.
* `pvlib` and a few other SWE-bench Lite instances cannot pass even with the gold patch in current images:
  gold-check an instance set before trusting a score on it.
