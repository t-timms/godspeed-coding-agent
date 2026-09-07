# SOTA Audit — Complete Agent Harness Benchmarks

## Audit Date: May 11, 2026 · Verified Against Live Leaderboards

*Soli Deo Gloria.*

---

## Complete Benchmarks Registry

After exhaustive research across all major coding agent research venues (ICLR, NeurIPS, ICML, arXiv) and live leaderboards, here is every SOTA-respected agent harness benchmark as of May 2026:

---

## Tier 1: Must-Have — Every SOTA Agent Reports These

### 1. SWE-bench Family *(THE standard)*

| Sub-benchmark | Instances | Scope | SOTA (May 2026) | Godspeed Target |
|---|:---:|---|:---:|:---:|
| **SWE-bench Verified** | 500 | Human-validated GitHub issues | **>74%** (Gemini 3 Pro + mini-SWE-agent v2) | 50%+ |
| **SWE-bench Lite** | 300 | Smaller evaluation subset | 62.7% (Claude Opus 4.6) | 50%+ |
| **SWE-bench Multilingual** | 300 (9 langs) | Cross-language bug fixes | TBD | 35%+ |
| **SWE-bench Multimodal** | 517 | Visual issue elements | TBD | TBD |

**Status (2026-02-23):** OpenAI retired **SWE-bench Verified** after an audit found
59.4% of audited hard-task test suites flawed plus training-data contamination.
**SWE-bench Pro** (proprietary-repo based) is the contamination-resistant reference —
frontier sits at 23–59% there vs ~75% on Verified, the honest residual. Verified/Lite
targets above are **baseline-only** (historical comparison); report Pro numbers going forward.

**Paper:** [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770) (Jimenez et al., ICLR 2024)  
**Leaderboard:** [swebench.com](https://www.swebench.com)

### 2. Aider Polyglot *(multi-language code editing)*

| Metric | SOTA (May 2026) | Godspeed Target |
|---|---|---|
| Pass@2 (225 exercises, 6 langs) | **88.0%** (gpt-5 high) | 65%+ |
| DeepSeek V3.2 Exp Reasoner | 74.2% ($1.30) | reference |
| Kimi K2 | 59.1% ($1.24) | Godspeed baseline driver |

**Leaderboard:** [aider.chat/docs/leaderboards](https://aider.chat/docs/leaderboards/)  
**Paper:** [aider.chat/2024/12/21/polyglot](https://aider.chat/2024/12/21/polyglot.html)

### 3. LiveCodeBench *(contamination-free, continuously updated)*

| Scenario | SOTA (May 2026) | Godspeed Target |
|---|---|---|
| Code Generation | ~60% (Claude Opus 4, GPT-5) | 45%+ |
| Code Execution | ~70% (GPT-5) | 55%+ |
| Test Output Prediction | ~65% (Claude Opus 4) | 50%+ |

**Paper:** [arxiv.org/abs/2403.07974](https://arxiv.org/abs/2403.07974) (Jain et al., 2024)  
**Leaderboard:** [livecodebench.github.io](https://livecodebench.github.io/leaderboard.html)

### 4. Terminal-Bench *(operational reliability)*

| Metric | SOTA (May 2026) | Godspeed Target |
|---|---|---|
| TB 2.1 (89 containerized terminal tasks, 5 attempts, Harbor) | **84.6%** (public leaderboard) | 50%+ |
| TB 2.0 harness variance (same model, harness-only change) | 58.0% → 74.7% | reference |

Measures operational reliability: driving a real containerized shell to a correct
end state across 89 human-validated terminal tasks, 5 attempts each, run via the
Harbor framework. Harness variance is a first-class concern — the same model moved
58.0% → 74.7% purely by harness change on TB 2.0, so report the harness version
with every score.

**Leaderboard:** [terminal-bench.com](https://terminal-bench.com)  
**Repo:** [github.com/terminal-ai/terminal-bench](https://github.com/terminal-ai/terminal-bench)

---

## Tier 2: Strongly Recommended — Widely Cited

### 5. BigCodeBench *(practical, diverse programming tasks)*

| Variant | SOTA (May 2026) | Godspeed Target |
|---|---|---|
| Full Instruct (1140 tasks, Pass@1) | ~48% (DeepSeek V3.2) | 40%+ |
| Hard Instruct (150 tasks, Pass@1) | ~40% | 30%+ |

**Paper:** [openreview.net/forum?id=YrycTjllL0](https://openreview.net/forum?id=YrycTjllL0) (ICLR 2025 Oral)  
**Leaderboard:** [bigcode-bench.github.io](https://bigcode-bench.github.io)

### 6. CRUXEval *(code reasoning & execution)*

| Sub-benchmark | What it tests |
|---|---|
| CRUXEval-I | Input prediction — given code + output, predict input |
| CRUXEval-O | Output prediction — given code + input, predict output |

**Paper:** [crux-eval.github.io/paper/cruxeval.pdf](https://crux-eval.github.io/paper/cruxeval.pdf) (Facebook Research)  
**Leaderboard:** [crux-eval.github.io](https://crux-eval.github.io/leaderboard.html)

### 7. Spider 2.0 *(enterprise text-to-SQL workflows)*

| Sub-benchmark | Instances | Type | SOTA |
|---|---|---|---|
| Spider 2.0-Snow | 547 | Text-to-SQL on Snowflake | **96.7%** (Genloop Sentinel v2 Pro) |
| Spider 2.0-DBT | 68 | **Code agent** task (dbt projects) | **58.8%** (Databao Agent) |
| Spider 2.0-Lite | 547 | Text-to-SQL multi-DB | **72.0%** (SOMA-SQL) |

**Paper:** [arxiv.org/abs/2411.07763](https://arxiv.org/abs/2411.07763) (ICLR 2025 Oral)  
**Leaderboard:** [spider2-sql.github.io](https://spider2-sql.github.io)

### 8. EvalPlus (HumanEval+/MBPP+) *(extended test cases)*

| Sub-benchmark | Instances | What it tests |
|---|---|---|
| HumanEval+ | 164 | Extended tests beyond original HumanEval |
| MBPP+ | 399 | Extended tests beyond original MBPP |

**Paper:** [openreview.net/forum?id=1qvx610Cu7](https://openreview.net/forum?id=1qvx610Cu7) (NeurIPS 2023)  
**Leaderboard:** [evalplus.github.io](https://evalplus.github.io/leaderboard.html)

---

## Tier 3: Emerging / Domain-Specific

### 9. CodeClash *(goal-oriented development)* ⭐ NEW Nov 2025
SWE-bench family expansion: evaluates agents as goal-oriented developers building apps and websites from natural language descriptions.  
**URL:** [codeclash.ai](https://codeclash.ai)

### 10. SWE-bench Multilingual *(9 languages)*
Cross-language bug fixes: Python, Java, JavaScript, TypeScript, C, C++, Go, Rust, Ruby.  
**URL:** [swebench.com/multilingual](https://www.swebench.com/multilingual-leaderboard.html)

### 11. RepoBench *(repository-level code completion)*
Cross-file code understanding at real repository scale.  
**Paper:** [arxiv.org/abs/2306.03091](https://arxiv.org/abs/2306.03091) (Liu et al., 2023)

### 12. CrossCodeEval *(cross-file completion)*
Multi-file understanding across Python, Java, TypeScript, C#.  
**URL:** [github.com/amazon-science/cceval](https://github.com/amazon-science/cceval)

### 13. Spider 2.0-DBT *(dbt code agent)*
68 real-world dbt project tasks — the only SQL "code agent" benchmark. Spider-agent baseline gets 14.7% with Sonnet, SOTA is 58.8%. This is underserved — godspeed's agent loop could dominate.  
**URL:** [spider2-sql.github.io](https://spider2-sql.github.io)

---

## Summary: All 13 Benchmarks

| # | Benchmark | Relevance to Godspeed | Status in Plan |
|---|-----------|:---:|:---:|
| 1 | SWE-bench Verified (500) | ⭐⭐⭐ Primary | ✅ Runner built |
| 2 | SWE-bench Lite (300) | ⭐⭐⭐ Primary | ✅ Runner built |
| 3 | Aider Polyglot (225) | ⭐⭐⭐ Primary | 📋 Planned |
| 4 | Terminal-Bench 2.1 (89) | ⭐⭐⭐ Primary | 📋 Planned |
| 5 | LiveCodeBench (300+) | ⭐⭐ Secondary | 📋 Planned |
| 6 | BigCodeBench (1140) | ⭐⭐ Secondary | 📋 Planned |
| 7 | CRUXEval | ⭐⭐ Secondary | ❌ Missing |
| 8 | Spider 2.0-DBT (68) | ⭐⭐ Agent niche | ❌ Missing |
| 9 | EvalPlus (HumanEval+/MBPP+) | ⭐ Model-level | ❌ Missing |
| 10 | CodeClash | ⭐ NEW | ❌ Missing |
| 11 | SWE-bench Multilingual (300) | ⭐ Niche | ❌ Missing |
| 12 | RepoBench | ⭐ Niche | ❌ Missing |
| 13 | CrossCodeEval | ⭐ Niche | ❌ Missing |

**Realistic target:** Tier 1 + Tier 2 (8 benchmarks) = covers 100% of what respected agents report.

---

## Reliability Gates

Benchmark scores are only meaningful if the underlying test runs are trustworthy:

- **pass^k in `test_runner`:** the `test_runner` tool accepts a `repeat` param
  (1–5, clamped). With `repeat=k`, the same test command runs up to k times and
  the verdict is PASS only if **all** k runs pass (k-of-k), stopping early on the
  first failure. Counters flaky-test reward hacking — a test that passes 1-in-3
  times no longer counts as green.
- **Read-only test dirs:** protect test fixtures from agent edits via FileWrite
  deny rules in `settings.yaml` (see `settings.yaml.example`, e.g.
  `FileWrite(tests/*)`), so the agent cannot "fix" a failing test to make it pass.

---

## Godspeed Mini ("Lightspeed") — The Strategy

Yes — create a bash-only mini version for SWE-bench, just like mini-SWE-agent did.

### Architecture comparison

| Dimension | Godspeed (full) | Godspeed Mini | mini-SWE-agent v2 |
|---|---|---|---|
| Agent code | 28K lines | 150 lines | 100 lines |
| Tools | 30+ custom | Bash only (`subprocess.run`) | Bash only |
| Shell model | Async stateful sessions | Stateless `subprocess.run` | Stateless `subprocess.run` |
| History | Hash-chained context | Linear append | Linear append |
| Permission engine | 4-tier deny-first | **Disabled for benchmarks** | N/A |
| Action format | Tool-calling API JSON | Plain text → bash | Plain text → bash |
| Model routing | Fallback chains | Model roulette (random per step) | Single model |
| Driver models | Any LiteLLM | Same (reuses LLM client) | Any LiteLLM |
| Audit trail | SHA-256 JSONL | None | None |

### What Godspeed Mini inherits from full Godspeed

| Feature | Value for benchmarks |
|---|---|
| LiteLLM client | 200+ providers, works with NIM free keys |
| NIM key rotation | 120 RPM, $0 cost |
| Pre-flight checks | Validates Docker, keys, disk before run |
| Resume checkpointing | Recovers from crash at last instance |
| RunLogger | Structured logs + heartbeat + failure forensics |
| Budget prompt injection | Prevents over-editing (the agent-in-loop fix) |
| Prediction format | sb-cli compatible JSONL |

### Expected impact

```
Godspeed full (agent-in-loop, Kimi K2.5):  30.4%  ← current baseline
Godspeed full (agent-in-loop, DeepSeek V4): 35-45% ← projected with better model
Godspeed Mini (bash-only, DeepSeek V4):     50-60% ← projected with simple agent
Godspeed Mini (bash + roulette + 4 keys):   55-65% ← projected with ensemble
SOTA (mini-SWE-agent + Gemini 3 Pro):       >74%   ← ultimate ceiling
```

### Implementation plan — Godspeed Mini

```python
# src/godspeed/agent/mini.py — ~150 lines
class MiniAgent:
    """Bash-only agent loop for benchmark runs."""
    
    def __init__(self, model: str, workdir: Path):
        self.llm = LLMClient(model)
        self.workdir = workdir
        
    async def run(self, problem: str, max_steps: int = 40) -> str:
        messages = [
            {"role": "system", "content": MINI_SYSTEM_PROMPT},
            {"role": "user", "content": f"Fix this issue:\n\n{problem}"},
        ]
        for step in range(max_steps):
            response = await self.llm.chat(messages)
            action = self._extract_bash(response.content)
            result = subprocess.run(action, shell=True, cwd=self.workdir, 
                                    capture_output=True, text=True, timeout=120)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": f"Output:\n{result.stdout}{result.stderr}"})
            if "SUBMIT_PATCH" in response.content:
                break
        return self._capture_diff()
```

### Run strategy

```bash
# Phase 1: Godspeed Mini (bash-only) for SWE-bench scores
godspeed-mini --model nvidia_nim/deepseek-ai/deepseek-v4-pro --benchmark swebench_verified

# Phase 2: Godspeed Full submits same results (it's the same agent family)
# Report as "Godspeed v0.5 with Mini scaffold"
```

This is the exact strategy mini-SWE-agent used to beat full SWE-agent. Less is more.

---

## Final Recommendation

1. **Build Godspeed Mini** — 150-line bash-only agent, Monday morning, ship by Wednesday
2. **Run all 8 Tier 1+2 benchmarks** with Mini + DeepSeek V4 Pro via NIM (free)
3. **Report honestly**: "Godspeed Mini (bash-only scaffold) + Godspeed Full (security-first agent)"
4. **Differentiate**: No other agent has both a benchmark-winning minimal scaffold AND a production-grade secured agent with 4-tier permissions + audit trail + Windows support

This gives Godspeed two lanes:
- **Benchmark lane:** Godspeed Mini (bash-only, competes with mini-SWE-agent)
- **Production lane:** Godspeed Full (secure, auditable, 30+ tools — what you actually deploy)
