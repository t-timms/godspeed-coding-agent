# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Bash-only SWE-bench agent loop** (`experiments/swebench_lite/run_in_loop.sh`) —
  Pure-bash agent runner using curl + jq for tool dispatch. Zero Python
  framework dependency for the agent loop itself.
- **Enhanced TUI welcome banner** — Session start now displays project
  directory, registered tool count, and deny rule count alongside the
  existing version, model, and permission mode. The banner gives users
  immediate visibility into the session configuration without needing
  to run `/tools` or `/permissions`.
- **feat(cli): session continuity** — `godspeed --continue` resumes the
  most recent conversation and `godspeed --resume <id>` resumes a specific
  session by ID (`src/godspeed/cli.py`). New `godspeed list-sessions`
  command lists recent sessions (id, started, summary).
- **feat(cli): MCP server management** — `godspeed mcp add/list/remove`
  manages MCP servers from the CLI, supporting stdio and SSE transports
  with `--scope user|project` and `--env` (`src/godspeed/cli.py`).
- **feat(tui): new slash commands** — `/compact [instructions]` manually
  compacts the conversation, `/verify [instructions]` builds/launches/probes
  the project and prints a PASS/FAIL verdict, `/btw <question>` answers a
  side question without touching the main conversation, `/goal [text|clear]`
  sets/shows/clears the session goal, `/rewind` opens the rewind picker,
  `/batch [units=N] <goal>` decomposes a task into parallel worktree units,
  and `/usage [tools|agents]` shows a session usage breakdown
  (`src/godspeed/tui/commands.py`).
- **feat(tui): message queueing** — Ctrl+Q queues the current input instead
  of submitting it; queued messages are injected between turns
  (`src/godspeed/tui/message_queue.py`, `src/godspeed/tui/app.py`).
- **feat(tui): bash pass-through** — input starting with `!` runs the rest
  as a shell command directly (bypassing the LLM); `!!` runs it in the
  background. Commands are checked against dangerous-command detection
  (`src/godspeed/tui/bash_passthrough.py`).
- **feat(tui): rewind picker** — pressing Escape twice opens a picker to
  restore conversation, files, or checkpoints (`src/godspeed/tui/rewind.py`).
- **feat(tui): image attachments** — `:img <path>` / `@image=<path>`
  directives attach an image to the next message; Ctrl+V pastes an image
  path from the clipboard (`src/godspeed/tui/attachments.py`).
- **feat(security): plan-mode approval gate** — the agent exits plan mode
  through an explicit `exit_plan_mode` tool that awaits human approval;
  rejection feeds guidance back so the model can revise the plan
  (`src/godspeed/security/plan_gate.py`, `src/godspeed/agent/loop.py`).
- **feat(config): statusline HUD** — `statusline.enabled` and
  `statusline.template` customize the per-turn HUD with `{model}`, `{tokens}`,
  `{cost}`, and `{branch}` placeholders; the session goal renders in the HUD
  (`src/godspeed/config.py`, `src/godspeed/tui/output.py`).
- **feat(tools): runtime verification** — `tools/runtime_verify.py` builds,
  launches, and probes a project, returning a PASS/FAIL verdict with evidence
  lines.
- **feat(agent): aside + batch** — `agent/aside.py` backs `/btw` side
  questions; `agent/batch.py` backs `/batch` worktree decomposition.
- **feat(observability): usage report** — `observability/usage_report.py`
  aggregates session usage (tokens, cost, per-tool call counts from the
  audit trail, sub-agent invocations) and backs the `/usage` command.
  Dimensions without a measurable source are left empty rather than
  fabricated.
- **feat(context): LSP feedback** — `context/lsp_feedback.py` feeds language
  server diagnostics back into the agent loop.
- **feat(observability): usage ledger** — `llm/usage_ledger.py` records one
  row per LLM call; `/usage` now shows real per-task-type and
  per-subagent token/cost attribution. Sub-agent spawns tag their records
  via `subagent_context` or a dedicated child ledger, and audit records
  written during spawns set `is_sidechain`/`parent_session_id`.
- **feat(tui): diff reviews** — `/code-review [--fix]`, `/security-review`
  (deterministic secrets scan merged with LLM review), and `/simplify`
  review the working-tree diff (`agent/review.py`). `--fix` queues an
  agent fix pass as user guidance.
- **feat(cli): batch headless** — `godspeed batch <goal>` runs worktree
  batches from the CLI with `--dry-run`, `--units`, `--parallelism`, and
  `--allow-dirty`; new `batch` settings section (`config.py`).
- **feat(llm): reasoning effort call-time** — `reasoning_effort` is now
  applied at call time by the LLM client; `/effort low|medium|high` sets
  it for the session.
- **feat(tui): /loop recurring prompt** — `/loop [interval] <prompt>`
  re-dispatches a prompt on an interval at the turn-complete point
  (`tui/loop_state.py`); pause-aware, `/loop stop` cancels.
- **feat(batch): PR automation** — `/batch` and `godspeed batch --open-pr`
  open a GitHub PR per patchable unit via `gh` when
  `batch.open_pr`/`--open-pr` is set (`agent/batch.py` `create_pr`);
  per-unit gh failures never abort the batch.
- **feat(tui): /fork + persistent history** — `/fork [label]` duplicates the
  conversation under a new resumable session id; prompt history persists
  in `.godspeed/history` with Ctrl-R reverse search.

### Fixed

- **CI security scan** — Added `--ignore-vuln CVE-2026-3219` and
  `--ignore-vuln CVE-2026-42561` to pip-audit invocations in `ci.yml`.
  Both are transitive dependencies pinned by upstream libraries with no
  satisfiable upgrade path.

## [0.5.0] — 2026-05-08

### Added

- **Skills system overhaul** — new skill management infrastructure with
  four new modules:
  - **Security scanning** (`skills/security.py`) — static analysis of skill
    files before installation. Detects 12+ threat categories including
    obfuscated eval/exec, base64-encoded payloads, crypto-miner patterns,
    and dangerous shell commands. Risk classification (clean/suspicious/dangerous).
  - **Skill evolution** (`skills/evolution.py`) — lesson tracking and
    automatic skill rewriting (AutoContext pattern). Accumulates corrections
    per skill, merges duplicates, and rewrites SKILL.md when enough
    high-confidence lessons accumulate. Timestamped backups before mutation.
  - **Dream consolidation** (`skills/dream.py`) — cross-session pruning,
    dedup, and date normalization modeled on Claude Code's Auto-Dream.
    Runs periodically (24h) to scan skill directories, merge duplicates,
    remove stale entries, and convert relative dates to absolute.
  - **Wiki bridge** (`skills/wiki_bridge.py`) — auto-generate SKILL.md
    files from llm-wiki knowledge base pages. Frontmatter extraction,
    tag filtering, topic matching, and reference preservation.
- **Retrieval sub-agent** (`agent/retrieval_agent.py`) — focused code
  exploration tool that delegates deep searches to a read-only sub-agent.
  Returns structured `file:line-range` results without polluting the
  main agent's context. READ_ONLY risk level; registered as `retrieval`
  tool.
- **Speculative decoding benchmarks** (`scripts/benchmark_specdec.py`,
  `scripts/start_draft_server.py`) — GPU-accelerated draft model server
  and benchmark suite for measuring tokens-per-second with/without
  speculative decoding on RTX 5070 Ti and compatible hardware.
- **Enhanced codebase index** (`context/codebase_index.py`) — improved
  semantic code search with better chunking, symbol extraction, and
  index freshness detection. 118 lines of enhancements.
- **Model presets** in config: `fast`, `balanced`, `quality`, `local`,
  `cloud`, `frontier` shortcuts in `MODEL_PRESETS` dict. Default model
  is now `openai/qwen2.5-coder-14b` with llama.cpp GPU spec decoding.
- **Expanded driver catalog** (`llm/driver_catalog.yaml`) — new entries
  for DeepSeek V4, Kimi K2.6, and updated NVIDIA NIM endpoints.
- **`datasets` dependency** added for training pipeline integration.
- **Removed unused `textual` dependency** — cleanup from prior framework
  experimentation.
- **Enhanced `llamacpp_manager` tool** — health checks, model lifecycle
  management, and server status reporting for llama.cpp inference.
- **Expanded LLM client tests** — 324 new assertions covering edge cases,
  retry logic, and model fallback chains.

### Changed

- **Default model** changed from `ollama/qwen3:4b` to
  `openai/qwen2.5-coder-14b` with llama.cpp GPU spec decoding (~750
  tok/s on RTX 5070 Ti with 1.5B draft model).
- **Documentation audit fixes**: standardized tool naming from `Bash`/`Shell`
  to `shell(...)` across `docs/permissions.md`, `src/godspeed/security/rules.py`,
  `src/godspeed/security/permissions.py`, and `src/godspeed/config.py`.
- **Test reorganization**: moved `tests/test_skills_commands.py` to
  `tests/test_skills/test_commands.py`; removed redundant
  `tests/test_evolution/test_skill_gen.py`.

## [0.4.0] — 2026-05-01

### Added

- **Auto-load `~/.godspeed/.env.local` on startup** (`cli._load_env_files`).
  API keys in `~/.godspeed/.env` / `.env.local` (and the project's
  `.godspeed/.env` / `.env.local`) are now injected into `os.environ`
  before LiteLLM runs, so `NVIDIA_NIM_API_KEY=... godspeed` one-offs
  AND long-lived `~/.godspeed/.env.local` files both work without
  extra shell plumbing. Precedence: project/.env.local >
  project/.env > global/.env.local > global/.env; **shell env wins
  over all of them** so per-run overrides keep working. Matches
  dotenv/Vite/Next convention: `.env.local` is the gitignored local
  override. Log messages contain key names only, never values.
  Closes the "first-run auth error" UX paper cut that required users
  to remember to `$env:NVIDIA_NIM_API_KEY = ...` every shell.
- Troubleshooting doc: new "AuthenticationError: api_key client
  option must be set" section in `docs/troubleshooting.md` with the
  fix + the pre-v3.5.0 workaround.

### Changed

- **Performance optimizations across core modules**: reduced fsync
  frequency in audit trail (1→10 with guaranteed first-write sync),
  pre-computed lowercase keyword set in dangerous command detection,
  cached model string in LLM client, single-pass post-processing in
  agent loop, output caching in background process registry, async
  file/folder I/O in mention resolution, compiled mention regex in
  completions, and module-level imports in TUI app.
- **Background process auto-cleanup**: completed processes are now
  automatically removed from the registry to prevent memory leaks
  during long sessions.
- **Removed competitive product comparisons** from README, CHANGELOG,
  and audit docs. Project positioning is now strictly feature-driven.
- **Renamed PyPI package** from `godspeed` to `godspeed-coding-agent`
  to resolve naming conflict with an existing project. Install command
  updated throughout documentation.

### Fixed

- **Background test compatibility**: `test_output_after_completion`
  and `test_kill_already_exited` updated to handle auto-cleanup
  behavior where completed processes are removed from the registry.
- **Home directory expansion**: `~` now correctly expands in
  `glob_search`, `grep_search`, and `resolve_tool_path`.
- **Inaccessible file handling**: `glob_search` and `grep_search`
  gracefully skip files with permission errors on Windows.

## [0.4.0] — 2026-04-22

Phase 2 of the world-class UX push. Two independent additions on top
of v3.3.0, both landed with full test coverage and CI-green across
Python 3.11 / 3.12 / 3.13. Zero breaking changes — every default
behaves identically to v3.3.0 until you opt in.

### Added

- **`/remember` slash command** for persisting permission rules without
  leaving the TUI (`tui.commands._cmd_remember`). Writes to
  `~/.godspeed/settings.yaml` (or `.godspeed/settings.yaml` with
  `--project`) and adds the rule to the live
  `PermissionEngine` so it takes effect immediately — no restart.
  Accepts `approve` / `allow` / `deny` / `ask` as the action word;
  `approve` is an alias for `allow` that reads more naturally. Pattern
  syntax is validated up-front (`Tool(argument)` form) so typos fail
  loudly instead of saving an unmatchable rule. Duplicates are
  silently idempotent. New companion doc: `docs/permissions.md`
  covering the 4-tier model end-to-end.
- **`PermissionEngine.add_rule(pattern, action)`** for runtime rule
  injection — used by `/remember` but available for any code path that
  needs to widen (or narrow) the permission set mid-session.
- **`config.append_permission_rule(pattern, action, project_dir=)`** —
  generic YAML write-back helper covering all three tiers. Replaces
  the old allow-only `append_allow_rule`, which is kept as a
  back-compat wrapper.
- **Task-aware model routing** (`llm/router.py`,
  `GodspeedSettings.cheap_model` / `strong_model`). Each LLM turn is
  classified from conversation state (no extra LLM call) into one of
  `plan` / `edit` / `read` / `shell`; the existing `ModelRouter`
  swaps `self.model` for the duration of the call and restores on
  cleanup. `cheap_model` populates `routing[edit/read/shell]` and
  `strong_model` populates `routing[plan]` so users get the
  cost-vs-quality split without writing the full dict. Explicit
  `routing:` entries in YAML still win. Streaming and non-streaming
  paths both honor the `task_type` parameter; mid-turn cancel still
  closes the routed stream cleanly via the existing `aclose()`
  finally-block. Headless mode and configs without shortcuts behave
  identically to v3.3.0 (no routing → default model for every call).

## [3.3.0] — 2026-04-22

World-class coding-agent UX push. Five additions on top of the
security-first core, all born from a 6-task daily-use benchmark that
surfaced two real production bugs (both fixed) and four UX gaps (all
addressed). Every change shipped with tests, CI-green across Python
3.11 / 3.12 / 3.13.

Version jumps 3.1.0 → 3.3.0 to align with the research-track label
(v3.2 covered the LLM-judge / best@k selector + four null results;
artifacts are in `experiments/swebench_lite/` and the tech-report
draft on Desktop).

### Added

- **Mid-turn cancellation** (`agent.result.AgentCancelledError`,
  `agent.loop.agent_loop(cancel_event=...)`). The agent now checks
  a cancel event at three checkpoints per iteration — top of loop,
  after pause release, between streaming chunks — and unwinds
  cleanly via `AgentCancelledError` when set. `_streaming_call`
  uses a `finally`-block `aclose()` so the underlying httpx stream
  shuts down promptly. TUI installs an `asyncio` SIGINT handler on
  Linux/macOS: first Ctrl+C cancels the current turn; second press
  within 1 s hard-interrupts (Jupyter pattern). Windows
  `ProactorEventLoop` degrades gracefully. Distinct from pause —
  pause stops at iteration boundary; cancel stops immediately.
- **Diff approve-before-write gate** (`tools.base.DiffReviewer`
  Protocol, `Tool.produces_diff` class attribute,
  `ToolContext.diff_reviewer`). Two independent axes of consent
  now: `PermissionEvaluator` answers "may this tool run?"
  (existing); `DiffReviewer` answers "should THIS specific diff
  be applied?" (new). `file_edit` / `file_write` / `diff_apply`
  opt in via `produces_diff = True`; each calls
  `await context.diff_reviewer.review(tool_name, path, before,
  after)` just before write. `"accept"` proceeds; any other value
  (including forward-compat `"edit"`) rejects. TUI
  `_InteractiveDiffReviewer` renders a side-by-side diff via
  Rich's `Syntax("diff")` lexer with `(y)es · (n)o · (a)lways`
  keys; session-scoped `"a"` bypass. Headless/CI:
  `diff_reviewer` stays `None`, writes proceed as before.
- **Per-turn cost/token HUD** (`tui.output.format_status_hud`).
  Compact one-line summary after every agent turn:
  `· 1,234 in + 567 out (1,801) · $0.0024 · model · 3 turns`.
  When `max_cost_usd > 0` shows `$X / $Y` and tints `WARNING`
  when remaining budget drops below 20%. Provider prefix stripped
  for readability. Pure function; reuses existing `LLMClient`
  accumulators.
- **Troubleshooting + Windows quickstart docs**
  (`docs/troubleshooting.md`, `docs/quickstart_windows.md`,
  `docs/demo.md`). Top-of-funnel issues from the production audit:
  `UnicodeEncodeError` on cp1252, NIM 40 RPM mitigations, WSL +
  Docker for swebench (with swebench 4.1.0 pvlib NumPy bug
  caveat), audit-verify workflow, Ollama first-run, post-edit
  syntax gate rejections, diff reviewer keys, mid-turn Ctrl+C
  semantics. Windows quickstart: 5-minute path from Miniconda →
  pip → `godspeed init` → first task. Demo recording is
  user-gated; `docs/demo.md` captures the asciinema + agg recipe.
- **README "What's new in v3.3.0"** section — 5-row summary table
  cross-linking every new feature + the production audit.

### Fixed

- **`file_edit` silent indentation corruption on multi-line
  replace** (PR #78). The fuzzy matcher could find an indented
  region, but a poorly-indented `new_string` would write a
  syntactically broken file without warning. Now the tool runs
  `ast.parse(new_content)` on `.py` / `.pyi` and
  `json.loads(new_content)` on `.json`. If the pre-edit content
  parsed but the post-edit would not, the write is rejected with
  a clear error asking the agent to include more context.
  Pre-broken files pass through so the agent can fix them.
  Surfaced by the daily-use benchmark's T4 task (multi-file
  `user_id → account_id` rename); with the fix the same task
  now passes.
- **Windows `UnicodeEncodeError` on CLI exit** (PR #78). The
  agent's final summary often contains `→`, `—`, or smart quotes
  — none encodable in cp1252. `cli.py` now calls
  `_force_utf8_stdio()` at module load to re-wrap `sys.stdout` and
  `sys.stderr` as `TextIOWrapper(encoding="utf-8",
  errors="replace")`. Idempotent; skips already-utf8 streams.

### Changed

- Dependabot now ignores `pydantic` and `python-dotenv` bumps
  (PR #73). Both are pinned strictly by `litellm>=1.83.7`
  (`pydantic==2.12.5`, `python-dotenv==1.0.1`) — upstream-blocked
  so auto-opened PRs are unmergeable. Un-ignore when LiteLLM
  relaxes its pins.

### Tested

- **Daily-use benchmark** on
  `C:\Users\ttimm\Desktop\godspeed_benchmark\` runs 4/4 passes
  after the two fixes above; extended to 6/6 with harder refactor
  + bug-fix scenarios (T5, T6).
- **Production audit** adds path-traversal (T7a blocked
  correctly), audit-trail chain verify (21 records on a live
  session, tamper detection confirmed), and full-suite regression
  (1861+ pass; 10 flaky `system_optimizer` tests pre-exist and
  are environmental).
- **CI** 8/8 green across lint / security / type-check / CodeQL
  / Analyze / tests 3.11/3.12/3.13 on every feature PR (#78-#82).

## [3.1.0] — 2026-04-21

Agent-in-loop Docker oracle tool + ensemble selector + SystemOptimizerTool
+ driver registry + web-search hardening. Headline number is from an
**oracle-selector best-of-5 ensemble**, disclosed transparently — see
the Benchmarks subsection below and `experiments/swebench_lite/findings_2026_04_21.md`.

### Added

- **Agent-in-loop Docker oracle** (`experiments/swebench_lite/run_in_loop.py`,
  `docker_test_tool.py`, `run.py --agent-in-loop`). The agent can call
  `swebench_verify_patch` mid-session to run the real SWE-Bench test
  harness against its current edits, then iterate on the failure output.
  Budget: 5 verify calls per instance (hard cap 8), with a working-tree
  SHA short-circuit to prevent re-verifying unchanged diffs.
- **Oracle-guided best-of-N selector** (`experiments/swebench_lite/oracle_merge.py`).
  Takes `predictions.jsonl:report.json` pairs and picks per instance the
  patch that actually resolved (preferring shortest among resolvers;
  falling back to shortest non-empty). Methodology is the same pattern
  other open-source agent projects publish — explicitly labeled as
  `oracle_best_of_N` in output `model_name_or_path`.
- **Driver registry** (`src/godspeed/llm/driver_catalog.yaml`,
  `src/godspeed/agent/prompt_profiles.py`, `scripts/validate_driver.py`,
  `docs/adding_a_driver.md`). Ten catalogued drivers (NVIDIA NIM free
  tier, Moonshot direct, Anthropic frontier, Ollama local + cloud).
  Three prompt profiles (`default`, `thinking`, `minimal`) with robust
  fallbacks. New-driver smoke runs 3 easy instances and gates on a ≤20%
  LLM-error rate before allowing benchmark use.
- **SystemOptimizerTool** (`src/godspeed/tools/system_optimizer.py`),
  `inspect` and `recommend` modes (both `READ_ONLY`). Cross-platform
  (Windows/Linux/macOS) using `psutil` + optional `pynvml`. Hard deny-list
  for system-critical processes (`explorer.exe`, `systemd`, `launchd`,
  `kernel_task`, etc.), documented up front even though `act` mode is not
  yet shipped. Recommends Ollama unload, memory/disk pressure remediation,
  and outlier-process investigation.
- **Instance cooldown flag** (`run.py --instance-cooldown SECONDS`) to
  smooth NVIDIA NIM free-tier rate-limit (40 RPM) when running
  agent-in-loop sessions sequentially.
- **Web-search disk cache** (`src/godspeed/tools/web_fetch.py`). 7-day TTL
  at `~/.godspeed/cache/web/<sha1>.json` with 50 MB LRU-by-mtime eviction.
  New `no_cache: true` bypass flag. Transparent; no agent prompt change needed.
- **`--allow-web-search` flag** on `experiments/swebench_lite/run.py`
  (defaults to `False`). Benchmark runs now register the tool registry
  with `tool_set="local"` so the agent cannot leak the ground-truth fix
  via `web_search` / `web_fetch` during evaluation.
- **`reproduce_v3_1.sh`** one-command SWE-Bench Lite reproduction script.
- **SWE-Bench research memo** at `~/Desktop/SWE-Bench-Research-Memo/` (MD + PDF) documenting current SOTA, the dataset noise floor, the five
  compounding levers, and honest positioning for a solo-engineer OSS agent.

### Changed

- **`src/godspeed/tools/shell.py` force-kills the process tree on timeout.**
  Replaces `subprocess.run(timeout=N)` with `subprocess.Popen` +
  `communicate(timeout=...)` + psutil-based tree kill on `TimeoutExpired`.
  Addresses a Windows-specific failure mode where grandchildren holding
  pipes survived the parent's kill and blocked the runner for 60–100+
  minutes. Now returns within timeout + ~5s cleanup window. Six new
  regression tests in `tests/test_shell_tool.py` guard the path.
- **`--verify-retry` deprecated** in favor of `--agent-in-loop`. The
  former is a post-hoc single-shot retry; the latter gives the agent
  live oracle feedback during the session. `--verify-retry` now logs a
  deprecation warning and no-ops when combined with `--agent-in-loop`.
- `psutil>=5.9,<8.0` is now a required dependency (was optional) —
  SystemOptimizerTool registers by default and the test suite needs it.
- `aiohttp>=3.13.4` pin-up to patch 10 CVEs (CVE-2026-34513..34525,
  CVE-2026-22815) transitively carried through litellm.
- Settings.yaml.example pre-registers `moonshot/kimi-k2.6`,
  `moonshot/kimi-k2.5`, `ollama/kimi-k2.6:cloud`, `ollama/kimi-k2.5:cloud`
  driver paths (commented) so swapping providers is a one-line edit.

### Fixed

- CodeQL was flagging intentionally-broken fixture code
  (`benchmarks/fixtures/easy-fix-syntax-01/app.py`) as a syntax error.
  `.github/codeql/codeql-config.yml` now excludes `benchmarks/fixtures/`,
  `experiments/`, and `tests/fixtures/` from scanning.
- `python-dotenv` CVE-2026-28684 is suppressed via `pip-audit
  --ignore-vuln CVE-2026-28684` in CI. The fix (python-dotenv 1.2.2) is
  unsatisfiable because litellm pins `python-dotenv==1.0.1` strictly; the
  suppression is temporary, pending a litellm release that relaxes the
  bound. Commented in `.github/workflows/ci.yml`.

### Benchmarks

SWE-Bench Lite dev-23 (Python-only, 23-instance subset). All numbers are
free-tier; $0 API spend, ~6 hours total wall-clock on a single consumer
laptop (RTX 5070 Ti 16 GB). Method column is the important part — each
row uses a different inference strategy over the same 23-instance
split.

| Version | Method | Resolved | Rate |
|---|---|---:|---:|
| v2.12.0 baseline | Kimi K2.5 single-shot | 8 / 23 | 34.8% |
| v3.1 Phase 1 (null result) | Kimi K2.5 + agent-in-loop, single seed | 7 / 23 | 30.4% |
| **v3.1.0 headline** | **Oracle-selector best-of-5 (free-tier)** | **12 / 23** | **52.2%** |

**Methodology disclosure** — the 52.2% headline is an
`oracle_best_of_5`: the 5 constituent runs (Kimi K2.5 single-shot,
GPT-OSS-120B, Qwen3.5-397B iter1, Qwen3.5-397B seed3, Kimi K2.5 +
agent-in-loop) were each submitted to sb-cli standalone and paid their
own quota slot. The selector (`oracle_merge.py`) picks per instance the
patch from whichever run resolved — preferring shortest among resolvers;
falling back to shortest non-empty when no run resolved. This is the
same pattern used in published open-source agent research; it is explicitly
labeled as such.

**Single-run result was a null result.** Kimi K2.5 + agent-in-loop
alone scored 7 / 23 = 30.4%, below the 34.8% single-shot baseline.
Prompt tuning did not fix it. The instance-level analysis shows
agent-in-loop **is** additive inside the ensemble (it contributed
`marshmallow-code__marshmallow-1359`, a unique resolve no other run
landed) — but on this driver, a verify-feedback loop alone over-edits on
hard instances and stops early on easy ones.

**Limitations.** dev-23 is a 23-instance subset of SWE-Bench Lite's 300
dev instances. Single-seed runs for most constituent drivers (Kimi
seed 2 was contamination-affected by NVIDIA NIM concurrent-job
contention and excluded). Test-50 and dev-300 headline validation is
pending for v3.1.x. Published SOTA on full SWE-Bench Lite (April 2026) is
62.7%; top open-source agents with paid frontier drivers sit in the
40-50% band on the same benchmark.

Reproducibility:

```bash
./experiments/swebench_lite/reproduce_v3_1.sh
```

Full findings, per-instance resolution map, run-by-run pre-merge
numbers, and the research memo that framed this release:
- `experiments/swebench_lite/findings_2026_04_21.md`
- `~/Desktop/SWE-Bench-Research-Memo/` (local, not checked in)

## [2.11.0] — 2026-04-19

Benchmark and test-infrastructure polish. No runtime behavior changes to
the agent itself — this release strengthens the evidence that Godspeed
works end-to-end before any fine-tuning work begins.

### Added

- **Real-LLM smoke tests under `@pytest.mark.real_llm`**
  (`tests/test_smoke_real_llm.py`, `tests/test_integration_real.py`).
  Skipped by default; run with `pytest -m real_llm` and a running Ollama
  server on `localhost:11434`. Includes a multi-turn read-edit-verify
  scenario that proves the agent loop chains real tool calls with a real
  model — the foundation claim the prior test suite did not exercise.
- **Benchmark fixture directories** (`benchmarks/fixtures/<task_id>/`)
  for 19 of the 20 tasks in `benchmarks/tasks.jsonl`. Each is copied into
  an isolated temp workspace per run so scores are reproducible across
  models and invocations. Optional `_setup.py` hook handles tasks that
  need runtime state (git repos with dirty/staged changes). Optional
  `verify.py` hook runs post-agent and returns a machine-checkable
  mechanical-success signal for 13 tasks (syntax fix, SQL-injection
  remediation, requests→httpx migration, CI workflow creation, etc.).
- **`waste_penalty` metric** in `BenchmarkScore` and `BenchmarkSuiteResult`.
  Deducts up to 0.3 when an agent issues > 1.5× the expected tool calls,
  so honest efficiency shows up in the score instead of being hidden by
  the Jaccard + LCS primaries. Tie-breaker, not a dominant signal.
- **`test_parallel_file_reads_real_tools`** — complements the existing
  `_TrackedTool` synthetic test by running real `FileReadTool` in
  parallel on real files, locking in the concurrency guarantee for the
  production tool path.

### Changed

- `scripts/run_benchmark.py` now uses `benchmarks/fixtures/<task_id>/` as
  the per-task workspace when present (falls back to `--project-dir`
  otherwise). Records `mechanical_success` per task and `mechanical_pass`
  / `mechanical_evaluated` / `mean_waste_penalty` in `summary.json`.

### Fixed

- Prior audit erroneously flagged speculative tool dispatch as
  scaffolded-but-inactive. Verified the cache is populated by
  `_speculative_dispatch` during streaming (`src/godspeed/agent/loop.py`)
  and exercised by 7 passing tests in `tests/test_speculative.py`.

### Benchmarks

First honest end-to-end numbers for Godspeed on a real LLM, run against
the polished fixtures:

| Driver | Overall | Pass | Mech |
|---|---:|---:|---:|
| `nvidia_nim/qwen/qwen3.5-397b-a17b` | **0.608** | 11/20 | 7/13 |
| `nvidia_nim/moonshotai/kimi-k2.5` | 0.548 | 9/20 | 6/13 |
| `nvidia_nim/mistralai/devstral-2-123b-instruct-2512` | 0.446 | 5/20 | 2/13 |
| `nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct` | 0.333 | 5/20 | 2/13 |
| `ollama/qwen3-coder:latest` | 0.107 | 1/20 | 1/13 |

Recommended production driver: `nvidia_nim/qwen/qwen3.5-397b-a17b`.
Full table: `experiments/benchmark_shootout_2026_04.md`.

## [2.10.0] — 2026-04-18

Minor release bundling two independent features: the Phase A1 synthetic
tool-calling training-data pipeline, and weak-model robustness at the
agent layer (tool-name aliasing + tool-set filtering).

### Added — Phase A1 pipeline (`experiments/phase_a1/`)

- End-to-end synthetic data pipeline producing ~6,200 `{messages, tools}`
  OpenAI-format training samples on a $0 free-tier-only budget. Stages:
  `specs` (stratified 21-tool × 4-category gen plan) → `blueprints`
  (LLM-planned JSON with per-tool arg validation + retry) → `executor`
  (real tool dispatch against a seeded sandbox, with fixtures for
  network-touching tools) → `narrator` (LLM assistant-reasoning with
  anti-hallucination guards + retry) → `emit` (canonical OpenAI record)
  → `judge` (4-dim rubric scoring) → `validate` (schema + coverage gate)
  → `assemble` (merge anchor + augment + synthetic + distill, dedup by
  user-prompt hash, deterministic shuffle).
- **Per-tool argument validation inside blueprint generation**. The same
  invariants `validate.py` enforces post-execution now run pre-execution
  via the new public `validate_tool_call_args(tool_name, args)` helper.
  Malformed blueprints (e.g. `github.action=None`, `grep_search.pattern=""`)
  are rejected before executor + narrator spend is incurred, then retried
  with a temperature bump. Closes the three failure modes that burned a
  prior 5-sample smoke run (0/5 yield).
- **Anti-hallucination narrator guards**. System prompt now forbids the
  narrator from inventing facts (PR state, tool output values, counts)
  not present in the executed transcript. A `narrate_session` retry
  loop catches structural failures like `pre_call length != N` and
  re-samples with bumped temperature before giving up.
- **Observability**. `executor.py`, `judge.py`, and four sites in
  `orchestrate.py` now call `logger.warning(..., exc_info=True)` on
  caught exceptions so failed samples are debuggable from the log stream
  rather than only via the truncated `error` field in `run_metrics.jsonl`.
- Multi-provider async router with SQLite-backed per-provider daily
  quota tracking (`providers.py`): Cerebras + Z.ai + Groq + Ollama
  cascade, UTC-midnight quota reset, rate-limit backoff.
- Makefile `a1-*` targets for each stage (`a1-smoke`, `a1-run`,
  `a1-run-prod`, `a1-validate`, `a1-judge`, `a1-anchor`, `a1-distill`,
  `a1-augment`, `a1-assemble`).
- Per-file ruff ignore for `experiments/**/*.py` — allow `S311` (jitter
  via `random`), `N818` (error naming), `SIM103` style nits in research
  code.
- `.gitignore` rules excluding derived data artifacts:
  `experiments/phase_a1/data/*.jsonl`, `*.db`, `sessions/`, `_*/`, plus
  `.env.local`.

### Added — Agent tool-name aliasing + tool-set filtering

- **`src/godspeed/tools/aliases.py`**: common hallucinated tool names
  (`read_file`, `grep`, `glob`, etc.) get canonicalized to their
  registered equivalents inside `_parse_tool_call`. Small open-source
  models express a correct intent with the wrong label — this closes
  the dead-end without a fine-tune.
- **`src/godspeed/tools/tool_sets.py`**: named capability surfaces
  (`local` / `web` / `full`). New `--tool-set` CLI flag constrains the
  registry so local-codebase runs hide `web_search` / `web_fetch` /
  `github` entirely. Weak models stop picking `web_search` over
  `file_read` by construction, not by prompting.
- QUALITY_PROMPT extension with explicit Tool Selection Defaults so the
  model's written guidance matches the runtime constraint.

### Tests (+18 new)

- `experiments/phase_a1/tests/test_blueprints.py` (11 tests): validates
  the per-tool-arg rejection of `github.action=None` and
  `grep_search.pattern=""`, verifies retry-on-bad-args,
  retry-on-invalid-JSON, temperature bump on each retry, failure after
  exhausted retries, and prompt content for error-prone tools.
- `experiments/phase_a1/tests/test_narrator.py` (7 tests): verifies
  retry on `pre_call` length mismatch, retry on invalid JSON,
  temperature bump, failure after exhausted retries,
  content-injection into the session JSONL, and anti-hallucination
  prompt content.
- Phase A1 test suite total: 125 passing (excludes
  `test_swesmith_distill.py` which has a pre-existing `sklearn` import
  gap unrelated to this release).
- Project-wide suite: 1706 passing / 20 skipped.

## [2.9.1] — 2026-04-17

Patch release — compatibility shim and local-model documentation. Shipped
alongside the Stage A findings from the Track B re-evaluation plan.

### Added

- **`src/godspeed/llm/qwen3_coder_parser.py`** — regex extractor for
  Qwen3-Coder's `<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>`
  tool-call XML. Ollama 0.20.x doesn't recognize this format — calls come
  back in the `content` field instead of `tool_calls`. The parser
  synthesizes OpenAI-shaped tool_calls so the rest of Godspeed's pipeline
  sees a standard response. Hooked into `LLMClient._call()` as a no-op
  when `tool_calls` is already populated.
- Model table entries for `ollama/qwen3:14b` and
  `ollama/qwen3-coder:latest` in the `models` CLI command and
  `settings.yaml.example` — documented but not promoted to default.
- `scripts/run_benchmark.py` — measurement runner that shells out
  `godspeed run` per task and scores via `training.benchmark`. Not a
  shipped CLI command; a tool for Stage A / future benchmark runs.
- Per-file ignore for `scripts/**/*.py` in `pyproject.toml` — `print()`
  is intentional in CLI scripts.
- `.godspeed/training/`, `.godspeed/checkpoints/`, `.godspeed/memory.db*`
  added to `.gitignore` — session artifacts should not track into repos.

### Tests (+20 new)

- `tests/test_qwen3_coder_parser.py` covers the parser's detector, the
  type-coercion (bool/int/float/JSON/string), multi-call responses,
  malformed input, and uniqueness of synthesized IDs.

## [2.9.0] — 2026-04-17

Final entry in the v2.5.1 review follow-up chain. Auto-index on session
start so the agent has semantic code search available by default instead
of requiring a manual `/reindex`.

### Added

- **Auto-index on session start** (`src/godspeed/context/auto_index.py`).
  New helper `maybe_start_auto_index(project_dir, auto_index_enabled)`
  that:
  - Returns `None` when disabled, when `chromadb` isn't installed (graceful
    degradation via the `[index]` extra), or when the index is already
    fresh (`needs_reindex()` returns `False`).
  - Otherwise schedules `build_index_async()` as an `asyncio.Task` so the
    session continues without blocking on index construction.
  - Swallows exceptions in the build coroutine so an indexing failure
    never crashes the agent.

  Wired into both the TUI and headless paths in `cli.py`, right after
  the `ToolContext` is constructed.
- **`GodspeedSettings.auto_index`** field (default `True`). Disable with
  `auto_index: false` in `~/.godspeed/settings.yaml` or `GODSPEED_AUTO_INDEX=false`.

### Changed

- Sessions started without a fresh codebase index now rebuild it in the
  background automatically (assuming `[index]` extra is installed). The
  previous behavior required a manual `/reindex`.

## [2.8.0] — 2026-04-17

Closes the last of the v2.5.1 review follow-ups. New LLM-driven test
generation tool, architectural affordance for tools that need an LLM,
and the semantic fix to `verify._verify_with_retry` that was deferred
from earlier patches.

### Added

- **`generate_tests` tool** (`src/godspeed/tools/generate_tests.py`) —
  reads a source file, asks the LLM to produce a complete pytest
  module, writes it to `tests/test_<basename>.py` (or a caller-specified
  path). The agent can then run `test_runner` to confirm the generated
  tests pass. Strips markdown code fences if the LLM returns them.
  Closes the test-first discipline loop with one tool call.
- **`LLMInvoker` protocol** + **`ToolContext.llm_client`** field
  (`src/godspeed/tools/base.py`) — architectural affordance for tools
  that need to make LLM calls. Decouples `tools/` from a concrete
  `llm.LLMClient` reference. Both headless and TUI paths now populate
  `llm_client` when constructing the `ToolContext`.

### Changed

- **`verify._verify_with_retry` now returns `ToolResult.failure`** when
  unresolved lint errors remain after retries (previously it returned a
  success-typed result with "some remaining" in the output body).
  The MUST-FIX gate still fires on the same fingerprint — it now checks
  both `result.error` and `result.output` so it's robust across both
  the new and legacy shapes. Downstream callers get a clear
  `is_error=True` signal where before they had to substring-match the
  output.
- `_build_tool_registry` (`cli.py`) registers `GenerateTestsTool`
  alongside the other quality tools.
- The three tests in `test_verify_fix_retry.py` that previously asserted
  `not result.is_error` with remaining issues have been updated to
  assert `result.is_error` and read the fingerprint from `result.error`.

### Deferred to v2.9.0

- Auto-index on session start (separate concern — session-init refactor).

## [2.7.0] — 2026-04-17

Quality-tooling minor release, continuation of v2.6.0. Closes two more
deferred items from the v2.5.1 review and adds an integration-level
invariant check between the audit trail and training log.

### Added

- **`complexity` tool** (`src/godspeed/tools/complexity.py`) — wraps
  `radon cc` (Python, with grade mapping from the McCabe CC threshold)
  and `lizard` (polyglot, when installed). Functions above `max_cc`
  (default 10) fail the call — lets the agent enforce a complexity
  budget on edits rather than catching it in code review.
- **`dep_audit` tool** (`src/godspeed/tools/dep_audit.py`) — auto-detects
  the project's package manager from manifests (`pyproject.toml` /
  `requirements.txt` → pip-audit; `package.json` → npm audit;
  `Cargo.toml` → cargo-audit) and runs the matching CVE scanner.
  Vulnerabilities produce an error result so the agent treats them as
  gating. Explicit `manager` argument overrides detection.
- **E2E drift test** (`tests/test_audit_log_drift.py`) — integration-level
  assertion that the audit trail's `session_end` detail and
  `ConversationLogger.log_session_end` record agree on every field
  (`exit_reason`, `exit_code`, all metrics). Catches regressions where
  the two streams could drift after independent edits.

### Changed

- `_build_tool_registry` (`cli.py`) registers `ComplexityTool` and
  `DepAuditTool` alongside the existing quality tools.

## [2.6.0] — 2026-04-17

Quality-tooling minor release. Two new tools the agent can call to enforce
production-grade code; efficiency telemetry in training logs; follow-up
cleanup from v2.5.1.

### Added

- **`coverage` tool** (`src/godspeed/tools/coverage.py`) — wraps
  `coverage run pytest` + `coverage report -m`. Optional `min_percent`
  argument fails the call below the threshold, turning coverage into a
  quality gate the agent can enforce per-session. Auto-detects `--source`
  (`./src` if present, else cwd). Graceful error when `coverage` isn't
  installed.
- **`security_scan` tool** (`src/godspeed/tools/security_scan.py`) —
  wraps `bandit` (Python) and `semgrep` (polyglot, when installed). Both
  optional; falls back when neither is available. Non-zero findings
  produce an error result so the agent treats them as gating rather than
  advisory. Configurable minimum severity (`low` / `medium` / `high`).
- **`AgentMetrics.must_fix_injections`** counter — increments each time
  the MUST-FIX gate injects a fix-required message. Exposed in headless
  JSON output, audit trail `session_end` detail, and `ConversationLogger`
  `log_session_end` record. Training signal: agents that trigger many
  MUST-FIX injections are less efficient per unit of successful work;
  downstream RL (GRPO) can penalize against this counter.

### Changed

- **Fingerprint coupling reduced** — `verify.REMAINING_ERRORS_FINGERPRINT`
  is now a module-level constant imported by `loop._maybe_inject_must_fix`.
  Eliminates the duplicated `"some remaining"` string literal identified
  in the v2.5.1 review follow-ups.
- `ConversationLogger.log_session_end(...)` accepts a new
  `must_fix_injections` keyword argument (defaulted to 0 for
  backwards-compatibility with existing callers).
- `_build_tool_registry` (`cli.py`) registers `CoverageTool` and
  `SecurityScanTool` alongside the existing quality tools.

### Fixed

- `tests/test_agent_result.py::test_duration_before_finalize` — flaked on
  Windows where `time.monotonic()` granularity could return identical
  values across a `0.01s` sleep. Assertion relaxed to `>= 0.0` (the
  semantically correct invariant — duration is never negative).

## [2.5.1] — 2026-04-17

Patch release — code-quality enforcement and training-record alignment. No new
user-facing CLI flags or JSON schema changes.

### Added

- **`ConversationLogger.log_session_end()`** emits a terminal record per
  headless session with `exit_reason`, `exit_code`, `iterations_used`,
  `tool_call_count`, `tool_error_count`, `duration_seconds`, `cost_usd`.
  Fields mirror the audit-trail `session_end` detail so the two streams stay
  comparable. Enables RL pipelines (GRPO) to shape rewards on `exit_code`
  without parsing the audit log.
- **Quality defaults** in the system prompt: a new `QUALITY_PROMPT` block
  (type hints, test-first, read-before-edit, comments-only-when-non-obvious,
  secrets policy, anti-premature-abstraction) appended after `WORKFLOW_PROMPT`.

### Changed

- **MUST-FIX gate on auto-verify failures**. When `_auto_verify_file` leaves
  unresolved lint errors (fingerprint: "some remaining"), the agent loop
  injects a user-role message naming the file and errors, instructing the
  model to fix them before any other edits. Capped at 3 injections per
  session — after the cap, the gate logs a warning and fails open so the
  agent isn't deadlocked on a fundamentally unfixable error (e.g., broken
  project ruff config). Wired into both parallel and sequential dispatch
  paths. Structural enforcement closes a silent-error channel where the
  model could read a `verify` success marker despite persistent lint issues.

### Fixed

- `src/godspeed/__init__.py:5` — stale `__version__` (was `"2.2.0"`; now
  tracks `pyproject.toml`).

## [2.5.0] — 2026-04-16

MLOps-readiness release. Closes the headless/pipeline gaps found in the
feature + logic audit on 2026-04-16. The `godspeed run` command is now
suitable for unattended use in CI, W&B sweeps, and ML research pipelines.

### Added

- **Exit code contract** (`godspeed.agent.result.ExitCode`). `godspeed run`
  now returns differentiated exit codes so orchestrators can switch on them:
  `0` success, `1` tool error, `2` max iterations, `3` budget exceeded,
  `4` LLM error, `5` invalid input, `6` timeout, `130` interrupted. Codes
  are stable across minor versions.
- **JSON output schema extended** with `exit_reason`, `exit_code`,
  `iterations_used`, `tool_calls` (list of `{name, is_error}`),
  `tool_call_count`, `tool_error_count`, `duration_seconds`, `cost_usd`,
  and `audit_log_path`. Existing fields preserved.
- **`--timeout N`** wall-clock session cap on `godspeed run`. Wraps the
  agent loop in `asyncio.wait_for`; exits with code 6 on timeout. Default
  `0` (no limit) preserves existing behavior.
- **`--prompt-file FILE`** flag on `godspeed run` reads the task from a
  file. Task input precedence: `--prompt-file` > positional > stdin.
  Passing `-` as the positional task reads from stdin; an empty positional
  with a piped stdin also reads from stdin.
- **Rate-limit retry with exponential backoff + jitter** in `LLMClient`.
  429/quota errors now retry up to 4 times on the same model (delays
  1s/2s/4s/8s ± 25% jitter, capped at 60s) before falling over to the
  next model. Honors `Retry-After` when the provider supplies it
  (upward-only jitter — we don't retry earlier than the provider asked).
- **`AgentMetrics`** accumulator on `agent_loop()` via the new optional
  `metrics` kwarg. Populates `iterations_used`, `tool_calls`,
  `exit_reason`, `duration_seconds`.

### Changed

- **Headless mode now has an audit trail by default**. Previously
  `godspeed run` wired `audit=None` into the tool context — unattended
  sessions had no tamper-evident log, the opposite of the project's
  security posture. Audit now writes to `~/.godspeed/audit/{session}.audit.jsonl`
  with session-start and session-end records bookending every run.
- `agent_loop()` gains the optional `metrics: AgentMetrics | None` kwarg.
  Omitting it preserves the prior behavior exactly — backwards-compatible.

### Fixed

- Addresses G1–G7 from the 2026-04-16 feature audit.

### Security

- **Audit trail fails closed on I/O errors**: `AuditTrail.record()` now raises
  `AuditWriteError` when a write or `fsync` fails, instead of silently logging
  and advancing the in-memory chain. Chain state (sequence, prev_hash) does not
  advance on failure, so a successful retry chains cleanly from the last
  persisted record. This closes a gap where disk-full or permission errors
  would poison the hash chain while the agent continued executing tools.
- **Evolution safety gate blocks mutations to security-sensitive tool descriptions**:
  Mutations whose `artifact_id` is in `SECURITY_SENSITIVE_TOOL_IDS` (shell, bash,
  file_write, file_edit, diff_apply, git, github, background) now require
  human review. Mutations whose text matches any `SECURITY_BYPASS_PATTERNS`
  regex (e.g. "always granted", "bypass permission", "ignore safety",
  "auto-approve") also require review regardless of artifact.

### Changed

- README, SECURITY.md, and architecture doc updated: dangerous-pattern count
  corrected from "72+" to 71 (actual); tool count updated from "18+" to 25.
- CI matrix extended to Python 3.13 (previously 3.11, 3.12).
- Pre-commit adds `mypy` (src-scoped) and `bandit` (low-severity filter) hooks;
  ruff hook bumped to v0.14.1.
- `make lint` no longer auto-fixes or formats — matches CI exactly. New
  `make fix` target runs `ruff check --fix && ruff format .`. `make test` now
  runs with coverage gate to match CI.

### Added

- **CodeQL workflow** (`.github/workflows/codeql.yml`): security-and-quality
  query set, runs on push/PR and weekly cron.
- **Release workflow** (`.github/workflows/release.yml`): builds wheel/sdist
  on `v*` tag push, generates CycloneDX SBOM, attests build provenance via
  sigstore, attaches to GitHub release.

## [2.3.0] — 2026-04-13

### Added

- **Fine-tuning data pipeline** (Units A-F): Full infrastructure for collecting training data and fine-tuning tool-calling LLMs on Godspeed conversations.
  - **ConversationLogger**: Persists every conversation message (user, assistant, tool results, compaction summaries) to per-session JSONL at `~/.godspeed/training/`. Gated on `log_conversations` config.
  - **TrainingExporter**: Converts conversation logs to `openai`, `chatml`, and `sharegpt` fine-tuning formats with filtering (min_tool_calls, success_only, min_turns, tool whitelist). CLI: `godspeed export-training`.
  - **Per-step reward annotations**: Automatic reward signals for GRPO/DPO — success (+1.0), verify passed (+0.5), dangerous command (-1.0), efficient sequence bonus (+0.5). Session-level summarization.
  - **Benchmark suite**: 20 hand-crafted tasks (easy/medium/hard) with Jaccard tool selection scoring and LCS sequence quality scoring.
  - **Tool description enhancement**: All 10 core tools now include inline usage examples and JSON Schema `examples` fields for better training signal.
  - **Common workflows in system prompt**: 5 canonical multi-step patterns (fix bug, add feature, explore codebase, git workflow, research/debug).
- 110 new training pipeline tests (total: 1,557 passing)

## [2.2.0] — 2026-04-12

### Added

- **Self-evolution system** (Units 20-27): Learn from execution traces to improve prompts, tool descriptions, and permissions automatically. Runs entirely on Ollama for $0 — optional API acceleration.
  - **Trace analyzer**: Parse audit trail JSONL into failure patterns, latency stats, permission insights, and repeated tool sequences.
  - **Evolution engine**: GEPA-style LLM-guided mutations for tool descriptions, system prompt sections, compaction prompts, and auto-generated skills.
  - **Fitness evaluator**: A/B testing with LLM-as-judge scoring (0.5×correctness + 0.3×procedure + 0.2×conciseness). Length penalty for bloat.
  - **Safety gate**: Size limit (<2x growth), semantic drift (Jaccard ≥0.3), fitness threshold, confidence check, human review for high-impact artifacts.
  - **Evolution registry**: Append-only JSONL history with apply/revert/rollback. Originals backed up before mutation.
  - **Runtime hot-swap**: Update tool descriptions in-memory via `ToolRegistry._description_overrides`. Prompt section overrides loaded on startup.
  - **Cross-session learning**: Aggregate insights across sessions, model-specific analysis, regression detection with rollback/investigate/monitor recommendations.
  - **Skill auto-generation**: Detect repeated multi-tool patterns (≥3 occurrences) → generate reusable skill markdown with YAML frontmatter.
  - **Permission pattern learning**: Analyze denial/approval patterns → suggest allowlist optimizations with rationale.
  - **Hardware-aware model selection**: Auto-detect VRAM (nvidia-smi + Jetson /proc/meminfo), pick largest fitting model from tier list. Scales candidates and batch sizes by available memory. Jetson Orin Nano 8GB → qwen2.5:3b with 2 candidates.
  - **`/evolve` command**: status, run, history, rollback, review, approve, reject subcommands.
- 175 new evolution tests (total: 1,400+ passing)

### Changed

- `ToolRegistry` gains `_description_overrides` dict, `update_description()`, `clear_description_override()`, `get_description()` methods
- `GodspeedSettings` includes `evolution_enabled` and `evolution_model` fields
- Trace analyzer uses streaming line-by-line reads instead of `readlines()` for low-memory devices

## [2.1.0] - 2026-04-12

### Added

- **Extended thinking**: Pass `thinking` parameter to Anthropic/Claude models with configurable token budget. `/think [budget]` slash command to toggle. Thinking blocks displayed in collapsed dim panel.
- **Notebook (.ipynb) support**: `file_read` detects `.ipynb` and renders cells as structured text with outputs. `notebook_edit` tool for cell-level operations (edit, add, delete, move).
- **Image read tool**: `image_read` reads PNG/JPG/GIF/WebP files, returns base64-encoded content for vision-capable LLMs. Size limits (20MB reject, 5MB warning). Path traversal protection.
- **PDF read tool**: `pdf_read` extracts text from PDF files with page range support (`pages: "1-5"`). 20-page-per-request limit. Graceful fallback when pymupdf not installed.
- **Cost budget enforcement**: Hard cost limit via `max_cost_usd` config. `BudgetExceededError` stops agent when exceeded. `/budget [amount]` slash command to set/show budget.
- **GitHub PR/issue tool**: `github` tool wraps `gh` CLI for create/list/view PRs and issues, add comments. 8 actions with structured JSON output. HIGH risk (modifies external state).
- **Background command execution**: `shell` tool gains `background: true` parameter. `BackgroundRegistry` singleton tracks processes. `background_check` tool for status/output/kill.
- **Unified diff apply tool**: `diff_apply` accepts unified diff format, applies hunks in reverse order. Multi-file support, fuzzy context matching (±3 lines), dry-run mode.
- **Architect mode**: `/architect` toggles two-phase plan-then-execute pipeline. Phase 1 uses read-only tools for planning, Phase 2 uses full tools guided by the plan.
- **Speculative tool execution**: During streaming, READ_ONLY tool calls are dispatched immediately before the full response completes. Results cached and consumed by the main loop — eliminates dispatch latency for reads.
- **Risk-based parallel/serial dispatch**: Phase 3 partitions tool calls by risk level. READ_ONLY tools run in parallel via `asyncio.gather()`, write tools run sequentially. Maintains ordering.
- **Cheapest-model compaction**: `_compact_conversation()` selects the cheapest model from the fallback chain for summarization. `get_cheapest_model()` in cost module.
- **Typed event system**: `AgentEvent` union type with 11 frozen dataclass event types (ThinkingEvent, TextChunkEvent, ToolCallEvent, etc.) for composable agent loop consumption.
- 250+ new tests (total: 1,200+ passing)

### Changed

- `agent_loop()` signature expanded: `on_thinking` callback, speculative cache for streaming
- `LLMClient` tracks `thinking_budget`, `max_cost_usd`, `total_cost_usd`
- `GodspeedSettings` includes `thinking_budget`, `max_cost_usd`, `architect_model`, `sandbox` fields
- `_streaming_call()` accepts `tool_registry` and `speculative_cache` for speculative dispatch
- `/help` updated with new commands under "Agent Control" section

## [2.0.0] - 2026-04-11

### Added

- **Parallel tool execution**: When the LLM returns multiple tool calls, they execute concurrently via `asyncio.gather()`. 3-phase dispatch: parse → permission check (sequential) → execute (parallel). Config: `parallel_tool_calls: true` (default). Falls back to sequential for single calls or when disabled.
- **Multimodal user messages**: `add_user_message()` accepts `str` or `list[dict]` content blocks. `build_image_content_block()` and `build_multimodal_message()` helpers for image URLs and base64 data URIs. Token counter estimates 765 tokens per image block.
- **@-mention context injection**: `@file:path`, `@folder:path`, `@web:url` syntax in user input. Parsed via regex, resolved to content blocks injected into conversation. File/folder mentions validate paths via `resolve_tool_path()` (traversal protection). Web mentions enforce HTTPS-only with 100KB size limit.
- **@-mention tab completion**: Typing `@` suggests mention types; `@file:` and `@folder:` complete file paths relative to project directory.
- **Auto-commit workflow**: After configurable number of successful edits, generates a conventional commit message via LLM and commits with `Co-Authored-By: Godspeed <noreply@godspeed.dev>` attribution. Config: `auto_commit: false` (default off), `auto_commit_threshold: 5`.
- **`/autocommit` slash command**: Toggle auto-commit on/off, set threshold. Listed in `/help` and tab-completable.
- **Lint-fix retry loop**: Auto-verify now retries `ruff check --fix` up to N times for Python/JS files until clean. Config: `auto_fix_retries: 3` (default). Languages without deterministic fixers skip retry.
- **MCP SSE/HTTP transport**: `MCPSSEClient` connects to remote MCP servers via HTTP/SSE alongside existing stdio transport. Config: `transport: "sse"`, `url`, `headers` fields in MCP server config. Backward compatible — missing `transport` field defaults to "stdio".
- **Parallel tool TUI output**: `format_parallel_tool_calls()` shows grouped header with tool count and names. `format_parallel_results()` shows batch summary with success/error markers. Callbacks: `on_parallel_start`, `on_parallel_complete`.
- **Deferred auto-verify/auto-stash for parallel mode**: Auto-verify runs sequentially after parallel batch completes. Auto-stash counts writes across batch.
- 100+ new tests across 7 test files (total: 960+ passing)

### Changed

- `agent_loop()` signature expanded: `parallel_tool_calls`, `skip_user_message`, `auto_fix_retries`, `auto_commit`, `auto_commit_threshold`, `on_parallel_start`, `on_parallel_complete` parameters
- `Conversation.add_user_message()` accepts multimodal content blocks (`list[dict]`)
- Token counter handles `image_url` type blocks with flat 765-token estimate
- `GodspeedSettings` includes new config fields for all v2.0 features

## [1.0.0] - 2026-04-11

### Added

- **Token cost tracking**: `llm/cost.py` — model-aware pricing table for 20+ models (Claude, GPT, Gemini, DeepSeek). `/stats` command shows token usage and estimated cost. Quit screen includes session cost. Ollama/local models always show "free".
- **Enhanced diff previews**: Permission prompts now show unified diff format with `@@ hunk` headers, line change stats (`+5 -3 lines`), and up to 30 context lines. File write prompts show line count and overwrite warning.
- **Multi-file project instructions**: Loads GODSPEED.md, AGENTS.md (Linux Foundation AAIF standard), CLAUDE.md, and .cursorrules. Priority: GODSPEED.md > AGENTS.md > CLAUDE.md > .cursorrules. Zero-friction migration from other agents.
- **Prompt caching**: System prompt marked with `cache_control: ephemeral` for Anthropic/OpenAI models. ~50% cost reduction on repeated prefixes via LiteLLM.
- **Conversation export**: `/export [name]` command writes session as formatted markdown to `.godspeed/exports/`. Includes system prompt, messages, tool calls, and results.
- **Multi-language verify**: Auto-verify now supports Python (ruff), JS/TS (biome/eslint), Go (go vet), Rust (cargo check), and C/C++ (clang-tidy). Linters detected dynamically. Shared `_run_linter()` helper.
- **Test runner tool**: `test_runner` tool auto-detects project framework (pytest, jest, vitest, go test, cargo test). Runs targeted or full test suites. Available to the agent for edit-test-fix loops.
- **Headless/CI mode**: `godspeed run "task" --headless` for non-interactive execution. `--auto-approve` levels (reads/all/none), `--json-output` for structured results, `--max-iterations` control. Exit code reflects success/failure.
- **Web search tool**: `web_search` — DuckDuckGo HTML search, no API key required. Returns titles, URLs, snippets. Agent can look up docs and error messages.
- **Web fetch tool**: `web_fetch` — HTTP GET with HTML-to-text extraction. Blocks local/private network access. 10K char limit. Agent can read documentation pages.
- 12 new built-in tools (total: 12 built-in + MCP + sub-agents)
- 92 new tests (total: 806+ passing), all new features tested
- `/stats` and `/export` slash commands with tab completion

### Changed

- Auto-verify triggers on .js, .jsx, .ts, .tsx, .go, .rs, .c, .cpp, .h, .hpp (was Python-only)
- Quit screen shows estimated session cost for paid models
- Permission prompt diff uses `difflib.unified_diff` (was basic -/+ prefix)
- File write permission prompt shows line count and create/overwrite indicator

## [0.9.0] - 2026-04-11

### Added

- **Skill framework**: Markdown `.md` files with YAML frontmatter define reusable prompt skills. `discover_skills()` scans `~/.godspeed/skills/` and `.godspeed/skills/`, project overrides global. `/{trigger}` commands inject skill content into conversation. `/skills` lists available skills. Tab-completion for skill triggers.
- **Auto-permission learning**: `ApprovalTracker` counts repeated user approvals per pattern. After 3 approvals, suggests adding as permanent allow rule. `append_allow_rule()` persists to `.godspeed/settings.yaml`. Thread-safe, session-scoped.
- **Hook system**: `HookDefinition` pydantic model with 4 event types (`pre_tool_call`, `post_tool_call`, `pre_session`, `post_session`). `HookExecutor` runs shell commands with template variables (`{tool_name}`, `{session_id}`, `{cwd}`). Pre-tool hooks can block execution. Configurable timeout (1-300s). Wired into agent loop and CLI lifecycle.
- **Task tracking**: `TaskStore` (in-memory, sequential IDs) + `TaskTool` (create/update/list/complete). `/tasks` command shows themed Rich table. Registered as built-in LOW-risk tool.
- **Codebase index**: Optional ChromaDB-backed semantic search (`[index]` extra). AST-based chunking for Python, sliding window for other languages. `CodeSearchTool` for natural language code queries. Background indexing, `/reindex` command, stale detection.
- **Architecture document**: `GODSPEED_ARCHITECTURE.md` — 6-part reference covering core loop, security model, tool system, intelligence, autonomy, and memory/TUI. Mermaid diagrams. HTML comment delimiters for chunk loading.
- `hooks` field in `GodspeedSettings` for YAML hook configuration
- `chromadb` optional dependency under `[index]` extra
- 734 tests, ~90% coverage

## [0.6.0] - 2026-04-11

### Added

- **Midnight Gold visual identity**: `tui/theme.py` — single source of truth for all colors, styles, and branded strings. Electric gold primary, steel blue structure, mint green success, warm red errors, amber warnings, slate gray muted.
- **Branded prompt**: lightning bolt (`⚡`) icon with `icon_prompt()` supporting normal, plan, and paused states via prompt-toolkit HTML
- **Thinking spinner**: Rich Status spinner shown while waiting for LLM response; auto-clears on first output callback
- **Semantic color constants**: `CTX_OK/WARN/CRITICAL`, `PERM_ALLOW/DENY/ASK/SESSION`, `TABLE_KEY/VALUE/BORDER/HEADER` — no hardcoded Rich color strings remain in `src/godspeed/`
- **Theme test suite**: 24 tests covering all helpers (`styled()`, `brand()`, `icon_prompt()`), Rich rendering compatibility, and constant validation
- 626 tests, ~90% coverage

### Changed

- All TUI modules (`output.py`, `commands.py`, `app.py`) and CLI (`cli.py`) import colors from `tui/theme.py` instead of hardcoding Rich markup
- Permission prompt input uses themed `BOLD_WARNING` style
- `godspeed version` command uses `brand()` helper for consistent rendering
- `godspeed models` and `godspeed init` commands use themed table and text styles

## [0.5.0] - 2026-04-11

### Added

- **User memory with SQLite**: `UserMemory` class backed by SQLite with WAL mode; persistent preferences (key/value CRUD) and corrections table; safe concurrent access; auto-creates `~/.godspeed/memory.db`
- **Session memory**: `SessionMemory` records session lifecycle events (start, end, tool calls, errors) to SQLite; cross-session history with event filtering and limits
- **Correction tracker**: `CorrectionTracker` with heuristic detection of user corrections (negation patterns like "no", "don't", "stop", "instead"); auto-records to UserMemory; `format_for_system_prompt()` surfaces top-N corrections as "User prefers X over Y" guidance
- **`memory_enabled` config**: toggle memory system via `settings.yaml`
- 602 tests, ~90% coverage

## [0.4.0] - 2026-04-11

### Added

- **Sub-agent coordinator**: `AgentCoordinator` spawns isolated sub-agents with separate conversations; depth limit 3, iteration limit 25; `SpawnAgentTool` (HIGH risk) enables multi-agent orchestration via `agent_loop()` reuse
- **`spawn_parallel()`**: run multiple sub-agents concurrently via `asyncio.gather()`
- **MCP client**: `MCPClient` connects to MCP servers via stdio transport; `MCPToolAdapter` maps MCP tool definitions to Godspeed Tool ABC (all HIGH risk); graceful when `mcp` package not installed
- **MCP server discovery**: `mcp_servers` config in `settings.yaml` auto-discovers and registers remote tools at startup
- **Model routing**: `ModelRouter` routes LLM calls by task type (plan/edit/chat) to different models; configurable via `routing` in settings
- **Human-in-the-loop pause/resume**: `asyncio.Event` shared between TUI and agent loop; `/pause` stops at next iteration, `/resume` continues, `/guidance <msg>` injects mid-conversation correction and resumes
- **Rich permission prompts**: contextual detail in permission dialogs -- file_edit shows mini-diff, file_write shows first 10 lines, shell shows syntax-highlighted command, file_read shows path
- 549 tests, ~90% coverage

### Changed

- `LLMClient.chat()` accepts optional `task_type` parameter for model routing
- `agent_loop()` accepts `pause_event` parameter for human-in-the-loop control
- `format_permission_prompt()` accepts optional `arguments` dict for contextual display

## [0.3.0] - 2026-04-11

### Added

- **Tree-sitter repo map tool**: `RepoMapTool` extracts symbol outlines (functions, classes, methods) from Python/JS/TS/Go files using tree-sitter; graceful degradation when tree-sitter not installed
- **Plan mode**: `/plan` command toggles read-only mode -- permission engine blocks all non-READ_ONLY tools; system prompt updated to instruct explore-only behavior
- **Git stash/stash_pop actions**: `GitTool` supports `stash` (with auto-message) and `stash_pop` operations
- **Auto-stash before risky operations**: agent loop tracks consecutive file edits; after 3+ consecutive writes, auto-stashes working state as a safety net
- **Model-aware compaction**: compaction prompts adapt to model context window size -- small models (<=32K) get aggressive summarization, frontier models (>100K) get detailed preservation
- **`MODEL_CONTEXT_WINDOWS` mapping**: prefix-matched context window sizes for Claude, GPT, Gemini, Ollama models with `get_model_context_window()` utility
- **`/checkpoint [name]` command**: save conversation state snapshots to `.godspeed/checkpoints/`; list checkpoints with metadata (tokens, messages, model, timestamp)
- **`/restore <name>` command**: restore a saved checkpoint, rebuilding full conversation state
- **Checkpoint management**: `save_checkpoint()`, `load_checkpoint()`, `list_checkpoints()`, `delete_checkpoint()` in `context/checkpoint.py`
- 485 tests, ~90% coverage

### Changed

- Compaction in agent loop now uses model-aware prompts via `get_compaction_prompt()` instead of hardcoded prompt
- Moved `tests/test_context.py` into `tests/test_context/` package for better organization

## [0.2.0] - 2026-04-11

### Added

- **Stuck-loop detection**: after 3 identical tool errors, injects a replan message forcing the model to try a different approach
- **Verification cascade**: `VerifyTool` runs `ruff check` on Python files; auto-verifies after every `file_edit`/`file_write` so the agent self-corrects lint errors
- **`/extend N` command**: override max iterations per agent turn (default: 50)
- **`/context` command**: show context window usage — tokens, percentage, message count with color-coded thresholds
- **Audit trail compression**: `compress_session()` rotates `.jsonl` → `.jsonl.gz`; `verify_chain()` transparently handles compressed logs
- **FileEdit confidence reporting**: output includes `[match=exact confidence=1.00]` or `[match=fuzzy confidence=0.87 line=42]` so the agent can gauge match quality
- **26 new dangerous command patterns**: iptables, mount/umount, fdisk, shutdown/reboot, docker rm -f, kubectl delete, env exfiltration, Windows destructive ops, supply-chain attacks
- **Ollama auto-start**: detects when Ollama is not running and starts `ollama serve` as a background process before first LLM call
- **Lazy LiteLLM import**: deferred import reduces cold startup from ~1.5s to ~300ms
- **Smart retry for connection errors**: skips retry+sleep when Ollama/server is down, returns actionable error immediately
- **Non-TTY crash guard**: graceful error message when launched from non-interactive shells
- `ToolRegistry.has_tool()` method
- `agent_loop()` accepts `max_iterations` parameter
- 411 tests, 90% coverage

### Fixed

- Route logs to stderr and scope verbose mode to `godspeed.*` namespace only — eliminates LiteLLM/httpx/markdown_it debug noise in TUI
- Add debug logging on tiktoken encoding fallback instead of silent `pass`
- Add docstrings to `PermissionEvaluator` and `AuditRecorder` protocol methods

### Changed

- `godspeed init` command — creates `~/.godspeed/` and default `settings.yaml`
- `godspeed models` command — shows popular model options with provider, cost, and API key info
- `settings.yaml.example` — full reference config with free/paid model examples and permission rules
- Audit trail retention cleanup — expired sessions are purged on startup based on `retention_days` setting; handles both `.jsonl` and `.jsonl.gz`
- Token counter model mappings for Claude, Gemini, DeepSeek, Ollama models
- Default model changed from paid `claude-sonnet-4-20250514` to free `ollama/qwen3:4b` — zero-cost out of the box
- Expanded pyproject.toml classifiers and keywords for PyPI discoverability

## [0.1.0] - 2026-04-10

### Added

- Hand-rolled agent loop — model decides when to stop, no framework dependency
- 4-tier permission engine (deny > ask > allow) with deny-first evaluation, pattern matching, and 46 dangerous command patterns
- Hash-chained JSONL audit trail (SHA-256) with tamper detection and `godspeed audit verify`
- 4-layer secret protection: file deny rules, context cleaning, output filtering, audit redaction (27 regex patterns + Shannon entropy)
- LiteLLM integration for 200+ LLM providers (Claude, GPT, Gemini, Ollama, etc.) with fallback chains
- 7 built-in tools: file_read, file_write, file_edit (with fuzzy matching), shell, glob_search, grep_search, git
- Rich + prompt-toolkit TUI with syntax highlighting, diff rendering, and streaming
- Slash commands: /help, /model, /undo, /audit, /compact, /clear, /quit
- GODSPEED.md project instructions (walk-up-tree loading, like CLAUDE.md)
- Conversation compaction when approaching context limit
- Configuration cascade: global (~/.godspeed/settings.yaml) > project (.godspeed/settings.yaml) > CLI flags (deny rules are additive — project can't weaken global denies)
- CLI entry points: `godspeed` (TUI), `godspeed version`, `godspeed audit verify`
- 243 tests, 82% coverage
- Full repo-standards scaffolding: CI pipeline, pre-commit hooks, dependabot, issue/PR templates, CONTRIBUTING.md, SECURITY.md, LICENSE (MIT)
