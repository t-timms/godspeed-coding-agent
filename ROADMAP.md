# Godspeed Roadmap

This roadmap describes the path from the current Alpha to a stable v1.0 release. Items are grouped by phase and ordered by priority within each phase.

---

## Phase 1: Foundation (Current — v0.4.x)

**Goal:** Stabilize the core agent loop, security model, and developer experience. Fix known rough edges before inviting broader testing.

> **Landed (feature wave):** CLI session continuity (`--continue`, `--resume <id>`, `list-sessions`), `godspeed mcp add/list/remove`, TUI commands (`/compact`, `/verify`, `/btw`, `/goal`, `/rewind`, `/batch`, `/usage`), message queueing (Ctrl+Q), bash pass-through (`!`/`!!`), esc-esc rewind picker, image attachments, plan-mode approval gate, and `statusline` HUD config.

- [x] Hash-chained audit trail with fail-closed I/O
- [x] 4-tier permission engine with dangerous-command detection
- [x] Schema-validated tool calls with retry logic
- [x] Parallel + speculative tool dispatch
- [x] Prompt evaluation harness for tool-call accuracy
- [x] Active metrics thresholds and alerting
- [x] Cross-platform CI (Ubuntu + Windows)
- [x] **Docker / devcontainer quickstart** — containerized environment for sandboxed testing
- [ ] **TUI screenshot / GIF in README** — visual proof of the interface
- [ ] **Coverage for user-facing code** — bring `cli.py` and `tui/*` into the coverage gate
- [ ] **Merge pending dependency updates** — keep Dependabot PRs current

**Exit criteria for Phase 1:**
- All CI checks green for 14 consecutive days
- No open P0/P1 bugs
- At least 3 external testers have run the full quickstart without hitting blockers

---

## Phase 2: Community Beta (v0.5.x — v0.9.x)

**Goal:** Build trust through external usage, gather feedback, and establish governance.

- [ ] Enable GitHub Discussions for informal feedback
- [ ] File `good first issue` items to guide new contributors
- [ ] Add VS Code extension (basic: send prompt, display tool calls, diff viewer)
- [ ] Improve error messages and onboarding (first-run wizard, `--tutorial` flag)
- [ ] Publish honest single-run benchmark on full SWE-Bench Lite (300 instances)
- [ ] Graduate `Development Status` from Alpha to Beta after 30 days of external usage with no breaking changes
- [ ] Establish a second regular reviewer or co-maintainer
- [ ] Security audit by a third party (at minimum, a paid CodeQL + manual review pass)

**Exit criteria for Phase 2:**
- 50+ GitHub stars (minimum social proof threshold)
- 5+ external contributors with merged PRs
- No breaking changes in the last 30 days
- Published benchmark on full SWE-Bench Lite with single-run methodology

---

## Phase 3: v1.0 Stable

**Goal:** Freeze the API/CLI surface, ship a release the community can depend on.

- [ ] Freeze CLI arguments, settings.yaml schema, and tool JSON Schema
- [ ] MkDocs / Material documentation site (not just README)
- [ ] Plugin ecosystem documentation (MCP servers, custom tools, hooks)
- [ ] Stable release notes and migration guide from v0.x
- [ ] Long-term support policy (security fixes for last 2 minor versions)

**Exit criteria for Phase 3:**
- v1.0.0 tagged
- All docs site pages reviewed by at least one external contributor
- 30 days since last breaking change

---

## Phase 4: Scale (Post-v1.0)

**Goal:** Expand capability, distribution, and sustainability.

- [ ] JetBrains IDE plugin
- [ ] Self-hosted web UI (optional companion to the CLI)
- [ ] Fine-tuning pipeline for custom models on user conversation logs
- [ ] Multi-agent orchestration beyond depth-3 coordinator
- [ ] OpenCollective or GitHub Sponsors for sustainability
- [ ] Annual third-party security audit

---

## How to influence this roadmap

- **Open a Discussion** for feature ideas or use-case requests
- **Open an Issue** for bugs or concrete proposals
- **Comment on existing issues** with your use case — we prioritize based on real-world demand, not speculation

This roadmap is a living document. Last updated: 2026-09-06.
