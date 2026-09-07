# GODSPEED.md — Godspeed Coding Agent

## Project

Godspeed is a security-first open-source coding agent built in Python. It
provides an interactive terminal UI (TUI) where an LLM can read files, write
code, run shell commands, and use 30+ built-in tools — all gated by a 4-tier
permission engine and recorded in a tamper-evident audit trail.

## Stack

- Python 3.11–3.14, Pydantic v2, pydantic-settings
- LiteLLM for 200+ LLM provider support
- prompt-toolkit + Rich for the TUI
- pytest + pytest-asyncio + pytest-cov for testing
- ruff for linting and formatting (line length 100)
- ty (Astral) for type checking; mypy as fallback
- pip-audit + bandit for security scanning
- uv for dependency management and build

## Development Commands

```bash
# Install all dependencies
uv sync --all-extras

# Lint and format
ruff check . --fix && ruff format .

# Type check
uv run ty check src/

# Security scan
uv run pip-audit --ignore-vuln CVE-2026-28684 --ignore-vuln CVE-2026-3219
uv run bandit -r src/ -ll

# Run tests
uv run pytest --cov

# Run the TUI locally
uv run python -m godspeed
```

## Code Standards

- `from __future__ import annotations` at the top of every module
- Type hints on all public functions
- No `print()` in production — use `logging.getLogger(__name__)`
- Structured logging: `logger.info("event key=%s", value)` (no f-strings)
- Specific exceptions only — never bare `except:`
- Tests for every new feature, especially security-related code
- Conventional Commits format: `feat(scope): description`

## Architecture Notes

- **Agent loop** (`src/godspeed/agent/loop.py`) — hand-rolled async ReAct loop.
  The LLM decides when to stop. Streaming + speculative dispatch for reads.
- **Permission engine** (`src/godspeed/security/`) — deny-first, 4 tiers.
  Every tool call is evaluated before execution.
- **Audit trail** (`src/godspeed/audit/`) — SHA-256 hash-chained JSONL.
  Fail-closed on write errors.
- **Tools** (`src/godspeed/tools/`) — 30+ built-in tools with JSON Schema.
  New tools extend the `Tool` ABC.
- **LLM layer** (`src/godspeed/llm/`) — LiteLLM wrapper with fallback chains,
  model routing by task type, token counting, cost tracking.
- **Godspeed Lite** (`src/godspeed/lite/`) — minimal bash-only agent for benchmarks (3 modes, model roulette)
- **Benchmarks** (`src/godspeed/benchmarks/`) — NIM key rotation, pre-flight checks, SWE-bench runner

## Security Checklist for Changes

- [ ] Permission engine changes have edge-case tests
- [ ] Dangerous command patterns have tests with real-world examples
- [ ] Secret detection patterns have tests with real-world examples
- [ ] Audit trail integrity is maintained (hash chain, fsync)
- [ ] No hardcoded secrets or API keys

## Testing Notes

- `pytest -m "not real_llm"` skips tests that need a running Ollama server.
- `pytest --cov --cov-fail-under=80` enforces coverage gate.
- Adversarial tests in `tests/test_adversarial.py` cover prompt injection
  and jailbreak attempts.
- Windows-specific code (shell detection, UTF-8 stdio wrapping, process tree
  killing) exists but is not yet covered by CI.

## License

MIT — see `LICENSE`.
