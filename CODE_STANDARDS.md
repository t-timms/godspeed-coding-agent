# Godspeed Code Standards

> **Audience:** Human contributors, AI coding agents, and automated tooling.
> **Companion files:** `CONTRIBUTING.md` (workflow), `CLAUDE.md` (agent quickstart),
> `SECURITY.md` (vulnerability reporting), `pyproject.toml` (tool config).

---

## 1. File-Level Requirements (Every `.py` File)

### Required
```python
from __future__ import annotations
```
First import in every module. No exceptions. Validated by ruff `FA` (flake8-future-annotations).

### Prohibited
- `print()` in production code — use `logging.getLogger(__name__)`. Validated by ruff `T20`.
- Bare `except:` — use specific exception types. Validated by ruff `E722`.
- `except Exception:` without re-raise or logging. Validated by ruff `BLE`.
- Unused imports or variables. Validated by ruff `F401`, `F811`, `F821`, `F841`.
- Unused function arguments without `_` prefix. Validated by ruff `ARG`.

### Required Structure
1. Shebang or encoding line (if needed)
2. Module docstring
3. `from __future__ import annotations`
4. Standard library imports
5. Third-party imports
6. Local imports
7. Module-level constants (UPPER_CASE)
8. Classes and functions

---

## 2. Type Annotations

### Functions
```python
def process(items: list[str], limit: int = 10) -> dict[str, int]:
    ...
```

### Methods
```python
class Registry:
    def get(self, name: str) -> Tool | None:
        ...
```

### Callbacks
Use type aliases for clarity:
```python
OnAssistantText = Callable[[str], None]
```

### Allowable Exception
`Any` is acceptable in tests, mocks, and protocol boundaries. Prefer `object` or a Protocol for public APIs.

---

## 3. Logging

### Correct
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Tool dispatched name=%s latency_ms=%.1f", name, latency)
logger.debug("Cache hit rate=%.1f%%", hit_rate)
logger.warning("Permission denied tool=%s reason=%s", tool, reason)
logger.error("Audit write failed session=%s", session_id, exc_info=True)
```

### Prohibited
- f-strings in logger calls (`logger.info(f"x={x}")`) — defeats lazy evaluation
- `print()` for any purpose other than CLI scripts in `scripts/`

---

## 4. Exception Handling

### Correct
```python
try:
    result = await risky_operation()
except ConnectionError:
    logger.warning("Connection failed, falling back")
    result = fallback()
except ValueError as exc:
    raise RuntimeError(f"Invalid state: {exc}") from exc
```

### Prohibited
```python
except:                          # bare except
except Exception:                # too broad without re-raise
    pass                         # silent failure
```

### Exception Chain
Always use `raise ... from exc` when wrapping exceptions to preserve the traceback chain.

---

## 5. Testing

### Test File Naming
- `tests/test_{module_name}.py` for module-level tests
- `tests/test_{package}/test_{module}.py` for subpackage tests

### Fixture Pattern
```python
import pytest
from pathlib import Path

@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    return tmp_path
```

### Async Tests
```python
@pytest.mark.asyncio
async def test_async_operation() -> None:
    result = await some_async_fn()
    assert result == expected
```

### Coverage
- Gate: 80% branch coverage (`fail_under = 80` in pyproject.toml)
- All new features must include tests
- Security code requires edge-case tests
- Mock external calls (LiteLLM, subprocess, network)
- Use `pytest -m "not real_llm"` to skip tests requiring a running LLM

---

## 6. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | snake_case | `agent_loop.py` |
| Packages | lowercase | `security/` |
| Classes | PascalCase | `PermissionEngine` |
| Functions/Methods | snake_case | `evaluate_tool_call()` |
| Constants | UPPER_CASE | `MAX_ITERATIONS` |
| Private members | `_leading_underscore` | `_model_lower` |
| Type aliases | PascalCase | `OnAssistantText` |
| Test functions | `test_{what}_{condition}` | `test_permission_denies_shell` |
| Test classes | `Test{Component}` | `TestPermissionEngine` |

---

## 7. Module Organization

### Package Structure
Each package must have:
- `__init__.py` with explicit `__all__` exports
- No circular imports between packages

### File Size Limits
- Ideal: < 500 lines
- Acceptable: < 1000 lines
- Review needed: > 1000 lines (consider splitting)
- Split candidates: `cli.py`, `agent/loop.py`, `tui/commands.py`

### Dependency Direction
```
tools/ ← security/ ← agent/ ← llm/ ← tui/
   ↓         ↓
context/  audit/
```

Lower packages must not import from higher packages. Tools don't import from agent. No circular dependencies.

---

## 8. Ruff Configuration

We use ruff for all linting and formatting. Configuration in `pyproject.toml`.

### Enabled Rule Sets
| Code | Source | Purpose |
|------|--------|---------|
| `E`, `W` | pycodestyle | Style errors and warnings |
| `F` | Pyflakes | Unused imports, undefined names |
| `I` | isort | Import ordering |
| `N` | pep8-naming | Naming conventions |
| `UP` | pyupgrade | Modern Python syntax |
| `B` | flake8-bugbear | Common bug patterns |
| `SIM` | flake8-simplify | Code simplification |
| `RUF` | Ruff-specific | Ruff-specific checks |
| `S` | flake8-bandit | Security (S603, S607 suppressed — subprocess intentional) |
| `T20` | flake8-print | Print statement detection |
| `FA` | flake8-future-annotations | `from __future__ import annotations` |
| `ARG` | flake8-unused-arguments | Unused function arguments |
| `BLE` | flake8-blind-except | Blind exception catching |
| `RET` | flake8-return | Return check consistency |
| `DOC` | pydoclint | Docstring coverage (preview) |

### Per-File Overrides
- `tests/**`: Allow `assert`, hardcoded passwords, nested `with` (S101, S105, S106, SIM117)
- `scripts/**`: Allow `print()` (T201)
- `experiments/**`: Relaxed for research code

---

## 9. Security Code (Additional Rules)

All code in `src/godspeed/security/` must follow these additional standards:

1. **Fail-closed:** Any ambiguity or error must result in denial, not allowance
2. **Secret redaction:** Never log, audit, or display API keys, tokens, or credentials
3. **Input validation:** Validate all user-controlled input before processing
4. **Pattern tests:** Every dangerous command regex needs a test with real-world examples
5. **Entropy checks:** Secret detection must use Shannon entropy ≥ 4.5 bits/char

---

## 10. AI Agent Contribution Policy

Godspeed is designed to be developed with AI coding agents. All AI-assisted contributions must:

1. **Disclose:** PR description must state AI assistance was used
2. **Verify:** Author must manually review every line of AI-generated code
3. **Test:** AI-generated code must pass the same test suite as human code
4. **Attribute:** If AI generates a novel approach, cite the prompt or conversation
5. **No secrets:** Never paste API keys, tokens, or `.env` content into AI prompts
6. **Review chain:** AI-generated code must be reviewed by at least one human before merge

This policy is based on [Ruff's AI Policy](https://github.com/astral-sh/.github/blob/main/AI_POLICY.md).

---

## 11. Commit Standards

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): description
fix(scope): description
docs: description
test: description
refactor(scope): description
security: description
chore: description
```

Scopes: `agent`, `security`, `tools`, `llm`, `tui`, `context`, `audit`, `config`, `cli`, `evolution`, `training`, `skills`, `mcp`, `eval`

---

## 12. Documentation Standards

- **Architecture:** `GODSPEED_ARCHITECTURE.md` — updated for major architectural changes
- **Changelog:** `CHANGELOG.md` — Keep a Changelog format, every user-facing change
- **Roadmap:** `ROADMAP.md` — living document, updated quarterly
- **Threat Model:** `THREAT_MODEL.md` — updated when security surface changes
- **Docstrings:** Google-style with Args/Returns/Raises sections for public APIs

---

## 13. Quality Gates (CI)

Every PR must pass:

| Gate | Command | Threshold |
|------|---------|-----------|
| Lint | `uv run ruff check .` | Zero errors |
| Format | `uv run ruff format . --check` | Clean |
| Type check | `uv run ty check src/` | Zero errors |
| Security | `uv run pip-audit` | Zero known vulns |
| SAST | `uv run bandit -r src/ -ll` | Zero HIGH/MEDIUM |
| Tests | `uv run pytest --cov -m "not real_llm"` | Pass |
| Coverage | `--cov-fail-under=80` | ≥ 80% |

---

## 14. Tools Quick Reference

```bash
# Lint and auto-fix
uv run ruff check . --fix

# Format
uv run ruff format .

# Type check
uv run ty check src/

# Security audit
uv run pip-audit
uv run bandit -r src/ -ll

# Dead code
uv run vulture src/godspeed/ --min-confidence 70

# Tests (skip real LLM)
uv run pytest -m "not real_llm" --cov

# Full test suite
uv run pytest --cov --cov-fail-under=80

# Pre-commit (run before every commit)
uv run pre-commit run --all-files
```

---

*Last updated: 2026-05-09. This file is authoritative — CLAUDE.md and CONTRIBUTING.md
defer to it for any conflicting guidance.*
