"""Every tracked Python source file must be UTF-8 and parseable.

`experiments/swebench_lite/run_in_loop.py` was committed as UTF-16LE (with BOM) in May 2026;
Python cannot import such a file ("source code string cannot contain null bytes"), which silently
broke `scripts/validate_driver.py` and the agent-in-loop benchmark path. Nothing in the suite
imported it, so nothing noticed. This test walks the tree so the next one fails in CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEARCH_DIRS = ("src", "scripts", "experiments", "tests", "benchmarks")
# benchmarks/fixtures hold deliberately broken code (e.g. easy-fix-syntax-01) for the agent to repair.
_SKIP_PARTS = {"__pycache__", ".venv", "node_modules", ".git", "fixtures"}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in _SEARCH_DIRS:
        base = _REPO_ROOT / name
        if not base.is_dir():
            continue
        files.extend(p for p in base.rglob("*.py") if not _SKIP_PARTS.intersection(p.parts))
    return sorted(files)


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_python_source_is_utf8_and_parses(path: Path) -> None:
    raw = path.read_bytes()
    assert raw[:2] not in (b"\xff\xfe", b"\xfe\xff"), f"{path} has a UTF-16 BOM; re-save as UTF-8"
    assert b"\x00" not in raw, f"{path} contains NUL bytes (UTF-16/UTF-32?); re-save as UTF-8"
    ast.parse(raw.decode("utf-8-sig"), filename=str(path))
