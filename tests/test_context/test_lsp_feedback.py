"""Tests for LSPFeedbackProvider — passive LSP diagnostics feedback."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from godspeed.context.lsp_feedback import (
    Diagnostic,
    FileDiagnostics,
    LSPFeedbackProvider,
    SEVERITY_ERROR,
    SEVERITY_HINT,
    SEVERITY_INFORMATION,
    SEVERITY_WARNING,
    _DIAG_CACHE_TTL,
    _pyright_severity,
)


# ── Diagnostic dataclass tests ────────────────────────────────────────


class TestDiagnostic:
    """Test Diagnostic frozen dataclass."""

    def test_severity_name_error(self) -> None:
        d = _make_diag(severity=SEVERITY_ERROR)
        assert d.severity_name == "error"
        assert d.is_error is True
        assert d.is_warning is False

    def test_severity_name_warning(self) -> None:
        d = _make_diag(severity=SEVERITY_WARNING)
        assert d.severity_name == "warning"
        assert d.is_error is False
        assert d.is_warning is True

    def test_severity_name_info(self) -> None:
        d = _make_diag(severity=SEVERITY_INFORMATION)
        assert d.severity_name == "info"

    def test_severity_name_hint(self) -> None:
        d = _make_diag(severity=SEVERITY_HINT)
        assert d.severity_name == "hint"

    def test_severity_name_unknown(self) -> None:
        d = _make_diag(severity=999)
        assert d.severity_name == "unknown"

    def test_frozen(self) -> None:
        d = _make_diag()
        with pytest.raises(AttributeError):
            d.line = 2  # type: ignore[misc]


# ── FileDiagnostics tests ─────────────────────────────────────────────


class TestFileDiagnostics:
    """Test FileDiagnostics counts."""

    def test_error_count(self) -> None:
        fd = FileDiagnostics(
            file_path="test.py",
            diagnostics=[
                _make_diag(severity=SEVERITY_ERROR),
                _make_diag(severity=SEVERITY_ERROR),
                _make_diag(severity=SEVERITY_WARNING),
            ],
            timestamp=0.0,
        )
        assert fd.error_count == 2
        assert fd.warning_count == 1

    def test_empty_diagnostics(self) -> None:
        fd = FileDiagnostics(file_path="test.py", diagnostics=[], timestamp=0.0)
        assert fd.error_count == 0
        assert fd.warning_count == 0


# ── LSPFeedbackProvider init tests ────────────────────────────────────


class TestProviderInit:
    """Test provider initialization and properties."""

    def test_default_init(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        assert p.enabled is True

    def test_disabled_init(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path, enabled=False)
        assert p.enabled is False

    def test_enable_disable_toggle(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        assert p.enabled is True
        p.enabled = False
        assert p.enabled is False
        p.enabled = True
        assert p.enabled is True

    def test_max_diagnostics_configurable(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path, max_diagnostics=5)
        assert p._max_diagnostics == 5


# ── Content hash tests ────────────────────────────────────────────────


class TestContentHash:
    """Test content hash computation."""

    def test_hash_from_content(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        h1 = p._content_hash(tmp_path / "x.py", "hello")
        h2 = p._content_hash(tmp_path / "x.py", "hello")
        h3 = p._content_hash(tmp_path / "x.py", "world")
        assert h1 == h2
        assert h1 != h3

    def test_hash_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("print('hi')")
        p = LSPFeedbackProvider(project_dir=tmp_path)
        h1 = p._content_hash(f)
        h2 = p._content_hash(f, content="print('hi')")
        assert h1 == h2

    def test_hash_unreadable_file(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        h = p._content_hash(tmp_path / "nonexistent.py")
        assert h == "unreadable"


# ── Cache tests ───────────────────────────────────────────────────────


class TestCache:
    """Test diagnostic cache behavior."""

    def test_cache_set_and_get(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fp = tmp_path / "test.py"
        fd = FileDiagnostics(
            file_path=str(fp),
            diagnostics=[_make_diag()],
            timestamp=time.time(),
        )
        p._set_cached(fp, "abc123", fd)
        result = p._get_cached(fp, "abc123")
        assert result is not None
        assert result.file_path == str(fp)

    def test_cache_miss(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        result = p._get_cached(tmp_path / "x.py", "nope")
        assert result is None

    def test_cache_expired(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fp = tmp_path / "test.py"
        fd = FileDiagnostics(
            file_path=str(fp),
            diagnostics=[_make_diag()],
            timestamp=time.time(),
        )
        p._set_cached(fp, "abc123", fd)
        # Manually age the cache entry
        cache_key = f"{fp}:abc123"
        p._cache[cache_key].created_at = time.time() - _DIAG_CACHE_TTL - 1
        result = p._get_cached(fp, "abc123")
        assert result is None
        assert cache_key not in p._cache

    def test_clear_cache_all(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fp = tmp_path / "test.py"
        fd = FileDiagnostics(
            file_path=str(fp),
            diagnostics=[_make_diag()],
            timestamp=time.time(),
        )
        p._set_cached(fp, "abc", fd)
        p._set_cached(fp, "def", fd)
        removed = p.clear_cache()
        assert removed == 2
        assert p.cache_size == 0

    def test_clear_cache_by_file(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fp1 = tmp_path / "a.py"
        fp2 = tmp_path / "b.py"
        fd = FileDiagnostics(
            file_path="",
            diagnostics=[_make_diag()],
            timestamp=time.time(),
        )
        p._set_cached(fp1, "h1", fd)
        p._set_cached(fp1, "h2", fd)
        p._set_cached(fp2, "h3", fd)
        removed = p.clear_cache(fp1)
        assert removed == 2
        assert p.cache_size == 1


# ── Formatting tests ──────────────────────────────────────────────────


class TestFormatting:
    """Test diagnostic formatting for context injection."""

    def test_format_for_prompt_empty(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        result = p.format_for_prompt([])
        assert result == ""

    def test_format_for_prompt_with_errors(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fd = FileDiagnostics(
            file_path="auth.py",
            diagnostics=[
                _make_diag(
                    file_path="auth.py",
                    severity=SEVERITY_ERROR,
                    message="Name not defined",
                    code="reportUndefinedVariable",
                    line=10,
                    column=5,
                ),
                _make_diag(
                    file_path="auth.py",
                    severity=SEVERITY_WARNING,
                    message="Unused import",
                    code="unusedImport",
                    line=1,
                    column=0,
                ),
            ],
            timestamp=time.time(),
        )
        result = p.format_for_prompt([fd])
        assert "LSP Diagnostics" in result
        assert "error: auth.py:10:5" in result
        assert "[reportUndefinedVariable]" in result
        assert "warning: auth.py:1:0" in result
        # Errors should come before warnings
        error_pos = result.index("error:")
        warn_pos = result.index("warning:")
        assert error_pos < warn_pos

    def test_format_for_prompt_truncation(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        diags = [_make_diag(severity=SEVERITY_WARNING, message=f"warn {i}") for i in range(50)]
        fd = FileDiagnostics(
            file_path="test.py",
            diagnostics=diags,
            timestamp=time.time(),
        )
        result = p.format_for_prompt([fd], max_total=10)
        assert result.count("warning:") == 10

    def test_format_for_tool_result_empty(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fd = FileDiagnostics(file_path="test.py", diagnostics=[], timestamp=0.0)
        result = p.format_for_tool_result(fd)
        assert "No diagnostics" in result
        assert "test.py" in result

    def test_format_for_tool_result_with_diags(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        fd = FileDiagnostics(
            file_path="auth.py",
            diagnostics=[
                _make_diag(
                    severity=SEVERITY_ERROR,
                    message="type error",
                    code="assignment",
                    line=42,
                    column=8,
                ),
                _make_diag(
                    severity=SEVERITY_WARNING,
                    message="unused var",
                    code="unusedVariable",
                    line=10,
                    column=0,
                ),
            ],
            timestamp=time.time(),
        )
        result = p.format_for_tool_result(fd)
        assert "auth.py" in result
        assert "error: L42:C8" in result
        assert "[assignment]" in result
        assert "1 errors" in result
        assert "1 warnings" in result


# ── Async diagnostics (mocked) tests ──────────────────────────────────


class TestAsyncDiagnostics:
    """Test async diagnostic collection with mocked LSP."""

    @pytest.mark.asyncio
    async def test_get_diagnostics_disabled(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path, enabled=False)
        result = await p.get_diagnostics(tmp_path / "test.py")
        assert result.diagnostics == []

    @pytest.mark.asyncio
    async def test_get_diagnostics_no_lsp(self, tmp_path: Path) -> None:
        """No LSP available returns empty diagnostics."""
        p = LSPFeedbackProvider(project_dir=tmp_path, enabled=True)
        # Force availability check (will fail if no pyright/ruff)
        p._available = True
        p._lsp_command = None
        result = await p.get_diagnostics(tmp_path / "test.py")
        assert result.diagnostics == []

    @pytest.mark.asyncio
    async def test_get_diagnostics_caches(self, tmp_path: Path) -> None:
        """Second call uses cache."""
        p = LSPFeedbackProvider(project_dir=tmp_path)
        p._available = True
        p._lsp_command = None  # No actual LSP

        fp = tmp_path / "test.py"
        fp.write_text("x = 1")
        r1 = await p.get_diagnostics(fp)
        r2 = await p.get_diagnostics(fp, content="x = 1")
        # Both should return empty (no LSP), but second should hit cache
        assert r1.diagnostics == []
        assert r2.diagnostics == []

    @pytest.mark.asyncio
    async def test_on_file_edited_unsupported_extension(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        result = await p.on_file_edited(tmp_path / "image.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_on_file_edited_disabled(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path, enabled=False)
        result = await p.on_file_edited(tmp_path / "test.py")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_diagnostics_batch(self, tmp_path: Path) -> None:
        p = LSPFeedbackProvider(project_dir=tmp_path)
        p._available = True
        p._lsp_command = None

        fps = [tmp_path / f"file{i}.py" for i in range(3)]
        results = await p.get_diagnostics_batch(fps)
        assert len(results) == 3
        for r in results:
            assert r.diagnostics == []


# ── Pyright severity mapping tests ────────────────────────────────────


class TestPyrightSeverity:
    """Test pyright severity string mapping."""

    def test_error(self) -> None:
        assert _pyright_severity("error") == SEVERITY_ERROR

    def test_warning(self) -> None:
        assert _pyright_severity("warning") == SEVERITY_WARNING

    def test_information(self) -> None:
        assert _pyright_severity("information") == SEVERITY_INFORMATION

    def test_hint(self) -> None:
        assert _pyright_severity("hint") == SEVERITY_HINT

    def test_unknown_defaults_warning(self) -> None:
        assert _pyright_severity("unknown_severity") == SEVERITY_WARNING

    def test_case_insensitive(self) -> None:
        assert _pyright_severity("ERROR") == SEVERITY_ERROR
        assert _pyright_severity("Warning") == SEVERITY_WARNING


# ── on_file_edited filtering tests ────────────────────────────────────


class TestOnFileEditedFiltering:
    """Test on_file_edited diagnostic filtering logic."""

    @pytest.mark.asyncio
    async def test_no_injection_without_errors_or_warnings(self, tmp_path: Path) -> None:
        """Files with < 3 warnings and no errors → None."""
        p = LSPFeedbackProvider(project_dir=tmp_path)
        p._available = True
        p._lsp_command = None

        fp = tmp_path / "clean.py"
        fd = FileDiagnostics(
            file_path=str(fp),
            diagnostics=[
                _make_diag(severity=SEVERITY_WARNING),
                _make_diag(severity=SEVERITY_HINT),
            ],
            timestamp=time.time(),
        )
        # Manually set cache to return non-empty diagnostics
        content_hash = p._content_hash(fp)
        p._set_cached(fp, content_hash, fd)

        result = await p.on_file_edited(fp)
        assert result is None

    @pytest.mark.asyncio
    async def test_inject_with_errors(self, tmp_path: Path) -> None:
        """Files with errors → inject."""
        p = LSPFeedbackProvider(project_dir=tmp_path)
        p._available = True
        p._lsp_command = None

        fp = tmp_path / "broken.py"
        fd = FileDiagnostics(
            file_path=str(fp),
            diagnostics=[
                _make_diag(severity=SEVERITY_ERROR, message="type error"),
            ],
            timestamp=time.time(),
        )
        content_hash = p._content_hash(fp)
        p._set_cached(fp, content_hash, fd)

        result = await p.on_file_edited(fp)
        assert result is not None
        assert "error:" in result

    @pytest.mark.asyncio
    async def test_inject_with_many_warnings(self, tmp_path: Path) -> None:
        """Files with >= 3 warnings → inject."""
        p = LSPFeedbackProvider(project_dir=tmp_path)
        p._available = True
        p._lsp_command = None

        fp = tmp_path / "warny.py"
        fd = FileDiagnostics(
            file_path=str(fp),
            diagnostics=[_make_diag(severity=SEVERITY_WARNING, message=f"w{i}") for i in range(3)],
            timestamp=time.time(),
        )
        content_hash = p._content_hash(fp)
        p._set_cached(fp, content_hash, fd)

        result = await p.on_file_edited(fp)
        assert result is not None
        assert "3 warnings" in result


# ── Helpers ───────────────────────────────────────────────────────────


def _make_diag(
    file_path: str = "test.py",
    line: int = 1,
    column: int = 0,
    severity: int = SEVERITY_WARNING,
    message: str = "test diagnostic",
    code: str | None = None,
) -> Diagnostic:
    """Create a Diagnostic with defaults for testing."""
    return Diagnostic(
        file_path=file_path,
        line=line,
        column=column,
        end_line=line,
        end_column=column + 1,
        severity=severity,
        message=message,
        code=code,
        source="test",
    )
