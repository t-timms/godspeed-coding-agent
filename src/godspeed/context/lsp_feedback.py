"""LSP diagnostics passive feedback — inject post-edit diagnostics as context.

Runs language server diagnostics on files after edit operations and
injects relevant warnings/errors as passive feedback into the conversation.
The agent sees diagnostic context without needing to explicitly run a
linter/verifier — reducing tool calls and improving self-correction.

This follows the openJiuwen pattern of passive middleware feedback:
the LSP runs as infrastructure, not as a tool call.

Design:
- Runs diagnostics via subprocess (pyright/basedpyright/ruff LSP)
- Caches results per (file_path, content_hash) to avoid redundant runs
- Injects diagnostics as role: "tool" messages with diagnostic metadata
- Integrates with the existing verify tool (ruff check) but at a
  different layer — LSP catches type errors and subtle issues that
  ruff's AST-based checks miss
- Gracefully degrades: no-op if no LSP server configured

References:
- openJiuwen Context Management: tools/middleware carry gains
- Claude Code: integrated LSP diagnostics in context
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Diagnostic severity levels (matching LSP spec)
SEVERITY_ERROR = 1
SEVERITY_WARNING = 2
SEVERITY_INFORMATION = 3
SEVERITY_HINT = 4

_SEVERITY_NAMES: dict[int, str] = {
    SEVERITY_ERROR: "error",
    SEVERITY_WARNING: "warning",
    SEVERITY_INFORMATION: "info",
    SEVERITY_HINT: "hint",
}

# Maximum diagnostics per file to avoid context overflow
_MAX_DIAGNOSTICS_PER_FILE: int = 15

# Maximum total diagnostics across all files
_MAX_TOTAL_DIAGNOSTICS: int = 30

# Cache TTL for diagnostics (seconds)
_DIAG_CACHE_TTL: float = 30.0

# Maximum number of cached diagnostic entries (LRU eviction)
_MAX_CACHE_ENTRIES: int = 256

# File extensions that benefit from LSP diagnostics
_LSP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
    }
)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A single diagnostic from an LSP server."""

    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int
    severity: int
    message: str
    code: str | None
    source: str

    @property
    def severity_name(self) -> str:
        """Human-readable severity name."""
        return _SEVERITY_NAMES.get(self.severity, "unknown")

    @property
    def is_error(self) -> bool:
        """Whether this diagnostic is an error."""
        return self.severity == SEVERITY_ERROR

    @property
    def is_warning(self) -> bool:
        """Whether this diagnostic is a warning."""
        return self.severity == SEVERITY_WARNING


@dataclass(frozen=True, slots=True)
class FileDiagnostics:
    """Diagnostics for a single file."""

    file_path: str
    diagnostics: list[Diagnostic]
    timestamp: float

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return sum(1 for d in self.diagnostics if d.is_error)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return sum(1 for d in self.diagnostics if d.is_warning)


@dataclass
class _CachedDiagnostics:
    """Cached diagnostics with timestamp for TTL."""

    result: FileDiagnostics
    created_at: float


class LSPFeedbackProvider:
    """Provides LSP diagnostics as passive feedback after file edits.

    Runs diagnostics asynchronously and caches results. Integrates with
    the agent loop to inject diagnostic context without explicit tool calls.

    Supports:
    - pyright / basedpyright (type checking)
    - ruff LSP (linting)
    - Custom LSP servers (any JSON-RPC based server)

    Gracefully degrades: returns empty diagnostics when no LSP server
    is configured or available.

    Args:
        project_dir: Project root for LSP server configuration.
        enabled: Whether LSP feedback is enabled.
        max_diagnostics: Maximum diagnostics to return per file.
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        enabled: bool = True,
        max_diagnostics: int = _MAX_DIAGNOSTICS_PER_FILE,
    ) -> None:
        self._project_dir = project_dir or Path.cwd()
        self._enabled = enabled
        self._max_diagnostics = max_diagnostics
        self._cache: OrderedDict[str, _CachedDiagnostics] = OrderedDict()
        self._max_cache_entries = _MAX_CACHE_ENTRIES
        self._available: bool | None = None
        self._lsp_command: list[str] | None = None

    @property
    def enabled(self) -> bool:
        """Whether LSP feedback is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable LSP feedback."""
        self._enabled = value

    def _is_available(self) -> bool:
        """Check if an LSP server is available for diagnostics."""
        if self._available is not None:
            return self._available

        if not self._enabled:
            self._available = False
            return False

        # Try to find a working LSP command
        self._lsp_command = self._find_lsp_command()
        self._available = self._lsp_command is not None

        if self._available:
            logger.info(
                "LSP feedback available command=%s",
                " ".join(self._lsp_command),
            )
        else:
            logger.debug("No LSP server available for passive feedback")

        return self._available

    def _find_lsp_command(self) -> list[str] | None:
        """Find an available LSP command for diagnostics.

        Priority: pyright → basedpyright → ruff check (as fallback).
        Returns the command list or None if nothing available.
        """
        # Check for pyright (most common LSP for Python)
        for cmd in ("pyright", "basedpyright"):
            if self._command_exists(cmd):
                return [cmd, "--outputjson", "--stdout"]

        # Fallback: ruff check for basic lint diagnostics
        if self._command_exists("ruff"):
            return ["ruff", "check", "--output-format=json", "--stdin-filename"]

        return None

    @staticmethod
    def _command_exists(cmd: str) -> bool:
        """Check if a command exists on PATH."""
        try:
            import shutil

            return shutil.which(cmd) is not None
        except Exception:
            return False

    def _content_hash(self, file_path: Path, content: str | None = None) -> str:
        """Compute a content hash for cache invalidation."""
        if content is not None:
            return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        try:
            data = file_path.read_bytes()
            return hashlib.sha256(data).hexdigest()[:16]
        except OSError:
            return "unreadable"

    def _get_cached(self, file_path: Path, content_hash: str) -> FileDiagnostics | None:
        """Get cached diagnostics if still valid."""
        cache_key = f"{file_path}:{content_hash}"
        cached = self._cache.get(cache_key)
        if cached is None:
            return None

        if time.time() - cached.created_at > _DIAG_CACHE_TTL:
            del self._cache[cache_key]
            return None
        self._cache.move_to_end(cache_key)
        return cached.result

    def _set_cached(self, file_path: Path, content_hash: str, result: FileDiagnostics) -> None:
        """Cache diagnostics with timestamp, evicting LRU entries when full."""
        cache_key = f"{file_path}:{content_hash}"
        self._cache[cache_key] = _CachedDiagnostics(
            result=result,
            created_at=time.time(),
        )
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)

    # ── Diagnostics collection ─────────────────────────────────────────

    async def get_diagnostics(
        self,
        file_path: Path,
        content: str | None = None,
    ) -> FileDiagnostics:
        """Get LSP diagnostics for a file.

        Uses caching to avoid redundant LSP runs. If content is provided,
        it's used for hash computation (avoids re-reading the file).

        Args:
            file_path: Path to the file to diagnose.
            content: Optional file content (for hash computation).

        Returns:
            FileDiagnostics with all diagnostics for the file.
        """
        if not self._is_available():
            return FileDiagnostics(
                file_path=str(file_path),
                diagnostics=[],
                timestamp=0.0,
            )

        content_hash = self._content_hash(file_path, content)
        cached = self._get_cached(file_path, content_hash)
        if cached is not None:
            return cached

        # Run diagnostics
        try:
            diagnostics = await self._run_diagnostics(file_path)
        except Exception as exc:
            logger.debug("LSP diagnostics failed for %s: %s", file_path, exc)
            diagnostics = []

        result = FileDiagnostics(
            file_path=str(file_path),
            diagnostics=diagnostics[: self._max_diagnostics],
            timestamp=time.time(),
        )

        self._set_cached(file_path, content_hash, result)
        return result

    async def get_diagnostics_batch(
        self,
        file_paths: list[Path],
    ) -> list[FileDiagnostics]:
        """Get diagnostics for multiple files concurrently.

        Args:
            file_paths: List of file paths to diagnose.

        Returns:
            List of FileDiagnostics, one per file.
        """
        tasks = [self.get_diagnostics(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        diagnostics_list: list[FileDiagnostics] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(
                    "LSP diagnostics failed for %s: %s",
                    file_paths[i],
                    result,
                )
                diagnostics_list.append(
                    FileDiagnostics(
                        file_path=str(file_paths[i]),
                        diagnostics=[],
                        timestamp=0.0,
                    )
                )
            else:
                diagnostics_list.append(result)

        return diagnostics_list

    # ── Diagnostics execution ──────────────────────────────────────────

    async def _run_diagnostics(self, file_path: Path) -> list[Diagnostic]:
        """Run LSP diagnostics on a file.

        Dispatches to the appropriate handler based on the LSP command.
        """
        if self._lsp_command is None:
            return []

        cmd = self._lsp_command

        if cmd[0] in ("pyright", "basedpyright"):
            return await self._run_pyright(file_path, cmd)
        if cmd[0] == "ruff":
            return await self._run_ruff_check(file_path)

        return []

    async def _run_pyright(
        self,
        file_path: Path,
        cmd: list[str],
    ) -> list[Diagnostic]:
        """Run pyright/basedpyright and parse JSON output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                str(file_path),
                cwd=str(self._project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=15.0,
            )
        except (TimeoutError, OSError) as exc:
            logger.debug("pyright timeout/error: %s", exc)
            return []

        if proc.returncode is None:
            return []

        try:
            data = json.loads(stdout.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

        diagnostics: list[Diagnostic] = []
        target = str(file_path)
        for diag in data.get("generalDiagnostics", []):
            file_diag_path = diag.get("file", "")
            # Only include diagnostics for the requested file (full-path match)
            if file_diag_path and not _paths_match(file_diag_path, target):
                continue

            range_info = diag.get("range", {})
            start = range_info.get("start", {})
            end = range_info.get("end", {})

            diagnostics.append(
                Diagnostic(
                    file_path=file_diag_path or str(file_path),
                    line=start.get("line", 0) + 1,  # LSP is 0-indexed
                    column=start.get("character", 0),
                    end_line=end.get("line", 0) + 1,
                    end_column=end.get("character", 0),
                    severity=_pyright_severity(diag.get("severity", "error")),
                    message=diag.get("message", ""),
                    code=diag.get("rule", diag.get("diagnosticId", None)),
                    source="pyright",
                )
            )

        return diagnostics

    async def _run_ruff_check(self, file_path: Path) -> list[Diagnostic]:
        """Run ruff check and parse JSON output."""
        try:
            content = file_path.read_bytes()
            proc = await asyncio.create_subprocess_exec(
                "ruff",
                "check",
                "--output-format=json",
                "--stdin-filename",
                str(file_path),
                cwd=str(self._project_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(input=content),
                timeout=10.0,
            )
        except (TimeoutError, OSError) as exc:
            logger.debug("ruff check timeout/error: %s", exc)
            return []

        try:
            data = json.loads(stdout.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

        diagnostics: list[Diagnostic] = []
        for item in data:
            location = item.get("location", {})
            end_location = item.get("end_location", {})
            code = item.get("code")

            diagnostics.append(
                Diagnostic(
                    file_path=str(file_path),
                    line=location.get("row", 1),
                    column=location.get("column", 0),
                    end_line=end_location.get("row", 1),
                    end_column=end_location.get("column", 0),
                    severity=SEVERITY_WARNING,  # ruff doesn't distinguish
                    message=item.get("message", ""),
                    code=str(code) if code else None,
                    source="ruff",
                )
            )

        return diagnostics

    # ── Formatting for context injection ───────────────────────────────

    def format_for_prompt(
        self,
        diagnostics: list[FileDiagnostics],
        max_total: int = _MAX_TOTAL_DIAGNOSTICS,
    ) -> str:
        """Format diagnostics as passive feedback for the system prompt.

        Produces compact, actionable diagnostic context. Errors first,
        then warnings. Truncated at max_total to prevent context overflow.

        Args:
            diagnostics: List of FileDiagnostics to format.
            max_total: Maximum total diagnostics to include.

        Returns:
            Formatted string for system prompt injection, or "" if empty.
        """
        all_diags: list[Diagnostic] = []
        for fd in diagnostics:
            all_diags.extend(fd.diagnostics)

        if not all_diags:
            return ""

        # Sort: errors first, then warnings, then info
        all_diags.sort(key=lambda d: d.severity)

        # Truncate to max_total
        all_diags = all_diags[:max_total]

        lines: list[str] = ["LSP Diagnostics (post-edit passive feedback):"]
        for diag in all_diags:
            file_name = Path(diag.file_path).name
            code_str = f" [{diag.code}]" if diag.code else ""
            lines.append(
                f"  {diag.severity_name}: {file_name}:{diag.line}:{diag.column}"
                f"{code_str} {diag.message}"
            )

        return "\n".join(lines)

    def format_for_tool_result(
        self,
        file_diagnostics: FileDiagnostics,
    ) -> str:
        """Format diagnostics as a tool result message content.

        Used when injecting diagnostics directly into conversation as
        a synthetic tool result (role: "tool").

        Args:
            file_diagnostics: Diagnostics for a single file.

        Returns:
            Formatted diagnostic string.
        """
        if not file_diagnostics.diagnostics:
            return f"No diagnostics for {Path(file_diagnostics.file_path).name}"

        parts: list[str] = [f"LSP diagnostics for {Path(file_diagnostics.file_path).name}:"]
        for diag in file_diagnostics.diagnostics:
            code_str = f" [{diag.code}]" if diag.code else ""
            parts.append(
                f"  {diag.severity_name}: L{diag.line}:C{diag.column}{code_str} {diag.message}"
            )

        parts.append(
            f"  ({file_diagnostics.error_count} errors, {file_diagnostics.warning_count} warnings)"
        )
        return "\n".join(parts)

    # ── Post-edit integration ──────────────────────────────────────────

    async def on_file_edited(
        self,
        file_path: Path,
        content: str | None = None,
    ) -> str | None:
        """Called after a file edit to collect and return diagnostic feedback.

        This is the primary integration point for the agent loop. After
        a file_edit or file_write tool call on a supported file, the agent
        loop calls this method. It returns diagnostic text to inject as
        a passive feedback message, or None if no diagnostics to report.

        Args:
            file_path: Path of the edited file.
            content: Optional new file content (for hash computation).

        Returns:
            Diagnostic feedback string, or None if no issues found.
        """
        if not self._is_available():
            return None

        if file_path.suffix.lower() not in _LSP_EXTENSIONS:
            return None

        file_diag = await self.get_diagnostics(file_path, content)
        if not file_diag.diagnostics:
            return None

        # Only inject if there are errors or significant warnings
        has_errors = file_diag.error_count > 0
        has_warnings = file_diag.warning_count >= 3

        if not has_errors and not has_warnings:
            return None

        return self.format_for_tool_result(file_diag)

    def clear_cache(self, file_path: Path | None = None) -> int:
        """Clear diagnostic cache.

        Args:
            file_path: If provided, clear only entries for this file.
                       If None, clear all cached diagnostics.

        Returns:
            Number of cache entries removed.
        """
        if file_path is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        file_str = str(file_path)
        to_remove = [k for k in self._cache if k.startswith(file_str)]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    @property
    def cache_size(self) -> int:
        """Number of cached diagnostic entries."""
        return len(self._cache)


# ── Helpers ───────────────────────────────────────────────────────────


def _pyright_severity(severity_str: str) -> int:
    """Convert pyright severity string to LSP severity integer."""
    mapping: dict[str, int] = {
        "error": SEVERITY_ERROR,
        "warning": SEVERITY_WARNING,
        "information": SEVERITY_INFORMATION,
        "hint": SEVERITY_HINT,
    }
    return mapping.get(severity_str.lower(), SEVERITY_WARNING)


def _paths_match(reported: str, target: str) -> bool:
    """Compare two file paths, normalizing separators and case on Windows."""
    norm_reported = Path(reported).resolve()
    norm_target = Path(target).resolve()
    if norm_reported == norm_target:
        return True
    return str(norm_reported).replace("\\", "/") == str(norm_target).replace("\\", "/")
