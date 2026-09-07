"""File edit tool — search/replace with fuzzy matching fallback."""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import tempfile
from difflib import SequenceMatcher
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

# Minimum similarity ratio for fuzzy matching
FUZZY_THRESHOLD = 0.8

# Extensions for which we run a post-edit syntax gate. If the edit changes
# the file from parseable to unparseable, we revert and surface the error —
# protecting against the "looks mostly right but actually broken" failure
# mode observed with multi-line edits that drop indentation.
_SYNTAX_GATED_EXTS = frozenset({".py", ".pyi"})


def _syntax_check(path_ext: str, before_content: str, after_content: str) -> str | None:
    """Return an error message if the edit broke syntax that previously parsed.

    Currently covers Python (.py, .pyi) via ast.parse and JSON via json.loads.
    Returns None if the edit is safe — either the file type is not gated, or
    the file was already broken before the edit (we don't blame the edit for
    pre-existing breakage), or the new content still parses.
    """
    ext = path_ext.lower()

    if ext not in _SYNTAX_GATED_EXTS and ext != ".json":
        return None

    is_python = ext in _SYNTAX_GATED_EXTS

    def _parse(source: str) -> None:
        if is_python:
            ast.parse(source)
        else:
            json.loads(source)

    # Skip the gate if the pre-edit content already didn't parse — the edit
    # isn't responsible for pre-existing breakage, and a blocked revert would
    # prevent the agent from fixing the underlying issue.
    try:
        _parse(before_content)
    except (SyntaxError, ValueError):
        return None

    try:
        _parse(after_content)
    except SyntaxError as exc:
        return (
            f"Post-edit syntax check failed — the edit would leave the file unparseable. "
            f"Line {exc.lineno}: {exc.msg}. "
            "This commonly means a multi-line replacement dropped indentation. "
            "Re-read the file and include enough surrounding context in old_string "
            "to preserve whitespace in new_string."
        )
    except ValueError as exc:
        return f"Post-edit JSON parse failed — the edit would produce invalid JSON: {exc}."

    return None


class FileEditTool(Tool):
    """Edit a file using search/replace.

    The most reliable edit format across all LLM models. Finds the exact
    old_string in the file and replaces it with new_string. Falls back to
    fuzzy matching (difflib.SequenceMatcher) when exact match fails due
    to whitespace drift.
    """

    produces_diff = True

    @property
    def name(self) -> str:
        return "file_edit"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_string with new_string. "
            "The old_string must be unique in the file (include surrounding context "
            "to ensure uniqueness). Read the file first to get exact content. "
            "Falls back to fuzzy matching if whitespace drifts.\n\n"
            "Example: file_edit(file_path='app.py', "
            "old_string='def hello():\\n    pass', "
            "new_string='def hello():\\n    return \"world\"')\n"
            "Example: file_edit(file_path='config.yaml', "
            "old_string='debug: false', new_string='debug: true')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative to project root)",
                    "examples": ["src/app.py", "config.yaml"],
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace (must be unique in file)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        replace_all = arguments.get("replace_all", False)

        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")
        if not isinstance(old_string, str) or not old_string:
            return ToolResult.failure("old_string must be a non-empty string")
        if not isinstance(new_string, str):
            return ToolResult.failure(
                f"new_string must be a string, got {type(new_string).__name__}"
            )
        if old_string == new_string:
            return ToolResult.failure("old_string and new_string must be different")

        try:
            resolved = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not resolved.exists():
            return ToolResult.failure(f"File not found: {file_path_str}")

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(f"Cannot edit binary file: {file_path_str}")

        # Normalize line endings to LF for consistent matching
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Try exact match first
        if old_string in content:
            count = content.count(old_string)
            if count > 1 and not replace_all:
                return ToolResult.failure(
                    f"old_string matches {count} locations in {file_path_str}. "
                    "Include more surrounding context to make it unique, "
                    "or set replace_all=true."
                )
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1
            match_info = f"[match=exact confidence=1.00 replacements={replacements}]"
        else:
            # Fuzzy matching fallback
            match_result = _fuzzy_find(content, old_string)
            if match_result is None:
                return ToolResult.failure(
                    f"old_string not found in {file_path_str}. "
                    "Read the file first to get the exact content."
                )
            start, end, ratio = match_result
            # Find the line number of the match
            match_line = content[:start].count("\n") + 1
            new_content = content[:start] + new_string + content[end:]
            replacements = 1
            match_info = f"[match=fuzzy confidence={ratio:.2f} line={match_line} replacements=1]"
            logger.info(
                "Fuzzy match used ratio=%.2f line=%d path=%s",
                ratio,
                match_line,
                file_path_str,
            )

        # Post-edit syntax gate: for .py / .pyi / .json, refuse edits that
        # would turn a parseable file into an unparseable one. Catches the
        # "multi-line replace drops indentation" failure mode.
        syntax_err = _syntax_check(resolved.suffix, content, new_content)
        if syntax_err is not None:
            logger.warning(
                "Rejected edit on %s — post-edit syntax gate failed: %s",
                file_path_str,
                syntax_err,
            )
            return ToolResult.failure(syntax_err)

        # Diff review gate: when a diff_reviewer is attached to the context
        # (typically the TUI), let the human accept or reject this specific
        # change before it hits disk. Headless / CI: diff_reviewer is None →
        # auto-accept, preserving backward-compatible behavior.
        if context.diff_reviewer is not None:
            decision = await context.diff_reviewer.review(
                tool_name=self.name,
                path=file_path_str,
                before=content,
                after=new_content,
            )
            if decision != "accept":
                logger.info(
                    "Edit rejected by diff_reviewer: path=%s decision=%s",
                    file_path_str,
                    decision,
                )
                return ToolResult.failure(
                    f"Edit rejected by reviewer for {file_path_str} — no changes written."
                )

        # Before-edit checkpoint: snapshot the on-disk original so the edit
        # is reversible via edit_checkpoints.restore_latest.
        from godspeed.tools.edit_checkpoints import snapshot_file

        snapshot_file(resolved, context.cwd, context.session_id)

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(suffix=resolved.suffix, dir=str(resolved.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.replace(tmp_path, str(resolved))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        logger.info(
            "Edited file path=%s replacements=%d",
            file_path_str,
            replacements,
        )
        return ToolResult.success(
            f"Replaced {replacements} occurrence(s) in {file_path_str} {match_info}"
        )


def _fuzzy_find(content: str, search: str) -> tuple[int, int, float] | None:
    """Find the best fuzzy match for search in content.

    Uses a sliding window approach with SequenceMatcher.
    Returns (start, end, ratio) of the best match, or None if below threshold.
    """
    # Normalize line endings — caller should also normalize, but be defensive
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    search = search.replace("\r\n", "\n").replace("\r", "\n")

    search_lines = search.splitlines()
    content_lines = content.splitlines()

    if not search_lines or not content_lines:
        return None

    window_size = len(search_lines)
    best_ratio = 0.0
    best_start_line = 0

    for i in range(len(content_lines) - window_size + 1):
        window = content_lines[i : i + window_size]
        window_text = "\n".join(window)
        ratio = SequenceMatcher(None, search, window_text).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start_line = i

    if best_ratio < FUZZY_THRESHOLD:
        return None

    # Convert line indices to character positions using join (handles edge cases)
    lines_before = content_lines[:best_start_line]
    start_pos = len("\n".join(lines_before)) + (1 if lines_before else 0)
    matched_lines = content_lines[best_start_line : best_start_line + window_size]
    end_pos = start_pos + len("\n".join(matched_lines))

    # Clamp to content length
    if end_pos > len(content):
        end_pos = len(content)

    return start_pos, end_pos, best_ratio
