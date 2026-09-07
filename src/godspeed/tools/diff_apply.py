"""Unified diff apply tool — parse and apply unified diffs to files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)


@dataclass
class Hunk:
    """A single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """All hunks targeting a single file."""

    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new_file: bool = False


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff string into structured FileDiff objects.

    Raises:
        ValueError: On malformed diff content.
    """
    diff_text = diff_text.replace("\r\n", "\n").replace("\r", "\n")

    file_diffs: list[FileDiff] = []
    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        if lines[i].startswith("--- "):
            old_header = lines[i]
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                msg = f"Expected '+++ ' after '--- ' at line {i + 1}"
                raise ValueError(msg)
            new_header = lines[i + 1]
            i += 2

            old_path = _extract_path(old_header)
            new_path = _extract_path(new_header)
            is_new = old_path == "/dev/null"

            file_diff = FileDiff(
                old_path=old_path,
                new_path=new_path,
                is_new_file=is_new,
            )

            # Parse hunks for this file
            while i < len(lines) and lines[i].startswith("@@"):
                hunk, i = _parse_hunk(lines, i)
                file_diff.hunks.append(hunk)

            file_diffs.append(file_diff)
        else:
            i += 1

    return file_diffs


def _extract_path(header: str) -> str:
    """Extract file path from --- or +++ header line.

    Handles formats like:
        --- a/path/to/file
        +++ b/path/to/file
        --- /dev/null
    """
    rest = header[4:].strip()
    if rest == "/dev/null":
        return rest
    if rest.startswith("a/") or rest.startswith("b/"):
        return rest[2:]
    return rest


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunk(lines: list[str], start: int) -> tuple[Hunk, int]:
    """Parse a single hunk starting at the @@ line.

    Returns the Hunk and the next line index after the hunk.
    """
    match = _HUNK_RE.match(lines[start])
    if not match:
        msg = f"Malformed hunk header at line {start + 1}: {lines[start]}"
        raise ValueError(msg)

    old_start = int(match.group(1))
    old_count = int(match.group(2)) if match.group(2) is not None else 1
    new_start = int(match.group(3))
    new_count = int(match.group(4)) if match.group(4) is not None else 1

    hunk = Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
    )

    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@") or line.startswith("--- "):
            break
        if line.startswith("+") or line.startswith("-") or line.startswith(" "):
            hunk.lines.append(line)
            i += 1
        elif line == "":
            # Bare empty line — could be context with stripped trailing space.
            # Treat as context if the next line continues the hunk.
            if i + 1 < len(lines) and (
                lines[i + 1].startswith("+")
                or lines[i + 1].startswith("-")
                or lines[i + 1].startswith(" ")
            ):
                hunk.lines.append(" ")
                i += 1
            else:
                break
        else:
            break

    return hunk, i


def _apply_hunk_to_lines(
    file_lines: list[str],
    hunk: Hunk,
    fuzzy_range: int = 3,
) -> tuple[list[str], bool]:
    """Apply a single hunk to file lines with fuzzy matching.

    Tries exact position first, then searches +/-fuzzy_range lines.

    Returns:
        Tuple of (new_lines, used_fuzzy). Raises ValueError on failure.
    """
    # Build expected old block (context + removals)
    old_block: list[str] = []
    for ln in hunk.lines:
        if ln.startswith(" ") or ln.startswith("-"):
            old_block.append(ln[1:])

    # Build new block (context + additions)
    new_block: list[str] = []
    for ln in hunk.lines:
        if ln.startswith(" ") or ln.startswith("+"):
            new_block.append(ln[1:])

    # 1-indexed to 0-indexed
    target_line = hunk.old_start - 1

    # Try exact position, then fuzzy offsets
    offsets = [0] + [d for delta in range(1, fuzzy_range + 1) for d in (-delta, delta)]

    for offset in offsets:
        pos = target_line + offset
        if pos < 0 or pos + len(old_block) > len(file_lines):
            continue

        candidate = file_lines[pos : pos + len(old_block)]
        if candidate == old_block:
            new_lines = file_lines[:pos] + new_block + file_lines[pos + len(old_block) :]
            return new_lines, offset != 0

    expected_preview = "\n".join(old_block[:5])
    actual_start = max(0, target_line)
    actual_end = min(len(file_lines), target_line + len(old_block))
    actual_preview = "\n".join(file_lines[actual_start:actual_end][:5])
    msg = (
        f"Hunk at line {hunk.old_start} failed to match.\n"
        f"Expected:\n{expected_preview}\n"
        f"Got:\n{actual_preview}"
    )
    raise ValueError(msg)


def apply_file_diff(
    file_diff: FileDiff,
    cwd: Path,
    *,
    dry_run: bool = False,
    fuzzy_range: int = 3,
) -> tuple[int, int]:
    """Apply all hunks for a single file.

    Args:
        file_diff: Parsed diff for one file.
        cwd: Working directory for path resolution.
        dry_run: If True, validate without writing.
        fuzzy_range: Max lines to search for fuzzy context match.

    Returns:
        Tuple of (hunks_applied, fuzzy_count).

    Raises:
        ValueError: If path resolution or hunk application fails.
    """
    target_path_str = file_diff.new_path

    resolved = resolve_tool_path(target_path_str, cwd)

    if file_diff.is_new_file:
        new_content_lines: list[str] = []
        for hunk in file_diff.hunks:
            for ln in hunk.lines:
                if ln.startswith("+"):
                    new_content_lines.append(ln[1:])
        if not dry_run:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(
                "\n".join(new_content_lines) + "\n",
                encoding="utf-8",
            )
        logger.info(
            "diff_apply action=%s path=%s hunks=%d",
            "dry_run_new" if dry_run else "created",
            target_path_str,
            len(file_diff.hunks),
        )
        return len(file_diff.hunks), 0

    if not resolved.exists():
        msg = f"File not found: {target_path_str}"
        raise ValueError(msg)

    content = resolved.read_text(encoding="utf-8")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    file_lines = content.split("\n")

    trailing_newline = content.endswith("\n")
    if trailing_newline and file_lines and file_lines[-1] == "":
        file_lines = file_lines[:-1]

    fuzzy_count = 0
    # Apply hunks in REVERSE order to preserve line numbers
    for hunk in reversed(file_diff.hunks):
        file_lines, used_fuzzy = _apply_hunk_to_lines(file_lines, hunk, fuzzy_range)
        if used_fuzzy:
            fuzzy_count += 1

    if not dry_run:
        out = "\n".join(file_lines)
        if trailing_newline:
            out += "\n"
        resolved.write_text(out, encoding="utf-8")

    logger.info(
        "diff_apply action=%s path=%s hunks=%d fuzzy=%d",
        "dry_run" if dry_run else "applied",
        target_path_str,
        len(file_diff.hunks),
        fuzzy_count,
    )
    return len(file_diff.hunks), fuzzy_count


class DiffApplyTool(Tool):
    """Apply a unified diff to one or more files.

    Parses standard unified diff format and applies hunks in reverse order
    to preserve line numbers. Supports fuzzy context matching (+-3 lines)
    when exact position doesn't match.
    """

    produces_diff = True

    @property
    def name(self) -> str:
        return "diff_apply"

    @property
    def description(self) -> str:
        return (
            "Apply a unified diff to files. Supports multi-file diffs, "
            "fuzzy context matching, dry-run validation, and new file creation."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": (
                        "Unified diff text to apply. Must include --- a/file and "
                        "+++ b/file headers and @@ hunk headers."
                    ),
                },
                "dry_run": {
                    "type": "boolean",
                    "description": ("If true, validate the diff can be applied without writing."),
                    "default": False,
                },
            },
            "required": ["diff"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        diff_text = arguments.get("diff", "")
        dry_run = bool(arguments.get("dry_run", False))

        if not isinstance(diff_text, str) or not diff_text.strip():
            return ToolResult.failure("diff must be a non-empty string")

        try:
            file_diffs = parse_unified_diff(diff_text)
        except ValueError as exc:
            return ToolResult.failure(f"Failed to parse diff: {exc}")

        if not file_diffs:
            return ToolResult.failure("No file diffs found in the provided diff content")

        # Diff review gate: when a reviewer is attached (typically the TUI)
        # and this is not a dry_run, let the human accept or reject the whole
        # multi-file diff before anything is written. One review call covers
        # the whole operation — diff_apply is treated as atomic from the
        # user's perspective. Dry runs skip the gate since nothing is written.
        if context.diff_reviewer is not None and not dry_run:
            path_summary = ", ".join(fd.new_path for fd in file_diffs[:5])
            if len(file_diffs) > 5:
                path_summary += f", ... ({len(file_diffs) - 5} more)"
            decision = await context.diff_reviewer.review(
                tool_name=self.name,
                path=path_summary,
                before="",  # raw diff IS the before/after delta — shown verbatim
                after=diff_text,
            )
            if decision != "accept":
                logger.info(
                    "diff_apply rejected by diff_reviewer: files=%d decision=%s",
                    len(file_diffs),
                    decision,
                )
                return ToolResult.failure(
                    f"Diff rejected by reviewer ({len(file_diffs)} files) — no changes written."
                )

        total_hunks = 0
        total_fuzzy = 0
        files_modified = 0

        for file_diff in file_diffs:
            try:
                if not dry_run:
                    # Before-edit checkpoint per target file (new files have
                    # nothing to snapshot) — keeps multi-file diffs reversible.
                    from godspeed.tools.edit_checkpoints import snapshot_file

                    snapshot_file(context.cwd / file_diff.new_path, context.cwd, context.session_id)
                hunks_applied, fuzzy_count = apply_file_diff(
                    file_diff, context.cwd, dry_run=dry_run
                )
            except ValueError as exc:
                return ToolResult.failure(str(exc))

            total_hunks += hunks_applied
            total_fuzzy += fuzzy_count
            files_modified += 1

        if dry_run:
            summary = f"Dry run: {total_hunks} hunks to {files_modified} files would apply"
        else:
            summary = f"Applied {total_hunks} hunks to {files_modified} files"

        if total_fuzzy > 0:
            summary += f" ({total_fuzzy} with fuzzy matching)"

        return ToolResult.success(summary)
