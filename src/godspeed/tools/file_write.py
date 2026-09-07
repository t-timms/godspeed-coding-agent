"""File write tool — create new files."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit for file writes


class FileWriteTool(Tool):
    """Create or overwrite a file with new content.

    Use file_edit for modifying existing files.
    Use file_write only for creating new files or complete rewrites.
    """

    produces_diff = True

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file and parent directories if needed. "
            "Overwrites existing content. Use file_edit for precise modifications.\n\n"
            "Example: file_write(file_path='src/utils.py', content='def helper():\\n    pass')\n"
            "Example: file_write(file_path='config.json', content='{\"debug\": true}')"
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
                    "description": "Path to the file to write (relative to project root)",
                    "examples": ["src/utils.py", "config.json", "tests/test_new.py"],
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")
        if not isinstance(content, str):
            return ToolResult.failure(f"content must be a string, got {type(content).__name__}")

        # Check file size limit
        if len(content.encode("utf-8")) > MAX_FILE_SIZE:
            return ToolResult.failure(
                f"File content exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB"
            )

        try:
            resolved = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        # Diff review gate: when a diff_reviewer is attached to the context
        # (typically the TUI), let the human accept or reject this specific
        # write before it hits disk. Read current content for the diff if the
        # file already exists; for new files, the "before" is empty string.
        if context.diff_reviewer is not None:
            try:
                before = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
            except UnicodeDecodeError:
                before = "<binary>"
            decision = await context.diff_reviewer.review(
                tool_name=self.name,
                path=file_path_str,
                before=before,
                after=content,
            )
            if decision != "accept":
                logger.info(
                    "Write rejected by diff_reviewer: path=%s decision=%s",
                    file_path_str,
                    decision,
                )
                return ToolResult.failure(
                    f"Write rejected by reviewer for {file_path_str} — no changes written."
                )

        try:
            # Before-edit checkpoint: snapshot the on-disk original (existing
            # files only) so the overwrite is reversible via
            # edit_checkpoints.restore_latest.
            if resolved.exists():
                from godspeed.tools.edit_checkpoints import snapshot_file

                snapshot_file(resolved, context.cwd, context.session_id)

            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to temp file, then replace atomically
            fd, tmp_path = tempfile.mkstemp(suffix=resolved.suffix, dir=str(resolved.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, str(resolved))
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

            logger.info("Wrote file path=%s size=%d", resolved, len(content))
            return ToolResult.success(f"Wrote {len(content)} bytes to {file_path_str}")

        except OSError as exc:
            return ToolResult.failure(f"Failed to write {file_path_str}: {exc}")
