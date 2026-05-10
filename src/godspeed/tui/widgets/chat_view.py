"""Chat view widget — RichLog-based scrollable conversation display."""

from __future__ import annotations

import difflib
import json
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from textual.widgets import RichLog


class ChatView(RichLog):
    """Scrollable conversation history using RichLog.

    Handles tool call/results, thinking, markdown rendering, and streaming.
    Provides a per-turn markdown buffer for proper assistant response display.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            **kwargs,
        )
        self._markdown_buffer: list[str] = []
        self._in_turn = False

    def write(
        self,
        content: str = "",
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
    ) -> Any:
        """Override write to allow blank-line calls (no-arg write() for spacing)."""
        return super().write(
            content, width=width, expand=expand, shrink=shrink, scroll_end=scroll_end
        )

    # -- Turn lifecycle -------------------------------------------------------

    def start_turn(self) -> None:
        self._markdown_buffer = []
        self._in_turn = True

    def end_turn(self) -> None:
        self._in_turn = False

    # -- Streaming & markdown -------------------------------------------------

    def write_chunk(self, chunk: str) -> None:
        """Buffer and write a streaming text chunk."""
        self._markdown_buffer.append(chunk)
        self.write(chunk)

    def write_markdown(self, text: str) -> None:
        """Write rendered markdown (used at end of assistant response)."""
        if not text.strip():
            return
        self.write(Markdown(text))

    # -- Tool calls -----------------------------------------------------------

    def write_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Render a tool call — compact for simple tools, expanded for complex."""
        from godspeed.tui.theme import BOLD_PRIMARY, DIM, NEUTRAL, styled

        marker = styled(">", NEUTRAL)
        tool = styled(name, BOLD_PRIMARY)

        # Compact: single-arg tools
        if name in ("file_read", "glob_search", "repo_map") and args.get("file_path"):
            self.write(f"  {marker} {tool}  {args['file_path']}")
            return

        if name == "grep_search" and args.get("pattern"):
            path = args.get("path", "")
            suffix = f"  {path}" if path else ""
            self.write(f"  {marker} {tool}  {styled(args['pattern'], DIM)}{suffix}")
            return

        if name == "git" and args.get("action"):
            action = args["action"]
            extra = args.get("message", args.get("branch", ""))
            suffix = f"  {extra}" if extra else ""
            self.write(f"  {marker} {tool}  {action}{suffix}")
            return

        # Shell with gutter
        if name == "shell" and args.get("command"):
            self.write(f"  {marker} {tool}")
            self._gutter_text(f"$ {args['command']}")
            return

        # File edit: compact diff
        if name == "file_edit" and args.get("file_path"):
            self.write(f"  {marker} {tool}  {args['file_path']}")
            if args.get("old_string") and args.get("new_string"):
                old_lines = args["old_string"].splitlines()
                new_lines = args["new_string"].splitlines()
                diff_out = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))
                diff_lines = diff_out[2:] if len(diff_out) > 2 else diff_out
                if diff_lines:
                    self._gutter_text("\n".join(diff_lines[:15]))
            return

        # File write: path + line count
        if name == "file_write" and args.get("file_path"):
            content = args.get("content", "")
            line_count = len(content.splitlines())
            count_label = styled(f"({line_count} lines)", DIM)
            self.write(f"  {marker} {tool}  {args['file_path']}  {count_label}")
            return

        # Default: JSON
        try:
            args_text = json.dumps(args, indent=2, default=str)
        except (TypeError, ValueError):
            args_text = str(args)
        self.write(f"  {marker} {tool}")
        self._gutter_text(args_text)

    # -- Tool results ---------------------------------------------------------

    _RESULT_MAX_LINES = 10
    _RESULT_MAX_CHARS = 2000

    def write_tool_result(
        self,
        name: str,
        result: str,
        is_error: bool = False,
        duration_ms: float = 0.0,
    ) -> None:
        """Render a tool result — compact for success, expanded for errors."""
        from godspeed.tui.theme import (
            BOLD_ERROR,
            DIM,
            ERROR,
            MARKER_ERROR,
            MARKER_SUCCESS,
            NEUTRAL,
            SUCCESS,
            styled,
        )

        timing = ""
        if duration_ms > 0:
            if duration_ms < 1000:
                timing = styled(f" ({duration_ms:.0f}ms)", NEUTRAL)
            else:
                timing = styled(f" ({duration_ms / 1000:.1f}s)", NEUTRAL)

        if is_error:
            marker = styled(MARKER_ERROR, ERROR)
            tool = styled(f"{name}", BOLD_ERROR)
            display = result
            if len(result) > self._RESULT_MAX_CHARS:
                remaining = len(result) - self._RESULT_MAX_CHARS
                display = result[: self._RESULT_MAX_CHARS] + f"\n... ({remaining} more chars)"
            lines = display.splitlines()
            if len(lines) <= 3:
                self.write(f"  {marker} {tool}{timing}  {lines[0] if lines else ''}")
                for line in lines[1:]:
                    self.write(f"    {line}")
            else:
                self.write(f"  {marker} {tool}{timing}")
                for line in lines[:20]:
                    self.write(f"    {styled(line, DIM)}")
                if len(lines) > 20:
                    self.write(f"    {styled(f'... ({len(lines) - 20} more lines)', DIM)}")
        else:
            marker = styled(MARKER_SUCCESS, SUCCESS)
            tool = styled(name, NEUTRAL)
            lines = result.splitlines()
            line_count = len(lines)
            if not result.strip():
                self.write(f"  {marker} {tool}{timing}")
                return
            if line_count <= self._RESULT_MAX_LINES and len(result) <= 500:
                self.write(f"  {marker} {tool}{timing}")
                for line in lines:
                    self.write(f"    {styled(line, DIM)}")
            else:
                preview_lines = lines[:3]
                self.write(f"  {marker} {tool}{timing}  {styled(f'({line_count} lines)', DIM)}")
                for line in preview_lines:
                    self.write(f"    {styled(line, DIM)}")
                if line_count > 3:
                    self.write(f"    {styled(f'... ({line_count - 3} more lines)', DIM)}")

    # -- Thinking -------------------------------------------------------------

    def write_thinking(self, text: str) -> None:
        """Render thinking content in a dim panel."""
        if not text.strip():
            return
        from godspeed.tui.theme import DIM as DIM_STYLE
        from godspeed.tui.theme import NEUTRAL
        from godspeed.tui.theme import styled as s

        display_text = text[:2000]
        if len(text) > 2000:
            display_text += f"\n... ({len(text) - 2000} chars truncated)"
        panel = Panel(
            s(display_text, DIM_STYLE),
            title=s("Thinking", NEUTRAL),
            border_style=NEUTRAL,
            expand=False,
            padding=(0, 1),
        )
        self.write(panel)

    # -- Permission -----------------------------------------------------------

    def write_permission_denied(self, tool_name: str, reason: str) -> None:
        from godspeed.tui.theme import BOLD_ERROR, ERROR, MARKER_ERROR, styled

        marker = styled(MARKER_ERROR, ERROR)
        self.write(f"  {marker} {styled('Blocked:', BOLD_ERROR)} {tool_name} -- {reason}")

    # -- Status messages ------------------------------------------------------

    def write_info(self, message: str) -> None:
        from godspeed.tui.theme import DIM, MARKER_INFO, NEUTRAL, styled

        self.write(f"  {styled(MARKER_INFO, NEUTRAL)} {styled(message, DIM)}")

    def write_success(self, message: str) -> None:
        from godspeed.tui.theme import MARKER_SUCCESS, SUCCESS, styled

        self.write(f"  {styled(MARKER_SUCCESS, SUCCESS)} {message}")

    def write_warning(self, message: str) -> None:
        from godspeed.tui.theme import MARKER_WARNING, WARNING, styled

        self.write(f"  {styled(MARKER_WARNING, WARNING)} {message}")

    def write_error(self, message: str) -> None:
        from godspeed.tui.theme import BOLD_ERROR, ERROR, MARKER_ERROR, styled

        self.write(f"  {styled(MARKER_ERROR, ERROR)} {styled(f'Error: {message}', BOLD_ERROR)}")

    def write_status(self, text: str) -> None:
        from godspeed.tui.theme import DIM, styled

        self.write(f"  {styled(text, DIM)}")

    # -- Internal helpers -----------------------------------------------------

    def _gutter_text(self, text: str) -> None:
        from godspeed.tui.theme import GUTTER, GUTTER_STYLE, styled

        gutter = styled(GUTTER, GUTTER_STYLE)
        for line in text.splitlines():
            self.write(f"    {gutter} {line}")
