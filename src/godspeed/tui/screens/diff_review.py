"""Diff review dialog — accept/reject proposed code changes."""

from __future__ import annotations

import difflib
from typing import Any, ClassVar

from textual.screen import Screen
from textual.widgets import Static

from godspeed.tui.theme import (
    BOLD_PRIMARY,
    BOLD_WARNING,
    DIM,
    ERROR,
    NEUTRAL,
    SUCCESS,
    WARNING,
    styled,
)


class DiffReviewDialog(Screen[str]):
    """Dialog to review and approve/reject a proposed code diff."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("y", "accept", "Yes"),
        ("n", "reject", "No"),
        ("a", "always", "Always"),
        ("escape", "reject", "Cancel"),
    ]

    def __init__(
        self,
        tool_name: str,
        path: str,
        before: str,
        after: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._path = path
        self._before = before
        self._after = after

    def compose(self: Any) -> Any:
        lines: list[str] = []
        lines.append(f"  {styled('Review proposed edit', BOLD_WARNING)}")
        lines.append("")
        lines.append(f"    {styled(self._tool_name, BOLD_PRIMARY)}  {self._path}")

        is_raw_diff = (not self._before and self._after.startswith("diff --git"))
        if is_raw_diff or self._after.startswith("---"):
            diff_text = self._after
        else:
            before_lines = self._before.splitlines()
            after_lines = self._after.splitlines()
            diff_lines_list = list(difflib.ndiff(before_lines, after_lines))
            added = sum(1 for line in diff_lines_list if line.startswith("+ "))
            removed = sum(1 for line in diff_lines_list if line.startswith("- "))
            lines.append(f"    {styled(f'+{added} -{removed} lines', DIM)}")
            diff_lines = list(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile="before",
                    tofile="after",
                    lineterm="",
                    n=3,
                )
            )
            diff_content = diff_lines[2:] if len(diff_lines) > 2 else diff_lines
            diff_text = "\n".join(diff_content[:80])
            if len(diff_content) > 80:
                diff_text += f"\n... ({len(diff_content) - 80} more lines)"

        # Render diff with color markup for Textual
        for line in diff_text.splitlines():
            if line.startswith("+"):
                lines.append(f"  [{SUCCESS}]{line}[/{SUCCESS}]")
            elif line.startswith("-"):
                lines.append(f"  [{ERROR}]{line}[/{ERROR}]")
            elif line.startswith("@@"):
                lines.append(f"  [{NEUTRAL}]{line}[/{NEUTRAL}]")
            else:
                lines.append(f"  {line}")

        lines.append("")
        lines.append(f"    {styled('Apply?', WARNING)} {styled('(y)es | (n)o | (a)lways', DIM)}")
        yield Static("\n".join(lines), id="diff-content")

    def action_accept(self) -> None:
        self.dismiss("accept")

    def action_reject(self) -> None:
        self.dismiss("reject")

    def action_always(self) -> None:
        self.dismiss("always")
