"""Diff viewer widget — display and approve/reject code diffs interactively."""

from __future__ import annotations

import difflib

from godspeed.tui.output import _output as output_console
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    BOLD_WARNING,
    DIM,
    WARNING,
    styled,
)


def show_diff_review(
    tool_name: str,
    path: str,
    before: str,
    after: str,
) -> str:
    """Display a diff review prompt using Rich and return user decision.

    Returns 'accept' or 'reject'.
    """
    from rich.syntax import Syntax

    from godspeed.tui.theme import SYNTAX_THEME

    output_console.console.print()
    output_console.console.print(
        f"  {styled('Review proposed edit', BOLD_WARNING)}"
    )
    output_console.console.print()
    output_console.console.print(
        f"    {styled(tool_name, BOLD_PRIMARY)}  {path}"
    )

    if (not before and after.startswith("diff --git")) or after.startswith("---"):
        diff_text = after
    else:
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        diff_lines_list = list(difflib.ndiff(before_lines, after_lines))
        added = sum(1 for line in diff_lines_list if line.startswith("+ "))
        removed = sum(1 for line in diff_lines_list if line.startswith("- "))
        output_console.console.print(
            f"    {styled(f'+{added} -{removed} lines', DIM)}"
        )
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

    output_console.console.print(
        Syntax(diff_text, "diff", theme=SYNTAX_THEME, word_wrap=True)
    )
    output_console.console.print(
        f"    {styled('Apply?', WARNING)} {styled('(y)es | (n)o', DIM)}"
    )

    try:
        answer = output_console.console.input(
            f"[{BOLD_WARNING}]  > [/{BOLD_WARNING}]"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "reject"

    if answer in ("y", "yes", ""):
        return "accept"
    return "reject"
