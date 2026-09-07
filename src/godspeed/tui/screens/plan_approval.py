"""Plan approval dialog — approve/reject the agent's plan to exit plan mode."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.screen import Screen
from textual.widgets import Static

from godspeed.tui.theme import (
    BOLD_PRIMARY,
    BOLD_WARNING,
    DIM,
    WARNING,
    styled,
)


class PlanApprovalDialog(Screen[str]):
    """Dialog to approve or reject the agent's plan for exiting plan mode."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("y", "approve", "Yes"),
        ("n", "reject", "No"),
        ("escape", "reject", "Cancel"),
    ]

    def __init__(self, plan: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._plan = plan

    def compose(self: Any) -> Any:
        lines: list[str] = []
        lines.append(f"  {styled('Plan approval requested', BOLD_WARNING)}")
        lines.append("")
        lines.append(f"    {styled('exit_plan_mode', BOLD_PRIMARY)}")
        lines.append("")
        plan_lines = self._plan.splitlines() or ["(no plan text provided)"]
        for line in plan_lines[:60]:
            lines.append(f"    {line}")
        if len(plan_lines) > 60:
            lines.append(f"    ... ({len(plan_lines) - 60} more lines)")
        lines.append("")
        lines.append(
            f"    {styled('Approve plan and exit plan mode?', WARNING)} "
            f"{styled('(y)es | (n)o', DIM)}"
        )
        yield Static("\n".join(lines), id="plan-content")

    def action_approve(self) -> None:
        self.dismiss("approve")

    def action_reject(self) -> None:
        self.dismiss("reject")
