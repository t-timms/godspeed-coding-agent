"""TUI wiring for the plan-mode approval gate.

Builds the ``approval_prompt`` callable the ``PlanApprovalGate`` awaits when
plan mode is active. Mirrors ``_InteractiveDiffReviewer``: push a Textual
dialog and map the dismissal to an approve/reject decision.
"""

from __future__ import annotations

from typing import Any

from godspeed.security.plan_gate import APPROVE, REJECT


def tui_plan_approval_prompt(app: Any) -> Any:
    """Return an async ``(plan: str) -> str`` prompt bound to *app*.

    The returned callable pushes a ``PlanApprovalDialog`` and returns
    ``APPROVE`` when the human approves, ``REJECT`` otherwise. On any dialog
    failure it rejects closed (fail-safe: plan mode stays on).
    """

    async def _prompt(plan: str) -> str:
        from godspeed.tui.screens.plan_approval import PlanApprovalDialog

        try:
            answer = await app.push_screen(
                PlanApprovalDialog(plan),
                wait_for_dismiss=True,
            )
        except Exception:
            return REJECT
        return APPROVE if answer == "approve" else REJECT

    return _prompt
