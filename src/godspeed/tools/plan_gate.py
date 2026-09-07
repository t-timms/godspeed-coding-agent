"""ExitPlanMode tool — presents the plan and requests approval to leave plan mode."""

from __future__ import annotations

from typing import Any

from godspeed.security.plan_gate import (
    PLAN_GATE_TOOL_NAME,
    PlanApprovalGate,
)
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult


class ExitPlanModeTool(Tool):
    """Tool that presents the plan and requests approval to exit plan mode.

    Declared READ_ONLY so it is exempt from the plan-mode block that denies
    non-READ_ONLY tools. When plan mode is active, calling it raises a
    structured approval request via ``PlanApprovalGate`` and awaits the
    human's decision. When plan mode is off, it returns a "not in plan mode"
    message rather than erroring.
    """

    def __init__(self, gate: PlanApprovalGate) -> None:
        self._gate = gate

    @property
    def name(self) -> str:
        return PLAN_GATE_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Present the current plan and request approval to exit plan mode. "
            "Call this when you have finished exploring and are ready to begin "
            "implementation. The user will approve or reject the plan."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "The plan to present for approval.",
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        plan = str(arguments.get("plan", "")).strip()
        message = await self._gate.request_approval(plan)
        if self._gate.approved:
            return ToolResult.ok(message)
        guidance = self._gate.guidance
        if guidance:
            return ToolResult.ok(f"{message}\n\nGuidance: {guidance}")
        return ToolResult.ok(message)
