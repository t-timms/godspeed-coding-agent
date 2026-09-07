"""Plan-mode approval gate — the awaitable interface for exiting plan mode.

The agent-loop side is authoritative: the ``ExitPlanModeTool`` calls
``PlanApprovalGate.request_approval`` which presents the plan and awaits the
human's decision. The TUI (or a test double) supplies an ``approval_prompt``
callable; when none is supplied the gate auto-approves (headless default).

Approval flips ``permission_engine.plan_mode`` off so implementation can
start. Rejection keeps plan mode on and records optional guidance that the
tool surfaces back to the model so it can revise the plan.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Tool name for the plan-mode exit gate.
PLAN_GATE_TOOL_NAME = "exit_plan_mode"

#: Decision strings returned by approval prompts.
APPROVE = "approve"
REJECT = "reject"

#: Message returned when the tool is called outside plan mode.
NOT_IN_PLAN_MODE_MSG = "Not in plan mode — nothing to approve."

#: Message returned when the plan is approved.
APPROVED_MSG = "Plan approved. Plan mode is now OFF — you may begin implementation."

#: Message returned when the plan is rejected.
REJECTED_MSG = "Plan rejected. Revise the plan based on the guidance and present it again."

#: Type alias for the async approval prompt: takes the plan text and returns
#: ``APPROVE`` or a rejection (optionally carrying guidance text).
ApprovalPrompt = Callable[[str], Awaitable[str]]


class PlanApprovalGate:
    """Awaitable approval gate for exiting plan mode.

    The gate owns the decision state and the permission-engine reference.
    ``request_approval`` is the single entry point the tool awaits; the TUI
    drives the outcome through ``approve`` / ``reject`` (or through the
    injected ``approval_prompt`` callable).
    """

    def __init__(
        self,
        permission_engine: Any | None = None,
        approval_prompt: ApprovalPrompt | None = None,
    ) -> None:
        self._engine = permission_engine
        self._approval_prompt = approval_prompt
        self._decision_event = asyncio.Event()
        self._approved = False
        self._guidance = ""
        self._pending_plan = ""
        self._pending = False

    @property
    def plan_mode_active(self) -> bool:
        """Whether plan mode is currently active."""
        return bool(getattr(self._engine, "plan_mode", False))

    @property
    def pending(self) -> bool:
        """Whether an approval request is currently awaiting a decision."""
        return self._pending

    @property
    def pending_plan(self) -> str:
        """The plan text currently awaiting approval."""
        return self._pending_plan

    @property
    def approved(self) -> bool:
        """Whether the most recent request was approved."""
        return self._approved

    @property
    def guidance(self) -> str:
        """Guidance recorded on rejection, for the model to revise the plan."""
        return self._guidance

    def attach_engine(self, permission_engine: Any) -> None:
        """Attach (or replace) the permission-engine reference."""
        self._engine = permission_engine

    async def request_approval(self, plan: str) -> str:
        """Present *plan* and await the human's decision.

        Returns a message describing the outcome. When plan mode is off,
        returns ``NOT_IN_PLAN_MODE_MSG`` without prompting.
        """
        if not self.plan_mode_active:
            return NOT_IN_PLAN_MODE_MSG

        self._pending = True
        self._pending_plan = plan
        self._decision_event.clear()
        self._approved = False
        self._guidance = ""

        if self._approval_prompt is not None:
            decision = await self._approval_prompt(plan)
            if decision == APPROVE:
                self.approve()
            else:
                guidance = decision if decision != REJECT else ""
                self.reject(guidance)
        else:
            # Headless default: auto-approve so plan mode can be exited.
            self.approve()

        await self._decision_event.wait()
        self._pending = False

        if self._approved:
            return APPROVED_MSG
        return REJECTED_MSG

    def approve(self) -> None:
        """Approve the pending plan — turns plan mode off."""
        if self._engine is not None:
            self._engine.plan_mode = False
        self._approved = True
        self._decision_event.set()

    def reject(self, guidance: str = "") -> None:
        """Reject the pending plan, keeping plan mode on.

        Args:
            guidance: Optional text the model should use to revise the plan.
        """
        self._approved = False
        self._guidance = guidance
        self._decision_event.set()
