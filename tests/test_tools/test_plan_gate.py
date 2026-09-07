"""Tests for the plan-mode approval gate and exit_plan_mode tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.security.permissions import ALLOW, DENY, PermissionEngine
from godspeed.security.plan_gate import (
    APPROVE,
    APPROVED_MSG,
    NOT_IN_PLAN_MODE_MSG,
    PLAN_GATE_TOOL_NAME,
    REJECT,
    REJECTED_MSG,
    PlanApprovalGate,
)
from godspeed.tools.base import RiskLevel, ToolCall, ToolContext
from godspeed.tools.plan_gate import ExitPlanModeTool
from godspeed.tools.tasks import TaskStore, build_continuation_nudge


@pytest.fixture()
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


class TestPlanApprovalGate:
    """Test the awaitable approval gate."""

    def test_plan_mode_active_reflects_engine(self) -> None:
        engine = PermissionEngine()
        gate = PlanApprovalGate(permission_engine=engine)
        assert gate.plan_mode_active is False
        engine.plan_mode = True
        assert gate.plan_mode_active is True

    def test_attach_engine(self) -> None:
        engine = PermissionEngine()
        gate = PlanApprovalGate()
        assert gate.plan_mode_active is False
        gate.attach_engine(engine)
        engine.plan_mode = True
        assert gate.plan_mode_active is True

    @pytest.mark.asyncio()
    async def test_request_approval_outside_plan_mode(self) -> None:
        engine = PermissionEngine()
        gate = PlanApprovalGate(permission_engine=engine)
        message = await gate.request_approval("My plan")
        assert message == NOT_IN_PLAN_MODE_MSG
        assert gate.pending is False

    @pytest.mark.asyncio()
    async def test_headless_auto_approves(self) -> None:
        engine = PermissionEngine()
        engine.plan_mode = True
        gate = PlanApprovalGate(permission_engine=engine)
        message = await gate.request_approval("My plan")
        assert message == APPROVED_MSG
        assert gate.approved is True
        assert engine.plan_mode is False

    @pytest.mark.asyncio()
    async def test_approval_prompt_approve(self) -> None:
        engine = PermissionEngine()
        engine.plan_mode = True
        gate = PlanApprovalGate(permission_engine=engine, approval_prompt=_prompt(APPROVE))
        message = await gate.request_approval("My plan")
        assert message == APPROVED_MSG
        assert engine.plan_mode is False

    @pytest.mark.asyncio()
    async def test_approval_prompt_reject_keeps_plan_mode(self) -> None:
        engine = PermissionEngine()
        engine.plan_mode = True
        gate = PlanApprovalGate(permission_engine=engine, approval_prompt=_prompt(REJECT))
        message = await gate.request_approval("My plan")
        assert message == REJECTED_MSG
        assert engine.plan_mode is True
        assert gate.guidance == ""

    @pytest.mark.asyncio()
    async def test_approval_prompt_reject_with_guidance(self) -> None:
        engine = PermissionEngine()
        engine.plan_mode = True
        gate = PlanApprovalGate(
            permission_engine=engine,
            approval_prompt=_prompt("Add tests first"),
        )
        message = await gate.request_approval("My plan")
        assert message == REJECTED_MSG
        assert engine.plan_mode is True
        assert gate.guidance == "Add tests first"

    @pytest.mark.asyncio()
    async def test_approve_flips_plan_mode(self) -> None:
        engine = PermissionEngine()
        engine.plan_mode = True
        gate = PlanApprovalGate(permission_engine=engine)
        gate.approve()
        assert gate.approved is True
        assert engine.plan_mode is False

    @pytest.mark.asyncio()
    async def test_reject_keeps_plan_mode_with_guidance(self) -> None:
        engine = PermissionEngine()
        engine.plan_mode = True
        gate = PlanApprovalGate(permission_engine=engine)
        gate.reject("Too vague")
        assert gate.approved is False
        assert engine.plan_mode is True
        assert gate.guidance == "Too vague"


def _prompt(decision: str):
    async def _prompt_impl(plan: str) -> str:
        return decision

    return _prompt_impl


class TestExitPlanModeTool:
    """Test the exit_plan_mode tool."""

    @pytest.fixture()
    def engine(self) -> PermissionEngine:
        return PermissionEngine()

    @pytest.fixture()
    def gate(self, engine: PermissionEngine) -> PlanApprovalGate:
        return PlanApprovalGate(permission_engine=engine)

    @pytest.fixture()
    def tool(self, gate: PlanApprovalGate) -> ExitPlanModeTool:
        return ExitPlanModeTool(gate)

    def test_name_and_risk(self, tool: ExitPlanModeTool) -> None:
        assert tool.name == PLAN_GATE_TOOL_NAME
        assert tool.risk_level == RiskLevel.READ_ONLY

    def test_schema_has_plan(self, tool: ExitPlanModeTool) -> None:
        schema = tool.get_schema()
        assert "plan" in schema["properties"]

    @pytest.mark.asyncio()
    async def test_execute_outside_plan_mode(
        self, tool: ExitPlanModeTool, context: ToolContext
    ) -> None:
        result = await tool.execute({"plan": "My plan"}, context)
        assert not result.is_error
        assert NOT_IN_PLAN_MODE_MSG in result.output

    @pytest.mark.asyncio()
    async def test_execute_approves(
        self, tool: ExitPlanModeTool, engine: PermissionEngine, context: ToolContext
    ) -> None:
        engine.plan_mode = True
        result = await tool.execute({"plan": "My plan"}, context)
        assert not result.is_error
        assert APPROVED_MSG in result.output
        assert engine.plan_mode is False

    @pytest.mark.asyncio()
    async def test_execute_reject_with_guidance(
        self, tool: ExitPlanModeTool, engine: PermissionEngine, context: ToolContext
    ) -> None:
        engine.plan_mode = True
        tool._gate._approval_prompt = _prompt("Add tests first")
        result = await tool.execute({"plan": "My plan"}, context)
        assert not result.is_error
        assert REJECTED_MSG in result.output
        assert "Add tests first" in result.output
        assert engine.plan_mode is True


class TestPlanModeExemption:
    """exit_plan_mode is exempt from the plan-mode block."""

    def test_exit_plan_mode_allowed_in_plan_mode(self) -> None:
        engine = PermissionEngine(
            tool_risk_levels={PLAN_GATE_TOOL_NAME: RiskLevel.READ_ONLY},
        )
        engine.plan_mode = True
        tc = ToolCall(tool_name=PLAN_GATE_TOOL_NAME, arguments={"plan": "My plan"})
        assert engine.evaluate(tc) == ALLOW

    def test_other_tools_still_blocked_in_plan_mode(self) -> None:
        engine = PermissionEngine(
            tool_risk_levels={"file_edit": RiskLevel.LOW},
        )
        engine.plan_mode = True
        tc = ToolCall(tool_name="file_edit", arguments={"file_path": "test.py"})
        assert engine.evaluate(tc) == DENY


class TestBuildContinuationNudge:
    """Test the continuation nudge builder."""

    def test_none_when_no_tasks(self) -> None:
        assert build_continuation_nudge([]) is None

    def test_none_when_all_completed(self) -> None:
        store = TaskStore()
        store.create("Done")
        store.complete(1)
        assert build_continuation_nudge(store.list_all()) is None

    def test_nudge_lists_open_tasks(self) -> None:
        store = TaskStore()
        store.create("Fix bug")
        store.create("Write tests")
        store.update(1, "in_progress")
        store.create("Done")
        store.complete(3)
        nudge = build_continuation_nudge(store.list_all())
        assert nudge is not None
        assert "Fix bug" in nudge
        assert "Write tests" in nudge
        assert "Done" not in nudge
        assert "in_progress" in nudge
