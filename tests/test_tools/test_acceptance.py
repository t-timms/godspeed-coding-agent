"""Tests for the acceptance-criteria tracker tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from godspeed.agent.completion_gate import (
    CompletionGateState,
    GateDecision,
    should_block,
)
from godspeed.tools.acceptance import (
    ACCEPTANCE_DIRNAME,
    ACCEPTANCE_FILENAME,
    AcceptanceContract,
    AcceptanceInitTool,
    AcceptanceStatusTool,
    AcceptanceUpdateTool,
)
from godspeed.tools.base import RiskLevel, ToolContext


@pytest.fixture()
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


def _contract_path(context: ToolContext) -> Path:
    return context.cwd / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME


class TestAcceptanceContract:
    """Test the AcceptanceContract data model."""

    def test_from_titles_all_failing(self) -> None:
        contract = AcceptanceContract.from_titles(["Fix bug", "Add tests"])
        assert len(contract.items) == 2
        assert all(item.status == "failing" for item in contract.items)
        assert all(item.evidence is None for item in contract.items)
        assert [item.id for item in contract.items] == [1, 2]

    def test_update_passing_requires_evidence(self) -> None:
        contract = AcceptanceContract.from_titles(["Fix bug"])
        with pytest.raises(ValueError):
            contract.update(1, "passing", evidence="")

    def test_update_passing_with_evidence(self) -> None:
        contract = AcceptanceContract.from_titles(["Fix bug"])
        item = contract.update(1, "passing", evidence="tests/test_x.py passed")
        assert item is not None
        assert item.status == "passing"
        assert item.evidence == "tests/test_x.py passed"

    def test_update_failing_always_allowed(self) -> None:
        contract = AcceptanceContract.from_titles(["Fix bug"])
        item = contract.update(1, "failing", evidence=None)
        assert item is not None
        assert item.status == "failing"

    def test_update_nonexistent(self) -> None:
        contract = AcceptanceContract.from_titles(["Fix bug"])
        assert contract.update(99, "passing", evidence="x") is None

    def test_failing_items(self) -> None:
        contract = AcceptanceContract.from_titles(["A", "B", "C"])
        contract.update(1, "passing", evidence="proof")
        failing = contract.failing_items()
        assert [item.id for item in failing] == [2, 3]

    def test_format_active_none_when_all_pass(self) -> None:
        contract = AcceptanceContract.from_titles(["A"])
        contract.update(1, "passing", evidence="proof")
        assert contract.format_active() is None

    def test_format_active_with_failing(self) -> None:
        contract = AcceptanceContract.from_titles(["A", "B"])
        active = contract.format_active()
        assert active is not None
        assert "A" in active
        assert "B" in active

    def test_format_status(self) -> None:
        contract = AcceptanceContract.from_titles(["A"])
        contract.update(1, "passing", evidence="proof")
        status = contract.format_status()
        assert "A" in status
        assert "passing" in status
        assert "evidence" in status

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME
        contract = AcceptanceContract.from_titles(["A", "B"])
        contract.update(1, "passing", evidence="proof")
        contract.save(path)
        loaded = AcceptanceContract.load(path)
        assert len(loaded.items) == 2
        assert loaded.get(1) is not None
        assert loaded.get(1).status == "passing"  # type: ignore[union-attr]
        assert loaded.get(1).evidence == "proof"  # type: ignore[union-attr]

    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME
        contract = AcceptanceContract.load(path)
        assert contract.items == []

    def test_load_corrupt_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        contract = AcceptanceContract.load(path)
        assert contract.items == []


class TestAcceptanceTools:
    """Test the acceptance tool executions."""

    @pytest.mark.asyncio()
    async def test_init_creates_all_failing_json(self, context: ToolContext) -> None:
        tool = AcceptanceInitTool()
        result = await tool.execute({"titles": ["Fix bug", "Add tests"]}, context)
        assert not result.is_error
        path = _contract_path(context)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["items"]) == 2
        assert all(item["status"] == "failing" for item in data["items"])
        assert all(item["evidence"] is None for item in data["items"])

    @pytest.mark.asyncio()
    async def test_init_requires_titles(self, context: ToolContext) -> None:
        tool = AcceptanceInitTool()
        result = await tool.execute({}, context)
        assert result.is_error

    @pytest.mark.asyncio()
    async def test_update_without_evidence_rejected(self, context: ToolContext) -> None:
        init = AcceptanceInitTool()
        await init.execute({"titles": ["Fix bug"]}, context)
        tool = AcceptanceUpdateTool()
        result = await tool.execute({"item_id": 1, "status": "passing"}, context)
        assert result.is_error
        assert "evidence" in (result.error or "")

    @pytest.mark.asyncio()
    async def test_update_with_evidence_flips(self, context: ToolContext) -> None:
        init = AcceptanceInitTool()
        await init.execute({"titles": ["Fix bug"]}, context)
        tool = AcceptanceUpdateTool()
        result = await tool.execute(
            {"item_id": 1, "status": "passing", "evidence": "tests passed"},
            context,
        )
        assert not result.is_error
        data = json.loads(_contract_path(context).read_text(encoding="utf-8"))
        assert data["items"][0]["status"] == "passing"
        assert data["items"][0]["evidence"] == "tests passed"

    @pytest.mark.asyncio()
    async def test_update_failing_allowed_without_evidence(self, context: ToolContext) -> None:
        init = AcceptanceInitTool()
        await init.execute({"titles": ["Fix bug"]}, context)
        tool = AcceptanceUpdateTool()
        result = await tool.execute({"item_id": 1, "status": "failing"}, context)
        assert not result.is_error

    @pytest.mark.asyncio()
    async def test_update_nonexistent_item(self, context: ToolContext) -> None:
        init = AcceptanceInitTool()
        await init.execute({"titles": ["Fix bug"]}, context)
        tool = AcceptanceUpdateTool()
        result = await tool.execute({"item_id": 99, "status": "failing"}, context)
        assert result.is_error

    @pytest.mark.asyncio()
    async def test_status_rendering(self, context: ToolContext) -> None:
        init = AcceptanceInitTool()
        await init.execute({"titles": ["Fix bug", "Add tests"]}, context)
        tool = AcceptanceStatusTool()
        result = await tool.execute({}, context)
        assert not result.is_error
        assert "Fix bug" in result.output
        assert "Add tests" in result.output
        assert "failing" in result.output

    @pytest.mark.asyncio()
    async def test_status_empty(self, context: ToolContext) -> None:
        tool = AcceptanceStatusTool()
        result = await tool.execute({}, context)
        assert not result.is_error
        assert "No acceptance criteria" in result.output

    def test_risk_levels(self) -> None:
        assert AcceptanceInitTool().risk_level == RiskLevel.LOW
        assert AcceptanceUpdateTool().risk_level == RiskLevel.LOW
        assert AcceptanceStatusTool().risk_level == RiskLevel.READ_ONLY

    def test_names(self) -> None:
        assert AcceptanceInitTool().name == "acceptance_init"
        assert AcceptanceUpdateTool().name == "acceptance_update"
        assert AcceptanceStatusTool().name == "acceptance_status"


class TestGateIntegration:
    """Test that failing acceptance items block the pre-completion gate."""

    def test_failing_items_block_stop(self, tmp_path: Path) -> None:
        context = ToolContext(cwd=tmp_path, session_id="test")
        init = AcceptanceInitTool()
        import asyncio

        asyncio.run(init.execute({"titles": ["Fix bug"]}, context))
        state = CompletionGateState(
            has_edits_since_verify=False,
            tasks_open=True,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.BLOCK

    def test_all_passing_does_not_block(self, tmp_path: Path) -> None:
        context = ToolContext(cwd=tmp_path, session_id="test")
        init = AcceptanceInitTool()
        update = AcceptanceUpdateTool()
        import asyncio

        asyncio.run(init.execute({"titles": ["Fix bug"]}, context))
        asyncio.run(
            update.execute(
                {"item_id": 1, "status": "passing", "evidence": "tests passed"},
                context,
            )
        )
        state = CompletionGateState(
            has_edits_since_verify=False,
            tasks_open=False,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.PASS
