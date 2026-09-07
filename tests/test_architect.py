"""Tests for architect mode — two-phase plan-then-execute pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.agent.architect import (
    ARCHITECT_SYSTEM_PROMPT,
    _filter_read_only,
    architect_loop,
)
from godspeed.agent.conversation import Conversation
from godspeed.llm.client import LLMClient
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

# -- Helpers ------------------------------------------------------------------


class _FakeTool(Tool):
    """Minimal tool stub for testing."""

    def __init__(self, name: str, risk: RiskLevel) -> None:
        self._name = name
        self._risk = risk

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk

    def get_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult.ok(f"{self._name} executed")


def _make_registry() -> ToolRegistry:
    """Build a registry with a mix of risk levels."""
    registry = ToolRegistry()
    registry.register(_FakeTool("file_read", RiskLevel.READ_ONLY))
    registry.register(_FakeTool("glob_search", RiskLevel.READ_ONLY))
    registry.register(_FakeTool("grep_search", RiskLevel.READ_ONLY))
    registry.register(_FakeTool("file_edit", RiskLevel.LOW))
    registry.register(_FakeTool("file_write", RiskLevel.HIGH))
    registry.register(_FakeTool("shell", RiskLevel.DESTRUCTIVE))
    return registry


def _make_context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session-001")


def _make_conversation() -> Conversation:
    return Conversation(
        system_prompt="You are a helpful assistant.",
        model="test-model",
        max_tokens=100_000,
    )


def _make_llm_client() -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client.model = "main-model"
    return client


# -- Tests: _filter_read_only ------------------------------------------------


class TestFilterReadOnly:
    """Tests for the read-only registry filter."""

    def test_returns_only_read_only_tools(self) -> None:
        registry = _make_registry()
        filtered = _filter_read_only(registry)

        tool_names = {t.name for t in filtered.list_tools()}
        assert tool_names == {"file_read", "glob_search", "grep_search"}

    def test_excludes_low_risk_tools(self) -> None:
        registry = _make_registry()
        filtered = _filter_read_only(registry)

        assert filtered.get("file_edit") is None

    def test_excludes_high_risk_tools(self) -> None:
        registry = _make_registry()
        filtered = _filter_read_only(registry)

        assert filtered.get("file_write") is None

    def test_excludes_destructive_tools(self) -> None:
        registry = _make_registry()
        filtered = _filter_read_only(registry)

        assert filtered.get("shell") is None

    def test_empty_registry_returns_empty(self) -> None:
        registry = ToolRegistry()
        filtered = _filter_read_only(registry)

        assert filtered.list_tools() == []

    def test_all_read_only_registry_passes_all(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("a", RiskLevel.READ_ONLY))
        registry.register(_FakeTool("b", RiskLevel.READ_ONLY))
        filtered = _filter_read_only(registry)

        assert len(filtered.list_tools()) == 2


# -- Tests: architect_loop ---------------------------------------------------


class TestArchitectLoop:
    """Tests for the two-phase architect loop."""

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_plan_phase_uses_read_only_registry(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """Phase 1 should receive a registry containing only READ_ONLY tools."""
        mock_agent_loop.side_effect = [
            "Step 1: Read main.py\nStep 2: Edit it",  # plan
            "Done implementing.",  # execute
        ]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        await architect_loop(
            user_input="Add logging to main.py",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        # First call is the plan phase
        plan_call_kwargs = mock_agent_loop.call_args_list[0].kwargs
        plan_registry = plan_call_kwargs["tool_registry"]
        plan_tool_names = {t.name for t in plan_registry.list_tools()}
        assert plan_tool_names == {"file_read", "glob_search", "grep_search"}

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_execute_phase_uses_full_registry(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """Phase 2 should receive the full (unfiltered) tool registry."""
        mock_agent_loop.side_effect = [
            "Step 1: Read main.py\nStep 2: Edit it",  # plan
            "Done implementing.",  # execute
        ]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        await architect_loop(
            user_input="Add logging to main.py",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        # Second call is the execute phase
        exec_call_kwargs = mock_agent_loop.call_args_list[1].kwargs
        exec_registry = exec_call_kwargs["tool_registry"]
        exec_tool_names = {t.name for t in exec_registry.list_tools()}
        assert "file_edit" in exec_tool_names
        assert "shell" in exec_tool_names
        assert len(exec_tool_names) == 6

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_plan_injected_into_execute_context(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """The plan from phase 1 should appear in the user_input of phase 2."""
        plan_text = "Step 1: Read file\nStep 2: Add import\nStep 3: Write file"
        mock_agent_loop.side_effect = [plan_text, "All done."]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        await architect_loop(
            user_input="Add logging",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        exec_call_kwargs = mock_agent_loop.call_args_list[1].kwargs
        exec_input = exec_call_kwargs["user_input"]
        assert plan_text in exec_input
        assert "Add logging" in exec_input
        assert "Execute this plan" in exec_input

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_plan_failure_returns_error(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """If the plan phase returns an error, architect_loop should return it."""
        mock_agent_loop.return_value = "Error: LLM call failed — timeout"

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        result = await architect_loop(
            user_input="Do something",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        assert result.startswith("Error:")
        assert mock_agent_loop.call_count == 1  # No execute phase

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_empty_plan_returns_error(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """If the plan phase returns empty string, return an error."""
        mock_agent_loop.return_value = ""

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        result = await architect_loop(
            user_input="Do something",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        assert "no output" in result.lower() or result.startswith("Error:")
        assert mock_agent_loop.call_count == 1

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_phase_change_callback_called(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """on_phase_change should be called for both phases."""
        mock_agent_loop.side_effect = ["The plan.", "Executed."]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        phase_changes: list[tuple[str, str]] = []

        def on_phase(phase: str, model: str) -> None:
            phase_changes.append((phase, model))

        await architect_loop(
            user_input="Build it",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
            on_phase_change=on_phase,
        )

        assert len(phase_changes) == 2
        assert phase_changes[0][0] == "plan"
        assert phase_changes[1][0] == "execute"

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_architect_model_switches_for_plan(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """When architect_model differs from main, it should switch during plan."""
        mock_agent_loop.side_effect = ["The plan.", "Done."]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()
        client.model = "main-model"

        phase_models: list[str] = []

        def on_phase(phase: str, model: str) -> None:
            phase_models.append(model)

        await architect_loop(
            user_input="Build it",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
            architect_model="planning-model",
            on_phase_change=on_phase,
        )

        assert phase_models[0] == "planning-model"
        assert phase_models[1] == "main-model"
        # Model should be restored after plan phase
        assert client.model == "main-model"

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_architect_model_restored_on_plan_error(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """Model must be restored even when the plan phase raises."""
        mock_agent_loop.side_effect = RuntimeError("plan failed")

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()
        client.model = "main-model"

        with pytest.raises(RuntimeError, match="plan failed"):
            await architect_loop(
                user_input="Build it",
                conversation=conversation,
                llm_client=client,
                tool_registry=registry,
                tool_context=context,
                architect_model="planning-model",
            )

        assert client.model == "main-model"

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_plan_conversation_uses_architect_system_prompt(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """Phase 1 conversation should use the ARCHITECT_SYSTEM_PROMPT."""
        mock_agent_loop.side_effect = ["Plan.", "Done."]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        await architect_loop(
            user_input="Build it",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        plan_call_kwargs = mock_agent_loop.call_args_list[0].kwargs
        plan_conv = plan_call_kwargs["conversation"]
        system_msg = plan_conv.messages[0]
        assert system_msg["content"] == ARCHITECT_SYSTEM_PROMPT

    @pytest.mark.asyncio
    @patch("godspeed.agent.architect.agent_loop", new_callable=AsyncMock)
    async def test_execute_uses_original_conversation(
        self, mock_agent_loop: AsyncMock, tmp_path: Path
    ) -> None:
        """Phase 2 should use the caller's conversation, not the plan conversation."""
        mock_agent_loop.side_effect = ["Plan.", "Done."]

        registry = _make_registry()
        context = _make_context(tmp_path)
        conversation = _make_conversation()
        client = _make_llm_client()

        await architect_loop(
            user_input="Build it",
            conversation=conversation,
            llm_client=client,
            tool_registry=registry,
            tool_context=context,
        )

        exec_call_kwargs = mock_agent_loop.call_args_list[1].kwargs
        assert exec_call_kwargs["conversation"] is conversation


# -- Tests: /architect command toggle -----------------------------------------


class TestArchitectCommand:
    """/architect command toggles architect_mode on the Commands instance."""

    def test_toggle_on(self) -> None:
        from godspeed.tui.commands import Commands

        cmds = Commands(
            conversation=MagicMock(),
            llm_client=MagicMock(),
            permission_engine=MagicMock(),
            audit_trail=None,
            session_id="test",
            cwd=Path("."),
            tool_registry=None,
        )
        assert cmds.architect_mode is False
        result = cmds.dispatch("/architect")
        assert result is not None
        assert result.handled is True
        assert cmds.architect_mode is True

    def test_toggle_off(self) -> None:
        from godspeed.tui.commands import Commands

        cmds = Commands(
            conversation=MagicMock(),
            llm_client=MagicMock(),
            permission_engine=MagicMock(),
            audit_trail=None,
            session_id="test",
            cwd=Path("."),
            tool_registry=None,
        )
        cmds.architect_mode = True
        result = cmds.dispatch("/architect")
        assert result is not None
        assert result.handled is True
        assert cmds.architect_mode is False

    def test_double_toggle_returns_to_original(self) -> None:
        from godspeed.tui.commands import Commands

        cmds = Commands(
            conversation=MagicMock(),
            llm_client=MagicMock(),
            permission_engine=MagicMock(),
            audit_trail=None,
            session_id="test",
            cwd=Path("."),
            tool_registry=None,
        )
        cmds.dispatch("/architect")
        cmds.dispatch("/architect")
        assert cmds.architect_mode is False
