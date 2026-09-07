"""Tests for the think tool."""

from __future__ import annotations

import pytest

from godspeed._bootstrap import _build_tool_registry
from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.think import MAX_THOUGHT_CHARS, ThinkTool
from godspeed.tools.tool_sets import TOOL_SET_LOCAL, get_allowed_tool_names


@pytest.fixture
def tool() -> ThinkTool:
    return ThinkTool()


class TestThinkToolMetadata:
    def test_name(self, tool: ThinkTool) -> None:
        assert tool.name == "think"

    def test_risk_level_is_read_only(self, tool: ThinkTool) -> None:
        assert tool.risk_level == RiskLevel.READ_ONLY

    def test_description_mentions_no_environment_change(self, tool: ThinkTool) -> None:
        assert "changes nothing" in tool.description

    def test_schema_requires_thought(self, tool: ThinkTool) -> None:
        schema = tool.get_schema()
        assert schema["required"] == ["thought"]
        assert schema["properties"]["thought"]["type"] == "string"
        assert schema["properties"]["next_actions"]["type"] == "array"

    def test_concurrency_safe(self, tool: ThinkTool) -> None:
        assert tool.concurrency_safe is True


class TestThinkToolExecution:
    @pytest.mark.asyncio
    async def test_records_thought(self, tool: ThinkTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"thought": "Check the traceback first."}, tool_context)
        assert not result.is_error
        assert result.output == "Thought recorded. Continue with your plan."

    @pytest.mark.asyncio
    async def test_confirmation_does_not_echo_thought(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        thought = "A very specific strategy the model wrote down."
        result = await tool.execute({"thought": thought}, tool_context)
        assert thought not in result.output

    @pytest.mark.asyncio
    async def test_empty_thought_rejected(self, tool: ThinkTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"thought": ""}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error

    @pytest.mark.asyncio
    async def test_whitespace_thought_rejected(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({"thought": "   \n\t "}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error

    @pytest.mark.asyncio
    async def test_missing_thought_rejected(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error

    @pytest.mark.asyncio
    async def test_non_string_thought_rejected(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({"thought": 12345}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error

    @pytest.mark.asyncio
    async def test_oversized_thought_rejected(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({"thought": "x" * (MAX_THOUGHT_CHARS + 1)}, tool_context)
        assert result.is_error
        assert "exceeds maximum length" in result.error
        assert str(MAX_THOUGHT_CHARS) in result.error

    @pytest.mark.asyncio
    async def test_max_length_thought_accepted(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({"thought": "x" * MAX_THOUGHT_CHARS}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_extra_arguments_ignored(
        self, tool: ThinkTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute(
            {"thought": "Plan A", "next_actions": ["run tests", "fix import"]}, tool_context
        )
        assert not result.is_error
        assert result.output == "Thought recorded. Continue with your plan."


class TestThinkToolRegistration:
    def test_registered_in_full_registry(self) -> None:
        registry, _ = _build_tool_registry("full")
        assert registry.has_tool("think")
        tool = registry.get("think")
        assert tool is not None
        assert tool.risk_level == RiskLevel.READ_ONLY

    def test_in_local_tool_set(self) -> None:
        allowed = get_allowed_tool_names(TOOL_SET_LOCAL)
        assert allowed is not None
        assert "think" in allowed
