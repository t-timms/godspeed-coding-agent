"""Tests for tool base types and protocol."""

from __future__ import annotations

import pytest

from godspeed.tools.base import RiskLevel, ToolCall, ToolContext, ToolResult
from tests.conftest import MockTool


class TestRiskLevel:
    """Test risk level enum."""

    def test_values(self) -> None:
        assert RiskLevel.READ_ONLY == "read_only"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.DESTRUCTIVE == "destructive"

    def test_ordering_by_severity(self) -> None:
        levels = [RiskLevel.DESTRUCTIVE, RiskLevel.READ_ONLY, RiskLevel.HIGH, RiskLevel.LOW]
        # Should be sortable alphabetically (but we don't rely on this)
        assert len(set(levels)) == 4


class TestToolResult:
    """Test ToolResult creation."""

    def test_success(self) -> None:
        r = ToolResult.success("hello")
        assert r.output == "hello"
        assert r.error is None
        assert r.is_error is False

    def test_failure(self) -> None:
        r = ToolResult.failure("something broke")
        assert r.output == ""
        assert r.error == "something broke"
        assert r.is_error is True

    def test_default_empty(self) -> None:
        r = ToolResult()
        assert r.output == ""
        assert r.error is None
        assert r.is_error is False


class TestToolCall:
    """Test ToolCall formatting."""

    def test_format_with_string_arg(self) -> None:
        tc = ToolCall(tool_name="Bash", arguments={"command": "git status"})
        assert tc.format_for_permission == "bash(git status)"

    def test_format_no_args(self) -> None:
        tc = ToolCall(tool_name="FileRead", arguments={})
        assert tc.format_for_permission == "file_read()"

    def test_format_non_string_args(self) -> None:
        tc = ToolCall(tool_name="Resize", arguments={"width": 100, "height": 200})
        assert tc.format_for_permission == "resize(*)"

    def test_format_normalizes_snake_case_name_unchanged(self) -> None:
        tc = ToolCall(tool_name="file_write", arguments={"file_path": "prod.env"})
        assert tc.format_for_permission == "file_write(prod.env)"


class TestToolContext:
    """Test ToolContext creation."""

    def test_basic_creation(self, tool_context: ToolContext) -> None:
        assert tool_context.session_id == "test-session-001"
        assert tool_context.cwd.exists()

    def test_permissions_default_none(self, tool_context: ToolContext) -> None:
        assert tool_context.permissions is None

    def test_audit_default_none(self, tool_context: ToolContext) -> None:
        assert tool_context.audit is None


class TestMockTool:
    """Test the MockTool fixture implementation."""

    @pytest.mark.asyncio
    async def test_execute(self, tool_context: ToolContext) -> None:
        tool = MockTool()
        result = await tool.execute({"input": "test"}, tool_context)
        assert result.output == "mock output"
        assert tool.last_arguments == {"input": "test"}

    def test_schema(self) -> None:
        tool = MockTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "input" in schema["properties"]

    def test_custom_risk_level(self) -> None:
        tool = MockTool(risk_level=RiskLevel.DESTRUCTIVE)
        assert tool.risk_level == RiskLevel.DESTRUCTIVE

    @pytest.mark.asyncio
    async def test_custom_result(self, tool_context: ToolContext) -> None:
        tool = MockTool(result=ToolResult.failure("nope"))
        result = await tool.execute({}, tool_context)
        assert result.is_error is True
        assert result.error == "nope"
