"""Tests for ToolRegistry.dispatch routing high-risk tools through sandbox policy."""

from __future__ import annotations

import asyncio

import pytest

from godspeed.sandbox.policy import NetworkRule, SandboxPolicy
from godspeed.tools.base import RiskLevel, ToolCall, ToolContext
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


@pytest.fixture
def sandbox_policy() -> SandboxPolicy:
    return SandboxPolicy(
        network_rules=[NetworkRule(pattern="evil.com", action="deny")],
    )


@pytest.fixture
def registry_with_sandbox(
    sandbox_policy: SandboxPolicy,
) -> ToolRegistry:
    return ToolRegistry(sandbox=sandbox_policy)


class TestDispatchSandboxGate:
    """Verify dispatch routes high-risk/destructive tools through sandbox policy."""

    def test_blocked_egress_web_fetch(
        self,
        registry_with_sandbox: ToolRegistry,
        tool_context: ToolContext,
    ) -> None:
        fetch_tool = MockTool(name="web_fetch", risk_level=RiskLevel.HIGH)
        registry_with_sandbox.register(fetch_tool)

        tool_call = ToolCall(
            tool_name="web_fetch",
            arguments={"url": "https://evil.com/payload"},
        )
        result = asyncio.run(registry_with_sandbox.dispatch(tool_call, tool_context))
        assert result.is_error
        assert "Sandbox denied" in result.error

    def test_contained_shell_blocked_egress(
        self,
        registry_with_sandbox: ToolRegistry,
        tool_context: ToolContext,
    ) -> None:
        shell_tool = MockTool(name="shell", risk_level=RiskLevel.HIGH)
        registry_with_sandbox.register(shell_tool)

        tool_call = ToolCall(
            tool_name="shell",
            arguments={"url": "https://evil.com/exfil"},
        )
        result = asyncio.run(registry_with_sandbox.dispatch(tool_call, tool_context))
        assert result.is_error
        assert "Sandbox denied" in result.error

    def test_contained_shell_no_network_target_passes(
        self,
        registry_with_sandbox: ToolRegistry,
        tool_context: ToolContext,
    ) -> None:
        shell_tool = MockTool(name="shell", risk_level=RiskLevel.HIGH)
        registry_with_sandbox.register(shell_tool)

        tool_call = ToolCall(
            tool_name="shell",
            arguments={"command": "ls -la"},
        )
        result = asyncio.run(registry_with_sandbox.dispatch(tool_call, tool_context))
        assert not result.is_error

    def test_destructive_tool_denied_by_sandbox(
        self,
        registry_with_sandbox: ToolRegistry,
        tool_context: ToolContext,
    ) -> None:
        diff_tool = MockTool(name="diff_apply", risk_level=RiskLevel.DESTRUCTIVE)
        registry_with_sandbox.register(diff_tool)

        tool_call = ToolCall(
            tool_name="diff_apply",
            arguments={"endpoint": "https://evil.com/exploit"},
        )
        result = asyncio.run(registry_with_sandbox.dispatch(tool_call, tool_context))
        assert result.is_error
        assert "Sandbox denied" in result.error

    def test_read_only_tool_skips_sandbox(
        self,
        registry_with_sandbox: ToolRegistry,
        tool_context: ToolContext,
    ) -> None:
        read_tool = MockTool(name="file_read", risk_level=RiskLevel.READ_ONLY)
        registry_with_sandbox.register(read_tool)

        tool_call = ToolCall(
            tool_name="file_read",
            arguments={"url": "https://evil.com/stuff"},
        )
        result = asyncio.run(registry_with_sandbox.dispatch(tool_call, tool_context))
        assert not result.is_error

    def test_no_sandbox_allows_high_risk(
        self,
        tool_context: ToolContext,
    ) -> None:
        registry = ToolRegistry(sandbox=None)
        shell_tool = MockTool(name="shell", risk_level=RiskLevel.HIGH)
        registry.register(shell_tool)

        tool_call = ToolCall(
            tool_name="shell",
            arguments={"url": "https://evil.com/exploit"},
        )
        result = asyncio.run(registry.dispatch(tool_call, tool_context))
        assert not result.is_error

    def test_low_risk_tool_skips_sandbox(
        self,
        registry_with_sandbox: ToolRegistry,
        tool_context: ToolContext,
    ) -> None:
        low_tool = MockTool(name="file_edit", risk_level=RiskLevel.LOW)
        registry_with_sandbox.register(low_tool)

        tool_call = ToolCall(tool_name="file_edit", arguments={})
        result = asyncio.run(registry_with_sandbox.dispatch(tool_call, tool_context))
        assert not result.is_error


class TestLowRiskSandboxGate:
    """LOW-risk write tools (file_write/edit/move, git, diff_apply) are sandbox-gated."""

    def test_low_risk_write_tool_blocked_path_denied(self, tool_context: ToolContext) -> None:
        policy = SandboxPolicy(blocked_paths=["/etc"])
        registry = ToolRegistry(sandbox=policy)
        registry.register(MockTool(name="file_edit", risk_level=RiskLevel.LOW))

        tool_call = ToolCall(tool_name="file_edit", arguments={"path": "/etc/passwd"})
        result = asyncio.run(registry.dispatch(tool_call, tool_context))
        assert result.is_error
        assert "Sandbox denied" in result.error

    def test_low_risk_write_tool_normal_path_passes(self, tool_context: ToolContext) -> None:
        policy = SandboxPolicy()
        registry = ToolRegistry(sandbox=policy)
        registry.register(MockTool(name="file_edit", risk_level=RiskLevel.LOW))

        tool_call = ToolCall(tool_name="file_edit", arguments={"path": "src/app.py"})
        result = asyncio.run(registry.dispatch(tool_call, tool_context))
        assert not result.is_error

    def test_non_bundled_low_tool_skips_gate(self, tool_context: ToolContext) -> None:
        policy = SandboxPolicy(blocked_paths=["/etc"])
        registry = ToolRegistry(sandbox=policy)
        registry.register(MockTool(name="web_search", risk_level=RiskLevel.LOW))

        tool_call = ToolCall(tool_name="web_search", arguments={"query": "hello"})
        result = asyncio.run(registry.dispatch(tool_call, tool_context))
        assert not result.is_error
