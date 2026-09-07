"""Tests for the sub-agent coordinator."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.coordinator import (
    MAX_SUB_AGENT_DEPTH,
    AgentCoordinator,
    SpawnAgentTool,
)
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_tool_response(tool_name: str, arguments: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[
            {
                "id": "call_sub_001",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
        finish_reason="tool_calls",
    )


class TestAgentCoordinator:
    """Test sub-agent spawning and isolation."""

    @pytest.mark.asyncio
    async def test_spawn_returns_result(self, tool_context: ToolContext) -> None:
        """Sub-agent runs and returns its final text."""
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=_make_text_response("Sub-task completed successfully.")
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Find all TODO comments")
        assert "completed successfully" in result

    @pytest.mark.asyncio
    async def test_spawn_uses_tools(self, tool_context: ToolContext) -> None:
        """Sub-agent can use tools from the shared registry."""
        registry = ToolRegistry()
        registry.register(
            MockTool(name="grep_search", result=ToolResult.success("TODO found in main.py:42"))
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("grep_search", {"pattern": "TODO"}),
                _make_text_response("Found 1 TODO in main.py line 42."),
            ]
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Search for TODOs")
        assert "TODO" in result
        assert client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_isolated_conversation(self, tool_context: ToolContext) -> None:
        """Each sub-agent gets its own conversation — no cross-contamination."""
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        # Spawn two agents — each should get independent conversations
        r1 = await coordinator.spawn("Task A")
        r2 = await coordinator.spawn("Task B")

        assert r1 == "Done."
        assert r2 == "Done."
        # Each spawn = 1 call (text response), so 2 total
        assert client.chat.call_count == 2

        # Verify messages are independent — check that "Task A" is NOT
        # in the messages sent for "Task B"
        second_call_messages = client.chat.call_args_list[1][1]["messages"]
        user_messages = [m for m in second_call_messages if m.get("role") == "user"]
        assert any("Task B" in m["content"] for m in user_messages)
        assert not any("Task A" in m.get("content", "") for m in user_messages)

    @pytest.mark.asyncio
    async def test_depth_limit(self, tool_context: ToolContext) -> None:
        """Depth limit prevents infinite sub-agent recursion."""
        registry = ToolRegistry()
        client = LLMClient(model="test")

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
            max_depth=2,
        )

        result = await coordinator.spawn("Task", depth=2)
        assert "maximum" in result.lower() or "depth" in result.lower()
        # LLM should NOT have been called
        assert not hasattr(client, "chat") or not getattr(client.chat, "called", False)

    @pytest.mark.asyncio
    async def test_iteration_limit_respected(self, tool_context: ToolContext) -> None:
        """Sub-agent respects its iteration limit."""
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))

        # Return tool calls forever — the iteration limit should stop it
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_tool_response("shell", {"command": "echo loop"}))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
            iteration_limit=3,
        )

        result = await coordinator.spawn("Loop forever")
        assert "maximum iterations" in result.lower()
        # Should have been called exactly 3 times (iteration limit)
        assert client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_failure_doesnt_crash(self, tool_context: ToolContext) -> None:
        """Sub-agent failure returns error string, doesn't crash parent."""
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Risky task")
        # Should return error message, not raise
        assert "error" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_spawn_parallel(self, tool_context: ToolContext) -> None:
        """Parallel spawning runs tasks concurrently."""
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Parallel task done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        results = await coordinator.spawn_parallel(["Task 1", "Task 2", "Task 3"])
        assert len(results) == 3
        assert all("done" in r.lower() for r in results)

    @pytest.mark.asyncio
    async def test_spawn_parallel_depth_limit(self, tool_context: ToolContext) -> None:
        """Parallel spawn at max depth returns error for all tasks."""
        registry = ToolRegistry()
        client = LLMClient(model="test")

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
            max_depth=1,
        )

        results = await coordinator.spawn_parallel(["A", "B"], depth=1)
        assert len(results) == 2
        assert all("depth" in r.lower() for r in results)


class TestSpawnAgentTool:
    """Test the SpawnAgentTool wrapper."""

    def test_metadata(self) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)
        assert tool.name == "spawn_agent"
        assert tool.risk_level == "high"
        schema = tool.get_schema()
        assert "task" in schema["properties"]
        assert schema["required"] == ["task"]

    def test_description(self) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)
        assert isinstance(tool.description, str)
        assert "sub-agent" in tool.description.lower() or "sub-task" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_execute_delegates_to_coordinator(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Sub-agent result."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute({"task": "Do something"}, tool_context)
        assert not result.is_error
        assert "Sub-agent result" in result.output

    @pytest.mark.asyncio
    async def test_execute_no_task(self, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)
        result = await tool.execute({}, tool_context)
        assert result.is_error
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_default_max_depth(self) -> None:
        assert MAX_SUB_AGENT_DEPTH == 3


class TestSpawnRetrieval:
    """Test spawn_retrieval for the retrieval sub-agent."""

    @pytest.mark.asyncio
    async def test_spawn_retrieval_returns_result(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=_make_text_response("file:src/main.py:10-20 -- Processor class")
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn_retrieval("Find Processor")
        assert "Processor" in result

    @pytest.mark.asyncio
    async def test_spawn_retrieval_failure(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("Retrieval failed"))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn_retrieval("Find stuff")
        assert "error" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_spawn_retrieval_max_iterations(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="grep_search", result=ToolResult.success("hit")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=_make_tool_response("grep_search", {"pattern": "TODO"})
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
            iteration_limit=2,
        )

        result = await coordinator.spawn_retrieval("Search")
        assert "maximum" in result.lower() or "iterations" in result.lower()


class TestRetrievalSubAgentTool:
    """Test the RetrievalSubAgentTool from retrieval_agent.py."""

    @pytest.mark.asyncio
    async def test_execute_delegates_to_spawn_retrieval(self, tool_context: ToolContext) -> None:
        from godspeed.agent.retrieval_agent import RetrievalSubAgentTool

        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=_make_text_response("file:src/auth.py:145-167 -- AuthManager")
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )
        tool = RetrievalSubAgentTool(coordinator)

        result = await tool.execute({"query": "find auth"}, tool_context)
        assert not result.is_error
        assert "AuthManager" in result.output

    @pytest.mark.asyncio
    async def test_execute_no_query(self, tool_context: ToolContext) -> None:
        from godspeed.agent.retrieval_agent import RetrievalSubAgentTool

        coordinator = AgentCoordinator(
            llm_client=LLMClient(model="test"),
            tool_registry=ToolRegistry(),
            tool_context=tool_context,
        )
        tool = RetrievalSubAgentTool(coordinator)

        result = await tool.execute({}, tool_context)
        assert result.is_error
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, tool_context: ToolContext) -> None:
        from godspeed.agent.retrieval_agent import RetrievalSubAgentTool

        coordinator = AgentCoordinator(
            llm_client=LLMClient(model="test"),
            tool_registry=ToolRegistry(),
            tool_context=tool_context,
        )
        tool = RetrievalSubAgentTool(coordinator)

        result = await tool.execute({"query": ""}, tool_context)
        assert result.is_error
        assert "required" in result.error.lower()

    def test_metadata(self) -> None:
        from godspeed.agent.retrieval_agent import RetrievalSubAgentTool

        coordinator = AsyncMock()
        tool = RetrievalSubAgentTool(coordinator)
        assert tool.name == "retrieval"
        assert tool.risk_level == "read_only"
        schema = tool.get_schema()
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_description(self) -> None:
        from godspeed.agent.retrieval_agent import RetrievalSubAgentTool

        coordinator = AsyncMock()
        tool = RetrievalSubAgentTool(coordinator)
        assert isinstance(tool.description, str)
        assert "retrieval" in tool.description.lower()


class TestSpawnWithTimeout:
    """Test sub-agent delegation with timeout."""

    @pytest.mark.asyncio
    async def test_spawn_error_propagates(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=TimeoutError("timed out"))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("timeout task")
        assert "error" in result.lower() or "Time" in result

    @pytest.mark.asyncio
    async def test_spawn_value_error_propagates(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=ValueError("bad input"))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("bad task")
        assert "error" in result.lower()


class TestSpawnRetrievalFailureTypes:
    """Test spawn_retrieval with various exception types."""

    @pytest.mark.asyncio
    async def test_retrieval_value_error(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=ValueError("bad retrieval input"))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn_retrieval("bad query")
        assert "error" in result.lower()


class TestSpawnCapsAndStagger:
    @pytest.mark.asyncio
    async def test_spawn_cap_enforced(self, tool_context: ToolContext) -> None:

        coord = AgentCoordinator(
            llm_client=LLMClient(model="test"),
            tool_registry=ToolRegistry(),
            tool_context=tool_context,
            max_spawns=0,
        )
        result = await coord.spawn("task")
        assert "cap (0) reached" in result
        assert coord._spawn_count == 1

    @pytest.mark.asyncio
    async def test_spawn_parallel_staggers_siblings(self, tool_context: ToolContext) -> None:
        import time as _time

        coord = AgentCoordinator(
            llm_client=LLMClient(model="test"),
            tool_registry=ToolRegistry(),
            tool_context=tool_context,
            stagger_seconds=0.05,
        )
        coord.spawn = AsyncMock(return_value="ok")
        t0 = _time.monotonic()
        results = await coord.spawn_parallel(["a", "b", "c"])
        elapsed = _time.monotonic() - t0
        assert results == ["ok", "ok", "ok"]
        assert elapsed >= 0.08
