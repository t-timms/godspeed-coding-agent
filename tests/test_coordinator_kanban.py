"""Tests for SubAgentConfig, per-agent model routing, and KanbanSwarm."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.agent.coordinator import (
    AgentCoordinator,
    CapabilityBundle,
    KanbanPlan,
    KanbanSwarm,
    SpawnAgentTool,
    SpawnKanbanTool,
    SubAgentConfig,
    WorkItem,
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


class TestSubAgentConfig:
    """Test SubAgentConfig dataclass."""

    def test_defaults(self) -> None:
        config = SubAgentConfig()
        assert config.model is None
        assert config.effort == "normal"
        assert config.max_iterations == 0
        assert config.tool_bundle is None
        assert config.system_prompt is None

    def test_iteration_limit_normal(self) -> None:
        config = SubAgentConfig(effort="normal")
        assert config.iteration_limit == 25

    def test_iteration_limit_low(self) -> None:
        config = SubAgentConfig(effort="low")
        assert config.iteration_limit == 10

    def test_iteration_limit_high(self) -> None:
        config = SubAgentConfig(effort="high")
        assert config.iteration_limit == 40

    def test_iteration_limit_unknown_falls_back(self) -> None:
        config = SubAgentConfig(effort="extreme")
        assert config.iteration_limit == 25

    def test_frozen(self) -> None:
        config = SubAgentConfig()
        with pytest.raises(AttributeError):
            config.model = "new-model"  # type: ignore[misc]


class TestAgentCoordinatorWithConfig:
    """Test per-subagent model routing."""

    @pytest.mark.asyncio
    async def test_spawn_with_model_override(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        mock_new_client = LLMClient(model="claude-opus-4")
        mock_new_client.chat = AsyncMock(return_value=_make_text_response("Done."))

        with patch.object(
            type(coordinator), "_build_llm_client_for_config", return_value=mock_new_client
        ):
            config = SubAgentConfig(model="claude-opus-4", effort="high")
            result = await coordinator.spawn("Do something", depth=0, config=config)
            assert result == "Done."

    @pytest.mark.asyncio
    async def test_spawn_uses_default_config_when_none(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Default."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Task", depth=0, config=None)
        assert result == "Default."

    @pytest.mark.asyncio
    async def test_spawn_parallel_with_configs(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Parallel done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        mock_client_a = LLMClient(model="model-a")
        mock_client_a.chat = AsyncMock(return_value=_make_text_response("Parallel done."))
        mock_client_b = LLMClient(model="model-b")
        mock_client_b.chat = AsyncMock(return_value=_make_text_response("Parallel done."))

        call_count = 0

        def build_side_effect(config):
            nonlocal call_count
            call_count += 1
            c = LLMClient(model=config.model or "test")
            c.chat = AsyncMock(return_value=_make_text_response("Parallel done."))
            return c

        with patch.object(
            type(coordinator), "_build_llm_client_for_config", side_effect=build_side_effect
        ):
            configs = [
                SubAgentConfig(model="model-a", effort="low"),
                SubAgentConfig(model="model-b", effort="high"),
            ]
            results = await coordinator.spawn_parallel(
                ["Task 1", "Task 2"], depth=0, configs=configs
            )
            assert len(results) == 2
            assert all("done" in r.lower() for r in results)

    @pytest.mark.asyncio
    async def test_spawn_parallel_configs_none(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("OK."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        results = await coordinator.spawn_parallel(["A", "B"], depth=0, configs=None)
        assert len(results) == 2

    def test_build_llm_client_for_config_none_model(self) -> None:
        from pathlib import Path

        registry = ToolRegistry()
        client = LLMClient(model="test")
        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=ToolContext(cwd=Path("."), session_id="s"),
        )
        config = SubAgentConfig()
        result = coordinator._build_llm_client_for_config(config)
        assert result is client

    @pytest.mark.asyncio
    async def test_spawn_with_tool_bundle(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("content")))
        registry.register(MockTool(name="complexity", result=ToolResult.success("5")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        config = SubAgentConfig(tool_bundle=CapabilityBundle.CORE)
        result = await coordinator.spawn("Read a file", depth=0, config=config)
        assert "Done" in result


class TestSpawnAgentToolWithModel:
    """Test SpawnAgentTool with per-agent model override."""

    def test_schema_includes_model_and_effort(self) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)
        schema = tool.get_schema()
        assert "model" in schema["properties"]
        assert "effort" in schema["properties"]
        assert schema["required"] == ["task"]

    @pytest.mark.asyncio
    async def test_execute_with_model_override(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Sub-agent result."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        mock_new_client = LLMClient(model="gpt-4o")
        mock_new_client.chat = AsyncMock(return_value=_make_text_response("Sub-agent result."))

        with patch.object(
            type(coordinator), "_build_llm_client_for_config", return_value=mock_new_client
        ):
            tool = SpawnAgentTool(coordinator)
            result = await tool.execute(
                {"task": "Do something", "model": "gpt-4o", "effort": "high"},
                tool_context,
            )
            assert not result.is_error
            assert "Sub-agent result" in result.output

    @pytest.mark.asyncio
    async def test_execute_no_model_uses_default(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Default."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute({"task": "Do something"}, tool_context)
        assert not result.is_error


class TestWorkItem:
    """Test WorkItem dataclass."""

    def test_defaults(self) -> None:
        item = WorkItem(id="w1", description="Do thing")
        assert item.files == []
        assert item.deps == []
        assert item.acceptance == []
        assert item.status == "pending"


class TestKanbanPlan:
    """Test KanbanPlan dataclass."""

    def test_done_when_all_verified(self) -> None:
        items = [
            WorkItem(id="a", description="A", status="verified"),
            WorkItem(id="b", description="B", status="verified"),
        ]
        plan = KanbanPlan(objective="test", items=items)
        assert plan.done is True

    def test_not_done_when_pending(self) -> None:
        items = [
            WorkItem(id="a", description="A", status="verified"),
            WorkItem(id="b", description="B", status="pending"),
        ]
        plan = KanbanPlan(objective="test", items=items)
        assert plan.done is False


class TestKanbanSwarm:
    """Test KanbanSwarm orchestration."""

    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")

        responses = [
            _make_text_response("Worker done: modified auth.py"),
            _make_text_response("VERIFIED: item-1"),
        ]
        client.chat = AsyncMock(side_effect=responses)

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        swarm = KanbanSwarm(coordinator)
        plan = KanbanPlan(
            objective="Fix auth bug",
            items=[
                WorkItem(
                    id="item-1",
                    description="Fix the auth module",
                    files=["src/auth.py"],
                    acceptance=["Tests pass"],
                ),
            ],
        )

        results = await swarm.execute(plan)
        assert "item-1" in results
        assert plan.done

    @pytest.mark.asyncio
    async def test_execute_plan_with_deps(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")

        responses = [
            _make_text_response("Worker item-1 done"),
            _make_text_response("VERIFIED: item-1"),
            _make_text_response("Worker item-2 done"),
            _make_text_response("VERIFIED: item-2"),
        ]
        client.chat = AsyncMock(side_effect=responses)

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        swarm = KanbanSwarm(coordinator)
        plan = KanbanPlan(
            objective="Refactor",
            items=[
                WorkItem(id="item-1", description="Create interface"),
                WorkItem(id="item-2", description="Implement interface", deps=["item-1"]),
            ],
        )

        results = await swarm.execute(plan)
        assert "item-1" in results
        assert "item-2" in results
        assert plan.done

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock()

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        swarm = KanbanSwarm(coordinator)
        plan = KanbanPlan(objective="Nothing to do", items=[])
        results = await swarm.execute(plan)
        assert results == {}

    @pytest.mark.asyncio
    async def test_worker_model_passed(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_text_response("Worker done"),
                _make_text_response("VERIFIED: w1"),
            ]
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        def build_side_effect(config):
            c = LLMClient(model=config.model or "test")
            c.chat = AsyncMock(
                side_effect=[
                    _make_text_response("Worker done"),
                    _make_text_response("VERIFIED: w1"),
                ]
            )
            return c

        with patch.object(
            type(coordinator),
            "_build_llm_client_for_config",
            side_effect=build_side_effect,
        ):
            swarm = KanbanSwarm(coordinator, worker_model="gpt-4o", verifier_model="claude-haiku")
            plan = KanbanPlan(
                objective="Test",
                items=[WorkItem(id="w1", description="Work")],
            )

            results = await swarm.execute(plan)
            assert "w1" in results


class TestSpawnKanbanTool:
    """Test SpawnKanbanTool."""

    def test_metadata(self) -> None:
        coordinator = AsyncMock()
        tool = SpawnKanbanTool(coordinator)
        assert tool.name == "kanban_swarm"
        assert tool.risk_level == "high"

    def test_schema(self) -> None:
        coordinator = AsyncMock()
        tool = SpawnKanbanTool(coordinator)
        schema = tool.get_schema()
        assert "objective" in schema["properties"]
        assert "items" in schema["properties"]
        assert "worker_model" in schema["properties"]
        assert schema["required"] == ["objective", "items"]

    @pytest.mark.asyncio
    async def test_execute_no_objective(self, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        tool = SpawnKanbanTool(coordinator)
        result = await tool.execute({}, tool_context)
        assert result.is_error
        assert "objective" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_no_items(self, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        tool = SpawnKanbanTool(coordinator)
        result = await tool.execute({"objective": "Do stuff", "items": []}, tool_context)
        assert result.is_error
        assert "work item" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_delegates_to_swarm(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_text_response("Worker done"),
                _make_text_response("VERIFIED: w1"),
            ]
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )
        tool = SpawnKanbanTool(coordinator)

        result = await tool.execute(
            {
                "objective": "Fix bug",
                "items": [{"id": "w1", "description": "Fix it"}],
            },
            tool_context,
        )
        assert not result.is_error
        assert "w1" in result.output
