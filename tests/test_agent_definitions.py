"""Tests for file-defined sub-agents and spawn_agent resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

import godspeed.skills.agent_loader as agent_loader_mod
from godspeed.agent.coordinator import CapabilityBundle, SpawnAgentTool, SubAgentConfig
from godspeed.skills.agent_loader import (
    AGENT_NAME_MAX_CHARS,
    AgentDefinition,
    AgentDefinitionError,
    load_agent_definitions,
)
from godspeed.tools.base import ToolContext


@pytest.fixture(autouse=True)
def _isolate_user_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the user agent dir at a temp dir so tests never touch ~/.godspeed."""
    monkeypatch.setattr(agent_loader_mod, "USER_AGENT_DIR", tmp_path / "user-agents")


def _write_agent(
    base: Path,
    name: str,
    frontmatter: dict,
    body: str = "Do the thing.",
) -> Path:
    """Write ``{base}/.godspeed/agents/{name}.md`` with YAML frontmatter."""
    agents_dir = base / ".godspeed" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.md"
    text = f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n{body}\n"
    path.write_text(text, encoding="utf-8")
    return path


def _write_user_agent(base: Path, name: str, frontmatter: dict, body: str = "User.") -> Path:
    """Write directly into the (monkeypatched) user agent dir."""
    user_dir = base / "user-agents"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{name}.md"
    text = f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n{body}\n"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadAgentDefinitions:
    """Loading agent definition files from project and user scopes."""

    def test_load_full_frontmatter(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path,
            "researcher",
            {
                "name": "researcher",
                "description": "Deep research agent",
                "model": "gpt-4o",
                "effort": "high",
                "max_iterations": 15,
                "tool_bundle": "readonly",
                "system_prompt": "You are a researcher.",
                "max_cost_usd": 2.5,
            },
            body="Gather facts and cite sources.",
        )

        agent = load_agent_definitions(tmp_path)["researcher"]
        assert agent.description == "Deep research agent"
        assert agent.model == "gpt-4o"
        assert agent.effort == "high"
        assert agent.max_iterations == 15
        assert agent.tool_bundle == CapabilityBundle.READONLY
        assert agent.max_cost_usd == 2.5
        assert agent.system_prompt == (
            "You are a researcher.\n\n---\n\nGather facts and cite sources."
        )

    def test_load_minimal_name_only(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "helper", {"name": "helper"}, body="Just help.")

        agent = load_agent_definitions(tmp_path)["helper"]
        assert agent.description == ""
        assert agent.model is None
        assert agent.effort == "normal"
        assert agent.max_iterations == 0
        assert agent.tool_bundle is None
        assert agent.max_cost_usd is None
        assert agent.system_prompt == "Just help."

    def test_name_falls_back_to_file_stem(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "stemmed", {}, body="No name in frontmatter.")

        definitions = load_agent_definitions(tmp_path)
        assert "stemmed" in definitions

    def test_project_wins_over_user(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "shared", {"name": "shared", "effort": "low"})
        _write_user_agent(tmp_path, "shared", {"name": "shared", "effort": "high"})

        definitions = load_agent_definitions(tmp_path)
        assert definitions["shared"].effort == "low"

    def test_user_scope_loaded(self, tmp_path: Path) -> None:
        _write_user_agent(tmp_path, "global", {"name": "global"})

        definitions = load_agent_definitions(tmp_path)
        assert "global" in definitions

    def test_malformed_frontmatter_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        agents_dir = tmp_path / ".godspeed" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "bad.md").write_text("no frontmatter here\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="godspeed.skills.agent_loader"):
            definitions = load_agent_definitions(tmp_path)

        assert definitions == {}
        assert any("frontmatter" in r.message for r in caplog.records)

    def test_invalid_name_skipped(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "bad name", {"name": "bad name"})

        assert load_agent_definitions(tmp_path) == {}

    def test_name_too_long_skipped(self, tmp_path: Path) -> None:
        long_name = "a" * (AGENT_NAME_MAX_CHARS + 1)
        _write_agent(tmp_path, long_name, {"name": long_name})

        assert load_agent_definitions(tmp_path) == {}

    def test_invalid_tool_bundle_skipped(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "weird", {"name": "weird", "tool_bundle": "bogus"})

        assert load_agent_definitions(tmp_path) == {}

    def test_invalid_effort_skipped(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "lazy", {"name": "lazy", "effort": "extreme"})

        assert load_agent_definitions(tmp_path) == {}

    def test_invalid_max_iterations_skipped(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "loopy", {"name": "loopy", "max_iterations": -1})

        assert load_agent_definitions(tmp_path) == {}

    def test_reload_returns_fresh_definitions(self, tmp_path: Path) -> None:
        _write_agent(tmp_path, "one", {"name": "one"})
        first = load_agent_definitions(tmp_path)
        _write_agent(tmp_path, "two", {"name": "two"})
        second = load_agent_definitions(tmp_path)

        assert first is not second
        assert set(first) == {"one"}
        assert set(second) == {"one", "two"}


class TestAgentDefinition:
    """AgentDefinition validation and conversion."""

    def test_rejects_invalid_name(self) -> None:
        with pytest.raises(AgentDefinitionError):
            AgentDefinition(name="Bad Name")

    def test_rejects_too_long_name(self) -> None:
        with pytest.raises(AgentDefinitionError):
            AgentDefinition(name="a" * (AGENT_NAME_MAX_CHARS + 1))

    def test_to_config_returns_sub_agent_config(self, tmp_path: Path) -> None:
        _write_agent(
            tmp_path,
            "coder",
            {"name": "coder", "model": "claude-x", "effort": "high", "max_iterations": 20},
        )
        agent = load_agent_definitions(tmp_path)["coder"]

        config = agent.to_config()
        assert isinstance(config, SubAgentConfig)
        assert config.model == "claude-x"
        assert config.effort == "high"
        assert config.max_iterations == 20
        assert config.system_prompt == agent.system_prompt


class TestSpawnAgentToolResolution:
    """spawn_agent resolves file-defined agents by name."""

    def test_schema_includes_agent_name(self) -> None:
        tool = SpawnAgentTool(AsyncMock())
        schema = tool.get_schema()
        assert "agent_name" in schema["properties"]
        assert schema["required"] == ["task"]

    @pytest.mark.asyncio
    async def test_spawn_with_agent_name(self, tmp_path: Path, tool_context: ToolContext) -> None:
        _write_agent(
            tmp_path,
            "reviewer",
            {
                "name": "reviewer",
                "description": "Reviews code",
                "effort": "high",
                "tool_bundle": "readonly",
            },
        )
        coordinator = AsyncMock()
        coordinator.spawn.return_value = "ok"
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute(
            {"task": "Review main.py", "agent_name": "reviewer"}, tool_context
        )

        assert not result.is_error
        coordinator.spawn.assert_awaited_once()
        args, kwargs = coordinator.spawn.await_args
        assert args[0] == "Review main.py"
        assert kwargs["depth"] == 0
        config = kwargs["config"]
        assert isinstance(config, SubAgentConfig)
        assert config.effort == "high"
        assert config.tool_bundle == CapabilityBundle.READONLY

    @pytest.mark.asyncio
    async def test_spawn_unknown_agent(self, tmp_path: Path, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute({"task": "Do it", "agent_name": "nope"}, tool_context)

        assert result.is_error
        assert "Unknown agent" in result.error
        coordinator.spawn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_named_agent_overrides_ad_hoc(
        self, tmp_path: Path, tool_context: ToolContext
    ) -> None:
        _write_agent(tmp_path, "reviewer", {"name": "reviewer", "effort": "low"})
        coordinator = AsyncMock()
        coordinator.spawn.return_value = "ok"
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute(
            {"task": "Review", "agent_name": "reviewer", "model": "gpt-x", "effort": "high"},
            tool_context,
        )

        assert not result.is_error
        _, kwargs = coordinator.spawn.await_args
        config = kwargs["config"]
        assert config.effort == "low"
        assert config.model is None

    @pytest.mark.asyncio
    async def test_no_agent_name_keeps_ad_hoc(self, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        coordinator.spawn.return_value = "ok"
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute({"task": "Do it", "model": "gpt-x"}, tool_context)

        assert not result.is_error
        _, kwargs = coordinator.spawn.await_args
        config = kwargs["config"]
        assert config.model == "gpt-x"
        assert config.effort == "normal"
