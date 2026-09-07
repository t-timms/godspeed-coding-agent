"""Tests for capability bundles and per-subagent tool filtering.

Bundles are defined in ``godspeed.agent.coordinator`` and applied by
``AgentCoordinator._filter_registry_for_bundle`` when spawning sub-agents.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from godspeed.agent.coordinator import (
    _BUNDLE_ALLOWED,
    CapabilityBundle,
    SubAgentConfig,
    AgentCoordinator,
)
from godspeed.tools.base import Tool, ToolContext
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


ALL_KNOWN_TOOLS = frozenset().union(*_BUNDLE_ALLOWED.values())


class TestCapabilityBundles:
    """Verify bundle definitions are correct and non-overlapping."""

    def test_all_bundles_have_known_tools(self) -> None:
        for bundle, allowed in _BUNDLE_ALLOWED.items():
            if bundle == CapabilityBundle.FULL:
                continue  # FULL is a passthrough sentinel, not a name list
            assert allowed, f"Bundle {bundle} must not be empty"

    def test_bundle_tools_are_non_empty_names(self) -> None:
        for allowed in _BUNDLE_ALLOWED.values():
            assert all(isinstance(name, str) and name for name in allowed)

    def test_full_bundle_is_passthrough(self) -> None:
        assert _BUNDLE_ALLOWED[CapabilityBundle.FULL] == set()

    def test_capability_bundle_enum_values(self) -> None:
        assert CapabilityBundle.CORE == "core"
        assert CapabilityBundle.READONLY == "readonly"
        assert CapabilityBundle.WRITE == "write"
        assert CapabilityBundle.FULL == "full"

    def test_core_bundle_contains_essential_tools(self) -> None:
        essentials = {"file_read", "file_write", "file_edit", "directory_list"}
        assert essentials.issubset(_BUNDLE_ALLOWED[CapabilityBundle.CORE])

    def test_readonly_bundle_has_no_write_tools(self) -> None:
        readonly = _BUNDLE_ALLOWED[CapabilityBundle.READONLY]
        assert "file_write" not in readonly
        assert "shell" not in readonly

    def test_write_bundle_superset_of_core(self) -> None:
        assert _BUNDLE_ALLOWED[CapabilityBundle.CORE].issubset(
            _BUNDLE_ALLOWED[CapabilityBundle.WRITE]
        )


def _make_coordinator(tools: list[Tool]) -> AgentCoordinator:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentCoordinator(
        llm_client=MagicMock(),
        tool_registry=registry,
        tool_context=MagicMock(spec=ToolContext),
    )


class TestBundleFiltering:
    """Verify _filter_registry_for_bundle returns a filtered subset view."""

    @pytest.fixture()
    def coordinator(self) -> AgentCoordinator:
        names = [
            "file_read",
            "file_write",
            "file_edit",
            "directory_list",
            "grep_search",
            "shell",
            "web_fetch",
        ]
        return _make_coordinator([MockTool(name=n) for n in names])

    def test_none_bundle_returns_full_registry(self, coordinator: AgentCoordinator) -> None:
        registry = coordinator._filter_registry_for_bundle(None)
        assert {t.name for t in registry.list_tools()} == {
            "file_read",
            "file_write",
            "file_edit",
            "directory_list",
            "grep_search",
            "shell",
            "web_fetch",
        }

    def test_full_bundle_returns_full_registry(self, coordinator: AgentCoordinator) -> None:
        registry = coordinator._filter_registry_for_bundle(CapabilityBundle.FULL)
        assert len(registry.list_tools()) == 7

    def test_readonly_bundle_filters_write_tools(self, coordinator: AgentCoordinator) -> None:
        registry = coordinator._filter_registry_for_bundle(CapabilityBundle.READONLY)
        names = {t.name for t in registry.list_tools()}
        assert "file_read" in names
        assert "file_write" not in names
        assert "shell" not in names

    def test_filtered_registry_is_new_instance(self, coordinator: AgentCoordinator) -> None:
        registry = coordinator._filter_registry_for_bundle(CapabilityBundle.CORE)
        assert registry is not coordinator._tool_registry

    def test_filtered_registry_preserves_sandbox(self, coordinator: AgentCoordinator) -> None:
        coordinator._tool_registry._sandbox = object()  # type: ignore[assignment]
        registry = coordinator._filter_registry_for_bundle(CapabilityBundle.CORE)
        assert registry._sandbox is coordinator._tool_registry._sandbox

    def test_bundle_config_defaults_to_none(self) -> None:
        assert SubAgentConfig().tool_bundle is None
