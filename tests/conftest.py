"""Shared test fixtures for Godspeed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from godspeed.config import GodspeedSettings
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult


@pytest.fixture(autouse=True)
def _ensure_logging_enabled() -> None:
    """Re-enable logging in case a prior test disabled it.

    Also resets the "godspeed" logger's own level: tests that exercise
    ``cli._setup_logging()`` (e.g. ``test_cli_more.py::TestSetupLogging``)
    set ``logging.getLogger("godspeed").level`` as a side effect and never
    reset it, since Python loggers are global singletons that persist for
    the whole test session. Left alone, that leaks into every other test
    under the ``godspeed.*`` namespace that runs afterward — any assertion
    on INFO/DEBUG log output (e.g. via ``caplog``) would silently see
    nothing, not because logging failed, but because an unrelated earlier
    test raised the effective level to WARNING.
    """
    logging.disable(logging.NOTSET)
    if not logging.root.handlers:
        logging.root.addHandler(logging.StreamHandler())
    logging.root.setLevel(logging.NOTSET)
    logging.getLogger("godspeed").setLevel(logging.NOTSET)


class MockTool(Tool):
    """A minimal tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        risk_level: RiskLevel = RiskLevel.READ_ONLY,
        result: ToolResult | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._risk_level = risk_level
        self._result = result or ToolResult.success("mock output")
        self.last_arguments: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk_level

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"},
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.last_arguments = arguments
        return self._result


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with .godspeed/."""
    godspeed_dir = tmp_path / ".godspeed"
    godspeed_dir.mkdir()
    return tmp_path


@pytest.fixture
def tool_context(tmp_project: Path) -> ToolContext:
    """Create a ToolContext for testing."""
    return ToolContext(cwd=tmp_project, session_id="test-session-001")


@pytest.fixture
def settings(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> GodspeedSettings:
    """Create GodspeedSettings with isolated config (no real config files)."""
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_project / ".godspeed-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_project / ".godspeed")
    return GodspeedSettings(project_dir=tmp_project)
