"""Tests for /compact, /context breakdown, and statusline template features."""

from __future__ import annotations

import asyncio
import re
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.config import GodspeedSettings, StatuslineSettings
from godspeed.context.compaction import compact_now
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands
from godspeed.tui.output import (
    DEFAULT_STATUSLINE_TEMPLATE,
    _render_statusline_template,
    format_status_hud,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(fn, *args, **kwargs) -> str:
    """Run a formatting function and capture its Rich console output (ANSI stripped)."""
    buf = StringIO()
    original = _output.console
    _output.console = _output.Console(file=buf, force_terminal=True, width=120)
    try:
        fn(*args, **kwargs)
    finally:
        _output.console = original
    return _ANSI_RE.sub("", buf.getvalue())


@pytest.fixture
def conversation() -> Conversation:
    return Conversation("You are a coding agent.", max_tokens=100_000)


@pytest.fixture
def commands(conversation: Conversation, tmp_path: Path) -> Commands:
    llm_client = MagicMock()
    llm_client.model = "test-model"
    llm_client.fallback_models = []
    llm_client.total_input_tokens = 0
    llm_client.total_output_tokens = 0
    return Commands(
        conversation=conversation,
        llm_client=llm_client,
        permission_engine=None,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
        tool_registry=None,
    )


def _fill_conversation(conversation: Conversation, count: int) -> None:
    """Add *count* user/assistant message pairs to the conversation."""
    for i in range(count):
        conversation.add_user_message(f"user message {i}")
        conversation.add_assistant_message(f"assistant reply {i}")


class TestCompactCommand:
    """Test /compact command behavior."""

    def test_compact_under_threshold_shows_nothing(self, commands: Commands) -> None:
        _fill_conversation(commands._conversation, 3)  # 6 messages + system = 7
        output = _capture(commands.dispatch, "/compact")
        assert "Nothing to compact" in output

    async def test_compact_registered(self, commands: Commands) -> None:
        _fill_conversation(commands._conversation, 10)
        result = commands.dispatch("/compact")
        assert result is not None
        assert result.handled
        await asyncio.sleep(0)  # let the background compaction task settle

    async def test_compact_now_passes_instructions(self) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        _fill_conversation(conversation, 5)

        llm_client = MagicMock()
        llm_client.model = "test-model"
        response = MagicMock()
        response.content = "Summary of the work."
        llm_client.chat = AsyncMock(return_value=response)

        result = await compact_now(
            conversation,
            llm_client,
            instructions="preserve all file paths",
        )

        assert result.applied
        assert result.messages_before == 10
        assert result.messages_after == 1
        # Instructions must be passed through into the summary prompt.
        call_args = llm_client.chat.call_args.kwargs["messages"]
        user_content = call_args[1]["content"]
        assert "preserve all file paths" in user_content
        # The summary replaced the history.
        assert "Summary of the work." in conversation.messages[-1]["content"]

    async def test_compact_now_without_instructions(self) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        _fill_conversation(conversation, 5)

        llm_client = MagicMock()
        llm_client.model = "test-model"
        response = MagicMock()
        response.content = "Summary."
        llm_client.chat = AsyncMock(return_value=response)

        result = await compact_now(conversation, llm_client)

        assert result.applied
        call_args = llm_client.chat.call_args.kwargs["messages"]
        user_content = call_args[1]["content"]
        assert "[User instructions for summary:" not in user_content

    async def test_compact_now_failure_reports_not_applied(self) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        _fill_conversation(conversation, 5)

        llm_client = MagicMock()
        llm_client.model = "test-model"
        llm_client.chat = MagicMock(side_effect=RuntimeError("boom"))

        result = await compact_now(conversation, llm_client)

        assert not result.applied
        assert result.messages_before == 10


class TestVerifyCommand:
    """Test /verify command behavior."""

    async def test_verify_registered(self, commands: Commands) -> None:
        result = commands.dispatch("/verify")
        assert result is not None
        assert result.handled
        await asyncio.sleep(0)  # let the background verification task settle

    async def test_verify_with_instructions(self, commands: Commands) -> None:
        result = commands.dispatch("/verify check the server starts")
        assert result is not None
        assert result.handled
        await asyncio.sleep(0)

    def test_verify_in_help(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/verify" in output


class TestContextBreakdown:
    """Test /context detailed token breakdown."""

    def test_context_shows_four_rows(self, commands: Commands) -> None:
        _fill_conversation(commands._conversation, 3)
        output = _capture(commands.dispatch, "/context")
        assert "System prompt" in output
        assert "Tool schemas" in output
        assert "Conversation" in output
        assert "Free space" in output

    def test_context_keeps_single_line_usage(self, commands: Commands) -> None:
        _fill_conversation(commands._conversation, 3)
        output = _capture(commands.dispatch, "/context")
        assert "tokens:" in output
        assert "messages:" in output

    def test_context_empty_conversation(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/context")
        assert "System prompt" in output
        assert "Free space" in output


class TestStatuslineTemplate:
    """Test statusline template substitution and fallback."""

    def test_template_substitution(self) -> None:
        output = _capture(
            format_status_hud,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0024,
            model="anthropic/claude-opus",
            turns=3,
            template="{model} | {tokens} | {cost} | {branch}",
        )
        assert "claude-opus" in output
        assert "1,500" in output
        assert "$0.0024" in output

    def test_default_template_constant(self) -> None:
        assert "{model}" in DEFAULT_STATUSLINE_TEMPLATE
        assert "{tokens}" in DEFAULT_STATUSLINE_TEMPLATE
        assert "{cost}" in DEFAULT_STATUSLINE_TEMPLATE
        assert "{branch}" in DEFAULT_STATUSLINE_TEMPLATE

    def test_invalid_template_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="godspeed.tui.output"):
            output = _capture(
                format_status_hud,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0,
                model="m",
                turns=1,
                template="{unknown_placeholder}",
            )
        assert "tokens" in output  # default HUD rendered
        assert any("Invalid statusline template" in r.message for r in caplog.records)

    def test_render_returns_none_for_bad_template(self) -> None:
        assert _render_statusline_template("{nope}", 1, 2, 0.0, "m") is None

    def test_render_substitutes_values(self) -> None:
        rendered = _render_statusline_template(
            "{model}:{tokens}:{cost}", 1000, 500, 0.01, "openai/gpt-4o"
        )
        assert rendered == "gpt-4o:1,500:$0.01"


class TestStatuslineConfig:
    """Test statusline config round-trip."""

    def test_config_round_trip(self) -> None:
        settings = GodspeedSettings(statusline={"enabled": True, "template": "{model} | {tokens}"})
        assert settings.statusline.enabled is True
        assert settings.statusline.template == "{model} | {tokens}"

    def test_config_defaults_disabled(self) -> None:
        settings = GodspeedSettings()
        assert settings.statusline.enabled is False
        assert settings.statusline.template == ""

    def test_statusline_settings_direct(self) -> None:
        sl = StatuslineSettings(enabled=True, template="{cost}")
        assert sl.enabled is True
        assert sl.template == "{cost}"
