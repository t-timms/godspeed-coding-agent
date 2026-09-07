"""Tests for the /fork session-duplication command."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(fn, *args, **kwargs) -> str:
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
    conv = Conversation("You are a coding agent.", max_tokens=100_000)
    conv.add_user_message("hello")
    return conv


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
        session_id="sess-base01",
        cwd=tmp_path,
        tool_registry=None,
    )


class TestFork:
    def test_missing_memory_graceful(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/fork")
        assert "Session memory" in output

    def test_fork_registers_and_saves_messages(self, commands: Commands) -> None:
        memory = MagicMock()
        commands._session_memory = memory
        _capture(commands.dispatch, "/fork")

        fork_id = memory.start_session.call_args[0][0]
        assert fork_id.startswith("sess-base01-fork-")
        memory.start_session.assert_called_once()
        saved = memory.save_messages.call_args[0]
        assert saved[0] == fork_id
        assert len(saved[1]) == len(commands._conversation.messages)
        # Live conversation is untouched (fork only copies).
        assert len(commands._conversation.messages) == 2

    def test_fork_label_in_summary(self, commands: Commands) -> None:
        memory = MagicMock()
        commands._session_memory = memory
        _capture(commands.dispatch, "/fork experiment A")
        summary = memory.end_session.call_args[1]["summary"]
        assert "experiment A" in summary
        assert "sess-base01" in summary

    def test_fork_help_entry(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/fork" in output
