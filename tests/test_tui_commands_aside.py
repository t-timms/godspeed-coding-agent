"""Tests for /btw, /goal, and /rewind TUI slash commands."""

from __future__ import annotations

import asyncio
import re
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.agent.aside import (
    BTW_MAX_TOKENS,
    build_btw_messages,
    snapshot_messages,
    verify_conversation_unchanged,
)
from godspeed.agent.conversation import Conversation
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands
from godspeed.tui.output import format_status_hud
from godspeed.tui.rewind import (
    RESTORE_BOTH,
    RESTORE_CONVERSATION,
    RESTORE_FILES,
    RESTORE_NONE,
    RewindEntry,
    parse_rewind_choice,
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


class TestBtwBuildMessages:
    """Test the pure aside helper for building btw message lists."""

    def test_build_btw_messages_appends_question(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = build_btw_messages(messages, "what is 2+2?")
        assert len(result) == 3
        assert result[-1] == {"role": "user", "content": "what is 2+2?"}
        # Original untouched
        assert len(messages) == 2

    def test_build_btw_messages_does_not_mutate_original(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        original = list(messages)
        build_btw_messages(messages, "question")
        assert messages == original

    def test_build_btw_messages_deep_copies(self) -> None:
        messages = [{"role": "user", "content": "hi", "extra": {"nested": [1, 2]}}]
        result = build_btw_messages(messages, "q")
        # Mutating the copy must not affect the original
        result[0]["extra"]["nested"].append(3)
        assert messages[0]["extra"]["nested"] == [1, 2]

    def test_build_btw_messages_empty_question_raises(self) -> None:
        with pytest.raises(ValueError):
            build_btw_messages([{"role": "system", "content": "s"}], "   ")

    def test_snapshot_and_verify(self) -> None:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        snap = snapshot_messages(messages)
        assert verify_conversation_unchanged(snap, messages)
        # Mutating the snapshot must not affect the original
        snap.append({"role": "user", "content": "extra"})
        assert not verify_conversation_unchanged(snap, messages)

    def test_btw_max_tokens_constant(self) -> None:
        assert BTW_MAX_TOKENS > 0


class TestBtwCommand:
    """Test /btw command behavior."""

    async def test_btw_empty_question_hint(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/btw")
        assert "Usage: /btw" in output

    async def test_btw_answers_and_leaves_conversation_unchanged(
        self, commands: Commands, conversation: Conversation
    ) -> None:
        conversation.add_user_message("main question")
        conversation.add_assistant_message("main answer")
        before = [dict(m) for m in conversation.messages]

        response = MagicMock()
        response.content = "The answer is 42."
        commands._llm_client.chat = AsyncMock(return_value=response)

        result = commands.dispatch("/btw what is the meaning of life?")
        assert result is not None
        assert result.handled
        await asyncio.sleep(0)  # let the background btw task settle

        # Main conversation byte-identical
        assert conversation.messages == before
        # LLM called with the copied message list (question appended)
        call_args = commands._llm_client.chat.call_args.kwargs["messages"]
        assert call_args[-1] == {"role": "user", "content": "what is the meaning of life?"}
        assert len(call_args) == len(before) + 1

    async def test_btw_llm_failure_restores_state(
        self, commands: Commands, conversation: Conversation
    ) -> None:
        conversation.add_user_message("main")
        before = [dict(m) for m in conversation.messages]

        commands._llm_client.chat = AsyncMock(side_effect=RuntimeError("boom"))

        result = commands.dispatch("/btw some question")
        assert result is not None
        assert result.handled
        await asyncio.sleep(0)

        # Conversation unchanged even on failure
        assert conversation.messages == before

    def test_btw_in_help(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/btw" in output


class TestGoalCommand:
    """Test /goal command behavior."""

    def test_goal_set(self, commands: Commands) -> None:
        result = commands.dispatch("/goal fix the login bug")
        assert result is not None
        assert result.handled
        assert commands._session_goal == "fix the login bug"

    def test_goal_show(self, commands: Commands) -> None:
        commands.dispatch("/goal my goal")
        output = _capture(commands.dispatch, "/goal")
        assert "my goal" in output

    def test_goal_show_empty(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/goal")
        assert "No session goal" in output

    def test_goal_clear(self, commands: Commands) -> None:
        commands.dispatch("/goal some goal")
        result = commands.dispatch("/goal clear")
        assert result is not None
        assert result.handled
        assert commands._session_goal == ""

    def test_goal_in_help(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/goal" in output


class TestGoalStatusHud:
    """Test goal appears in status HUD output."""

    def test_goal_appears_in_hud(self) -> None:
        output = _capture(
            format_status_hud,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0,
            model="m",
            turns=1,
            goal="fix login",
        )
        assert "goal: fix login" in output

    def test_no_goal_when_empty(self) -> None:
        output = _capture(
            format_status_hud,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0,
            model="m",
            turns=1,
            goal="",
        )
        assert "goal:" not in output


class TestRewindCommand:
    """Test /rewind command behavior."""

    def test_rewind_no_entries(self, commands: Commands) -> None:
        with patch("godspeed.tui.rewind.collect_rewind_entries", return_value=[]):
            output = _capture(commands.dispatch, "/rewind")
        assert "No rewind checkpoints" in output

    def test_rewind_lists_entries(self, commands: Commands) -> None:
        entries = [
            RewindEntry(kind="conversation", name="cp1", detail="3 messages"),
            RewindEntry(kind="files", name="app.py", detail="snapshot: x"),
        ]
        with (
            patch("godspeed.tui.rewind.collect_rewind_entries", return_value=entries),
            patch("rich.console.Console.input", side_effect=["1", "n"]),
        ):
            output = _capture(commands.dispatch, "/rewind")
        assert "Rewind Checkpoints" in output
        assert "cp1" in output
        assert "app.py" in output

    def test_rewind_invalid_selection(self, commands: Commands) -> None:
        entries = [RewindEntry(kind="conversation", name="cp1", detail="d")]
        with patch("godspeed.tui.rewind.collect_rewind_entries", return_value=entries):
            output = _capture(commands.dispatch, "/rewind abc")
        assert "Rewind cancelled" in output

    def test_rewind_out_of_range(self, commands: Commands) -> None:
        entries = [RewindEntry(kind="conversation", name="cp1", detail="d")]
        with patch("godspeed.tui.rewind.collect_rewind_entries", return_value=entries):
            output = _capture(commands.dispatch, "/rewind 5")
        assert "Rewind cancelled" in output

    def test_rewind_restores_conversation(
        self, commands: Commands, conversation: Conversation
    ) -> None:
        entries = [RewindEntry(kind="conversation", name="cp1", detail="d")]
        with (
            patch("godspeed.tui.rewind.collect_rewind_entries", return_value=entries),
            patch("rich.console.Console.input", return_value="c"),
            patch(
                "godspeed.tui.rewind.restore_conversation",
                return_value="Restored conversation checkpoint [cp1]",
            ) as mock_restore,
        ):
            output = _capture(commands.dispatch, "/rewind 1")
        assert "Restored conversation checkpoint" in output
        mock_restore.assert_called_once_with(conversation, "cp1", commands._cwd)

    def test_rewind_restores_files(self, commands: Commands) -> None:
        entries = [RewindEntry(kind="files", name="app.py", detail="d")]
        with (
            patch("godspeed.tui.rewind.collect_rewind_entries", return_value=entries),
            patch("rich.console.Console.input", return_value="f"),
            patch(
                "godspeed.tui.rewind.restore_files",
                return_value="Restored 1 file(s): app.py",
            ) as mock_restore,
        ):
            output = _capture(commands.dispatch, "/rewind 1")
        assert "Restored 1 file(s)" in output
        mock_restore.assert_called_once_with(commands._cwd, commands._session_id)

    def test_rewind_conversation_mode_on_file_entry_warns(self, commands: Commands) -> None:
        entries = [RewindEntry(kind="files", name="app.py", detail="d")]
        with (
            patch("godspeed.tui.rewind.collect_rewind_entries", return_value=entries),
            patch("rich.console.Console.input", return_value="c"),
        ):
            output = _capture(commands.dispatch, "/rewind 1")
        assert "no conversation to restore" in output

    def test_rewind_in_help(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/rewind" in output


class TestParseRewindChoice:
    """Test rewind restore-mode parsing maps to functions."""

    def test_parse_conversation(self) -> None:
        assert parse_rewind_choice("c") == RESTORE_CONVERSATION
        assert parse_rewind_choice("conversation") == RESTORE_CONVERSATION

    def test_parse_files(self) -> None:
        assert parse_rewind_choice("f") == RESTORE_FILES
        assert parse_rewind_choice("files") == RESTORE_FILES

    def test_parse_both(self) -> None:
        assert parse_rewind_choice("b") == RESTORE_BOTH
        assert parse_rewind_choice("both") == RESTORE_BOTH

    def test_parse_none(self) -> None:
        assert parse_rewind_choice("n") == RESTORE_NONE
        assert parse_rewind_choice("q") == RESTORE_NONE
        assert parse_rewind_choice("") == RESTORE_NONE

    def test_parse_unknown(self) -> None:
        assert parse_rewind_choice("x") == RESTORE_NONE
