"""Tests for session persistence + resume (--continue / --resume / --list-sessions).

Covers:
- resume-most-recent via ``--continue``
- resume-by-id via ``--resume <id>``
- unknown session id => exit code 5 (INVALID_INPUT)
- auto-generated summary population on ``end_session``
- bootstrap injection into the conversation (messages or summary marker)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.result import ExitCode
from godspeed.memory.session import SessionMemory


@pytest.fixture
def session_mem(tmp_path: Path) -> SessionMemory:
    mem = SessionMemory(db_path=tmp_path / "resume_test.db")
    yield mem
    mem.close()


def _settings_with_global_dir(tmp_path: Path) -> object:
    from godspeed.config import GodspeedSettings

    return GodspeedSettings(global_dir=tmp_path)


def _memory_at(tmp_path: Path) -> SessionMemory:
    """SessionMemory at the path _resolve_resume_context expects (global_dir/memory.db)."""
    return SessionMemory(db_path=tmp_path / "memory.db")


class TestResumeMostRecent:
    """`--continue` resolves the most recently started session."""

    def test_continue_returns_most_recent_session(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_resume_context

        mem = _memory_at(tmp_path)
        try:
            mem.start_session("old", "m1")
            mem.start_session("new", "m2")
            mem.save_messages("new", [{"role": "user", "content": "hello"}])
        finally:
            mem.close()

        settings = _settings_with_global_dir(tmp_path)
        ctx = _resolve_resume_context(continue_session=True, resume_session=None, settings=settings)

        assert ctx is not None
        assert ctx["session_id"] == "new"
        assert ctx["messages"] == [{"role": "user", "content": "hello"}]

    def test_continue_with_no_sessions_exits_5(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_resume_context

        settings = _settings_with_global_dir(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _resolve_resume_context(continue_session=True, resume_session=None, settings=settings)
        assert excinfo.value.code == int(ExitCode.INVALID_INPUT)

    def test_no_resume_flags_returns_none(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_resume_context

        settings = _settings_with_global_dir(tmp_path)
        assert (
            _resolve_resume_context(continue_session=False, resume_session=None, settings=settings)
            is None
        )


class TestResumeById:
    """`--resume <id>` resolves a specific session."""

    def test_resume_by_id_returns_session(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_resume_context

        mem = _memory_at(tmp_path)
        try:
            mem.start_session("abc-123", "m1")
            mem.save_messages("abc-123", [{"role": "user", "content": "fix the bug"}])
        finally:
            mem.close()

        settings = _settings_with_global_dir(tmp_path)
        ctx = _resolve_resume_context(
            continue_session=False, resume_session="abc-123", settings=settings
        )

        assert ctx is not None
        assert ctx["session_id"] == "abc-123"
        assert ctx["messages"] == [{"role": "user", "content": "fix the bug"}]

    def test_resume_unknown_id_exits_5(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_resume_context

        settings = _settings_with_global_dir(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _resolve_resume_context(
                continue_session=False, resume_session="does-not-exist", settings=settings
            )
        assert excinfo.value.code == int(ExitCode.INVALID_INPUT)

    def test_resume_returns_summary_when_no_messages(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_resume_context

        mem = _memory_at(tmp_path)
        try:
            mem.start_session("abc-123", "m1")
            mem.end_session("abc-123", summary="Fixed the auth bug")
        finally:
            mem.close()

        settings = _settings_with_global_dir(tmp_path)
        ctx = _resolve_resume_context(
            continue_session=False, resume_session="abc-123", settings=settings
        )

        assert ctx is not None
        assert ctx["session_id"] == "abc-123"
        assert ctx["messages"] is None
        assert ctx["summary"] == "Fixed the auth bug"


class TestSummaryPopulation:
    """end_session always stores a non-empty summary."""

    def test_end_session_with_empty_summary_auto_generates(
        self, session_mem: SessionMemory
    ) -> None:
        session_mem.start_session("s1", "m")
        session_mem.save_messages(
            "s1",
            [
                {"role": "user", "content": "Refactor the auth module"},
                {"role": "assistant", "content": "Done."},
            ],
        )
        session_mem.end_session("s1", summary="")

        s = session_mem.get_session("s1")
        assert s is not None
        assert s["summary"] != ""
        assert "turns=1" in s["summary"]
        assert "auth module" in s["summary"]

    def test_end_session_with_explicit_summary_keeps_it(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m")
        session_mem.end_session("s1", summary="Explicit summary")
        s = session_mem.get_session("s1")
        assert s is not None
        assert s["summary"] == "Explicit summary"

    def test_generate_summary_counts_turns_and_tools(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "done"},
        ]
        summary = SessionMemory.generate_summary(messages)
        assert "turns=2" in summary
        assert "assistant=2" in summary
        assert "tool_calls=2" in summary
        assert "topic: first" in summary


class TestBootstrapInjection:
    """Resume bootstraps context into the conversation."""

    def _conversation(self) -> Conversation:
        return Conversation(system_prompt="sys", model="test-model")

    def test_restore_messages_replaces_history(self) -> None:
        conv = self._conversation()
        conv.add_user_message("original")
        conv.restore_messages([{"role": "user", "content": "restored"}])
        assert conv.messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "restored"},
        ]

    def test_add_system_message_injects_marker(self) -> None:
        conv = self._conversation()
        conv.add_system_message("[resumed: abc-123]\n\nPrevious session summary:\nFixed bug")
        roles = [m["role"] for m in conv.messages]
        assert roles == ["system", "system"]
        assert "[resumed: abc-123]" in conv.messages[-1]["content"]

    def test_bootstrap_with_messages_restores_full_history(self) -> None:
        from godspeed.tui.textual_app import GodspeedTextualApp

        app = GodspeedTextualApp.__new__(GodspeedTextualApp)
        app._resume_context = {
            "session_id": "abc-123",
            "messages": [{"role": "user", "content": "hello"}],
            "summary": "",
        }
        app._resume_notice = None
        conv = self._conversation()
        app._apply_resume_bootstrap(conv)
        assert conv.messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert app._resume_notice is not None
        assert "abc-123" in app._resume_notice

    def test_bootstrap_without_messages_injects_summary_marker(self) -> None:
        from godspeed.tui.textual_app import GodspeedTextualApp

        app = GodspeedTextualApp.__new__(GodspeedTextualApp)
        app._resume_context = {
            "session_id": "abc-123",
            "messages": None,
            "summary": "Fixed the auth bug",
        }
        app._resume_notice = None
        conv = self._conversation()
        app._apply_resume_bootstrap(conv)
        assert len(conv.messages) == 2
        assert conv.messages[-1]["role"] == "system"
        assert "[resumed: abc-123]" in conv.messages[-1]["content"]
        assert "Fixed the auth bug" in conv.messages[-1]["content"]


class TestListSessionsCommand:
    """`godspeed --list-sessions` prints a table of recent sessions."""

    def test_list_sessions_prints_table(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import list_sessions

        mem = _memory_at(tmp_path)
        try:
            mem.start_session("abc-123", "m1")
            mem.end_session("abc-123", summary="Fixed the auth bug")
        finally:
            mem.close()

        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.environ["GODSPEED_GLOBAL_DIR"] = str(tmp_path)
            result = runner.invoke(list_sessions, [], standalone_mode=False)

        assert result.exit_code == 0
        assert "Recent Sessions" in result.output
        assert "abc-123" in result.output
        assert "Fixed the auth bug" in result.output

    def test_list_sessions_empty(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import list_sessions

        runner = CliRunner()
        with runner.isolated_filesystem():
            import os

            os.environ["GODSPEED_GLOBAL_DIR"] = str(tmp_path)
            result = runner.invoke(list_sessions, [], standalone_mode=False)

        assert result.exit_code == 0
        assert "No sessions recorded yet." in result.output


class TestCliFlagWiring:
    """The --continue / --resume flags are registered on the main group."""

    def test_continue_flag_registered(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--continue" in result.output
        assert "--resume" in result.output

    def test_list_sessions_command_registered(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "list-sessions" in result.output
