"""Tests for src/godspeed/tui/rewind.py — ESC+ESC rewind picker helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from godspeed.tui.rewind import (
    RESTORE_BOTH,
    RESTORE_CONVERSATION,
    RESTORE_FILES,
    RESTORE_NONE,
    REWIND_WINDOW_SECONDS,
    collect_rewind_entries,
    parse_rewind_choice,
    restore_conversation,
    restore_files,
)


class TestParseRewindChoice:
    """Verify single-character choice mapping."""

    def test_conversation_choices(self) -> None:
        assert parse_rewind_choice("c") == RESTORE_CONVERSATION
        assert parse_rewind_choice("conversation") == RESTORE_CONVERSATION
        assert parse_rewind_choice("conv") == RESTORE_CONVERSATION
        assert parse_rewind_choice("C") == RESTORE_CONVERSATION

    def test_files_choices(self) -> None:
        assert parse_rewind_choice("f") == RESTORE_FILES
        assert parse_rewind_choice("files") == RESTORE_FILES
        assert parse_rewind_choice("file") == RESTORE_FILES

    def test_both_choices(self) -> None:
        assert parse_rewind_choice("b") == RESTORE_BOTH
        assert parse_rewind_choice("both") == RESTORE_BOTH

    def test_none_choices(self) -> None:
        assert parse_rewind_choice("n") == RESTORE_NONE
        assert parse_rewind_choice("none") == RESTORE_NONE
        assert parse_rewind_choice("q") == RESTORE_NONE
        assert parse_rewind_choice("cancel") == RESTORE_NONE
        assert parse_rewind_choice("") == RESTORE_NONE

    def test_unknown_returns_none(self) -> None:
        assert parse_rewind_choice("x") == RESTORE_NONE
        assert parse_rewind_choice("!!") == RESTORE_NONE

    def test_window_constant(self) -> None:
        assert REWIND_WINDOW_SECONDS == 0.8


class TestRestoreConversation:
    """Verify conversation checkpoint restore."""

    def test_restores_messages_in_order(self, tmp_path) -> None:
        conversation = MagicMock()
        conversation.messages = [{"role": "system"}, {"role": "user", "content": "hi"}]
        conversation.token_count = 42
        checkpoint = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there", "tool_calls": None},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ]
        }
        with patch("godspeed.context.checkpoint.load_checkpoint", return_value=checkpoint):
            summary = restore_conversation(conversation, "cp-1", tmp_path)

        conversation.clear.assert_called_once_with()
        conversation.add_user_message.assert_called_once_with("hello")
        conversation.add_assistant_message.assert_called_once_with(
            content="hi there", tool_calls=None
        )
        conversation.add_tool_result.assert_called_once_with(
            tool_call_id="call_1", content="result"
        )
        assert "cp-1" in summary

    def test_missing_checkpoint_returns_notice(self, tmp_path) -> None:
        conversation = MagicMock()
        with patch("godspeed.context.checkpoint.load_checkpoint", return_value=None):
            summary = restore_conversation(conversation, "missing", tmp_path)
        assert "not found" in summary.lower()
        conversation.clear.assert_not_called()


class TestRestoreFiles:
    """Verify latest-snapshot file restore."""

    def test_restores_latest_snapshot_per_file(self, tmp_path) -> None:
        files_dir = tmp_path / "checkpoints"
        files_dir.mkdir()
        (files_dir / "20250101-120000_000_main.py").write_text("old")
        (files_dir / "20250101-120001_001_main.py").write_text("new")
        (files_dir / "20250101-120002_000_util.py").write_text("util")

        with (
            patch("godspeed.tui.rewind.edit_checkpoints_dir", return_value=files_dir),
            patch("godspeed.tui.rewind.restore_latest", return_value="restored") as mock_restore,
        ):
            summary = restore_files(tmp_path, "sess-1")

        assert "2 file(s)" in summary
        assert "main.py" in summary
        assert "util.py" in summary
        # One restore per unique original file, despite two main.py snapshots
        assert mock_restore.call_count == 2

    def test_no_checkpoints_dir(self, tmp_path) -> None:
        with patch(
            "godspeed.tui.rewind.edit_checkpoints_dir",
            return_value=tmp_path / "missing",
        ):
            summary = restore_files(tmp_path, "sess-1")
        assert "No file checkpoints" in summary


class TestCollectRewindEntries:
    """Verify rewind candidate collection."""

    def test_combines_conversation_and_files_sorted_newest_first(self, tmp_path) -> None:
        conv_cps = [
            {"name": "cp-old", "message_count": 1, "token_count": 10, "timestamp": 100.0},
            {"name": "cp-new", "message_count": 2, "token_count": 20, "timestamp": 300.0},
        ]
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        (files_dir / "20250101-120000_000_a.py").write_text("x")
        (files_dir / "20250101-120001_000_b.py").write_text("y")

        with (
            patch("godspeed.tui.rewind.list_conv_checkpoints", return_value=conv_cps),
            patch("godspeed.tui.rewind.edit_checkpoints_dir", return_value=files_dir),
        ):
            entries = collect_rewind_entries(tmp_path, "sess-1")

        assert len(entries) == 4
        kinds = {e.kind for e in entries}
        assert kinds == {"conversation", "files"}
        timestamps = [e.timestamp for e in entries]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_caps_at_max_entries(self, tmp_path) -> None:
        conv_cps = [
            {"name": f"cp-{i}", "message_count": 0, "token_count": 0, "timestamp": float(i)}
            for i in range(20)
        ]
        with (
            patch("godspeed.tui.rewind.list_conv_checkpoints", return_value=conv_cps),
            patch("godspeed.tui.rewind.edit_checkpoints_dir", return_value=tmp_path / "missing"),
        ):
            entries = collect_rewind_entries(tmp_path, "sess-1", max_entries=5)
        assert len(entries) == 5
