"""Tests for conversation logger — JSONL persistence for training data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from godspeed.training.conversation_logger import ConversationLogger


@pytest.fixture
def logger_dir(tmp_path: Path) -> Path:
    return tmp_path / "training"


@pytest.fixture
def clog(logger_dir: Path) -> ConversationLogger:
    cl = ConversationLogger(session_id="test-session-001", output_dir=logger_dir)
    yield cl
    cl.close()


def _read_lines(clog: ConversationLogger, path: Path) -> list[dict]:
    """Read JSONL records, flushing the logger first to ensure data is on disk."""
    clog.flush()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestConversationLogger:
    def test_creates_file_on_first_write(self, clog: ConversationLogger) -> None:
        assert not clog.path.exists()
        clog.log_user("hello")
        assert clog.path.exists()

    def test_tool_result_secrets_are_redacted(self, clog: ConversationLogger) -> None:
        clog.log_tool_result(
            "call_1",
            "file_read",
            'aws_key = "AKIAIOSFODNN7EXAMPLE"',  # canonical AWS docs example
        )
        records = _read_lines(clog, clog.path)
        raw = clog.path.read_text()
        assert "AKIAIOSFODNN7EXAMPLE" not in raw
        assert "[REDACTED" in records[0]["content"]

    def test_log_system(self, clog: ConversationLogger) -> None:
        clog.log_system("You are Godspeed.")
        records = _read_lines(clog, clog.path)
        assert len(records) == 1
        assert records[0]["role"] == "system"
        assert records[0]["content"] == "You are Godspeed."
        assert "timestamp" in records[0]
        assert records[0]["session_id"] == "test-session-001"

    def test_log_user_text(self, clog: ConversationLogger) -> None:
        clog.log_user("Fix the bug in auth.py")
        records = _read_lines(clog, clog.path)
        assert records[0]["role"] == "user"
        assert records[0]["content"] == "Fix the bug in auth.py"

    def test_log_user_multimodal(self, clog: ConversationLogger) -> None:
        blocks = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        clog.log_user(blocks)
        records = _read_lines(clog, clog.path)
        assert records[0]["content"] == blocks

    def test_log_assistant_text_only(self, clog: ConversationLogger) -> None:
        clog.log_assistant(content="I found the issue.")
        records = _read_lines(clog, clog.path)
        assert records[0]["role"] == "assistant"
        assert records[0]["content"] == "I found the issue."
        assert "tool_calls" not in records[0]

    def test_log_assistant_with_tool_calls(self, clog: ConversationLogger) -> None:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "file_read",
                    "arguments": '{"file_path": "app.py"}',
                },
            }
        ]
        clog.log_assistant(content="", tool_calls=tool_calls)
        records = _read_lines(clog, clog.path)
        assert records[0]["tool_calls"] == tool_calls

    def test_log_assistant_with_thinking(self, clog: ConversationLogger) -> None:
        clog.log_assistant(content="Result", thinking="I need to think...")
        records = _read_lines(clog, clog.path)
        assert records[0]["thinking"] == "I need to think..."

    def test_log_tool_result(self, clog: ConversationLogger) -> None:
        clog.log_tool_result(
            tool_call_id="call_1",
            tool_name="file_read",
            content="1\tdef hello(): pass",
            is_error=False,
        )
        records = _read_lines(clog, clog.path)
        assert records[0]["role"] == "tool"
        assert records[0]["tool_call_id"] == "call_1"
        assert records[0]["name"] == "file_read"
        assert records[0]["is_error"] is False
        assert records[0]["step"] == 1

    def test_log_tool_result_error(self, clog: ConversationLogger) -> None:
        clog.log_tool_result(
            tool_call_id="call_2",
            tool_name="shell",
            content="command not found",
            is_error=True,
        )
        records = _read_lines(clog, clog.path)
        assert records[0]["is_error"] is True

    def test_step_counter_increments(self, clog: ConversationLogger) -> None:
        assert clog.step_count == 0
        clog.log_tool_result("c1", "file_read", "ok")
        assert clog.step_count == 1
        clog.log_tool_result("c2", "file_edit", "ok")
        assert clog.step_count == 2

    def test_log_compaction(self, clog: ConversationLogger) -> None:
        clog.log_compaction(
            summary="User asked to fix auth. We read and edited auth.py.",
            messages_before=25,
            messages_after=1,
        )
        records = _read_lines(clog, clog.path)
        assert records[0]["role"] == "meta"
        assert records[0]["event"] == "compaction"
        assert records[0]["summary"].startswith("User asked")
        assert records[0]["messages_before"] == 25
        assert records[0]["messages_after"] == 1

    def test_full_conversation_flow(self, clog: ConversationLogger) -> None:
        """Verify a realistic multi-turn conversation with tool calls."""
        clog.log_system("You are Godspeed.")
        clog.log_user("Read app.py")
        clog.log_assistant(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": '{"file_path":"app.py"}'},
                }
            ],
        )
        clog.log_tool_result("call_1", "file_read", "1\tdef main(): pass")
        clog.log_assistant(content="The file has a main function.")

        records = _read_lines(clog, clog.path)
        assert len(records) == 5
        roles = [r["role"] for r in records]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_close_and_reopen(self, clog: ConversationLogger) -> None:
        clog.log_user("first")
        clog.close()
        # Re-open by logging again (file opened in append mode)
        clog.log_user("second")
        records = _read_lines(clog, clog.path)
        assert len(records) == 2

    def test_jsonl_format_valid(self, clog: ConversationLogger) -> None:
        """Each line must be valid JSON independently."""
        clog.log_user("hello")
        clog.log_assistant("world")
        lines = clog.path.read_text().splitlines()
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_no_thinking_field_when_empty(self, clog: ConversationLogger) -> None:
        clog.log_assistant(content="just text")
        records = _read_lines(clog, clog.path)
        assert "thinking" not in records[0]

    def test_no_tool_calls_field_when_empty(self, clog: ConversationLogger) -> None:
        clog.log_assistant(content="just text")
        records = _read_lines(clog, clog.path)
        assert "tool_calls" not in records[0]

    def test_path_property(self, clog: ConversationLogger, logger_dir: Path) -> None:
        assert clog.path == logger_dir / "test-session-001.conversation.jsonl"

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"
        cl = ConversationLogger(session_id="deep", output_dir=deep_dir)
        cl.log_user("test")
        assert cl.path.exists()
        cl.close()
