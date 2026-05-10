"""Tests for the ChatView widget — tool calls, results, streaming, markdown."""

from __future__ import annotations

import pytest


def _get_content(view) -> str:
    """Get all written content from ChatView deferred renders."""
    from io import StringIO

    from rich.console import Console

    parts: list[str] = []
    for d in view._deferred_renders:
        if d.content:
            if isinstance(d.content, str):
                parts.append(d.content)
            else:
                buf = StringIO()
                c = Console(file=buf, force_terminal=True, width=120, no_color=True)
                c.print(d.content)
                parts.append(buf.getvalue())
    return "\n".join(parts)


class TestChatViewToolCall:
    """Verify write_tool_call renders correctly for each tool shape."""

    @pytest.fixture
    def view(self):
        from godspeed.tui.widgets.chat_view import ChatView

        return ChatView()

    def test_file_read_compact(self, view):
        view.write_tool_call("file_read", {"file_path": "src/main.py"})
        content = _get_content(view)
        assert "file_read" in content
        assert "src/main.py" in content

    def test_grep_shows_pattern(self, view):
        view.write_tool_call("grep_search", {"pattern": "def test"})
        content = _get_content(view)
        assert "def test" in content

    def test_grep_with_path(self, view):
        view.write_tool_call("grep_search", {"pattern": "def test", "path": "src/"})
        content = _get_content(view)
        assert "def test" in content
        assert "src/" in content

    def test_git_shows_action(self, view):
        view.write_tool_call("git", {"action": "diff", "branch": "main"})
        content = _get_content(view)
        assert "diff" in content
        assert "main" in content

    def test_shell_shows_command(self, view):
        view.write_tool_call("shell", {"command": "pytest -v"})
        content = _get_content(view)
        assert "pytest -v" in content

    def test_file_edit_shows_path(self, view):
        view.write_tool_call("file_edit", {"file_path": "src/main.py"})
        content = _get_content(view)
        assert "src/main.py" in content

    def test_file_edit_with_diff(self, view):
        view.write_tool_call(
            "file_edit",
            {"file_path": "src/main.py", "old_string": "hello", "new_string": "world"},
        )
        content = _get_content(view)
        assert "src/main.py" in content

    def test_file_write_shows_line_count(self, view):
        view.write_tool_call(
            "file_write", {"file_path": "src/main.py", "content": "line1\nline2\nline3"}
        )
        content = _get_content(view)
        assert "src/main.py" in content
        assert "3 lines" in content

    def test_glob_search_compact(self, view):
        view.write_tool_call("glob_search", {"file_path": "src/**/*.py"})
        content = _get_content(view)
        assert "src/**/*.py" in content

    def test_repo_map_compact(self, view):
        view.write_tool_call("repo_map", {"file_path": "src/"})
        content = _get_content(view)
        assert "src/" in content

    def test_default_json_args(self, view):
        view.write_tool_call("unknown_tool", {"key": "value", "num": 42})
        content = _get_content(view)
        assert "unknown_tool" in content

    def test_non_serializable_args(self, view):
        view.write_tool_call("unknown_tool", {"obj": object()})
        content = _get_content(view)
        assert "unknown_tool" in content


class TestChatViewToolResult:
    """Verify write_tool_result renders success/error correctly."""

    @pytest.fixture
    def view(self):
        from godspeed.tui.widgets.chat_view import ChatView

        return ChatView()

    def test_success_short(self, view):
        view.write_tool_result("file_read", "content of file", is_error=False)
        content = _get_content(view)
        assert "file_read" in content
        assert "content of file" in content

    def test_success_empty(self, view):
        view.write_tool_result("shell", "", is_error=False)
        content = _get_content(view)
        assert "shell" in content

    def test_error_result(self, view):
        view.write_tool_result("shell", "command not found", is_error=True)
        content = _get_content(view)
        assert "shell" in content
        assert "command not found" in content

    def test_long_success_shows_line_count(self, view):
        long_text = "\n".join(f"line {i}" for i in range(20))
        view.write_tool_result("file_read", long_text, is_error=False)
        content = _get_content(view)
        assert "20 lines" in content

    def test_long_error_truncated(self, view):
        long_text = "x" * 3000
        view.write_tool_result("shell", long_text, is_error=True)
        content = _get_content(view)
        assert "more chars" in content or "..." in content

    def test_timing_milliseconds(self, view):
        view.write_tool_result("shell", "done", is_error=False, duration_ms=500)
        content = _get_content(view)
        assert "500ms" in content or "0.5s" in content

    def test_timing_seconds(self, view):
        view.write_tool_result("shell", "done", is_error=False, duration_ms=1500)
        content = _get_content(view)
        assert "1.5s" in content

    def test_timing_zero_not_shown(self, view):
        view.write_tool_result("file_read", "done", is_error=False, duration_ms=0)
        content = _get_content(view)
        assert "ms" not in content


class TestChatViewStreaming:
    """Verify streaming and markdown rendering."""

    @pytest.fixture
    def view(self):
        from godspeed.tui.widgets.chat_view import ChatView

        return ChatView()

    def test_write_chunk_buffers(self, view):
        view.start_turn()
        view.write_chunk("Hello ")
        view.write_chunk("World")
        assert len(view._markdown_buffer) == 2
        content = _get_content(view)
        assert "Hello" in content
        assert "World" in content

    def test_write_markdown_renders(self, view):
        view.write_markdown("# Title\n\nContent")
        content = _get_content(view)
        assert "Title" in content
        assert "Content" in content

    def test_write_markdown_empty_skipped(self, view):
        view.write_markdown("")
        view.write_markdown("   ")
        content = _get_content(view)
        assert content.strip() == ""

    def test_end_turn_resets(self, view):
        view.start_turn()
        view.end_turn()
        assert not view._in_turn

    def test_start_turn_clears_buffer(self, view):
        view.start_turn()
        view.write_chunk("old")
        view.start_turn()
        assert view._markdown_buffer == []


class TestChatViewStatusMessages:
    """Verify status/info/error methods."""

    @pytest.fixture
    def view(self):
        from godspeed.tui.widgets.chat_view import ChatView

        return ChatView()

    def test_write_info(self, view):
        view.write_info("Note")
        content = _get_content(view)
        assert "Note" in content

    def test_write_success(self, view):
        view.write_success("Done")
        content = _get_content(view)
        assert "Done" in content

    def test_write_warning(self, view):
        view.write_warning("Careful")
        content = _get_content(view)
        assert "Careful" in content

    def test_write_error(self, view):
        view.write_error("Boom")
        content = _get_content(view)
        assert "Boom" in content

    def test_write_status(self, view):
        view.write_status("Idle")
        content = _get_content(view)
        assert "Idle" in content

    def test_write_permission_denied(self, view):
        view.write_permission_denied("shell", "blocked")
        content = _get_content(view)
        assert "shell" in content
        assert "blocked" in content

    def test_write_thinking(self, view):
        view.write_thinking("Let me think about this carefully...")
        content = _get_content(view)
        assert "think about this" in content

    def test_write_thinking_empty(self, view):
        view.write_thinking("")
        content = _get_content(view)
        assert content.strip() == ""

    def test_write_thinking_truncated(self, view):
        long_text = "x" * 3000
        view.write_thinking(long_text)
        content = _get_content(view)
        assert "truncated" in content


class TestChatViewWriteMethod:
    """Verify the write() override works with all signatures."""

    @pytest.fixture
    def view(self):
        from godspeed.tui.widgets.chat_view import ChatView

        return ChatView()

    def test_write_no_args(self, view):
        view.write()
        content = _get_content(view)
        assert content.strip() == ""

    def test_write_with_content(self, view):
        view.write("hello")
        content = _get_content(view)
        assert "hello" in content

    def test_write_with_width(self, view):
        view.write("hello", width=40)
        content = _get_content(view)
        assert "hello" in content

    def test_write_with_expand(self, view):
        view.write("hello", expand=True)
        content = _get_content(view)
        assert "hello" in content

    def test_write_with_all_args(self, view):
        view.write("hello", width=40, expand=True, shrink=False, scroll_end=True)
        content = _get_content(view)
        assert "hello" in content


class TestChatViewGutter:
    """Verify gutter helper renders correctly."""

    @pytest.fixture
    def view(self):
        from godspeed.tui.widgets.chat_view import ChatView

        return ChatView()

    def test_gutter_single_line(self, view):
        view._gutter_text("hello")
        content = _get_content(view)
        assert "hello" in content

    def test_gutter_multi_line(self, view):
        view._gutter_text("line1\nline2\nline3")
        content = _get_content(view)
        assert "line1" in content
        assert "line2" in content
        assert "line3" in content
