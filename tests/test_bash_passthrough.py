"""Tests for src/godspeed/tui/bash_passthrough.py — ! / !! shell pass-through."""

from __future__ import annotations

import pytest

from godspeed.tui.bash_passthrough import (
    BACKGROUND_PREFIX,
    FOREGROUND_PREFIX,
    BashCommand,
    check_dangerous,
    parse_bash_command,
)


class TestParseBashCommand:
    """Verify ! / !! prefix parsing."""

    def test_no_prefix_returns_none(self) -> None:
        assert parse_bash_command("ls -la") is None
        assert parse_bash_command("") is None
        assert parse_bash_command("!not-a-command") is not None

    def test_foreground_prefix(self) -> None:
        result = parse_bash_command("!ls -la")
        assert result == BashCommand(command="ls -la", background=False)

    def test_foreground_strips_whitespace(self) -> None:
        result = parse_bash_command("!  ls -la  ")
        assert result == BashCommand(command="ls -la", background=False)

    def test_background_prefix(self) -> None:
        result = parse_bash_command("!!python server.py")
        assert result == BashCommand(command="python server.py", background=True)

    def test_background_strips_whitespace(self) -> None:
        result = parse_bash_command("!!  python server.py  ")
        assert result == BashCommand(command="python server.py", background=True)

    def test_empty_foreground_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bash_command("!")

    def test_empty_background_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bash_command("!!")

    def test_prefix_constants(self) -> None:
        assert FOREGROUND_PREFIX == "!"
        assert BACKGROUND_PREFIX == "!!"


class TestCheckDangerous:
    """Verify dangerous-command detection is wired through."""

    def test_safe_command_returns_empty(self) -> None:
        assert check_dangerous("echo hello") == []

    def test_dangerous_command_returns_descriptions(self) -> None:
        dangers = check_dangerous("rm -rf /")
        assert isinstance(dangers, list)
        assert len(dangers) > 0
