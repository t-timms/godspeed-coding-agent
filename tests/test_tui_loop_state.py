"""Tests for /loop — interval parsing, loop state, and turn dispatch."""

from __future__ import annotations

import asyncio
import re
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands
from godspeed.tui.loop_state import (
    LOOP_DEFAULT_INTERVAL_SECONDS,
    LoopState,
    is_loop_interval,
    parse_loop_interval,
)
from godspeed.tui.message_queue import MessageQueue

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


class TestParseLoopInterval:
    """Interval parsing: bare seconds, suffixes, invalid, zero rejection."""

    def test_bare_seconds(self) -> None:
        assert parse_loop_interval("30") == 30.0

    def test_seconds_suffix(self) -> None:
        assert parse_loop_interval("90s") == 90.0

    def test_minutes_suffix(self) -> None:
        assert parse_loop_interval("5m") == 300.0

    def test_hours_suffix(self) -> None:
        assert parse_loop_interval("2h") == 7200.0

    def test_uppercase_suffix(self) -> None:
        assert parse_loop_interval("5M") == 300.0

    def test_whitespace_tolerated(self) -> None:
        assert parse_loop_interval(" 5m ") == 300.0

    def test_invalid_suffix_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_loop_interval("5x")

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_loop_interval("abc")

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_loop_interval("0")

    def test_zero_with_suffix_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_loop_interval("0m")

    def test_is_loop_interval_shape(self) -> None:
        assert is_loop_interval("30")
        assert is_loop_interval("5m")
        assert is_loop_interval("2h")
        assert not is_loop_interval("5x")
        assert not is_loop_interval("abc")

    def test_default_interval_constant(self) -> None:
        assert LOOP_DEFAULT_INTERVAL_SECONDS == 60


class TestLoopState:
    """Pure loop-state logic: due computation, counter, stop."""

    def test_starts_disabled(self) -> None:
        state = LoopState()
        assert not state.enabled
        assert state.prompt == ""
        assert state.interval == LOOP_DEFAULT_INTERVAL_SECONDS
        assert state.last_dispatched_at is None
        assert state.turn_count == 0

    def test_start_enables_and_resets(self) -> None:
        state = LoopState()
        state.start("check status", 300.0)
        assert state.enabled
        assert state.prompt == "check status"
        assert state.interval == 300.0
        assert state.last_dispatched_at is None
        assert state.turn_count == 0

    def test_is_due_when_never_dispatched(self) -> None:
        state = LoopState()
        state.start("check status", 300.0)
        assert state.is_due(0.0)

    def test_is_due_only_after_interval(self) -> None:
        state = LoopState()
        state.start("check status", 300.0)
        state.mark_dispatched(100.0)
        assert not state.is_due(100.0 + 299.0)
        assert state.is_due(100.0 + 300.0)

    def test_disabled_never_due(self) -> None:
        state = LoopState()
        assert not state.is_due(0.0)
        assert not state.is_due(10_000.0)

    def test_mark_dispatched_increments_counter_and_returns_header(self) -> None:
        state = LoopState()
        state.start("check status", 60.0)
        assert state.mark_dispatched(10.0) == "[loop turn 1] check status"
        assert state.mark_dispatched(70.0) == "[loop turn 2] check status"
        assert state.turn_count == 2
        assert state.last_dispatched_at == 70.0

    def test_stop_clears_state(self) -> None:
        state = LoopState()
        state.start("check status", 60.0)
        state.mark_dispatched(10.0)
        state.stop()
        assert not state.enabled
        assert state.prompt == ""
        assert state.interval == LOOP_DEFAULT_INTERVAL_SECONDS
        assert state.last_dispatched_at is None
        assert state.turn_count == 0
        assert not state.is_due(999.0)

    def test_stop_does_not_touch_message_queue(self) -> None:
        queue = MessageQueue()
        queue.enqueue("[loop turn 1] check status")
        state = LoopState()
        state.start("check status", 60.0)
        state.stop()
        # Stop only clears loop state — a queued loop message survives.
        assert queue.drain() == ["[loop turn 1] check status"]


class TestLoopCommand:
    """/loop command parsing and status output."""

    def test_loop_status_not_looping(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/loop")
        assert "Not looping" in output

    def test_loop_start_default_interval(self, commands: Commands) -> None:
        result = commands.dispatch("/loop check status")
        assert result is not None
        assert result.handled
        assert commands._loop_state.enabled
        assert commands._loop_state.prompt == "check status"
        assert commands._loop_state.interval == LOOP_DEFAULT_INTERVAL_SECONDS

    def test_loop_start_with_interval(self, commands: Commands) -> None:
        commands.dispatch("/loop 5m check status")
        assert commands._loop_state.enabled
        assert commands._loop_state.prompt == "check status"
        assert commands._loop_state.interval == 300.0

    def test_loop_start_with_seconds_suffix(self, commands: Commands) -> None:
        commands.dispatch("/loop 90s check status")
        assert commands._loop_state.interval == 90.0

    def test_loop_start_with_hours(self, commands: Commands) -> None:
        commands.dispatch("/loop 2h check status")
        assert commands._loop_state.interval == 7200.0

    def test_loop_zero_interval_rejected(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/loop 0 check status")
        assert "Interval must be positive" in output
        assert not commands._loop_state.enabled

    def test_loop_missing_prompt_usage(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/loop 5m")
        assert "Usage: /loop" in output
        assert not commands._loop_state.enabled

    def test_loop_stop(self, commands: Commands) -> None:
        commands.dispatch("/loop check status")
        assert commands._loop_state.enabled
        result = commands.dispatch("/loop stop")
        assert result is not None
        assert result.handled
        assert not commands._loop_state.enabled
        assert commands._loop_state.prompt == ""

    def test_loop_status_shows_prompt_and_interval(self, commands: Commands) -> None:
        commands.dispatch("/loop 5m check status")
        output = _capture(commands.dispatch, "/loop")
        assert "check status" in output
        assert "300" in output

    def test_loop_non_interval_leading_token_is_prompt_data(self, commands: Commands) -> None:
        commands.dispatch("/loop 5x check status")
        assert commands._loop_state.prompt == "5x check status"
        assert commands._loop_state.interval == LOOP_DEFAULT_INTERVAL_SECONDS

    def test_loop_in_help(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/loop [interval] <prompt>" in output
        assert "prompt loops on interval" in output


class TestMaybeDispatchLoopTurn:
    """Commands._maybe_dispatch_loop_turn with a real MessageQueue."""

    def _wire_queue(self, commands: Commands) -> MessageQueue:
        queue = MessageQueue()
        commands._message_queue = queue
        return queue

    def test_enqueues_loop_prompt_with_header(self, commands: Commands) -> None:
        queue = self._wire_queue(commands)
        commands.dispatch("/loop check status")

        dispatched = commands._maybe_dispatch_loop_turn(now=100.0)
        assert dispatched is True
        assert queue.drain() == ["[loop turn 1] check status"]

    def test_timer_reset_prevents_immediate_redispatch(self, commands: Commands) -> None:
        queue = self._wire_queue(commands)
        commands.dispatch("/loop 5m check status")
        commands._maybe_dispatch_loop_turn(now=100.0)
        queue.drain()

        # Same instant: not due yet.
        assert commands._maybe_dispatch_loop_turn(now=100.0) is False
        assert len(queue) == 0
        # 4m59s later: still not due.
        assert commands._maybe_dispatch_loop_turn(now=100.0 + 299.0) is False
        assert len(queue) == 0
        # 5m elapsed: due again, counter incremented.
        assert commands._maybe_dispatch_loop_turn(now=100.0 + 300.0) is True
        assert queue.drain() == ["[loop turn 2] check status"]

    def test_paused_skips_dispatch_and_timer_resumes(self, commands: Commands) -> None:
        queue = self._wire_queue(commands)
        commands.dispatch("/loop check status")
        commands._pause_event = asyncio.Event()
        commands._pause_event.clear()  # paused

        assert commands._maybe_dispatch_loop_turn(now=100.0) is False
        assert len(queue) == 0

        # Resume: timer untouched, still due immediately.
        commands._pause_event.set()
        assert commands._maybe_dispatch_loop_turn(now=100.0) is True
        assert queue.drain() == ["[loop turn 1] check status"]

    def test_disabled_loop_does_not_dispatch(self, commands: Commands) -> None:
        queue = self._wire_queue(commands)
        assert commands._maybe_dispatch_loop_turn(now=100.0) is False
        assert len(queue) == 0

    def test_no_queue_wired_is_noop(self, commands: Commands) -> None:
        commands.dispatch("/loop check status")
        # Fixture leaves _message_queue None — dispatch must not crash.
        assert commands._maybe_dispatch_loop_turn(now=100.0) is False

    async def test_fake_app_drain_loop(self, commands: Commands) -> None:
        """Simulate the app's drain site: drain, dispatch, drain again."""
        queue = self._wire_queue(commands)
        commands.dispatch("/loop 5m check status")

        async def drain_once(now: float) -> list[str]:
            """Mimic the app's turn-completion drain site."""
            queued = queue.drain()
            if queued:
                return queued
            if commands._maybe_dispatch_loop_turn(now=now):
                return queue.drain()
            return []

        # After simulated turn 1: loop turn dispatched immediately.
        assert await drain_once(now=0.0) == ["[loop turn 1] check status"]
        # After the loop turn: not due yet.
        assert await drain_once(now=10.0) == []
        # Interval elapsed: next loop turn dispatched.
        assert await drain_once(now=300.0) == ["[loop turn 2] check status"]
