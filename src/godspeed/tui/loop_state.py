"""Loop state for the /loop TUI command.

Pure state holder plus interval parsing so the recurring-prompt logic
can be unit-tested without any interactive mocking. The app checks the
state at the turn-completion point and enqueues the loop prompt into
the message queue when due.
"""

from __future__ import annotations

import re

LOOP_DEFAULT_INTERVAL_SECONDS = 60

_INTERVAL_RE = re.compile(r"^(\d+)([smh])?$")

_UNIT_MULTIPLIERS = {"s": 1, "m": 60, "h": 3600}


def is_loop_interval(text: str) -> bool:
    """Return True when *text* is a syntactically valid interval spec.

    Matches a bare number (seconds) or an ``Ns``/``Nm``/``Nh`` suffix.
    Zero is syntactically valid but rejected by :func:`parse_loop_interval`.
    """
    return _INTERVAL_RE.match(text.strip().lower()) is not None


def parse_loop_interval(text: str) -> float:
    """Parse an interval spec into seconds.

    Accepts a bare number (seconds) or an ``Ns``/``Nm``/``Nh`` suffix.
    Raises ``ValueError`` for invalid specs and for zero intervals.
    """
    match = _INTERVAL_RE.match(text.strip().lower())
    if match is None:
        raise ValueError(f"Invalid interval: {text!r}. Use seconds, or a suffix like 5m / 2h.")
    value = int(match.group(1))
    multiplier = _UNIT_MULTIPLIERS.get(match.group(2), 1)
    seconds = value * multiplier
    if seconds <= 0:
        raise ValueError("Interval must be positive.")
    return float(seconds)


class LoopState:
    """State for the recurring /loop prompt.

    Holds the enabled flag, prompt, interval, and the monotonic clock
    value of the last dispatch. The clock is passed in as a parameter
    so tests can drive due/not-due transitions deterministically.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.prompt = ""
        self.interval = LOOP_DEFAULT_INTERVAL_SECONDS
        self.last_dispatched_at: float | None = None
        self.turn_count = 0

    def start(self, prompt: str, interval: float) -> None:
        """Enable the loop with the given prompt and interval."""
        self.enabled = True
        self.prompt = prompt
        self.interval = interval
        self.last_dispatched_at = None
        self.turn_count = 0

    def stop(self) -> None:
        """Disable the loop, resetting all state.

        Only touches this state object — a loop message already sitting
        in the message queue is left alone and will still be processed.
        """
        self.enabled = False
        self.prompt = ""
        self.interval = LOOP_DEFAULT_INTERVAL_SECONDS
        self.last_dispatched_at = None
        self.turn_count = 0

    def is_due(self, now: float) -> bool:
        """Return True when the loop is enabled and the interval elapsed."""
        if not self.enabled:
            return False
        if self.last_dispatched_at is None:
            return True
        return now - self.last_dispatched_at >= self.interval

    def mark_dispatched(self, now: float) -> str:
        """Record a dispatch and return the loop message with its header.

        The header is ``[loop turn N]`` where N is a 1-based counter.
        Each dispatch resets the timer to *now*.
        """
        self.turn_count += 1
        self.last_dispatched_at = now
        return f"[loop turn {self.turn_count}] {self.prompt}"
