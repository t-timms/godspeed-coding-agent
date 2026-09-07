"""Tests for src/godspeed/tui/message_queue.py — FIFO message queue."""

from __future__ import annotations

import pytest

from godspeed.tui.message_queue import MessageQueue


class TestMessageQueue:
    """Verify FIFO semantics and type safety."""

    def test_starts_empty(self) -> None:
        queue = MessageQueue()
        assert len(queue) == 0
        assert not queue
        assert queue.dequeue() is None
        assert queue.peek() is None
        assert queue.drain() == []

    def test_enqueue_dequeue_fifo(self) -> None:
        queue = MessageQueue()
        queue.enqueue("first")
        queue.enqueue("second")
        queue.enqueue("third")
        assert len(queue) == 3
        assert queue.dequeue() == "first"
        assert queue.dequeue() == "second"
        assert queue.dequeue() == "third"
        assert queue.dequeue() is None

    def test_peek_does_not_remove(self) -> None:
        queue = MessageQueue()
        queue.enqueue("only")
        assert queue.peek() == "only"
        assert len(queue) == 1
        assert queue.dequeue() == "only"

    def test_drain_returns_all_in_order_and_empties(self) -> None:
        queue = MessageQueue()
        queue.enqueue("a")
        queue.enqueue("b")
        assert queue.drain() == ["a", "b"]
        assert len(queue) == 0
        assert queue.drain() == []

    def test_bool_reflects_contents(self) -> None:
        queue = MessageQueue()
        assert not queue
        queue.enqueue("x")
        assert queue
        queue.dequeue()
        assert not queue

    def test_iteration_yields_fifo_order(self) -> None:
        queue = MessageQueue()
        queue.enqueue("a")
        queue.enqueue("b")
        assert list(queue) == ["a", "b"]

    def test_repr(self) -> None:
        queue = MessageQueue()
        queue.enqueue("a")
        assert repr(queue) == "MessageQueue(['a'])"

    def test_enqueue_rejects_non_string(self) -> None:
        queue = MessageQueue()
        with pytest.raises(TypeError):
            queue.enqueue(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            queue.enqueue(None)  # type: ignore[arg-type]
        assert len(queue) == 0
