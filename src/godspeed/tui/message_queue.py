"""Message queue for the Godspeed TUI.

When the agent is streaming/running, user input is queued instead of being
lost or ignored. After the current turn completes, queued messages are
injected in order at the safe point between turns.

The queue is intentionally a small, pure helper so it can be unit-tested
without any interactive mocking.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator


class MessageQueue:
    """FIFO queue of pending user messages.

    Thread-safe enough for the TUI's single-threaded asyncio loop: all
    mutations happen on the event loop, so no locking is required.
    """

    def __init__(self) -> None:
        self._items: deque[str] = deque()

    def enqueue(self, message: str) -> None:
        """Add a message to the back of the queue."""
        if not isinstance(message, str):
            msg = f"message must be a str, got {type(message).__name__}"
            raise TypeError(msg)
        self._items.append(message)

    def dequeue(self) -> str | None:
        """Remove and return the front message, or None if empty."""
        if not self._items:
            return None
        return self._items.popleft()

    def drain(self) -> list[str]:
        """Remove and return all queued messages in FIFO order.

        Returns an empty list when the queue is empty.
        """
        items = list(self._items)
        self._items.clear()
        return items

    def peek(self) -> str | None:
        """Return the front message without removing it, or None if empty."""
        if not self._items:
            return None
        return self._items[0]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"MessageQueue({list(self._items)!r})"
