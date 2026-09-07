"""Side-question (btw) helpers for the Godspeed TUI.

Provides pure functions to build a temporary message list for an aside
question and verify that the main conversation is left untouched.  The
command handler in ``tui/commands.py`` orchestrates the LLM call; this
module owns the *data* decisions only.

Design:
    /btw snapshots the current conversation, appends the question as a
    user message, runs a single assistant turn with a small token budget,
    prints the answer, then **discards** the snapshot.  The main
    conversation is byte-identical afterward.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Maximum output tokens for a btw turn — keeps cost and latency minimal.
BTW_MAX_TOKENS: int = 512


def build_btw_messages(
    messages: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """Build the message list for a side-question LLM call.

    The returned list is a **deep copy** of *messages* with the user's
    *question* appended.  The original list is never mutated.

    Args:
        messages: The current conversation message list (including the
            system prompt at index 0).
        question: The user's btw question text.

    Returns:
        A new message list ready for ``LLMClient.chat()``.

    Raises:
        ValueError: If *question* is empty or whitespace-only.
    """
    if not question or not question.strip():
        raise ValueError("btw question must not be empty")

    snapshot = copy.deepcopy(messages)
    snapshot.append({"role": "user", "content": question.strip()})
    return snapshot


def snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep copy of *messages* for later comparison / restore.

    Used by the command handler to capture the conversation state *before*
    the btw call so it can verify nothing leaked.
    """
    return copy.deepcopy(messages)


def verify_conversation_unchanged(
    original: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> bool:
    """Return True if *current* is byte-identical to *original*.

    This is a post-btw safety check — if the handler accidentally mutated
    the real conversation, this will catch it.
    """
    return original == current
