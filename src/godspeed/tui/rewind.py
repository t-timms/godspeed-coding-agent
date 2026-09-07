"""ESC+ESC rewind picker for the Godspeed TUI.

Pressing Escape twice quickly (~0.8s) during idle input opens a rewind
picker: a numbered list of recent per-prompt checkpoints (from
``edit_checkpoints.py``) plus conversation checkpoints. The user chooses
what to restore: conversation, files, both, or none (cancel).

The restore logic is factored into pure functions so it can be unit-tested
without interactive mocking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from godspeed.context.checkpoint import list_checkpoints as list_conv_checkpoints
from godspeed.tools.edit_checkpoints import (
    checkpoints_dir as edit_checkpoints_dir,
)
from godspeed.tools.edit_checkpoints import (
    restore_latest,
)

logger = logging.getLogger(__name__)

#: Time window (seconds) within which two Esc presses trigger rewind.
REWIND_WINDOW_SECONDS = 0.8

#: Restore choice vocabulary.
RESTORE_CONVERSATION = "conversation"
RESTORE_FILES = "files"
RESTORE_BOTH = "both"
RESTORE_NONE = "none"


@dataclass(frozen=True)
class RewindEntry:
    """A single rewind candidate."""

    kind: str  # "conversation" | "files"
    name: str
    detail: str
    timestamp: float = 0.0


def collect_rewind_entries(
    cwd: Path,
    session_id: str,
    *,
    max_entries: int = 10,
) -> list[RewindEntry]:
    """Collect recent rewind candidates.

    Combines conversation checkpoints (from ``checkpoint.py``) and
    per-file edit checkpoints (from ``edit_checkpoints.py``), newest
    first, capped at *max_entries*.

    Args:
        cwd: Project working directory.
        session_id: Current session id.
        max_entries: Maximum number of entries to return.

    Returns:
        List of ``RewindEntry`` objects, newest first.
    """
    entries: list[RewindEntry] = []

    # Conversation checkpoints
    try:
        for cp in list_conv_checkpoints(cwd):
            entries.append(
                RewindEntry(
                    kind="conversation",
                    name=cp.get("name", "unnamed"),
                    detail=(
                        f"{cp.get('message_count', 0)} messages, {cp.get('token_count', 0)} tokens"
                    ),
                    timestamp=cp.get("timestamp", 0.0),
                )
            )
    except OSError as exc:
        logger.warning("Failed to list conversation checkpoints: %s", exc)

    # Per-file edit checkpoints — one entry per file with snapshots
    try:
        files_dir = edit_checkpoints_dir(cwd, session_id)
        if files_dir.is_dir():
            # Group snapshot files by original filename
            seen: set[str] = set()
            for snapshot in sorted(files_dir.iterdir(), reverse=True):
                # Snapshot names look like: 20250101-120000_000_original.py
                parts = snapshot.name.split("_", 2)
                if len(parts) < 3:
                    continue
                original_name = parts[2]
                if original_name in seen:
                    continue
                seen.add(original_name)
                entries.append(
                    RewindEntry(
                        kind="files",
                        name=original_name,
                        detail=f"snapshot: {snapshot.name}",
                        timestamp=snapshot.stat().st_mtime,
                    )
                )
    except OSError as exc:
        logger.warning("Failed to list file checkpoints: %s", exc)

    # Sort newest first, cap
    entries.sort(key=lambda e: e.timestamp, reverse=True)
    return entries[:max_entries]


def parse_rewind_choice(choice: str) -> str:
    """Map a user's single-character choice to a restore action.

    Accepts ``c``/``conversation``, ``f``/``files``, ``b``/``both``,
    ``n``/``none``/``q``/``cancel``. Returns one of the
    ``RESTORE_*`` constants. Unknown input returns ``RESTORE_NONE``.
    """
    normalized = choice.strip().lower()
    if normalized in ("c", "conversation", "conv"):
        return RESTORE_CONVERSATION
    if normalized in ("f", "files", "file"):
        return RESTORE_FILES
    if normalized in ("b", "both"):
        return RESTORE_BOTH
    if normalized in ("n", "none", "q", "cancel", ""):
        return RESTORE_NONE
    return RESTORE_NONE


def restore_conversation(
    conversation: Any,
    checkpoint_name: str,
    cwd: Path,
) -> str:
    """Restore a conversation checkpoint into the given conversation.

    Returns a human-readable summary of what was restored.
    """
    from godspeed.context.checkpoint import load_checkpoint

    data = load_checkpoint(checkpoint_name, cwd)
    if data is None:
        return f"Checkpoint not found: {checkpoint_name}"

    conversation.clear()
    for msg in data.get("messages", []):
        role = msg.get("role", "")
        if role == "user":
            conversation.add_user_message(msg.get("content", ""))
        elif role == "assistant":
            conversation.add_assistant_message(
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
            )
        elif role == "tool":
            conversation.add_tool_result(
                tool_call_id=msg.get("tool_call_id", ""),
                content=msg.get("content", ""),
            )

    token_count = conversation.token_count
    msg_count = len(conversation.messages) - 1  # exclude system prompt
    return (
        f"Restored conversation checkpoint [{checkpoint_name}] "
        f"({msg_count} messages, {token_count:,} tokens)"
    )


def restore_files(
    cwd: Path,
    session_id: str,
    *,
    max_files: int = 20,
) -> str:
    """Restore the latest file snapshots for all edited files.

    Returns a human-readable summary of what was restored.
    """
    files_dir = edit_checkpoints_dir(cwd, session_id)
    if not files_dir.is_dir():
        return "No file checkpoints found for this session."

    restored: list[str] = []
    seen: set[str] = set()
    for snapshot in sorted(files_dir.iterdir(), reverse=True):
        parts = snapshot.name.split("_", 2)
        if len(parts) < 3:
            continue
        original_name = parts[2]
        if original_name in seen:
            continue
        seen.add(original_name)
        target = cwd / original_name
        result = restore_latest(target, cwd, session_id)
        if result is not None:
            restored.append(original_name)
        if len(restored) >= max_files:
            break

    if not restored:
        return "No file checkpoints could be restored."
    return f"Restored {len(restored)} file(s): {', '.join(restored)}"
