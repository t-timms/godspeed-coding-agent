"""Persist conversation messages to JSONL for training data collection.

Every user message, assistant response (with tool_calls), tool result, and
compaction event is logged to a per-session JSONL file. This captures the
full conversation flow that the audit trail misses (audit records tool metadata
but not the actual conversation content).

Gated on ``GodspeedSettings.log_conversations``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Append-only JSONL logger for conversation messages.

    Each line is a self-contained JSON object with at minimum a ``role`` field.
    Compatible with OpenAI fine-tuning format after transformation by the
    ``TrainingExporter``.
    """

    _FLUSH_INTERVAL = 10  # flush every N writes; close() always flushes

    def __init__(self, session_id: str, output_dir: Path) -> None:
        self._session_id = session_id
        self._path = output_dir / f"{session_id}.conversation.jsonl"
        self._file: TextIO | None = None
        self._step = 0
        self._writes_since_flush = 0

    # -- public API ----------------------------------------------------------

    def log_system(self, content: str) -> None:
        """Log the system prompt (typically once per session)."""
        self._write({"role": "system", "content": content})

    def log_user(self, content: str | list[dict[str, Any]]) -> None:
        """Log a user message (plain text or multimodal content blocks)."""
        self._write({"role": "user", "content": content})

    def log_assistant(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        thinking: str = "",
    ) -> None:
        """Log an assistant response with optional tool calls and thinking."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if thinking:
            msg["thinking"] = thinking
        self._write(msg)

    def log_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        is_error: bool = False,
        is_dangerous: bool = False,
    ) -> None:
        """Log a tool execution result."""
        self._step += 1
        self._write(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": content,
                "is_error": is_error,
                "dangerous_command": is_dangerous,
                "step": self._step,
            }
        )

    def log_compaction(
        self,
        summary: str,
        messages_before: int,
        messages_after: int,
    ) -> None:
        """Log a compaction event with the summary text.

        The audit trail records compaction but discards the actual summary.
        This preserves it for training data reconstruction.
        """
        self._write(
            {
                "role": "meta",
                "event": "compaction",
                "summary": summary,
                "messages_before": messages_before,
                "messages_after": messages_after,
            }
        )

    def log_session_end(
        self,
        exit_reason: str,
        exit_code: int,
        iterations_used: int,
        tool_call_count: int,
        tool_error_count: int,
        duration_seconds: float,
        cost_usd: float,
        must_fix_injections: int = 0,
        lines_added: int | None = None,
        lines_removed: int | None = None,
    ) -> None:
        """Terminal record per session.

        Fields mirror the audit-trail session_end detail so the two streams
        stay comparable. Downstream RL (GRPO) reads exit_code to shape
        rewards: +1.0 on SUCCESS, -0.5 on TOOL_ERROR, -1.0 on
        MAX_ITERATIONS, etc. See ml-lab phase4_grpo.yaml for the mapping.

        `must_fix_injections` (v2.6.0+) is a quality signal: agents that
        triggered many fix-required injections are less efficient per unit
        of successful work.
        """
        record = {
            "role": "session_end",
            "exit_reason": exit_reason,
            "exit_code": exit_code,
            "iterations_used": iterations_used,
            "tool_call_count": tool_call_count,
            "tool_error_count": tool_error_count,
            "must_fix_injections": must_fix_injections,
            "duration_seconds": round(duration_seconds, 3),
            "cost_usd": round(cost_usd, 6),
        }
        if lines_added is not None:
            record["lines_added"] = lines_added
        if lines_removed is not None:
            record["lines_removed"] = lines_removed
        self._write(record)

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                logger.debug("Could not flush/close conversation log file")
            finally:
                self._file = None

    def flush(self) -> None:
        """Flush buffered writes to disk without closing."""
        if self._file is not None:
            self._file.flush()
            self._writes_since_flush = 0

    @property
    def path(self) -> Path:
        """Path to the JSONL file."""
        return self._path

    @property
    def step_count(self) -> int:
        """Number of tool results logged so far."""
        return self._step

    # -- internals -----------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        """Serialize one record as a JSON line and flush."""
        record["timestamp"] = datetime.now(tz=UTC).isoformat()
        record["session_id"] = self._session_id

        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._file.write(line + "\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._FLUSH_INTERVAL:
            self._file.flush()
            self._writes_since_flush = 0

    def __del__(self) -> None:
        self.close()
