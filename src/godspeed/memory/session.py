"""Session memory — persistent session event logging.

Records session events (start, end, tool calls, errors) to SQLite for
cross-session learning and context resumption.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_SESSIONS = 200
_MAX_EVENTS_PER_SESSION = 200

_MAX_EVENT_DETAIL_CHARS = 4000

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    project_dir TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON session_events(event_type);

CREATE TABLE IF NOT EXISTS session_messages (
    session_id TEXT PRIMARY KEY,
    messages TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""


class SessionMemory:
    """Persistent session event store backed by SQLite.

    Records session lifecycle events for cross-session learning:
    - session_start / session_end
    - tool_call, tool_error
    - user_correction
    - compaction

    Shares the same database as UserMemory (WAL mode, safe concurrent access).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".godspeed" / "memory.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_INIT_SQL)
        self._conn: sqlite3.Connection | None = conn
        logger.info("session_memory.init db_path=%s", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Return the active connection, raising if closed."""
        if self._conn is None:
            msg = "Database connection is closed"
            raise RuntimeError(msg)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Session lifecycle -----------------------------------------------------

    def start_session(self, session_id: str, model: str, project_dir: str = "") -> None:
        """Record a new session start."""
        from godspeed.memory.redact import redact_or_fail

        model = redact_or_fail(model)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, model, started_at, project_dir) "
            "VALUES (?, ?, ?, ?)",
            (session_id, model, time.time(), project_dir),
        )
        conn.commit()
        self.cleanup_old_entries()
        logger.info("session_memory.start session_id=%s model=%s", session_id, model)

    def end_session(self, session_id: str, summary: str = "") -> None:
        """Record session end with optional summary.

        If ``summary`` is empty, a compact summary is auto-generated from the
        session's stored conversation messages (see :meth:`generate_summary`),
        so the stored summary is never empty for a session that had messages.
        """
        from godspeed.memory.redact import redact_or_fail

        if not summary:
            messages = self.get_messages(session_id)
            if messages:
                summary = self.generate_summary(messages)
        summary = redact_or_fail(summary)[:_MAX_EVENT_DETAIL_CHARS]
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (time.time(), summary, session_id),
        )
        conn.commit()
        logger.info("session_memory.end session_id=%s", session_id)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent sessions, newest first."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_most_recent_session(self) -> dict[str, Any] | None:
        """Return the most recently started session, or None if none exist."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    # -- Message persistence ----------------------------------------------------

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist the full conversation message list for a session.

        Messages are stored as JSON so a later ``--continue``/``--resume`` can
        restore the exact conversation. Only the non-system messages are
        expected (the system prompt is rebuilt per-session).
        """
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO session_messages (session_id, messages, updated_at) "
            "VALUES (?, ?, ?)",
            (session_id, json.dumps(messages, ensure_ascii=False), time.time()),
        )
        conn.commit()
        logger.info(
            "session_memory.save_messages session_id=%s count=%d", session_id, len(messages)
        )

    def get_messages(self, session_id: str) -> list[dict[str, Any]] | None:
        """Return the persisted message list for a session, or None if none stored."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT messages FROM session_messages WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["messages"])
        except (json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, list) else None

    @staticmethod
    def generate_summary(messages: list[dict[str, Any]], max_chars: int = 4000) -> str:
        """Generate a compact, deterministic summary of a conversation.

        Counts user/assistant turns and tool calls, and includes the first
        user message as a topic hint. Used to populate ``end_session`` when no
        explicit summary is provided.
        """
        user_turns = 0
        assistant_turns = 0
        tool_calls = 0
        first_user: str = ""
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                user_turns += 1
                if not first_user:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        first_user = content.strip()
            elif role == "assistant":
                assistant_turns += 1
                if msg.get("tool_calls"):
                    tool_calls += len(msg["tool_calls"])
            elif role == "tool":
                tool_calls += 1

        parts = [f"turns={user_turns} assistant={assistant_turns} tool_calls={tool_calls}"]
        if first_user:
            snippet = first_user.replace("\n", " ")[:200]
            parts.append(f"topic: {snippet}")
        summary = " | ".join(parts)
        return summary[:max_chars]

    # -- Events ----------------------------------------------------------------

    def record_event(
        self,
        session_id: str,
        event_type: str,
        detail: str = "",
    ) -> int:
        """Record a session event. Returns the event ID."""
        from godspeed.memory.redact import redact_or_fail

        detail = redact_or_fail(detail)[:_MAX_EVENT_DETAIL_CHARS]
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO session_events (session_id, event_type, detail, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, event_type, detail, time.time()),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get events for a session, optionally filtered by type."""
        conn = self._get_conn()
        if event_type:
            cursor = conn.execute(
                "SELECT * FROM session_events "
                "WHERE session_id = ? AND event_type = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, event_type, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        return [dict(row) for row in cursor.fetchall()]

    def event_count(self, session_id: str) -> int:
        """Count events for a session."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def cleanup_old_entries(
        self,
        max_sessions: int = _MAX_SESSIONS,
        max_events_per_session: int = _MAX_EVENTS_PER_SESSION,
    ) -> tuple[int, int]:
        """Prune oldest sessions and their events beyond caps.

        Returns:
            (sessions_deleted, events_deleted) counts.
        """
        conn = self._get_conn()

        sessions_deleted = 0
        cursor = conn.execute("SELECT COUNT(*) AS cnt FROM sessions")
        cnt = cursor.fetchone()
        if cnt is not None and cnt["cnt"] > max_sessions:
            excess = cnt["cnt"] - max_sessions
            old = conn.execute(
                "SELECT id FROM sessions ORDER BY started_at ASC LIMIT ?",
                (excess,),
            ).fetchall()
            ids = [r["id"] for r in old]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM session_events WHERE session_id IN ({placeholders})",  # noqa: S608
                    ids,
                )
                conn.execute(
                    f"DELETE FROM sessions WHERE id IN ({placeholders})",  # noqa: S608
                    ids,
                )
                sessions_deleted = len(ids)

        events_deleted = 0
        cursor = conn.execute(
            "SELECT session_id, COUNT(*) AS cnt FROM session_events "
            "GROUP BY session_id HAVING cnt > ?",
            (max_events_per_session,),
        )
        for row in cursor.fetchall():
            sid = row["session_id"]
            excess = row["cnt"] - max_events_per_session
            old_events = conn.execute(
                "SELECT id FROM session_events "
                "WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
                (sid, excess),
            ).fetchall()
            eids = [r["id"] for r in old_events]
            if eids:
                placeholders = ",".join("?" for _ in eids)
                conn.execute(
                    f"DELETE FROM session_events WHERE id IN ({placeholders})",  # noqa: S608
                    eids,
                )
                events_deleted += len(eids)

        if sessions_deleted or events_deleted:
            conn.commit()
            logger.info(
                "session_memory.cleanup sessions_deleted=%d events_deleted=%d",
                sessions_deleted,
                events_deleted,
            )
        return sessions_deleted, events_deleted
