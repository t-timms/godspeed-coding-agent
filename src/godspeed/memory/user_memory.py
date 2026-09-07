"""User memory — persistent preferences and corrections stored in SQLite.

Database lives at ~/.godspeed/memory.db with WAL mode for safe concurrent access.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_MAX_PREF_VALUE_CHARS = 4000
_MAX_CORRECTIONS = 500
_MAX_CORRECTION_CHARS = 2000

_INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original TEXT NOT NULL,
    corrected TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corrections_created ON corrections(created_at DESC);
"""


class UserMemory:
    """Persistent user memory backed by SQLite.

    Stores:
    - preferences: key/value pairs (coding style, model prefs, etc.)
    - corrections: user corrections of agent behavior for learning

    Thread-safe via WAL mode. All writes are serialized by SQLite.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".godspeed" / "memory.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_INIT_SQL)

        # Check/set schema version
        cursor = self._conn.execute("SELECT version FROM schema_version")
        row = cursor.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
            )
            self._conn.commit()
        logger.info("user_memory.init db_path=%s", self._db_path)

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

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

    # -- Preferences -----------------------------------------------------------

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get a preference value by key."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row is not None else default

    def set(self, key: str, value: str) -> None:
        """Set a preference value (upsert)."""
        from godspeed.memory.redact import redact_or_fail

        key = redact_or_fail(key)
        value = redact_or_fail(value)[:_MAX_PREF_VALUE_CHARS]
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, value, now),
        )
        conn.commit()
        logger.debug("user_memory.set key=%s", key)

    def delete(self, key: str) -> bool:
        """Delete a preference. Returns True if it existed."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def list_preferences(self) -> list[dict[str, Any]]:
        """List all preferences as dicts with key, value, updated_at."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT key, value, updated_at FROM preferences ORDER BY key")
        return [dict(row) for row in cursor.fetchall()]

    # -- Corrections -----------------------------------------------------------

    def record_correction(self, original: str, corrected: str, context: str = "") -> int:
        """Record a user correction. Returns the correction ID."""
        from godspeed.memory.redact import redact_or_fail

        original = redact_or_fail(original)[:_MAX_CORRECTION_CHARS]
        corrected = redact_or_fail(corrected)[:_MAX_CORRECTION_CHARS]
        context = redact_or_fail(context)[:_MAX_CORRECTION_CHARS]
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            "INSERT INTO corrections (original, corrected, context, created_at) "
            "VALUES (?, ?, ?, ?)",
            (original, corrected, context, now),
        )
        conn.commit()
        correction_id = cursor.lastrowid
        logger.info(
            "user_memory.record_correction id=%d original=%s corrected=%s",
            correction_id,
            original[:50],
            corrected[:50],
        )
        self.cleanup_old_entries()
        return correction_id  # type: ignore[return-value]

    def get_corrections(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent corrections, newest first."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id, original, corrected, context, created_at "
            "FROM corrections ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_correction(self, correction_id: int) -> bool:
        """Delete a correction by ID. Returns True if it existed."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))
        conn.commit()
        return cursor.rowcount > 0

    def correction_count(self) -> int:
        """Return total number of corrections."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM corrections")
        row = cursor.fetchone()
        return row[0] if row else 0

    def cleanup_old_entries(self, max_corrections: int = _MAX_CORRECTIONS) -> int:
        """Prune oldest corrections beyond the cap. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) AS cnt FROM corrections")
        cnt = cursor.fetchone()
        if cnt is None or cnt["cnt"] <= max_corrections:
            return 0
        excess = cnt["cnt"] - max_corrections
        rows = conn.execute(
            "SELECT id FROM corrections ORDER BY created_at ASC LIMIT ?",
            (excess,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        del_cursor = conn.execute(
            f"DELETE FROM corrections WHERE id IN ({placeholders})",  # noqa: S608
            ids,
        )
        conn.commit()
        return del_cursor.rowcount
