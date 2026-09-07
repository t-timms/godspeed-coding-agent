"""Durable memory substrate — Letta-inspired layered persistent memory.

Provides three memory tiers on top of the existing SQLite preferences/corrections:

1. **User profile** — durable structured user identity (role, goals, bio)
   that persists across sessions and informs all interactions.
2. **Project memory** — per-project decisions, patterns, and context that
   survive compaction and session boundaries.
3. **Semantic recall** — optional ChromaDB-backed vector search over memories
   for relevance-filtered retrieval during context assembly.

Design references:
- Letta durable memory (59.1% vs 41.6% Terminal-Bench)
- AHE finding: tools/middleware/memory carry gains; prompt-only regresses

All methods are synchronous. Wrap in ``asyncio.to_thread()`` from async code.
Secrets are redacted before every write. SQLite WAL mode preserved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_SCHEMA_VERSION = 2

_MEMORY_INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- User profile: durable structured identity
CREATE TABLE IF NOT EXISTS user_profile (
    id TEXT PRIMARY KEY DEFAULT 'default',
    role TEXT NOT NULL DEFAULT '',
    goals TEXT NOT NULL DEFAULT '[]',
    bio TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Project memory: per-project decisions, patterns, context
CREATE TABLE IF NOT EXISTS project_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'general',
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    relevance_score REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_project_key ON project_memory(project_key);
CREATE INDEX IF NOT EXISTS idx_project_type ON project_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_project_relevance ON project_memory(relevance_score DESC);

-- Semantic recall index metadata (vectors live in ChromaDB)
CREATE TABLE IF NOT EXISTS recall_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_type TEXT NOT NULL,
    entry_key TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_id TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recall_type ON recall_entries(entry_type);
CREATE INDEX IF NOT EXISTS idx_recall_key ON recall_entries(entry_key);
"""

# Maximum length for memory content before truncation (tokens ≈ chars/4)
_MAX_CONTENT_CHARS = 4000

# Maximum number of project memories to return per query
_MAX_PROJECT_MEMORIES = 20

# Hard row cap per project — oldest memories beyond this are pruned on cleanup
_MAX_PROJECT_MEMORY_ROWS: int = 500

# Hard row cap for recall_entries
_MAX_RECALL_ENTRIES: int = 2000

# Relevance decay factor: memories lose relevance exponentially with age
_RELEVANCE_DECAY_HALF_LIFE_DAYS = 90

# Maximum user profile goals
_MAX_GOALS = 20


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single memory entry for semantic recall."""

    id: int
    entry_type: str
    entry_key: str
    content: str
    embedding_id: str | None
    created_at: float


@dataclass(frozen=True, slots=True)
class ProjectMemoryEntry:
    """A project-scoped memory entry."""

    id: int
    project_key: str
    memory_type: str
    content: str
    metadata: dict[str, Any]
    relevance_score: float
    access_count: int
    created_at: float
    last_accessed: float


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Durable user profile for cross-session continuity."""

    role: str = ""
    goals: list[str] = field(default_factory=list)
    bio: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


class MemoryStore:
    """Durable memory substrate backed by SQLite + optional ChromaDB.

    Extends the existing UserMemory (preferences/corrections) with:

    - User profile (Letta-style durable identity)
    - Project memory (per-project decisions/patterns)
    - Semantic recall (optional ChromaDB vector search)

    Thread-safe via WAL mode. Secrets redacted before every write.

    Args:
        db_path: Path to SQLite database. Defaults to ``~/.godspeed/memory.db``.
        project_key: Project identifier for scoped memory. Defaults to ``""``.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        project_key: str = "",
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".godspeed" / "memory.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._project_key = project_key
        self._conn: sqlite3.Connection | None = None
        self._recall_collection: Any | None = None
        self._recall_available: bool | None = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with memory schema."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_MEMORY_INIT_SQL)

        # Check/set schema version
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_profile'"
        )
        if cursor.fetchone() is not None:
            cursor = self._conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            if row is None or row[0] < _MEMORY_SCHEMA_VERSION:
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (_MEMORY_SCHEMA_VERSION,),
                )
                self._conn.commit()

        logger.info(
            "memory_store.init db_path=%s project_key=%s",
            self._db_path,
            self._project_key,
        )

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

    @property
    def project_key(self) -> str:
        """Return the project key."""
        return self._project_key

    def _get_conn(self) -> sqlite3.Connection:
        """Return the active connection, raising if closed."""
        if self._conn is None:
            msg = "Memory store connection is closed"
            raise RuntimeError(msg)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._recall_collection = None

    # ── Secrets redaction ──────────────────────────────────────────────

    @staticmethod
    def _redact_content(text: str) -> str:
        """Redact secrets from content before persisting to memory.

        Delegates to the shared ``redact_or_fail`` helper which is
        fail-closed: if the redaction module is unavailable the entire
        string is replaced with ``[REDACTED]``.
        """
        from godspeed.memory.redact import redact_or_fail

        return redact_or_fail(text)

    @staticmethod
    def _truncate_content(text: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        """Truncate content to prevent unbounded memory growth.

        The truncation marker is included within the budget so the
        returned string never exceeds ``max_chars``.
        """
        if len(text) <= max_chars:
            return text
        marker = f"\n...[truncated {len(text) - max_chars} chars]"
        return text[: max_chars - len(marker)] + marker

    @staticmethod
    def _redact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        from godspeed.memory.redact import redact_or_fail

        def _deep(value: Any) -> Any:
            if isinstance(value, str):
                return redact_or_fail(value)
            if isinstance(value, dict):
                return {k: _deep(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_deep(v) for v in value]
            return value

        return {k: _deep(v) for k, v in metadata.items()}

    # ── User Profile (Letta-style durable identity) ────────────────────

    def get_profile(self) -> UserProfile:
        """Get the user profile. Returns empty profile if none exists."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM user_profile WHERE id = 'default'")
        row = cursor.fetchone()
        if row is None:
            return UserProfile()

        goals: list[str] = []
        try:
            goals = json.loads(row["goals"])
        except (json.JSONDecodeError, TypeError):
            goals = []

        return UserProfile(
            role=row["role"] or "",
            goals=goals,
            bio=row["bio"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_profile(
        self,
        role: str | None = None,
        goals: list[str] | None = None,
        bio: str | None = None,
    ) -> UserProfile:
        """Update the user profile (partial update, merges goals).

        Args:
            role: User's role (e.g. "senior python engineer").
            goals: List of goals (replaces existing if provided).
            bio: User bio (replaces existing if provided).

        Returns:
            The updated profile.
        """
        current = self.get_profile()
        now = time.time()

        new_role = role if role is not None else current.role
        new_goals = goals if goals is not None else current.goals
        new_bio = bio if bio is not None else current.bio

        # Enforce limits
        new_goals = new_goals[:_MAX_GOALS]
        new_bio = self._truncate_content(new_bio)

        new_role = self._redact_content(new_role)
        new_goals = [self._redact_content(g) for g in new_goals]
        new_bio = self._redact_content(new_bio)

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO user_profile (id, role, goals, bio, created_at, updated_at) "
            "VALUES ('default', ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "role = excluded.role, goals = excluded.goals, "
            "bio = excluded.bio, updated_at = excluded.updated_at",
            (new_role, json.dumps(new_goals), new_bio, current.created_at or now, now),
        )
        conn.commit()
        logger.info(
            "memory_store.profile_updated role=%s goals=%d bio_len=%d",
            new_role,
            len(new_goals),
            len(new_bio),
        )
        return self.get_profile()

    def add_goal(self, goal: str) -> UserProfile:
        """Add a goal to the user profile. Deduplicates."""
        profile = self.get_profile()
        goal = self._truncate_content(goal, max_chars=200)
        goal = self._redact_content(goal)
        if goal and goal not in profile.goals:
            new_goals = [*profile.goals, goal]
            return self.update_profile(goals=new_goals)
        return profile

    def remove_goal(self, goal: str) -> UserProfile:
        """Remove a goal from the user profile."""
        profile = self.get_profile()
        new_goals = [g for g in profile.goals if g != goal]
        return self.update_profile(goals=new_goals)

    def format_profile_for_prompt(self) -> str:
        """Format the user profile for system prompt injection.

        Returns a compact representation suitable for context assembly.
        """
        profile = self.get_profile()
        if not profile.role and not profile.goals and not profile.bio:
            return ""

        parts: list[str] = ["User Profile:"]
        if profile.role:
            parts.append(f"  Role: {profile.role}")
        if profile.goals:
            parts.append(f"  Goals: {'; '.join(profile.goals)}")
        if profile.bio:
            # Truncate bio for prompt injection (keep under 500 chars)
            bio = profile.bio[:500]
            if len(profile.bio) > 500:
                bio += "..."
            parts.append(f"  Context: {bio}")
        return "\n".join(parts)

    # ── Project Memory (per-project decisions/patterns) ─────────────────

    def cleanup_old_entries(
        self,
        project_cap: int = _MAX_PROJECT_MEMORY_ROWS,
        recall_cap: int = _MAX_RECALL_ENTRIES,
    ) -> tuple[int, int]:
        """Prune oldest project memories and recall entries beyond caps.

        Returns:
            (project_deleted, recall_deleted) counts.
        """
        conn = self._get_conn()

        project_deleted = 0
        cursor = conn.execute(
            "SELECT COUNT(*) AS cnt FROM project_memory WHERE project_key = ?",
            (self._project_key,),
        )
        cnt = cursor.fetchone()
        if cnt is not None and cnt["cnt"] > project_cap:
            excess = cnt["cnt"] - project_cap
            rows = conn.execute(
                "SELECT id FROM project_memory WHERE project_key = ? "
                "ORDER BY last_accessed ASC LIMIT ?",
                (self._project_key, excess),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM project_memory WHERE id IN ({placeholders})",  # noqa: S608
                    ids,
                )
                project_deleted = len(ids)

        recall_deleted = 0
        cursor = conn.execute("SELECT COUNT(*) AS cnt FROM recall_entries")
        cnt = cursor.fetchone()
        if cnt is not None and cnt["cnt"] > recall_cap:
            excess = cnt["cnt"] - recall_cap
            rows = conn.execute(
                "SELECT id FROM recall_entries ORDER BY created_at ASC LIMIT ?",
                (excess,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM recall_entries WHERE id IN ({placeholders})",  # noqa: S608
                    ids,
                )
                recall_deleted = len(ids)

        if project_deleted or recall_deleted:
            conn.commit()
            logger.info(
                "memory_store.cleanup project_deleted=%d recall_deleted=%d",
                project_deleted,
                recall_deleted,
            )
        return project_deleted, recall_deleted

    def store_project_memory(
        self,
        content: str,
        memory_type: str = "general",
        metadata: dict[str, Any] | None = None,
        relevance_score: float = 1.0,
        max_rows: int = _MAX_PROJECT_MEMORY_ROWS,
    ) -> int:
        """Store a project-scoped memory.

        Args:
            content: Memory content (will be redacted and truncated).
            memory_type: Category (decision, pattern, error_fix, convention, general).
            metadata: Optional metadata dict (serialized as JSON).
            relevance_score: Initial relevance score (0.0-1.0).

        Returns:
            The memory entry ID.
        """
        content = self._redact_content(content)
        content = self._truncate_content(content)

        if not content.strip():
            return -1

        conn = self._get_conn()
        now = time.time()
        meta_json = json.dumps(self._redact_metadata(metadata or {}))
        cursor = conn.execute(
            "INSERT INTO project_memory "
            "(project_key, memory_type, content, metadata, relevance_score, "
            "access_count, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (
                self._project_key,
                memory_type,
                content,
                meta_json,
                relevance_score,
                now,
                now,
            ),
        )
        conn.commit()
        memory_id = cursor.lastrowid
        logger.info(
            "memory_store.project_stored id=%d type=%s project=%s content_len=%d",
            memory_id,
            memory_type,
            self._project_key,
            len(content),
        )

        # Update semantic index if available
        self._update_recall_index(memory_id, memory_type, content)

        # Auto-prune if beyond row cap
        cursor_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM project_memory WHERE project_key = ?",
            (self._project_key,),
        )
        cnt_row = cursor_count.fetchone()
        if cnt_row is not None and cnt_row["cnt"] > max_rows:
            self.cleanup_old_entries(project_cap=max_rows)

        return memory_id  # type: ignore[return-value]

    def recall_project_memories(
        self,
        memory_type: str | None = None,
        limit: int = _MAX_PROJECT_MEMORIES,
        min_relevance: float = 0.0,
    ) -> list[ProjectMemoryEntry]:
        """Recall project memories, ordered by relevance and recency.

        Args:
            memory_type: Optional filter by type.
            limit: Maximum entries to return.
            min_relevance: Minimum relevance score threshold.

        Returns:
            List of ProjectMemoryEntry objects.
        """
        conn = self._get_conn()
        now = time.time()

        if memory_type:
            cursor = conn.execute(
                "SELECT * FROM project_memory "
                "WHERE project_key = ? AND memory_type = ? AND relevance_score >= ? "
                "ORDER BY relevance_score DESC, last_accessed DESC LIMIT ?",
                (self._project_key, memory_type, min_relevance, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM project_memory "
                "WHERE project_key = ? AND relevance_score >= ? "
                "ORDER BY relevance_score DESC, last_accessed DESC LIMIT ?",
                (self._project_key, min_relevance, limit),
            )

        entries: list[ProjectMemoryEntry] = []
        for row in cursor.fetchall():
            # Update access tracking
            conn.execute(
                "UPDATE project_memory SET access_count = access_count + 1, "
                "last_accessed = ? WHERE id = ?",
                (now, row["id"]),
            )
            meta: dict[str, Any] = {}
            if row["metadata"]:
                meta = json.loads(row["metadata"])
            entries.append(
                ProjectMemoryEntry(
                    id=row["id"],
                    project_key=row["project_key"],
                    memory_type=row["memory_type"],
                    content=row["content"],
                    metadata={k: v for k, v in meta.items() if k != "embedding_id"},
                    relevance_score=row["relevance_score"],
                    access_count=row["access_count"],
                    created_at=row["created_at"],
                    last_accessed=row["last_accessed"],
                )
            )

        if entries:
            conn.commit()

        return entries

    def decay_project_memory_relevance(
        self, half_life_days: float = _RELEVANCE_DECAY_HALF_LIFE_DAYS
    ) -> int:
        """Apply time-based relevance decay to all project memories.

        Memories that are accessed frequently resist decay. This models
        the Ebbinghaus forgetting curve — accessed memories stay relevant.

        Args:
            half_life_days: Half-life in days for the exponential decay.

        Returns:
            Number of memories updated.
        """
        conn = self._get_conn()
        half_life_secs = half_life_days * 86400
        now = time.time()

        cursor = conn.execute(
            "SELECT id, relevance_score, access_count, last_accessed "
            "FROM project_memory WHERE project_key = ?",
            (self._project_key,),
        )

        updated = 0
        for row in cursor.fetchall():
            age_secs = now - row["last_accessed"]
            decay = 0.5 ** (age_secs / half_life_secs)
            # Boost for frequently accessed memories (log scale)
            access_boost = min(1.0 + 0.1 * (row["access_count"] ** 0.5), 2.0)
            new_score = min(1.0, row["relevance_score"] * decay * access_boost)
            new_score = max(0.0, new_score)

            if abs(new_score - row["relevance_score"]) > 0.01:
                conn.execute(
                    "UPDATE project_memory SET relevance_score = ? WHERE id = ?",
                    (new_score, row["id"]),
                )
                updated += 1

        conn.commit()
        if updated:
            logger.info("memory_store.decay_updated count=%d", updated)
        return updated

    def delete_project_memory(self, memory_id: int) -> bool:
        """Delete a project memory by ID, cascading to ChromaDB if available."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM project_memory WHERE id = ? AND project_key = ?",
            (memory_id, self._project_key),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            self._delete_recall_entry(memory_id)
        conn.commit()
        return deleted

    def format_project_memories_for_prompt(self, limit: int = 10) -> str:
        """Format project memories for system prompt injection.

        Returns compact, relevance-filtered project context.
        """
        memories = self.recall_project_memories(limit=limit, min_relevance=0.1)
        if not memories:
            return ""

        lines = ["Project Memories:"]
        for m in memories:
            content = m.content[:200]
            if len(m.content) > 200:
                content += "..."
            lines.append(f"  [{m.memory_type}] {content}")
        return "\n".join(lines)

    # ── Semantic Recall (optional ChromaDB) ─────────────────────────────

    def _is_recall_available(self) -> bool:
        """Check if ChromaDB is available for semantic recall."""
        if self._recall_available is not None:
            return self._recall_available
        try:
            import chromadb  # noqa: F401

            self._recall_available = True
        except ImportError:
            self._recall_available = False
        return self._recall_available

    def _get_recall_collection(self) -> Any | None:
        """Get or create the ChromaDB collection for memory recall."""
        if self._recall_collection is not None:
            return self._recall_collection
        if not self._is_recall_available():
            return None

        import chromadb

        recall_path = self._db_path.parent / "recall_index"
        recall_path.mkdir(parents=True, exist_ok=True)

        try:
            client = chromadb.PersistentClient(path=str(recall_path))
            self._recall_collection = client.get_or_create_collection(
                name="memory_recall",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.warning("recall_index init failed: %s", exc)
            self._recall_available = False
            return None

        return self._recall_collection

    def _update_recall_index(
        self,
        memory_id: int,
        entry_type: str,
        content: str,
    ) -> None:
        """Add or update a memory entry in the semantic recall index."""
        collection = self._get_recall_collection()
        if collection is None:
            return

        embedding_id = f"mem_{memory_id}"
        try:
            collection.upsert(
                ids=[embedding_id],
                documents=[content],
                metadatas=[
                    {
                        "entry_type": entry_type,
                        "memory_id": memory_id,
                        "project_key": self._project_key,
                    }
                ],
            )
            # Track the embedding ID in SQLite
            conn = self._get_conn()
            conn.execute(
                "UPDATE project_memory SET metadata = json_set(metadata, '$.embedding_id', ?) "
                "WHERE id = ?",
                (embedding_id, memory_id),
            )
            conn.commit()
        except Exception as exc:
            logger.debug("recall_index upsert failed: %s", exc)

    def _delete_recall_entry(self, memory_id: int) -> None:
        """Remove a memory from ChromaDB + SQLite recall_entries."""
        collection = self._get_recall_collection()
        if collection is not None:
            try:
                collection.delete(ids=[f"mem_{memory_id}"])
            except Exception as exc:
                logger.debug("recall_index delete failed: %s", exc)
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM recall_entries WHERE entry_key = ?",
            (f"mem_{memory_id}",),
        )

    def semantic_recall(
        self,
        query: str,
        entry_type: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Semantic search over all memories via ChromaDB embeddings.

        Gracefully degrades to empty list when chromadb is not installed.

        Args:
            query: Natural language query.
            entry_type: Optional filter by memory type.
            top_k: Maximum results.

        Returns:
            List of MemoryEntry objects ordered by relevance.
        """
        collection = self._get_recall_collection()
        if collection is None:
            return []

        try:
            where_filter: dict[str, Any] | None = None
            if entry_type:
                where_filter = {"entry_type": entry_type}

            doc_count = collection.count()
            if doc_count == 0:
                return []

            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(top_k, doc_count),
            }
            if where_filter:
                kwargs["where"] = where_filter

            results = collection.query(**kwargs)

            entries: list[MemoryEntry] = []
            if results and results["ids"] and results["ids"][0]:
                for doc_id, doc, meta in zip(
                    results["ids"][0],
                    results["documents"][0] if results["documents"] else [],
                    results["metadatas"][0] if results["metadatas"] else [],
                    strict=False,
                ):
                    entries.append(
                        MemoryEntry(
                            id=meta.get("memory_id", 0),
                            entry_type=meta.get("entry_type", ""),
                            entry_key=doc_id,
                            content=doc,
                            embedding_id=doc_id,
                            created_at=0.0,
                        )
                    )
            return entries

        except Exception as exc:
            logger.debug("semantic_recall failed: %s", exc)
            return []

    # ── Combined recall (all tiers) ────────────────────────────────────

    def recall_all(
        self,
        query: str | None = None,
        project_limit: int = 10,
        profile: bool = True,
    ) -> str:
        """Recall from all memory tiers and format for system prompt.

        This is the primary entry point for context assembly. Returns
        a formatted string combining:

        1. User profile (if available)
        2. Project memories (relevance-filtered)
        3. Semantic recall results (if query provided and ChromaDB available)

        Args:
            query: Optional query for semantic recall.
            project_limit: Max project memories to include.
            profile: Whether to include user profile.

        Returns:
            Formatted memory context for system prompt injection.
        """
        parts: list[str] = []

        # Tier 1: User profile
        if profile:
            profile_text = self.format_profile_for_prompt()
            if profile_text:
                parts.append(profile_text)

        # Tier 2: Project memories
        project_text = self.format_project_memories_for_prompt(limit=project_limit)
        if project_text:
            parts.append(project_text)

        # Tier 3: Semantic recall
        if query and self._is_recall_available():
            semantic_entries = self.semantic_recall(query, top_k=5)
            if semantic_entries:
                lines = ["Recalled Context:"]
                for entry in semantic_entries:
                    content = entry.content[:150]
                    if len(entry.content) > 150:
                        content += "..."
                    lines.append(f"  [{entry.entry_type}] {content}")
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # ── Async wrappers ──────────────────────────────────────────────────

    async def recall_all_async(
        self,
        query: str | None = None,
        project_limit: int = 10,
        profile: bool = True,
    ) -> str:
        """Async wrapper for recall_all (runs sync DB in thread)."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.recall_all(query=query, project_limit=project_limit, profile=profile),
        )

    async def store_project_memory_async(
        self,
        content: str,
        memory_type: str = "general",
        metadata: dict[str, Any] | None = None,
        relevance_score: float = 1.0,
    ) -> int:
        """Async wrapper for store_project_memory."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.store_project_memory(
                content,
                memory_type=memory_type,
                metadata=metadata,
                relevance_score=relevance_score,
            ),
        )
