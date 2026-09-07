"""Tests for bounded-growth memory: row caps, cleanup, and ChromaDB cascade."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.memory.session import SessionMemory
from godspeed.memory.store import MemoryStore
from godspeed.memory.user_memory import UserMemory


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db", project_key="proj")
    yield s
    s.close()


@pytest.fixture
def user_memory(tmp_path: Path) -> UserMemory:
    um = UserMemory(db_path=tmp_path / "umemory.db")
    yield um
    um.close()


@pytest.fixture
def session_memory(tmp_path: Path) -> SessionMemory:
    sm = SessionMemory(db_path=tmp_path / "smemory.db")
    yield sm
    sm.close()


class TestMemoryStoreBounded:
    def test_store_project_memory_auto_prunes_beyond_cap(self, store: MemoryStore) -> None:
        for i in range(10):
            store.store_project_memory(f"memory {i}", max_rows=5)
        entries = store.recall_project_memories(limit=100)
        assert len(entries) <= 5

    def test_cleanup_old_entries_returns_counts(self, store: MemoryStore) -> None:
        for i in range(10):
            store.store_project_memory(f"memory {i}", max_rows=100)
        deleted_project, deleted_recall = store.cleanup_old_entries(project_cap=3)
        assert deleted_project == 7
        assert deleted_recall == 0
        assert len(store.recall_project_memories(limit=100)) == 3

    def test_cleanup_old_entries_noop_when_under_cap(self, store: MemoryStore) -> None:
        store.store_project_memory("only one")
        deleted_project, deleted_recall = store.cleanup_old_entries(project_cap=10)
        assert deleted_project == 0
        assert deleted_recall == 0

    def test_delete_project_memory_cascades_recall_entries(self, store: MemoryStore) -> None:
        mid = store.store_project_memory("to delete")
        assert store.delete_project_memory(mid) is True
        conn = store._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM recall_entries WHERE entry_key = ?",
            (f"mem_{mid}",),
        ).fetchone()
        assert row["cnt"] == 0

    def test_delete_project_memory_missing_returns_false(self, store: MemoryStore) -> None:
        assert store.delete_project_memory(99999) is False


class TestUserMemoryBounded:
    def test_record_correction_auto_prunes(self, user_memory: UserMemory) -> None:
        for i in range(505):
            user_memory.record_correction(f"orig {i}", f"fixed {i}")
        assert user_memory.correction_count() <= 500

    def test_cleanup_old_entries_returns_count(self, user_memory: UserMemory) -> None:
        for i in range(10):
            user_memory.record_correction(f"orig {i}", f"fixed {i}")
        deleted = user_memory.cleanup_old_entries(max_corrections=3)
        assert deleted == 7
        assert user_memory.correction_count() == 3

    def test_cleanup_old_entries_noop_when_under_cap(self, user_memory: UserMemory) -> None:
        user_memory.record_correction("a", "b")
        assert user_memory.cleanup_old_entries(max_corrections=10) == 0


class TestSessionMemoryBounded:
    def test_start_session_triggers_cleanup(self, session_memory: SessionMemory) -> None:
        for i in range(10):
            session_memory.start_session(f"sess-{i}", "model")
        # After 10 starts with cap 200, nothing pruned
        assert len(session_memory.list_sessions(limit=100)) == 10

    def test_cleanup_old_entries_prunes_sessions(self, session_memory: SessionMemory) -> None:
        for i in range(10):
            session_memory.start_session(f"sess-{i}", "model")
        deleted_sessions, deleted_events = session_memory.cleanup_old_entries(max_sessions=3)
        assert deleted_sessions == 7
        assert deleted_events == 0
        assert len(session_memory.list_sessions(limit=100)) == 3

    def test_cleanup_old_entries_prunes_events_per_session(
        self, session_memory: SessionMemory
    ) -> None:
        session_memory.start_session("sess-1", "model")
        for i in range(10):
            session_memory.record_event("sess-1", "tool_call", f"detail {i}")
        deleted_sessions, deleted_events = session_memory.cleanup_old_entries(
            max_sessions=100, max_events_per_session=3
        )
        assert deleted_sessions == 0
        assert deleted_events == 7
        assert session_memory.event_count("sess-1") == 3
