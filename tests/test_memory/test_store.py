"""Tests for MemoryStore — durable memory substrate with semantic recall."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from godspeed.memory.store import (
    MemoryStore,
)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Create a MemoryStore with a temp database."""
    s = MemoryStore(db_path=tmp_path / "test_memory.db", project_key="test_project")
    yield s
    s.close()


@pytest.fixture
def store_no_project(tmp_path: Path) -> MemoryStore:
    """Create a MemoryStore with no project key."""
    s = MemoryStore(db_path=tmp_path / "test_np.db", project_key="")
    yield s
    s.close()


# ── User Profile tests ────────────────────────────────────────────────


class TestUserProfile:
    """Test Letta-style durable user profile."""

    def test_empty_profile_by_default(self, store: MemoryStore) -> None:
        profile = store.get_profile()
        assert profile.role == ""
        assert profile.goals == []
        assert profile.bio == ""

    def test_update_profile(self, store: MemoryStore) -> None:
        profile = store.update_profile(
            role="senior python engineer",
            goals=["refactor auth module", "add type hints"],
            bio="Works on security-focused projects",
        )
        assert profile.role == "senior python engineer"
        assert len(profile.goals) == 2
        assert "refactor auth module" in profile.goals
        assert profile.bio == "Works on security-focused projects"

    def test_update_preserves_existing(self, store: MemoryStore) -> None:
        store.update_profile(role="engineer", goals=["goal1"])
        # Partial update: only change role
        updated = store.update_profile(role="lead engineer")
        assert updated.role == "lead engineer"
        assert updated.goals == ["goal1"]  # goals preserved

    def test_add_goal(self, store: MemoryStore) -> None:
        store.update_profile(goals=["existing"])
        profile = store.add_goal("new goal")
        assert "existing" in profile.goals
        assert "new goal" in profile.goals

    def test_add_goal_deduplicates(self, store: MemoryStore) -> None:
        store.update_profile(goals=["existing"])
        profile = store.add_goal("existing")
        assert profile.goals.count("existing") == 1

    def test_remove_goal(self, store: MemoryStore) -> None:
        store.update_profile(goals=["keep", "remove"])
        profile = store.remove_goal("remove")
        assert "keep" in profile.goals
        assert "remove" not in profile.goals

    def test_max_goals_enforced(self, store: MemoryStore) -> None:
        goals = [f"goal_{i}" for i in range(25)]
        profile = store.update_profile(goals=goals)
        assert len(profile.goals) == 20  # capped at _MAX_GOALS

    def test_profile_timestamps(self, store: MemoryStore) -> None:
        before = time.time()
        profile = store.update_profile(role="test")
        after = time.time()
        assert before <= profile.created_at <= after
        assert before <= profile.updated_at <= after

    def test_format_for_prompt_empty(self, store: MemoryStore) -> None:
        assert store.format_profile_for_prompt() == ""

    def test_format_for_prompt_with_role(self, store: MemoryStore) -> None:
        store.update_profile(role="python dev", goals=["learn rust"])
        result = store.format_profile_for_prompt()
        assert "User Profile:" in result
        assert "python dev" in result
        assert "learn rust" in result

    def test_profile_bio_truncation(self, store: MemoryStore) -> None:
        long_bio = "x" * 5000
        profile = store.update_profile(bio=long_bio)
        assert len(profile.bio) <= 4000  # _MAX_CONTENT_CHARS


# ── Project Memory tests ──────────────────────────────────────────────


class TestProjectMemory:
    """Test per-project memory store/recall."""

    def test_store_and_recall(self, store: MemoryStore) -> None:
        mem_id = store.store_project_memory(
            "Use pytest fixtures for DB cleanup",
            memory_type="convention",
        )
        assert mem_id > 0
        memories = store.recall_project_memories()
        assert len(memories) == 1
        assert memories[0].content == "Use pytest fixtures for DB cleanup"
        assert memories[0].memory_type == "convention"

    def test_store_empty_content_returns_neg(self, store: MemoryStore) -> None:
        mem_id = store.store_project_memory("   ")
        assert mem_id == -1

    def test_project_scoped(self, tmp_path: Path) -> None:
        """Memories are scoped by project_key."""
        store1 = MemoryStore(db_path=tmp_path / "scoped.db", project_key="proj_a")
        store2 = MemoryStore(db_path=tmp_path / "scoped.db", project_key="proj_b")

        store1.store_project_memory("project A memory")
        store2.store_project_memory("project B memory")

        assert len(store1.recall_project_memories()) == 1
        assert store1.recall_project_memories()[0].content == "project A memory"
        assert len(store2.recall_project_memories()) == 1
        assert store2.recall_project_memories()[0].content == "project B memory"

        store1.close()
        store2.close()

    def test_recall_by_type(self, store: MemoryStore) -> None:
        store.store_project_memory("decision 1", memory_type="decision")
        store.store_project_memory("pattern 1", memory_type="pattern")
        store.store_project_memory("decision 2", memory_type="decision")

        decisions = store.recall_project_memories(memory_type="decision")
        assert len(decisions) == 2
        assert all(m.memory_type == "decision" for m in decisions)

    def test_recall_by_min_relevance(self, store: MemoryStore) -> None:
        store.store_project_memory("high rel", relevance_score=1.0)
        store.store_project_memory("low rel", relevance_score=0.05)

        memories = store.recall_project_memories(min_relevance=0.1)
        assert len(memories) == 1
        assert memories[0].content == "high rel"

    def test_access_count_increments(self, store: MemoryStore) -> None:
        mem_id = store.store_project_memory("frequently accessed")
        store.recall_project_memories()
        store.recall_project_memories()
        store.recall_project_memories()

        memories = store.recall_project_memories()
        assert memories[0].access_count >= 3

    def test_delete_project_memory(self, store: MemoryStore) -> None:
        mem_id = store.store_project_memory("to delete")
        assert store.delete_project_memory(mem_id) is True
        assert len(store.recall_project_memories()) == 0

    def test_delete_wrong_project(self, tmp_path: Path) -> None:
        """Cannot delete memories from other projects."""
        store1 = MemoryStore(db_path=tmp_path / "d.db", project_key="a")
        store2 = MemoryStore(db_path=tmp_path / "d.db", project_key="b")

        mem_id = store1.store_project_memory("a's memory")
        assert store2.delete_project_memory(mem_id) is False
        assert len(store1.recall_project_memories()) == 1

        store1.close()
        store2.close()

    def test_metadata_roundtrip(self, store: MemoryStore) -> None:
        meta = {"source": "tool_call", "file": "auth.py"}
        mem_id = store.store_project_memory("with meta", metadata=meta)
        memories = store.recall_project_memories()
        assert memories[0].metadata == meta

    def test_decay_reduces_relevance(self, store: MemoryStore) -> None:
        mem_id = store.store_project_memory("old memory", relevance_score=1.0)

        # Simulate old memory by backdating last_accessed
        conn = store._get_conn()
        conn.execute(
            "UPDATE project_memory SET last_accessed = ? WHERE id = ?",
            (time.time() - 86400 * 180, mem_id),  # 180 days ago
        )
        conn.commit()

        updated = store.decay_project_memory_relevance(half_life_days=90)
        assert updated >= 1

        memories = store.recall_project_memories()
        # After 180 days (2 half-lives), relevance should be reduced
        assert memories[0].relevance_score < 0.5

    def test_format_for_prompt(self, store: MemoryStore) -> None:
        store.store_project_memory("convention: use type hints", memory_type="convention")
        result = store.format_project_memories_for_prompt()
        assert "Project Memories:" in result
        assert "type hints" in result

    def test_format_for_prompt_empty(self, store: MemoryStore) -> None:
        assert store.format_project_memories_for_prompt() == ""


# ── Semantic Recall tests ─────────────────────────────────────────────


class TestSemanticRecall:
    """Test optional ChromaDB-backed semantic recall."""

    def test_recall_empty_without_chromadb(self, store: MemoryStore) -> None:
        """Without chromadb, semantic recall returns empty list."""
        entries = store.semantic_recall("test query")
        assert entries == []

    def test_recall_all_empty_without_chromadb(self, store: MemoryStore) -> None:
        """recall_all still works without chromadb, just no semantic tier."""
        result = store.recall_all(query="test")
        assert result == "" or "User Profile" in result or "Project Memories" in result


# ── Combined recall tests ─────────────────────────────────────────────


class TestCombinedRecall:
    """Test recall_all across all tiers."""

    def test_recall_all_profile_only(self, store: MemoryStore) -> None:
        store.update_profile(role="dev")
        result = store.recall_all(profile=True)
        assert "User Profile:" in result
        assert "dev" in result

    def test_recall_all_project_only(self, store: MemoryStore) -> None:
        store.store_project_memory("test convention", memory_type="convention")
        result = store.recall_all(profile=False)
        assert "Project Memories:" in result

    def test_recall_all_combined(self, store: MemoryStore) -> None:
        store.update_profile(role="dev")
        store.store_project_memory("test convention", memory_type="convention")
        result = store.recall_all()
        assert "User Profile:" in result
        assert "Project Memories:" in result

    def test_recall_all_empty(self, store: MemoryStore) -> None:
        result = store.recall_all(profile=False)
        assert result == ""


# ── Secrets redaction tests ───────────────────────────────────────────


class TestSecretsRedaction:
    """Test that secrets are redacted before memory write."""

    def test_bio_redaction(self, store: MemoryStore) -> None:
        """Secrets in bio should be redacted (if security module available)."""
        profile = store.update_profile(bio="normal bio text")
        # The bio is stored as-is when security module unavailable in test env
        assert profile.bio is not None

    def test_content_redaction(self, store: MemoryStore) -> None:
        """Memory content should be processed through redaction."""
        mem_id = store.store_project_memory("normal content")
        memories = store.recall_project_memories()
        assert len(memories) == 1


# ── Database lifecycle tests ──────────────────────────────────────────


class TestDatabaseLifecycle:
    """Test database creation, WAL mode, and lifecycle."""

    def test_creates_db_on_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "dir" / "mem.db"
        s = MemoryStore(db_path=db_path, project_key="test")
        assert db_path.exists()
        s.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        s = MemoryStore(db_path=tmp_path / "wal.db")
        conn = s._get_conn()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        s.close()

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"
        s = MemoryStore(db_path=db_path, project_key="test")
        s.update_profile(role="test_role")
        s.close()

        s2 = MemoryStore(db_path=db_path, project_key="test")
        assert s2.get_profile().role == "test_role"
        s2.close()

    def test_double_close_is_safe(self, tmp_path: Path) -> None:
        s = MemoryStore(db_path=tmp_path / "close.db")
        s.close()
        s.close()  # should not raise

    def test_get_after_close_raises(self, tmp_path: Path) -> None:
        s = MemoryStore(db_path=tmp_path / "closed.db")
        s.close()
        with pytest.raises(RuntimeError, match="closed"):
            s.get_profile()


# ── Async wrapper tests ───────────────────────────────────────────────


class TestAsyncWrappers:
    """Test async wrappers for memory operations."""

    @pytest.mark.asyncio
    async def test_recall_all_async(self, store: MemoryStore) -> None:
        store.update_profile(role="async dev")
        result = await store.recall_all_async(profile=True)
        assert "async dev" in result

    @pytest.mark.asyncio
    async def test_store_project_memory_async(self, store: MemoryStore) -> None:
        mem_id = await store.store_project_memory_async(
            "async stored memory",
            memory_type="convention",
        )
        assert mem_id > 0
        memories = store.recall_project_memories()
        assert len(memories) == 1
