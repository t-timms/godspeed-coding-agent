"""Tests for secrets redaction across all memory write paths.

Covers:
- redact_or_fail() fail-closed behaviour
- sk-*/github_pat redacted in preferences (set)
- sk-*/github_pat redacted in corrections (record_correction)
- sk-*/github_pat redacted in session events (record_event)
- sk-*/github_pat redacted in user profile (role, goals, bio)
- truncation caps honoured
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.memory.redact import redact_or_fail
from godspeed.memory.session import SessionMemory
from godspeed.memory.store import MemoryStore
from godspeed.memory.user_memory import UserMemory


_OPENAI_KEY = "sk-proj-abc123def456ghi789jkl012mno"
_GITHUB_PAT = "github_pat_11ABCDEF0123456789_abcdefghijklmnopqrstuvwxyz0123456"


# ── redact_or_fail unit tests ──────────────────────────────────────────


class TestRedactOrFail:
    """Unit tests for the shared redaction helper."""

    def test_clean_text_unchanged(self) -> None:
        assert redact_or_fail("hello world") == "hello world"

    def test_empty_string_passthrough(self) -> None:
        assert redact_or_fail("") == ""

    def test_openai_key_redacted(self) -> None:
        result = redact_or_fail(f"key is {_OPENAI_KEY}")
        assert _OPENAI_KEY not in result
        assert "[REDACTED]" in result

    def test_github_pat_redacted(self) -> None:
        result = redact_or_fail(f"token: {_GITHUB_PAT}")
        assert _GITHUB_PAT not in result
        assert "[REDACTED]" in result

    def test_fail_closed_on_import_error(self) -> None:
        with patch(
            "godspeed.memory.redact.redact_or_fail",
            side_effect=ImportError("no module"),
        ):
            pass
        # Simulate the actual failure path by patching the internal import
        with patch.dict("sys.modules", {"godspeed.security.secrets": None}):
            result = redact_or_fail(f"secret={_OPENAI_KEY}")
        assert _OPENAI_KEY not in result

    def test_fail_closed_returns_full_redacted_on_exception(self) -> None:
        with patch(
            "godspeed.security.secrets.redact_secrets",
            side_effect=RuntimeError("boom"),
        ):
            result = redact_or_fail(f"token={_OPENAI_KEY}")
        assert result == "[REDACTED]"
        assert _OPENAI_KEY not in result


# ── UserMemory redaction ──────────────────────────────────────────────


@pytest.fixture
def user_mem(tmp_path: Path) -> UserMemory:
    m = UserMemory(db_path=tmp_path / "test_mem.db")
    yield m
    m.close()


class TestUserMemoryRedaction:
    """Verify secrets are redacted in preferences and corrections."""

    def test_set_redacts_openai_key(self, user_mem: UserMemory) -> None:
        user_mem.set("api_key", f"my key is {_OPENAI_KEY}")
        stored = user_mem.get("api_key")
        assert stored is not None
        assert _OPENAI_KEY not in stored
        assert "[REDACTED]" in stored

    def test_set_redacts_github_pat(self, user_mem: UserMemory) -> None:
        user_mem.set("token", _GITHUB_PAT)
        stored = user_mem.get("token")
        assert stored is not None
        assert _GITHUB_PAT not in stored

    def test_record_correction_redacts_secrets(self, user_mem: UserMemory) -> None:
        cid = user_mem.record_correction(
            original=f"used key {_OPENAI_KEY}",
            corrected=f"use {_GITHUB_PAT} instead",
            context="fix secret leak",
        )
        corrections = user_mem.get_corrections()
        assert len(corrections) == 1
        assert _OPENAI_KEY not in corrections[0]["original"]
        assert _GITHUB_PAT not in corrections[0]["corrected"]
        assert "[REDACTED]" in corrections[0]["original"]
        assert "[REDACTED]" in corrections[0]["corrected"]

    def test_clean_prefs_stored_unchanged(self, user_mem: UserMemory) -> None:
        user_mem.set("theme", "dark")
        assert user_mem.get("theme") == "dark"


# ── SessionMemory redaction ──────────────────────────────────────────


@pytest.fixture
def sess_mem(tmp_path: Path) -> SessionMemory:
    m = SessionMemory(db_path=tmp_path / "session_test.db")
    yield m
    m.close()


class TestSessionMemoryRedaction:
    """Verify secrets are redacted in session event details."""

    def _start(self, sm: SessionMemory) -> None:
        sm.start_session("s1", "gpt-4")

    def test_record_event_redacts_openai_key(self, sess_mem: SessionMemory) -> None:
        self._start(sess_mem)
        eid = sess_mem.record_event("s1", "tool_call", detail=f"exec with {_OPENAI_KEY}")
        events = sess_mem.get_events("s1")
        assert len(events) == 1
        assert _OPENAI_KEY not in events[0]["detail"]
        assert "[REDACTED]" in events[0]["detail"]

    def test_record_event_redacts_github_pat(self, sess_mem: SessionMemory) -> None:
        self._start(sess_mem)
        sess_mem.record_event("s1", "git_push", detail=f"pushed token {_GITHUB_PAT}")
        events = sess_mem.get_events("s1")
        assert _GITHUB_PAT not in events[0]["detail"]

    def test_clean_events_stored_unchanged(self, sess_mem: SessionMemory) -> None:
        self._start(sess_mem)
        sess_mem.record_event("s1", "session_start", detail="normal log line")
        events = sess_mem.get_events("s1")
        assert events[0]["detail"] == "normal log line"


# ── MemoryStore profile redaction ─────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "test_store.db", project_key="test")
    yield s
    s.close()


class TestStoreProfileRedaction:
    """Verify secrets are redacted in user profile role, goals, and bio."""

    def test_role_redacted(self, store: MemoryStore) -> None:
        profile = store.update_profile(role=f"engineer key={_OPENAI_KEY}")
        assert _OPENAI_KEY not in profile.role

    def test_goals_redacted(self, store: MemoryStore) -> None:
        profile = store.update_profile(
            goals=[f"deploy with {_GITHUB_PAT}", "ship feature X"],
        )
        for g in profile.goals:
            assert _GITHUB_PAT not in g
        assert "ship feature X" in profile.goals[1]

    def test_bio_redacted(self, store: MemoryStore) -> None:
        profile = store.update_profile(bio=f"My key: {_OPENAI_KEY}")
        assert _OPENAI_KEY not in profile.bio
        assert "[REDACTED]" in profile.bio

    def test_add_goal_redacted(self, store: MemoryStore) -> None:
        profile = store.add_goal(f"ship with {_OPENAI_KEY}")
        assert _OPENAI_KEY not in profile.goals[0]

    def test_clean_profile_unchanged(self, store: MemoryStore) -> None:
        profile = store.update_profile(
            role="python dev",
            goals=["write tests"],
            bio="likes coffee",
        )
        assert profile.role == "python dev"
        assert profile.goals == ["write tests"]
        assert profile.bio == "likes coffee"

    def test_project_memory_redacted(self, store: MemoryStore) -> None:
        mid = store.store_project_memory(f"config: {_OPENAI_KEY}")
        entries = store.recall_project_memories()
        assert len(entries) == 1
        assert _OPENAI_KEY not in entries[0].content


# ── Truncation caps ──────────────────────────────────────────────────


class TestTruncationCaps:
    """Verify truncation prevents unbounded memory growth."""

    def test_set_truncates_long_value(self, user_mem: UserMemory) -> None:
        long_val = "x" * 5000
        user_mem.set("big", long_val)
        stored = user_mem.get("big")
        assert stored is not None
        assert len(stored) <= 4000

    def test_correction_truncates_long_field(self, user_mem: UserMemory) -> None:
        long_text = "y" * 3000
        user_mem.record_correction(original=long_text, corrected="short")
        corrections = user_mem.get_corrections()
        assert len(corrections[0]["original"]) <= 2000

    def test_event_truncates_long_detail(self, sess_mem: SessionMemory) -> None:
        sess_mem.start_session("s2", "model")
        long_detail = "z" * 5000
        sess_mem.record_event("s2", "tool_call", detail=long_detail)
        events = sess_mem.get_events("s2")
        assert len(events[0]["detail"]) <= 4000

    def test_store_profile_bio_truncated(self, store: MemoryStore) -> None:
        long_bio = "a" * 5000
        profile = store.update_profile(bio=long_bio)
        assert len(profile.bio) <= 4000
