"""Tests for the single-writer session lease (turn-level durability).

Covers:
- atomic acquisition (O_CREAT | O_EXCL) and lease file contents
- LeaseHeldError when a live lease is held by another process
- stale-lease stealing (heartbeat age > STALE_SECONDS)
- corrupt/unreadable lease treated as stale
- release removes the file; context-manager usage
- background heartbeat refreshes heartbeat_at
- schedule_release_on_task_done releases when the task finishes
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from godspeed.agent.session_lease import (
    STALE_SECONDS,
    LeaseHeldError,
    SessionLease,
)


def _old_iso(seconds_ago: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds_ago))


def _write_lease(path: Path, *, pid: int, heartbeat_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session_id": "sess-1",
                "pid": pid,
                "host": "test-host",
                "acquired_at": heartbeat_at,
                "heartbeat_at": heartbeat_at,
            }
        ),
        encoding="utf-8",
    )


class TestAcquire:
    def test_acquire_creates_lease_file(self, tmp_path: Path) -> None:
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            assert lease.lease_path.exists()
            data = json.loads(lease.lease_path.read_text(encoding="utf-8"))
            assert data["session_id"] == "sess-1"
            assert data["pid"] == __import__("os").getpid()
            assert "host" in data
            assert "acquired_at" in data
            assert "heartbeat_at" in data
        finally:
            asyncio.run(lease.release())

    def test_acquire_uses_monkeypatched_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.session_lease.os.getpid", lambda: 424242)
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            data = json.loads(lease.lease_path.read_text(encoding="utf-8"))
            assert data["pid"] == 424242
        finally:
            asyncio.run(lease.release())

    def test_lease_file_mode_is_private(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("Windows ignores os.open mode bits")
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            mode = lease.lease_path.stat().st_mode & 0o777
            assert mode == 0o600
        finally:
            asyncio.run(lease.release())


class TestLeaseHeld:
    def test_live_lease_raises_held_error(self, tmp_path: Path) -> None:
        first = SessionLease("sess-1", tmp_path)
        asyncio.run(first.acquire())
        try:
            second = SessionLease("sess-1", tmp_path)
            with pytest.raises(LeaseHeldError) as excinfo:
                asyncio.run(second.acquire())
            assert excinfo.value.other_pid == __import__("os").getpid()
            assert excinfo.value.session_id == "sess-1"
            assert "pid" in str(excinfo.value)
        finally:
            asyncio.run(first.release())

    def test_released_lease_can_be_reacquired(self, tmp_path: Path) -> None:
        first = SessionLease("sess-1", tmp_path)
        asyncio.run(first.acquire())
        asyncio.run(first.release())
        second = SessionLease("sess-1", tmp_path)
        asyncio.run(second.acquire())
        try:
            assert second.lease_path.exists()
        finally:
            asyncio.run(second.release())


class TestStaleStealing:
    def test_stale_lease_is_stolen(self, tmp_path: Path) -> None:
        _write_lease(
            tmp_path / ".godspeed" / "leases" / "sess-1.json",
            pid=999,
            heartbeat_at=_old_iso(STALE_SECONDS + 60),
        )
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            data = json.loads(lease.lease_path.read_text(encoding="utf-8"))
            assert data["pid"] == __import__("os").getpid()
        finally:
            asyncio.run(lease.release())

    def test_fresh_lease_is_not_stolen(self, tmp_path: Path) -> None:
        _write_lease(
            tmp_path / ".godspeed" / "leases" / "sess-1.json",
            pid=999,
            heartbeat_at=_old_iso(1),
        )
        lease = SessionLease("sess-1", tmp_path)
        with pytest.raises(LeaseHeldError):
            asyncio.run(lease.acquire())

    def test_corrupt_lease_is_treated_as_stale(self, tmp_path: Path) -> None:
        path = tmp_path / ".godspeed" / "leases" / "sess-1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            assert lease.lease_path.exists()
        finally:
            asyncio.run(lease.release())


class TestRelease:
    def test_release_removes_file(self, tmp_path: Path) -> None:
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        assert lease.lease_path.exists()
        asyncio.run(lease.release())
        assert not lease.lease_path.exists()

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        asyncio.run(lease.release())
        asyncio.run(lease.release())

    def test_context_manager_releases(self, tmp_path: Path) -> None:
        async def _run() -> None:
            async with SessionLease("sess-1", tmp_path) as lease:
                assert lease.lease_path.exists()
            assert not lease.lease_path.exists()

        asyncio.run(_run())


class TestHeartbeat:
    def test_heartbeat_refreshes_timestamp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.session_lease.HEARTBEAT_SECONDS", 0.05)
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            first = json.loads(lease.lease_path.read_text(encoding="utf-8"))["heartbeat_at"]
            asyncio.run(asyncio.sleep(0.15))
            second = json.loads(lease.lease_path.read_text(encoding="utf-8"))["heartbeat_at"]
            assert second >= first
        finally:
            asyncio.run(lease.release())

    def test_heartbeat_write_failure_is_nonfatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.session_lease.HEARTBEAT_SECONDS", 0.05)

        def _boom() -> None:
            raise OSError("disk full")

        monkeypatch.setattr("godspeed.agent.session_lease.SessionLease._write_heartbeat", _boom)
        lease = SessionLease("sess-1", tmp_path)
        asyncio.run(lease.acquire())
        try:
            asyncio.run(asyncio.sleep(0.15))
            assert lease.lease_path.exists()
        finally:
            asyncio.run(lease.release())


class TestScheduleReleaseOnTaskDone:
    def test_release_happens_when_task_finishes(self, tmp_path: Path) -> None:
        lease_path = tmp_path / ".godspeed" / "leases" / "sess-1.json"

        async def _main() -> None:
            lease = SessionLease("sess-1", tmp_path)
            await lease.acquire()
            lease.schedule_release_on_task_done()
            assert lease.lease_path.exists()

        async def _runner() -> None:
            task = asyncio.create_task(_main())
            await task
            # Let the done-callback's scheduled release task run.
            await asyncio.sleep(0.05)

        asyncio.run(_runner())
        assert not lease_path.exists()
