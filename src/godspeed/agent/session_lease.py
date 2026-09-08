"""Single-writer session lease for turn-level durability.

A lease file at ``{cwd}/.godspeed/leases/{session_id}.json`` guarantees a
single writer per session across processes. Acquisition is atomic via
``O_CREAT | O_EXCL``; staleness is decided purely by heartbeat age so a
crashed process's lease can be stolen after ``STALE_SECONDS``.

Fail-closed: when a live lease exists, ``acquire()`` raises
``LeaseHeldError`` naming the holding pid so the caller can refuse to
start a second writer.
"""

from __future__ import annotations

import asyncio
import calendar
import contextlib
import json
import logging
import os
import socket
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STALE_SECONDS = 120
HEARTBEAT_SECONDS = 30

_GODSPEED_DIR_NAME = ".godspeed"
_LEASES_DIR_NAME = "leases"


def _iso_now() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _epoch_from_iso(iso: str) -> float:
    """Parse an ISO-8601 UTC timestamp back to epoch seconds."""
    try:
        return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return 0.0


class LeaseHeldError(RuntimeError):
    """Raised when a live lease is held by another process."""

    def __init__(self, session_id: str, other_pid: int, lease_path: Path) -> None:
        self.session_id = session_id
        self.other_pid = other_pid
        self.lease_path = lease_path
        super().__init__(
            f"Session {session_id!r} is already active in another process "
            f"(pid={other_pid}). Refusing to start a second writer. "
            f"Lease file: {lease_path}"
        )


class SessionLease:
    """File-based single-writer lease with a background heartbeat.

    Usage::

        async with SessionLease(session_id, cwd):
            ...

    ``acquire()`` is atomic (``O_CREAT | O_EXCL``). A lease whose heartbeat
    is older than ``STALE_SECONDS`` is considered dead and is stolen
    (deleted, then re-acquired). A live lease raises ``LeaseHeldError``.
    """

    def __init__(self, session_id: str, cwd: Path) -> None:
        self.session_id = session_id
        self.lease_path = cwd / _GODSPEED_DIR_NAME / _LEASES_DIR_NAME / f"{session_id}.json"
        self._heartbeat_task: asyncio.Task | None = None
        self._acquired = False
        self._acquired_at = _iso_now()

    # -- lifecycle -----------------------------------------------------

    async def __aenter__(self) -> SessionLease:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()

    async def acquire(self) -> None:
        """Acquire the lease, stealing it if stale.

        Raises:
            LeaseHeldError: a live lease is held by another process.
        """
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if self._is_stale():
                logger.warning(
                    "Lease %s is stale — stealing from pid=%s",
                    self.session_id,
                    self._read_pid(),
                )
                self.lease_path.unlink(missing_ok=True)
                fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            else:
                raise LeaseHeldError(self.session_id, self._read_pid(), self.lease_path) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._payload(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        self._acquired = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Lease acquired session=%s path=%s", self.session_id, self.lease_path)

    async def release(self) -> None:
        """Stop the heartbeat and remove the lease file. Idempotent."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        if self._acquired:
            self.lease_path.unlink(missing_ok=True)
            self._acquired = False
            logger.info("Lease released session=%s", self.session_id)

    def release_sync(self) -> None:
        """Best-effort synchronous release (no heartbeat cancellation).

        Used as a fallback when the event loop is already closed (e.g.
        during interpreter shutdown) and an async release is impossible.
        """
        if self._acquired:
            self.lease_path.unlink(missing_ok=True)
            self._acquired = False
            logger.info("Lease released (sync) session=%s", self.session_id)

    def schedule_release_on_task_done(self) -> None:
        """Release the lease when the current asyncio task finishes.

        The agent loop cannot wrap its ~400-line body in ``try/finally``
        without re-indenting it, so this registers a done-callback that
        schedules an async release on the running loop. Falls back to a
        synchronous release if the loop is already closed.
        """
        task = asyncio.current_task()
        if task is None:
            return

        def _on_done(_task: asyncio.Task) -> None:
            try:
                loop = asyncio.get_running_loop()
                self._release_task = loop.create_task(self.release())
            except RuntimeError:
                self.release_sync()

        task.add_done_callback(_on_done)

    # -- heartbeat -----------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                self._write_heartbeat()
            except OSError as exc:
                logger.warning(
                    "Lease heartbeat write failed session=%s error=%s",
                    self.session_id,
                    exc,
                )

    def _write_heartbeat(self) -> None:
        """Atomically refresh the heartbeat timestamp (tmp + replace)."""
        tmp = self.lease_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._payload()), encoding="utf-8")
        os.replace(tmp, self.lease_path)

    # -- helpers -------------------------------------------------------

    def _payload(self) -> dict:
        return {
            "session_id": self.session_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": self._acquired_at,
            "heartbeat_at": _iso_now(),
        }

    def _read_lease(self) -> dict | None:
        try:
            data = json.loads(self.lease_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _read_pid(self) -> int:
        data = self._read_lease()
        if data is None:
            return -1
        pid = data.get("pid")
        return pid if isinstance(pid, int) else -1

    def _is_stale(self) -> bool:
        """A lease is stale when its heartbeat is older than STALE_SECONDS.

        An unreadable/corrupt lease is treated as stale so a wedged file
        cannot block the session forever.
        """
        data = self._read_lease()
        if data is None:
            logger.warning("Lease %s unreadable — treating as stale", self.session_id)
            return True
        heartbeat = data.get("heartbeat_at")
        if not isinstance(heartbeat, str):
            return True
        return time.time() - _epoch_from_iso(heartbeat) > STALE_SECONDS
