"""Hash-chained JSONL audit trail — tamper-evident logging.

Every action is recorded as a typed event. Each record includes the SHA-256
hash of the previous record, creating a verifiable chain. Secrets are
redacted before writing.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from godspeed.audit.events import AuditEventType, AuditRecord
from godspeed.audit.redactor import redact_audit_detail

logger = logging.getLogger(__name__)


class AuditWriteError(OSError):
    """Raised when an audit record cannot be durably persisted.

    A tamper-evident log is only meaningful if every event reaches disk.
    Callers must fail closed: stop executing tool calls until audit recovers.
    """


class AuditTrail:
    """Append-only, hash-chained JSONL audit log.

    Async-safe: all writes are serialized through an asyncio lock to protect
    the hash chain (_sequence and _prev_hash) from concurrent mutations.
    Blocking I/O (write, flush, fsync) is offloaded to a thread pool so the
    event loop never stalls.

    One file per session: {session_id}.audit.jsonl
    """

    def __init__(self, log_dir: Path, session_id: str) -> None:
        self._log_dir = log_dir
        self._session_id = session_id
        self._prev_hash = ""
        self._record_count = 0
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._file: Any | None = None
        self._writes_since_sync = 0
        # fsync every N records — 10x fewer syscalls while maintaining durability
        self._fsync_interval = 10

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"{session_id}.audit.jsonl"

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def record_count(self) -> int:
        return self._record_count

    def _write_record(self, event: AuditRecord) -> None:
        """Synchronous helper: write the record to disk under the lock."""
        try:
            line = event.model_dump_json() + "\n"
            file_just_opened = self._file is None
            if file_just_opened:
                self._file = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115
            self._file.write(line)  # type: ignore[union-attr]
            self._file.flush()  # type: ignore[union-attr]
            self._writes_since_sync += 1
            if file_just_opened or self._writes_since_sync >= self._fsync_interval:
                os.fsync(self._file.fileno())  # type: ignore[union-attr]
                self._writes_since_sync = 0
        except OSError as exc:
            logger.error(
                "Audit write failed path=%s error=%s — refusing to advance chain",
                self._log_path,
                exc,
            )
            raise AuditWriteError(f"audit write failed: {exc}") from exc

    def record(
        self,
        event_type: AuditEventType | str,
        detail: dict | None = None,
        outcome: str = "success",
        **extra: Any,
    ) -> AuditRecord:
        """Append a record to the audit trail (sync version).

        Thread-safe. For async contexts, prefer :meth:`arecord` to avoid
        blocking the event loop. Extra keyword arguments set optional record
        fields (parent_session_id, is_sidechain, cost_usd, lines_added,
        lines_removed) before hashing.
        """
        safe_detail = redact_audit_detail(detail or {})

        with self._sync_lock:
            event = AuditRecord(
                session_id=self._session_id,
                sequence=self._sequence,
                action_type=AuditEventType(event_type)
                if isinstance(event_type, str)
                else event_type,
                action_detail=safe_detail,
                outcome=outcome,
                prev_hash=self._prev_hash,
            )
            for key, value in extra.items():
                setattr(event, key, value)

            record_json = event.model_dump_json(exclude={"record_hash"})
            event.record_hash = hashlib.sha256(record_json.encode()).hexdigest()

            try:
                self._write_record(event)
            except AuditWriteError:
                raise

            self._prev_hash = event.record_hash
            self._sequence += 1
            self._record_count += 1

        logger.debug(
            "Audit record type=%s outcome=%s seq=%d hash=%s",
            event_type,
            outcome,
            event.sequence,
            event.record_hash[:12],
        )
        return event

    async def arecord(
        self,
        event_type: AuditEventType | str,
        detail: dict | None = None,
        outcome: str = "success",
        **extra: Any,
    ) -> AuditRecord:
        """Append a record to the audit trail (async version).

        Blocking I/O (file open, write, flush, fsync) is offloaded to a
        thread pool so the asyncio event loop never stalls. Extra keyword
        arguments set optional record fields before hashing.
        """
        safe_detail = redact_audit_detail(detail or {})

        async with self._lock:
            event = AuditRecord(
                session_id=self._session_id,
                sequence=self._sequence,
                action_type=AuditEventType(event_type)
                if isinstance(event_type, str)
                else event_type,
                action_detail=safe_detail,
                outcome=outcome,
                prev_hash=self._prev_hash,
            )
            for key, value in extra.items():
                setattr(event, key, value)

            record_json = event.model_dump_json(exclude={"record_hash"})
            event.record_hash = hashlib.sha256(record_json.encode()).hexdigest()

            try:
                await asyncio.to_thread(self._write_record, event)
            except AuditWriteError:
                raise

            self._prev_hash = event.record_hash
            self._sequence += 1
            self._record_count += 1

        logger.debug(
            "Audit record type=%s outcome=%s seq=%d hash=%s",
            event_type,
            outcome,
            event.sequence,
            event.record_hash[:12],
        )
        return event

    def close(self) -> None:
        """Close the audit file handle, doing a final fsync."""
        if self._file is not None:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError:
                logger.warning("Failed to fsync audit log during close")
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    async def aclose(self) -> None:
        """Async close: offloads fsync/close to a thread pool."""
        if self._file is not None:
            try:
                await asyncio.to_thread(self._file.flush)
                await asyncio.to_thread(os.fsync, self._file.fileno())
            except OSError:
                logger.warning("Failed to fsync audit log during aclose")
            await asyncio.to_thread(self._file.close)
            self._file = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> AuditTrail:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Safety net: if the trail is GC'd without an explicit close(),
        # flush and release the underlying file handle.  Swallow all
        # errors — we are in a destructor and must not raise.
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
            self._file = None

    def compress_session(self) -> Path | None:
        """Compress the current session's audit log to gzip.

        Rotates {session}.audit.jsonl → {session}.audit.jsonl.gz.
        The uncompressed file is removed after successful compression.

        Returns:
            Path to the compressed file, or None if nothing to compress.
        """
        if not self._log_path.exists() or self._log_path.stat().st_size == 0:
            return None

        # Close persistent handle before compression so the file is free
        self.close()

        gz_path = self._log_path.with_suffix(".jsonl.gz")
        try:
            with open(self._log_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                f_out.writelines(f_in)
            self._log_path.unlink()
            logger.info(
                "Compressed audit log session=%s records=%d",
                self._session_id,
                self._record_count,
            )
            return gz_path
        except OSError as exc:
            logger.error("Audit compression failed session=%s error=%s", self._session_id, exc)
            # Clean up partial gz if it exists
            if gz_path.exists():
                gz_path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _read_log_lines(log_path: Path) -> list[str]:
        """Read lines from a log file, handling both plain and gzipped formats."""
        if log_path.suffix == ".gz" or str(log_path).endswith(".jsonl.gz"):
            with gzip.open(log_path, "rt", encoding="utf-8") as f:
                return f.readlines()
        with open(log_path, encoding="utf-8") as f:
            return f.readlines()

    def verify_chain(self, log_path: Path | None = None) -> tuple[bool, str]:
        """Verify the hash chain integrity of a session log.

        Supports both plain .jsonl and compressed .jsonl.gz files.

        Args:
            log_path: Path to verify. Defaults to current session's log.
                      Checks both uncompressed and compressed paths.

        Returns:
            (is_valid, message) tuple.
        """
        target = log_path or self._log_path
        # If the uncompressed file doesn't exist, try the compressed version
        if not target.exists():
            gz_target = target.with_suffix(".jsonl.gz")
            if gz_target.exists():
                target = gz_target
            else:
                return True, "No audit log to verify"

        try:
            lines = self._read_log_lines(target)
        except (OSError, gzip.BadGzipFile) as exc:
            return False, f"Cannot read log: {exc}"

        prev_hash = ""
        line_num = 0

        for raw_line in lines:
            line_num += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                return False, f"Line {line_num}: invalid JSON"

            record = AuditRecord.model_validate(data)

            if record.prev_hash != prev_hash:
                return (
                    False,
                    f"Line {line_num}: prev_hash mismatch "
                    f"(expected {prev_hash[:12]}..., got {record.prev_hash[:12]}...)",
                )

            record_json = record.model_copy(update={"record_hash": ""}).model_dump_json(
                exclude={"record_hash"}
            )
            expected_hash = hashlib.sha256(record_json.encode()).hexdigest()

            if record.record_hash != expected_hash:
                return (
                    False,
                    f"Line {line_num}: record_hash mismatch "
                    f"(expected {expected_hash[:12]}..., got {record.record_hash[:12]}...)",
                )

            prev_hash = record.record_hash

        return True, f"Chain verified: {line_num} records"

    def cleanup_expired(self, retention_days: int = 30) -> int:
        """Remove audit logs older than retention_days.

        Scans the log directory for session files (both plain and compressed)
        and deletes those whose last modification time exceeds the retention
        period. Skips the current session's log file.

        Returns:
            Number of expired log files removed.
        """
        import time

        if retention_days <= 0:
            return 0

        cutoff = time.time() - (retention_days * 86400)
        removed = 0

        # Check both .jsonl and .jsonl.gz files
        patterns = ["*.audit.jsonl", "*.audit.jsonl.gz"]
        for pattern in patterns:
            for log_file in self._log_dir.glob(pattern):
                # Never delete the current session's log (compressed or not)
                if (
                    log_file == self._log_path
                    or log_file.stem.replace(".audit", "") == self._session_id
                ):
                    continue
                try:
                    if log_file.stat().st_mtime < cutoff:
                        log_file.unlink()
                        removed += 1
                        logger.info("Removed expired audit log path=%s", log_file.name)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove expired audit log path=%s error=%s", log_file, exc
                    )

        if removed:
            logger.info("Audit cleanup removed=%d retention_days=%d", removed, retention_days)
        return removed
