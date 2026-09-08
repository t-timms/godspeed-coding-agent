"""Append-only turn journal for crash-durable turn tracking.

Each turn (one LLM call plus its tool executions) is recorded as a JSONL
record at ``{cwd}/.godspeed/turns/{session_id}.jsonl``. On restart,
``reconcile()`` reports which turns completed and which were left
incomplete by a crash.

Journal writes are fail-safe: a write error logs a warning and never
breaks the agent loop (unlike the audit trail, which is fail-closed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_GODSPEED_DIR_NAME = ".godspeed"
_TURNS_DIR_NAME = "turns"
_FINGERPRINT_TAIL = 20  # last N messages hashed into the request fingerprint

OUTCOME_RUNNING = "running"
OUTCOME_COMPLETED = "completed"
OUTCOME_INTERRUPTED = "interrupted"
OUTCOME_ERROR = "error"

_FINISHED_OUTCOMES = frozenset({OUTCOME_COMPLETED, OUTCOME_ERROR})


def _iso_now() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def request_fingerprint(
    model: str,
    messages: list,
    tools: list,
    tail: int = _FINGERPRINT_TAIL,
) -> str:
    """Stable sha256 fingerprint of the request that starts a turn.

    Hashes the model name, the last ``tail`` messages, and the tool list
    so a restarted session can recognize an identical request.
    """
    payload = {
        "model": model,
        "messages_tail": messages[-tail:],
        "tools": tools,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ReconcileResult:
    """Outcome of reconciling a journal after a restart."""

    last_completed_seq: int
    incomplete: list[int]
    verdict: str  # "clean" | "incomplete" | "empty" | "error"

    @property
    def has_incomplete(self) -> bool:
        return bool(self.incomplete)


class TurnJournal:
    """Append-only JSONL journal of turns for one session.

    Usage::

        journal = TurnJournal(session_id, cwd)
        seq = journal.start_turn(fingerprint)
        ...
        journal.complete_turn(seq)
    """

    def __init__(self, session_id: str, cwd: Path) -> None:
        self.session_id = session_id
        self.path = cwd / _GODSPEED_DIR_NAME / _TURNS_DIR_NAME / f"{session_id}.jsonl"
        self._running: dict[int, dict] = {}  # seq -> record for in-flight turns
        self._last_seq = self._scan_last_seq()

    def _scan_last_seq(self) -> int:
        """Read the journal once at init to find the highest seq."""
        last = 0
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                seq = record.get("seq")
                if isinstance(seq, int) and seq > last:
                    last = seq
        except OSError:
            pass
        return last

    def start_turn(self, fingerprint: str) -> int:
        """Append a ``running`` record and return its seq.

        Idempotent per turn: each call appends a fresh record with
        ``seq = last + 1``.
        """
        seq = self._last_seq + 1
        self._last_seq = seq
        record = {
            "seq": seq,
            "started_at": _iso_now(),
            "request_fingerprint": fingerprint,
            "finished_at": None,
            "outcome": OUTCOME_RUNNING,
        }
        self._running[seq] = record
        self._append(record)
        return seq

    def complete_turn(
        self,
        seq: int,
        outcome: str = OUTCOME_COMPLETED,
        error: str | None = None,
    ) -> None:
        """Append the completion record for a started turn.

        ``outcome`` is one of ``"completed" | "interrupted" | "error"``.
        """
        started = self._running.get(seq, {})
        record = {
            "seq": seq,
            "started_at": started.get("started_at", _iso_now()),
            "request_fingerprint": started.get("request_fingerprint", ""),
            "finished_at": _iso_now(),
            "outcome": outcome,
            "error": error,
        }
        self._running.pop(seq, None)
        self._append(record)

    def _append(self, record: dict) -> None:
        """Fail-safe append: log a warning, never raise."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
        except OSError as exc:
            logger.warning(
                "Turn journal write failed session=%s seq=%s error=%s",
                self.session_id,
                record.get("seq"),
                exc,
            )


def reconcile(session_id: str, cwd: Path) -> ReconcileResult:
    """Reconcile a journal after a restart.

    The last record per seq wins (a turn may have a ``running`` record
    followed by a completion record). ``completed`` and ``error`` count as
    finished; ``running`` and ``interrupted`` count as incomplete.

    Verdicts:
    - ``"clean"``: journal exists and every turn finished.
    - ``"incomplete"``: at least one turn was left unfinished.
    - ``"empty"``: no journal (or no records) — nothing to reconcile.
    - ``"error"``: journal exists but is unreadable/corrupt.
    """
    path = cwd / _GODSPEED_DIR_NAME / _TURNS_DIR_NAME / f"{session_id}.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ReconcileResult(last_completed_seq=0, incomplete=[], verdict="empty")
    except OSError:
        return ReconcileResult(last_completed_seq=0, incomplete=[], verdict="error")

    states: dict[int, dict] = {}
    valid_lines = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        valid_lines += 1
        try:
            record = json.loads(line)
        except ValueError:
            continue
        seq = record.get("seq")
        if isinstance(seq, int):
            states[seq] = record

    if not states:
        if valid_lines:
            return ReconcileResult(last_completed_seq=0, incomplete=[], verdict="error")
        return ReconcileResult(last_completed_seq=0, incomplete=[], verdict="empty")

    last_completed = 0
    incomplete: list[int] = []
    for seq, record in sorted(states.items()):
        outcome = record.get("outcome")
        if outcome in _FINISHED_OUTCOMES:
            last_completed = seq
        else:
            incomplete.append(seq)

    verdict = "clean" if not incomplete else "incomplete"
    return ReconcileResult(
        last_completed_seq=last_completed,
        incomplete=incomplete,
        verdict=verdict,
    )
