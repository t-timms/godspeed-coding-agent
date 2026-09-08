"""Tests for the append-only turn journal (turn-level durability).

Covers:
- start_turn appends a ``running`` record with seq = last + 1
- complete_turn appends the completion record
- seq increments across turns
- reconcile verdicts: clean / incomplete / empty / error
- request_fingerprint stability and sensitivity
- fail-safe writes (never raise, even when the journal is unwritable)
"""

from __future__ import annotations

import json
from pathlib import Path

from godspeed.agent.turn_journal import (
    TurnJournal,
    reconcile,
    request_fingerprint,
)


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class TestStartTurn:
    def test_start_turn_appends_running_record(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        assert seq == 1
        records = _records(journal.path)
        assert len(records) == 1
        assert records[0]["seq"] == 1
        assert records[0]["outcome"] == "running"
        assert records[0]["request_fingerprint"] == "fp-1"
        assert records[0]["finished_at"] is None

    def test_seq_increments_across_turns(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        assert journal.start_turn("fp-1") == 1
        assert journal.start_turn("fp-2") == 2
        assert journal.start_turn("fp-3") == 3

    def test_seq_continues_after_restart(self, tmp_path: Path) -> None:
        first = TurnJournal("sess-1", tmp_path)
        first.start_turn("fp-1")
        first.complete_turn(1)
        second = TurnJournal("sess-1", tmp_path)
        assert second.start_turn("fp-2") == 2


class TestCompleteTurn:
    def test_complete_turn_appends_completion(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        journal.complete_turn(seq)
        records = _records(journal.path)
        assert len(records) == 2
        assert records[1]["seq"] == 1
        assert records[1]["outcome"] == "completed"
        assert records[1]["finished_at"] is not None

    def test_complete_turn_with_error(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        journal.complete_turn(seq, outcome="error", error="boom")
        records = _records(journal.path)
        assert records[1]["outcome"] == "error"
        assert records[1]["error"] == "boom"

    def test_complete_turn_interrupted(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        journal.complete_turn(seq, outcome="interrupted")
        records = _records(journal.path)
        assert records[1]["outcome"] == "interrupted"


class TestReconcile:
    def test_clean_when_all_completed(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        for i in range(3):
            seq = journal.start_turn(f"fp-{i}")
            journal.complete_turn(seq)
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "clean"
        assert result.last_completed_seq == 3
        assert result.incomplete == []

    def test_incomplete_when_running_left(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        journal.start_turn("fp-1")
        journal.complete_turn(1)
        journal.start_turn("fp-2")  # never completed — crash
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "incomplete"
        assert result.last_completed_seq == 1
        assert result.incomplete == [2]

    def test_incomplete_when_interrupted(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        journal.complete_turn(seq, outcome="interrupted")
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "incomplete"
        assert result.incomplete == [1]

    def test_error_outcome_counts_as_completed(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        journal.complete_turn(seq, outcome="error", error="boom")
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "clean"
        assert result.last_completed_seq == 1

    def test_empty_when_no_journal(self, tmp_path: Path) -> None:
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "empty"
        assert result.last_completed_seq == 0
        assert result.incomplete == []

    def test_error_when_journal_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / ".godspeed" / "turns" / "sess-1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json\n", encoding="utf-8")
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "error"

    def test_last_record_per_seq_wins(self, tmp_path: Path) -> None:
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")
        journal.complete_turn(seq)
        # Simulate a crash that left a stale "running" record after the
        # completion (should not happen with correct usage, but reconcile
        # must be robust): append a duplicate seq with running outcome.
        with journal.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"seq": 1, "outcome": "running"}) + "\n")
        result = reconcile("sess-1", tmp_path)
        assert result.verdict == "incomplete"
        assert result.incomplete == [1]


class TestRequestFingerprint:
    def test_stable_for_same_input(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"name": "shell"}]
        fp1 = request_fingerprint("model-a", messages, tools)
        fp2 = request_fingerprint("model-a", messages, tools)
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_differs_for_different_model(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"name": "shell"}]
        assert request_fingerprint("model-a", messages, tools) != request_fingerprint(
            "model-b", messages, tools
        )

    def test_differs_for_different_messages(self) -> None:
        tools = [{"name": "shell"}]
        assert request_fingerprint(
            "m", [{"role": "user", "content": "a"}], tools
        ) != request_fingerprint("m", [{"role": "user", "content": "b"}], tools)

    def test_differs_for_different_tools(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        assert request_fingerprint("m", messages, [{"name": "shell"}]) != request_fingerprint(
            "m", messages, [{"name": "read"}]
        )


class TestFailSafe:
    def test_write_failure_does_not_raise(self, tmp_path: Path) -> None:
        # Make the journal path unwritable by placing a FILE where the
        # .godspeed directory should be — mkdir raises FileExistsError.
        blocker = tmp_path / ".godspeed"
        blocker.write_text("i am a file", encoding="utf-8")
        journal = TurnJournal("sess-1", tmp_path)
        seq = journal.start_turn("fp-1")  # must not raise
        journal.complete_turn(seq)  # must not raise

    def test_scan_last_seq_ignores_corrupt_lines(self, tmp_path: Path) -> None:
        path = tmp_path / ".godspeed" / "turns" / "sess-1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "{bad\n" + json.dumps({"seq": 7, "outcome": "completed"}) + "\n", encoding="utf-8"
        )
        journal = TurnJournal("sess-1", tmp_path)
        assert journal.start_turn("fp-1") == 8
