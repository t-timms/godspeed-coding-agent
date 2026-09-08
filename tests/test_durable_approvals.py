"""Tests for crash-durable pending approvals (permission engine).

Covers:
- approval_fingerprint stability and sensitivity
- ASK decisions persist a pending record BEFORE prompting
- record_decision records the user's answer
- replay of a recorded decision on the next session (one-shot)
- no pending_dir => no persistence, plain ASK
- MAX_PENDING cap evicts the oldest records
"""

from __future__ import annotations

import json
from pathlib import Path

from godspeed.security.permissions import (
    ALLOW,
    ASK,
    DENY,
    MAX_PENDING,
    PermissionEngine,
    approval_fingerprint,
)
from godspeed.tools.base import ToolCall


def _engine(tmp_path: Path) -> PermissionEngine:
    return PermissionEngine(pending_dir=tmp_path / ".godspeed" / "pending_approvals")


def _shell_call(command: str = "ls") -> ToolCall:
    return ToolCall(tool_name="shell", arguments={"command": command})


class TestApprovalFingerprint:
    def test_stable_for_same_call(self) -> None:
        fp1 = approval_fingerprint("shell", {"command": "ls"})
        fp2 = approval_fingerprint("shell", {"command": "ls"})
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_differs_for_different_arguments(self) -> None:
        assert approval_fingerprint("shell", {"command": "ls"}) != approval_fingerprint(
            "shell", {"command": "rm -rf /"}
        )

    def test_differs_for_different_tool(self) -> None:
        assert approval_fingerprint("shell", {"command": "ls"}) != approval_fingerprint(
            "read", {"command": "ls"}
        )

    def test_argument_order_does_not_matter(self) -> None:
        fp1 = approval_fingerprint("shell", {"a": 1, "b": 2})
        fp2 = approval_fingerprint("shell", {"b": 2, "a": 1})
        assert fp1 == fp2


class TestPersistPending:
    def test_ask_persists_record_before_prompt(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        decision = engine.evaluate(_shell_call())
        assert decision == ASK
        pending_dir = tmp_path / ".godspeed" / "pending_approvals"
        files = list(pending_dir.glob("*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text(encoding="utf-8"))
        assert record["tool_name"] == "shell"
        assert "command" in record["arguments_json"]
        assert "requested_at" in record
        assert "rule_suggestion" in record
        assert "decision" not in record

    def test_no_pending_dir_means_no_persistence(self, tmp_path: Path) -> None:
        engine = PermissionEngine()
        decision = engine.evaluate(_shell_call())
        assert decision == ASK
        assert not (tmp_path / ".godspeed").exists()

    def test_allow_and_deny_do_not_persist(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.add_rule("shell(ls)", "allow")
        assert engine.evaluate(_shell_call()) == ALLOW
        engine.add_rule("shell(rm *)", "deny")
        assert engine.evaluate(_shell_call("rm -rf /")) == DENY
        pending_dir = tmp_path / ".godspeed" / "pending_approvals"
        assert not pending_dir.exists() or not list(pending_dir.glob("*.json"))


class TestRecordDecision:
    def test_record_decision_updates_record(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.evaluate(_shell_call())
        fingerprint = approval_fingerprint("shell", {"command": "ls"})
        engine.record_decision(fingerprint, True)
        record = json.loads(
            (tmp_path / ".godspeed" / "pending_approvals" / f"{fingerprint}.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["decision"] == "approved"
        assert "decided_at" in record

    def test_record_decision_missing_record_is_noop(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.record_decision("does-not-exist", True)  # must not raise


class TestReplay:
    def test_replay_approved_returns_allow(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.evaluate(_shell_call())
        engine.record_decision(approval_fingerprint("shell", {"command": "ls"}), True)
        decision = engine.evaluate(_shell_call())
        assert decision == ALLOW
        assert "replayed from pending record" in decision.reason

    def test_replay_denied_returns_deny(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.evaluate(_shell_call())
        engine.record_decision(approval_fingerprint("shell", {"command": "ls"}), False)
        decision = engine.evaluate(_shell_call())
        assert decision == DENY
        assert "replayed from pending record" in decision.reason

    def test_replay_is_one_shot(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.evaluate(_shell_call())
        engine.record_decision(approval_fingerprint("shell", {"command": "ls"}), True)
        assert engine.evaluate(_shell_call()) == ALLOW
        # Record consumed — the next identical call asks again.
        assert engine.evaluate(_shell_call()) == ASK

    def test_no_decision_means_reask(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.evaluate(_shell_call())  # persisted, no decision recorded
        decision = engine.evaluate(_shell_call())
        assert decision == ASK

    def test_replay_matches_only_identical_call(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine.evaluate(_shell_call("ls"))
        engine.record_decision(approval_fingerprint("shell", {"command": "ls"}), True)
        # Different arguments -> different fingerprint -> no replay.
        assert engine.evaluate(_shell_call("pwd")) == ASK


class TestMaxPending:
    def test_cap_evicts_oldest(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        for i in range(MAX_PENDING + 10):
            engine.evaluate(_shell_call(f"cmd-{i}"))
        pending_dir = tmp_path / ".godspeed" / "pending_approvals"
        files = list(pending_dir.glob("*.json"))
        assert len(files) <= MAX_PENDING
