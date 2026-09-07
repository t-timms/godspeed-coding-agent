"""Tests for the pre-completion gate (pure logic, no async)."""

from __future__ import annotations

from godspeed.agent.completion_gate import (
    COMPLETION_GATE_MAX_BLOCKS,
    CompletionGateState,
    GateDecision,
    get_checklist_message,
    should_block,
)


class TestShouldBlock:
    """Test the should_block decision logic."""

    def test_clean_no_edits_no_tasks(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=False,
            tasks_open=False,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.PASS

    def test_edits_since_verify_blocks(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=False,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.BLOCK

    def test_open_tasks_block(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=False,
            tasks_open=True,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.BLOCK

    def test_both_edits_and_tasks_block(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=True,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.BLOCK

    def test_second_stop_passes(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=True,
            stop_attempts=COMPLETION_GATE_MAX_BLOCKS + 1,
        )
        assert should_block(state) == GateDecision.PASS

    def test_forced_stop_skips_gate(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=True,
            stop_attempts=1,
            forced_stop=True,
        )
        assert should_block(state) == GateDecision.PASS

    def test_first_stop_attempt_blocks(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=False,
            stop_attempts=1,
        )
        assert should_block(state) == GateDecision.BLOCK

    def test_zero_stop_attempts_blocks_when_pending(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=False,
            stop_attempts=0,
        )
        assert should_block(state) == GateDecision.BLOCK

    def test_no_edits_no_tasks_zero_attempts_passes(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=False,
            tasks_open=False,
            stop_attempts=0,
        )
        assert should_block(state) == GateDecision.PASS

    def test_custom_max_blocks(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=False,
            stop_attempts=3,
        )
        assert should_block(state, max_blocks=5) == GateDecision.BLOCK

    def test_custom_max_blocks_exceeded(self) -> None:
        state = CompletionGateState(
            has_edits_since_verify=True,
            tasks_open=False,
            stop_attempts=6,
        )
        assert should_block(state, max_blocks=5) == GateDecision.PASS


class TestChecklistMessage:
    """Test the checklist message is non-empty and contains key phrases."""

    def test_message_content(self) -> None:
        msg = get_checklist_message()
        assert "verify" in msg
        assert "tests" in msg
        assert "tasks" in msg
        assert "DONE" in msg

    def test_message_not_empty(self) -> None:
        assert len(get_checklist_message()) > 0
