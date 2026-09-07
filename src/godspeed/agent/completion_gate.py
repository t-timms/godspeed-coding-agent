"""Pre-completion gate — structural guard against premature stop.

When the model signals it wants to STOP and (a) file edits/writes happened
since the last successful verification, or (b) the TaskStore has open tasks
that were touched this session, the loop blocks the stop *once* and injects
a structured completion-checklist message. On the second stop attempt, stop
proceeds normally.

Skipped entirely when:
- No edits since verify AND no open tasks.
- Headless budget/timeout forced stop.
- User interrupted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

COMPLETION_GATE_MAX_BLOCKS: int = 1

_COMPLETION_CHECKLIST: str = (
    "Before stopping: run verify on changed files; run tests if any test file "
    "was touched; confirm all tasks are completed or explicitly parked. If "
    "verification is already green, restate DONE."
)


class GateDecision(StrEnum):
    """Outcome of the pre-completion gate check."""

    PASS = "pass"  # noqa: S105
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class CompletionGateState:
    """Pure snapshot of gate-relevant state for the ``should_block`` decision.

    Kept frozen so the helper is side-effect-free and trivially testable.
    Loop integration constructs this from mutable ``_LoopState`` fields each
    time the model signals a stop.
    """

    has_edits_since_verify: bool
    tasks_open: bool
    stop_attempts: int
    forced_stop: bool = False


def should_block(
    state: CompletionGateState,
    max_blocks: int = COMPLETION_GATE_MAX_BLOCKS,
) -> GateDecision:
    """Determine whether the pre-completion gate should block the stop.

    Args:
        state: Immutable snapshot of loop-relevant state.
        max_blocks: Maximum number of times the gate may block before
            yielding (``COMPLETION_GATE_MAX_BLOCKS``).

    Returns:
        ``GateDecision.BLOCK`` when the gate should inject the checklist
        message and let the loop continue; ``GateDecision.PASS`` otherwise.
    """
    # Skip on forced stop (budget/timeout) or user interrupt
    if state.forced_stop:
        logger.debug("Completion gate: skip (forced stop)")
        return GateDecision.PASS

    # Nothing to check — clean slate
    if not state.has_edits_since_verify and not state.tasks_open:
        logger.debug("Completion gate: skip (clean — no edits, no open tasks)")
        return GateDecision.PASS

    # Already exhausted block budget — fail open
    if state.stop_attempts > max_blocks:
        logger.debug(
            "Completion gate: pass (stop_attempts=%d > max_blocks=%d)",
            state.stop_attempts,
            max_blocks,
        )
        return GateDecision.PASS

    # First attempt with pending work — block once
    logger.info(
        "Completion gate: BLOCK (edits=%s, tasks_open=%s, stop_attempts=%d)",
        state.has_edits_since_verify,
        state.tasks_open,
        state.stop_attempts,
    )
    return GateDecision.BLOCK


def get_checklist_message() -> str:
    """Return the structured completion-checklist message for injection."""
    return _COMPLETION_CHECKLIST
