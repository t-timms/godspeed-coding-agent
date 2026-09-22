"""Laya fast permission pre-classifier — advisory only, never authoritative.

Laya (https://huggingface.co/convaiinnovations/laya) is a ~421M-param,
non-autoregressive "System 1" decision model: given a state and a set of
typed questions, it returns typed answers with calibrated confidence in a
single ~33ms forward pass.

This module wires it in strictly *after* ``PermissionEngine.evaluate()``
(``security/permissions.py``) has already decided ``ASK`` for a bash/shell
command — as a read-only annotation on the decision's ``reason``/``metadata``,
never as an input to the decision itself. ``annotate_ask_decision`` cannot
change a ``PermissionDecision.action``; that is enforced by construction, not
convention (see its body). Laya failing, timing out, or being uninstalled
always degrades to "no annotation" — it must never fail closed, since an
outage here must not affect the actual permission decision.

Scope: interactive (TUI) only. Headless already hard-denies HIGH/DESTRUCTIVE
risk with no ASK moment to annotate, so there is nothing for Laya to attach to
there.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from godspeed.security.dangerous import detect_dangerous_command
from godspeed.security.permissions import ASK, PermissionDecision

if TYPE_CHECKING:
    from godspeed.config import LayaSettings
    from godspeed.tools.base import ToolCall

logger = logging.getLogger(__name__)

_laya_available: bool | None = None
_laya_available_lock = threading.Lock()


def _is_laya_available() -> bool:
    """Check whether the ``laya`` package is installed. Cached after first call."""
    global _laya_available
    if _laya_available is None:
        with _laya_available_lock:
            if _laya_available is None:
                try:
                    import laya  # noqa: F401

                    _laya_available = True
                except ImportError:
                    _laya_available = False
    return _laya_available


@dataclass
class LayaAdvisory:
    """One Laya read on a shell command. Advisory only — see module docstring."""

    risk_category: str  # "read_only" | "low" | "moderate" | "destructive"
    risk_category_confidence: float | None
    is_destructive: float  # calibrated probability, 0.0-1.0
    raw: dict[str, Any] = field(repr=False)


class LayaAdvisor:
    """Lazily loads and caches a single Laya agent — pays model-load cost once."""

    _instance: LayaAdvisor | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._agent: Any | None = None
        self._agent_load_failed = False
        self._agent_lock = threading.Lock()
        # Single worker: Laya's own thread-safety under concurrent predict()
        # calls on one loaded model isn't confirmed, so calls are serialized
        # rather than risking a race inside the model. Known v1 trade-off:
        # if a turn issues multiple ASK-tier shell calls concurrently
        # (agent_loop's asyncio.gather), later ones can spuriously hit their
        # timeout while queued behind an earlier call, since
        # future.result(timeout=...) clocks from submission, not from when
        # execution actually starts. Fails neutral either way (annotation is
        # just dropped), so this is a missed-advisory cost, not a
        # correctness or security one.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-advisor")

    @classmethod
    def get(cls) -> LayaAdvisor:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _agent_or_none(self) -> Any | None:
        if self._agent is not None:
            return self._agent
        if self._agent_load_failed:
            return None
        with self._agent_lock:
            if self._agent is None and not self._agent_load_failed:
                try:
                    import laya

                    self._agent = laya.load("convaiinnovations/laya")
                except Exception:
                    logger.warning("Laya failed to load — advisory disabled", exc_info=True)
                    self._agent_load_failed = True
                    return None
            return self._agent

    def get_advisory(
        self, command: str, dangers: list[str], *, timeout_ms: int = 200
    ) -> LayaAdvisory | None:
        """Score *command*. Returns ``None`` on any failure — never raises.

        ``predict()`` is synchronous with no native timeout, so it runs on a
        dedicated single worker thread and is bounded with ``.result(timeout=...)``;
        a hung call just leaks that one thread rather than blocking the caller.
        """
        agent = self._agent_or_none()
        if agent is None:
            return None
        try:
            future = self._executor.submit(self._predict, agent, command, dangers)
            result = future.result(timeout=max(timeout_ms, 1) / 1000)
        except FutureTimeoutError:
            logger.warning("Laya advisory timed out after %dms — continuing without it", timeout_ms)
            return None
        except Exception:
            logger.warning("Laya advisory failed — continuing without it", exc_info=True)
            return None
        return result

    @staticmethod
    def _predict(agent: Any, command: str, dangers: list[str]) -> LayaAdvisory:
        state = {"command": command, "regex_flags": dangers}
        questions = {
            "risk_category": {
                "type": "choice",
                "instructions": (
                    "Classify the actual risk of running `command` in a coding agent's shell tool."
                ),
                "criteria": {
                    "read_only": "no side effects: listing, viewing, searching, status checks",
                    "low": "reversible local changes: editing a tracked file, installing a package",
                    "moderate": (
                        "harder-to-reverse changes scoped to this repo/session: "
                        "force-push, deleting untracked files"
                    ),
                    "destructive": (
                        "irreversible data loss or damage beyond this repo/session: "
                        "recursive deletes of system paths, disk operations, "
                        "credential exfiltration"
                    ),
                },
            },
            "is_destructive": {
                "type": "noul",
                "instructions": (
                    "Would running `command` cause irreversible data loss, system "
                    "damage, or destructive side effects?"
                ),
            },
        }
        result = agent.predict(state, questions)
        answers = result.get("answers", {}) if isinstance(result, dict) else {}

        category_answer = answers.get("risk_category", {})
        category = category_answer.get("choice", "") if isinstance(category_answer, dict) else ""
        category_confidence = (
            category_answer.get("confidence") if isinstance(category_answer, dict) else None
        )

        destructive_answer = answers.get("is_destructive", {})
        destructive_prob = (
            destructive_answer.get("noul", 0.0) if isinstance(destructive_answer, dict) else 0.0
        )

        return LayaAdvisory(
            risk_category=str(category),
            risk_category_confidence=(
                float(category_confidence) if category_confidence is not None else None
            ),
            is_destructive=float(destructive_prob),
            raw=result if isinstance(result, dict) else {},
        )


def annotate_ask_decision(
    decision: PermissionDecision, tool_call: ToolCall, laya_settings: LayaSettings
) -> PermissionDecision:
    """Attach a Laya advisory to an ``ASK``-tier shell/bash decision.

    Returns *decision* unchanged (same object) in every case where Laya
    doesn't apply or fails — this function can only ever return a decision
    with the *same* ``.action`` it was given; it has no code path that sets
    ``.action`` to anything else. Called only from the interactive TUI
    permission proxies.
    """
    if not laya_settings.enabled or decision.action != ASK:
        return decision
    if tool_call.tool_name.lower() not in ("bash", "shell"):
        return decision
    if not _is_laya_available():
        return decision

    command = ""
    if isinstance(tool_call.arguments, dict):
        raw_command = tool_call.arguments.get("command", "")
        if isinstance(raw_command, str):
            command = raw_command
    if not command:
        return decision

    try:
        dangers = detect_dangerous_command(command)
    except Exception:
        dangers = []

    advisory = LayaAdvisor.get().get_advisory(command, dangers, timeout_ms=laya_settings.timeout_ms)
    if advisory is None:
        return decision
    if (
        advisory.risk_category_confidence is not None
        and advisory.risk_category_confidence < laya_settings.confidence_threshold
    ):
        return decision

    return PermissionDecision(
        decision.action,
        f"{decision.reason} [laya: {advisory.risk_category}, "
        f"destructive={advisory.is_destructive:.0%}]",
        metadata={**decision.metadata, "laya": advisory.raw},
    )
