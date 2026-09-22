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

Only the ``is_destructive`` question is used, not a multi-way risk category.
A hand-labeled accuracy check against this checkpoint (24 shell commands,
covering read_only/low/moderate/destructive) found the base checkpoint's
4-way categorical classification scores ~37.5% — barely above the 25% random
baseline for four classes, and not usable. ``is_destructive`` (a
boolean/probability question) scored 87.5% on the same set and carries real
signal. This matches an independent finding (eesel.ai's review of Laya) that
its headline accuracy figures belong to a checkpoint fine-tuned on that
specific benchmark, not the zero-shot base checkpoint used here.

The same validation run found confidence never approached 0.7 across 40 test
cases (permission + router questions combined) — the highest observed was
0.64. There is no reliable calibration yet for what a "trustworthy"
confidence value looks like for this question, so this module does not
hard-gate on one: it always attaches the annotation when Laya succeeds and
shows the raw probability so a human can judge the signal's strength
themselves, rather than hiding it behind an opaque, unvalidated threshold.

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


_PERMISSION_QUESTIONS: dict[str, dict[str, Any]] = {
    "is_destructive": {
        "type": "noul",
        "instructions": (
            "Would running `command` cause irreversible data loss, system "
            "damage, or destructive side effects?"
        ),
    },
}


@dataclass
class LayaAdvisory:
    """One Laya read on a shell command. Advisory only — see module docstring."""

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

    def ask(
        self, state: Any, questions: dict[str, dict[str, Any]], *, timeout_ms: int = 200
    ) -> dict[str, Any] | None:
        """Ask Laya *questions* about *state*. Returns the raw ``predict()``
        result, or ``None`` on any failure — never raises.

        ``predict()`` is synchronous with no native timeout, so it runs on a
        dedicated single worker thread and is bounded with
        ``.result(timeout=...)``; a hung call just leaks that one thread
        rather than blocking the caller. Shared by every Laya-backed feature
        (permission advisory, task-type routing, ...) so the checkpoint is
        loaded once regardless of how many question sets get asked of it.
        """
        agent = self._agent_or_none()
        if agent is None:
            return None
        try:
            future = self._executor.submit(self._call_predict, agent, state, questions)
            return future.result(timeout=max(timeout_ms, 1) / 1000)
        except FutureTimeoutError:
            logger.warning("Laya ask() timed out after %dms — continuing without it", timeout_ms)
            return None
        except Exception:
            logger.warning("Laya ask() failed — continuing without it", exc_info=True)
            return None

    @staticmethod
    def _call_predict(
        agent: Any, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        result = agent.predict(state, questions)
        return result if isinstance(result, dict) else {}

    def get_advisory(
        self, command: str, dangers: list[str], *, timeout_ms: int = 200
    ) -> LayaAdvisory | None:
        """Score *command* for destructiveness. Returns ``None`` on any failure."""
        state = {"command": command, "regex_flags": dangers}
        result = self.ask(state, _PERMISSION_QUESTIONS, timeout_ms=timeout_ms)
        if result is None:
            return None
        answers = result.get("answers", {}) if isinstance(result, dict) else {}
        destructive_answer = answers.get("is_destructive", {})
        destructive_prob = (
            destructive_answer.get("noul", 0.0) if isinstance(destructive_answer, dict) else 0.0
        )
        return LayaAdvisory(is_destructive=float(destructive_prob), raw=result)


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

    return PermissionDecision(
        decision.action,
        f"{decision.reason} [laya: destructive={advisory.is_destructive:.0%}]",
        metadata={**decision.metadata, "laya": advisory.raw},
    )
