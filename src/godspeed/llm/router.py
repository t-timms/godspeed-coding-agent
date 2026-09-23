"""Task-aware model routing — classify each LLM turn so cheap models
handle simple continuations and a strong model handles fresh planning.

Pairs with:
- ``LLMClient.chat(..., task_type=)`` and ``stream_chat(..., task_type=)``
  in :mod:`godspeed.llm.client` — both apply the routing swap via
  :class:`ModelRouter`.
- ``GodspeedSettings.cheap_model`` / ``strong_model`` / ``architect_model``
  shortcuts in :mod:`godspeed.config` — populate ``routing[<type>]`` so
  users don't have to learn the underlying dict syntax.

The classifier is rule-based (no extra LLM call). It inspects the last
assistant message in the conversation and decides what kind of work the
model is *most likely* about to do next based on what it just did.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from godspeed.config import LayaSettings

logger = logging.getLogger(__name__)

# Canonical task-type strings used as routing keys throughout Godspeed.
# Kept as module constants so callers don't sprinkle string literals.
TASK_PLAN: Final[str] = "plan"
TASK_EDIT: Final[str] = "edit"
TASK_READ: Final[str] = "read"
TASK_SHELL: Final[str] = "shell"
TASK_COMPACTION: Final[str] = "compaction"
TASK_ARCHITECT: Final[str] = "architect"

TASK_TYPES: Final[tuple[str, ...]] = (
    TASK_PLAN,
    TASK_EDIT,
    TASK_READ,
    TASK_SHELL,
    TASK_COMPACTION,
    TASK_ARCHITECT,
)

# Tools that mutate the working tree. A model continuing after one of
# these is likely making more edits or doing a quick verify — both
# cheap-model territory.
_EDIT_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "file_edit",
        "file_write",
        "diff_apply",
        "notebook_edit",
        "generate_tests",
    }
)

# Tools that execute commands or hit external services.
_SHELL_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "shell",
        "git",
        "github",
        "test_runner",
    }
)

# Read-only tools — search, inspection, audit. Continuation after these
# is usually "look at one more thing then decide" — cheap.
_READ_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "file_read",
        "pdf_read",
        "image_read",
        "glob_search",
        "grep_search",
        "code_search",
        "repo_map",
        "web_search",
        "web_fetch",
        "dep_audit",
        "security_scan",
        "complexity",
        "coverage",
        "verify",
        "system_optimizer",
        "background_check",
        "tasks",
    }
)


def _extract_tool_names(message: dict[str, Any]) -> list[str]:
    """Pull tool names out of a single assistant message.

    LiteLLM/OpenAI shape: ``message["tool_calls"]`` is a list of
    ``{"id": ..., "function": {"name": ..., "arguments": ...}, ...}``.
    Returns ``[]`` for non-assistant messages or messages with no
    tool calls.
    """
    if message.get("role") != "assistant":
        return []
    raw_calls = message.get("tool_calls") or []
    names: list[str] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _last_assistant_tools(messages: Sequence[dict[str, Any]]) -> list[str]:
    """Tool names from the most recent assistant turn, or ``[]``.

    Walks the conversation backwards. Returns ``[]`` when the most
    recent assistant turn had no tools, or when no assistant turn
    exists yet (fresh conversation).
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return _extract_tool_names(msg)
    return []


def classify_task_type(messages: Sequence[dict[str, Any]]) -> str:
    """Classify the upcoming LLM call from current conversation state.

    Heuristic (cheap, deterministic, no LLM call):

    - **plan**: no prior assistant turn, OR the most recent assistant
      turn made no tool calls (model previously stopped — user is
      kicking off something new). Routes to the strongest available
      model since this is where reasoning matters most.
    - **edit**: most recent assistant turn called any write tool.
      Continuation is likely follow-on edits or a quick verify.
    - **shell**: most recent assistant turn called any execute tool
      (shell/git/github/test_runner). Continuation is interpreting
      output.
    - **read**: most recent assistant turn called *only* read-only tools.
      Continuation is "consider one more thing then decide".
    - Falls back to **plan** when the prior turn called tools we
      don't classify (newly added, or external/MCP) — safer to keep
      the strong model than to silently downgrade reasoning.
    """
    tools = _last_assistant_tools(messages)
    if not tools:
        return TASK_PLAN
    # Highest-stakes wins: an edit-then-search batch is still "edit".
    if any(t in _EDIT_TOOLS for t in tools):
        return TASK_EDIT
    if any(t in _SHELL_TOOLS for t in tools):
        return TASK_SHELL
    if all(t in _READ_TOOLS for t in tools):
        return TASK_READ
    return TASK_PLAN


# task_types classify_task_type can hand back that Laya is allowed to
# escalate away from. plan/architect/compaction are already the top of the
# ladder — nothing to escalate to.
_ESCALATABLE_TASK_TYPES: Final[frozenset[str]] = frozenset({TASK_EDIT, TASK_READ, TASK_SHELL})


def get_laya_difficulty_score(
    request_text: str, laya_settings: LayaSettings | None
) -> float | None:
    """Ask Laya how hard *request_text* is (``router_questions()``'s
    ``difficulty`` score, continuous 0-3). Returns ``None`` on a disabled/
    unset ``laya_settings``, empty *request_text*, or any Laya error/
    timeout/absence — never raises.

    ``request_text`` should be the actual user request driving this turn —
    e.g. ``agent_loop``'s own ``user_input`` parameter — **not** derived
    from scanning conversation history for the latest ``role="user"``
    message. ``agent_loop`` injects synthetic ``role="user"`` messages
    mid-turn (continuation nudges, acceptance summaries; see
    ``Conversation.add_user_message`` — it doesn't distinguish human from
    synthetic), so a later "most recent user message" would score
    administrative boilerplate instead of the real request on iteration 2+
    of the same turn.

    Blocking (bounded by ``laya_settings.timeout_ms``, same as
    ``LayaAdvisor.ask()``) — callers on an event loop should wrap this in
    ``asyncio.to_thread``, the way ``agent_loop`` does, calling it once per
    turn rather than once per iteration since the request's difficulty
    doesn't change as the agent works through it.
    """
    if laya_settings is None or not laya_settings.enabled or not request_text:
        return None

    from godspeed.security.laya_advisor import LayaAdvisor, _is_laya_available, extract_answer

    if not _is_laya_available():
        return None

    try:
        import laya

        router_questions = laya.router_questions()
    except Exception:
        logger.warning("Laya router_questions() unavailable — routing unchanged", exc_info=True)
        return None

    result = LayaAdvisor.get().ask(
        {"request": request_text},
        router_questions,
        timeout_ms=laya_settings.timeout_ms,
    )
    difficulty_answer = extract_answer(result, "difficulty")
    score = difficulty_answer.get("score") if difficulty_answer else None
    return score if isinstance(score, int | float) else None


def maybe_escalate_task_type(
    task_type: str, difficulty_score: float | None, laya_settings: LayaSettings | None
) -> str:
    """Escalate *task_type* toward ``"plan"`` if *difficulty_score* (from
    ``get_laya_difficulty_score``) meets
    ``laya_settings.difficulty_escalate_threshold``.

    ``classify_task_type`` looks *backward* — what tools did the last
    assistant turn call — which says nothing about how hard the user's
    actual request is. Laya's ``router_questions()`` preset reads the
    request itself, prospectively, so the two signals are complementary
    rather than redundant.

    Escalate-only, mirroring ``security/laya_advisor.py``'s "advisory adds
    caution, never removes it" principle: this can only move *task_type*
    toward the strong-model tier, never away from it. Fails neutral — a
    ``None`` *difficulty_score* (disabled, unset, or any Laya failure — see
    ``get_laya_difficulty_score``) or an already-strong *task_type* both
    return *task_type* unchanged.

    Pure and cheap by design (no I/O): meant to be called every loop
    iteration against a *difficulty_score* computed once per turn, unlike
    ``get_laya_difficulty_score`` itself.
    """
    if difficulty_score is None or laya_settings is None:
        return task_type
    if task_type not in _ESCALATABLE_TASK_TYPES:
        return task_type
    if difficulty_score >= laya_settings.difficulty_escalate_threshold:
        logger.info("Laya escalated task_type %s -> plan (score=%.2f)", task_type, difficulty_score)
        return TASK_PLAN
    return task_type
