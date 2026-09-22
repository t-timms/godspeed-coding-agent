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
# ladder — nothing to escalate to, and skipping them avoids paying for a
# Laya call on requests that don't need one.
_ESCALATABLE_TASK_TYPES: Final[frozenset[str]] = frozenset({TASK_EDIT, TASK_READ, TASK_SHELL})


def _last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    """Plain text of the most recent user-role message, or ``""``.

    Mirrors the block-extraction pattern in ``llm/client.py``'s response
    parsing: ``content`` is either a plain string or a list of content
    blocks (e.g. text + image, OpenAI multimodal format) — only the text
    blocks matter for a difficulty read.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
            return text
        return ""
    return ""


def maybe_escalate_task_type(
    task_type: str, messages: Sequence[dict[str, Any]], laya_settings: LayaSettings | None
) -> str:
    """Escalate *task_type* toward ``"plan"`` if Laya judges the request
    harder than ``classify_task_type``'s tool-based bucketing assumed.

    ``classify_task_type`` looks *backward* — what tools did the last
    assistant turn call — which says nothing about how hard the user's
    actual request is. Laya's ``router_questions()`` preset reads the
    request itself, prospectively, so the two signals are complementary
    rather than redundant.

    Escalate-only, mirroring ``security/laya_advisor.py``'s "advisory adds
    caution, never removes it" principle: this can only move *task_type*
    toward the strong-model tier, never away from it. Fails neutral — a
    disabled/unset ``laya_settings``, an already-strong task_type, or any
    Laya error/timeout/absence all return *task_type* unchanged, and in the
    already-strong case Laya isn't even called.
    """
    if laya_settings is None or not laya_settings.enabled:
        return task_type
    if task_type not in _ESCALATABLE_TASK_TYPES:
        return task_type

    request_text = _last_user_text(messages)
    if not request_text:
        return task_type

    from godspeed.security.laya_advisor import LayaAdvisor, _is_laya_available

    if not _is_laya_available():
        return task_type

    try:
        import laya

        router_questions = laya.router_questions()
    except Exception:
        logger.warning("Laya router_questions() unavailable — routing unchanged", exc_info=True)
        return task_type

    result = LayaAdvisor.get().ask(
        {"request": request_text},
        router_questions,
        timeout_ms=laya_settings.timeout_ms,
    )
    if result is None:
        return task_type

    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    difficulty_answer = answers.get("difficulty", {})
    score = difficulty_answer.get("score") if isinstance(difficulty_answer, dict) else None
    if not isinstance(score, int | float):
        return task_type

    if score >= laya_settings.difficulty_escalate_threshold:
        logger.info("Laya escalated task_type %s -> plan (score=%.2f)", task_type, score)
        return TASK_PLAN
    return task_type
