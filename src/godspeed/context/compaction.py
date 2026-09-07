"""Conversation compaction — graduated compaction ladder.

Deterministic stages (applied by ``GraduatedCompactor.apply_stages``):
  Stage 1 (75%): budget_reduction — drop verbose tool outputs, keep structure
  Stage 2 (60%): snip — remove low-signal turns, keep decisions + tool calls
  Stage 3 (45%): microcompact — collapse tool call runs to GCG summaries
  Stage 4 (30%): context_collapse — keep only tool call metadata + GCG refs

Async LLM stage (invoked separately via ``emergency_compact``):
  Stage 5 (15%): auto_compact — emergency LLM summarization

GCG node IDs survive ALL stages. Content does not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.config import get_model_context_window
from godspeed.llm.client import LLMClient

logger = logging.getLogger(__name__)

SMALL_CONTEXT_THRESHOLD = 32_768
LARGE_CONTEXT_THRESHOLD = 100_000

# Stage 2 signal-score threshold: messages scoring >= this are preserved.
# Tuned so short acknowledgements ("ok", "done") are dropped but
# planning discussions, questions, and code blocks survive.
SIGNAL_SCORE_THRESHOLD: float = 1.5

_DECISION_LEXICON: frozenset[str] = frozenset(
    {
        "decided",
        "decision",
        "approach",
        "proposal",
        "plan",
        "strategy",
        "design",
        "architecture",
        "pattern",
        "factory",
        "singleton",
        "adapter",
        "refactor",
        "instead",
        "trade-off",
        "tradeoff",
        "consider",
        "propose",
        "alternative",
        "suggest",
        "recommend",
        "rationale",
        "because",
        "option",
        "consensus",
        "agree",
        "disagree",
        "concern",
    }
)


def _signal_score(content: str) -> float:
    """Multi-signal information-density score for Stage 2 compaction decisions."""
    if not content:
        return 0.0
    score = 0.0
    score += min(len(content) / 1_000, 2.0)
    score += content.count("```") * 1.5
    lower = content.lower()
    score += sum(1.0 for word in _DECISION_LEXICON if word in lower)
    score += content.count("?") * 0.5
    return score


COMPACTION_PROMPT_SMALL = """\
Aggressively summarize this coding agent conversation.

Keep ONLY:
- The current task and its status
- File paths modified (just the paths, not contents)
- The last error encountered (if any)
- Key user instructions and corrections
- Active tool schemas the user was referencing

Discard everything else. Be extremely brief — under 500 words.
The summary replaces the full history, so omit anything non-essential.
"""

COMPACTION_PROMPT_MEDIUM = """\
Summarize the following conversation between a user and a coding agent.

You MUST preserve:
- Architectural decisions made
- File paths that were modified or created
- Unresolved issues or errors
- The current task state and what was accomplished
- User preferences, instructions, and corrections
- Active tool schemas or conventions being used

You MUST discard:
- Redundant tool outputs (e.g., full file contents already edited)
- Repeated attempts that were superseded
- Verbose error tracebacks (keep the error message, not the full trace)

Be concise but complete. The summary will replace the conversation history,
so anything not in the summary is lost forever.
"""

COMPACTION_PROMPT_LARGE = """\
Summarize this coding agent conversation, preserving maximum context.

You MUST preserve:
- Architectural decisions and rationale
- All file paths modified or created, with a brief note on each change
- Unresolved issues, errors, and their context
- Current task state — what was accomplished, what remains
- User preferences, instructions, and corrections given
- Active tool schemas or conventions being used
- Last 3 tool results (summarized, not verbatim)
- Key code patterns or conventions discovered

You may discard:
- Verbatim file contents that were read but not modified
- Redundant intermediate tool outputs
- Full stack traces (keep the error message and location)

Be thorough. With a large context window, it is better to preserve
too much than too little.
"""

COMPACTION_SYSTEM_PROMPT = COMPACTION_PROMPT_MEDIUM


# ── 5-stage graduated compaction ladder ────────────────────────────────────


@dataclass
class CompactionStage:
    """A single compaction stage in the graduated ladder."""

    name: str
    threshold_pct: float
    preserves: list[str]
    strategy: Callable[[CompactionContext], list[dict[str, Any]]]


@dataclass
class CompactionContext:
    """Context passed to compaction stage strategies."""

    messages: list[dict[str, Any]]
    token_count: int
    max_tokens: int
    gcg: Any | None = None
    gcg_symbol_ids: list[str] = field(default_factory=list)


@dataclass
class CompactionResult:
    """Result of running a compaction stage."""

    stage_name: str = ""
    messages_before: int = 0
    messages_after: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    applied: bool = False


# ── Stage strategies ─────────────────────────────────────────────────────


def _drop_verbose_tool_outputs(ctx: CompactionContext) -> list[dict[str, Any]]:
    """Stage 1: Drop verbose tool outputs, keep structure."""
    result: list[dict[str, Any]] = []
    for msg in ctx.messages:
        role = msg.get("role", "")
        if role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 3000:
                truncated = content[:1000] + (f"\n... [truncated {len(content) - 1000} chars]")
                result.append({**msg, "content": truncated})
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


def _remove_low_signal_turns(ctx: CompactionContext) -> list[dict[str, Any]]:
    """Stage 2: Remove low-signal turns, keep decisions + tool calls."""
    result: list[dict[str, Any]] = []
    for msg in ctx.messages:
        role = msg.get("role", "")
        if role == "tool":
            result.append(msg)
        elif role == "assistant":
            if msg.get("tool_calls"):
                result.append(msg)
            else:
                content = msg.get("content", "")
                if content and _signal_score(content) >= SIGNAL_SCORE_THRESHOLD:
                    result.append(msg)
                else:
                    continue
        else:
            result.append(msg)
    return result


def _collapse_tool_runs_to_gcg_summaries(ctx: CompactionContext) -> list[dict[str, Any]]:
    """Stage 3: Collapse tool call runs to GCG summaries."""
    result: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    for msg in ctx.messages:
        role = msg.get("role", "")
        if role == "tool":
            pending_tool_results.append(msg)
        else:
            if pending_tool_results and len(pending_tool_results) > 3:
                tool_names = sorted(
                    {t.get("tool_call_id", "?").split("-")[0] for t in pending_tool_results}
                )
                gcg_refs = _build_gcg_summary(ctx, pending_tool_results)
                summary = (
                    f"[compacted: {len(pending_tool_results)} tool results] "
                    f"Tools: {', '.join(tool_names)}\n{gcg_refs}"
                )
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": "compaction",
                        "content": summary,
                    }
                )
            else:
                result.extend(pending_tool_results)
            pending_tool_results = []
            result.append(msg)

    if pending_tool_results:
        result.extend(pending_tool_results)

    return result


def _keep_metadata_only(ctx: CompactionContext) -> list[dict[str, Any]]:
    """Stage 4: Keep only tool call metadata + GCG refs."""
    result: list[dict[str, Any]] = []
    for msg in ctx.messages:
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            result.append({**msg, "content": content})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                brief = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                result.append(
                    {
                        "role": "assistant",
                        "content": f"[tool calls: {brief}]",
                    }
                )
            else:
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    content = content[:200] + "..."
                result.append({**msg, "content": content})
        elif role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            result.append({**msg, "content": content})

    if ctx.gcg_symbol_ids:
        result.append(
            {
                "role": "user",
                "content": f"[GCG references preserved: {len(ctx.gcg_symbol_ids)} symbols]",
            }
        )

    return result


async def _llm_emergency_summarize(
    ctx: CompactionContext,
    llm_client: LLMClient,
    model: str,
) -> list[dict[str, Any]]:
    """Stage 5: Emergency LLM summarization."""
    prompt = get_compaction_prompt(model)
    text = _messages_to_text(ctx.messages)
    response = await llm_client.chat(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
    )
    return [
        {
            "role": "user",
            "content": (
                "[Conversation compacted. Summary of previous work:]\n\n"
                f"{response.content}\n\n"
                "[Continue from where we left off.]"
            ),
        }
    ]


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Convert message list to plain text for compaction."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            parts.append(f"[{role}]: {content}")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(f"[tool_call]: {fn.get('name', '?')}({fn.get('arguments', '')})")
    return "\n".join(parts)


def _build_gcg_summary(
    ctx: CompactionContext,
    results: list[dict[str, Any]],
) -> str:
    """Build GCG reference summary from tool results."""
    if not ctx.gcg:
        return ""
    paths: set[str] = set()
    for r in results:
        content = r.get("content", "")
        if not isinstance(content, str):
            continue
        # Extract file paths from tool output (heuristic)
        for line in content.split("\n"):
            if line.strip().startswith("file:") or line.strip().startswith("/"):
                paths.add(line.strip())
    if not paths:
        return ""
    return "GCG refs: " + ", ".join(sorted(paths)[:10])


# ── Stage ladder ───────────────────────────────────────────────────────────

STAGE_BUDGET_REDUCTION: int = 0
STAGE_SNIP: int = 1
STAGE_MICROCOMPACT: int = 2
STAGE_CONTEXT_COLLAPSE: int = 3
EMERGENCY_STAGE_IDX: int = 4

COMPACTION_STAGES: list[CompactionStage] = [
    CompactionStage(
        name="budget_reduction",
        threshold_pct=0.75,
        preserves=["tool_calls", "file_paths", "decisions", "guidance", "gcg_refs"],
        strategy=_drop_verbose_tool_outputs,
    ),
    CompactionStage(
        name="snip",
        threshold_pct=0.60,
        preserves=["tool_calls", "file_paths", "decisions", "gcg_refs"],
        strategy=_remove_low_signal_turns,
    ),
    CompactionStage(
        name="microcompact",
        threshold_pct=0.45,
        preserves=["tool_calls", "gcg_refs"],
        strategy=_collapse_tool_runs_to_gcg_summaries,
    ),
    CompactionStage(
        name="context_collapse",
        threshold_pct=0.30,
        preserves=["gcg_refs"],
        strategy=_keep_metadata_only,
    ),
]


class GraduatedCompactor:
    """Orchestrates the graduated compaction ladder.

    Deterministic stages 0-3 are applied via ``apply_stages``.
    Stage 4 (auto_compact) is async and handled by ``emergency_compact``.

    Args:
        stages: List of compaction stages. Defaults to COMPACTION_STAGES.
        gcg: Optional GCG instance for GCG-aware compaction.
    """

    def __init__(
        self,
        stages: list[CompactionStage] | None = None,
        gcg: Any | None = None,
    ) -> None:
        self._stages = stages or COMPACTION_STAGES
        self.gcg = gcg
        self._last_stage_idx: int = -1
        self._failed_stages: set[int] = set()
        self._context_pct: float = 0.0

    @property
    def context_pct(self) -> float:
        """Current context usage as a fraction (0.0-1.0)."""
        return self._context_pct

    def reset(self) -> None:
        """Reset compaction state for a new session."""
        self._last_stage_idx = -1
        self._failed_stages: set[int] = set()
        self._context_pct = 0.0

    def get_stage_for_context(self, token_count: int, max_tokens: int) -> int:
        """Get the stage index for current context usage.

        Returns the index of the highest-priority stage whose threshold
        is at or below the current usage percentage. -1 if no stage applies.
        """
        self._context_pct = token_count / max_tokens if max_tokens > 0 else 0.0

        # Find the highest threshold that is <= current usage
        best_idx = -1
        best_threshold = -1.0
        for i, stage in enumerate(self._stages):
            if stage.threshold_pct <= self._context_pct and stage.threshold_pct > best_threshold:
                best_threshold = stage.threshold_pct
                best_idx = i

        return best_idx

    def apply_stages(
        self,
        conversation: Conversation,
        token_count: int,
        max_tokens: int,
    ) -> list[CompactionResult]:
        """Apply all needed compaction stages up to current usage.

        Returns list of CompactionResult for each stage applied.
        Stages are additive: if at 30%, applies stages 1-4.
        """
        target_idx = self.get_stage_for_context(token_count, max_tokens)
        if target_idx <= self._last_stage_idx:
            return []

        results: list[CompactionResult] = []
        ctx = CompactionContext(
            messages=conversation._messages,
            token_count=token_count,
            max_tokens=max_tokens,
            gcg=self.gcg,
        )

        for i in range(self._last_stage_idx + 1, target_idx + 1):
            if i in self._failed_stages:
                continue
            stage = self._stages[i]
            try:
                new_messages = stage.strategy(ctx)
                before_count = len(ctx.messages)
                after_count = len(new_messages)

                conversation.replace_messages(new_messages)

                ctx.messages = new_messages
                self._last_stage_idx = max(self._last_stage_idx, i)

                results.append(
                    CompactionResult(
                        stage_name=stage.name,
                        messages_before=before_count,
                        messages_after=after_count,
                        tokens_before=token_count,
                        tokens_after=conversation.token_count,
                        applied=True,
                    )
                )
                logger.info(
                    "Compaction stage=%s messages=%d→%d pct=%.0f%%",
                    stage.name,
                    before_count,
                    after_count,
                    self._context_pct * 100,
                )
            except Exception as exc:
                logger.warning("Compaction stage %s failed: %s", stage.name, exc)
                self._failed_stages.add(i)

        return results

    async def emergency_compact(
        self,
        conversation: Conversation,
        llm_client: LLMClient,
        model: str,
    ) -> CompactionResult:
        """Stage 5 emergency LLM summarization."""
        before = len(conversation._messages)
        tokens_before = conversation.token_count

        ctx = CompactionContext(
            messages=conversation._messages,
            token_count=tokens_before,
            max_tokens=conversation.max_tokens,
            gcg=self.gcg,
        )
        new_messages = await _llm_emergency_summarize(ctx, llm_client, model)
        conversation.replace_messages(new_messages)
        self._last_stage_idx = max(self._last_stage_idx, EMERGENCY_STAGE_IDX)

        logger.info("Emergency compaction messages=%d→%d", before, len(new_messages))
        return CompactionResult(
            stage_name="auto_compact",
            messages_before=before,
            messages_after=len(new_messages),
            tokens_before=tokens_before,
            tokens_after=conversation.token_count,
            applied=True,
        )


# ── Existing API (backwards compatible) ────────────────────────────────────


def get_compaction_prompt(model: str) -> str:
    """Select the compaction prompt based on model context window size."""
    context_size = get_model_context_window(model)

    if context_size <= SMALL_CONTEXT_THRESHOLD:
        logger.debug("Using small compaction prompt model=%s context=%d", model, context_size)
        return COMPACTION_PROMPT_SMALL
    if context_size > LARGE_CONTEXT_THRESHOLD:
        logger.debug("Using large compaction prompt model=%s context=%d", model, context_size)
        return COMPACTION_PROMPT_LARGE

    logger.debug("Using medium compaction prompt model=%s context=%d", model, context_size)
    return COMPACTION_PROMPT_MEDIUM


async def compact_if_needed(
    conversation: Conversation,
    llm_client: LLMClient,
    model: str | None = None,
) -> bool:
    """Check if compaction is needed and perform it.

    Args:
        conversation: The conversation to compact.
        llm_client: LLM client for the summarization call.
        model: Model name for selecting the compaction prompt.

    Returns True if compaction was performed.
    """
    if not conversation.is_near_limit:
        return False

    model_name = model or getattr(llm_client, "model", "")
    prompt = get_compaction_prompt(model_name) if model_name else COMPACTION_PROMPT_MEDIUM

    logger.info(
        "Compaction triggered tokens=%d threshold=%d model=%s",
        conversation.token_count,
        int(conversation.max_tokens * conversation.compaction_threshold),
        model_name,
    )

    context = conversation.get_compaction_context()
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    try:
        response = await llm_client.chat(messages=summary_messages)
        conversation.compact(response.content)
        logger.info("Compaction complete new_tokens=%d", conversation.token_count)
        return True
    except Exception as exc:
        logger.error("Compaction failed: %s", exc, exc_info=True)
        return False


async def compact_now(
    conversation: Conversation,
    llm_client: LLMClient,
    model: str | None = None,
    instructions: str = "",
) -> CompactionResult:
    """Force an immediate LLM-based compaction of the conversation.

    Unlike :func:`compact_if_needed`, this always compacts regardless of the
    context threshold. Optional *instructions* guide what the compaction
    summary preserves — they are prepended as a line in the summary prompt
    (the existing compaction call does not accept instructions directly).

    Args:
        conversation: The conversation to compact.
        llm_client: LLM client for the summarization call.
        model: Model name for selecting the compaction prompt.
        instructions: Optional guidance for what the summary should preserve.

    Returns:
        A :class:`CompactionResult` describing the compaction attempt.
    """
    model_name = model or getattr(llm_client, "model", "")
    prompt = get_compaction_prompt(model_name) if model_name else COMPACTION_PROMPT_MEDIUM

    messages_before = len(conversation._messages)
    tokens_before = conversation.token_count

    logger.info(
        "Manual compaction triggered tokens=%d messages=%d model=%s",
        tokens_before,
        messages_before,
        model_name,
    )

    context = conversation.get_compaction_context()
    if instructions:
        context = f"[User instructions for summary: {instructions}]\n\n{context}"

    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    try:
        response = await llm_client.chat(messages=summary_messages, task_type="compaction")
        conversation.compact(response.content)
        logger.info(
            "Manual compaction complete messages=%d→%d new_tokens=%d",
            messages_before,
            len(conversation._messages),
            conversation.token_count,
        )
        return CompactionResult(
            stage_name="manual_compact",
            messages_before=messages_before,
            messages_after=len(conversation._messages),
            tokens_before=tokens_before,
            tokens_after=conversation.token_count,
            applied=True,
        )
    except Exception as exc:
        logger.error("Manual compaction failed: %s", exc, exc_info=True)
        return CompactionResult(
            stage_name="manual_compact",
            messages_before=messages_before,
            messages_after=len(conversation._messages),
            tokens_before=tokens_before,
            tokens_after=conversation.token_count,
            applied=False,
        )
