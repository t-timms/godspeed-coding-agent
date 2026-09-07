"""Core agent loop — the heart of Godspeed.

Hand-rolled loop following patterns proven by top-performing open-source
coding agents. The model decides when to stop. No framework overhead.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from godspeed.agent.completion_gate import (
    CompletionGateState,
    GateDecision,
    get_checklist_message,
    should_block,
)
from godspeed.agent.conversation import Conversation
from godspeed.agent.result import AgentCancelledError, AgentMetrics, ExitReason
from godspeed.hooks import HookEvent
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.llm.router import classify_task_type
from godspeed.observability.metrics import LoopMetrics, MetricsSink
from godspeed.security.dangerous import detect_dangerous_command
from godspeed.security.secrets import detect_secrets
from godspeed.tools.base import ToolCall, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry
from godspeed.tools.tasks import build_continuation_nudge

logger = logging.getLogger(__name__)

__all__ = ["MAX_ITERATIONS", "MAX_RETRIES", "MAX_SPECULATIVE_CACHE_SIZE", "agent_loop"]

MAX_ITERATIONS = 50
MAX_RETRIES = 3
STUCK_LOOP_THRESHOLD = 3
AUTO_STASH_THRESHOLD = 3
MUST_FIX_CAP = 3
MAX_SPECULATIVE_CACHE_SIZE = 10  # Max concurrent speculative tasks per iteration
# Tools that are idempotent and safe to dispatch speculatively even if their
# formal risk level is LOW (e.g. web_fetch hits external servers but is
# read-only from the local system perspective).
_SPECULATIVE_ALLOWLIST: set[str] = {
    "web_fetch",
    "repo_map",
}
VERIFIABLE_EXTENSIONS = (
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
)

# Patterns that strip meta-commentary the model might emit despite prompt instructions
_META_COMMENTARY_PATTERNS: tuple[str, ...] = (
    "No function call is needed",
    "No tool call is needed",
    "I don't need to use any tools",
    "I don't need any tools",
    "No tools are needed",
    "function call is not needed",
    "tool call is not needed",
    "No function call needed",
    "No tool call needed",
)


def _strip_meta_commentary(text: str) -> str:
    """Remove meta-commentary phrases the model emits despite prompt instructions.

    Lightweight safety net — runs on every text-only response before display.
    """
    for phrase in _META_COMMENTARY_PATTERNS:
        text = text.replace(phrase, "").strip()
    # Clean up punctuation artifacts left by removal (". . " or leading ". ")
    text = text.replace(". . ", ". ").replace(". .", ".")
    if text.startswith(". "):
        text = text[2:]
    # Clean up double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def _last_message_is_nudge(messages: list[dict[str, Any]], nudge: str) -> bool:
    """Return True when the last message is already the given nudge.

    Guards against re-injecting the same nudge on consecutive iterations —
    the model may legitimately respond to the nudge with a tool call, and we
    don't want to stack duplicate nudges in the conversation.
    """
    if not messages:
        return False
    last = messages[-1]
    return last.get("role") == "user" and last.get("content") == nudge


# Callback type aliases for clarity
OnAssistantText = Callable[[str], None]
OnToolCall = Callable[[str, dict[str, Any]], None]
OnToolResult = Callable[[str, ToolResult], None]
OnPermissionDenied = Callable[[str, str], None]
OnChunk = Callable[[str], None]
OnParallelStart = Callable[[list[tuple[str, dict[str, Any]]]], None]
OnParallelComplete = Callable[[list[tuple[str, str, bool]]], None]
OnThinking = Callable[[str], None]


@dataclass
class _LoopState:
    """Mutable state and loop config shared across helper functions."""

    # --- Mutable state ---
    consecutive_writes: int = 0
    consecutive_successful_edits: int = 0
    recent_change_descriptions: list[str] = field(default_factory=list)
    auto_stashed: bool = False
    must_fix_injections: int = 0
    recent_error_hashes: list[str] = field(default_factory=list)
    speculative_cache: dict[str, asyncio.Task[ToolResult]] = field(default_factory=dict)
    pending_background_tasks: list[asyncio.Task] = field(default_factory=list)
    verify_failure_count: int = 0
    budget_prompt_injected: bool = False
    has_edits_since_verify: bool = False
    stop_attempts: int = 0

    # --- Loop config (set once at loop start) ---
    auto_fix_retries: int = 3
    auto_commit: bool = False
    auto_commit_threshold: int = 5
    stuck_threshold: int = 3
    stash_threshold: int = 3
    must_fix_cap: int = 3
    max_speculative_cache_size: int = 10
    competition_mode: bool = False
    llm_max_retries: int = 3
    llm_retry_delay: float = 2.0
    overflow_compacted: bool = False
    budget_verify_cap: int = 3


async def agent_loop(
    user_input: str,
    conversation: Conversation,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    on_assistant_text: OnAssistantText | None = None,
    on_tool_call: OnToolCall | None = None,
    on_tool_result: OnToolResult | None = None,
    on_permission_denied: OnPermissionDenied | None = None,
    on_assistant_chunk: OnChunk | None = None,
    max_iterations: int | None = None,
    pause_event: asyncio.Event | None = None,
    cancel_event: asyncio.Event | None = None,
    hook_executor: Any | None = None,
    retrieval_subagent: Any | None = None,
    graduated_compactor: Any | None = None,
    parallel_tool_calls: bool = True,
    skip_user_message: bool = False,
    auto_fix_retries: int = 3,
    auto_commit: bool = False,
    auto_commit_threshold: int = 5,
    max_retries: int | None = None,
    stuck_loop_threshold: int | None = None,
    auto_stash_threshold: int | None = None,
    must_fix_cap: int | None = None,
    max_speculative_cache_size: int | None = None,
    on_parallel_start: OnParallelStart | None = None,
    on_parallel_complete: OnParallelComplete | None = None,
    on_thinking: OnThinking | None = None,
    metrics: AgentMetrics | None = None,
    metrics_sink: MetricsSink | None = None,
    competition_mode: bool = False,
    llm_max_retries: int = 3,
    llm_retry_delay: float = 2.0,
    budget_verify_cap: int = 3,
    task_store: Any | None = None,
    completion_gate: bool = False,
) -> str:
    """Run the agent loop until the model stops calling tools.

    Flow:
    1. Add user input to conversation
    2. Send conversation + tool schemas to LLM
    3. If response has tool_calls: check permissions, execute, record results
    4. If response is text-only: return it (model decided to stop)
    5. On malformed response: retry up to MAX_RETRIES

    Args:
        user_input: The user's message.
        conversation: Conversation history manager.
        llm_client: LLM client for API calls.
        tool_registry: Registry of available tools.
        tool_context: Execution context for tools.
        on_assistant_text: Callback(text) for complete assistant output.
        on_tool_call: Callback(tool_name, args) before tool execution.
        on_tool_result: Callback(tool_name, result) after tool execution.
        on_permission_denied: Callback(tool_name, reason) when permission denied.
        on_assistant_chunk: Callback(text) for streaming chunks. When provided,
            uses streaming LLM calls instead of batch calls.
        max_iterations: Override the default iteration limit (MAX_ITERATIONS).
        max_retries: Override MAX_RETRIES for malformed tool calls.
        stuck_loop_threshold: Override effective_stuck_threshold for stuck detection.
        auto_stash_threshold: Override effective_stash_threshold for auto-stash.
        must_fix_cap: Override effective_must_fix_cap for must-fix injections.
        pause_event: Optional asyncio.Event for pause/resume. When cleared,
            the loop waits at the top of each iteration until set again.
        cancel_event: Optional asyncio.Event for mid-turn cancellation.
            When set, the loop raises AgentCancelledError at the next safe
            checkpoint — between streaming chunks, before an LLM call,
            or before dispatching tools — so the user can interrupt a
            long-running turn immediately instead of waiting for the
            iteration boundary. The TUI binds Ctrl+C to set this event.
        parallel_tool_calls: Execute multiple tool calls concurrently when True
            (default). Falls back to sequential when False or for single calls.
        task_store: Optional TaskStore whose open tasks drive a continuation
            nudge injected into the model's context before each LLM call.

    Returns:
        The final assistant text response.
    """
    iteration_limit = max_iterations if max_iterations is not None else MAX_ITERATIONS
    effective_max_retries = max_retries if max_retries is not None else MAX_RETRIES

    if not skip_user_message and user_input:
        conversation.add_user_message(user_input)
    tool_schemas = tool_registry.get_schemas()

    retries = 0
    final_text = ""
    state = _LoopState(
        auto_fix_retries=auto_fix_retries,
        auto_commit=auto_commit and not competition_mode,
        auto_commit_threshold=auto_commit_threshold,
        stuck_threshold=(
            stuck_loop_threshold if stuck_loop_threshold is not None else STUCK_LOOP_THRESHOLD
        ),
        stash_threshold=(
            auto_stash_threshold if auto_stash_threshold is not None else AUTO_STASH_THRESHOLD
        ),
        must_fix_cap=(
            0 if competition_mode else (must_fix_cap if must_fix_cap is not None else MUST_FIX_CAP)
        ),
        max_speculative_cache_size=(
            max_speculative_cache_size
            if max_speculative_cache_size is not None
            else MAX_SPECULATIVE_CACHE_SIZE
        ),
        competition_mode=competition_mode,
        llm_max_retries=llm_max_retries,
        llm_retry_delay=llm_retry_delay,
        budget_verify_cap=budget_verify_cap,
    )

    loop_metrics = metrics.loop if metrics is not None else LoopMetrics()

    for iteration in range(iteration_limit):
        iter_t0 = time.monotonic()

        # Cancel check: before pause check, so a cancel delivered during a
        # pause doesn't strand the loop. Raises AgentCancelledError; caller unwinds.
        _check_cancel(cancel_event)

        # Clear stale speculative tasks from previous iteration
        for task in state.speculative_cache.values():
            task.cancel()
        state.speculative_cache.clear()

        # Pause/resume: if pause_event exists and is cleared, wait for it
        if pause_event is not None and not pause_event.is_set():
            logger.info("Agent loop paused at iteration=%d", iteration)
            await pause_event.wait()
            logger.info("Agent loop resumed at iteration=%d", iteration)

        # Cancel check #2: may have been set while we were paused.
        _check_cancel(cancel_event)

        logger.debug("Agent loop iteration=%d tokens=%d", iteration, conversation.token_count)

        # Check context thresholds and apply graduated compaction
        if not state.competition_mode:
            await _check_context_and_compact(
                conversation,
                llm_client,
                hook_executor,
                graduated_compactor,
                loop_metrics,
            )

        await _drain_background_tasks(state, conversation, metrics)

        # Task-aware routing: classify the upcoming call from conversation
        # state. Cheap heuristic (no extra LLM call); resolves to one of
        # plan/edit/read/shell. The router translates that to a model
        # via settings.routing (or the cheap_model/strong_model shortcuts).
        task_type = classify_task_type(conversation.messages)

        # Continuation nudge: if the task store has open tasks, remind the
        # model to keep working through them before it decides to stop.
        if task_store is not None:
            nudge = build_continuation_nudge(task_store.list_all())
            if nudge is not None and not _last_message_is_nudge(conversation.messages, nudge):
                conversation.add_user_message(nudge)

        # Acceptance summary: surface failing criteria alongside the task nudge.
        acceptance_summary = _acceptance_summary(tool_context)
        if acceptance_summary is not None and not _last_message_is_nudge(
            conversation.messages, acceptance_summary
        ):
            conversation.add_user_message(acceptance_summary)

        # Call LLM (streaming or batch) with retry for transient errors
        llm_t0 = time.monotonic()
        response: ChatResponse | None = None
        last_exc: Exception | None = None
        for llm_attempt in range(state.llm_max_retries + 1):
            try:
                if on_assistant_chunk is not None:
                    response = await _streaming_call(
                        llm_client,
                        conversation.messages,
                        tool_schemas if tool_schemas else None,
                        on_assistant_chunk,
                        tool_registry=tool_registry,
                        tool_context=tool_context,
                        speculative_cache=state.speculative_cache,
                        cancel_event=cancel_event,
                        task_type=task_type,
                        max_speculative_cache_size=state.max_speculative_cache_size,
                    )
                else:
                    response = await llm_client.chat(
                        messages=conversation.messages,
                        tools=tool_schemas if tool_schemas else None,
                        task_type=task_type,
                    )
                break
            except AgentCancelledError:
                # Finalize with INTERRUPTED and unwind — don't wrap in LLM_ERROR.
                logger.info("Agent loop cancelled mid-turn at iteration=%d", iteration)
                if metrics is not None:
                    metrics.iterations_used = iteration
                    metrics.finalize(ExitReason.INTERRUPTED)
                raise
            except Exception as exc:
                # Import here to avoid circular import at module level
                from godspeed.llm.client import BudgetExceededError

                if isinstance(exc, BudgetExceededError):
                    msg = (
                        f"Budget exceeded (${exc.spent:.4f} / ${exc.limit:.2f} limit). "
                        "Use /budget to increase the limit."
                    )
                    logger.warning("Budget exceeded spent=%.4f limit=%.2f", exc.spent, exc.limit)
                    if hook_executor is not None:
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            functools.partial(
                                hook_executor.fire,
                                HookEvent.BUDGET_EXCEEDED,
                                cost_usd=exc.spent,
                            ),
                        )
                    if metrics is not None:
                        metrics.iterations_used = iteration
                        metrics.finalize(ExitReason.BUDGET_EXCEEDED)
                    return msg
                if _is_context_overflow(exc) and not state.overflow_compacted:
                    state.overflow_compacted = True
                    logger.warning(
                        "Context overflow at iteration=%d — compacting and retrying once",
                        iteration,
                    )
                    max_toks = conversation.max_tokens
                    toks = max(conversation.token_count, max_toks)
                    if graduated_compactor is not None:
                        graduated_compactor.apply_stages(conversation, toks, max_toks)
                        await graduated_compactor.emergency_compact(
                            conversation, llm_client, getattr(llm_client, "model", "")
                        )
                    else:
                        await _compact_conversation(conversation, llm_client)
                    loop_metrics.record_compaction()
                    continue
                last_exc = exc
                if llm_attempt < state.llm_max_retries:
                    delay = state.llm_retry_delay * (2**llm_attempt)
                    logger.warning(
                        "LLM call failed attempt=%d/%d delay=%.1fs error=%s",
                        llm_attempt + 1,
                        state.llm_max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("LLM call failed error=%s", exc, exc_info=True)
                    if metrics is not None:
                        metrics.iterations_used = iteration
                        metrics.finalize(ExitReason.LLM_ERROR)
                    return f"Error: LLM call failed — {exc}"
        if response is None:
            # Defensive — should not happen because we return early on final failure
            return f"Error: LLM call failed — {last_exc}"
        loop_metrics.record_llm_call(time.monotonic() - llm_t0)
        loop_metrics.record_token_count(conversation.token_count)

        # Display thinking blocks (extended thinking for Anthropic models)
        if response.thinking and on_thinking:
            on_thinking(response.thinking)

        # Handle text response (model decided to stop)
        if not response.has_tool_calls:
            final_text = _strip_meta_commentary(response.content)
            if completion_gate:
                state.stop_attempts += 1
                gate_state = CompletionGateState(
                    has_edits_since_verify=state.has_edits_since_verify,
                    tasks_open=_tasks_open(task_store, tool_context),
                    stop_attempts=state.stop_attempts,
                )
                if should_block(gate_state) == GateDecision.BLOCK:
                    if final_text:
                        conversation.add_assistant_message(
                            content=final_text,
                            reasoning_content=response.thinking,
                        )
                    conversation.add_user_message(get_checklist_message())
                    logger.info(
                        "Completion gate blocked stop attempt=%d edits=%s",
                        state.stop_attempts,
                        state.has_edits_since_verify,
                    )
                    continue
            if final_text:
                # NEW: Pass reasoning_content for DeepSeek V4 multi-turn
                conversation.add_assistant_message(
                    content=final_text,
                    reasoning_content=response.thinking,  # Store thinking/reasoning
                )
                # Skip Markdown re-render if we already streamed the text
                if on_assistant_text and on_assistant_chunk is None:
                    on_assistant_text(final_text)
            if metrics is not None:
                metrics.iterations_used = iteration + 1
                metrics.finalize(ExitReason.STOPPED)
            return final_text

        # Handle tool calls
        # Always use standard message handling to maintain proper API compatibility
        # DeepSeek V4: reasoning_content stripped before API calls in _call_deepseek_direct
        conversation.add_assistant_message(
            content=response.content,
            tool_calls=response.tool_calls,
            reasoning_content=response.thinking,
        )

        if response.content and on_assistant_text:
            on_assistant_text(response.content)

        # --- Phase 1: Parse and pre-flight all tool calls ---
        parsed_calls: list[tuple[dict[str, Any], ToolCall | None]] = []
        for raw_tc in response.tool_calls:
            parsed_calls.append((raw_tc, _parse_tool_call(raw_tc)))

        # --- Phase 2: Permission checks + pre-tool hooks (parallel) ---
        # Malformed calls are trivial O(1) checks handled sequentially first.
        # Permission and hook evaluations for valid calls are I/O-bound and
        # batched with asyncio.gather for latency reduction.
        permitted: list[ToolCall] = []
        valid_calls: list[ToolCall] = []
        for raw_tc, tool_call in parsed_calls:
            _check_cancel(cancel_event)
            if tool_call is None:
                retries += 1
                if retries > effective_max_retries:
                    if metrics is not None:
                        metrics.iterations_used = iteration + 1
                        metrics.finalize(ExitReason.TOOL_ERROR)
                    return "Error: Too many malformed tool calls from the model."
                conversation.add_tool_result(
                    tool_call_id=raw_tc.get("id", ""),
                    content=(
                        "Error: Malformed tool call. Please try again with valid JSON arguments."
                    ),
                )
                continue
            retries = 0
            valid_calls.append(tool_call)

        if valid_calls:

            async def _eval_one(tc: ToolCall) -> ToolCall | None:
                if tool_context.permissions is not None:
                    if hook_executor is not None:
                        serialized_args = json.dumps(tc.arguments, default=str)
                        if tc.tool_name == "shell" and detect_dangerous_command(
                            str(tc.arguments.get("command", ""))
                        ):
                            await asyncio.get_running_loop().run_in_executor(
                                None,
                                functools.partial(
                                    hook_executor.fire,
                                    HookEvent.DANGEROUS_COMMAND,
                                    tool=tc.tool_name,
                                    pattern=str(tc.arguments.get("command", ""))[:200],
                                ),
                            )
                        if detect_secrets(serialized_args):
                            await asyncio.get_running_loop().run_in_executor(
                                None,
                                functools.partial(
                                    hook_executor.fire,
                                    HookEvent.SECRET_DETECTED,
                                    tool=tc.tool_name,
                                    pattern="arguments",
                                ),
                            )
                    if inspect.iscoroutinefunction(tool_context.permissions.evaluate):
                        decision = await tool_context.permissions.evaluate(tc)
                    else:
                        decision = tool_context.permissions.evaluate(tc)
                    if decision == "deny":
                        reason = f"Permission denied for {tc.format_for_permission}"
                        logger.info("Permission denied tool=%s", tc.tool_name)
                        loop_metrics.record_tool_denial()
                        if on_permission_denied:
                            on_permission_denied(tc.tool_name, reason)
                        if hook_executor is not None:
                            await asyncio.get_running_loop().run_in_executor(
                                None,
                                functools.partial(
                                    hook_executor.fire,
                                    HookEvent.PERMISSION_DENIED,
                                    tool=tc.tool_name,
                                    pattern=reason,
                                ),
                            )
                        conversation.add_tool_result(
                            tool_call_id=tc.call_id,
                            content=(
                                f"DENIED: {reason}. "
                                "This tool call was blocked by the permission engine."
                            ),
                        )
                        return None
                    if hook_executor is not None:
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            functools.partial(
                                hook_executor.fire,
                                HookEvent.POST_PERMISSION_CHECK,
                                tool=tc.tool_name,
                                decision="granted",
                            ),
                        )

                if on_tool_call:
                    on_tool_call(tc.tool_name, tc.arguments)

                if hook_executor is not None:
                    hook_ok = await asyncio.get_running_loop().run_in_executor(
                        None, hook_executor.run_pre_tool, tc.tool_name
                    )
                    if not hook_ok:
                        logger.info("Pre-tool hook blocked tool=%s", tc.tool_name)
                        conversation.add_tool_result(
                            tool_call_id=tc.call_id,
                            content="BLOCKED: Pre-tool hook returned non-zero exit.",
                        )
                        return None

                return tc

            results = await asyncio.gather(*[_eval_one(tc) for tc in valid_calls])
            permitted = [r for r in results if r is not None]

        if not permitted:
            await _drain_background_tasks(state, conversation, metrics)
            loop_metrics.record_iteration(time.monotonic() - iter_t0)
            if metrics_sink is not None:
                metrics_sink.emit("loop_iteration", loop_metrics.to_dict())
            continue

        # Intercept navigation tools through retrieval subagent (Phase 3)
        navigation_tools = {"code_search", "grep", "glob", "repo_map"}
        if retrieval_subagent is not None:
            nav_calls = [tc for tc in permitted if tc.tool_name in navigation_tools]
            non_nav_calls = [tc for tc in permitted if tc.tool_name not in navigation_tools]

            for tc in nav_calls:
                query = tc.arguments.get("pattern") or tc.arguments.get("query", "")
                if query:
                    r = await retrieval_subagent.retrieve(query=query)
                    fmt = retrieval_subagent.format_spans_for_agent(r.spans)
                    conversation.add_tool_result(
                        tool_call_id=tc.call_id,
                        content=fmt,
                    )
                    if on_tool_result:
                        on_tool_result(
                            tc.tool_name,
                            ToolResult(
                                output=fmt,
                                is_error=not r.spans,
                            ),
                        )
                else:
                    conversation.add_tool_result(
                        tool_call_id=tc.call_id,
                        content="Retrieval requires a pattern or query argument.",
                    )

            permitted = non_nav_calls

        if not permitted:
            await _drain_background_tasks(state, conversation, metrics)
            loop_metrics.record_iteration(time.monotonic() - iter_t0)
            if metrics_sink is not None:
                metrics_sink.emit("loop_iteration", loop_metrics.to_dict())
            continue

        # --- Phase 4: Execute tools (parallel or sequential) ---
        if parallel_tool_calls and len(permitted) > 1:
            await _dispatch_parallel(
                permitted,
                tool_registry,
                tool_context,
                hook_executor,
                on_tool_result,
                on_parallel_start,
                on_parallel_complete,
                metrics,
                state,
                conversation,
                cancel_event,
                llm_client,
            )
        else:
            await _dispatch_sequential(
                permitted,
                tool_registry,
                tool_context,
                hook_executor,
                on_tool_result,
                metrics,
                state,
                conversation,
                cancel_event,
                llm_client,
            )

        loop_metrics.record_iteration(time.monotonic() - iter_t0)
        if metrics_sink is not None:
            metrics_sink.emit("loop_iteration", loop_metrics.to_dict())

        # Drain completed background tasks from this iteration before proceeding
        await _drain_background_tasks(state, conversation, metrics)

    if metrics is not None:
        metrics.iterations_used = iteration_limit
        metrics.finalize(ExitReason.MAX_ITERATIONS)
    return "Error: Reached maximum iterations. The task may be too complex for a single turn."


async def _dispatch_parallel(
    permitted: list[ToolCall],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    hook_executor: Any | None,
    on_tool_result: OnToolResult | None,
    on_parallel_start: OnParallelStart | None,
    on_parallel_complete: OnParallelComplete | None,
    metrics: AgentMetrics | None,
    state: _LoopState,
    conversation: Conversation,
    cancel_event: asyncio.Event | None,
    llm_client: LLMClient,
) -> None:
    """Execute tools in parallel (read-only) and sequential (write) batches.

    Handles auto-verify, auto-stash, auto-commit, and stuck-loop detection.
    """
    from godspeed.tools.base import RiskLevel

    def _is_concurrency_safe(tc: ToolCall) -> bool:
        tool = tool_registry.get(tc.tool_name)
        if tool is None:
            return False
        return bool(getattr(tool, "concurrency_safe", tool.risk_level == RiskLevel.READ_ONLY))

    all_calls = list(permitted)
    if on_parallel_start:
        on_parallel_start([(tc.tool_name, tc.arguments) for tc in all_calls])

    t0 = time.monotonic()
    results: list[ToolResult] = []
    batch: list[ToolCall] = []

    async def _flush_batch() -> None:
        if not batch:
            return
        sem = asyncio.Semaphore(10)

        async def _run(tc: ToolCall) -> ToolResult:
            async with sem:
                return await tool_registry.dispatch(tc, tool_context)

        coros = []
        for c in batch:
            cached_task = state.speculative_cache.pop(c.call_id, None)
            if cached_task is not None:
                logger.debug("Speculative hit tool=%s call_id=%s", c.tool_name, c.call_id)
                if metrics is not None:
                    metrics.loop.record_speculative_hit()
                coros.append(cached_task)
            else:
                if metrics is not None:
                    metrics.loop.record_speculative_miss()
                coros.append(_run(c))
        results.extend(await asyncio.gather(*coros))
        batch.clear()

    for tc in all_calls:
        if _is_concurrency_safe(tc):
            batch.append(tc)
            continue
        await _flush_batch()
        _check_cancel(cancel_event)
        results.append(await tool_registry.dispatch(tc, tool_context))
    await _flush_batch()

    permitted_ordered = all_calls
    batch_latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Parallel dispatch completed tools=%d latency_ms=%.1f",
        len(permitted_ordered),
        batch_latency_ms,
    )

    # Process results
    for tool_call, result in zip(permitted_ordered, results, strict=True):
        _check_cancel(cancel_event)
        if hook_executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, hook_executor.run_post_tool, tool_call.tool_name
            )
        if on_tool_result:
            on_tool_result(tool_call.tool_name, result)
        if metrics is not None:
            metrics.record_tool_call(tool_call.tool_name, result.is_error)
            metrics.loop.record_tool_call(
                tool_call.tool_name,
                duration_sec=batch_latency_ms / 1000 / len(permitted_ordered),
                is_error=result.is_error,
            )
        if tool_context.audit is not None:
            await tool_context.audit.arecord(
                event_type="tool_call",
                detail={
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output_length": len(result.output),
                    "is_error": result.is_error,
                    "latency_ms": round(batch_latency_ms / len(permitted_ordered), 1),
                    "parallel": True,
                },
                outcome="error" if result.is_error else "success",
            )
        result_content = result.error if result.is_error else result.output
        conversation.add_tool_result(
            tool_call_id=tool_call.call_id,
            content=result_content or "",
        )

    # Post-processing: auto-verify, auto-stash, auto-commit, stuck-loop
    await _post_process_results(
        permitted_ordered,
        results,
        state,
        conversation,
        tool_registry,
        tool_context,
        llm_client,
        metrics,
        hook_executor,
    )

    if on_parallel_complete:
        on_parallel_complete(
            [
                (tc.tool_name, str(r.error) if r.is_error else str(r.output), r.is_error)
                for tc, r in zip(permitted_ordered, results, strict=True)
            ]
        )


async def _dispatch_sequential(
    permitted: list[ToolCall],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    hook_executor: Any | None,
    on_tool_result: OnToolResult | None,
    metrics: AgentMetrics | None,
    state: _LoopState,
    conversation: Conversation,
    cancel_event: asyncio.Event | None,
    llm_client: LLMClient,
) -> None:
    """Execute tools sequentially, one at a time."""
    for tool_call in permitted:
        _check_cancel(cancel_event)
        t0 = time.monotonic()
        cached_task = state.speculative_cache.pop(tool_call.call_id, None)
        if cached_task is not None:
            logger.debug(
                "Speculative hit (sequential) tool=%s call_id=%s",
                tool_call.tool_name,
                tool_call.call_id,
            )
            if metrics is not None:
                metrics.loop.record_speculative_hit()
            result = await cached_task
        else:
            if metrics is not None:
                metrics.loop.record_speculative_miss()
            result = await tool_registry.dispatch(tool_call, tool_context)
        latency_sec = time.monotonic() - t0

        if hook_executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, hook_executor.run_post_tool, tool_call.tool_name
            )
        if on_tool_result:
            on_tool_result(tool_call.tool_name, result)
        if metrics is not None:
            metrics.record_tool_call(tool_call.tool_name, result.is_error)
            metrics.loop.record_tool_call(
                tool_call.tool_name,
                duration_sec=latency_sec,
                is_error=result.is_error,
            )
        if tool_context.audit is not None:
            await tool_context.audit.arecord(
                event_type="tool_call",
                detail={
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output_length": len(result.output),
                    "is_error": result.is_error,
                    "latency_ms": round(latency_sec * 1000, 1),
                },
                outcome="error" if result.is_error else "success",
            )
        result_content = result.error if result.is_error else result.output
        conversation.add_tool_result(
            tool_call_id=tool_call.call_id,
            content=result_content or "",
        )
        await _post_process_single_result(
            tool_call,
            result,
            state,
            conversation,
            tool_registry,
            tool_context,
            llm_client,
            metrics,
            hook_executor,
        )


async def _post_process_results(
    tool_calls: list[ToolCall],
    results: list[ToolResult],
    state: _LoopState,
    conversation: Conversation,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    llm_client: LLMClient,
    metrics: AgentMetrics | None,
    hook_executor: Any | None = None,
) -> None:
    """Post-process a batch of tool results (auto-verify, auto-stash, etc.).

    Optimized: single-pass iteration over tool_calls/results to avoid repeated zip.
    """
    # Pre-compute zipped pairs once to avoid repeated iteration
    paired = list(zip(tool_calls, results, strict=True))

    # Single-pass: collect write operations and errors
    write_results: list[tuple[ToolCall, ToolResult]] = []
    has_non_write = False

    for tc, result in paired:
        # Check for write operations
        if not result.is_error and tc.tool_name in ("file_edit", "file_write", "diff_apply"):
            write_results.append((tc, result))
            state.has_edits_since_verify = True
        elif tc.tool_name not in ("file_edit", "file_write", "diff_apply"):
            has_non_write = True

        if not result.is_error and tc.tool_name in ("verify", "test_runner"):
            state.has_edits_since_verify = False

        # Stuck-loop detection (inline to avoid another pass)
        if result.is_error:
            error_hash = hashlib.sha256((result.error or "").encode()).hexdigest()
            state.recent_error_hashes.append(error_hash)
            if len(state.recent_error_hashes) > state.stuck_threshold:
                state.recent_error_hashes.pop(0)
        else:
            state.recent_error_hashes.clear()

        # Auto-verify for write operations — fire-and-forget in background.
        # The next iteration drains pending tasks before the LLM call.
        if (
            not result.is_error
            and tc.tool_name in ("file_edit", "file_write")
            and tool_registry.has_tool("verify")
        ):
            file_path = tc.arguments.get("file_path", "")
            if file_path and file_path.endswith(VERIFIABLE_EXTENSIONS):
                task = asyncio.create_task(
                    _auto_verify_background(
                        file_path,
                        tc.call_id,
                        tool_registry,
                        tool_context,
                        state.auto_fix_retries,
                    )
                )
                state.pending_background_tasks.append(task)

        # Update write tracking from collected results
    batch_writes = len(write_results)
    if has_non_write:
        state.consecutive_writes = batch_writes
        state.consecutive_successful_edits = batch_writes
        state.recent_change_descriptions = [
            f"{tc.tool_name} {tc.arguments.get('file_path', '?')}" for tc, _ in write_results
        ]
    else:
        state.consecutive_writes += batch_writes

    # Stuck-loop detection: inject hint if needed
    if (
        len(state.recent_error_hashes) == state.stuck_threshold
        and len(set(state.recent_error_hashes)) == 1
    ):
        logger.warning("Stuck loop detected: %d identical errors", state.stuck_threshold)
        last_error = state.recent_error_hashes[0] if state.recent_error_hashes else ""
        if hook_executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    hook_executor.fire,
                    HookEvent.STUCK_LOOP_DETECTED,
                    iterations=state.stuck_threshold,
                    last_error=last_error,
                ),
            )
        conversation.add_user_message(
            f"You have failed {state.stuck_threshold} times with the same error. "
            "Stop, explain what is wrong, and try a completely different approach."
        )
        state.recent_error_hashes.clear()

    # Auto-stash check
    if (
        state.consecutive_writes >= state.stash_threshold
        and not state.auto_stashed
        and tool_registry.has_tool("git")
    ):
        stash_call = ToolCall(
            tool_name="git",
            arguments={"action": "stash"},
            call_id=tool_calls[-1].call_id,
        )
        stash_result = await tool_registry.dispatch(stash_call, tool_context)
        if (
            not stash_result.is_error
            and "nothing to stash" not in (stash_result.output or "").lower()
        ):
            state.auto_stashed = True
            logger.info(
                "Auto-stash triggered after %d consecutive writes", state.consecutive_writes
            )
            conversation.add_tool_result(
                tool_call_id=stash_call.call_id,
                content=(
                    f"[Auto-stash] Saved working state after "
                    f"{state.consecutive_writes} consecutive file edits. "
                    "Use git stash_pop to restore if needed."
                ),
            )

    # Auto-commit tracking (uses write_results from single pass)
    for tc, _r in write_results:
        state.consecutive_successful_edits += 1
        desc = f"{tc.tool_name} {tc.arguments.get('file_path', '?')}"
        state.recent_change_descriptions.append(desc)

    # Budget prompt: inject after N write operations to prevent over-editing.
    # Addresses the agent-in-loop regression where verify feedback causes the
    # agent to produce increasingly larger patches that break previously-passing
    # tests. The budget prompt forces timely submission.
    if (
        batch_writes > 0
        and state.consecutive_writes >= state.budget_verify_cap
        and not state.budget_prompt_injected
    ):
        state.budget_prompt_injected = True
        logger.info(
            "Budget prompt injected after %d writes (cap=%d)",
            state.consecutive_writes,
            state.budget_verify_cap,
        )
        conversation.add_user_message(
            f"You have made {state.consecutive_writes} edits so far. "
            "When the current change passes verification, submit the result. "
            "Avoid unnecessary further edits."
        )

    if state.auto_commit and state.consecutive_successful_edits >= state.auto_commit_threshold:
        committed = await _try_auto_commit(
            list(state.recent_change_descriptions),
            tool_context,
            llm_client,
            conversation,
            tool_calls[-1].call_id,
        )
        if committed:
            state.consecutive_successful_edits = 0
            state.recent_change_descriptions.clear()


async def _post_process_single_result(
    tool_call: ToolCall,
    result: ToolResult,
    state: _LoopState,
    conversation: Conversation,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    llm_client: LLMClient,
    metrics: AgentMetrics | None,
    hook_executor: Any | None = None,
) -> None:
    """Post-process a single tool result."""
    if (
        not result.is_error
        and tool_call.tool_name in ("file_edit", "file_write")
        and tool_registry.has_tool("verify")
    ):
        file_path = tool_call.arguments.get("file_path", "")
        if file_path and file_path.endswith(VERIFIABLE_EXTENSIONS):
            task = asyncio.create_task(
                _auto_verify_background(
                    file_path,
                    tool_call.call_id,
                    tool_registry,
                    tool_context,
                    state.auto_fix_retries,
                )
            )
            state.pending_background_tasks.append(task)

    if not result.is_error and tool_call.tool_name in ("file_edit", "file_write", "diff_apply"):
        state.consecutive_writes += 1
        state.has_edits_since_verify = True
        if (
            state.consecutive_writes >= state.stash_threshold
            and not state.auto_stashed
            and tool_registry.has_tool("git")
        ):
            stash_call = ToolCall(
                tool_name="git",
                arguments={"action": "stash"},
                call_id=tool_call.call_id,
            )
            stash_result = await tool_registry.dispatch(stash_call, tool_context)
            if (
                not stash_result.is_error
                and "nothing to stash" not in (stash_result.output or "").lower()
            ):
                state.auto_stashed = True
                logger.info(
                    "Auto-stash triggered after %d consecutive writes", state.consecutive_writes
                )
                conversation.add_tool_result(
                    tool_call_id=stash_call.call_id,
                    content=(
                        f"[Auto-stash] Saved working state after "
                        f"{state.consecutive_writes} consecutive file edits. "
                        "Use git stash_pop to restore if needed."
                    ),
                )
        state.consecutive_successful_edits += 1
        desc = f"{tool_call.tool_name} {tool_call.arguments.get('file_path', '?')}"
        state.recent_change_descriptions.append(desc)
        # Budget prompt injection for sequential dispatch path
        if state.consecutive_writes >= state.budget_verify_cap and not state.budget_prompt_injected:
            state.budget_prompt_injected = True
            logger.info(
                "Budget prompt injected after %d writes (cap=%d, seq)",
                state.consecutive_writes,
                state.budget_verify_cap,
            )
            conversation.add_user_message(
                f"You have made {state.consecutive_writes} edits. "
                "If the fix is correct, STOP NOW and submit. "
                "If verify/tests still fail after 2 more attempts, submit your "
                "best effort — a partial fix is better than none at all."
            )
        if state.auto_commit and state.consecutive_successful_edits >= state.auto_commit_threshold:
            committed = await _try_auto_commit(
                list(state.recent_change_descriptions),
                tool_context,
                llm_client,
                conversation,
                tool_call.call_id,
            )
            if committed:
                state.consecutive_successful_edits = 0
                state.recent_change_descriptions.clear()
    elif not result.is_error and tool_call.tool_name in ("verify", "test_runner"):
        state.has_edits_since_verify = False
    else:
        state.consecutive_writes = 0
        state.consecutive_successful_edits = 0
        state.recent_change_descriptions = []

    # Stuck-loop detection
    if result.is_error:
        error_hash = hashlib.sha256((result.error or "").encode()).hexdigest()
        state.recent_error_hashes.append(error_hash)
        if len(state.recent_error_hashes) > state.stuck_threshold:
            state.recent_error_hashes.pop(0)
        if (
            len(state.recent_error_hashes) == state.stuck_threshold
            and len(set(state.recent_error_hashes)) == 1
        ):
            logger.warning("Stuck loop detected: %d identical errors", state.stuck_threshold)
            last_error = state.recent_error_hashes[0] if state.recent_error_hashes else ""
            if hook_executor is not None:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        hook_executor.fire,
                        HookEvent.STUCK_LOOP_DETECTED,
                        iterations=state.stuck_threshold,
                        last_error=last_error,
                    ),
                )
            conversation.add_user_message(
                f"You have failed {state.stuck_threshold} times with the same error. "
                "Stop, explain what is wrong, and try a completely different approach."
            )
            state.recent_error_hashes.clear()
    else:
        state.recent_error_hashes.clear()


async def _auto_verify_file(
    file_path: str,
    parent_call_id: str,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    auto_fix_retries: int,
) -> ToolResult:
    """Run auto-verify on a file, using the retry loop when retries > 0.

    Falls back to plain verify dispatch when retries are disabled (0).
    """
    from pathlib import Path

    from godspeed.tools.verify import _EXTENSION_MAP, _verify_with_retry

    resolved = Path(file_path) if Path(file_path).is_absolute() else (tool_context.cwd / file_path)
    suffix = resolved.suffix.lower()
    lang = _EXTENSION_MAP.get(suffix)

    if auto_fix_retries > 0 and lang is not None:
        # Run in thread to avoid blocking the event loop
        return await asyncio.to_thread(
            _verify_with_retry,
            resolved=resolved,
            display_path=file_path,
            lang=lang,
            cwd=tool_context.cwd,
            max_retries=auto_fix_retries,
        )

    # Fallback: plain verify dispatch (one-shot)
    verify_call = ToolCall(
        tool_name="verify",
        arguments={"file_path": file_path},
        call_id=parent_call_id,
    )
    return await tool_registry.dispatch(verify_call, tool_context)


@dataclass
class _VerifyResult:
    call_id: str
    output: str
    must_fix_file: str
    must_fix_text: str
    must_fix_increment: bool


async def _auto_verify_background(
    file_path: str,
    call_id: str,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    auto_fix_retries: int,
) -> _VerifyResult:
    """Run auto-verify in background. Returns a result for the drain to apply.

    Never mutates conversation or state directly — the caller processes results
    in the main loop, avoiding data races with in-flight LLM calls.
    """
    verify_result = await _auto_verify_file(
        file_path, call_id, tool_registry, tool_context, auto_fix_retries
    )
    verify_text = verify_result.error or verify_result.output or ""
    from godspeed.tools.verify import REMAINING_ERRORS_FINGERPRINT

    return _VerifyResult(
        call_id=call_id,
        output=verify_result.output or "",
        must_fix_file=file_path,
        must_fix_text=verify_text,
        must_fix_increment=REMAINING_ERRORS_FINGERPRINT in verify_text,
    )


async def _drain_background_tasks(
    state: _LoopState,
    conversation: Conversation,
    metrics: AgentMetrics | None,
) -> None:
    """Non-blocking drain of completed background tasks.

    Only processes tasks that have already completed. Never awaits in-flight
    tasks, avoiding data races where a task mutates ``conversation`` while
    an LLM call is in progress.
    """
    if not state.pending_background_tasks:
        return
    # Yield to event loop so recently-scheduled tasks have a chance to execute
    # before we check ``t.done()``. This is needed because ``create_task`` only
    # schedules the coroutine — it doesn't run until the next ``await``.
    await asyncio.sleep(0)
    still_pending: list[asyncio.Task] = []
    for t in state.pending_background_tasks:
        if not t.done():
            still_pending.append(t)
            continue
        exc = t.exception()
        if exc:
            logger.warning("Background task failed: %s", exc)
            continue
        result: _VerifyResult = t.result()
        conversation.add_tool_result(
            tool_call_id=result.call_id,
            content=result.output,
        )
        if not result.must_fix_increment:
            state.has_edits_since_verify = False
        state.must_fix_injections = _maybe_inject_must_fix(
            conversation,
            result.must_fix_file,
            result.must_fix_text,
            state.must_fix_injections,
            metrics,
            state.must_fix_cap,
        )
        # Track verify failures for budget prompt injection
        if result.must_fix_increment:
            state.verify_failure_count += 1
            if (
                state.verify_failure_count >= state.budget_verify_cap
                and not state.budget_prompt_injected
            ):
                state.budget_prompt_injected = True
                logger.info(
                    "Verify-failure budget prompt injected after %d failures",
                    state.verify_failure_count,
                )
                conversation.add_user_message(
                    f"Verify has failed {state.verify_failure_count} times. "
                    "Consider whether the core problem is addressed; if so, "
                    "submit the current state as the result."
                )
    state.pending_background_tasks = still_pending


def _maybe_inject_must_fix(
    conversation: Conversation,
    file_path: str,
    verify_output: str,
    injections: int,
    metrics: AgentMetrics | None,
    effective_must_fix_cap: int,
) -> int:
    """Force the model to address unresolved lint errors after auto-verify.

    verify_with_retry returns a success ToolResult even when lint errors
    persist (fingerprint: verify.REMAINING_ERRORS_FINGERPRINT). Without
    this gate the model sees a success marker and can proceed to unrelated
    edits while quality silently degrades. On detection, inject a
    user-role message naming the file and errors so the constraint is
    in-conversation.

    Caps at effective_must_fix_cap injections per session. After the cap we log a
    warning and fail open — better to let the agent try a different tack
    than to deadlock on a fundamentally unfixable error (broken ruff
    config, upstream dep bug, etc.).

    When `metrics` is provided, each successful injection is recorded so
    downstream RL can shape rewards against agent efficiency.
    """
    from godspeed.tools.verify import REMAINING_ERRORS_FINGERPRINT

    if REMAINING_ERRORS_FINGERPRINT not in (verify_output or ""):
        return injections
    if injections >= effective_must_fix_cap:
        logger.warning(
            "MUST-FIX cap reached for file=%s; allowing agent to proceed",
            file_path,
        )
        return injections
    conversation.add_user_message(
        f"VERIFY FAILED on {file_path}. Unresolved lint errors remain "
        f"after auto-fix attempts:\n\n{verify_output}\n\n"
        "You MUST fix these errors before any other edits or writes."
    )
    logger.info(
        "MUST-FIX injected file=%s count=%d/%d",
        file_path,
        injections + 1,
        effective_must_fix_cap,
    )
    if metrics is not None:
        metrics.record_must_fix_injection()
    return injections + 1


async def _try_auto_commit(
    change_descriptions: list[str],
    tool_context: ToolContext,
    llm_client: LLMClient,
    conversation: Conversation,
    parent_call_id: str,
) -> bool:
    """Attempt an auto-commit with LLM-generated message. Returns True on success."""
    from godspeed.agent.auto_commit import auto_commit, generate_commit_message

    try:
        message = await generate_commit_message(change_descriptions, llm_client)
        result = await auto_commit(tool_context.cwd, message)
        if not result.is_error:
            logger.info("Auto-commit succeeded message=%s", message)
            conversation.add_tool_result(
                tool_call_id=f"{parent_call_id}_autocommit",
                content=f"[Auto-commit] {result.output}",
            )
            return True
        logger.warning("Auto-commit failed: %s", result.error)
    except Exception as exc:
        logger.warning("Auto-commit error: %s", exc)
    return False


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall | None:
    """Parse a raw tool call from the LLM response.

    Returns None if the tool call is malformed (invalid JSON arguments, etc.).
    Common tool-name hallucinations (``read_file``, ``grep``, ``glob``, etc.)
    are rewritten to their canonical names via
    ``godspeed.tools.aliases.canonicalize_tool_name`` so weak models don't
    dead-end on a correct intent expressed with the wrong label.
    """
    from godspeed.tools.aliases import canonicalize_tool_name

    try:
        func = raw.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        arguments = json.loads(args_str) if isinstance(args_str, str) else args_str

        if not name:
            return None

        return ToolCall(
            tool_name=canonicalize_tool_name(name),
            arguments=arguments,
            call_id=raw.get("id", ""),
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Malformed tool call: %s", raw)
        return None


def _acceptance_summary(tool_context: ToolContext | None) -> str | None:
    """Return a one-line summary of failing acceptance items, or None."""
    if tool_context is None:
        return None
    from godspeed.tools.acceptance import (
        ACCEPTANCE_DIRNAME,
        ACCEPTANCE_FILENAME,
        AcceptanceContract,
    )

    contract_path = tool_context.cwd / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME
    if not contract_path.exists():
        return None
    active = AcceptanceContract.load(contract_path).format_active()
    if active is None:
        return None
    return f"Acceptance criteria still failing:\n{active}"


def _tasks_open(task_store: Any | None, tool_context: ToolContext | None) -> bool:
    """Return True when open tasks or failing acceptance items remain."""
    if task_store is not None:
        if any(t.status != "completed" for t in task_store.list_all()):
            return True
    if tool_context is None:
        return False
    from godspeed.tools.acceptance import (
        ACCEPTANCE_DIRNAME,
        ACCEPTANCE_FILENAME,
        AcceptanceContract,
    )

    contract_path = tool_context.cwd / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME
    if not contract_path.exists():
        return False
    return bool(AcceptanceContract.load(contract_path).failing_items())


def _is_context_overflow(exc: Exception) -> bool:
    """Match provider context-overflow errors (the prompt_too_long family)."""
    if "contextwindowoverflow" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    markers = (
        "prompt_too_long",
        "prompt is too long",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "too many tokens",
        "input length exceeds",
        "reduce the length",
    )
    return any(marker in text for marker in markers)


async def _check_context_and_compact(
    conversation: Conversation,
    llm_client: LLMClient,
    hook_executor: Any | None,
    graduated_compactor: Any | None,
    loop_metrics: LoopMetrics,
) -> None:
    """Check context thresholds and apply graduated compaction if needed.

    Fires context threshold hooks and uses the GraduatedCompactor when
    available. Falls back to simple LLM compaction otherwise.
    """
    max_toks = conversation.max_tokens
    toks = conversation.token_count
    pct = toks / max_toks if max_toks > 0 else 0.0

    if hook_executor is not None:
        if pct >= 0.75:
            await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    hook_executor.fire,
                    HookEvent.CONTEXT_THRESHOLD_75,
                    token_count=toks,
                ),
            )
        elif pct >= 0.50:
            await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    hook_executor.fire,
                    HookEvent.CONTEXT_THRESHOLD_50,
                    token_count=toks,
                ),
            )
        elif pct >= 0.25:
            await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    hook_executor.fire,
                    HookEvent.CONTEXT_THRESHOLD_25,
                    token_count=toks,
                ),
            )

    if graduated_compactor is not None:
        results = graduated_compactor.apply_stages(conversation, toks, max_toks)
        for r in results:
            if r.applied:
                loop_metrics.record_compaction()
                if hook_executor is not None:
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        functools.partial(
                            hook_executor.fire,
                            HookEvent.POST_COMPACTION,
                            stage=r.stage_name,
                            tokens_before=r.tokens_before,
                            tokens_after=r.tokens_after,
                        ),
                    )

        # Emergency compaction (stage 5)
        if graduated_compactor.get_stage_for_context(toks, max_toks) >= 4:
            model = getattr(llm_client, "model", "")
            await graduated_compactor.emergency_compact(conversation, llm_client, model)
            loop_metrics.record_compaction()
        return

    # Fallback: simple LLM compaction
    if conversation.is_near_limit:
        await _compact_conversation(conversation, llm_client)
        loop_metrics.record_compaction()


async def _compact_conversation(conversation: Conversation, llm_client: LLMClient) -> None:
    """Compact conversation by summarizing history via a separate LLM call.

    Uses model-aware compaction prompts — small models get aggressive summarization,
    frontier models get detailed preservation. Picks the cheapest available model
    from the fallback chain to minimize compaction cost.
    """
    from godspeed.context.compaction import get_compaction_prompt
    from godspeed.llm.cost import get_cheapest_model

    model_name = getattr(llm_client, "model", "")
    logger.info("Compacting conversation tokens=%d model=%s", conversation.token_count, model_name)

    # Use cheapest model for compaction
    candidates = [model_name, *getattr(llm_client, "fallback_models", [])]
    cheapest = get_cheapest_model(candidates)
    if cheapest and cheapest != model_name:
        logger.info("Compaction using cheaper model=%s (instead of %s)", cheapest, model_name)

    prompt = get_compaction_prompt(cheapest or model_name)
    context = conversation.get_compaction_context()
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    try:
        response = await llm_client.chat(
            messages=summary_messages,
            task_type="compaction",
        )
        conversation.compact(response.content)
    except Exception as exc:
        logger.error("Compaction failed error=%s", exc, exc_info=True)
        # Don't crash — try truncation as fallback
        with contextlib.suppress(Exception):
            conversation.compact(f"[Compaction failed: {exc}. Retaining most recent context.]")


def _check_cancel(cancel_event: asyncio.Event | None) -> None:
    """Raise AgentCancelledError if the event has been set.

    Called at checkpoint boundaries inside the agent loop: top of
    iteration, between streaming chunks, before tool dispatch. Cheap
    (single atomic is_set() read) — safe to sprinkle liberally.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise AgentCancelledError("cancel_event set by caller")


async def _streaming_call(
    llm_client: LLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    on_chunk: Callable[[str], None],
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
    speculative_cache: dict[str, asyncio.Task[ToolResult]] | None = None,
    cancel_event: asyncio.Event | None = None,
    task_type: str | None = None,
    max_speculative_cache_size: int = MAX_SPECULATIVE_CACHE_SIZE,
) -> ChatResponse:
    """Make a streaming LLM call, invoking on_chunk for each text delta.

    When tool_registry and tool_context are provided, speculatively dispatches
    READ_ONLY tool calls as soon as the final response arrives — before the
    main loop processes them. Results are stored in speculative_cache so the
    main loop can await them instead of re-dispatching.

    When cancel_event is provided, the chunk loop checks it between each
    yielded chunk and raises AgentCancelledError — closing the underlying
    litellm stream promptly (its aclose() fires on generator cleanup).

    Returns the final complete ChatResponse for conversation history.
    """
    final_response: ChatResponse | None = None

    stream = llm_client.stream_chat(messages=messages, tools=tools, task_type=task_type)
    try:
        async for chunk in stream:
            if chunk.finish_reason is None and chunk.content:
                # Intermediate chunk — stream text to caller first, THEN
                # check cancel. This way the user sees the text the model
                # already produced before we unwind — a cleaner UX than
                # cutting off mid-word.
                on_chunk(chunk.content)
            elif chunk.finish_reason is not None:
                # Final aggregated response
                final_response = chunk

            # Cancel checkpoint: between chunks. If the caller (TUI signal
            # handler, headless SIGINT) set cancel_event during the last
            # chunk's on_chunk callback — or any time before now — we raise
            # AgentCancelledError here. The generator cleanup path in `finally`
            # closes the underlying HTTP stream.
            _check_cancel(cancel_event)
    finally:
        # Ensure the underlying async generator is closed on cancel OR on
        # any exception. aclose() is idempotent and cheap.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()

    if final_response is None:
        # Stream ended without a finish_reason — shouldn't happen but be safe
        return ChatResponse(content="", tool_calls=[], finish_reason="stop", usage={})

    # Speculative execution: start READ_ONLY tools immediately
    if (
        final_response.has_tool_calls
        and tool_registry is not None
        and tool_context is not None
        and speculative_cache is not None
    ):
        _speculative_dispatch(
            final_response.tool_calls,
            tool_registry,
            tool_context,
            speculative_cache,
            max_size=max_speculative_cache_size,
        )

    return final_response


def _speculative_dispatch(
    raw_tool_calls: list[dict[str, Any]],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    cache: dict[str, asyncio.Task[ToolResult]],
    max_size: int = MAX_SPECULATIVE_CACHE_SIZE,
) -> None:
    """Start READ_ONLY (and allowlisted) tool calls speculatively as background tasks.

    Parses each tool call and checks risk level. If READ_LOW or in the
    _SPECULATIVE_ALLOWLIST, dispatches immediately and stores the
    asyncio.Task in cache keyed by call_id. The main loop checks the cache
    before dispatching to avoid double work.

    Enforces *max_size* to prevent unbounded growth.
    """
    from godspeed.tools.base import RiskLevel

    for raw_tc in raw_tool_calls:
        # Enforce cache size limit to prevent memory growth
        if len(cache) >= max_size:
            logger.debug(
                "Speculative cache full (max=%d), skipping remaining dispatches",
                max_size,
            )
            break

        parsed = _parse_tool_call(raw_tc)
        if parsed is None:
            continue

        tool = tool_registry.get(parsed.tool_name)
        if tool is None:
            continue
        is_safe = (
            tool.risk_level == RiskLevel.READ_ONLY or parsed.tool_name in _SPECULATIVE_ALLOWLIST
        )
        if not is_safe:
            continue

        call_id = parsed.call_id
        if call_id and call_id not in cache:
            logger.debug("Speculative dispatch tool=%s call_id=%s", parsed.tool_name, call_id)
            task = asyncio.create_task(tool_registry.dispatch(parsed, tool_context))
            cache[call_id] = task
