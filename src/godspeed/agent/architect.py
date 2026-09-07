"""Architect mode — two-phase plan-then-execute pipeline."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.llm.client import LLMClient
from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

ARCHITECT_SYSTEM_PROMPT = (
    "You are a code architect. Analyze the request and produce a detailed, "
    "step-by-step implementation plan. Use read-only tools (file_read, glob_search, "
    "grep_search, repo_map) to explore the codebase. "
    "Do NOT write any code or modify any files. "
    "Output a structured plan with numbered steps."
)


async def architect_loop(
    user_input: str,
    conversation: Conversation,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    architect_model: str = "",
    on_phase_change: Any = None,  # Callable[[str, str], None] — (phase, model)
    **kwargs: Any,
) -> str:
    """Run architect mode: plan phase -> execute phase.

    Phase 1 (Plan): Calls architect_model (or main model) with read-only tools
    and a planning system prompt. Produces a plan.

    Phase 2 (Execute): Injects the plan as context, calls main model with full
    tools to implement.
    """
    # Phase 1: Plan
    plan_model = architect_model or llm_client.model
    if on_phase_change:
        on_phase_change("plan", plan_model)

    # Create a filtered registry with only READ_ONLY tools
    read_only_registry = _filter_read_only(tool_registry)

    # Create a separate conversation for planning
    plan_conversation = Conversation(
        system_prompt=ARCHITECT_SYSTEM_PROMPT,
        model=plan_model,
        max_tokens=conversation.max_tokens,
    )

    # derive(), not with_model(): the swap is unsafe across awaits and
    # mid-session model changes invalidate the provider prompt cache.
    plan_client = llm_client.derive(plan_model)
    plan = await agent_loop(
        user_input=user_input,
        conversation=plan_conversation,
        llm_client=plan_client,
        tool_registry=read_only_registry,
        tool_context=tool_context,
        parallel_tool_calls=True,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "on_assistant_text",
                "on_tool_call",
                "on_tool_result",
                "on_assistant_chunk",
                "on_thinking",
                "max_iterations",
            }
        },
    )
    llm_client.adopt(plan_client)

    if not plan or plan.startswith("Error:"):
        return plan or "Error: Architect planning phase produced no output."

    # Phase 2: Execute
    if on_phase_change:
        on_phase_change("execute", llm_client.model)

    # Inject plan as context for the execution phase
    plan_context = (
        f"An architect has analyzed the request and produced this implementation plan:\n\n"
        f"---\n{plan}\n---\n\n"
        f"Original request: {user_input}\n\n"
        f"Execute this plan. Follow the steps precisely. You have full tool access."
    )

    result = await agent_loop(
        user_input=plan_context,
        conversation=conversation,
        llm_client=llm_client,
        tool_registry=tool_registry,
        tool_context=tool_context,
        parallel_tool_calls=True,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "on_assistant_text",
                "on_tool_call",
                "on_tool_result",
                "on_assistant_chunk",
                "on_thinking",
                "max_iterations",
                "on_permission_denied",
                "hook_executor",
                "auto_fix_retries",
                "auto_commit",
                "auto_commit_threshold",
                "on_parallel_start",
                "on_parallel_complete",
            }
        },
    )

    return result


def _filter_read_only(registry: ToolRegistry) -> ToolRegistry:
    """Create a new registry containing only READ_ONLY tools."""
    filtered = ToolRegistry()
    for tool in registry.list_tools():
        if tool.risk_level == RiskLevel.READ_ONLY:
            filtered.register(tool)
    return filtered
