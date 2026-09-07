"""Hook configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from godspeed.hooks import HookEvent


class HookDefinition(BaseModel):
    """A single hook that runs a shell command at a lifecycle event.

    Events (25 total):
        Session: session_start, session_end
        Permission: pre_permission_check, post_permission_check, permission_denied
        Tool: pre_tool_call, post_tool_call
        Context: pre_compaction, post_compaction, context_threshold_75,
            context_threshold_50, context_threshold_25
        Subagent: pre_subagent_spawn, post_subagent_complete, post_subagent_stop,
            subagent_error
        Safety: secret_detected, dangerous_command, stuck_loop_detected,
            budget_exceeded
        Workflow: workflow_phase_complete, workflow_complete, workflow_rejected
        Misc: stop, notification

    Template variables in ``command``:
        {tool_name}: Name of the tool being called (tool events only).
        {session_id}: Current session ID.
        {cwd}: Working directory.
        {project_dir}: Project directory (same as cwd).
        {gs_event}: Event name (GS_EVENT).
        {gs_path}: File path (GS_PATH).
        {gs_cost_usd}: Running cost in USD (GS_COST_USD).
        {gs_timestamp}: ISO 8601 timestamp (GS_TIMESTAMP).
    """

    event: HookEvent
    command: str
    tools: list[str] | None = Field(
        default=None,
        description="Tool names to match. None = all tools.",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Max seconds for hook execution.",
    )
    source: str = Field(
        default="",
        description=(
            "Provenance of this hook: empty = user settings (trusted); "
            "otherwise the config file path it was loaded from."
        ),
    )
