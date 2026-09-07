"""Hook system — run shell commands at agent lifecycle events."""

from __future__ import annotations

from enum import StrEnum


class HookEvent(StrEnum):
    """Lifecycle events that hooks can subscribe to.

    Events are grouped by category:
    - Session: session lifecycle
    - Permission: permission engine decisions
    - Tool: tool execution lifecycle
    - Context: context management and compaction
    - Subagent: sub-agent spawn and completion
    - Safety: security and safety events
    - Workflow: orchestrated workflow phases
    - Misc: stop, notification
    """

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Permission lifecycle
    PRE_PERMISSION_CHECK = "pre_permission_check"
    POST_PERMISSION_CHECK = "post_permission_check"
    PERMISSION_DENIED = "permission_denied"

    # Tool lifecycle
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"

    # Context management
    PRE_COMPACTION = "pre_compaction"
    POST_COMPACTION = "post_compaction"
    CONTEXT_THRESHOLD_75 = "context_threshold_75"
    CONTEXT_THRESHOLD_50 = "context_threshold_50"
    CONTEXT_THRESHOLD_25 = "context_threshold_25"

    # Subagent lifecycle
    PRE_SUBAGENT_SPAWN = "pre_subagent_spawn"
    POST_SUBAGENT_COMPLETE = "post_subagent_complete"
    POST_SUBAGENT_STOP = "post_subagent_stop"
    SUBAGENT_ERROR = "subagent_error"

    # Safety events
    SECRET_DETECTED = "secret_detected"  # noqa: S105
    DANGEROUS_COMMAND = "dangerous_command"
    STUCK_LOOP_DETECTED = "stuck_loop_detected"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Workflow
    WORKFLOW_PHASE_COMPLETE = "workflow_phase_complete"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_REJECTED = "workflow_rejected"

    # Misc
    STOP = "stop"
    NOTIFICATION = "notification"
