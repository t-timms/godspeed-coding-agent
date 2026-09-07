"""Tool registry — discovery, schema generation, validation, and dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from godspeed.tools.base import RiskLevel, Tool, ToolCall, ToolContext, ToolResult

if TYPE_CHECKING:
    from godspeed.sandbox.policy import SandboxPolicy

logger = logging.getLogger(__name__)

# Transient errors that warrant automatic retry.
# Network timeouts, connection resets, and temporary resource exhaustion
# are recoverable; logic errors and permission denials are not.
_TRANSIENT_ERROR_PATTERNS = (
    "timeout",
    "connection refused",
    "connection reset",
    "connection aborted",
    "temporary failure",
    "resource temporarily unavailable",
    "rate limit",
    "too many requests",
    "service unavailable",
    "502",
    "503",
    "504",
)

_SANDBOX_LOW_RISK_TOOLS = frozenset({"file_write", "file_edit", "file_move", "git", "diff_apply"})


def _validate_tool_arguments(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Validate tool arguments against the tool's JSON Schema.

    Returns an error message if validation fails, or None if valid.
    """
    schema = tool.get_schema()
    if not schema:
        return None

    # Check required fields
    required = schema.get("required", [])
    for field_name in required:
        if field_name not in arguments:
            return f"Missing required argument: '{field_name}'"

    # Check property types if defined
    properties = schema.get("properties", {})
    for field_name, value in arguments.items():
        if field_name in properties:
            expected_type = properties[field_name].get("type")
            if expected_type:
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "array": (list, tuple),
                    "object": dict,
                }
                expected = type_map.get(expected_type)
                if expected and not isinstance(value, expected):  # type: ignore[arg-type]
                    return (
                        f"Argument '{field_name}' expected type '{expected_type}', "
                        f"got '{type(value).__name__}'"
                    )

    return None


def _is_transient_error(error_msg: str) -> bool:
    """Check if an error message indicates a transient, retryable failure."""
    error_lower = error_msg.lower()
    return any(pattern in error_lower for pattern in _TRANSIENT_ERROR_PATTERNS)


class ToolRegistry:
    """Central registry for all available tools.

    Handles tool registration, schema generation for LLM APIs,
    argument validation, and dispatching tool calls with automatic
    retry on transient failures.
    """

    def __init__(self, max_retries: int = 2, sandbox: SandboxPolicy | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._description_overrides: dict[str, str] = {}  # tool_name -> override
        self._schema_cache: list[dict[str, Any]] | None = None
        self._max_retries = max_retries  # retries beyond the initial attempt
        self._sandbox = sandbox

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if tool.name in self._tools:
            msg = f"Tool '{tool.name}' is already registered"
            raise ValueError(msg)
        self._tools[tool.name] = tool
        self._schema_cache = None  # Invalidate cache
        logger.debug("Registered tool: %s (risk=%s)", tool.name, tool.risk_level)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def update_description(self, tool_name: str, description: str) -> bool:
        """Set a runtime description override for a tool.

        The override is used in get_schemas() instead of the tool's built-in
        description. Used by the self-evolution system to hot-swap descriptions.

        Returns:
            True if the tool exists and the override was set.
        """
        if tool_name not in self._tools:
            return False
        self._description_overrides[tool_name] = description
        self._schema_cache = None  # Invalidate cache
        logger.debug("Description override set tool=%s len=%d", tool_name, len(description))
        return True

    def clear_description_override(self, tool_name: str) -> None:
        """Remove a description override, reverting to the built-in description."""
        if self._description_overrides.pop(tool_name, None) is not None:
            self._schema_cache = None  # Invalidate cache

    def get_description(self, tool_name: str) -> str | None:
        """Get the effective description for a tool (override or built-in)."""
        if tool_name in self._description_overrides:
            return self._description_overrides[tool_name]
        tool = self._tools.get(tool_name)
        return tool.description if tool else None

    def get_schemas(self) -> list[dict[str, Any]]:
        """Generate tool schemas in the format expected by LLM APIs.

        Returns a list of tool definitions compatible with OpenAI/Anthropic
        function calling format (LiteLLM normalizes this). Uses description
        overrides from the self-evolution system when available.

        Results are cached until a tool is registered or a description
        override changes.
        """
        if self._schema_cache is not None:
            return self._schema_cache

        schemas = []
        for tool in self._tools.values():
            description = self._description_overrides.get(tool.name, tool.description)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": tool.get_schema(),
                    },
                }
            )
        self._schema_cache = schemas
        return schemas

    async def dispatch(self, tool_call: ToolCall, context: ToolContext) -> ToolResult:
        """Dispatch a tool call to the correct tool implementation.

        Validates arguments against the tool's JSON Schema before execution.
        Retries on transient failures (network timeouts, connection resets,
        rate limits) with exponential backoff.

        Args:
            tool_call: The tool call to execute.
            context: Execution context.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            return ToolResult.failure(
                f"Unknown tool: '{tool_call.tool_name}'. "
                f"Available: {', '.join(sorted(self._tools.keys()))}"
            )

        # Schema validation before execution
        validation_error = _validate_tool_arguments(tool, tool_call.arguments)
        if validation_error:
            logger.warning(
                "Schema validation failed tool=%s error=%s",
                tool_call.tool_name,
                validation_error,
            )
            return ToolResult.failure(
                f"Invalid arguments for '{tool_call.tool_name}': {validation_error}"
            )

        if self._sandbox is not None and (
            tool.risk_level in (RiskLevel.HIGH, RiskLevel.DESTRUCTIVE)
            or (tool.risk_level == RiskLevel.LOW and tool.name in _SANDBOX_LOW_RISK_TOOLS)
        ):
            from godspeed.sandbox.policy import evaluate_sandbox

            sandbox_result = evaluate_sandbox(tool_call, self._sandbox)
            if not sandbox_result.sandbox_ok:
                return ToolResult.failure(f"Sandbox denied: {sandbox_result.reason}")

        # Execute with retry on transient failures
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await tool.execute(tool_call.arguments, context)
            except Exception as exc:
                last_error = exc
                error_msg = str(exc)
                if not _is_transient_error(error_msg) or attempt >= self._max_retries:
                    break
                # Exponential backoff: 0.1s, 0.2s, 0.4s, ...
                import asyncio

                delay = 0.1 * (2**attempt)
                logger.info(
                    "Transient error on attempt %d for tool=%s, retrying in %.1fs: %s",
                    attempt + 1,
                    tool_call.tool_name,
                    delay,
                    error_msg,
                )
                await asyncio.sleep(delay)

        logger.error(
            "Tool execution failed after %d attempts tool=%s error=%s",
            self._max_retries + 1,
            tool_call.tool_name,
            last_error,
            exc_info=True,
        )
        return ToolResult.failure(f"Tool '{tool_call.tool_name}' failed: {last_error}")
