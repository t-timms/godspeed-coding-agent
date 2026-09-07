"""Tool protocol, base types, and risk classification."""

from __future__ import annotations

import abc
import functools
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from godspeed.sandbox.policy_types import SandboxPolicy


class RiskLevel(StrEnum):
    """4-tier risk classification for tools.

    Determines default permission behavior:
    - READ_ONLY: auto-allowed, no prompt
    - LOW: ask once, then session-scoped allow
    - HIGH: ask every time (unless pattern-matched)
    - DESTRUCTIVE: blocked by default, requires explicit enable
    """

    READ_ONLY = "read_only"
    LOW = "low"
    HIGH = "high"
    DESTRUCTIVE = "destructive"


class ToolResult(BaseModel):
    """Result returned from a tool execution."""

    output: str = ""
    error: str | None = None
    is_error: bool = False

    @classmethod
    def ok(cls, output: str) -> ToolResult:
        """Create a successful result."""
        return cls(output=output)

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        """Create an error result."""
        return cls(output="", error=error, is_error=True)

    # Keep 'success' as an alias for backwards compat with existing code
    success = ok


class ToolCall(BaseModel):
    """A request to execute a tool with specific arguments."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""

    @functools.cached_property
    def format_for_permission(self) -> str:
        """Cached: 'ToolName(arg_summary)' for permission rule matching.

        Prioritizes security-relevant keys over arbitrary first-string-value.
        For shell-like tools, always uses the 'command' key so that a benign
        'description' field cannot shadow the actual command being executed.
        """
        if isinstance(self.arguments, str):
            return f"{self.tool_name}({self.arguments})"
        if isinstance(self.arguments, dict):
            if not self.arguments:
                return f"{self.tool_name}()"
            if "command" in self.arguments:
                return f"{self.tool_name}({self.arguments['command']})"
            if "file_path" in self.arguments:
                return f"{self.tool_name}({self.arguments['file_path']})"
            if "action" in self.arguments:
                action = self.arguments["action"]
                return f"{self.tool_name}({action})"
            for value in self.arguments.values():
                if isinstance(value, str):
                    return f"{self.tool_name}({value})"
        return f"{self.tool_name}(*)"


@runtime_checkable
class PermissionEvaluator(Protocol):
    """Protocol for permission evaluation — avoids circular imports."""

    def evaluate(self, tool_call: ToolCall) -> Any:
        """Evaluate a tool call and return a permission decision."""
        pass


@runtime_checkable
class AuditRecorder(Protocol):
    """Protocol for audit recording — avoids circular imports."""

    def record(
        self,
        event_type: str,
        detail: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> Any:
        """Record an audit event. Returns the persisted record."""

    async def arecord(
        self,
        event_type: str,
        detail: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> Any:
        """Async variant of record."""
        pass


@runtime_checkable
class LLMInvoker(Protocol):
    """Protocol for tools that need to make LLM calls.

    Separated from the LLMClient concrete class so tools can accept any
    callable with an ``async chat(messages=...) -> response-with-content``
    surface. Avoids a hard dependency from tools/ on llm/.
    """

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:  # pragma: no cover
        """Send messages; return an object with a ``.content`` string attribute."""
        pass


@runtime_checkable
class DiffReviewer(Protocol):
    """Optional hook that lets a human approve / reject a concrete diff
    BEFORE a file is written.

    Permission (``PermissionEvaluator``) answers the question "is this
    tool allowed to run at all?" It fires once per tool invocation.

    ``DiffReviewer`` answers the question "should THIS specific change be
    applied?" It fires once per pending write, with the actual before /
    after content in hand. Two independent axes of consent.

    When ``ToolContext.diff_reviewer`` is ``None``, diff-producing tools
    write without review (headless / CI default). When present, the TUI
    (or a test double) implements the Protocol to prompt the user.
    """

    async def review(
        self,
        *,
        tool_name: str,
        path: str,
        before: str,
        after: str,
    ) -> str:
        """Return ``"accept"`` to apply the change or ``"reject"`` to skip it.

        Future return values (``"edit"``, etc) are reserved; implementations
        should treat anything other than ``"accept"`` as a reject for forward
        compatibility.
        """
        pass


WEB_CALL_SESSION_CAP = 200


class ToolContext(BaseModel):
    """Execution context passed to every tool."""

    cwd: Path
    session_id: str
    permissions: PermissionEvaluator | None = None
    audit: AuditRecorder | None = None
    llm_client: LLMInvoker | None = None
    diff_reviewer: DiffReviewer | None = None
    sandbox: SandboxPolicy | None = None
    web_call_count: int = 0
    lines_added: int = 0
    lines_removed: int = 0

    model_config = {"arbitrary_types_allowed": True}


class Tool(abc.ABC):
    """Abstract base class for all Godspeed tools.

    Every tool declares its name, description, risk level, and parameter schema.
    The same protocol is used for built-in tools, MCP tools, and future
    computer-use tools.
    """

    #: When True, the tool writes a file whose before/after content should be
    #: gated through ``ToolContext.diff_reviewer`` (if one is configured)
    #: before the write actually happens. Default False — read-only tools and
    #: shell-like tools opt out. File edit / write / diff-apply tools opt in.
    produces_diff: bool = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM system prompt."""
        pass

    @property
    @abc.abstractmethod
    def risk_level(self) -> RiskLevel:
        """Risk classification determining default permission behavior."""
        pass

    @property
    def concurrency_safe(self) -> bool:
        """Whether this tool may run concurrently with other safe tools.

        Drives loop batching: consecutive concurrency-safe calls execute in
        one gathered batch (bounded by a semaphore); the first non-safe call
        forces a serial flush. Defaults to READ_ONLY; override for tools
        whose reads are safe despite a higher declared risk level.
        """
        return self.risk_level == RiskLevel.READ_ONLY

    @abc.abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the tool's parameters.

        This schema is sent to the LLM as part of the tool definition.
        Must follow the JSON Schema spec used by LLM tool-calling APIs.
        """
        pass

    @abc.abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with validated arguments.

        Args:
            arguments: Validated arguments matching get_schema().
            context: Execution context (cwd, session, permissions, audit).

        Returns:
            ToolResult with output or error.
        """
        pass


def run_external_tool(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int = 60,
    max_output_chars: int = 5000,
    check_binary: str = "",
) -> ToolResult:
    """Run an external CLI tool and return a formatted ToolResult.

    Checks that the binary exists, runs the command with a timeout,
    truncates output, and returns success or failure.

    This is a shared helper used by complexity, coverage, dep_audit,
    and security_scan tools to avoid repeating the same boilerplate.
    """
    import shutil
    import subprocess

    if check_binary and not shutil.which(check_binary):
        return ToolResult.failure(f"{check_binary} is not installed. Install it to use this tool.")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except FileNotFoundError:
        return ToolResult.failure(f"Command not found: {cmd[0]}")

    output = result.stdout
    if result.stderr and result.returncode != 0:
        output += f"\n[stderr]\n{result.stderr}"

    if len(output) > max_output_chars:
        output = output[: max_output_chars - 3] + "..."

    if result.returncode != 0:
        return ToolResult.failure(output)
    return ToolResult.success(output)
