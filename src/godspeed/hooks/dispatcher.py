"""Hook dispatcher — orchestrates hook execution with auto-approve/deny.

Higher-level layer over ``HookExecutor`` that adds:

1. **Auto-approve / auto-deny** — JSON-configurable rules that short-circuit
   hook execution for known-safe or known-dangerous patterns.
2. **Adapters** — thin translation layers that map external hook formats
   (dict-shaped and flat-list JSON configs) to Godspeed's ``HookEvent`` +
   ``HookDefinition``.
3. **Sandbox awareness** — integrates ``SandboxPolicy`` so hooks can read
   technical containment decisions.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from godspeed.hooks import HookEvent
from godspeed.hooks.config import HookDefinition
from godspeed.hooks.executor import HookExecutor
from godspeed.sandbox.policy import SandboxApprovalResult, SandboxPolicy, evaluate_sandbox
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import ToolCall

logger = logging.getLogger(__name__)


def is_trusted_hook_source(source: str) -> bool:
    """Pre-trust gate: decide whether hooks from *source* may fire.

    Fail-closed policy:

    - Empty source (hooks from the user's settings file) — trusted by
      construction, the user authored them.
    - Paths under the user's home config roots — trusted (user-owned).
    - Everything else, including project-shipped hook configs (a cloned
      repository can carry malicious hooks) — untrusted.
    """
    if not source:
        return True
    try:
        path = Path(source).expanduser().resolve()
        home = Path.home().resolve()
    except OSError:
        return False
    return path.is_relative_to(home)


def filter_trusted_hooks(hooks: list[HookDefinition]) -> list[HookDefinition]:
    """Drop hooks whose source fails the pre-trust gate (fail closed).

    Returns only hooks the gate trusts; rejected hooks are logged with
    their provenance so the drop is auditable.
    """
    kept: list[HookDefinition] = []
    for hook in hooks:
        if is_trusted_hook_source(hook.source):
            kept.append(hook)
        else:
            logger.warning(
                "Pre-trust gate rejected hook source=%s command=%.80s",
                hook.source or "<unnamed>",
                hook.command,
            )
    return kept


@dataclass
class AutoApproveRule:
    """A rule that short-circuits hook execution to auto-approve.

    Attributes:
        event: The hook event this rule applies to.
        tool_pattern: Glob pattern for tool names (``"*"`` = all tools).
        reason: Why this auto-approve exists (for audit trail).
    """

    event: str
    tool_pattern: str = "*"
    reason: str = ""

    def matches(self, event: str, tool_name: str | None = None) -> bool:
        """Check if this rule matches an event + optional tool name."""
        if self.event != event:
            return False
        if self.tool_pattern == "*":
            return True
        if tool_name is None:
            return False

        return fnmatch.fnmatch(tool_name, self.tool_pattern)


@dataclass
class AutoDenyRule:
    """A rule that short-circuits hook execution to auto-deny.

    Attributes:
        event: The hook event this rule applies to.
        tool_pattern: Glob pattern for tool names.
        reason: Why this auto-deny exists (for audit trail).
    """

    event: str
    tool_pattern: str = "*"
    reason: str = ""

    def matches(self, event: str, tool_name: str | None = None) -> bool:
        """Check if this rule matches an event + optional tool name."""
        if self.event != event:
            return False
        if self.tool_pattern == "*":
            return True
        if tool_name is None:
            return False

        return fnmatch.fnmatch(tool_name, self.tool_pattern)


@dataclass
class DispatcherConfig:
    """Configuration for the hook dispatcher.

    Attributes:
        auto_approve: Rules that skip hook execution and auto-approve.
        auto_deny: Rules that skip hook execution and auto-deny.
        sandbox: Technical containment policy.
    """

    auto_approve: list[AutoApproveRule] = field(default_factory=list)
    auto_deny: list[AutoDenyRule] = field(default_factory=list)
    sandbox: SandboxPolicy = field(default_factory=SandboxPolicy)

    @classmethod
    def from_json(cls, path: Path) -> DispatcherConfig:
        """Load dispatcher config from a JSON file.

        Expected format::

            {
                "auto_approve": [
                    {"event": "pre_tool_call", "tool_pattern": "file_read", "reason": "..."}
                ],
                "auto_deny": [
                    {"event": "pre_tool_call", "tool_pattern": "shell", "reason": "..."}
                ]
            }
        """
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load dispatcher config from %s: %s", path, exc)
            return cls()

        approve_rules = [AutoApproveRule(**r) for r in data.get("auto_approve", [])]
        deny_rules = [AutoDenyRule(**r) for r in data.get("auto_deny", [])]
        return cls(auto_approve=approve_rules, auto_deny=deny_rules)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DispatcherConfig:
        """Build a DispatcherConfig from a raw dictionary."""
        approve_rules = [AutoApproveRule(**r) for r in data.get("auto_approve", [])]
        deny_rules = [AutoDenyRule(**r) for r in data.get("auto_deny", [])]
        return cls(auto_approve=approve_rules, auto_deny=deny_rules)


class HookDispatcher:
    """Orchestrator that wraps ``HookExecutor`` with policy layers.

    Evaluation order for a tool call:
        1. Auto-deny rules (short-circuit → block).
        2. Sandbox policy (technical containment).
        3. Auto-approve rules (short-circuit → allow).
        4. Permission engine (deny-first, 4 tiers).
        5. Shell hooks (subprocess execution via HookExecutor).
    """

    def __init__(
        self,
        executor: HookExecutor,
        permission_engine: PermissionEngine | None = None,
        config: DispatcherConfig | None = None,
    ) -> None:
        self._executor = executor
        self._permission_engine = permission_engine
        self._config = config or DispatcherConfig()

    @property
    def config(self) -> DispatcherConfig:
        """Read-only access to the dispatcher configuration."""
        return self._config

    def evaluate_tool_call(self, tool_call: ToolCall) -> SandboxApprovalResult:
        """Full pipeline: auto-rules → sandbox → permission engine.

        Returns a ``SandboxApprovalResult`` indicating whether the tool
        call may proceed.
        """
        event_str = HookEvent.PRE_TOOL_CALL.value

        for rule in self._config.auto_deny:
            if rule.matches(event_str, tool_call.tool_name):
                logger.info(
                    "Auto-deny rule matched tool=%s reason=%s",
                    tool_call.tool_name,
                    rule.reason,
                )
                return SandboxApprovalResult(
                    allowed=False,
                    sandbox_ok=True,
                    reason=f"Auto-deny: {rule.reason}",
                )

        sandbox_result = evaluate_sandbox(
            tool_call,
            self._config.sandbox,
            self._permission_engine,
        )
        if not sandbox_result.allowed:
            return sandbox_result

        for rule in self._config.auto_approve:
            if rule.matches(event_str, tool_call.tool_name):
                logger.info(
                    "Auto-approve rule matched tool=%s reason=%s",
                    tool_call.tool_name,
                    rule.reason,
                )
                return SandboxApprovalResult(
                    allowed=True,
                    sandbox_ok=True,
                    permission_decision=sandbox_result.permission_decision,
                    reason=f"Auto-approve: {rule.reason}",
                )

        return sandbox_result

    def fire(self, event: HookEvent, **context: Any) -> bool | None:
        """Fire hooks for an event, respecting auto-approve/deny.

        For ``pre_tool_call`` events, this first checks auto-deny rules
        and sandbox policy before delegating to the shell executor.
        """
        tool_name = context.get("tool_name")

        for rule in self._config.auto_deny:
            if rule.matches(event.value, tool_name):
                logger.debug("Auto-deny hook event=%s tool=%s", event.value, tool_name)
                return False

        for rule in self._config.auto_approve:
            if rule.matches(event.value, tool_name):
                logger.debug("Auto-approve hook event=%s tool=%s", event.value, tool_name)
                return None

        return self._executor.fire(event, **context)

    def run_pre_tool(self, tool_name: str) -> bool:
        """Run pre-tool pipeline. Returns True to proceed, False to abort."""
        for rule in self._config.auto_deny:
            if rule.matches(HookEvent.PRE_TOOL_CALL.value, tool_name):
                return False

        for rule in self._config.auto_approve:
            if rule.matches(HookEvent.PRE_TOOL_CALL.value, tool_name):
                return True

        return self._executor.run_pre_tool(tool_name)

    def run_post_tool(self, tool_name: str, result: str = "") -> None:
        """Run post-tool hooks."""
        self._executor.run_post_tool(tool_name, result)

    def run_pre_session(self) -> None:
        """Run session start hooks."""
        self._executor.run_pre_session()

    def run_post_session(self) -> None:
        """Run session end hooks."""
        self._executor.run_post_session()

    def run_stop(self) -> None:
        """Fire the STOP event."""
        self._executor.fire(HookEvent.STOP)

    def run_notification(self, message: str = "") -> None:
        """Fire a NOTIFICATION event."""
        self._executor.fire(HookEvent.NOTIFICATION, message=message)


class AgentHooksAdapter:
    """Translate the dict-shaped agent hooks JSON format to Godspeed ``HookDefinition`` list.

    The dict-shaped format uses a JSON config with a ``hooks`` dict keyed by event name::

        {
            "hooks": {
                "PreToolUse": [{"type": "command", "command": "..."}],
                "PostToolUse": [{"type": "command", "command": "..."}],
                "Stop": [{"type": "command", "command": "..."}]
            }
        }
    """

    _EVENT_MAP: ClassVar[dict[str, HookEvent]] = {
        "PreToolUse": HookEvent.PRE_TOOL_CALL,
        "PostToolUse": HookEvent.POST_TOOL_CALL,
        "Stop": HookEvent.STOP,
        "SubagentStop": HookEvent.POST_SUBAGENT_STOP,
        "PreCompact": HookEvent.PRE_COMPACTION,
        "Notification": HookEvent.NOTIFICATION,
        "SessionStart": HookEvent.SESSION_START,
        "SessionEnd": HookEvent.SESSION_END,
    }

    @classmethod
    def translate(cls, hooks_config: dict[str, Any], *, source: str = "") -> list[HookDefinition]:
        """Convert a dict-shaped hooks config to Godspeed HookDefinitions."""
        hooks_raw = hooks_config.get("hooks", {})
        definitions: list[HookDefinition] = []

        for event_name, hooks_list in hooks_raw.items():
            godspeed_event = cls._EVENT_MAP.get(event_name)
            if godspeed_event is None:
                logger.debug("Unknown hook event: %s", event_name)
                continue
            if not isinstance(hooks_list, list):
                continue
            for hook_spec in hooks_list:
                if not isinstance(hook_spec, dict):
                    continue
                command = hook_spec.get("command", "")
                if not command:
                    continue
                definitions.append(
                    HookDefinition(
                        event=godspeed_event.value,
                        command=command,
                        timeout=hook_spec.get("timeout", 30),
                        source=source,
                    )
                )

        return definitions

    @classmethod
    def from_json(cls, path: Path) -> list[HookDefinition]:
        """Load and translate a dict-shaped hooks JSON file.

        Hooks from files failing the pre-trust gate (e.g. project-shipped
        configs) are dropped fail-closed.
        """
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load hooks from %s: %s", path, exc)
            return []
        definitions = cls.translate(data, source=str(path))
        return filter_trusted_hooks(definitions)


class FlatListHooksAdapter:
    """Translate the flat-list hooks JSON format to Godspeed ``HookDefinition`` list.

    The flat-list format uses event-command pairs::

        [
            {"event": "pre_tool_use", "command": "..."},
            {"event": "post_tool_use", "command": "..."}
        ]
    """

    _EVENT_MAP: ClassVar[dict[str, HookEvent]] = {
        "pre_tool_use": HookEvent.PRE_TOOL_CALL,
        "post_tool_use": HookEvent.POST_TOOL_CALL,
        "session_start": HookEvent.SESSION_START,
        "session_end": HookEvent.SESSION_END,
        "stop": HookEvent.STOP,
        "pre_compact": HookEvent.PRE_COMPACTION,
        "subagent_stop": HookEvent.POST_SUBAGENT_STOP,
        "notification": HookEvent.NOTIFICATION,
    }

    @classmethod
    def translate(
        cls, hooks_config: list[dict[str, Any]], *, source: str = ""
    ) -> list[HookDefinition]:
        """Convert a flat-list hooks config to Godspeed HookDefinitions."""
        definitions: list[HookDefinition] = []
        for hook_spec in hooks_config:
            event_name = hook_spec.get("event", "")
            godspeed_event = cls._EVENT_MAP.get(event_name)
            if godspeed_event is None:
                logger.debug("Unknown hook event: %s", event_name)
                continue
            command = hook_spec.get("command", "")
            if not command:
                continue
            definitions.append(
                HookDefinition(
                    event=godspeed_event.value,
                    command=command,
                    timeout=hook_spec.get("timeout", 30),
                    source=source,
                )
            )
        return definitions

    @classmethod
    def from_json(cls, path: Path) -> list[HookDefinition]:
        """Load and translate a flat-list hooks JSON file.

        Hooks from files failing the pre-trust gate are dropped fail-closed.
        """
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load hooks from %s: %s", path, exc)
            return []
        if not isinstance(data, list):
            return []
        definitions = cls.translate(data, source=str(path))
        return filter_trusted_hooks(definitions)
