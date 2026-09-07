"""Hook executor — runs shell commands at agent lifecycle events."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from godspeed.hooks import HookEvent
from godspeed.hooks.config import HookDefinition

logger = logging.getLogger(__name__)

_SHELL_META = set("|<>;&")


def _needs_shell(command: str) -> bool:
    """Check if a command contains shell metacharacters that require shell=True."""
    return any(ch in command for ch in _SHELL_META)


_SHELL_UNSAFE_RE = re.compile(r"[;|<>&`\$!#*?{}()\[\]\\\"' \t\n]")


def _sanitize_template_var(value: str) -> str:
    """Escape shell-unsafe characters in a template variable value.

    Prevents injection via user-controlled values (tool_name, paths, etc.)
    being interpolated into hook commands.  Only characters that could be
    interpreted by a shell are escaped; cosmetic characters are preserved.
    """
    if not value:
        return value
    return _SHELL_UNSAFE_RE.sub(lambda m: "\\" + m.group(0), value)


class HookExecutor:
    """Execute hooks at agent lifecycle events.

    Pre-tool hooks can block tool execution by returning a non-zero exit code.
    All hooks run synchronously in a subprocess with a timeout.

    Args:
        hooks: List of hook definitions to execute.
        cwd: Working directory for subprocess execution.
        session_id: Current session ID for template expansion.
    """

    def __init__(
        self,
        hooks: list[HookDefinition],
        cwd: Path,
        session_id: str,
    ) -> None:
        self._hooks = hooks
        self._cwd = cwd
        self._session_id = session_id

    def run_pre_tool(self, tool_name: str) -> bool:
        """Run pre_tool_call hooks. Returns True to proceed, False to abort."""
        return self.fire(HookEvent.PRE_TOOL_CALL, tool_name=tool_name) is not False

    def run_post_tool(self, tool_name: str, result: str = "") -> None:
        """Run post_tool_call hooks."""
        self.fire(HookEvent.POST_TOOL_CALL, tool_name=tool_name)

    def run_pre_session(self) -> None:
        """Run pre_session hooks."""
        self.fire(HookEvent.SESSION_START)

    def run_post_session(self) -> None:
        """Run post_session hooks."""
        self.fire(HookEvent.SESSION_END)

    def fire(
        self,
        event: HookEvent,
        **context: Any,
    ) -> bool | None:
        """Fire hooks for a specific event.

        Args:
            event: The HookEvent to fire.
            **context: Additional context variables to pass to hook commands.

        Returns:
            - None: No hooks fired or all hooks succeeded (advisory).
            - False: A pre_* hook returned non-zero (blocks execution).
            - True: Should not happen (hooks are advisory except pre_*).
        """
        hooks_for_event = [h for h in self._hooks if h.event == event.value]
        if not hooks_for_event:
            return None

        base_context = {
            "gs_event": event.value,
            "gs_session_id": self._session_id,
            "gs_timestamp": datetime.now(UTC).isoformat(),
            **context,
        }

        blocked = False
        for hook in hooks_for_event:
            tool_name = base_context.get("tool_name")
            if tool_name is not None and not self._matches_tool(hook, tool_name):
                continue

            exit_code = self._execute(hook, base_context)
            if exit_code != 0:
                logger.warning(
                    "Hook blocked event=%s command=%s exit=%d",
                    event.value,
                    hook.command,
                    exit_code,
                )
                if event.value.startswith("pre_"):
                    blocked = True

        return False if blocked else None

    def _matches_tool(self, hook: HookDefinition, tool_name: str) -> bool:
        """Check if a hook applies to the given tool."""
        return hook.tools is None or tool_name in hook.tools

    def _execute(self, hook: HookDefinition, context: dict[str, Any]) -> int:
        """Execute a hook command with template variable expansion.

        Returns the exit code (0 = success).
        """
        raw_vars: dict[str, str] = {
            "session_id": self._session_id,
            "cwd": str(self._cwd),
            "project_dir": str(self._cwd),
            **{k: str(v) for k, v in context.items()},
        }
        template_vars = {k: _sanitize_template_var(v) for k, v in raw_vars.items()}

        try:
            command = hook.command.format(**template_vars)
        except KeyError as exc:
            logger.warning(
                "Hook template error command=%s missing_var=%s",
                hook.command,
                exc,
            )
            return 1

        logger.debug(
            "Executing hook event=%s command=%s timeout=%d",
            hook.event,
            command,
            hook.timeout,
        )

        use_shell = _needs_shell(command)
        try:
            if use_shell:
                result = subprocess.run(  # noqa: S602 # nosec B602 - trusted hook config, template vars sanitized
                    command,
                    shell=True,
                    cwd=self._cwd,
                    timeout=hook.timeout,
                    capture_output=True,
                    text=True,
                )
            else:
                cmd_args = shlex.split(command)
                result = subprocess.run(
                    cmd_args,
                    shell=False,
                    cwd=self._cwd,
                    timeout=hook.timeout,
                    capture_output=True,
                    text=True,
                )
            if result.stdout:
                logger.debug("Hook stdout: %s", result.stdout.strip())
            if result.stderr:
                logger.debug("Hook stderr: %s", result.stderr.strip())
            return result.returncode
        except ValueError as exc:
            logger.warning(
                "Hook command parse error command=%s error=%s",
                command,
                exc,
            )
            return 1
        except subprocess.TimeoutExpired:
            logger.warning(
                "Hook timed out event=%s command=%s timeout=%ds",
                hook.event,
                command,
                hook.timeout,
            )
            return 1
        except OSError as exc:
            logger.warning(
                "Hook execution failed event=%s command=%s error=%s",
                hook.event,
                command,
                exc,
            )
            return 1
