"""Sandbox policy — bridges technical containment and approval decisions.

Two orthogonal questions:

1. **Sandbox**: What can the agent *technically* access?  Controlled by
   ``SandboxPolicy.writable_paths``, ``network_rules``, and Docker settings.
   This is a static configuration that doesn't change per tool call.

2. **Approval**: Is this specific tool call *allowed* right now?  Controlled
   by the existing ``PermissionEngine`` (deny-first, 4 tiers) with session
   grants, dangerous-command detection, and plan mode.

``SandboxPolicy.evaluate()`` composes both dimensions: a tool call must
pass the sandbox check AND the approval check to proceed.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from godspeed.sandbox.policy_types import NetworkRule, SandboxPolicy  # noqa: F401
from godspeed.security.permissions import ALLOW, PermissionDecision, PermissionEngine
from godspeed.tools.base import ToolCall

logger = logging.getLogger(__name__)

_SECRET_ENV_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9_]*"
    r"(?:API_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIALS?|ACCESS_KEY|PRIVATE_KEY)"
    r"|OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SESSION_TOKEN|GITHUB_TOKEN|HF_TOKEN)\b"
)


@dataclass
class SandboxApprovalResult:
    """Result of a combined sandbox + approval evaluation.

    Attributes:
        allowed: Whether the tool call may proceed.
        sandbox_ok: Whether the sandbox check passed.
        permission_decision: The PermissionEngine decision (if evaluated).
        reason: Human-readable explanation.
    """

    allowed: bool
    sandbox_ok: bool
    permission_decision: PermissionDecision | None = None
    reason: str = ""


_URL_RE = re.compile(r"https?://([^:/\s]+)(?::(\d+))?")

_PATH_LIKE_RE = re.compile(r"(?:^|\s)([/\\]?[\w./\\-]+\.\w+)")


def _extract_network_target(tool_call: ToolCall) -> tuple[str | None, int | None]:
    """Best-effort extraction of hostname:port from tool arguments."""
    args = tool_call.arguments
    if not isinstance(args, dict):
        return None, None

    for key in ("url", "host", "hostname", "endpoint"):
        val = args.get(key)
        if isinstance(val, str):
            match = _URL_RE.search(val)
            if match:
                hostname = match.group(1)
                port = int(match.group(2)) if match.group(2) else None
                return hostname, port
            return val, None
    return None, None


_WRITE_TOOLS = frozenset({"file_write", "file_edit", "file_move", "diff_apply", "git"})
_PATH_KEYS = ("file_path", "path", "source", "destination")
_GIT_WRITE_ACTIONS = frozenset({"commit", "undo", "stash", "stash_pop"})


def evaluate_sandbox(
    tool_call: ToolCall,
    sandbox: SandboxPolicy,
    permission_engine: PermissionEngine | None = None,
) -> SandboxApprovalResult:
    """Evaluate a tool call against both sandbox policy and permission engine.

    The evaluation order is:
    1. Sandbox check (path writability / network access) — fast, static.
    2. Permission engine check (deny-first, 4 tiers) — per-call, dynamic.

    Both must pass for the tool call to proceed.
    """
    sandbox_ok = True
    sandbox_reason = ""

    args = tool_call.arguments
    if isinstance(args, dict):
        for key in _PATH_KEYS:
            val = args.get(key)
            if not isinstance(val, str):
                continue
            if tool_call.tool_name in _WRITE_TOOLS:
                if not sandbox.is_path_writable(val):
                    sandbox_ok = False
                    sandbox_reason = f"Path not writable: {val}"
                    break
            elif not sandbox.is_path_readable(val):
                sandbox_ok = False
                sandbox_reason = f"Path not readable: {val}"
                break

        if sandbox_ok and tool_call.tool_name == "git":
            action = args.get("action", "")
            if action in _GIT_WRITE_ACTIONS:
                cwd = str(Path(args.get("cwd", ".")).resolve())
                if not sandbox.is_path_writable(cwd):
                    sandbox_ok = False
                    sandbox_reason = f"Git write action blocked — cwd not writable: {cwd}"

    # Evaluate network rules whenever rules exist (non-empty) or network
    # is globally disabled.
    if sandbox_ok and (sandbox.network_rules or not sandbox.enable_network):
        hostname, port = _extract_network_target(tool_call)
        if hostname and not sandbox.is_network_allowed(hostname, port):
            sandbox_ok = False
            sandbox_reason = f"Network access denied: {hostname}:{port or '*'}"

    if not sandbox_ok:
        return SandboxApprovalResult(
            allowed=False,
            sandbox_ok=False,
            reason=sandbox_reason,
        )

    if permission_engine is None:
        return SandboxApprovalResult(allowed=True, sandbox_ok=True, reason="No permission engine")

    decision = permission_engine.evaluate(tool_call)
    allowed = decision.action == ALLOW

    return SandboxApprovalResult(
        allowed=allowed,
        sandbox_ok=True,
        permission_decision=decision,
        reason=decision.reason,
    )


def _expand_token(token: str) -> str:
    """Expand env vars and ~ then resolve to an absolute path."""
    expanded = os.path.expandvars(token)
    expanded = os.path.expanduser(expanded)
    return str(Path(expanded).resolve())


def validate_shell_command(command: str, sandbox: SandboxPolicy) -> tuple[bool, str]:
    """Check a shell command against sandbox blocked paths.

    Uses ``shlex.split`` to tokenize, expands each token (env vars, ``~``,
    relative paths), and compares resolved paths against resolved blocked
    paths.  Substring matches inside a command word are *not* sufficient —
    the resolved token must be equal to or under a blocked path.

    Returns ``(allowed, reason)``.
    """
    if not sandbox.blocked_paths and not _SECRET_ENV_RE.search(command):
        return True, ""

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    resolved_blocked = [str(Path(bp).expanduser().resolve()) for bp in sandbox.blocked_paths]
    resolved_norm = [SandboxPolicy._norm(rb) for rb in resolved_blocked]

    for token in tokens:
        resolved = _expand_token(token)
        rn = SandboxPolicy._norm(resolved)
        for i in range(len(resolved_blocked)):
            target = resolved_norm[i]
            if rn == target or rn.startswith(target + os.sep) or rn.startswith(target + "/"):
                return False, f"Command references blocked path: {token} -> {resolved}"

    secret_env = _SECRET_ENV_RE.search(command)
    if secret_env:
        return False, f"Secret environment variable referenced: {secret_env.group(0)}"

    return True, ""


def build_sandbox_policy(
    blocked_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
) -> SandboxPolicy:
    """Construct a SandboxPolicy from configuration values.

    When no blocked_paths are provided, ships secure defaults that deny
    access to sensitive system locations regardless of platform.
    """
    defaults: list[str] = []
    if blocked_paths is None:
        defaults = [
            "/etc/shadow",
            "/etc/passwd",
            "/proc",
            "/sys",
            "/dev",
            "~/.ssh",
            "~/.gnupg",
            "~/.aws/credentials",
            "~/.config/gcloud",
            "~/.kube/config",
            "~/.docker/config.json",
            ".env",
            "credentials.json",
            ".netrc",
        ]
    return SandboxPolicy(
        blocked_paths=blocked_paths if blocked_paths is not None else defaults,
        writable_paths=writable_paths or [],
    )
