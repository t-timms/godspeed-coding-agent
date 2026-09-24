"""Shell tool — run shell commands via subprocess."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from godspeed.sandbox.kernel import (
    ExecutionPlan,
    KernelEnforcement,
    SandboxConstructionError,
    format_enforcement_header,
    plan_execution,
)
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_COMMAND_LENGTH = 10000  # 10K characters max for shell commands

# Cap on combined stdout/stderr returned to the model after a command
# completes. 8000 chars keeps a single command's output readable inside the
# context window while still surfacing the head of the output (where build
# errors and test summaries usually land). Mirrors test_runner's truncation
# spirit (5000 there); shell output is capped higher because commands like
# `pytest -v` legitimately produce more. The timeout path is exempt — it
# already tails the last 2000 chars of each stream.
MAX_OUTPUT_CHARS = 8000


class _ShellNotFoundError(Exception):
    """Raised when the shell executable cannot be found."""


class _ShellTimeoutError(Exception):
    """Raised when a shell command exceeds its timeout."""


class _SandboxUnavailableError(Exception):
    """Raised when kernel sandboxing was requested but cannot be enforced."""


class _SandboxSetupError(Exception):
    """Raised when kernel sandbox setup fails after process creation."""


def _truncate_output(output: str) -> str:
    """Cap combined command output at MAX_OUTPUT_CHARS, keeping the head.

    Mirrors test_runner's truncation: keep the first MAX_OUTPUT_CHARS chars
    and append a marker reporting how many chars were cut so the model knows
    output was truncated. Output at or under the cap passes through unchanged.
    """
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    truncated = len(output) - MAX_OUTPUT_CHARS
    return output[:MAX_OUTPUT_CHARS] + f"\n... ({truncated} chars truncated)"


def _kill_process_tree(pid: int) -> None:
    """Force-kill a process and all its descendants.

    Why this exists:
      ``subprocess.run(..., timeout=N)`` is documented to kill the child
      on TimeoutExpired, but on Windows (and sometimes on Linux with
      certain pipe configurations) that kill does NOT propagate to
      grandchildren. When the agent runs ``shell(command='python')`` the
      shell spawns git-bash which spawns an interactive Python — killing
      git-bash leaves Python holding stdout/stderr pipes, and
      subprocess.run blocks indefinitely waiting for them to close.

      Observed in SWE-Bench dev-23 attempt #3: instance sqlfluff-1517
      hung for ~100 minutes after a bare ``python`` REPL call despite
      the tool's 120s timeout. Instance sqlfluff-1733 hung ~60 min on a
      recursive ``sqlfluff fix``. Both required manual PID kill to
      unstick.

    This helper uses psutil's ``children(recursive=True)`` to walk the
    tree and issue kill() to each — which translates to
    ``TerminateProcess`` on Windows and SIGKILL on Unix. Cross-platform.

    Best-effort: if any process in the tree has already exited we skip
    it silently. Never raises to the caller.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available; cannot force-kill process tree for pid=%d", pid)
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            child.kill()
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        parent.kill()


_shell_cache: list[str] | None = None
_shell_lock = threading.Lock()

_WINDOWS_GIT_BASH_CANDIDATES: tuple[Path, ...] = (
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "Git"
    / "bin"
    / "bash.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "bin" / "bash.exe",
)


SHELL_WRAPPER_ENV = "GODSPEED_SHELL_WRAPPER"


def _detect_shell() -> list[str]:
    """Return the shell command prefix for the current platform (cached, thread-safe).

    On POSIX, ``$GODSPEED_SHELL_WRAPPER`` (an executable invoked as ``WRAPPER -c COMMAND``)
    replaces ``/bin/bash``. Benchmark runs use it to put every agent command in an isolated
    namespace (``scripts/agent_shell_isolate.sh``). It is read once and cached, so set it before
    the first command runs; the wrapper itself may read per-task variables on every call.
    """
    global _shell_cache
    if _shell_cache is not None:
        return _shell_cache
    with _shell_lock:
        if _shell_cache is None:
            wrapper = os.environ.get(SHELL_WRAPPER_ENV, "").strip()
            if platform.system() == "Windows":
                _shell_cache = _detect_windows_shell()
            elif wrapper:
                _shell_cache = [wrapper, "-c"]
            else:
                _shell_cache = ["/bin/bash", "-c"]
    return _shell_cache


def _detect_windows_shell() -> list[str]:
    """Pick the best available Windows shell prefix.

    Preference order:
    1. Git Bash from standard install locations (real bash, POSIX semantics).
    2. Git Bash found on PATH — but never the Microsoft Store WSL stub in
       ``WindowsApps``, which is broken when WSL is not installed and fails
       with ``REGDB_E_CLASSNOTREG`` + UTF-16 stderr.
    3. ``cmd.exe /c`` as the final fallback.
    """
    git_bash: str | None = None

    for candidate in _WINDOWS_GIT_BASH_CANDIDATES:
        if candidate.is_file():
            git_bash = str(candidate)
            break

    if git_bash is None:
        path_bash = shutil.which("bash")
        if path_bash and "windowsapps" not in path_bash.lower():
            git_bash = path_bash

    if git_bash:
        return [git_bash, "-c"]
    return ["cmd.exe", "/c"]


class ShellTool(Tool):
    """Run shell commands via subprocess.

    Each invocation is stateless — shell state (cwd changes, env vars) does not
    persist between calls. The working directory is always set from context.cwd.
    Cross-platform: uses bash on Unix, git-bash (or cmd fallback) on Windows.
    """

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Run a shell command and capture stdout/stderr. "
            "Each command runs independently (stateless). "
            "Use absolute paths or paths relative to the project root. "
            "Confirm with the user before destructive commands (rm, git push --force, etc.). "
            "Set background=true for long-running commands, then use "
            "background_check to poll status.\n\n"
            "Example: shell(command='pytest tests/ -v')\n"
            "Example: shell(command='pip install requests', timeout=60)\n"
            "Example: shell(command='npm run build', background=true)"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Timeout in seconds (default: {DEFAULT_TIMEOUT}, max: {MAX_TIMEOUT})"
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run in background and return immediately. "
                        "Use background_check tool to poll status."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = arguments.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.failure("command must be a non-empty string")

        # Check command length limit
        if len(command) > MAX_COMMAND_LENGTH:
            return ToolResult.failure(
                f"Command exceeds maximum length of {MAX_COMMAND_LENGTH} characters"
            )

        # Validate command against sandbox blocked paths
        from godspeed.sandbox.policy import validate_shell_command

        sandbox = context.sandbox
        if sandbox is not None:
            allowed, reason = validate_shell_command(command, sandbox)
            if not allowed:
                logger.warning("Shell command blocked by sandbox: %s", reason)
                return ToolResult.failure(f"Blocked by sandbox policy: {reason}")

        # Background execution
        if arguments.get("background", False):
            return await self._execute_background(command, context)

        raw_timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        if not isinstance(raw_timeout, int):
            try:
                raw_timeout = int(raw_timeout)
            except (TypeError, ValueError):
                return ToolResult.failure(
                    f"timeout must be an integer, got {type(raw_timeout).__name__}"
                )
        if raw_timeout <= 0:
            return ToolResult.failure("timeout must be positive")
        timeout = min(raw_timeout, MAX_TIMEOUT)

        shell_prefix = _detect_shell()
        logger.info("shell.execute command=%r timeout=%d", command, timeout)

        # Kernel sandbox planning — fail CLOSED when kernel sandboxing was
        # requested but cannot be enforced (never silently unsandboxed).
        plan: ExecutionPlan | None = None
        header = ""
        if sandbox is not None and sandbox.kernel_enforced:
            try:
                plan = plan_execution([*shell_prefix, command], cwd=context.cwd, policy=sandbox)
            except SandboxConstructionError as exc:
                logger.error("sandbox.construction-failed: %s", exc)
                return ToolResult.failure(f"Sandbox construction failed: {exc}")
            if plan.report.strategy != KernelEnforcement.JOB_OBJECT and not plan.report.enforced:
                details = "; ".join(plan.report.details)
                logger.error("sandbox.unavailable: %s", details)
                return ToolResult.failure(f"Sandbox unavailable: {details}")
            header = format_enforcement_header(plan.report)

        # Use Popen + communicate(timeout=...) instead of subprocess.run so
        # we can explicitly kill the process tree on timeout. subprocess.run's
        # timeout cleanup is unreliable on Windows when the child has holding
        # pipes (see _kill_process_tree docstring).
        def _run_sync() -> tuple[int, str, str, str]:
            """Run the command synchronously; called via run_in_executor."""
            effective_header = header
            proc: subprocess.Popen[str] | None = None
            try:
                popen_kwargs: dict[str, Any] = dict(
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(context.cwd),
                )
                if plan is not None:
                    if plan.preexec is not None:
                        popen_kwargs["preexec_fn"] = plan.preexec
                    if plan.env:
                        popen_kwargs["env"] = {**os.environ, **plan.env}
                    if plan.pass_fds:
                        popen_kwargs["pass_fds"] = plan.pass_fds
                    if plan.creationflags and sys.platform == "win32":
                        popen_kwargs["creationflags"] = plan.creationflags
                proc = subprocess.Popen(
                    plan.argv if plan is not None else [*shell_prefix, command],
                    **popen_kwargs,
                )
            except FileNotFoundError as exc:
                raise _ShellNotFoundError(exc) from exc

            if plan is not None and plan.post_start is not None:
                try:
                    updated = plan.post_start(proc)
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        proc.kill()
                    raise _SandboxSetupError(exc) from exc
                if updated is not None:
                    effective_header = format_enforcement_header(updated)

            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                return proc.returncode, stdout, stderr, effective_header
            except subprocess.TimeoutExpired:
                logger.warning(
                    "shell.timeout pid=%d command=%r timeout=%d - force-killing process tree",
                    proc.pid,
                    command,
                    timeout,
                )
                _kill_process_tree(proc.pid)
                # After killing the tree, drain any buffered output so the
                # underlying pipe FDs close and we don't leak them. Give it
                # a short window; if still blocked, move on with empty output.
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = "", ""
                tail = ""
                if stdout:
                    tail += f"\nSTDOUT tail:\n{stdout[-2000:]}"
                if stderr:
                    tail += f"\nSTDERR tail:\n{stderr[-2000:]}"
                raise _ShellTimeoutError(tail) from None
            finally:
                if proc is not None and proc.returncode is None:
                    with contextlib.suppress(Exception):
                        proc.kill()
                if plan is not None and plan.cleanup is not None:
                    with contextlib.suppress(Exception):
                        plan.cleanup()

        try:
            returncode, stdout, stderr, header = await asyncio.get_running_loop().run_in_executor(
                None, _run_sync
            )
        except _ShellNotFoundError as exc:
            return ToolResult.failure(f"Shell not found: {exc}")
        except _ShellTimeoutError as exc:
            return ToolResult.failure(
                f"Command timed out after {timeout}s and was force-killed "
                f"(including any child processes).{exc.args[0]}"
            )
        except _SandboxSetupError as exc:
            return ToolResult.failure(f"Sandbox setup failed: {exc}")

        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"
        output = _truncate_output(output)

        if header:
            output = f"{header}\n{output}"

        if returncode != 0:
            return ToolResult.failure(f"Exit code {returncode}\n{output}")

        return ToolResult.success(output)

    async def _execute_background(self, command: str, context: ToolContext) -> ToolResult:
        """Spawn a command in the background and return its process ID."""
        import time

        from godspeed.tools.background import (
            MAX_CONCURRENT,
            BackgroundProcess,
            BackgroundRegistry,
            _collect_output,
        )

        registry = BackgroundRegistry.get()

        if registry.active_count >= MAX_CONCURRENT:
            return ToolResult.failure(
                f"Too many background processes ({registry.active_count}/{MAX_CONCURRENT}). "
                "Kill some before starting new ones."
            )

        shell_prefix = _detect_shell()
        logger.info("shell.background command=%r", command)

        kernel_note = ""
        if context.sandbox is not None and context.sandbox.kernel_enforced:
            kernel_note = "sandbox: background execution NOT kernel-sandboxed (v1 limitation)\n"

        proc = await asyncio.create_subprocess_exec(
            *shell_prefix,
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(context.cwd),
        )

        pid = registry.next_id()
        bg_proc = BackgroundProcess(
            id=pid,
            command=command,
            process=proc,
            started_at=time.monotonic(),
        )
        # Start collecting output in background
        bg_proc._collection_task = asyncio.create_task(_collect_output(bg_proc))
        registry.add(bg_proc)

        return ToolResult.success(
            f"{kernel_note}Started background process {pid}\n"
            f"Command: {command}\n"
            f"Use background_check to poll status."
        )
