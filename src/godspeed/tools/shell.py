"""Shell tool — run shell commands via subprocess."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_COMMAND_LENGTH = 10000  # 10K characters max for shell commands


class _ShellNotFoundError(Exception):
    """Raised when the shell executable cannot be found."""


class _ShellTimeoutError(Exception):
    """Raised when a shell command exceeds its timeout."""


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


def _detect_shell() -> list[str]:
    """Return the shell command prefix for the current platform (cached, thread-safe)."""
    global _shell_cache
    if _shell_cache is not None:
        return _shell_cache
    with _shell_lock:
        if _shell_cache is None:
            if platform.system() != "Windows":
                _shell_cache = ["/bin/bash", "-c"]
            else:
                _shell_cache = _detect_windows_shell()
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

        # Use Popen + communicate(timeout=...) instead of subprocess.run so
        # we can explicitly kill the process tree on timeout. subprocess.run's
        # timeout cleanup is unreliable on Windows when the child has holding
        # pipes (see _kill_process_tree docstring).
        def _run_sync() -> tuple[int, str, str]:
            """Run the command synchronously; called via run_in_executor."""
            proc: subprocess.Popen[str] | None = None
            try:
                proc = subprocess.Popen(
                    [*shell_prefix, command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(context.cwd),
                )
            except FileNotFoundError as exc:
                raise _ShellNotFoundError(exc) from exc

            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                return proc.returncode, stdout, stderr
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

        try:
            returncode, stdout, stderr = await asyncio.get_running_loop().run_in_executor(
                None, _run_sync
            )
        except _ShellNotFoundError as exc:
            return ToolResult.failure(f"Shell not found: {exc}")
        except _ShellTimeoutError as exc:
            return ToolResult.failure(
                f"Command timed out after {timeout}s and was force-killed "
                f"(including any child processes).{exc.args[0]}"
            )

        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

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
            f"Started background process {pid}\n"
            f"Command: {command}\n"
            f"Use background_check to poll status."
        )
