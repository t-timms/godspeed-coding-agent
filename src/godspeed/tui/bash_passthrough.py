"""Bash pass-through for the Godspeed TUI.

Input starting with ``!`` runs the rest as a shell command directly,
bypassing the LLM. ``!!`` runs in background (fire and forget).

Security: commands are checked against the existing dangerous-command
detection (``godspeed.security.dangerous``) and routed through the same
permission-relevant check function used by the shell tool. The audit
trail hook is wired when an ``AuditTrail`` is available.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from godspeed.security.dangerous import detect_dangerous_command

logger = logging.getLogger(__name__)

#: Prefix for foreground shell pass-through.
FOREGROUND_PREFIX = "!"

#: Prefix for background shell pass-through (fire and forget).
BACKGROUND_PREFIX = "!!"


@dataclass(frozen=True)
class BashCommand:
    """A parsed bash pass-through command."""

    command: str
    background: bool = False


def parse_bash_command(raw_input: str) -> BashCommand | None:
    """Parse a ``!`` / ``!!`` prefixed input into a BashCommand.

    Returns ``None`` when the input is not a bash pass-through (does not
    start with ``!``). Raises ``ValueError`` for an empty command after
    the prefix (e.g. ``!`` or ``!!`` with nothing after).
    """
    if not raw_input.startswith("!"):
        return None

    if raw_input.startswith("!!"):
        command = raw_input[2:].strip()
        if not command:
            msg = "empty command after '!!'"
            raise ValueError(msg)
        return BashCommand(command=command, background=True)

    command = raw_input[1:].strip()
    if not command:
        msg = "empty command after '!'"
        raise ValueError(msg)
    return BashCommand(command=command, background=False)


def check_dangerous(command: str) -> list[str]:
    """Return danger descriptions for a command, or [] if safe."""
    return detect_dangerous_command(command)


def _detect_shell() -> list[str]:
    """Return the shell command prefix for the current platform."""
    if platform.system() != "Windows":
        return ["/bin/bash", "-c"]
    # Prefer git-bash on Windows, fall back to cmd.exe
    for candidate in (
        Path(shutil.which("bash") or ""),
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ):
        if candidate.is_file():
            return [str(candidate), "-c"]
    return ["cmd.exe", "/c"]


async def run_foreground(
    command: str,
    cwd: Path,
    *,
    timeout: int = 120,
    on_output: Any | None = None,
) -> tuple[int, str]:
    """Run a shell command in the foreground, streaming output.

    Args:
        command: The shell command to execute.
        cwd: Working directory.
        timeout: Timeout in seconds.
        on_output: Optional async callback ``(text: str) -> None`` invoked
            with each chunk of stdout/stderr as it arrives.

    Returns:
        ``(returncode, combined_output)``.
    """
    shell_prefix = _detect_shell()
    proc = await asyncio.create_subprocess_exec(
        *shell_prefix,
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )

    output_parts: list[str] = []

    async def _read_stream(stream: Any) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.readline()
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            output_parts.append(text)
            if on_output is not None:
                await on_output(text)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _read_stream(proc.stdout),
                _read_stream(proc.stderr),
            ),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        output_parts.append(f"\n[Command timed out after {timeout}s and was killed]")
        return 124, "".join(output_parts)

    returncode = await proc.wait()
    return returncode, "".join(output_parts)


async def run_background(command: str, cwd: Path) -> int:
    """Run a shell command in the background (fire and forget).

    Returns the process ID.
    """
    shell_prefix = _detect_shell()
    proc = await asyncio.create_subprocess_exec(
        *shell_prefix,
        command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(cwd),
    )
    logger.info("Background bash pass-through pid=%d command=%r", proc.pid, command)
    return proc.pid
