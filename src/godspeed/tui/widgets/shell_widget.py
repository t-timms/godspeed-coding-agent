"""Shell widget — persistent PTY shell with interactive terminal support.

Uses pywinpty (ConPTY) on Windows for real terminal I/O — vim, nano, etc.
Falls back to subprocess.Popen on missing pywinpty.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
from typing import ClassVar

from textual.widgets import RichLog

try:
    from pywinpty import PtyProcess

    _HAS_PTY = True
except ImportError:
    _HAS_PTY = False

logger = logging.getLogger(__name__)

_SHELL = os.environ.get("COMSPEC", "cmd.exe")


class ShellWidget(RichLog):
    """Interactive shell with persistent process state (cd, set, etc.).

    Uses pywinpty for full terminal support when available (vim, nano, htop).
    Falls back to subprocess.Popen for basic CLI tool support.
    """

    BINDINGS: ClassVar[list] = [
        ("escape", "focus_input", "Back to input"),
    ]

    def __init__(self, cwd: str | None = None, **kwargs):
        super().__init__(id="shell-log", highlight=False, markup=False, wrap=True, **kwargs)
        self._cwd = cwd or os.getcwd()
        self._proc: PtyProcess | subprocess.Popen | None = None
        self._reader_task: asyncio.Task | None = None
        self._running = False

    @property
    def _pty_available(self) -> bool:
        return _HAS_PTY

    def on_mount(self) -> None:
        self._start_shell()
        backend = "pty" if self._pty_available else "pipe"
        self.write(f"[dim]Shell [{backend}] — {self._cwd}[/dim]")

    def _start_shell(self) -> None:
        if self._running:
            return
        try:
            if self._pty_available:
                self._proc = PtyProcess.spawn(
                    [_SHELL],
                    cwd=self._cwd,
                    env=os.environ.copy(),
                    dimensions=(80, 24),
                )
            else:
                self._proc = subprocess.Popen(
                    [_SHELL],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self._cwd,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            self._running = True
            self._reader_task = asyncio.create_task(self._read_output())
        except Exception as exc:
            self.write(f"[#a45252]Failed to start shell: {exc}[/#a45252]")
            logger.exception("Shell startup failed")

    async def _read_output(self) -> None:
        if not self._proc:
            return
        loop = asyncio.get_event_loop()
        while self._running and self._proc.poll() is None:
            try:
                if self._pty_available and hasattr(self._proc, "read"):
                    data = await loop.run_in_executor(None, self._read_pty_chunk)
                    if data:
                        self.write(data.rstrip("\n\r"))
                elif self._proc.stdout:
                    line = await loop.run_in_executor(None, self._proc.stdout.readline)
                    if line:
                        self.write(line.rstrip("\n"))
                    else:
                        break
            except Exception:
                break

    def _read_pty_chunk(self) -> str:
        if not self._proc or not hasattr(self._proc, "read"):
            return ""
        try:
            return self._proc.read(4096)
        except Exception:
            return ""

    def send_command(self, command: str) -> None:
        if not self._running or self._proc is None:
            self._start_shell()
        if not self._proc:
            return
        try:
            self._proc.write(command + "\r\n")
        except Exception as exc:
            self.write(f"[#a45252]Error: {exc}[/#a45252]")

    def stop(self) -> None:
        self._running = False
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            with contextlib.suppress(Exception):
                self._proc.terminate()
