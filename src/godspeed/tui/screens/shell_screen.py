"""Shell screen — full-screen terminal with persistent state."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Input

from godspeed.tui.widgets.shell_widget import ShellWidget


class ShellScreen(Screen):
    """Full-screen shell with input area and persistent process."""

    BINDINGS: ClassVar[list] = [
        Binding("ctrl+r", "dismiss", "Chat"),
        Binding("escape", "focus_input", "Input", show=False),
    ]

    def __init__(self, cwd: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd

    def compose(self: Any) -> Any:
        yield ShellWidget(cwd=self._cwd)
        with Horizontal(id="input-area"):
            yield Input(placeholder="$ command...", id="shell-input")
        yield Footer()

    def on_mount(self: Any) -> None:
        self.query_one("#shell-input", Input).focus()

    def on_input_submitted(self: Any, message: Input.Submitted) -> None:
        cmd = message.value.strip()
        if not cmd:
            return
        self.query_one("#shell-input", Input).value = ""
        if cmd.lower() in ("exit", "quit"):
            self.action_dismiss()
            return
        shell = self.query_one("#shell-log", ShellWidget)
        shell.write(f"[bold]$ {cmd}[/bold]")
        shell.send_command(cmd)

    def action_dismiss(self: Any) -> None:
        shell = self.query_one("#shell-log", ShellWidget)
        shell.stop()
        self.app.pop_screen()
