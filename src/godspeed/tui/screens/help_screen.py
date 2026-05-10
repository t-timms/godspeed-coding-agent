"""Help screen — full keybinding reference and command listing."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.screen import Screen
from textual.widgets import Static

from godspeed.tui.theme import BOLD_PRIMARY, DIM, NEUTRAL, styled


class HelpScreen(Screen):
    """Full-screen help overlay showing all keybindings and slash commands."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, commands: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._commands = commands

    def compose(self: Any) -> Any:
        yield Static(self._build_help_text(), id="help-content")

    def action_dismiss(self: Any) -> None:
        self.app.pop_screen()

    def _build_help_text(self: Any) -> str:
        rule = styled("-" * 30, NEUTRAL)
        lines: list[str] = []

        lines.append(f"  {styled('Godspeed Help', BOLD_PRIMARY)}")
        lines.append(f"  {rule}")
        lines.append("")

        groups: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Keyboard Shortcuts",
                [
                    ("Enter", "Submit prompt"),
                    ("Escape+Enter", "Insert newline (multiline)"),
                    ("Ctrl+C", "Cancel agent (press twice for hard stop)"),
                    ("Ctrl+P", "Command palette / slash commands"),
                    ("Ctrl+S", "Session browser"),
                    ("Escape", "Focus prompt input"),
                    ("F1", "This help screen"),
                ],
            ),
            (
                "Session Commands",
                [
                    ("/model [name]", "Show or switch model"),
                    ("/models", "List installed Ollama models"),
                    ("/clear /c", "Clear conversation history"),
                    ("/stats /s", "Show token usage and cost"),
                    ("/export /e", "Export conversation as markdown"),
                    ("/quit /q", "Exit Godspeed"),
                ],
            ),
            (
                "Agent Control",
                [
                    ("/plan /p", "Toggle plan mode (read-only)"),
                    ("/extend /x [N]", "Set max iterations per turn"),
                    ("/architect", "Toggle architect mode"),
                    ("/think /t [budget]", "Toggle extended thinking"),
                    ("/budget /b [amount]", "Show/set cost budget"),
                    ("/pause", "Pause agent loop"),
                    ("/resume", "Resume paused agent"),
                    ("/guidance <msg>", "Inject guidance and resume"),
                ],
            ),
            (
                "Context",
                [
                    ("/context /ctx", "Show context window usage"),
                    ("/checkpoint /cp [name]", "Save/list checkpoints"),
                    ("/restore /rs <name>", "Restore a checkpoint"),
                    ("/tasks", "Show task list"),
                    ("/reindex", "Rebuild codebase index"),
                ],
            ),
            (
                "Security",
                [
                    ("/audit /a", "Show audit trail and verify"),
                    ("/permissions", "Show permission rules"),
                    ("/remember <act> <pat>", "Persist permission rule"),
                    ("/undo /u", "Undo last git commit"),
                ],
            ),
        ]

        for group_name, cmds in groups:
            lines.append(f"  {styled(group_name, NEUTRAL)}")
            lines.append(f"  {styled('-' * 30, NEUTRAL)}")
            for cmd_name, desc in cmds:
                lines.append(f"    {styled(cmd_name, BOLD_PRIMARY):32s} {styled(desc, DIM)}")
            lines.append("")

        lines.append(f"  {styled('Press Escape or q to close', DIM)}")
        return "\n".join(lines)
