"""Permission dialog screen — approve/deny/always prompt for tool calls."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.screen import Screen
from textual.widgets import Static

from godspeed.tui.theme import BOLD_PRIMARY, BOLD_WARNING, DIM, styled


class PermissionDialog(Screen[str]):
    """Dialog for permission prompts — approve, deny, or always-allow."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("y", "approve", "Yes"),
        ("n", "deny", "No"),
        ("a", "always_allow", "Always"),
        ("escape", "deny", "Cancel"),
    ]

    def __init__(
        self,
        tool_name: str,
        reason: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._reason = reason
        self._arguments = arguments or {}

    def compose(self: Any) -> Any:
        lines = [
            f"  {styled('Permission required', BOLD_WARNING)}",
            "",
            f"    {styled(self._tool_name, BOLD_PRIMARY)}",
        ]
        args = self._arguments
        if args.get("file_path"):
            lines.append(f"    {args['file_path']}")
        if args.get("command"):
            lines.append(f"    $ {args['command']}")
        if args.get("pattern"):
            lines.append(f"    {styled(args['pattern'], DIM)}")
        lines.append("")
        lines.append(f"    {styled(self._reason, DIM)}")
        lines.append("")
        lines.append(f"  {styled('(y)es | (n)o | (a)lways | Escape to cancel', DIM)}")
        yield Static("\n".join(lines), id="permission-content")

    def action_approve(self) -> None:
        self.dismiss("yes")

    def action_deny(self) -> None:
        self.dismiss("no")

    def action_always_allow(self) -> None:
        self.dismiss("always")
