"""Session list screen — interactive browser with previews and resume."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from godspeed.tui.theme import BOLD_PRIMARY, DIM, NEUTRAL, styled


class SessionListScreen(Screen):
    """Interactive session browser — preview, select, resume past sessions."""

    BINDINGS: ClassVar[list] = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("enter", "resume_session", "Resume"),
        Binding("r", "resume_session", "Resume"),
    ]

    def __init__(
        self,
        project_dir: Path,
        on_resume: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._project_dir = project_dir
        self._on_resume = on_resume

    def compose(self: Any) -> Any:
        yield Static(self._build_header(), id="sessions-header")
        yield ListView(id="sessions-list")
        yield Static(self._build_footer(), id="sessions-footer")

    def on_mount(self: Any) -> None:
        self._populate_list()
        if self.query_one("#sessions-list", ListView).children:
            self.query_one("#sessions-list", ListView).focus()

    def _build_header(self: Any) -> str:
        return (
            f"\n  {styled('Past Sessions', BOLD_PRIMARY)}\n"
            f"  {styled('─' * 40, NEUTRAL)}\n"
        )

    def _build_footer(self: Any) -> str:
        return (
            f"\n  {styled('Enter/r = resume | Escape/q = close', DIM)}"
        )

    def _populate_list(self: Any) -> None:
        sessions_dir = self._project_dir / ".godspeed" / "sessions"
        if not sessions_dir.exists():
            return

        session_files = sorted(sessions_dir.glob("*.jsonl"), reverse=True)[:30]
        if not session_files:
            return

        list_view = self.query_one("#sessions-list", ListView)
        for sf in session_files:
            try:
                lines = sf.read_text().splitlines()
                event_count = len(lines)
                mtime = sf.stat().st_mtime
                when = datetime.fromtimestamp(mtime, tz=UTC).strftime("%b %d %H:%M")

                # Extract model + summary from first/last events
                model = self._extract_model(lines)
                summary = self._extract_summary(lines)

                meta = f"{when}  {event_count} events"
                if model:
                    meta += f"  {model}"
                if summary:
                    meta += f"\n    {summary[:80]}"

                item = ListItem(
                    Label(f"  {styled(sf.stem[:12], BOLD_PRIMARY)}  {styled(meta, DIM)}")
                )
                list_view.append(item)
            except OSError:
                continue

    def _extract_model(self, lines: list[str]) -> str:
        import json

        if lines:
            try:
                data = json.loads(lines[0])
                return data.get("detail", {}).get("model", "")
            except (json.JSONDecodeError, KeyError):
                pass
        return ""

    def _extract_summary(self, lines: list[str]) -> str:
        import json

        if len(lines) > 1:
            try:
                data = json.loads(lines[-1])
                return data.get("detail", {}).get("exit_reason", "")
            except (json.JSONDecodeError, KeyError):
                pass
        return ""

    def action_resume(self: Any) -> None:
        list_view = self.query_one("#sessions-list", ListView)
        if list_view.index is not None and list_view.index < len(list_view.children):
            item = list_view.children[list_view.index]
            label = item.query_one(Label)
            text = str(label.renderable) if label.renderable else ""
            sid = text.strip().split(" ")[0].strip()
            if self._on_resume:
                self._on_resume(sid)
                self.app.pop_screen()

    def action_dismiss(self: Any) -> None:
        self.app.pop_screen()
