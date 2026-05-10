"""Splash screen — shown during boot with loading progress."""

from __future__ import annotations

import logging
from typing import Any

from textual.screen import Screen
from textual.widgets import Static

from godspeed.tui.theme import BOLD_PRIMARY, DIM, styled

logger = logging.getLogger(__name__)


class SplashScreen(Screen):
    """Loading screen displayed while backend initializes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._status: str = "Starting..."

    def compose(self: Any) -> Any:
        from godspeed import __version__

        lines = [
            "",
            f"  {styled('>', BOLD_PRIMARY)} {styled('godspeed', BOLD_PRIMARY)}"
            f"  {styled(f'v{__version__}', DIM)}",
            "",
            f"  {styled(self._status, DIM)}",
        ]
        yield Static("\n".join(lines), id="splash-content")

    def update_status(self, text: str) -> None:
        self._status = text
        try:
            widget = self.query_one("#splash-content", Static)
            from godspeed import __version__

            lines = [
                "",
                f"  {styled('>', BOLD_PRIMARY)} {styled('godspeed', BOLD_PRIMARY)}"
                f"  {styled(f'v{__version__}', DIM)}",
                "",
                f"  {styled(self._status, DIM)}",
            ]
            widget.update("\n".join(lines))
        except Exception:
            logger.debug("Could not update splash status", exc_info=True)
