"""Godspeed TUI — terminal user interface built with Textual and Rich.

The TUI layer comprises:
- textual_app.py  — Textual App subclass (primary interface)
- output.py       — Rich formatting utilities (markdown, diffs, panels, status)
- theme.py        — Earth-tone color palette and brand identity
- commands.py     — Slash command registry, fuzzy dispatch, and aliases
- mentions.py     — @-mention parsing and resolution (file, folder, web)
- completions.py  — Tab completion for slash commands and file paths
- screens/        — Textual Screen subclasses (chat, help, session list)
- widgets/        — Custom Textual Widget subclasses (chat view, prompt, status bar)
"""

from __future__ import annotations
