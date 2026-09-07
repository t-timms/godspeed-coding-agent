"""Output style suffixes for the agent system prompt.

Styles append a short instruction block to the system prompt that shapes
how the agent presents its work. Built-ins ship with the product; custom
styles live in ``{project}/.godspeed/styles/*.md`` and are validated like
agent definitions (lowercase, no spaces, <= 32 chars). Malformed files
are skipped with a warning — a bad style never crashes the loader.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

STYLE_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
STYLE_NAME_MAX_CHARS = 32

# Built-in styles: name -> system-prompt suffix. ``default`` has no
# suffix (``None``); the others append an instructing paragraph.
BUILT_IN_STYLES: dict[str, str | None] = {
    "default": None,
    "explanatory": (
        "## Output Style: Explanatory\n"
        "Explain your choices as you work. For each substantive change, "
        "briefly state what you changed, why, and the alternative you "
        "considered. Keep each explanation to one or two sentences so it "
        "never crowds out the work itself."
    ),
    "learning": (
        "## Output Style: Learning\n"
        "Add a short learning hint after each substantive change that "
        "teaches the underlying pattern, then end with one reflective "
        "question about the change — for example, what trade-off was made "
        "or what would break if done differently. Keep hints brief and "
        "specific to the change."
    ),
}


def load_custom_styles(project_dir: Path) -> dict[str, str]:
    """Load custom output styles from ``{project_dir}/.godspeed/styles/*.md``.

    The style name is the filename stem, validated like agent definitions
    (lowercase, no spaces, <= 32 chars). Malformed or unreadable files are
    skipped with a warning. Returns a fresh dict on every call — no
    caching, no shared mutable state.
    """
    styles: dict[str, str] = {}
    styles_dir = Path(project_dir) / ".godspeed" / "styles"
    if not styles_dir.is_dir():
        return styles
    for path in sorted(styles_dir.glob("*.md")):
        name = path.stem
        if not STYLE_NAME_RE.match(name):
            logger.warning("Style file %s has invalid name %r — skipping", path, name)
            continue
        if len(name) > STYLE_NAME_MAX_CHARS:
            logger.warning(
                "Style file %s name %r exceeds %d chars — skipping",
                path,
                name,
                STYLE_NAME_MAX_CHARS,
            )
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Cannot read style file %s: %s", path, exc)
            continue
        if not text:
            logger.warning("Style file %s is empty — skipping", path)
            continue
        styles[name] = text
    return styles


def resolve_style(name: str, project_dir: Path) -> str | None:
    """Resolve a style name to its system-prompt suffix.

    Built-ins win over custom styles with the same name. Returns ``None``
    for the ``default`` style (no suffix) and for unknown names.
    """
    if name in BUILT_IN_STYLES:
        return BUILT_IN_STYLES[name]
    return load_custom_styles(project_dir).get(name)
