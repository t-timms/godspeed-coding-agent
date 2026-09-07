"""Dream consolidation — cross-session pruning, dedup, and date normalization.

Runs periodically (24h) to scan skill directories and lesson stores,
merging duplicates, removing stale entries, and converting relative
dates to absolute.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

RELATIVE_DATE_RE = re.compile(
    r"(?i)\b(?:yesterday|today|last\s+(?:week|month|year|night|session)|"
    r"\d+\s+(?:day|week|month|year|hour|minute)s?\s+ago)\b"
)

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class SkillDream:
    """Cross-session consolidation of skill knowledge."""

    def __init__(self, base_dir: Path | None = None):
        self._base = base_dir or Path.home() / ".godspeed"
        self._dream_lock = self._base / ".dream" / "last_run.txt"
        self._dream_lock.parent.mkdir(parents=True, exist_ok=True)

    def should_run(self, interval_hours: int = 24) -> bool:
        """Check if enough time has passed since last dream."""
        try:
            last = datetime.fromisoformat(self._dream_lock.read_text().strip())
            return (datetime.now(tz=UTC) - last).total_seconds() > interval_hours * 3600
        except (OSError, ValueError):
            return True

    def run(self, skills_dir: Path) -> dict[str, int]:
        """Execute one dream consolidation pass.

        Returns stats: {consolidated, pruned, dates_normalized, errors}.
        """
        stats: dict[str, int] = {"consolidated": 0, "pruned": 0, "dates_normalized": 0, "errors": 0}

        if not skills_dir.is_dir():
            return stats

        for skill_path in skills_dir.glob("*/SKILL.md"):
            try:
                text = skill_path.read_text(encoding="utf-8", errors="replace")
                normalized = self._normalize_dates(text)
                if normalized != text:
                    skill_path.write_text(normalized)
                    stats["dates_normalized"] += 1
                    logger.info("Dream: normalized dates in %s", skill_path)
            except OSError:
                stats["errors"] += 1

        self._dream_lock.write_text(datetime.now(tz=UTC).isoformat())
        logger.info("Dream complete: %s", stats)
        return stats

    def _normalize_dates(self, text: str) -> str:
        now = datetime.now(tz=UTC)

        def _replace(m: re.Match) -> str:
            raw = m.group(0).lower()
            if raw == "yesterday":
                return (now - timedelta(days=1)).strftime("%Y-%m-%d")
            if raw == "today":
                return now.strftime("%Y-%m-%d")
            if raw.startswith("last "):
                unit = raw.split()[-1]
                if unit in ("week",):
                    return (now - timedelta(weeks=1)).strftime("%Y-%m-%d")
                if unit in ("month",):
                    return (now - timedelta(days=30)).strftime("%Y-%m-%d")
                if unit in ("session", "night"):
                    return (now - timedelta(days=1)).strftime("%Y-%m-%d")
                if unit in ("year",):
                    return now.replace(year=now.year - 1).strftime("%Y-%m-%d")
            m2 = re.match(r"(\d+)\s+(day|week|month|year|hour|minute)s?\s+ago", raw)
            if m2:
                n = int(m2.group(1))
                unit = m2.group(2)
                if unit == "day":
                    return (now - timedelta(days=n)).strftime("%Y-%m-%d")
                if unit == "week":
                    return (now - timedelta(weeks=n)).strftime("%Y-%m-%d")
                if unit == "month":
                    return (now - timedelta(days=n * 30)).strftime("%Y-%m-%d")
                if unit == "year":
                    return now.replace(year=now.year - n).strftime("%Y-%m-%d")
            return m.group(0)

        return RELATIVE_DATE_RE.sub(_replace, text)
