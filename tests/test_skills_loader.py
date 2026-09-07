"""Tests for skill loading budgets: 250-char description cap + 1% context budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.skills import loader as loader_mod
from godspeed.skills.loader import (
    SKILL_DESCRIPTION_MAX_CHARS,
    Skill,
    discover_skills,
    filter_context_budget,
)


@pytest.fixture(autouse=True)
def _isolated_skill_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep discovery away from the developer's real skill directories."""
    monkeypatch.setattr(loader_mod, "DEFAULT_SKILL_DIRS", [])
    monkeypatch.setattr(loader_mod, "_find_project_root", lambda: None)


def _write_skill(base: Path, name: str, description: str, body: str = "Do the thing.") -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


class TestDescriptionBudget:
    def test_long_description_truncated_to_cap(self, tmp_path: Path) -> None:
        long_desc = "x" * 400
        _write_skill(tmp_path, "long-desc", long_desc)
        skills = discover_skills(extra_dirs=[tmp_path])
        assert len(skills) == 1
        assert len(skills[0].description) == SKILL_DESCRIPTION_MAX_CHARS + 3  # cap + "..."
        assert skills[0].description.endswith("...")

    def test_short_description_untouched(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "short-desc", "Short and sweet")
        skills = discover_skills(extra_dirs=[tmp_path])
        assert skills[0].description == "Short and sweet"


class TestContextBudget:
    def _skill(self, name: str, body_chars: int) -> Skill:
        return Skill(
            name=name,
            description="d",
            trigger=name,
            content="c" * body_chars,
            path=Path(f"/tmp/{name}/SKILL.md"),
        )

    def test_oversized_skill_refused(self) -> None:
        # 1% of 200k tokens = 2000 tokens = 8000 chars
        skills = [self._skill("huge", 8001)]
        assert filter_context_budget(skills, context_window_tokens=200_000) == []

    def test_fitting_skill_kept(self) -> None:
        skills = [self._skill("fine", 8000)]
        kept = filter_context_budget(skills, context_window_tokens=200_000)
        assert [s.name for s in kept] == ["fine"]

    def test_mixed_set_filtered(self) -> None:
        skills = [self._skill("tiny", 100), self._skill("huge", 999_999)]
        kept = filter_context_budget(skills, context_window_tokens=200_000)
        assert [s.name for s in kept] == ["tiny"]
