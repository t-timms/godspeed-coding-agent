"""Full SKILL.md standard — discovery, parsing, progressive disclosure, hub management."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

SKILL_DESCRIPTION_MAX_CHARS = 250
SKILL_CONTEXT_BUDGET_FRACTION = 0.01
_CHARS_PER_TOKEN = 4

DEFAULT_SKILL_DIRS: list[Path] = [
    Path.home() / ".config" / "opencode" / "skills",
    Path.home() / ".claude" / "skills",
    Path.home() / ".agents" / "skills",
    Path.home() / ".godspeed" / "skills",
]

PROJECT_SKILL_DIRS: list[Path] = [
    Path(".opencode") / "skills",
    Path(".claude") / "skills",
    Path(".agents") / "skills",
    Path(".godspeed") / "skills",
]


class SkillError(Exception):
    """Base for skill system errors."""


class SkillSecurityError(SkillError):
    """Skill failed security scan."""


@dataclass
class SkillFiles:
    """Bundled supporting files for a skill."""

    references: list[Path] = field(default_factory=list)
    scripts: list[Path] = field(default_factory=list)
    assets: list[Path] = field(default_factory=list)


@dataclass
class Skill:
    """A skill following the open Agent Skills standard.

    Progressive disclosure tiers:
      Tier 0 (always loaded):  name, description
      Tier 1 (on activation):   full SKILL.md body + references/
      Tier 2 (on demand):       scripts/, assets/
    """

    name: str
    description: str
    trigger: str
    content: str
    path: Path
    frontmatter: dict[str, Any] = field(default_factory=dict)
    files: SkillFiles = field(default_factory=SkillFiles)
    version: str = ""
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    hash: str = ""

    def __post_init__(self):
        if not SKILL_NAME_RE.match(self.name):
            msg = f"Skill name {self.name!r} does not match required pattern"
            raise SkillError(msg)


def _compute_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _find_project_root() -> Path | None:
    """Walk up from cwd to find git worktree root."""
    cwd = Path.cwd().resolve()
    for parent in [cwd, *list(cwd.parents)]:
        if (parent / ".git").exists():
            return parent
    return None


def _skill_dirs() -> list[Path]:
    """Full ordered list of skill discovery paths (global then project, later wins)."""
    dirs: list[Path] = []

    for d in DEFAULT_SKILL_DIRS:
        dirs.append(d)

    root = _find_project_root()
    if root:
        for d in PROJECT_SKILL_DIRS:
            dirs.append(root / d)

    return dirs


def _check_skill_path(path: Path) -> bool:
    """A valid skill lives in ``{name}/SKILL.md`` ."""
    return path.name == "SKILL.md" and path.parent.parent != path.parent


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str] | None:
    """Extract YAML frontmatter and body from a SKILL.md string.

    Returns ``(frontmatter_dict, body_string)`` or ``None`` on failure.
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return None

    end = stripped.find("---", 3)
    if end == -1:
        return None

    fm_str = stripped[3:end].strip()
    body = stripped[end + 3 :].strip()

    try:
        fm = yaml.safe_load(fm_str)
    except yaml.YAMLError:
        return None

    if not isinstance(fm, dict):
        return None

    return fm, body


def _load_skill_directory(skill_dir: Path) -> Skill | None:
    """Load a skill from a ``{name}/SKILL.md`` directory.

    Scans for ``references/``, ``scripts/``, ``assets/`` subdirs.
    """
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.is_file():
        return None

    try:
        text = skill_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", skill_path, exc)
        return None

    parsed = _parse_frontmatter(text)
    if parsed is None:
        return None

    fm, body = parsed
    name = fm.get("name", skill_dir.name)
    description = fm.get("description", "")

    if not name or not description:
        logger.warning("Skill %s missing name or description in frontmatter", skill_path)
        return None

    if not SKILL_NAME_RE.match(name):
        logger.warning("Skill %s has invalid name %r", skill_path, name)
        return None

    if len(description) > SKILL_DESCRIPTION_MAX_CHARS:
        logger.warning(
            "Skill %s description %d chars exceeds %d — truncating",
            skill_path,
            len(description),
            SKILL_DESCRIPTION_MAX_CHARS,
        )
        description = description[:SKILL_DESCRIPTION_MAX_CHARS].rstrip() + "..."

    if skill_dir.name != name:
        logger.warning("Skill dir %s != name %s — using dir name", skill_dir.name, name)

    trigger = fm.get("trigger", name)
    license_val = fm.get("license", "")
    compatibility = fm.get("compatibility", "")
    metadata = fm.get("metadata", {})

    refs: list[Path] = []
    scripts: list[Path] = []
    assets: list[Path] = []

    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        refs = sorted(ref_dir.iterdir())

    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        scripts = sorted(scripts_dir.iterdir())

    assets_dir = skill_dir / "assets"
    if assets_dir.is_dir():
        assets = sorted(assets_dir.iterdir())

    return Skill(
        name=name,
        description=description,
        trigger=trigger,
        content=body,
        path=skill_path,
        frontmatter=fm,
        files=SkillFiles(references=refs, scripts=scripts, assets=assets),
        version=fm.get("version", ""),
        license=license_val,
        compatibility=compatibility,
        metadata=metadata or {},
        created_at=_stat_ctime(skill_path),
        updated_at=_stat_mtime(skill_path),
        hash=_compute_hash(text),
    )


def _stat_ctime(path: Path) -> datetime | None:
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_ctime, tz=UTC)
    except OSError:
        return None


def _stat_mtime(path: Path) -> datetime | None:
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime, tz=UTC)
    except OSError:
        return None


def discover_skills(extra_dirs: list[Path] | None = None) -> list[Skill]:
    """Discover skills from standard + extra directories.

    Later directories override earlier ones on ``trigger`` match.
    Project skills beat global skills. Extra dirs take highest priority.
    """
    dirs = _skill_dirs()
    if extra_dirs:
        dirs.extend(extra_dirs)

    seen: dict[str, Skill] = {}
    for base in dirs:
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            skill = _load_skill_directory(child)
            if skill is not None:
                tr = skill.trigger
                if tr in seen:
                    logger.info("Skill trigger %r overridden by %s", tr, child)
                seen[tr] = skill

    result = list(seen.values())
    logger.info("Discovered %d skills from %d directories", len(result), len(dirs))
    return result


def filter_context_budget(
    skills: list[Skill],
    context_window_tokens: int = 200_000,
) -> list[Skill]:
    """Drop skills whose body would exceed the per-skill context budget.

    A skill's full SKILL.md body enters the context on activation; a body
    larger than ``SKILL_CONTEXT_BUDGET_FRACTION`` (1%) of the context window
    would dominate the conversation the moment it fires. Oversized skills are
    refused at discovery (fail closed) rather than truncated mid-content,
    which would produce misleading partial instructions.
    """
    max_chars = int(context_window_tokens * SKILL_CONTEXT_BUDGET_FRACTION) * _CHARS_PER_TOKEN
    kept: list[Skill] = []
    for skill in skills:
        if len(skill.content) > max_chars:
            logger.warning(
                "Skill %s body %d chars exceeds 1%% context budget (%d) — refusing",
                skill.name,
                len(skill.content),
                max_chars,
            )
            continue
        kept.append(skill)
    return kept


class SkillHub:
    """Marketplace-like install/update/remove for skills with provenance tracking.

    Stores a ``.godspeed/skills/.hub/lock.json`` lock file so every
    installed skill is pinned by content hash.
    """

    def __init__(self, base_dir: Path | None = None):
        self._base = base_dir or Path.home() / ".godspeed" / "skills"
        self._hub_dir = self._base / ".hub"
        self._lock_file = self._hub_dir / "lock.json"
        self._hub_dir.mkdir(parents=True, exist_ok=True)
        self._lock: dict[str, Any] = self._load_lock()

    def _load_lock(self) -> dict[str, Any]:
        try:
            return json.loads(self._lock_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {"skills": {}, "installed_at": None}

    def _save_lock(self) -> None:
        self._lock["installed_at"] = datetime.now(tz=UTC).isoformat()
        self._lock_file.write_text(json.dumps(self._lock, indent=2))

    def install(self, name: str, source_path: Path) -> Skill:
        """Install a skill from ``source_path`` (a directory containing SKILL.md).

        Copies the directory to ``~/.godspeed/skills/{name}/``, scans for
        security issues, and records provenance in the lock file.
        """
        target = self._base / name
        if target.exists():
            msg = f"Skill {name!r} already installed at {target}"
            raise SkillError(msg)

        skill = _load_skill_directory(source_path)
        if skill is None:
            msg = f"Invalid skill in {source_path}"
            raise SkillError(msg)

        from godspeed.skills.security import scan_skill

        issues = scan_skill(source_path)
        if issues:
            details = "; ".join(issues[:5])
            if len(issues) > 5:
                details += f" (+{len(issues) - 5} more)"
            msg = f"Skill {name!r} failed security scan: {details}"
            raise SkillSecurityError(msg)

        shutil.copytree(source_path, target)

        self._lock["skills"][name] = {
            "source": str(source_path),
            "hash": skill.hash,
            "installed_at": datetime.now(tz=UTC).isoformat(),
            "version": skill.version,
        }
        self._save_lock()

        logger.info("Installed skill name=%s from=%s", name, source_path)
        return skill

    def remove(self, name: str) -> None:
        target = self._base / name
        if not target.exists():
            msg = f"Skill {name!r} not installed"
            raise SkillError(msg)

        shutil.rmtree(target)
        self._lock["skills"].pop(name, None)
        self._save_lock()
        logger.info("Removed skill name=%s", name)

    def list_installed(self) -> list[dict[str, Any]]:
        return [{"name": k, **v} for k, v in self._lock.get("skills", {}).items()]

    def verify_integrity(self, name: str) -> bool:
        """Check that installed skill's content hash matches the lock file."""
        skill_dir = self._base / name
        skill = _load_skill_directory(skill_dir)
        if skill is None:
            return False

        entry = self._lock.get("skills", {}).get(name)
        if entry is None:
            return False

        return skill.hash == entry.get("hash")

    def quarantine(self, name: str) -> None:
        """Move a tampered skill out of the way."""
        target = self._base / name
        quar_dir = self._hub_dir / "quarantine"
        quar_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.move(str(target), str(quar_dir / name))
            logger.warning("Quarantined skill name=%s", name)
