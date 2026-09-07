"""File-defined sub-agents — load named agents from ``.godspeed/agents/*.md``.

Each agent is a Markdown file with YAML frontmatter describing model, effort,
tool access, and system prompt overrides. Project agents live in
``{project}/.godspeed/agents/`` and win over user agents in
``~/.godspeed/agents/`` on name collision. Malformed or invalid files are
skipped with a warning (fail closed) — a bad definition never crashes a spawn.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from godspeed.agent.coordinator import CapabilityBundle, SubAgentConfig

logger = logging.getLogger(__name__)

AGENT_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
AGENT_NAME_MAX_CHARS = 32
USER_AGENT_DIR = Path.home() / ".godspeed" / "agents"
SYSTEM_PROMPT_DELIMITER = "---"
# Mirrors coordinator._EFFORT_ITERATIONS keys; kept explicit so a typo'd
# effort in a definition fails closed instead of silently changing limits.
_VALID_EFFORTS = frozenset({"low", "normal", "high"})


class AgentDefinitionError(Exception):
    """Raised when an agent definition is invalid."""


@dataclass(frozen=True)
class AgentDefinition(SubAgentConfig):
    """A file-defined sub-agent: ``SubAgentConfig`` plus name and description."""

    name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not AGENT_NAME_RE.match(self.name):
            msg = f"Agent name {self.name!r} does not match required pattern"
            raise AgentDefinitionError(msg)
        if len(self.name) > AGENT_NAME_MAX_CHARS:
            msg = f"Agent name {self.name!r} exceeds {AGENT_NAME_MAX_CHARS} chars"
            raise AgentDefinitionError(msg)

    def to_config(self) -> SubAgentConfig:
        """Return an equivalent base ``SubAgentConfig`` for spawning."""
        return SubAgentConfig(
            model=self.model,
            effort=self.effort,
            max_iterations=self.max_iterations,
            tool_bundle=self.tool_bundle,
            system_prompt=self.system_prompt,
            max_cost_usd=self.max_cost_usd,
        )


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str] | None:
    """Extract YAML frontmatter and body from an agent markdown string.

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


def _build_system_prompt(system_prompt: str | None, body: str) -> str:
    """Combine frontmatter ``system_prompt`` and markdown body into one prompt."""
    if not system_prompt:
        return body
    if not body:
        return system_prompt
    return f"{system_prompt}\n\n{SYSTEM_PROMPT_DELIMITER}\n\n{body}"


def _load_agent_file(path: Path) -> AgentDefinition | None:
    """Load a single agent definition file, or ``None`` if invalid."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read agent file %s: %s", path, exc)
        return None

    parsed = _parse_frontmatter(text)
    if parsed is None:
        logger.warning("Agent file %s has no valid YAML frontmatter", path)
        return None

    fm, body = parsed

    name = fm.get("name", path.stem)
    if not isinstance(name, str) or not name:
        logger.warning("Agent file %s missing name in frontmatter", path)
        return None
    if not AGENT_NAME_RE.match(name):
        logger.warning("Agent file %s has invalid name %r", path, name)
        return None
    if len(name) > AGENT_NAME_MAX_CHARS:
        logger.warning(
            "Agent file %s name %r exceeds %d chars",
            path,
            name,
            AGENT_NAME_MAX_CHARS,
        )
        return None

    description = fm.get("description", "")
    if not isinstance(description, str):
        logger.warning("Agent file %s description must be a string", path)
        return None

    model = fm.get("model")
    if model is not None and not isinstance(model, str):
        logger.warning("Agent file %s model must be a string", path)
        return None

    effort = fm.get("effort", "normal")
    if effort not in _VALID_EFFORTS:
        logger.warning("Agent file %s has invalid effort %r", path, effort)
        return None

    max_iterations = fm.get("max_iterations", 0)
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 0
    ):
        logger.warning("Agent file %s max_iterations must be a non-negative int", path)
        return None

    tool_bundle = fm.get("tool_bundle")
    if tool_bundle is not None:
        try:
            tool_bundle = CapabilityBundle(tool_bundle)
        except ValueError:
            logger.warning("Agent file %s has invalid tool_bundle %r", path, tool_bundle)
            return None

    system_prompt = fm.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        logger.warning("Agent file %s system_prompt must be a string", path)
        return None

    max_cost_usd = fm.get("max_cost_usd")
    if max_cost_usd is not None:
        if isinstance(max_cost_usd, bool) or not isinstance(max_cost_usd, (int, float)):
            logger.warning("Agent file %s max_cost_usd must be a number", path)
            return None
        max_cost_usd = float(max_cost_usd)

    return AgentDefinition(
        name=name,
        description=description,
        model=model,
        effort=effort,
        max_iterations=max_iterations,
        tool_bundle=tool_bundle,
        system_prompt=_build_system_prompt(system_prompt, body),
        max_cost_usd=max_cost_usd,
    )


def load_agent_definitions(project_dir: Path) -> dict[str, AgentDefinition]:
    """Load named sub-agent definitions from project and user scopes.

    Discovery order: ``{project_dir}/.godspeed/agents/*.md`` first, then
    ``~/.godspeed/agents/*.md``. Project definitions win on name collision.
    Malformed or invalid files are skipped with a warning (fail closed).

    Returns a fresh dict on every call — no caching, no shared mutable state.
    """
    definitions: dict[str, AgentDefinition] = {}
    for base in (Path(project_dir) / ".godspeed" / "agents", USER_AGENT_DIR):
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.md")):
            definition = _load_agent_file(path)
            if definition is None:
                continue
            if definition.name in definitions:
                logger.info("Agent %r overridden by %s", definition.name, path)
                continue
            definitions[definition.name] = definition
    return definitions
