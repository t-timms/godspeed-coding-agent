"""Permission rule types for the 4-tier permission engine."""

from __future__ import annotations

import fnmatch
import logging
import re
from enum import StrEnum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_pattern_tool_name(pattern: str) -> str:
    """Normalize the tool-name token of a rule pattern to snake_case.

    Only the token before the first ``(`` is normalized; the argument
    glob is left untouched. Uses the canonical ``normalize_tool_name``
    from ``tools.base`` so rule patterns and formatted calls always
    agree.
    """
    from godspeed.tools.base import normalize_tool_name

    head, sep, rest = pattern.partition("(")
    if not sep:
        return normalize_tool_name(head)
    return normalize_tool_name(head) + sep + rest


def _compile_glob(pattern: str) -> re.Pattern:
    """Compile a normalized fnmatch glob to a case-insensitive regex once.

    Case-insensitive as defense in depth: file extensions compare
    case-insensitively on Windows filesystems, and a deny rule must never
    be silently dead because of a case mismatch.
    """
    return re.compile(fnmatch.translate(normalize_pattern_tool_name(pattern)), re.IGNORECASE)


class RuleAction(StrEnum):
    """What happens when a rule matches."""

    DENY = "deny"
    ALLOW = "allow"
    ASK = "ask"


class PermissionRule(BaseModel):
    """A permission rule matching tool calls by pattern.

    Pattern format: 'ToolName(argument_pattern)'
    Examples:
        - 'shell(git *)' — matches any git command
        - 'FileRead(.env)' — matches reading .env
        - 'shell(*)' — matches any shell command
        - 'FileRead(*.pem)' — matches reading any .pem file

    The glob pattern is compiled to a regex at construction time for
    ~3-5x faster matching per evaluation.
    """

    pattern: str
    action: RuleAction

    def model_post_init(self, _context: object) -> None:
        self._compiled: re.Pattern = _compile_glob(self.pattern)

    def matches(self, tool_call_str: str) -> bool:
        """Check if this rule matches a formatted tool call string (compiled regex)."""
        return bool(self._compiled.match(tool_call_str))


def parse_rules(patterns: list[str], action: RuleAction) -> list[PermissionRule]:
    """Parse a list of pattern strings into PermissionRule objects."""
    return [PermissionRule(pattern=p, action=action) for p in patterns]
