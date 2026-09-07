"""Named tool groups for constraining the agent's capability surface.

Small open-source models often pick the wrong tool when handed all 20+
schemas. A ``--tool-set local`` run hides web-facing tools entirely, so
``file_read`` beats ``web_search`` on local-codebase tasks without
needing a stronger prompt or a fine-tune.

Named sets are intentionally small and stable — if you add a new tool,
decide which set(s) it belongs in here. Anything not explicitly listed
falls through to ``full``.
"""

from __future__ import annotations

from types import MappingProxyType

# Tools that touch the public internet or external services.
_WEB_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "web_search",
        "web_fetch",
        "github",
    }
)

# Tools scoped to the local project + filesystem + shell.
# Everything NOT in the web set is considered local by default.
_EXPLICIT_LOCAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "file_read",
        "file_write",
        "file_edit",
        "grep_search",
        "glob_search",
        "code_search",
        "shell",
        "git",
        "repo_map",
        "verify",
        "test_runner",
        "coverage",
        "complexity",
        "think",
        "security_scan",
        "dep_audit",
        "generate_tests",
        "notebook_edit",
        "pdf_read",
        "image_read",
        "diff_apply",
        "tasks",
        "acceptance_init",
        "acceptance_update",
        "acceptance_status",
        "background_check",
        "db_query",
        "exit_plan_mode",
    }
)

# SWE-Bench tool set: only tools actually useful for bug fixing in
# isolated temp repos. Drops 10 analysis/docs tools whose schemas
# bloat every API call without ever being selected by the agent.
_EXPLICIT_SWEBENCH_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "file_read",
        "file_write",
        "file_edit",
        "grep_search",
        "glob_search",
        "shell",
        "git",
        "verify",
        "test_runner",
        "diff_apply",
        "file_move",
        "repo_map",
    }
)

TOOL_SET_LOCAL = "local"
TOOL_SET_WEB = "web"
TOOL_SET_FULL = "full"
TOOL_SET_SWEBENCH = "swebench"

VALID_TOOL_SETS: frozenset[str] = frozenset(
    {TOOL_SET_LOCAL, TOOL_SET_WEB, TOOL_SET_FULL, TOOL_SET_SWEBENCH}
)

# Public mapping: tool-set name → the frozenset of names allowed in that set.
# For ``full``, we return None below so callers can short-circuit registry
# filtering entirely rather than build a set that contains everything.
_SET_DEFINITIONS: MappingProxyType[str, frozenset[str]] = MappingProxyType(
    {
        TOOL_SET_LOCAL: _EXPLICIT_LOCAL_TOOL_NAMES,
        TOOL_SET_WEB: _EXPLICIT_LOCAL_TOOL_NAMES | _WEB_TOOL_NAMES,
        TOOL_SET_SWEBENCH: _EXPLICIT_SWEBENCH_TOOL_NAMES,
    }
)


def get_allowed_tool_names(tool_set: str) -> frozenset[str] | None:
    """Return the frozenset of names allowed in ``tool_set``.

    ``None`` means "no filter — register every available tool" (i.e. the
    ``full`` set). That's a different shape from an empty set because an
    empty set would disable every tool; None is the explicit opt-out.

    Raises ValueError on unknown set names rather than silently returning
    full — a typo shouldn't accidentally expose a superset.
    """
    if tool_set not in VALID_TOOL_SETS:
        msg = f"Unknown tool set {tool_set!r}. Valid options: {sorted(VALID_TOOL_SETS)}"
        raise ValueError(msg)
    if tool_set == TOOL_SET_FULL:
        return None
    return _SET_DEFINITIONS[tool_set]
