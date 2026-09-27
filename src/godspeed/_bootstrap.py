"""Bootstrap helpers shared between CLI and MCP server.

Extracted from ``cli.py`` to break cyclic imports between ``cli.py`` and
``mcp_server/server.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from godspeed.config import DEFAULT_GLOBAL_DIR, GodspeedSettings

logger = logging.getLogger(__name__)


def _parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("Could not read env file %s: %s", path, exc)
        return result
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


# Variables that change WHAT CODE RUNS rather than what an API call authenticates with. The shell
# tool hands ``os.environ`` to every agent command, so a cloned repo's ``.godspeed/.env`` must not
# be able to set them: ``BASH_ENV`` is sourced by every non-interactive ``bash -c``, ``LD_PRELOAD``
# is loaded into every process, ``PATH`` decides which ``git`` runs, and
# ``GODSPEED_SHELL_WRAPPER`` replaces the shell binary outright. ``~/.godspeed/.env`` is the user's
# own file and is not filtered. This is a denylist of the known interpreter, loader and shell
# hooks; it cannot promise to name every one, so a project env file remains untrusted input.
_EXEC_ENV_NAMES = frozenset(
    {
        "PATH",
        "PATHEXT",
        "COMSPEC",
        "SHELL",
        "BASH_ENV",
        "ENV",
        "IFS",
        "CDPATH",
        "PROMPT_COMMAND",
        "PS4",
        "SHELLOPTS",
        "BASHOPTS",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONBREAKPOINT",
        "PYTHONUSERBASE",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
        "RUBYLIB",
        "CLASSPATH",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "_JAVA_OPTIONS",
        "GIT_SSH_COMMAND",
        "GIT_EXTERNAL_DIFF",
        "GIT_ASKPASS",
        "GIT_EXEC_PATH",
        "GIT_PAGER",
        "SSH_ASKPASS",
        "PAGER",
        "EDITOR",
        "VISUAL",
    }
)
_EXEC_ENV_PREFIXES = ("LD_", "DYLD_", "BASH_FUNC_", "GODSPEED_SHELL")


def _is_exec_env(key: str) -> bool:
    upper = key.upper()
    return upper in _EXEC_ENV_NAMES or upper.startswith(_EXEC_ENV_PREFIXES)


def _load_env_files(project_dir: Path | None = None) -> list[tuple[Path, list[str]]]:
    candidates: list[Path] = [
        DEFAULT_GLOBAL_DIR / ".env",
        DEFAULT_GLOBAL_DIR / ".env.local",
    ]
    project_files: list[Path] = []
    if project_dir is not None:
        project_files = [
            project_dir / ".godspeed" / ".env",
            project_dir / ".godspeed" / ".env.local",
        ]
        candidates.extend(project_files)

    resolved: dict[str, str] = {}
    contributions: list[tuple[Path, list[str]]] = []
    for path in candidates:
        parsed = _parse_env_file(path)
        if path in project_files:
            blocked = sorted(k for k in parsed if _is_exec_env(k))
            if blocked:
                logger.warning(
                    "Ignoring %d execution-affecting variable(s) in %s: %s. A project env file may "
                    "set API keys and endpoints, not what code runs; set these in your shell or in "
                    "~/.godspeed/.env if you really want them.",
                    len(blocked),
                    path,
                    ", ".join(blocked),
                )
                parsed = {k: v for k, v in parsed.items() if k not in blocked}
        if not parsed:
            continue
        contributions.append((path, sorted(parsed.keys())))
        resolved.update(parsed)

    loaded: list[tuple[Path, list[str]]] = []
    injected_keys: set[str] = set()
    for key, value in resolved.items():
        if key in os.environ:
            continue
        os.environ[key] = value
        injected_keys.add(key)

    for path, keys in contributions:
        effective = [k for k in keys if k in injected_keys]
        if not effective:
            continue
        loaded.append((path, effective))
        logger.info(
            "Loaded %d env var(s) from %s: %s",
            len(effective),
            path,
            ", ".join(effective),
        )
    return loaded


def _build_tool_registry(
    tool_set: str = "full",
    settings: GodspeedSettings | None = None,
) -> tuple:
    from godspeed.tools.tool_sets import get_allowed_tool_names

    allowed = get_allowed_tool_names(tool_set)
    from godspeed.tools.base import RiskLevel
    from godspeed.tools.file_edit import FileEditTool
    from godspeed.tools.file_move import FileMoveTool
    from godspeed.tools.file_read import FileReadTool
    from godspeed.tools.file_write import FileWriteTool
    from godspeed.tools.registry import ToolRegistry

    sandbox = None
    if settings is not None:
        from godspeed.config import SandboxMode
        from godspeed.sandbox.policy import build_sandbox_policy

        sandbox = build_sandbox_policy(
            blocked_paths=settings.sandbox_settings.blocked_paths or None,
            writable_paths=settings.sandbox_settings.writable_paths or None,
            enable_network=not settings.sandbox_settings.network_restricted,
            kernel_enforced=settings.sandbox_settings.mode == SandboxMode.KERNEL,
        )

    registry = ToolRegistry(sandbox=sandbox)
    risk_levels: dict[str, RiskLevel] = {}

    from godspeed.tools.background import BackgroundCheckTool
    from godspeed.tools.complexity import ComplexityTool
    from godspeed.tools.coverage import CoverageTool
    from godspeed.tools.db_query import DbQueryTool
    from godspeed.tools.dep_audit import DepAuditTool
    from godspeed.tools.generate_tests import GenerateTestsTool
    from godspeed.tools.git import GitTool
    from godspeed.tools.glob_search import GlobSearchTool
    from godspeed.tools.grep_search import GrepSearchTool
    from godspeed.tools.notebook import NotebookEditTool
    from godspeed.tools.repo_map import RepoMapTool
    from godspeed.tools.security_scan import SecurityScanTool
    from godspeed.tools.shell import ShellTool
    from godspeed.tools.test_runner import TestRunnerTool
    from godspeed.tools.think import ThinkTool
    from godspeed.tools.traceback_analyzer import TracebackAnalyzerTool
    from godspeed.tools.verify import VerifyTool
    from godspeed.tools.web_fetch import WebFetchTool
    from godspeed.tools.web_search import WebSearchTool

    try:
        from godspeed.tools.stock_price import StockPriceTool

        _stock_price_available = True
    except ImportError:
        _stock_price_available = False

    tools: list = [
        FileReadTool(),
        TracebackAnalyzerTool(),
        FileWriteTool(),
        FileEditTool(),
        FileMoveTool(),
        ShellTool(),
        GlobSearchTool(),
        GrepSearchTool(),
        GitTool(),
        VerifyTool(),
        RepoMapTool(),
        TestRunnerTool(),
        CoverageTool(),
        SecurityScanTool(),
        ComplexityTool(),
        ThinkTool(),
        DepAuditTool(),
        GenerateTestsTool(),
        WebSearchTool(),
        WebFetchTool(),
        NotebookEditTool(),
        BackgroundCheckTool(),
        DbQueryTool(),
    ]

    if _stock_price_available:
        tools.append(StockPriceTool())

    try:
        from godspeed.tools.image_read import ImageReadTool

        tools.append(ImageReadTool())
    except ImportError:
        logger.debug("image_read not available")

    try:
        from godspeed.tools.pdf_read import PdfReadTool

        tools.append(PdfReadTool())
    except ImportError:
        logger.debug("pdf_read not available")

    try:
        from godspeed.tools.github import GithubTool

        tools.append(GithubTool())
    except ImportError:
        logger.debug("github not available")

    try:
        from godspeed.tools.diff_apply import DiffApplyTool

        tools.append(DiffApplyTool())
    except ImportError:
        logger.debug("diff_apply not available")

    try:
        from godspeed.tools.ollama_manager import OllamaTool

        tools.append(OllamaTool())
    except ImportError:
        logger.debug("ollama_manager not available")

    try:
        from godspeed.tools.llamacpp_manager import LlamaCppTool

        tools.append(LlamaCppTool())
    except ImportError:
        logger.debug("llamacpp_manager not available")

    from godspeed.tools.acceptance import (
        AcceptanceInitTool,
        AcceptanceStatusTool,
        AcceptanceUpdateTool,
    )

    tools.extend([AcceptanceInitTool(), AcceptanceUpdateTool(), AcceptanceStatusTool()])

    for tool in tools:
        if allowed is not None and tool.name not in allowed:
            continue
        registry.register(tool)
        risk_levels[tool.name] = tool.risk_level

    return registry, risk_levels
