"""Configuration with pydantic-settings. Supports env vars, .env, and YAML config files."""

from __future__ import annotations

import logging
from collections import OrderedDict
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_DIR = Path.home() / ".godspeed"

# Schema version for settings.yaml — bump when adding/removing top-level keys
# so unknown-key warnings can distinguish "newer config" from typos.
CONFIG_SCHEMA_VERSION = 1

# YAML config cache: path -> (mtime, data)
# Bounded to 32 entries with LRU eviction to avoid unbounded growth in
# long-running processes while keeping frequently-accessed configs hot.
_MAX_YAML_CACHE_SIZE: int = 32
_yaml_cache: OrderedDict[Path, tuple[float, dict[str, Any]]] = OrderedDict()


def _load_yaml_cached(path: Path) -> dict[str, Any] | None:
    """Load YAML from *path* with mtime-based caching.

    Returns the parsed dict, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _yaml_cache.get(path)
    if cached is not None and cached[0] == mtime:
        # Promote to most-recently-used
        _yaml_cache.move_to_end(path)
        return cached[1]
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        data = {}
    # Evict least-recently-used entries when cache grows beyond limit
    while len(_yaml_cache) >= _MAX_YAML_CACHE_SIZE:
        _yaml_cache.popitem(last=False)
    _yaml_cache[path] = (mtime, data)
    return data


_PERMISSION_MODES = ("strict", "normal", "yolo")
_SANDBOX_MODES = ("none", "docker")
_EXECUTION_MODES = ("tool", "codeact")


class PermissionMode(StrEnum):
    """Permission enforcement tier.

    ``strict``:  Deny-first, every tool call requires approval.
    ``normal``:  Default — allow/deny/ask rules apply.
    ``yolo``:    No permission checks (INSECURE).
    """

    STRICT = "strict"
    NORMAL = "normal"
    YOLO = "yolo"


class SandboxMode(StrEnum):
    """Sandboxing strategy for tool execution.

    ``none``:   No sandboxing.
    ``docker``: Run tools inside a Docker container.
    """

    NONE = "none"
    DOCKER = "docker"


class ExecutionMode(StrEnum):
    """Agent execution strategy.

    ``tool`` (default):  Agent uses tool calls as its primary action mechanism.
    ``codeact``:         Agent writes code blocks that are auto-executed.
    """

    TOOL = "tool"
    CODEACT = "codeact"


# Model context windows — used for model-aware compaction prompts.
# Keys are prefixes matched against the model string.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Frontier models with large context
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "gemini-2": 1_000_000,
    "gemini-1.5": 1_000_000,
    "gemini-pro": 32_000,
    # Open-source models (common Ollama defaults)
    "ollama/qwen3": 32_768,
    "ollama/llama3": 8_192,
    "ollama/llama3.1": 128_000,
    "ollama/mistral": 32_768,
    "ollama/codellama": 16_384,
    "ollama/deepseek": 32_768,
    "ollama/gemma": 8_192,
    "ollama/phi": 16_384,
    # Local llama.cpp models
    "openai/qwen3.6": 32_768,
    "llamacpp/qwen3.6": 32_768,
    "qwen3.6": 32_768,
    "openai/qwen2.5-coder": 32_768,
    "openai/zaya1": 131_072,
    "zaya1": 131_072,
}


def get_model_context_window(model: str) -> int:
    """Get the context window size for a model by prefix matching.

    Returns the matched size or 32_768 as a safe default.
    """
    model_lower = model.lower()
    # Try longest prefix match first for specificity
    best_match = ""
    best_size = 32_768  # safe default
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model_lower.startswith(prefix.lower()) and len(prefix) > len(best_match):
            best_match = prefix
            best_size = size
    return best_size


DEFAULT_PROJECT_DIR = Path(".godspeed")


class PermissionSettings(BaseSettings):
    """Permission rules configuration."""

    deny: list[str] = Field(
        default_factory=lambda: [
            # Environment files
            "FileRead(.env)",
            "FileRead(.env.*)",
            "FileRead(.env.local)",
            "FileRead(.env.production)",
            "FileRead(.env.staging)",
            # Keys and certificates
            "FileRead(*.pem)",
            "FileRead(*.key)",
            "FileRead(*.p12)",
            "FileRead(*.pfx)",
            "FileRead(*.jks)",
            # Credential files
            "FileRead(*credentials*)",
            "FileRead(*secret*)",
            # Cloud provider configs
            "FileRead(.aws/*)",
            "FileRead(.gcloud/*)",
            "FileRead(.azure/*)",
            # SSH keys
            "FileRead(.ssh/*)",
            # Docker secrets
            "FileRead(docker-compose*.secret*)",
            # Also block writing to sensitive files
            "FileWrite(.env)",
            "FileWrite(.env.*)",
            "FileWrite(.ssh/*)",
            "FileWrite(.aws/*)",
        ]
    )
    allow: list[str] = Field(
        default_factory=lambda: [
            "shell(git *)",
            "shell(ruff *)",
            "shell(pytest *)",
            "shell(make *)",
        ]
    )
    ask: list[str] = Field(default_factory=lambda: ["shell(*)"])

    model_config = SettingsConfigDict(extra="ignore")


class AuditSettings(BaseSettings):
    """Audit trail configuration."""

    enabled: bool = True
    retention_days: int = 30

    model_config = SettingsConfigDict(extra="ignore")


class ContextSettings(BaseSettings):
    """Context management configuration."""

    project_instructions: str = "GODSPEED.md"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(
        default_factory=lambda: [
            "node_modules/",
            ".venv/",
            "__pycache__/",
            "*.pyc",
            ".git/",
        ]
    )

    model_config = SettingsConfigDict(extra="ignore")


class SandboxSettings(BaseSettings):
    """Sandboxing configuration for tool execution."""

    mode: SandboxMode = SandboxMode.NONE
    image: str = "python:3.12-slim"
    timeout_seconds: int = 120
    network_enabled: bool = False

    blocked_paths: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)

    model_config = SettingsConfigDict(extra="ignore")


class StatuslineSettings(BaseSettings):
    """Statusline (per-turn HUD) configuration.

    When ``enabled`` and a ``template`` is set, the HUD renders the template
    with ``{model}``, ``{tokens}``, ``{cost}``, and ``{branch}`` placeholders.
    An empty template uses the built-in default. Disabled (default) keeps the
    standard HUD.
    """

    enabled: bool = False
    template: str = ""

    model_config = SettingsConfigDict(extra="ignore")


class BatchSettings(BaseSettings):
    """Batch execution configuration for ``/batch`` and ``godspeed batch``.

    Controls default parallelism, PR creation, and worktree location for
    parallel worktree-based task decomposition runs.
    """

    parallelism: int = 5
    open_pr: bool = False
    worktree_dir: str = ""

    model_config = SettingsConfigDict(extra="ignore")


class GodspeedSettings(BaseSettings):
    """Root configuration for Godspeed."""

    # Model presets for speed + quality balance.
    # "cloud" and "frontier" are API-based (no local VRAM needed).
    # "fast", "balanced", "quality" are local Ollama models.
    # "local" is llama.cpp-based with speculative decoding (~750 tok/s).
    MODEL_PRESETS: ClassVar[dict[str, str]] = {
        "local": "openai/qwen2.5-coder-14b",
        "zaya": "openai/zaya1-8b",
    }

    # LLM — default to local Qwen2.5-Coder 14B via llama.cpp (GPU spec
    # decoding with 1.5B draft, ~750 tok/s on RTX 5070 Ti).
    # Experimental: ZAYA1-8B NF4 via transformers (~7.2GB VRAM, thinking profile).
    # Override via settings.yaml, GODSPEED_MODEL env var, or -m flag.
    model: str = "openai/qwen2.5-coder-14b"
    fallback_models: list[str] = Field(default_factory=list)

    # Paths
    project_dir: Path = Path(".")
    global_dir: Path = DEFAULT_GLOBAL_DIR

    # Security
    permission_mode: PermissionMode = PermissionMode.NORMAL

    # Context
    max_context_tokens: int = 100_000
    compaction_threshold: float = 0.8

    # Model routing — map task types to specific models
    routing: dict[str, str] = Field(default_factory=dict)

    # MCP servers — each entry is a dict with keys:
    #   name:      str   — unique server identifier (required)
    #   transport: str   — "stdio" (default) or "sse"
    #   command:   str   — executable for stdio transport
    #   args:      list  — CLI args for stdio transport
    #   env:       dict  — extra env vars for stdio subprocess
    #   url:       str   — base URL for sse transport (e.g. "http://localhost:3001")
    #   headers:   dict  — HTTP headers for sse transport (e.g. Authorization)
    mcp_servers: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "MCP server configurations. Each entry supports 'stdio' transport "
            "(command, args, env) or 'sse' transport (url, headers). "
            "Defaults to stdio when transport is omitted."
        ),
    )

    # Hooks — shell commands at lifecycle events
    hooks: list[dict[str, Any]] = Field(default_factory=list)

    # GitHub Actions — workflow configuration
    # {
    #   "workflows": [{"name": "Review", "on": ["pull_request"], "runs-on": "ubuntu-latest"}],
    #   "token": "${{ secrets.GITHUB_TOKEN }}",
    # }
    github_actions: dict[str, Any] = Field(default_factory=dict)

    # Agent behavior
    execution_mode: ExecutionMode = ExecutionMode.TOOL
    parallel_tool_calls: bool = True
    auto_fix_retries: int = 3  # lint-fix retry rounds (0 = one-shot, no auto-fix)
    auto_commit: bool = False
    auto_commit_threshold: int = 5

    # Loop control — configurable magic numbers (advanced users)
    max_iterations: int = 50  # max agent loop iterations per turn
    max_retries: int = 3  # max retries for malformed tool calls
    stuck_loop_threshold: int = 3  # consecutive identical errors before warning
    auto_stash_threshold: int = 3  # consecutive writes before auto-stash
    must_fix_cap: int = 3  # max must-fix injections per session
    max_speculative_cache_size: int = 10  # max concurrent speculative tasks per iteration

    # Per-session lifetime caps (Claude Code parity)
    max_spawns_per_session: int = 200
    max_web_calls_per_session: int = 200
    session_timeout_seconds: int = 0  # 0 = no wall-clock limit
    stagger_seconds: float = 0.25  # sibling spawn stagger for prompt-cache prefixes

    # MCP servers whose tools get MEDIUM risk (untrusted servers default HIGH)
    mcp_trusted_servers: list[str] = Field(default_factory=list)

    # Thinking — extended thinking for Anthropic/Claude models
    thinking_budget: int = 0  # 0 = disabled; >0 = budget_tokens for thinking blocks

    # Cost budget — hard limit on session spend (0 = unlimited)
    max_cost_usd: float = 0.0

    # Reasoning effort — for reasoning-capable models (o1, o3, o4-mini, etc.)
    reasoning_effort: str = ""  # "", "low", "medium", "high"

    # Architect mode — two-model pipeline (plan then execute)
    architect_model: str = ""  # model for planning phase; empty = use main model

    # Task-aware routing shortcuts. Populate `routing` for the canonical
    # task types (see godspeed.llm.router) without requiring users to
    # learn the dict syntax. Explicit `routing:` entries always win.
    #
    #   strong_model  → routing["plan"]      (fresh reasoning turns)
    #   cheap_model   → routing["edit"], routing["read"], routing["shell"]
    #
    # Leave empty to skip the shortcut for that tier.
    cheap_model: str = ""
    strong_model: str = ""

    # Sandboxing
    sandbox: SandboxMode = SandboxMode.NONE

    # Self-evolution — learn from execution traces to improve prompts/tools
    evolution_enabled: bool = False  # enable with /evolve or config
    evolution_model: str = ""  # model for mutations/judging; empty = auto-detect
    evolution_max_cost_usd: float = 0.50  # hard cap per evolution cycle
    evolution_min_sessions: int = 10  # sessions between auto-runs

    # Training data — log full conversations for fine-tuning
    log_conversations: bool = True

    # Memory
    memory_enabled: bool = True

    # Codebase auto-indexing — build a ChromaDB semantic index in the
    # background on session start (when the `[index]` extra is installed
    # and the project has no fresh index). Non-blocking; the session
    # continues without waiting for it to complete.
    auto_index: bool = True

    # Nested settings
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    sandbox_settings: SandboxSettings = Field(default_factory=SandboxSettings)
    statusline: StatuslineSettings = Field(default_factory=StatuslineSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)

    @model_validator(mode="after")
    def reconcile_sandbox_modes(self) -> GodspeedSettings:
        if self.sandbox != SandboxMode.NONE and self.sandbox_settings.mode == SandboxMode.NONE:
            self.sandbox_settings.mode = self.sandbox
        return self

    @field_validator("permission_mode", mode="before")
    @classmethod
    def validate_permission_mode(cls, v: Any) -> Any:
        if v not in _PERMISSION_MODES:
            msg = f"permission_mode must be one of {_PERMISSION_MODES}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "model must be a non-empty string"
            raise ValueError(msg)
        return v.strip()

    @field_validator("sandbox", mode="before")
    @classmethod
    def validate_sandbox(cls, v: Any) -> Any:
        if v not in _SANDBOX_MODES:
            msg = f"sandbox must be one of {_SANDBOX_MODES}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("execution_mode", mode="before")
    @classmethod
    def validate_execution_mode(cls, v: Any) -> Any:
        if v not in _EXECUTION_MODES:
            msg = f"execution_mode must be one of {_EXECUTION_MODES}, got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("auto_fix_retries")
    @classmethod
    def validate_auto_fix_retries(cls, v: int) -> int:
        if v < 0:
            msg = f"auto_fix_retries must be >= 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("thinking_budget")
    @classmethod
    def validate_thinking_budget(cls, v: int) -> int:
        if v < 0:
            msg = f"thinking_budget must be >= 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_cost_usd")
    @classmethod
    def validate_max_cost_usd(cls, v: float) -> float:
        if v < 0.0:
            msg = f"max_cost_usd must be >= 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("auto_commit_threshold")
    @classmethod
    def validate_auto_commit_threshold(cls, v: int) -> int:
        if v < 1:
            msg = f"auto_commit_threshold must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    model_config = SettingsConfigDict(
        env_prefix="GODSPEED_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("compaction_threshold")
    @classmethod
    def validate_compaction_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            msg = f"compaction_threshold must be between 0 and 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_context_tokens")
    @classmethod
    def validate_max_context_tokens(cls, v: int) -> int:
        if v < 1000:
            msg = f"max_context_tokens must be at least 1000, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def populate_routing_from_shortcuts(self) -> GodspeedSettings:
        """Fill ``routing[<task_type>]`` from the cheap/strong/architect shortcuts.

        Explicit ``routing:`` entries always win — the shortcuts only
        fill gaps via ``setdefault``. Empty-string field values are
        treated as "not set" and skipped. Canonical task types come
        from :mod:`godspeed.llm.router`.
        """
        if self.cheap_model:
            for task_type in ("edit", "read", "shell"):
                self.routing.setdefault(task_type, self.cheap_model)
        if self.strong_model:
            self.routing.setdefault("plan", self.strong_model)
        if self.architect_model:
            self.routing.setdefault("architect", self.architect_model)
        return self

    @model_validator(mode="after")
    def validate_collections(self) -> GodspeedSettings:
        """Validate hooks and mcp_servers entries have required fields."""
        for hook in self.hooks:
            if not isinstance(hook, dict):
                logger.warning("Skipping non-dict hook entry: %s", hook)
                continue
            if not hook.get("command"):
                logger.warning("Hook entry missing required 'command' key: %s", hook)

        for server in self.mcp_servers:
            if not isinstance(server, dict):
                logger.warning("Skipping non-dict MCP server entry: %s", server)
                continue
            if not server.get("name"):
                logger.warning("MCP server entry missing required 'name' key: %s", server)

        # Validate routing model names are non-empty
        for task_type, model_name in self.routing.items():
            if not model_name or not model_name.strip():
                logger.warning("Empty model name in routing[%r] — ignoring", task_type)

        return self

    @model_validator(mode="after")
    def warn_insecure_settings(self) -> GodspeedSettings:
        """Log warnings for insecure configuration settings."""
        # Check for yolo mode (no permission checks)
        if self.permission_mode == "yolo":
            logger.warning(
                "INSECURE: permission_mode='yolo' disables all permission checks. "
                "Any tool can be executed without user approval."
            )

        # Check for empty deny list
        if not self.permissions.deny:
            logger.warning(
                "No deny rules configured in permissions.deny. "
                "Consider adding deny rules for sensitive files (e.g., .env, *.key, .ssh/*)."
            )

        # Check if audit is disabled
        if not self.audit.enabled:
            logger.warning(
                "INSECURE: audit.enabled=False disables the audit trail. "
                "This reduces accountability and makes security incidents harder to investigate."
            )

        # Check for very permissive allow rules
        for rule in self.permissions.allow:
            if rule == "*" or rule == "**":
                logger.warning(
                    "INSECURE: Allow rule '%s' permits ALL tools without restriction.",
                    rule,
                )

        # Check for sandbox=none with yolo mode
        if self.sandbox == "none" and self.permission_mode == "yolo":
            logger.warning(
                "HIGHLY INSECURE: sandbox='none' with permission_mode='yolo'. "
                "Tools run without any sandboxing or permission checks."
            )

        # Check for very high context tokens (potential memory issue)
        if self.max_context_tokens > 500_000:
            logger.warning(
                "Very high max_context_tokens=%d may cause memory issues. "
                "Consider reducing to 200000 or less.",
                self.max_context_tokens,
            )

        return self

    @model_validator(mode="before")
    @classmethod
    def load_yaml_configs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load and merge YAML config files (global > project). Env vars take precedence.

        Uses mtime-based caching to avoid re-parsing YAML on every
        ``GodspeedSettings()`` instantiation.
        """
        merged: dict[str, Any] = {}

        # Load global config (cached)
        global_config = DEFAULT_GLOBAL_DIR / "settings.yaml"
        try:
            global_data = _load_yaml_cached(global_config)
            if global_data is not None:
                merged.update(global_data)
        except yaml.YAMLError as exc:
            logger.warning("Malformed global settings.yaml: %s — skipping", exc)

        # Load project config (overrides global, except deny rules which merge)
        project_config = DEFAULT_PROJECT_DIR / "settings.yaml"
        try:
            project_data = _load_yaml_cached(project_config)
            if project_data is not None:
                _merge_configs(merged, project_data)
        except yaml.YAMLError as exc:
            logger.warning("Malformed project settings.yaml: %s — skipping", exc)

        # Warn on unknown top-level keys (typos / newer configs)
        _warn_unknown_keys(merged)

        # Env vars / constructor args take final precedence
        merged.update({k: v for k, v in data.items() if v is not None})
        return merged


def append_permission_rule(
    pattern: str,
    action: str,
    project_dir: Path | None = None,
) -> Path | None:
    """Append a permission rule to the project or global ``settings.yaml``.

    Reads existing YAML, adds the pattern under ``permissions.<action>``,
    writes back. Preserves existing content. Duplicate patterns are
    silently skipped (re-running the command is idempotent).

    Args:
        pattern: Permission pattern to add (e.g. ``"shell(git status)"``).
        action: One of ``"allow" | "deny" | "ask"``.
        project_dir: Project directory for ``.godspeed/settings.yaml``.
            Falls back to the global settings file when ``None``.

    Returns:
        The :class:`Path` written on success, or ``None`` on OS error.

    Raises:
        ValueError: if ``action`` is not one of the three valid tiers.
    """
    if action not in ("allow", "deny", "ask"):
        msg = f"action must be 'allow' | 'deny' | 'ask', got {action!r}"
        raise ValueError(msg)

    # Determine which settings file to write to
    if project_dir is not None:
        settings_path = project_dir / ".godspeed" / "settings.yaml"
    else:
        settings_path = DEFAULT_GLOBAL_DIR / "settings.yaml"

    try:
        if settings_path.exists() and settings_path.is_file():
            with open(settings_path, encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError as exc:
                    logger.warning("Malformed %s: %s — rebuilding", settings_path, exc)
                    data = {}
        else:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}

        if not isinstance(data, dict):
            data = {}

        permissions = data.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            data["permissions"] = permissions
        rule_list = permissions.setdefault(action, [])
        if not isinstance(rule_list, list):
            logger.warning("Corrupted rule list for action=%s — rebuilding", action)
            rule_list = []
            permissions[action] = rule_list

        if pattern not in rule_list:
            rule_list.append(pattern)

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Appended %s rule '%s' to %s", action, pattern, settings_path)
        return settings_path
    except OSError as exc:
        logger.warning("Failed to write %s rule: %s", action, exc)
        return None


def append_allow_rule(pattern: str, project_dir: Path | None = None) -> bool:
    """Back-compat wrapper around :func:`append_permission_rule` for allow rules."""
    return append_permission_rule(pattern, "allow", project_dir) is not None


def append_mcp_server(
    entry: dict[str, Any],
    project_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path | None:
    """Add or replace an MCP server entry in ``settings.yaml``.

    Reads existing YAML, adds *entry* under the ``mcp_servers`` list, writes
    back. Preserves existing content. When a server with the same ``name``
    already exists and ``force`` is False, raises :class:`ValueError`; with
    ``force=True`` the existing entry is replaced in place.

    Args:
        entry: MCP server dict (``name``, ``transport``, ``command``, ``args``,
            ``env``, ``url``, ...). Must include a ``name`` key.
        project_dir: Project directory for ``.godspeed/settings.yaml``.
            Falls back to the global settings file when ``None``.
        force: Replace an existing entry with the same name instead of erroring.

    Returns:
        The :class:`Path` written on success, or ``None`` on OS error.

    Raises:
        ValueError: if *entry* has no ``name``, or a server with the same name
            already exists and ``force`` is False.
    """
    name = entry.get("name")
    if not name:
        msg = "MCP server entry requires a 'name' key"
        raise ValueError(msg)

    if project_dir is not None:
        settings_path = project_dir / ".godspeed" / "settings.yaml"
    else:
        settings_path = DEFAULT_GLOBAL_DIR / "settings.yaml"

    try:
        if settings_path.exists() and settings_path.is_file():
            with open(settings_path, encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError as exc:
                    logger.warning("Malformed %s: %s — rebuilding", settings_path, exc)
                    data = {}
        else:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}

        if not isinstance(data, dict):
            data = {}

        servers = data.setdefault("mcp_servers", [])
        if not isinstance(servers, list):
            logger.warning("Corrupted mcp_servers list — rebuilding")
            servers = []
            data["mcp_servers"] = servers

        replaced = False
        for i, existing in enumerate(servers):
            if isinstance(existing, dict) and existing.get("name") == name:
                if not force:
                    msg = f"MCP server '{name}' already exists (use --force to replace)"
                    raise ValueError(msg)
                servers[i] = entry
                replaced = True
                break
        if not replaced:
            servers.append(entry)

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Wrote MCP server '%s' to %s", name, settings_path)
        return settings_path
    except OSError as exc:
        logger.warning("Failed to write MCP server '%s': %s", name, exc)
        return None


def remove_mcp_server(
    name: str,
    project_dir: Path | None = None,
) -> tuple[bool, Path | None]:
    """Remove an MCP server entry by name from ``settings.yaml``.

    Args:
        name: Name of the MCP server to remove.
        project_dir: Project directory for ``.godspeed/settings.yaml``.
            Falls back to the global settings file when ``None``.

    Returns:
        A ``(found, path)`` tuple. ``found`` is True when an entry was removed;
        ``path`` is the settings file written, or ``None`` when the file was
        missing/malformed or an OS error occurred.
    """
    if project_dir is not None:
        settings_path = project_dir / ".godspeed" / "settings.yaml"
    else:
        settings_path = DEFAULT_GLOBAL_DIR / "settings.yaml"

    try:
        if not settings_path.exists() or not settings_path.is_file():
            return False, None
        with open(settings_path, encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                logger.warning("Malformed %s: %s", settings_path, exc)
                return False, None
        if not isinstance(data, dict):
            return False, None

        servers = data.get("mcp_servers")
        if not isinstance(servers, list):
            return False, None

        new_servers = [s for s in servers if not (isinstance(s, dict) and s.get("name") == name)]
        if len(new_servers) == len(servers):
            return False, None

        if new_servers:
            data["mcp_servers"] = new_servers
        else:
            data.pop("mcp_servers", None)

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Removed MCP server '%s' from %s", name, settings_path)
        return True, settings_path
    except OSError as exc:
        logger.warning("Failed to remove MCP server '%s': %s", name, exc)
        return False, None


_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "fallback_models",
        "project_dir",
        "global_dir",
        "permission_mode",
        "max_context_tokens",
        "compaction_threshold",
        "routing",
        "mcp_servers",
        "hooks",
        "github_actions",
        "execution_mode",
        "parallel_tool_calls",
        "auto_fix_retries",
        "auto_commit",
        "auto_commit_threshold",
        "max_iterations",
        "max_retries",
        "stuck_loop_threshold",
        "auto_stash_threshold",
        "must_fix_cap",
        "max_speculative_cache_size",
        "thinking_budget",
        "max_cost_usd",
        "reasoning_effort",
        "architect_model",
        "cheap_model",
        "strong_model",
        "sandbox",
        "sandbox_settings",
        "evolution_enabled",
        "evolution_model",
        "evolution_max_cost_usd",
        "evolution_min_sessions",
        "log_conversations",
        "memory_enabled",
        "auto_index",
        "permissions",
        "audit",
        "context",
        "statusline",
        "batch",
    }
)


def _warn_unknown_keys(data: dict[str, Any]) -> None:
    """Log a warning for unknown top-level config keys (typos / newer configs)."""
    for key in data:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            logger.warning(
                "Unknown settings key %r (schema version %d) — ignoring. "
                "Check for typos or a newer config than this version supports.",
                key,
                CONFIG_SCHEMA_VERSION,
            )


def _merge_configs(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Merge override into base. Deny rules are additive (project can't weaken global denies)."""
    for key, value in override.items():
        if key == "permissions" and isinstance(value, dict):
            base_perms = base.setdefault("permissions", {})
            # Deny rules are additive — project can only add more denies
            if "deny" in value:
                existing = base_perms.get("deny", [])
                base_perms["deny"] = list(dict.fromkeys(existing + value["deny"]))
            # Allow and ask rules: project overrides global
            if "allow" in value:
                base_perms["allow"] = value["allow"]
            if "ask" in value:
                base_perms["ask"] = value["ask"]
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_configs(base[key], value)
        else:
            base[key] = value
