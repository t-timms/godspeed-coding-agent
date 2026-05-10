"""Main Textual TUI application for Godspeed.

Replaces the prompt-toolkit REPL with a full Textual TUI featuring:
- Instant boot via splash screen — backend loads progressively
- Dual theme support — dark (default) and light (cream paper, eye-safe)
- Persistent status bar with model, tokens, cost, permission mode
- Scrollable RichLog for conversation history
- Multi-line TextArea prompt input
- Help screen (F1), Sessions screen (Ctrl+S), Command palette (Ctrl+P)
- Fuzzy file picker (@-mention triggers dropdown)
- Async-native event loop (no run_in_executor)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from godspeed.security.permissions import ALLOW, ASK, PermissionDecision, PermissionEngine
from godspeed.tools.base import RiskLevel
from godspeed.tools.registry import ToolRegistry
from godspeed.tui.output import (
    format_session_summary,
    set_compact_mode,
)

logger = logging.getLogger(__name__)

# -- Eye-safe palettes ---------------------------------------------------------

_DARK = Theme(
    name="godspeed-dark",
    primary="#c17c5b",
    secondary="#8b7355",
    warning="#d4a817",
    error="#a45252",
    success="#87a96b",
    accent="#87a96b",
    foreground="#e6dac8",
    background="#0e0c09",
    surface="#0e0c09",
    panel="#151210",
    dark=True,
    variables={
        "text-muted": "#5c5246",
        "border": "#25221e",
        "border-focus": "#c17c5b",
        "selection": "#2d2823",
    },
)

_LIGHT = Theme(
    name="godspeed-light",
    primary="#b86e4a",
    secondary="#8b7355",
    warning="#b8860f",
    error="#8b3028",
    success="#6b8e3d",
    accent="#6b8e3d",
    foreground="#2d2318",
    background="#faf5ee",
    surface="#faf5ee",
    panel="#f0e8d8",
    dark=False,
    variables={
        "text-muted": "#8b7355",
        "border": "#d4c8b0",
        "border-focus": "#b86e4a",
        "selection": "#e8ddcc",
    },
)


class GodspeedTextualApp(App):
    """Main Textual application for Godspeed coding agent."""

    CSS = """
    ChatScreen { layout: grid; grid-rows: 1fr auto auto auto; grid-size: 1; }

    #chat-log {
        height: 1fr; border: none; padding: 1 2;
        background: $surface; color: $foreground;
    }

    #input-area {
        height: auto; padding: 1 2; border-top: solid $border;
    }

    #prompt-input {
        min-height: 2; max-height: 10; border: none;
        background: $panel; color: $foreground; padding: 0 1;
        border-left: solid $border;
    }

    #prompt-input:focus {
        border-left: solid $primary;
    }

    Footer {
        dock: bottom; height: 1; padding: 0 2;
        background: $surface; color: $text-muted; border-top: solid $border;
    }

    * {
        scrollbar-background: $surface; scrollbar-color: $border;
        scrollbar-color-hover: $text-muted; scrollbar-color-active: $primary;
    }

    #file-picker {
        display: none; height: auto; max-height: 12;
        margin: 0 2; border: solid $primary;
        background: $surface; color: $foreground;
        overflow-y: auto;
    }

    #file-tree {
        dock: left; width: 30; display: none;
        border-right: solid $border; background: $surface;
    }

    #file-tree:focus {
        border-right: solid $primary;
    }

    #help-content, #sessions-content, #permission-content, #diff-content {
        padding: 2 3; height: 1fr; background: $surface; color: $foreground;
    }

    #sessions-header, #sessions-footer {
        height: auto; padding: 0 2; background: $surface; color: $foreground;
    }

    #sessions-list {
        height: 1fr; background: $surface; border: none;
    }

    #sessions-list ListView > ListItem {
        padding: 0 1; color: $foreground;
    }

    #sessions-list ListView > ListItem.--highlight {
        background: $primary; color: $surface;
    }
    """

    BINDINGS: ClassVar[list] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+t", "toggle_theme", "Theme", show=True),
    ]

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return {
            "text-muted": "#5c5246",
            "border": "#25221e",
            "border-focus": "#c17c5b",
            "selection": "#2d2823",
        }

    def __init__(
        self,
        settings: Any,
        registry: ToolRegistry,
        risk_levels: dict[str, Any],
        effective_model: str,
        effective_project_dir: Path,
        session_id: str,
        permission_mode: str | None = None,
        execution_mode: str = "tool",
        audit_dir: Path | None = None,
        compact: bool = False,
    ) -> None:
        super().__init__()

        for theme in (_DARK, _LIGHT):
            self.register_theme(theme)
        self.theme = "godspeed-dark"

        self._settings = settings
        self._tools = registry
        self._risk_levels = risk_levels
        self._effective_model = effective_model
        self._effective_project_dir = effective_project_dir
        self._session_id = session_id
        self._permission_mode = permission_mode
        self._execution_mode = execution_mode
        self._audit_dir = audit_dir
        self._compact = compact

        self._splash = None
        self._llm_client = None
        self._tool_registry = registry
        self._tool_context = None
        self._tool_context_stored = None
        self._conversation = None
        self._permission_engine = None
        self._audit_trail = None
        self._correction_tracker = None
        self._session_memory = None
        self._hook_executor = None
        self._skill_dream = None
        self._commands = None
        self._chat_screen = None
        self._approval_tracker = None
        self._diff_reviewer = None

        self._turn_count = 0
        self._tool_calls = 0
        self._tool_errors = 0
        self._tool_denied = 0
        self._start_time = time.monotonic()

    def on_mount(self: Any) -> None:
        import asyncio

        from godspeed.tui.screens.splash import SplashScreen

        splash = SplashScreen()
        self.push_screen(splash)
        self._splash = splash
        self._init_task = asyncio.create_task(self._init())

    async def _init(self: Any) -> None:
        splash = self._splash
        if splash is None:
            return

        def _status(text: str) -> None:
            if splash is not None:
                splash.update_status(text)

        try:
            _status("Loading settings...")
            await asyncio.sleep(0)

            from godspeed.agent.conversation import Conversation
            from godspeed.agent.system_prompt import build_system_prompt
            from godspeed.audit.trail import AuditTrail
            from godspeed.context.project_instructions import load_project_instructions
            from godspeed.llm.client import LLMClient
            from godspeed.security.permissions import PermissionEngine
            from godspeed.tools.base import ToolContext
            from godspeed.tools.tasks import TaskStore, TaskTool

            settings = self._settings
            effective_model = self._effective_model
            effective_project_dir = self._effective_project_dir
            session_id = self._session_id

            set_compact_mode(self._compact)

            _status("Building tool registry...")
            await asyncio.sleep(0)
            task_store = TaskStore()
            task_tool = TaskTool(task_store)
            self._tools.register(task_tool)
            self._risk_levels[task_tool.name] = task_tool.risk_level

            _status("Setting up permissions...")
            await asyncio.sleep(0)
            permission_engine = PermissionEngine(
                deny_patterns=settings.permissions.deny,
                allow_patterns=settings.permissions.allow,
                ask_patterns=settings.permissions.ask,
                tool_risk_levels=self._risk_levels,
            )
            self._permission_engine = permission_engine

            audit_trail = None
            if settings.audit.enabled:
                _status("Initializing audit trail...")
                await asyncio.sleep(0)
                effective_audit_dir = self._audit_dir or (settings.global_dir / "audit")
                audit_trail = AuditTrail(log_dir=effective_audit_dir, session_id=session_id)
                audit_trail.record(
                    event_type="session_start",
                    detail={"model": effective_model, "project_dir": str(effective_project_dir)},
                )
                audit_trail.cleanup_expired(settings.audit.retention_days)
            self._audit_trail = audit_trail

            conversation_logger = None
            if settings.log_conversations:
                from godspeed.training.conversation_logger import ConversationLogger
                training_dir = settings.global_dir / "training"
                conversation_logger = ConversationLogger(
                    session_id=session_id, output_dir=training_dir
                )

            memory_hints = ""
            correction_tracker = None
            session_memory = None
            if settings.memory_enabled:
                _status("Loading memory...")
                await asyncio.sleep(0)
                from godspeed.memory.corrections import CorrectionTracker
                from godspeed.memory.session import SessionMemory
                from godspeed.memory.user_memory import UserMemory
                db_path = settings.global_dir / "memory.db"
                user_memory = UserMemory(db_path=db_path)
                correction_tracker = CorrectionTracker(user_memory)
                session_memory = SessionMemory(db_path=db_path)
                session_memory.start_session(
                    session_id, effective_model, str(effective_project_dir)
                )
                prefs = user_memory.list_preferences()
                corrections = correction_tracker.format_for_system_prompt(n=5)
                if prefs or corrections:
                    parts = []
                    if prefs:
                        parts.append("User preferences:")
                        for p in prefs:
                            parts.append(f"- {p['key']}: {p['value']}")
                    if corrections:
                        parts.append(corrections)
                    memory_hints = "\n".join(parts)
            self._correction_tracker = correction_tracker
            self._session_memory = session_memory

            _status("Building system prompt...")
            await asyncio.sleep(0)
            project_instructions = load_project_instructions(
                effective_project_dir, settings.context.project_instructions
            )
            system_prompt = build_system_prompt(
                tools=self._tools.list_tools(),
                project_instructions=project_instructions,
                cwd=effective_project_dir,
                execution_mode=self._execution_mode,
                memory_hints=memory_hints or None,
            )

            _status("Connecting LLM client...")
            await asyncio.sleep(0)
            from godspeed.llm.client import ModelRouter
            router = ModelRouter(routing=settings.routing) if settings.routing else None
            llm_client = LLMClient(
                model=effective_model,
                fallback_models=settings.fallback_models,
                router=router,
                thinking_budget=settings.thinking_budget,
                max_cost_usd=settings.max_cost_usd,
            )
            self._llm_client = llm_client

            tool_context = ToolContext(
                cwd=effective_project_dir,
                session_id=session_id,
                permissions=permission_engine,
                audit=audit_trail,
                llm_client=llm_client,
            )
            self._tool_context = tool_context
            self._tool_context_stored = tool_context

            conversation = Conversation(
                system_prompt=system_prompt,
                model=effective_model,
                max_tokens=settings.max_context_tokens,
                compaction_threshold=settings.compaction_threshold,
                conversation_logger=conversation_logger,
            )
            self._conversation = conversation

            from godspeed.tui.commands import CommandResult, Commands
            self._commands = Commands(
                conversation=conversation,
                llm_client=llm_client,
                permission_engine=permission_engine,
                audit_trail=audit_trail,
                session_id=session_id,
                cwd=tool_context.cwd,
            )
            self._commands._task_store = task_store
            self._commands._codebase_index = None

            def _cmd_theme(_args):
                self.action_toggle_theme()
                return CommandResult(handled=True)
            self._commands.register("theme", _cmd_theme)

            if permission_engine is not None:
                from godspeed.security.approval_tracker import ApprovalTracker
                self._approval_tracker = ApprovalTracker()
                tool_context.permissions = _InteractivePermissionProxy(
                    permission_engine, self, approval_tracker=self._approval_tracker,
                )
            self._diff_reviewer = _InteractiveDiffReviewer(self)
            tool_context.diff_reviewer = self._diff_reviewer

            from godspeed.tui.screens.chat import ChatScreen
            self._chat_screen = ChatScreen(
                llm_client=llm_client,
                tool_registry=self._tools,
                tool_context=tool_context,
                conversation=conversation,
                permission_engine=permission_engine,
                audit_trail=audit_trail,
                session_id=session_id,
                commands=self._commands,
                hook_executor=None,
                correction_tracker=correction_tracker,
            )
            self.push_screen(self._chat_screen)
            _status("Ready")
            await asyncio.sleep(0)

            # Background: start inference server, connect tools, load skills
            self._bg_task = asyncio.create_task(
                self._init_background(
                    effective_model, settings, effective_project_dir,
                    conversation, llm_client,
                )
            )

        except Exception as exc:
            logger.exception("Init failed: %s", exc)
            _status(f"Error: {exc}")

    def action_toggle_theme(self: Any) -> None:
        self.theme = (
            "godspeed-light" if self.theme == "godspeed-dark" else "godspeed-dark"
        )

    async def _init_background(
        self,
        effective_model: str,
        settings: Any,
        effective_project_dir: Any,
        conversation: Any,
        llm_client: Any,
    ) -> None:
        """Lazy init: inference server, MCP, skills, code index."""
        try:
            from godspeed.context.auto_index import maybe_start_auto_index
            maybe_start_auto_index(effective_project_dir, settings.auto_index)

            if effective_model.lower().startswith("ollama"):
                from godspeed.cli import _ensure_ollama
                await asyncio.to_thread(_ensure_ollama)
            elif effective_model.lower().startswith(("llamacpp/", "openai/")):
                from godspeed.cli import _ensure_llamacpp
                await asyncio.to_thread(_ensure_llamacpp)

            if settings.mcp_servers:
                from godspeed.mcp.client import MCPClient, MCPServerConfig
                from godspeed.mcp.tool_adapter import adapt_mcp_tools
                mcp_client = MCPClient()
                if mcp_client.available:
                    for server_cfg in settings.mcp_servers:
                        config = MCPServerConfig(
                            name=server_cfg.get("name", "unknown"),
                            command=server_cfg.get("command", ""),
                            args=server_cfg.get("args", []),
                            env=server_cfg.get("env", {}),
                            transport=server_cfg.get("transport", "stdio"),
                            url=server_cfg.get("url"),
                            headers=server_cfg.get("headers"),
                        )
                        try:
                            definitions = await asyncio.wait_for(
                                mcp_client.connect(config), timeout=10.0
                            )
                            for tool in adapt_mcp_tools(definitions, mcp_client):
                                self._tools.register(tool)
                                self._risk_levels[tool.name] = tool.risk_level
                        except TimeoutError:
                            logger.warning("MCP server %s timed out", config.name)
                        except Exception as exc:
                            logger.warning("MCP server %s failed: %s", config.name, exc)

            from godspeed.context.codebase_index import CodebaseIndex
            idx = CodebaseIndex(project_dir=effective_project_dir)
            if idx.is_available:
                from godspeed.tools.code_search import CodeSearchTool
                code_search_tool = CodeSearchTool(idx)
                self._tools.register(code_search_tool)
                self._risk_levels[code_search_tool.name] = code_search_tool.risk_level
                if idx.needs_reindex():
                    self._reindex_task = asyncio.create_task(idx.build_index_async())

            from godspeed.skills.dream import SkillDream
            from godspeed.skills.evolution import SkillEvolution
            from godspeed.skills.loader import SkillHub, discover_skills
            skill_dirs = [
                settings.global_dir / "skills",
                effective_project_dir / ".godspeed" / "skills",
            ]
            skills = discover_skills(skill_dirs)
            skills_dir = Path.home() / ".godspeed" / "skills"
            self._skill_dream = SkillDream()

            hook_executor = None
            if settings.hooks:
                from godspeed.hooks.config import HookDefinition
                from godspeed.hooks.executor import HookExecutor
                hook_defs = [HookDefinition(**h) for h in settings.hooks]
                hook_executor = HookExecutor(
                    hooks=hook_defs, cwd=effective_project_dir,
                    session_id=self._session_id,
                )
                hook_executor.run_pre_session()

            if self._chat_screen:
                self._chat_screen._hook_executor = hook_executor

            if skills:
                from godspeed.skills.commands import register_skill_commands
                register_skill_commands(
                    self._commands, conversation, skills,
                    evolution=SkillEvolution(), hub=SkillHub(),
                    dream=self._skill_dream, skills_dir=skills_dir,
                    llm_client=llm_client,
                )

            from godspeed.tui.app import _schedule_dream
            _schedule_dream(self._skill_dream)

        except Exception as exc:
            logger.exception("Background init failed: %s", exc)

    async def action_quit(self: Any) -> None:
        duration = time.monotonic() - self._start_time
        format_session_summary(
            duration_secs=duration,
            input_tokens=self._llm_client.total_input_tokens,
            output_tokens=self._llm_client.total_output_tokens,
            tool_calls=self._tool_calls,
            tool_errors=self._tool_errors,
            tool_denied=self._tool_denied,
            model=self._llm_client.model,
            session_id=self._session_id,
        )

        if self._session_memory is not None:
            self._session_memory.end_session(
                self._session_id,
                summary=(
                    f"turns={self._turn_count}"
                    f" tools={self._tool_calls}"
                    f" errors={self._tool_errors}"
                ),
            )
        if self._audit_trail is not None:
            self._audit_trail.record(
                event_type="session_end",
                detail={"reason": "user_quit"},
            )

        self.exit()


class _InteractivePermissionProxy:
    """Wraps PermissionEngine to intercept ASK decisions with an interactive prompt."""

    def __init__(
        self,
        engine: PermissionEngine,
        app: Any,
        approval_tracker: Any | None = None,
    ) -> None:
        self._engine = engine
        self._app = app
        self._tracker = approval_tracker

    async def evaluate(self, tool_call: Any) -> PermissionDecision:
        decision = self._engine.evaluate(tool_call)
        if decision != ASK:
            return decision

        args = getattr(tool_call, "arguments", None) or {}
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        try:
            answer = await self._app.push_screen(
                PermissionDialog(tool_call.tool_name, decision.reason, arguments=args),
                wait_for_dismiss=True,
            )
        except Exception:
            answer = "no"

        if answer in ("y", "yes"):
            pattern = tool_call.format_for_permission
            if self._tracker is not None:
                self._tracker.record_approval(pattern)
                if self._tracker.should_suggest(pattern):
                    self._suggest_auto_permission(pattern)
            return PermissionDecision(ALLOW, "user approved")

        if answer in ("a", "always"):
            pattern = tool_call.format_for_permission
            risk = self._engine._tool_risk_levels.get(
                tool_call.tool_name, RiskLevel.HIGH
            )
            if risk == RiskLevel.LOW:
                self._engine.grant_tool_session_permission(tool_call.tool_name)
                if self._tracker is not None:
                    self._tracker.record_approval(f"{tool_call.tool_name}(*)")
                return PermissionDecision(
                    ALLOW, f"session grant: {tool_call.tool_name}(*)"
                )
            self._engine.grant_session_permission(pattern)
            return PermissionDecision(ALLOW, f"session grant: {pattern}")

        return PermissionDecision("deny", "user denied")

    def _suggest_auto_permission(self, pattern: str) -> None:
        for rule in self._engine.allow_rules:
            if rule == pattern:
                return

        self._engine.grant_session_permission(pattern)
        try:
            from godspeed.tui.theme import BOLD_PRIMARY, DIM

            chat_view = self._app._chat_screen.query_one("#chat-log")
            chat_view.write()
            chat_view.write(
                f"  [{DIM}]Tip: run [{BOLD_PRIMARY}]/remember approve {pattern}"
                f"[/{BOLD_PRIMARY}] to auto-approve permanently.[/{DIM}]"
            )
        except Exception as exc:
            logger.debug("Could not write permission suggestion: %s", exc)


class _InteractiveDiffReviewer:
    """Implements ToolContext.DiffReviewer by prompting the human via Textual dialog."""

    def __init__(self, app: Any) -> None:
        self._always_accept = False
        self._app = app

    async def review(
        self,
        *,
        tool_name: str,
        path: str,
        before: str,
        after: str,
    ) -> str:
        if self._always_accept:
            return "accept"

        from godspeed.tui.screens.diff_review import DiffReviewDialog

        try:
            answer = await self._app.push_screen(
                DiffReviewDialog(tool_name, path, before, after),
                wait_for_dismiss=True,
            )
        except Exception:
            return "reject"

        if answer in ("y", "yes", "accept", ""):
            return "accept"
        if answer in ("a", "always"):
            self._always_accept = True
            return "accept"
        return "reject"
