"""Slash commands for the Godspeed TUI."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any, ClassVar

from godspeed.config import append_permission_rule
from godspeed.tui import output as _output
from godspeed.tui.loop_state import (
    LOOP_DEFAULT_INTERVAL_SECONDS,
    LoopState,
    is_loop_interval,
    parse_loop_interval,
)
from godspeed.tui.output import (
    format_error,
    format_info,
    format_stats,
    format_success,
    format_warning,
)
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    CTX_CRITICAL,
    CTX_OK,
    CTX_WARN,
    DIM,
    NEUTRAL,
    PERM_ALLOW,
    PERM_ASK,
    PERM_DENY,
    PERM_SESSION,
    RULE_CHAR,
    SUCCESS,
    TABLE_BORDER,
    TABLE_KEY,
    TABLE_VALUE,
    WARNING,
    styled,
)

logger = logging.getLogger(__name__)


class CommandResult:
    """Result from executing a slash command."""

    def __init__(
        self,
        handled: bool = True,
        should_quit: bool = False,
        message: str = "",
    ) -> None:
        self.handled = handled
        self.should_quit = should_quit
        self.message = message


CommandHandler = Callable[..., CommandResult]


class Commands:
    """Registry of slash commands with dispatch.

    Usage:
        commands = Commands(...)
        result = commands.dispatch("/help")
    """

    def __init__(
        self,
        conversation: Any,
        llm_client: Any,
        permission_engine: Any,
        audit_trail: Any | None,
        session_id: str,
        cwd: Path,
        pause_event: Any | None = None,
        tool_registry: Any | None = None,
    ) -> None:
        self._conversation = conversation
        self._llm_client = llm_client
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id
        self._cwd = cwd
        self._pause_event = pause_event
        self._tool_registry = tool_registry

        self.max_iterations: int | None = None  # None = use default
        self.auto_commit: bool = False
        self.auto_commit_threshold: int = 5
        self.architect_mode: bool = False
        self.whisper_mode: bool = False
        self._session_goal: str = ""
        self._loop_state = LoopState()

        self._handlers: dict[str, CommandHandler] = {}
        self._handlers["/help"] = self._cmd_help
        self._handlers["/model"] = self._cmd_model
        self._handlers["/clear"] = self._cmd_clear
        self._handlers["/undo"] = self._cmd_undo
        self._handlers["/audit"] = self._cmd_audit
        self._handlers["/permissions"] = self._cmd_permissions
        self._handlers["/remember"] = self._cmd_remember
        self._handlers["/extend"] = self._cmd_extend
        self._handlers["/context"] = self._cmd_context
        self._handlers["/compact"] = self._cmd_compact
        self._handlers["/verify"] = self._cmd_verify
        self._handlers["/batch"] = self._cmd_batch
        self._handlers["/plan"] = self._cmd_plan
        self._handlers["/checkpoint"] = self._cmd_checkpoint
        self._handlers["/restore"] = self._cmd_restore
        self._handlers["/pause"] = self._cmd_pause
        self._handlers["/resume"] = self._cmd_resume
        self._handlers["/guidance"] = self._cmd_guidance
        self._handlers["/tasks"] = self._cmd_tasks
        self._handlers["/reindex"] = self._cmd_reindex
        self._handlers["/stats"] = self._cmd_stats
        self._handlers["/usage"] = self._cmd_usage
        self._handlers["/autocommit"] = self._cmd_autocommit
        self._handlers["/architect"] = self._cmd_architect
        self._handlers["/think"] = self._cmd_think
        self._handlers["/budget"] = self._cmd_budget
        self._handlers["/evolve"] = self._cmd_evolve
        self._handlers["/export"] = self._cmd_export
        self._handlers["/quit"] = self._cmd_quit
        self._handlers["/exit"] = self._cmd_quit
        self._handlers["/review"] = self._cmd_review
        self._handlers["/sessions"] = self._cmd_sessions
        self._handlers["/skill"] = self._cmd_skill
        self._handlers["/actions"] = self._cmd_actions
        self._handlers["/scan"] = self._cmd_scan
        self._handlers["/models"] = self._cmd_models
        self._handlers["/correct"] = self._cmd_correct
        self._handlers["/preferences"] = self._cmd_preferences
        self._handlers["/tools"] = self._cmd_tools
        self._handlers["/diff"] = self._cmd_diff
        self._handlers["/whisper"] = self._cmd_whisper
        self._handlers["/btw"] = self._cmd_btw
        self._handlers["/goal"] = self._cmd_goal
        self._handlers["/rewind"] = self._cmd_rewind
        self._handlers["/fork"] = self._cmd_fork
        self._handlers["/effort"] = self._cmd_effort
        self._handlers["/loop"] = self._cmd_loop
        self._handlers["/code-review"] = self._cmd_code_review
        self._handlers["/security-review"] = self._cmd_security_review
        self._handlers["/simplify"] = self._cmd_simplify

        # Short aliases for power-user productivity
        self._handlers["/q"] = self._cmd_quit
        self._handlers["/h"] = self._cmd_help
        self._handlers["/m"] = self._cmd_model
        self._handlers["/c"] = self._cmd_clear
        self._handlers["/s"] = self._cmd_stats
        self._handlers["/u"] = self._cmd_undo
        self._handlers["/p"] = self._cmd_plan
        self._handlers["/a"] = self._cmd_audit
        self._handlers["/e"] = self._cmd_export
        self._handlers["/b"] = self._cmd_budget
        self._handlers["/x"] = self._cmd_extend
        self._handlers["/ctx"] = self._cmd_context
        self._handlers["/t"] = self._cmd_think
        self._handlers["/r"] = self._cmd_review
        self._handlers["/l"] = self._cmd_models
        self._handlers["/cp"] = self._cmd_checkpoint
        self._handlers["/rs"] = self._cmd_restore

    # External references — set after Commands init
    _task_store: Any | None = None
    _codebase_index: Any | None = None
    _message_queue: Any | None = None
    _session_memory: Any | None = None

    def register(self, name: str, handler: CommandHandler) -> None:
        """Register a custom slash command."""
        if not name.startswith("/"):
            name = "/" + name
        self._handlers[name] = handler

    def dispatch(self, raw_input: str) -> CommandResult | None:
        """Dispatch a slash command with fuzzy matching for typos.

        Returns None if input is not a command.
        """
        stripped = raw_input.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = self._handlers.get(cmd)
        if handler is None:
            from difflib import get_close_matches

            suggestions = get_close_matches(cmd, list(self._handlers), n=3, cutoff=0.5)
            if suggestions:
                names = ", ".join(suggestions)
                format_info(f"Unknown: {cmd}. Did you mean: {names}?")
            else:
                format_error(f"Unknown command: {cmd}. Type /help for available commands.")
            return CommandResult(handled=True)

        return handler(args)

    # -- Built-in command handlers ------------------------------------------------

    def _cmd_help(self, _args: str = "") -> CommandResult:
        """Show available commands — grouped by category."""
        rule = styled(RULE_CHAR * 40, NEUTRAL)
        _output.console.print()
        _output.console.print(f"  {styled('Commands', BOLD_PRIMARY)}")
        _output.console.print(f"  {rule}")

        groups: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Session",
                [
                    ("/model [name|preset]", "Show or switch the active model (supports presets)"),
                    ("/models", "List installed Ollama models and presets"),
                    ("/scan", "Scan hardware and recommend optimal models"),
                    ("/clear", "Clear conversation history"),
                    ("/stats", "Show token usage and estimated cost"),
                    (r"/usage \[scope]", "Show usage breakdown (tokens, tools, agents)"),
                    ("/export [name]", "Export conversation as markdown"),
                    ("/correct <msg>", "Record a correction for future sessions"),
                    ("/preferences", "Show stored user preferences"),
                    ("/tools", "List available tools with descriptions"),
                    ("/diff", "Show git diff of all session changes"),
                    ("/quit, /exit", "Exit Godspeed"),
                ],
            ),
            (
                "Agent Control",
                [
                    ("/plan", "Toggle plan mode (read-only)"),
                    ("/extend [N]", "Set max iterations per turn"),
                    ("/autocommit [on|off|N]", "Toggle auto-commit or set threshold"),
                    ("/architect", "Toggle architect mode (plan then execute)"),
                    ("/think [budget]", "Toggle extended thinking or set token budget"),
                    ("/budget [amount]", "Show/set cost budget in USD"),
                    ("/evolve [cmd]", "Self-evolution: status|run|history|rollback|review"),
                    ("/verify [instructions]", "Build, launch, and observe the project"),
                    ("/batch [units=N] <goal>", "Decompose a task into parallel worktree units"),
                    ("/pause", "Pause the agent loop"),
                    ("/resume", "Resume a paused agent"),
                    ("/guidance <msg>", "Inject guidance and resume"),
                    ("/btw <question>", "Side-question without corrupting conversation"),
                    ("/goal [text|clear]", "Set, show, or clear the session goal"),
                    ("/effort low|medium|high", "Set reasoning effort (read at call time)"),
                    (r"/loop \[interval] <prompt>", "prompt loops on interval"),
                ],
            ),
            (
                "Context",
                [
                    ("/context", "Show context window usage"),
                    ("/compact [instructions]", "Manually compact the conversation"),
                    ("/rewind", "Rewind picker: restore conversation/files/checkpoints"),
                    ("/fork [label]", "Duplicate the session for later resume"),
                    ("/checkpoint [name]", "Save/list checkpoints"),
                    ("/restore <name>", "Restore a checkpoint"),
                    ("/tasks", "Show task list"),
                    ("/reindex", "Rebuild codebase search index"),
                ],
            ),
            (
                "Review",
                [
                    ("/code-review [--fix]", "Review the working-tree diff for bugs and cleanups"),
                    ("/security-review", "Security scan: deterministic secrets scan + LLM review"),
                    ("/simplify", "Cleanup-only review (dead code, duplication, naming)"),
                ],
            ),
            (
                "Security",
                [
                    ("/audit", "Show audit trail and verify chain"),
                    ("/permissions", "Show permission rules"),
                    ("/remember <act> <pat>", "Persist a permission rule to settings.yaml"),
                    ("/undo", "Undo last git commit"),
                ],
            ),
        ]

        for group_name, cmds in groups:
            _output.console.print()
            _output.console.print(f"  {styled(group_name, NEUTRAL)}")
            for cmd_name, desc in cmds:
                _output.console.print(
                    f"    {styled(cmd_name, BOLD_PRIMARY):28s} {styled(desc, DIM)}"
                )

        _output.console.print()
        return CommandResult(handled=True)

    def _cmd_model(self, args: str = "") -> CommandResult:
        """Show or switch the active model. Supports preset names."""
        from godspeed.config import GodspeedSettings

        arg = args.strip()

        if arg:
            preset_models = GodspeedSettings.MODEL_PRESETS
            resolved = preset_models.get(arg.lower())

            if resolved:
                old_model = self._llm_client.model
                self._llm_client.model = resolved
                format_success(
                    f"Model switched: [{NEUTRAL}]{old_model}[/{NEUTRAL}]"
                    f" -> [{BOLD_PRIMARY}]{resolved}[/{BOLD_PRIMARY}]"
                    f"  [{DIM}](preset: {arg.lower()})[/{DIM}]"
                )

                if resolved.lower().startswith("ollama"):
                    from godspeed.tools.ollama_manager import is_model_installed

                    model_tag = resolved.removeprefix("ollama/")
                    if not is_model_installed(model_tag):
                        format_warning(
                            f"Model {model_tag!r} is not installed locally. "
                            f"Pull it with: godspeed ollama pull {model_tag}"
                        )
            else:
                old_model = self._llm_client.model
                self._llm_client.model = arg
                new_model = self._llm_client.model
                format_success(
                    f"Model switched: [{NEUTRAL}]{old_model}[/{NEUTRAL}]"
                    f" -> [{BOLD_PRIMARY}]{new_model}[/{BOLD_PRIMARY}]"
                )

                if arg.lower().startswith("ollama/"):
                    from godspeed.tools.ollama_manager import is_model_installed

                    model_tag = arg.removeprefix("ollama/")
                    if not is_model_installed(model_tag):
                        format_warning(
                            f"Model {model_tag!r} is not installed locally. "
                            f"Pull it with: godspeed ollama pull {model_tag}"
                        )
        else:
            model = self._llm_client.model
            from godspeed.config import GodspeedSettings

            presets = GodspeedSettings.MODEL_PRESETS
            matched_preset = ""
            for pname, pmodel in presets.items():
                if pmodel == model:
                    matched_preset = pname
                    break

            if matched_preset:
                format_info(
                    f"Active model: [{BOLD_PRIMARY}]{model}[/{BOLD_PRIMARY}]"
                    f"  [{DIM}](preset: {matched_preset})[/{DIM}]"
                )
            else:
                format_info(f"Active model: [{BOLD_PRIMARY}]{model}[/{BOLD_PRIMARY}]")
            if self._llm_client.fallback_models:
                fallbacks = ", ".join(self._llm_client.fallback_models)
                _output.console.print(f"    [{DIM}]Fallbacks: {fallbacks}[/{DIM}]")
            _output.console.print(f"    [{DIM}]Presets: {', '.join(presets.keys())}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_clear(self, _args: str = "") -> CommandResult:
        """Clear conversation history."""
        self._conversation.clear()
        format_info("Conversation cleared.")
        return CommandResult(handled=True)

    def _cmd_undo(self, _args: str = "") -> CommandResult:
        """Undo last git commit with git reset --soft HEAD~1."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=10,
            )
            if result.returncode != 0:
                format_error("Not a git repository or no commits to undo.")
                return CommandResult(handled=True)

            last_commit = result.stdout.strip()
            format_info(f"Undoing: {last_commit}")

            undo_result = subprocess.run(
                ["git", "reset", "--soft", "HEAD~1"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=10,
            )
            if undo_result.returncode == 0:
                format_success("Last commit undone (changes preserved in staging).")
            else:
                format_error(f"git reset failed: {undo_result.stderr.strip()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            format_error(f"Failed to run git: {exc}")

        return CommandResult(handled=True)

    def _cmd_audit(self, _args: str = "") -> CommandResult:
        """Show audit trail stats and verify chain."""
        if self._audit_trail is None:
            format_info("Audit trail is disabled.")
            return CommandResult(handled=True)

        from rich.table import Table

        table = Table(show_header=False, border_style=NEUTRAL, expand=False)
        table.add_column("Key", style=TABLE_KEY)
        table.add_column("Value", style=TABLE_VALUE)
        table.add_row("Session", self._session_id[:12] + "...")
        table.add_row("Records", str(self._audit_trail.record_count))
        table.add_row("Log file", str(self._audit_trail.log_path))

        _output.console.print(table)

        # Verify chain integrity
        is_valid, message = self._audit_trail.verify_chain()
        if is_valid:
            format_success(f"Chain integrity: VALID -- {message}")
        else:
            format_error(f"Chain integrity: BROKEN -- {message}")

        return CommandResult(handled=True)

    def _cmd_permissions(self, _args: str = "") -> CommandResult:
        """Show current permission rules."""
        from rich.table import Table

        if self._permission_engine is None:
            format_info("Permission engine not loaded.")
            return CommandResult(handled=True)

        table = Table(title="Permission Rules", border_style=WARNING, expand=False)
        table.add_column("Action", style="bold")
        table.add_column("Pattern")

        for rule in self._permission_engine.deny_rules:
            table.add_row(f"[{PERM_DENY}]DENY[/{PERM_DENY}]", rule.pattern)
        for rule in self._permission_engine.allow_rules:
            table.add_row(f"[{PERM_ALLOW}]ALLOW[/{PERM_ALLOW}]", rule.pattern)
        for rule in self._permission_engine.ask_rules:
            table.add_row(f"[{PERM_ASK}]ASK[/{PERM_ASK}]", rule.pattern)

        # Session grants
        for grant in self._permission_engine.session_grants:
            table.add_row(f"[{PERM_SESSION}]SESSION[/{PERM_SESSION}]", grant)

        _output.console.print(table)
        return CommandResult(handled=True)

    # Map user-friendly action aliases to the canonical rule tier.
    # "approve" reads more naturally than "allow" in a conversational
    # CLI — `/remember approve Shell(pytest*)`.
    _REMEMBER_ACTIONS: ClassVar[dict[str, str]] = {
        "approve": "allow",
        "allow": "allow",
        "deny": "deny",
        "ask": "ask",
    }

    def _cmd_remember(self, args: str = "") -> CommandResult:
        """Persist a permission rule to settings.yaml.

        Usage:
            /remember approve Shell(pytest *)        — global allow
            /remember deny FileWrite(*.env*)         — global deny
            /remember ask Shell(rm *)                — global ask
            /remember approve Shell(make) --project  — scope to this repo

        The rule is written to ~/.godspeed/settings.yaml (or the
        project's .godspeed/settings.yaml with --project) AND added
        to the live permission engine so it takes effect immediately —
        no restart needed.
        """

        if self._permission_engine is None:
            format_error("Permission engine not loaded — /remember unavailable.")
            return CommandResult(handled=True)

        raw = args.strip()
        if not raw:
            format_info(
                "Usage: /remember <approve|deny|ask> <Pattern> [--project]\n"
                "  e.g.  /remember approve Shell(pytest *)\n"
                "        /remember deny FileWrite(*.env*)"
            )
            return CommandResult(handled=True)

        # Parse trailing --project flag (scope selector).
        tokens = raw.split()
        scope_project = False
        if tokens and tokens[-1] == "--project":
            scope_project = True
            tokens = tokens[:-1]

        if len(tokens) < 2:
            format_error(
                "Need both an action and a pattern. Example: /remember approve Shell(pytest *)"
            )
            return CommandResult(handled=True)

        action_word = tokens[0].lower()
        action = self._REMEMBER_ACTIONS.get(action_word)
        if action is None:
            format_error(
                f"Unknown action {action_word!r}. Expected one of: approve, allow, deny, ask."
            )
            return CommandResult(handled=True)

        # Everything after the action word is the pattern — rejoin so
        # patterns with spaces like `Shell(git *)` work unquoted.
        pattern = " ".join(tokens[1:]).strip()

        # Minimal syntactic validation — full glob semantics are
        # validated by fnmatch at match time. We just require the
        # Tool(argument) shape so users get an immediate error for
        # obvious typos instead of a silently-saved unmatchable rule.
        if "(" not in pattern or not pattern.endswith(")"):
            format_error(
                f"Pattern must be in Tool(argument) form, got: {pattern!r}\n"
                "  e.g.  Shell(pytest *), FileRead(*.pem), FileWrite(.env*)"
            )
            return CommandResult(handled=True)

        # Persist to YAML.
        written_path = append_permission_rule(
            pattern=pattern,
            action=action,
            project_dir=self._cwd if scope_project else None,
        )
        if written_path is None:
            format_error(
                "Failed to write rule to settings.yaml. Check filesystem permissions or see logs."
            )
            return CommandResult(handled=True)

        # Add to the live permission engine so it takes effect this turn.
        try:
            self._permission_engine.add_rule(pattern, action)
        except ValueError as exc:
            format_error(f"Permission engine rejected rule: {exc}")
            return CommandResult(handled=True)

        scope_label = "project" if scope_project else "global"
        action_upper = action.upper()
        format_success(f"Remembered {action_upper} {pattern}  ({scope_label}: {written_path})")
        logger.info(
            "Rule persisted action=%s pattern=%s scope=%s path=%s",
            action,
            pattern,
            scope_label,
            written_path,
        )
        return CommandResult(handled=True)

    def _cmd_plan(self, _args: str = "") -> CommandResult:
        """Toggle plan mode — read-only, explore and plan only."""
        if self._permission_engine is None:
            format_error("Permission engine not loaded — cannot toggle plan mode.")
            return CommandResult(handled=True)

        self._permission_engine.plan_mode = not self._permission_engine.plan_mode
        if self._permission_engine.plan_mode:
            format_warning("Plan mode ON — read-only tools only. Use /plan again to exit.")
        else:
            format_success("Plan mode OFF — full tool access restored.")
        return CommandResult(handled=True)

    def _cmd_extend(self, args: str = "") -> CommandResult:
        """Set or show the max iterations per agent turn."""
        from godspeed.agent.loop import MAX_ITERATIONS

        if not args.strip():
            current = self.max_iterations if self.max_iterations is not None else MAX_ITERATIONS
            format_info(
                f"Max iterations: [{BOLD_PRIMARY}]{current}[/{BOLD_PRIMARY}]"
                f" (default: {MAX_ITERATIONS})"
            )
            return CommandResult(handled=True)

        try:
            value = int(args.strip())
        except ValueError:
            format_error(f"Invalid number: {args.strip()}")
            return CommandResult(handled=True)

        if value < 1:
            format_error("Max iterations must be at least 1.")
            return CommandResult(handled=True)

        self.max_iterations = value
        format_success(f"Max iterations set to [{BOLD_PRIMARY}]{value}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_autocommit(self, args: str = "") -> CommandResult:
        """Toggle auto-commit or set the file-change threshold."""
        arg = args.strip().lower()

        if arg == "on":
            self.auto_commit = True
            logger.info("autocommit toggled state=on threshold=%d", self.auto_commit_threshold)
            format_success(
                f"Auto-commit [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}]"
                f" (threshold: {self.auto_commit_threshold} files)"
            )
        elif arg == "off":
            self.auto_commit = False
            logger.info("autocommit toggled state=off")
            format_info(f"Auto-commit [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
        elif arg:
            # Numeric threshold
            try:
                value = int(arg)
            except ValueError:
                format_error(f"Invalid argument: {args.strip()}. Use on, off, or a number.")
                return CommandResult(handled=True)

            if value < 1:
                format_error("Threshold must be at least 1.")
                return CommandResult(handled=True)

            self.auto_commit_threshold = value
            self.auto_commit = True
            logger.info(
                "autocommit threshold_set threshold=%d state=on", self.auto_commit_threshold
            )
            format_success(
                f"Auto-commit [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}]"
                f" — threshold set to [{BOLD_PRIMARY}]{value}[/{BOLD_PRIMARY}] files"
            )
        else:
            # No args — toggle
            self.auto_commit = not self.auto_commit
            state = "ON" if self.auto_commit else "OFF"
            logger.info(
                "autocommit toggled state=%s threshold=%d",
                state.lower(),
                self.auto_commit_threshold,
            )
            format_info(
                f"Auto-commit [{BOLD_PRIMARY}]{state}[/{BOLD_PRIMARY}]"
                f" (threshold: {self.auto_commit_threshold} files)"
            )

        return CommandResult(handled=True)

    def _cmd_architect(self, _args: str = "") -> CommandResult:
        """Toggle architect mode — two-phase plan-then-execute."""
        self.architect_mode = not self.architect_mode
        if self.architect_mode:
            format_success(
                f"Architect mode [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}] "
                "— requests will be planned before execution"
            )
        else:
            format_info(f"Architect mode [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_whisper(self, _args: str = "") -> CommandResult:
        """Toggle whisper mode — suppress tool output during agent runs."""
        self.whisper_mode = not self.whisper_mode
        if self.whisper_mode:
            format_warning(
                f"Whisper mode [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}]"
                " — tool output hidden during agent runs"
            )
        else:
            format_success(
                f"Whisper mode [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}] — tool output visible"
            )
        return CommandResult(handled=True)

    # ------------------------------------------------------------------
    # /btw — side-question without corrupting conversation history
    # ------------------------------------------------------------------

    def _cmd_btw(self, args: str = "") -> CommandResult:
        """Answer a side-question without modifying the main conversation.

        Usage:
            /btw <question>

        Snapshots the current messages, appends the question, runs ONE
        assistant turn with a small token budget, prints the answer under
        a ``btw`` heading, then discards the snapshot.  The main
        conversation is byte-identical afterward.

        If the LLM call fails, a graceful error is printed and state is
        still restored.
        """
        import asyncio

        from rich.panel import Panel

        from godspeed.agent.aside import (
            build_btw_messages,
            snapshot_messages,
            verify_conversation_unchanged,
        )

        question = args.strip()
        if not question:
            format_error("Usage: /btw <your question>")
            return CommandResult(handled=True)

        # Deep-copy current state so we can verify nothing leaked.
        original_snapshot = snapshot_messages(self._conversation.messages)

        try:
            btw_messages = build_btw_messages(self._conversation.messages, question)
        except ValueError as exc:
            format_error(str(exc))
            return CommandResult(handled=True)

        async def _run_btw() -> None:
            try:
                response = await self._llm_client.chat(
                    messages=btw_messages,
                    tools=None,
                    task_type="chat",
                )
                answer = response.content or "(no response)"

                _output.console.print()
                panel = Panel(
                    answer,
                    title=f"[{BOLD_PRIMARY}]btw[/{BOLD_PRIMARY}]",
                    border_style=DIM,
                    expand=False,
                    padding=(0, 1),
                )
                _output.console.print(panel)
            except Exception as exc:
                logger.warning("btw LLM call failed: %s", exc)
                format_error(f"btw failed: {exc}")
            finally:
                # Safety: verify the main conversation was never touched.
                if not verify_conversation_unchanged(
                    original_snapshot, self._conversation.messages
                ):
                    logger.error("btw handler leaked into main conversation — restoring snapshot")
                    self._conversation.replace_messages([dict(m) for m in original_snapshot[1:]])

        asyncio.create_task(_run_btw())  # noqa: RUF006
        format_info("asking btw...")
        return CommandResult(handled=True)

    # ------------------------------------------------------------------
    # /goal — session goal tracking
    # ------------------------------------------------------------------

    def _cmd_goal(self, args: str = "") -> CommandResult:
        """Set, show, or clear the session goal.

        Usage:
            /goal                   — show the current goal
            /goal <text>            — set the session goal
            /goal clear             — remove the goal

        The goal is stored in-memory on the Commands instance and
        surfaces in the status HUD when set.
        """
        arg = args.strip()

        if not arg:
            # Show current goal
            if self._session_goal:
                format_info(f"Session goal: [{BOLD_PRIMARY}]{self._session_goal}[/{BOLD_PRIMARY}]")
            else:
                format_info("No session goal set. Use /goal <text> to set one.")
            return CommandResult(handled=True)

        if arg.lower() == "clear":
            self._session_goal = ""
            format_info("Session goal cleared.")
            return CommandResult(handled=True)

        self._session_goal = arg
        format_success(f"Session goal: [{BOLD_PRIMARY}]{arg}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    # ------------------------------------------------------------------
    # /effort — session reasoning effort
    # ------------------------------------------------------------------

    def _cmd_effort(self, args: str = "") -> CommandResult:
        """Set, show, or clear the session reasoning effort.

        Usage:
            /effort                       — show current effort
            /effort low|medium|high       — set effort
            /effort clear                 — reset to default
        """
        level = args.strip().lower()
        if not level:
            current = getattr(self._llm_client, "reasoning_effort", "") or "(default)"
            format_info(f"Reasoning effort: [{BOLD_PRIMARY}]{current}[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)
        if level == "clear":
            self._llm_client.reasoning_effort = ""
            format_info("Reasoning effort reset to default.")
            return CommandResult(handled=True)
        if level not in ("low", "medium", "high"):
            format_error("Usage: /effort low|medium|high|clear")
            return CommandResult(handled=True)
        self._llm_client.reasoning_effort = level
        format_success(f"Reasoning effort set to [{BOLD_PRIMARY}]{level}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    # ------------------------------------------------------------------
    # /fork — duplicate the session for later resume
    # ------------------------------------------------------------------

    def _cmd_fork(self, args: str = "") -> CommandResult:
        """Duplicate the current conversation into a resumable fork.

        Usage:
            /fork [label]

        Writes the current messages under a new session id, registers the
        fork in session memory, and leaves the live session untouched. The
        fork can be resumed later with ``godspeed --resume <id>``.
        """
        import uuid as _uuid

        if self._session_memory is None:
            format_error(
                "Session memory is unavailable in this context; fork needs "
                "a session store to register the copy."
            )
            return CommandResult(handled=True)

        label = args.strip()
        fork_id = f"{self._session_id}-fork-{_uuid.uuid4().hex[:6]}"
        messages = [dict(m) for m in self._conversation.messages]
        self._session_memory.start_session(
            fork_id, model=getattr(self._llm_client, "model", ""), project_dir=str(self._cwd)
        )
        self._session_memory.save_messages(fork_id, messages)
        summary = (
            f"Forked from session {self._session_id}"
            + (f" ({label})" if label else "")
            + f"; {len(messages)} messages."
        )
        self._session_memory.end_session(fork_id, summary=summary)
        format_success(f"Forked session: [{BOLD_PRIMARY}]{fork_id}[/{BOLD_PRIMARY}]")
        format_info(f"Resume later with: godspeed --resume {fork_id}")
        return CommandResult(handled=True)

    # ------------------------------------------------------------------
    # /loop — recurring prompt on an interval
    # ------------------------------------------------------------------

    def _cmd_loop(self, args: str = "") -> CommandResult:
        """Run a prompt on a recurring interval.

        Usage:
            /loop <prompt>            — loop every 60s
            /loop 5m <prompt>         — loop every 5 minutes
            /loop                     — show loop status
            /loop stop                — stop looping

        The prompt is re-dispatched as a user turn after each completed
        agent turn once the interval has elapsed. Note: if the loop
        prompt itself is "/loop stop", the dispatched turn will cancel
        the loop — treated as data, acceptable.
        """
        arg = args.strip()

        if not arg:
            if self._loop_state.enabled:
                format_info(
                    f"Looping every {self._loop_state.interval:g}s: "
                    f"[{BOLD_PRIMARY}]{self._loop_state.prompt}[/{BOLD_PRIMARY}]"
                )
            else:
                format_info("Not looping. Use /loop <prompt> to start.")
            return CommandResult(handled=True)

        if arg.lower() == "stop":
            self._loop_state.stop()
            format_info("Loop stopped.")
            return CommandResult(handled=True)

        try:
            interval, prompt = self._split_loop_args(arg)
        except ValueError as exc:
            format_error(str(exc))
            return CommandResult(handled=True)

        if not prompt:
            format_error("Usage: /loop [interval] <prompt>")
            return CommandResult(handled=True)

        self._loop_state.start(prompt, interval)
        format_success(f"Looping every {interval:g}s: [{BOLD_PRIMARY}]{prompt}[/{BOLD_PRIMARY}]")
        logger.info("loop started interval=%s prompt=%r", interval, prompt)
        return CommandResult(handled=True)

    def _split_loop_args(self, arg: str) -> tuple[float, str]:
        """Split '/loop [interval] <prompt>' args into (interval, prompt).

        The first whitespace-delimited token is treated as an interval
        when it is a syntactically valid spec (bare number or Ns/Nm/Nh
        suffix); otherwise the whole argument is the prompt with the
        default interval. Raises ``ValueError`` for a zero interval.
        """
        first, _, rest = arg.partition(" ")
        if is_loop_interval(first):
            interval = parse_loop_interval(first)
            return interval, rest.strip()
        return LOOP_DEFAULT_INTERVAL_SECONDS, arg

    def _maybe_dispatch_loop_turn(self, now: float | None = None) -> bool:
        """Enqueue the loop prompt if the loop is enabled and due.

        Called by the app right after draining the message queue at the
        turn-completion point. Returns True when a loop turn was
        enqueued so the app can process it immediately.

        Skips while paused: loop turns must never stall on the pause
        event. The timer is left untouched while paused, so the loop
        resumes on schedule once unpaused.
        """
        if not self._loop_state.enabled:
            return False
        if self._pause_event is not None and not self._pause_event.is_set():
            return False
        if self._message_queue is None:
            return False
        current = time.monotonic() if now is None else now
        if not self._loop_state.is_due(current):
            return False
        message = self._loop_state.mark_dispatched(current)
        self._message_queue.enqueue(message)
        logger.debug(
            "loop turn dispatched turn=%d prompt=%r",
            self._loop_state.turn_count,
            self._loop_state.prompt,
        )
        return True

    # ------------------------------------------------------------------
    # /code-review, /security-review, /simplify — diff review commands
    # ------------------------------------------------------------------

    def _run_review(
        self,
        mode: str,
        args: str,
        extra_scan: Callable[[], list[str]] | None = None,
    ) -> CommandResult:
        """Shared single-shot diff review used by the three review commands."""
        import asyncio

        from rich.panel import Panel

        from godspeed.agent.review import (
            collect_diff,
            findings_from_response,
            format_findings,
            parse_review_args,
            review_prompt,
        )

        flags, positional = parse_review_args(args)
        diff = collect_diff(self._cwd)
        if diff.error:
            format_error(diff.error)
            return CommandResult(handled=True)

        deterministic_findings = extra_scan() if extra_scan is not None else []
        title = mode.replace("-", " ").title() + " Review"

        if "--fix" not in flags and mode != "security":

            async def _run() -> None:
                try:
                    text_prompt = review_prompt(mode, diff, extra=positional)
                    response = await self._llm_client.chat(
                        messages=[{"role": "user", "content": text_prompt}],
                        tools=None,
                        task_type="verification",
                    )
                    findings = findings_from_response(response.content or "")
                    if deterministic_findings:
                        findings = deterministic_findings + findings
                    _output.console.print()
                    _output.console.print(
                        Panel(
                            format_findings(findings, "No issues found."),
                            title=f"[{BOLD_PRIMARY}] {title} [/{BOLD_PRIMARY}]",
                            border_style=DIM if not findings else "error",
                            expand=False,
                            padding=(0, 1),
                        )
                    )
                except Exception as exc:
                    logger.warning("review LLM call failed: %s", exc)
                    format_error(
                        f"{mode} review failed: {exc}. Fix findings manually or rerun "
                        "with --fix to queue an agent fix pass."
                    )

            asyncio.create_task(_run())  # noqa: RUF006
            format_info(f"Running {mode} review...")
            return CommandResult(handled=True)

        if "--fix" in flags:
            try:
                prompt = review_prompt(mode, diff)
            except ValueError as exc:
                format_error(str(exc))
                return CommandResult(handled=True)
            self._conversation.add_user_message(
                f"[User guidance]: Fix the findings from the {mode} review of the "
                f"current diff:\n{prompt}"
            )
            if deterministic_findings:
                self._conversation.add_user_message(
                    "[User guidance]: Deterministic findings to fix:\n"
                    + "\n".join(deterministic_findings)
                )
            format_success(
                "Fix instructions queued as user guidance — they run on the next agent turn."
            )
            return CommandResult(handled=True)

        async def _run_security() -> None:
            findings: list[str] = list(deterministic_findings)
            try:
                text_prompt = review_prompt(mode, diff, extra=positional)
                response = await self._llm_client.chat(
                    messages=[{"role": "user", "content": text_prompt}],
                    tools=None,
                    task_type="verification",
                )
                findings.extend(f"[review] {f}" for f in findings_from_response(response.content))
            except Exception as exc:
                logger.warning("security review LLM step failed: %s", exc)
                findings.append(
                    "[note] LLM security review unavailable; "
                    "showing deterministic secrets scan only."
                )
            _output.console.print()
            _output.console.print(
                Panel(
                    format_findings(findings, "No issues found."),
                    title=f"[{BOLD_PRIMARY}] {title} [/{BOLD_PRIMARY}]",
                    border_style=DIM if not findings else "error",
                    expand=False,
                    padding=(0, 1),
                )
            )

        asyncio.create_task(_run_security())  # noqa: RUF006
        format_info("Running security review (deterministic scan + LLM review)...")
        return CommandResult(handled=True)

    def _cmd_code_review(self, args: str = "") -> CommandResult:
        """Review the working-tree diff for bugs, risks, and cleanups."""
        return self._run_review("code", args)

    def _cmd_simplify(self, args: str = "") -> CommandResult:
        """Cleanup-only review of the working-tree diff."""
        return self._run_review("simplify", args)

    def _cmd_security_review(self, args: str = "") -> CommandResult:
        """Security scan of the working-tree diff (secrets + LLM review)."""
        from godspeed.agent.review import collect_diff, security_scan_files

        scan_diff = collect_diff(self._cwd)
        if scan_diff.error:
            format_error(scan_diff.error)
            return CommandResult(handled=True)

        def _extra_scan() -> list[str]:
            return security_scan_files(scan_diff.changed_files, self._cwd)

        return self._run_review("security", args, extra_scan=_extra_scan)

    # ------------------------------------------------------------------
    # /rewind — slash alias for the ESC-ESC rewind picker
    # ------------------------------------------------------------------

    def _cmd_rewind(self, args: str = "") -> CommandResult:
        """Rewind picker: list available checkpoints and restore.

        Usage:
            /rewind          — list checkpoints and prompt for selection
            /rewind <number> — directly select checkpoint by number

        Reuses the same ``collect_rewind_entries``, ``parse_rewind_choice``
        and restore functions that the ESC-ESC keybinding path uses,
        ensuring both entry points stay consistent.  See ``tui/rewind.py``
        for the underlying restore logic.
        """
        from godspeed.tui.rewind import (
            RESTORE_BOTH,
            RESTORE_CONVERSATION,
            RESTORE_FILES,
            RESTORE_NONE,
            collect_rewind_entries,
            parse_rewind_choice,
            restore_conversation,
            restore_files,
        )

        entries = collect_rewind_entries(self._cwd, self._session_id)
        if not entries:
            format_info("No rewind checkpoints available.")
            return CommandResult(handled=True)

        _output.console.print()
        _output.console.print(f"  {styled('Rewind Checkpoints', BOLD_PRIMARY)}")
        _output.console.print(f"  {styled(RULE_CHAR * 30, NEUTRAL)}")
        for idx, entry in enumerate(entries, 1):
            kind_label = styled(f"[{entry.kind[:3]}]", NEUTRAL)
            _output.console.print(
                f"    {styled(str(idx), BOLD_PRIMARY)}. "
                f"{kind_label} {styled(entry.name, NEUTRAL)} — {styled(entry.detail, DIM)}"
            )
        _output.console.print(f"  [{DIM}]Choose an entry number, or 0 to cancel.[/{DIM}]")

        selection = args.strip()
        if not selection:
            try:
                selection = _output.console.input(f"[{WARNING}]  > [/{WARNING}]").strip()
            except (KeyboardInterrupt, EOFError):
                format_info("Rewind cancelled.")
                return CommandResult(handled=True)

        if not selection.isdigit():
            format_info("Rewind cancelled.")
            return CommandResult(handled=True)

        idx = int(selection)
        if idx < 1 or idx > len(entries):
            format_info("Rewind cancelled.")
            return CommandResult(handled=True)

        chosen = entries[idx - 1]

        _output.console.print(
            f"  [{DIM}]Restore what? [c]onversation / [f]iles / [b]oth / [n]one[/{DIM}]"
        )
        try:
            choice = _output.console.input(f"[{WARNING}]  > [/{WARNING}]").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "n"

        action = parse_rewind_choice(choice)
        if action == RESTORE_NONE:
            format_info("Rewind cancelled.")
            return CommandResult(handled=True)

        if action in (RESTORE_CONVERSATION, RESTORE_BOTH):
            if chosen.kind == "conversation":
                summary = restore_conversation(self._conversation, chosen.name, self._cwd)
                format_success(summary)
            else:
                format_warning(
                    f"Entry {chosen.name} is a file checkpoint — no conversation to restore."
                )

        if action in (RESTORE_FILES, RESTORE_BOTH):
            summary = restore_files(self._cwd, self._session_id)
            format_success(summary)

        return CommandResult(handled=True)

    def _cmd_think(self, args: str = "") -> CommandResult:
        """Toggle extended thinking or set the thinking token budget."""
        arg = args.strip()

        if not arg:
            # Toggle: off → default 10k, on → off
            current = self._llm_client.thinking_budget
            if current > 0:
                self._llm_client.thinking_budget = 0
                format_info(f"Extended thinking [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
            else:
                self._llm_client.thinking_budget = 10_000
                format_success(
                    f"Extended thinking [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}] (budget: 10,000 tokens)"
                )
            return CommandResult(handled=True)

        if arg.lower() == "off":
            self._llm_client.thinking_budget = 0
            format_info(f"Extended thinking [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        try:
            budget = int(arg.replace(",", "").replace("_", ""))
        except ValueError:
            format_error(f"Invalid budget: {arg}. Use a number or 'off'.")
            return CommandResult(handled=True)

        if budget < 1000:
            format_error("Thinking budget must be at least 1,000 tokens.")
            return CommandResult(handled=True)

        self._llm_client.thinking_budget = budget
        format_success(
            f"Extended thinking [{BOLD_PRIMARY}]ON[/{BOLD_PRIMARY}] (budget: {budget:,} tokens)"
        )
        return CommandResult(handled=True)

    def _cmd_budget(self, args: str = "") -> CommandResult:
        """Show or set the cost budget in USD."""
        from godspeed.llm.cost import format_cost

        arg = args.strip()

        if not arg:
            # Show current cost and budget
            spent = self._llm_client.total_cost_usd
            limit = self._llm_client.max_cost_usd
            model = self._llm_client.model
            input_tokens = self._llm_client.total_input_tokens
            output_tokens = self._llm_client.total_output_tokens

            spent_str = format_cost(spent)
            if limit > 0:
                pct = spent / limit * 100
                format_info(
                    f"Cost: [{BOLD_PRIMARY}]{spent_str}[/{BOLD_PRIMARY}]"
                    f" / ${limit:.2f} ({pct:.0f}%)"
                )
            else:
                format_info(
                    f"Cost: [{BOLD_PRIMARY}]{spent_str}[/{BOLD_PRIMARY}]"
                    f" [{DIM}](no budget limit)[/{DIM}]"
                )
            _output.console.print(
                f"    [{DIM}]{input_tokens:,} input + {output_tokens:,} output tokens"
                f" ({model})[/{DIM}]"
            )
            return CommandResult(handled=True)

        if arg.lower() in ("off", "unlimited", "0"):
            self._llm_client.max_cost_usd = 0.0
            format_info(f"Cost budget [{BOLD_PRIMARY}]unlimited[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        # Strip $ prefix if present
        cleaned = arg.lstrip("$")
        try:
            limit = float(cleaned)
        except ValueError:
            format_error(f"Invalid amount: {arg}. Use a number like 5.00 or 'off'.")
            return CommandResult(handled=True)

        if limit <= 0:
            format_error("Budget must be positive. Use 'off' to disable.")
            return CommandResult(handled=True)

        self._llm_client.max_cost_usd = limit
        format_success(f"Cost budget set to [{BOLD_PRIMARY}]${limit:.2f}[/{BOLD_PRIMARY}]")
        return CommandResult(handled=True)

    def _cmd_evolve(self, args: str = "") -> CommandResult:
        """Self-evolution system commands."""
        from godspeed.evolution.registry import EvolutionRegistry

        parts = args.strip().split(None, 1)
        subcmd = parts[0] if parts else "status"

        # Use global dir for evolution storage
        evo_dir = self._cwd / ".godspeed" / "evolution"

        if subcmd == "status":
            try:
                registry = EvolutionRegistry(evo_dir)
                stats = registry.stats()
                format_info(
                    f"[{BOLD_PRIMARY}]Evolution Status[/{BOLD_PRIMARY}]\n"
                    f"  Total mutations: {stats['total_mutations']}\n"
                    f"  Active: {stats['active']}\n"
                    f"  Reverted: {stats['reverted']}\n"
                    f"  Safety passed: {stats['safety_passed']}\n"
                    f"  Safety failed: {stats['safety_failed']}\n"
                    f"  Avg fitness: {stats['avg_fitness']:.3f}"
                )
            except Exception:
                format_info(
                    "No evolution data yet. "
                    f"Run [{BOLD_PRIMARY}]/evolve run[/{BOLD_PRIMARY}] to start."
                )
            return CommandResult(handled=True)

        if subcmd == "history":
            artifact_id = parts[1] if len(parts) > 1 else ""
            if not artifact_id:
                format_error("Usage: /evolve history <artifact_id>")
                return CommandResult(handled=True)

            registry = EvolutionRegistry(evo_dir)
            history = registry.get_history(artifact_id)
            if not history:
                format_info(
                    f"No evolution history for [{BOLD_PRIMARY}]{artifact_id}[/{BOLD_PRIMARY}]"
                )
            else:
                for rec in history:
                    if rec.applied_at and not rec.reverted_at:
                        status = "active"
                    elif rec.reverted_at:
                        status = "reverted"
                    else:
                        status = "candidate"
                    format_info(
                        f"  [{DIM}]{rec.record_id}[/{DIM}] fitness={rec.fitness_overall:.3f} "
                        f"status={status} model={rec.model_used}"
                    )
            return CommandResult(handled=True)

        if subcmd == "rollback":
            record_id = parts[1] if len(parts) > 1 else ""
            if not record_id:
                format_error("Usage: /evolve rollback <record_id>")
                return CommandResult(handled=True)

            registry = EvolutionRegistry(evo_dir)
            record = registry.get_record(record_id)
            if record is None:
                format_error(f"Record not found: {record_id}")
            else:
                registry.mark_reverted(record_id)
                format_success(f"Rolled back [{BOLD_PRIMARY}]{record_id}[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        if subcmd == "review":
            registry = EvolutionRegistry(evo_dir)
            records = [
                r
                for r in registry._load_records()
                if r.safety_passed and not r.applied_at and r.requires_review
            ]
            if not records:
                format_info("No pending reviews.")
            else:
                for rec in records:
                    format_info(
                        f"  [{BOLD_PRIMARY}]{rec.record_id}[/{BOLD_PRIMARY}] "
                        f"{rec.artifact_type}:{rec.artifact_id} "
                        f"fitness={rec.fitness_overall:.3f}"
                    )
                format_info(f"\nApprove with: [{DIM}]/evolve approve <id>[/{DIM}]")
            return CommandResult(handled=True)

        if subcmd == "run":
            if self._tool_registry is None:
                format_error("Tool registry not available.")
                return CommandResult(handled=True)

            audit_dir = None
            if self._audit_trail is not None:
                audit_dir = getattr(self._audit_trail, "_log_dir", None)
            if audit_dir is None:
                format_error("Audit trail not available — evolution requires audit logs.")
                return CommandResult(handled=True)

            format_info(f"[{BOLD_PRIMARY}]Evolution run[/{BOLD_PRIMARY}] — analyzing traces...")

            import asyncio

            from godspeed.evolution.orchestrator import EvolutionOrchestrator
            from godspeed.tools.registry import ToolRegistry

            registry = self._tool_registry
            if registry is None or not isinstance(registry, ToolRegistry):
                format_error("Tool registry not available for evolution.")
                return CommandResult(handled=True)

            async def _run() -> None:
                orch = EvolutionOrchestrator(
                    tool_registry=registry,
                    audit_dir=audit_dir,
                    evolution_model=getattr(self._llm_client, "model", ""),
                    llm_client=self._llm_client,
                )
                report = await orch.run_cycle()
                if report.get("skipped"):
                    format_info(f"Evolution skipped: {report.get('reason', 'unknown')}")
                    return
                format_success(
                    f"Evolution complete: generated={report['mutations_generated']} "
                    f"applied={report['mutations_applied']} "
                    f"rejected={report['mutations_rejected']} "
                    f"cost=${report.get('cost_usd_delta', 0):.4f}"
                )
                if report.get("errors"):
                    for err in report["errors"]:
                        format_warning(f"  error: {err}")

            # Schedule in background so TUI stays responsive
            asyncio.create_task(_run())  # noqa: RUF006
            return CommandResult(handled=True)

        format_error(
            f"Unknown subcommand: {subcmd}\n  Usage: /evolve [status|run|history|rollback|review]"
        )
        return CommandResult(handled=True)

    def _cmd_context(self, _args: str = "") -> CommandResult:
        """Show context window usage with a detailed token breakdown."""
        from rich.table import Table

        from godspeed.llm.token_counter import count_message_tokens, count_tokens

        tokens = self._conversation.token_count
        max_tokens = self._conversation.max_tokens
        pct = (tokens / max_tokens * 100) if max_tokens > 0 else 0

        if pct < 50:
            color = CTX_OK
        elif pct < 80:
            color = CTX_WARN
        else:
            color = CTX_CRITICAL

        _output.console.print(
            f"  [{color}]tokens: {tokens:,} / {max_tokens:,} ({pct:.0f}%)[/{color}]"
        )
        msg_count = len(self._conversation.messages)
        _output.console.print(f"  [{DIM}]messages: {msg_count}[/{DIM}]")

        model = getattr(self._llm_client, "model", "") or self._conversation.model
        messages = self._conversation.messages
        system_tokens = count_message_tokens([messages[0]], model) if messages else 0
        conv_tokens = count_message_tokens(messages[1:], model) if len(messages) > 1 else 0

        tool_tokens = 0
        if self._tool_registry is not None:
            try:
                schemas = self._tool_registry.get_schemas()
                tool_tokens = count_tokens(str(schemas), model)
            except Exception:
                logger.warning("Failed to count tool schema tokens", exc_info=True)

        free_space = max(0, max_tokens - tokens)

        table = Table(show_header=False, border_style=NEUTRAL, expand=False, padding=(0, 2))
        table.add_column("Component", style=TABLE_KEY)
        table.add_column("Tokens", style=TABLE_VALUE, justify="right")
        table.add_row("System prompt", f"{system_tokens:,}")
        table.add_row("Tool schemas", f"{tool_tokens:,}")
        table.add_row("Conversation", f"{conv_tokens:,}")
        table.add_row("Free space", f"{free_space:,}")
        _output.console.print(table)
        return CommandResult(handled=True)

    def _cmd_compact(self, args: str = "") -> CommandResult:
        """Manually trigger conversation compaction.

        Usage:
            /compact [instructions]

        Optional instructions guide what the compaction summary preserves.
        Compaction runs in the background so the TUI stays responsive.
        """
        import asyncio

        from godspeed.context.compaction import compact_now

        instructions = args.strip()

        # Sensible minimum: don't compact tiny conversations.
        if len(self._conversation.messages) < 10:
            format_info("Nothing to compact — fewer than 10 messages.")
            return CommandResult(handled=True)

        async def _run() -> None:
            result = await compact_now(
                self._conversation,
                self._llm_client,
                instructions=instructions,
            )
            if result.applied:
                format_success(
                    f"Compacted: {result.messages_before} messages -> "
                    f"{result.messages_after} messages"
                )
            else:
                format_error("Compaction failed.")

        asyncio.create_task(_run())  # noqa: RUF006
        format_info("Compacting conversation...")
        return CommandResult(handled=True)

    def _cmd_verify(self, args: str = "") -> CommandResult:
        """Build, launch, and observe the project to confirm it actually runs.

        Usage:
            /verify [instructions]

        Optional instructions are recorded as context for the agent (they do
        not change the verification steps). The verifier runs in the
        background so the TUI stays responsive, then prints a PASS/FAIL
        verdict panel with evidence lines.
        """
        import asyncio

        from godspeed.tools.runtime_verify import RuntimeVerifier

        instructions = args.strip()
        if instructions:
            logger.info("verify.instructions=%s", instructions)

        async def _run() -> None:
            verifier = RuntimeVerifier(self._cwd)
            verdict = await asyncio.to_thread(verifier.verify)

            if verdict.passed:
                format_success(f"[{BOLD_PRIMARY}]Runtime Verify: PASS[/{BOLD_PRIMARY}]")
            else:
                format_error(f"[{BOLD_PRIMARY}]Runtime Verify: FAIL[/{BOLD_PRIMARY}]")

            for line in verdict.evidence:
                _output.console.print(f"  [{DIM}]{line}[/{DIM}]")

            if not verdict.passed:
                _output.console.print()
                format_warning(
                    "The agent can be asked to fix the failure. "
                    "Describe the issue and it will attempt a repair."
                )

        asyncio.create_task(_run())  # noqa: RUF006
        format_info("Running runtime verification...")
        return CommandResult(handled=True)

    def _cmd_batch(self, args: str = "") -> CommandResult:
        """Decompose a task into parallel units and run each in an isolated worktree.

        Usage:
            /batch [units=N] <goal>

        Each unit runs as a sub-agent in its own git worktree, up to 5 in
        parallel. Results are collected and printed when all units finish.
        """
        import asyncio

        from godspeed.agent.batch import (
            MAX_BATCH_UNITS,
            BatchGitError,
            WorktreeBatchRunner,
            decompose_task,
        )
        from godspeed.agent.coordinator import AgentCoordinator
        from godspeed.tools.base import ToolContext

        if self._tool_registry is None:
            format_error("Tool registry not available — /batch requires tools.")
            return CommandResult(handled=True)

        num_units = 5
        goal = args.strip()
        match = re.match(r"^units=(\d+)\s+(.+)$", goal, re.DOTALL)
        if match:
            num_units = int(match.group(1))
            goal = match.group(2).strip()
        if not goal:
            format_error("Usage: /batch [units=N] <goal>")
            return CommandResult(handled=True)
        if num_units > MAX_BATCH_UNITS:
            format_error(f"units must be at most {MAX_BATCH_UNITS}.")
            return CommandResult(handled=True)

        tool_context = ToolContext(
            cwd=self._cwd,
            session_id=self._session_id,
            permissions=self._permission_engine,
            audit=self._audit_trail,
            llm_client=self._llm_client,
        )
        coordinator = AgentCoordinator(
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
            tool_context=tool_context,
        )

        async def _run() -> None:
            try:
                plan = await decompose_task(
                    goal,
                    num_units_hint=num_units,
                    llm_client=self._llm_client,
                )
            except ValueError as exc:
                format_error(f"Batch decomposition failed: {exc}")
                return
            format_info(f"Batch plan: {len(plan.units)} unit(s) — running in isolated worktrees.")
            runner = WorktreeBatchRunner(
                working_dir=self._cwd,
                coordinator=coordinator,
            )
            try:
                results = await runner.run(plan)
            except BatchGitError as exc:
                format_error(f"Batch aborted: {exc}")
                return
            for result in results:
                if result.ok:
                    format_success(f"[batch] {result.id}: done — {result.summary[:120]}")
                else:
                    format_error(f"[batch] {result.id}: FAILED — {result.summary[:120]}")
                    if result.worktree_path is not None:
                        format_warning(
                            f"[batch] {result.id}: worktree left at {result.worktree_path}"
                        )
            ok_count = sum(1 for r in results if r.ok)
            format_info(f"Batch finished: {ok_count}/{len(results)} units succeeded.")

        asyncio.create_task(_run())  # noqa: RUF006
        format_info("Starting batch...")
        return CommandResult(handled=True)

    def _cmd_checkpoint(self, args: str = "") -> CommandResult:
        """Save a checkpoint or list available checkpoints."""
        from godspeed.context.checkpoint import list_checkpoints, save_checkpoint

        name = args.strip()

        if not name or name == "list":
            # List checkpoints
            checkpoints = list_checkpoints(self._cwd)
            if not checkpoints:
                format_info("No checkpoints saved yet.")
                return CommandResult(handled=True)

            from datetime import datetime

            from rich.table import Table

            table = Table(title="Checkpoints", border_style=TABLE_BORDER, expand=False)
            table.add_column("Name", style=BOLD_PRIMARY)
            table.add_column("Time", style=NEUTRAL)
            table.add_column("Model")
            table.add_column("Tokens", justify="right")
            table.add_column("Messages", justify="right")

            for cp in checkpoints:
                ts = datetime.fromtimestamp(cp["timestamp"], tz=UTC)
                table.add_row(
                    cp["name"],
                    ts.strftime("%Y-%m-%d %H:%M"),
                    cp["model"],
                    f"{cp['token_count']:,}",
                    str(cp["message_count"]),
                )

            _output.console.print(table)
            return CommandResult(handled=True)

        # Save checkpoint
        system_msg = self._conversation.messages[0]
        system_prompt = system_msg.get("content", "")
        # Messages excluding system prompt
        messages = self._conversation.messages[1:]

        path = save_checkpoint(
            name=name,
            system_prompt=system_prompt,
            messages=messages,
            model=self._llm_client.model,
            token_count=self._conversation.token_count,
            project_dir=self._cwd,
        )
        format_success(
            f"Checkpoint saved: [{BOLD_PRIMARY}]{name}[/{BOLD_PRIMARY}]  [{DIM}]{path}[/{DIM}]"
        )
        return CommandResult(handled=True)

    def _cmd_restore(self, args: str = "") -> CommandResult:
        """Restore a saved checkpoint."""
        from godspeed.context.checkpoint import load_checkpoint

        name = args.strip()
        if not name:
            format_error("Usage: /restore <name>")
            return CommandResult(handled=True)

        data = load_checkpoint(name, self._cwd)
        if data is None:
            format_error(f"Checkpoint not found: {name}")
            return CommandResult(handled=True)

        # Restore conversation state
        self._conversation.clear()
        for msg in data.get("messages", []):
            role = msg.get("role", "")
            if role == "user":
                self._conversation.add_user_message(msg.get("content", ""))
            elif role == "assistant":
                self._conversation.add_assistant_message(
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls"),
                )
            elif role == "tool":
                self._conversation.add_tool_result(
                    tool_call_id=msg.get("tool_call_id", ""),
                    content=msg.get("content", ""),
                )

        token_count = self._conversation.token_count
        msg_count = len(self._conversation.messages) - 1  # exclude system prompt
        format_success(
            f"Restored checkpoint: [{BOLD_PRIMARY}]{name}[/{BOLD_PRIMARY}]"
            f" ({msg_count} messages, {token_count:,} tokens)"
        )
        return CommandResult(handled=True)

    def _cmd_pause(self, _args: str = "") -> CommandResult:
        """Pause the agent loop at the next iteration."""
        if self._pause_event is None:
            format_error("Pause/resume not available in this session.")
            return CommandResult(handled=True)

        self._pause_event.clear()
        format_warning("Agent paused. Use /resume or /guidance <msg>.")
        return CommandResult(handled=True)

    def _cmd_resume(self, _args: str = "") -> CommandResult:
        """Resume a paused agent loop."""
        if self._pause_event is None:
            format_error("Pause/resume not available in this session.")
            return CommandResult(handled=True)

        if self._pause_event.is_set():
            format_info("Agent is not paused.")
            return CommandResult(handled=True)

        self._pause_event.set()
        format_success("Agent resumed.")
        return CommandResult(handled=True)

    def _cmd_guidance(self, args: str = "") -> CommandResult:
        """Inject guidance as a user message and resume the paused agent."""
        if not args.strip():
            format_error("Usage: /guidance <your guidance message>")
            return CommandResult(handled=True)

        # Inject guidance into conversation
        self._conversation.add_user_message(f"[User guidance]: {args.strip()}")
        format_info(f"Guidance injected: {args.strip()}")

        # Resume if paused
        if self._pause_event is not None and not self._pause_event.is_set():
            self._pause_event.set()
            format_success("Agent resumed with guidance.")

        return CommandResult(handled=True)

    def _cmd_tasks(self, _args: str = "") -> CommandResult:
        """Show current task list."""
        if self._task_store is None:
            format_info("Task tracking not enabled.")
            return CommandResult()

        tasks = self._task_store.list_all()
        if not tasks:
            format_info("No tasks.")
            return CommandResult()

        from rich.table import Table

        table = Table(title="Tasks", border_style=TABLE_BORDER, expand=False)
        table.add_column("ID", style=BOLD_PRIMARY, width=4)
        table.add_column("Title")
        table.add_column("Status")

        for t in tasks:
            if t.status == "completed":
                status_style = f"[{SUCCESS}]{t.status}[/{SUCCESS}]"
            elif t.status == "in_progress":
                status_style = f"[{WARNING}]{t.status}[/{WARNING}]"
            else:
                status_style = f"[{NEUTRAL}]{t.status}[/{NEUTRAL}]"
            table.add_row(str(t.id), t.title, status_style)

        _output.console.print(table)
        return CommandResult()

    def _cmd_reindex(self, _args: str = "") -> CommandResult:
        """Rebuild the codebase search index."""
        if self._codebase_index is None:
            format_info("Codebase index not available.")
            _output.console.print(f"  [{DIM}]Install with: pip install godspeed[index][/{DIM}]")
            return CommandResult()

        if not self._codebase_index.is_available:
            format_error(f"ChromaDB not installed. [{DIM}]pip install godspeed[index][/{DIM}]")
            return CommandResult()

        if self._codebase_index.is_building:
            format_warning("Index is already building...")
            return CommandResult()

        import asyncio

        format_info("Rebuilding codebase index...")
        asyncio.get_event_loop().create_task(self._codebase_index.build_index_async())
        format_success("Reindex started in background.")
        return CommandResult()

    def _cmd_stats(self, _args: str = "") -> CommandResult:
        """Show session statistics including token usage and estimated cost."""
        from godspeed.llm.cost import estimate_cost

        input_tokens = self._llm_client.total_input_tokens
        output_tokens = self._llm_client.total_output_tokens
        cost = estimate_cost(self._llm_client.model, input_tokens, output_tokens)

        format_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._llm_client.model,
            session_id=self._session_id,
            cost=cost if cost > 0 else None,
        )
        return CommandResult(handled=True)

    def _cmd_usage(self, args: str = "") -> CommandResult:
        """Show session usage breakdown: tokens, tools, and sub-agents.

        Usage:
            /usage            — session totals + per-task-type breakdown
            /usage tools      — tool-call counts from the audit trail
            /usage agents     — sub-agent invocations (if instrumented)

        Only measurable data is shown. Dimensions the codebase does not
        track (e.g. per-task-type token attribution, per-subagent tokens)
        are stated as gaps rather than fabricated.
        """
        from rich.panel import Panel
        from rich.table import Table

        from godspeed.llm.cost import format_cost
        from godspeed.observability.usage_report import from_client

        scope = args.strip().lower()

        if scope == "tools":
            return self._usage_tools()

        if scope == "agents":
            return self._usage_agents()

        if scope:
            format_error(f"Unknown scope: {scope}. Use /usage, /usage tools, or /usage agents.")
            return CommandResult(handled=True)

        ledger = getattr(self._llm_client, "usage_ledger", None)
        report = from_client(self._llm_client, ledger)

        _output.console.print()
        _output.console.print(f"  {styled('Session Usage', BOLD_PRIMARY)}")
        _output.console.print(f"  {styled(RULE_CHAR * 30, NEUTRAL)}")

        summary = Table(show_header=False, border_style=NEUTRAL, expand=False, padding=(0, 2))
        summary.add_column("Metric", style=TABLE_KEY)
        summary.add_column("Value", style=TABLE_VALUE, justify="right")
        summary.add_row("Input tokens", f"{report.total_input_tokens:,}")
        summary.add_row("Output tokens", f"{report.total_output_tokens:,}")
        summary.add_row("Total tokens", f"{report.total_tokens:,}")
        summary.add_row("Estimated cost", format_cost(report.total_cost_usd))
        _output.console.print(summary)

        if report.by_task_type:
            tt_table = Table(
                title="By Task Type",
                border_style=TABLE_BORDER,
                expand=False,
            )
            tt_table.add_column("Task Type", style=BOLD_PRIMARY)
            tt_table.add_column("Calls", justify="right")
            tt_table.add_column("Input", justify="right")
            tt_table.add_column("Output", justify="right")
            tt_table.add_column("Cost", justify="right")
            for task_type, row in sorted(report.by_task_type.items()):
                tt_table.add_row(
                    task_type,
                    str(row.calls),
                    f"{row.input_tokens:,}",
                    f"{row.output_tokens:,}",
                    format_cost(row.cost_usd),
                )
            _output.console.print(tt_table)
        else:
            _output.console.print(
                Panel(
                    "No task-type calls recorded yet — attribution appears "
                    "after the first LLM call in this session.",
                    title="Task Type Breakdown",
                    border_style=DIM,
                    expand=False,
                    padding=(0, 1),
                )
            )

        return CommandResult(handled=True)

    def _usage_tools(self) -> CommandResult:
        """Render the ``tools`` scope: tool-call counts from the audit trail."""
        from rich.panel import Panel
        from rich.table import Table

        from godspeed.llm.cost import format_cost
        from godspeed.observability.usage_report import from_audit

        if self._audit_trail is None:
            format_info("Audit trail is disabled — no tool-call data available.")
            return CommandResult(handled=True)

        report = from_audit(self._audit_trail)

        _output.console.print()
        _output.console.print(f"  {styled('Tool Usage', BOLD_PRIMARY)}")
        _output.console.print(f"  {styled(RULE_CHAR * 30, NEUTRAL)}")

        if not report.by_tool:
            _output.console.print(
                Panel(
                    "No tool calls recorded in the audit trail yet.",
                    title="Tool Calls",
                    border_style=DIM,
                    expand=False,
                    padding=(0, 1),
                )
            )
            return CommandResult(handled=True)

        table = Table(title="Tool Calls", border_style=TABLE_BORDER, expand=False)
        table.add_column("Tool", style=BOLD_PRIMARY)
        table.add_column("Calls", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Cost", justify="right")
        for tool_name, row in sorted(report.by_tool.items(), key=lambda kv: -kv[1].calls):
            table.add_row(
                tool_name,
                str(row.calls),
                f"{row.input_tokens:,}",
                f"{row.output_tokens:,}",
                format_cost(row.cost_usd),
            )
        _output.console.print(table)

        _output.console.print(
            Panel(
                "Per-tool token/cost attribution is not recorded in the audit "
                "trail — only call counts are measurable.",
                title="Note",
                border_style=DIM,
                expand=False,
                padding=(0, 1),
            )
        )
        return CommandResult(handled=True)

    def _usage_agents(self) -> CommandResult:
        """Render the ``agents`` scope: sub-agent invocations and cost share."""
        from rich.panel import Panel
        from rich.table import Table

        from godspeed.llm.cost import format_cost
        from godspeed.observability.usage_report import SubagentRow, from_audit

        _output.console.print()
        _output.console.print(f"  {styled('Sub-Agent Usage', BOLD_PRIMARY)}")
        _output.console.print(f"  {styled(RULE_CHAR * 30, NEUTRAL)}")

        sub_cost = getattr(self._llm_client, "total_sub_agent_cost", None)
        if not isinstance(sub_cost, (int, float)):
            sub_cost = 0.0

        if sub_cost > 0:
            table = Table(show_header=False, border_style=NEUTRAL, expand=False, padding=(0, 2))
            table.add_column("Metric", style=TABLE_KEY)
            table.add_column("Value", style=TABLE_VALUE, justify="right")
            table.add_row("Aggregate sub-agent cost", format_cost(float(sub_cost)))
            _output.console.print(table)

        by_subagent: dict[str, SubagentRow] = {}
        ledger = getattr(self._llm_client, "usage_ledger", None)
        if ledger is not None and hasattr(ledger, "by_subagent"):
            for agent_name, row in ledger.by_subagent().items():
                if agent_name == "parent":
                    continue
                by_subagent[agent_name] = SubagentRow(
                    calls=row.calls,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    cost_usd=row.cost_usd,
                )
        if not by_subagent and self._audit_trail is not None:
            by_subagent = from_audit(self._audit_trail).by_subagent

        if by_subagent:
            sa_table = Table(title="Sub-Agents", border_style=TABLE_BORDER, expand=False)
            sa_table.add_column("Session", style=BOLD_PRIMARY)
            sa_table.add_column("Calls", justify="right")
            sa_table.add_column("Input Tokens", justify="right")
            sa_table.add_column("Output Tokens", justify="right")
            sa_table.add_column("Cost", justify="right")
            for agent_name, row in sorted(by_subagent.items()):
                sa_table.add_row(
                    agent_name,
                    str(row.calls),
                    f"{row.input_tokens:,}",
                    f"{row.output_tokens:,}",
                    format_cost(row.cost_usd),
                )
            _output.console.print(sa_table)
            return CommandResult(handled=True)

        _output.console.print(
            Panel(
                "No per-subagent usage recorded in this session: either no "
                "sub-agents have run, or none made instrumented LLM calls.",
                title="Note",
                border_style=DIM,
                expand=False,
                padding=(0, 1),
            )
        )
        return CommandResult(handled=True)

    def _cmd_export(self, args: str = "") -> CommandResult:
        """Export the current conversation as a markdown file."""
        from datetime import datetime

        export_dir = self._cwd / ".godspeed" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        name = args.strip() or self._session_id[:12]
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"{name}_{timestamp}.md"

        lines = ["# Godspeed Session Export\n"]
        lines.append(f"- **Session**: {self._session_id}")
        lines.append(f"- **Model**: {self._llm_client.model}")
        lines.append(f"- **Exported**: {datetime.now(tz=UTC).isoformat()}\n")
        lines.append("---\n")

        for msg in self._conversation.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                lines.append("## System Prompt\n")
                lines.append(f"```\n{content[:500]}\n```\n")
                if len(content) > 500:
                    lines.append(f"*({len(content) - 500} chars truncated)*\n")
            elif role == "user":
                lines.append(f"## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"## Assistant\n\n{content}\n")
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        lines.append(
                            f"**Tool call**: `{func.get('name', '?')}`\n"
                            f"```json\n{func.get('arguments', '{}')}\n```\n"
                        )
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "?")
                lines.append(f"## Tool Result ({tool_id})\n")
                lines.append(f"```\n{content[:1000]}\n```\n")
                if len(content) > 1000:
                    lines.append(f"*({len(content) - 1000} chars truncated)*\n")

        export_path.write_text("\n".join(lines), encoding="utf-8")
        format_success(f"Exported to: [{DIM}]{export_path}[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_quit(self, _args: str = "") -> CommandResult:
        """Exit Godspeed — session summary shown by app.py."""
        return CommandResult(handled=True, should_quit=True)

    def _cmd_review(self, _args: str = "") -> CommandResult:
        """Show git status and diff review of changes in this session."""

        git_path = shutil.which("git")
        if git_path is None:
            format_error("Git not found.")
            return CommandResult(handled=True)

        # Check if in a git repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                format_error("Not a git repository.")
                return CommandResult(handled=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            format_error("Git not available.")
            return CommandResult(handled=True)

        from rich.table import Table

        table = Table(title="Git Status", border_style=TABLE_BORDER, expand=False)
        table.add_column("File", style=BOLD_PRIMARY)
        table.add_column("Status")

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                format_error("git status failed.")
                return CommandResult(handled=True)

            lines = result.stdout.strip().splitlines()
            if not lines:
                format_info("No uncommitted changes.")
                return CommandResult(handled=True)

            for line in lines[:20]:  # Limit to 20 files
                if len(line) < 3:
                    continue
                status = line[:2]
                filepath = line[3:]
                status_icon = (
                    "[success]M[/success]"
                    if status == " M"
                    else "[warning]A[/warning]"
                    if status == "??"
                    else "[error]D[/error]"
                    if "D" in status
                    else "[error]?[/error]"
                )
                table.add_row(filepath, status_icon)

            _output.console.print(table)

            # Show diff summary
            diff_result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                format_info(f"Diff summary:\n{DIM}{diff_result.stdout.strip()}[/{DIM}]")

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            format_error(f"Git command failed: {e}")
            return CommandResult(handled=True)

        return CommandResult(handled=True)

    def _cmd_sessions(self, _args: str = "") -> CommandResult:
        """List past sessions from .godspeed/sessions/."""
        from datetime import datetime

        sessions_dir = self._cwd / ".godspeed" / "sessions"
        if not sessions_dir.exists():
            format_info("No past sessions.")
            return CommandResult(handled=True)

        session_files = sorted(sessions_dir.glob("*.jsonl"), reverse=True)[:20]
        if not session_files:
            format_info("No past sessions.")
            return CommandResult(handled=True)

        from rich.table import Table

        table = Table(title="Past Sessions", border_style=TABLE_BORDER, expand=False)
        table.add_column("Session", style=BOLD_PRIMARY)
        table.add_column("When")
        table.add_column("Messages")

        for sf in session_files:
            name = sf.stem
            try:
                lines = sf.read_text().splitlines()
                count = len(lines)
                mtime = sf.stat().st_mtime
                when = datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%d %H:%M")
            except OSError:
                continue

            table.add_row(name[:12], when, str(count))

        _output.console.print(table)
        return CommandResult(handled=True)

    def _cmd_skill(self, args: str = "") -> CommandResult:
        """Manage custom skills: list, add, remove."""
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        arg = parts[1] if len(parts) > 1 else ""

        from godspeed.skills.loader import discover_skills

        if action == "list":
            dirs = [
                self._cwd / ".godspeed" / "skills",
                self._cwd / ".godspeed" / "commands",
            ]
            skills = discover_skills(dirs)
            if not skills:
                format_info("No custom skills found.")
                return CommandResult(handled=True)

            from rich.table import Table

            table = Table(title="Custom Skills", border_style=TABLE_BORDER, expand=False)
            table.add_column("Skill", style=BOLD_PRIMARY)
            table.add_column("Trigger")
            table.add_column("Description")

            for s in skills:
                table.add_row(s.name, s.trigger, s.description[:50])

            _output.console.print(table)
            return CommandResult(handled=True)

        if action == "add":
            if not arg:
                format_error("Usage: /skill add <name>")
                return CommandResult(handled=True)

            skill_path = self._cwd / ".godspeed" / "commands" / f"{arg}.md"
            if skill_path.exists():
                format_warning(f"Skill already exists: {arg}")
                return CommandResult(handled=True)

            skill_path.parent.mkdir(parents=True, exist_ok=True)
            template = f"""---
name: {arg}
description: Custom skill for {arg}
trigger: {arg}
---
# {arg} skill

Describe what this skill does here.
"""
            skill_path.write_text(template, encoding="utf-8")
            format_success(f"Created: [{BOLD_PRIMARY}]{skill_path.name}[/{BOLD_PRIMARY}]")
            format_info("Edit the skill file and restart.")
            return CommandResult(handled=True)

        if action == "remove":
            if not arg:
                format_error("Usage: /skill remove <name>")
                return CommandResult(handled=True)

            skill_path = self._cwd / ".godspeed" / "commands" / f"{arg}.md"
            if not skill_path.exists():
                format_error(f"Skill not found: {arg}")
                return CommandResult(handled=True)

            skill_path.unlink()
            format_success(f"Removed: [{BOLD_PRIMARY}]{arg}[/{BOLD_PRIMARY}]")
            return CommandResult(handled=True)

        format_error("Usage: /skill [list|add|remove] <name>")
        return CommandResult(handled=True)

    def _cmd_actions(self, _args: str = "") -> CommandResult:
        """Run or list GitHub Actions workflows."""

        gh_path = shutil.which("gh")
        if gh_path is None:
            format_error("GitHub CLI (gh) not installed.")
            return CommandResult(handled=True)

        try:
            result = subprocess.run(
                ["gh", "run", "list", "--limit", "5"],
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                format_error("No GitHub repository or workflows found.")
                return CommandResult(handled=True)

            if not result.stdout.strip():
                format_info("No recent workflow runs.")
                return CommandResult(handled=True)

            from rich.table import Table

            table = Table(title="GitHub Actions", border_style=TABLE_BORDER, expand=False)
            table.add_column("Workflow", style=BOLD_PRIMARY)
            table.add_column("Status")
            table.add_column("When")

            for line in result.stdout.strip().splitlines()[:10]:
                parts = line.split()
                if len(parts) >= 3:
                    table.add_row(parts[0], parts[1], " ".join(parts[2:]))

            _output.console.print(table)

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            format_error(f"Failed to list workflows: {e}")
            return CommandResult(handled=True)

        return CommandResult(handled=True)

    def _cmd_scan(self, _args: str = "") -> CommandResult:
        """Scan hardware and recommend optimal models per preset tier."""
        from godspeed.evolution.hardware import format_machine_report

        report = format_machine_report()
        _output.console.print(report)
        return CommandResult(handled=True)

    def _cmd_models(self, _args: str = "") -> CommandResult:
        """List installed Ollama models and available presets."""
        from rich.table import Table

        from godspeed.config import GodspeedSettings
        from godspeed.tools.ollama_manager import list_models

        presets = GodspeedSettings.MODEL_PRESETS

        preset_descriptions = {
            "fast": "Local, low VRAM, fast (5.1GB)",
            "balanced": "Local, medium VRAM, strong (9GB)",
            "quality": "Local, high VRAM, best (15GB)",
            "cloud": "NVIDIA NIM free tier, no GPU",
            "frontier": "Claude, best quality, paid API",
        }

        table = Table(title="Model Presets", border_style=TABLE_BORDER, expand=False)
        table.add_column("Preset", style=BOLD_PRIMARY)
        table.add_column("Model", style=NEUTRAL)
        table.add_column("Description")

        for name, model in presets.items():
            desc = preset_descriptions.get(name, "")
            table.add_row(name, model, desc)

        _output.console.print(table)

        installed = list_models()
        if installed:
            _output.console.print()
            inst_table = Table(
                title="Installed Ollama Models",
                border_style=TABLE_BORDER,
                expand=False,
            )
            inst_table.add_column("Name", style=BOLD_PRIMARY)
            inst_table.add_column("Size", justify="right")

            for m in sorted(installed, key=lambda x: x.name):
                inst_table.add_row(m.name, f"{m.size_gb:.1f} GB")

            _output.console.print(inst_table)
        else:
            _output.console.print(f"\n  [{DIM}]No local Ollama models found.[/{DIM}]")
            _output.console.print(f"  [{DIM}]Pull one with: godspeed ollama pull rnj-1:8b[/{DIM}]")

        _output.console.print(
            f"\n  [{DIM}]Switch with /model <preset> or /model <model_name>[/{DIM}]"
        )
        return CommandResult(handled=True)

    def _cmd_correct(self, args: str = "") -> CommandResult:
        """Record an explicit user correction in memory."""
        if not args.strip():
            format_error("Usage: /correct <instruction>  (e.g. /correct always use Path objects)")
            return CommandResult(handled=True)

        from godspeed.memory.user_memory import UserMemory

        db_path = Path.home() / ".godspeed" / "memory.db"
        user_memory = UserMemory(db_path=db_path)
        user_memory.record_correction(
            original="(explicit /correct command)",
            corrected=args.strip(),
            context="user-command",
        )
        format_success("Correction recorded. It will appear in future session system prompts.")
        return CommandResult(handled=True)

    def _cmd_preferences(self, _args: str = "") -> CommandResult:
        """Show stored user preferences and corrections."""
        from godspeed.memory.user_memory import UserMemory

        db_path = Path.home() / ".godspeed" / "memory.db"
        user_memory = UserMemory(db_path=db_path)
        prefs = user_memory.list_preferences()
        corrections = user_memory.get_corrections(limit=10)

        from rich.table import Table

        if prefs:
            table = Table(title="Preferences", border_style=TABLE_BORDER, expand=False)
            table.add_column("Key", style=BOLD_PRIMARY)
            table.add_column("Value")
            for p in prefs:
                table.add_row(p["key"], p["value"])
            _output.console.print(table)
        else:
            format_info("No preferences stored yet.")

        if corrections:
            ctable = Table(title="Recent Corrections", border_style=TABLE_BORDER, expand=False)
            ctable.add_column("ID", style=BOLD_PRIMARY)
            ctable.add_column("Correction")
            for c in corrections:
                ctable.add_row(str(c["id"]), c["corrected"][:60])
            _output.console.print(ctable)

        return CommandResult(handled=True)

    def _cmd_tools(self, _args: str = "") -> CommandResult:
        """List available tools with risk levels and descriptions."""
        from rich.table import Table

        if self._tool_registry is None:
            format_error("Tool registry not available.")
            return CommandResult(handled=True)

        tools = self._tool_registry.list_tools()
        table = Table(title="Available Tools", border_style=TABLE_BORDER, expand=False)
        table.add_column("Tool", style=BOLD_PRIMARY)
        table.add_column("Risk", style=NEUTRAL)
        table.add_column("Description", style=DIM)

        for tool in sorted(tools, key=lambda t: t.name):
            risk = (
                str(tool.risk_level.value)
                if hasattr(tool.risk_level, "value")
                else str(tool.risk_level)
            )
            desc = tool.description.split("\n")[0][:80]
            table.add_row(tool.name, risk, desc)

        _output.console.print(table)
        _output.console.print(f"  [{DIM}]{len(tools)} tools available[/{DIM}]")
        return CommandResult(handled=True)

    def _cmd_diff(self, _args: str = "") -> CommandResult:
        """Show git diff of all changes in the current session."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True,
                text=True,
                cwd=str(self._cwd),
                timeout=10,
            )
            if result.returncode != 0:
                format_info("No changes detected or not a git repository.")
                return CommandResult(handled=True)

            output = result.stdout.strip()
            if not output:
                format_info("No changes detected in working tree.")
                return CommandResult(handled=True)

            _output.console.print()
            _output.console.print(f"  {styled('Session Changes', BOLD_PRIMARY)}")
            _output.console.print(styled(f"  {'-' * 40}", NEUTRAL))

            lines = output.splitlines()
            for line in lines[:20]:
                _output.console.print(f"  {styled(line, DIM)}")
            if len(lines) > 20:
                _output.console.print(f"  [{DIM}]... ({len(lines) - 20} more lines)[/{DIM}]")

            _output.console.print()

            # Detailed diff
            result_detailed = subprocess.run(
                ["git", "diff", "--unified=3"],
                capture_output=True,
                text=True,
                cwd=str(self._cwd),
                timeout=10,
            )
            if result_detailed.stdout.strip():
                from rich.syntax import Syntax

                diff_text = result_detailed.stdout.strip()[:4000]
                syntax = Syntax(diff_text, "diff", theme="monokai", word_wrap=True)
                _output.console.print(syntax)
                _output.console.print()

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            format_error(f"Could not run git diff: {exc}")

        return CommandResult(handled=True)
