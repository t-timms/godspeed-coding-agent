"""Slash commands for the Godspeed TUI."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any, ClassVar

from godspeed.config import append_permission_rule
from godspeed.tui import output as _output
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
        self._handlers["/plan"] = self._cmd_plan
        self._handlers["/checkpoint"] = self._cmd_checkpoint
        self._handlers["/restore"] = self._cmd_restore
        self._handlers["/pause"] = self._cmd_pause
        self._handlers["/resume"] = self._cmd_resume
        self._handlers["/guidance"] = self._cmd_guidance
        self._handlers["/tasks"] = self._cmd_tasks
        self._handlers["/reindex"] = self._cmd_reindex
        self._handlers["/stats"] = self._cmd_stats
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
                    ("/pause", "Pause the agent loop"),
                    ("/resume", "Resume a paused agent"),
                    ("/guidance <msg>", "Inject guidance and resume"),
                ],
            ),
            (
                "Context",
                [
                    ("/context", "Show context window usage"),
                    ("/checkpoint [name]", "Save/list checkpoints"),
                    ("/restore <name>", "Restore a checkpoint"),
                    ("/tasks", "Show task list"),
                    ("/reindex", "Rebuild codebase search index"),
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
                f"Whisper mode [{BOLD_PRIMARY}]OFF[/{BOLD_PRIMARY}]"
                " — tool output visible"
            )
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
        """Show context window usage."""
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
