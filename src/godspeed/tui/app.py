"""Main TUI application for Godspeed ΓÇö prompt-toolkit input, Rich output."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.status import Status

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.agent.result import AgentCancelledError
from godspeed.audit.trail import AuditTrail
from godspeed.config import GodspeedSettings
from godspeed.llm.client import LLMClient
from godspeed.security.permissions import ALLOW, ASK, PermissionDecision, PermissionEngine
from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.registry import ToolRegistry
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands
from godspeed.tui.completions import GodspeedCompleter
from godspeed.tui.mentions import parse_mentions, resolve_mentions
from godspeed.tui.output import (
    format_assistant_text,
    format_diff_review_prompt,
    format_error,
    format_parallel_results,
    format_parallel_tool_calls,
    format_permission_denied,
    format_permission_prompt,
    format_session_summary,
    format_status_hud,
    format_thinking,
    format_tool_call,
    format_tool_result,
    format_turn_separator,
    format_welcome,
    is_compact_mode,
    set_compact_mode,
)
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    BOLD_WARNING,
    DIM,
    ERROR,
    NEUTRAL,
    PROMPT_ICON,
    icon_prompt,
    styled,
)

logger = logging.getLogger(__name__)


def _schedule_dream(dream: Any) -> None:
    """Schedule background dream consolidation if 24h interval has elapsed."""
    from godspeed.skills.dream import SkillDream

    if not isinstance(dream, SkillDream):
        return
    if not dream.should_run():
        logger.debug("Dream consolidation skipped — interval not elapsed")
        return
    skills_dir = Path.home() / ".godspeed" / "skills"

    async def _run_dream() -> None:
        stats = dream.run(skills_dir)
        logger.info("Background dream consolidation: %s", stats)

    asyncio.ensure_future(_run_dream())  # noqa: RUF006


def _build_key_bindings() -> KeyBindings:
    """Build prompt-toolkit key bindings.

    - Enter: submit input
    - Escape+Enter: insert newline for multiline input
    - Ctrl+C: abort current input
    """
    bindings = KeyBindings()

    @bindings.add(Keys.Enter)
    def _submit(event: Any) -> None:
        """Enter submits the input."""
        event.current_buffer.validate_and_handle()

    @bindings.add(Keys.Escape, Keys.Enter)
    def _newline(event: Any) -> None:
        """Escape+Enter inserts a newline for multiline input."""
        event.current_buffer.insert_text("\n")

    return bindings


class TUIApp:
    """Main TUI application orchestrating input, agent loop, and output.

    Wires together: prompt-toolkit input -> slash commands / agent loop -> Rich output.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        conversation: Conversation,
        permission_engine: PermissionEngine | None,
        audit_trail: AuditTrail | None,
        session_id: str,
        skills: list[Any] | None = None,
        extra_completions: list[tuple[str, str]] | None = None,
        hook_executor: Any | None = None,
        task_store: Any | None = None,
        codebase_index: Any | None = None,
        correction_tracker: Any | None = None,
        session_memory: Any | None = None,
        compact: bool = False,
        skill_evolution: Any | None = None,
        skill_hub: Any | None = None,
        skill_dream: Any | None = None,
        skills_dir: Any | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._conversation = conversation
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id
        self._correction_tracker = correction_tracker
        self._session_memory = session_memory

        set_compact_mode(compact)

        # Pause/resume event for human-in-the-loop
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start in running state

        # Mid-turn cancel: set by Ctrl+C while the agent is running. Cleared
        # each time a new turn starts. Distinct from _pause_event ΓÇö pause
        # stalls at iteration boundary, cancel unwinds immediately.
        self._cancel_event = asyncio.Event()

        # Per-session turn counter, displayed in the status HUD.
        self._turn_count = 0

        self._commands = Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=permission_engine,
            audit_trail=audit_trail,
            session_id=session_id,
            cwd=tool_context.cwd,
            pause_event=self._pause_event,
            tool_registry=tool_registry,
        )

        # Wire task store and codebase index for commands
        if task_store is not None:
            self._commands._task_store = task_store
        if codebase_index is not None:
            self._commands._codebase_index = codebase_index

        # Register skill commands with full skill system features
        if skills:
            from godspeed.skills.commands import register_skill_commands

            register_skill_commands(
                self._commands,
                conversation,
                skills,
                evolution=skill_evolution,
                hub=skill_hub,
                dream=skill_dream,
                skills_dir=skills_dir,
                llm_client=self._llm_client,
            )

        self._skill_dream = skill_dream

        self._completer = GodspeedCompleter(
            cwd=tool_context.cwd,
            extra_commands=extra_completions,
        )
        self._key_bindings = _build_key_bindings()
        self._hook_executor = hook_executor

        # Patch the permission check to handle ASK interactively
        self._original_permissions = tool_context.permissions
        if permission_engine is not None:
            from godspeed.security.approval_tracker import ApprovalTracker

            self._approval_tracker = ApprovalTracker()
            tool_context.permissions = _InteractivePermissionProxy(
                permission_engine,
                approval_tracker=self._approval_tracker,
            )

        # Diff-review gate: diff-producing tools (file_edit, file_write,
        # diff_apply) consult this reviewer just before writing. TUI only ΓÇö
        # headless/CI path leaves diff_reviewer None so writes proceed as
        # before.
        self._diff_reviewer = _InteractiveDiffReviewer()
        tool_context.diff_reviewer = self._diff_reviewer

        # Track last SIGINT time for the "press twice" pattern.
        self._last_sigint_monotonic = 0.0

    def _on_sigint(self) -> None:
        """Handle Ctrl+C during agent execution.

        First press sets cancel_event (clean unwind). Second press within
        1s raises KeyboardInterrupt (hard interrupt).
        """
        now = time.monotonic()
        if self._cancel_event.is_set() and (now - self._last_sigint_monotonic) < 1.0:
            raise KeyboardInterrupt
        self._last_sigint_monotonic = now
        self._cancel_event.set()

    def _get_permission_mode(self) -> str:
        """Return the current permission mode string for display."""
        if self._permission_engine is None:
            return "normal"
        if getattr(self._permission_engine, "plan_mode", False):
            return "plan"
        # Check for strict/yolo mode
        deny_count = len(getattr(self._permission_engine, "deny_rules", []))
        has_wildcard_deny = any(
            r.pattern in ("*", "Shell(*)", "FileWrite(*)", "FileEdit(*)")
            for r in getattr(self._permission_engine, "deny_rules", [])
        )
        if deny_count > 5 or has_wildcard_deny:
            return "strict"
        # If all tools are auto-approved (no ask rules), it's "yolo"
        ask_count = len(getattr(self._permission_engine, "ask_rules", []))
        if ask_count == 0 and deny_count == 0:
            return "yolo"
        return "normal"

    def _get_prompt_state(self) -> str:
        """Return the prompt state string for icon_prompt()."""
        if self._permission_engine is not None and getattr(
            self._permission_engine, "plan_mode", False
        ):
            return "plan"
        if self._pause_event is not None and not self._pause_event.is_set():
            return "paused"
        return ""

    async def run(self) -> None:
        """Run the main TUI loop."""
        start_time = time.monotonic()
        tool_calls = 0
        tool_errors = 0
        tool_denied = 0

        # Background dream consolidation (non-blocking, 24h interval)
        if self._skill_dream is not None:
            _schedule_dream(self._skill_dream)

        # Display welcome with config summary
        tool_names = (
            [t.name for t in self._tool_registry.list_tools()] if self._tool_registry else None
        )
        deny_patterns = (
            [r.pattern for r in self._permission_engine.deny_rules]
            if self._permission_engine
            else None
        )
        format_welcome(
            model=self._llm_client.model,
            project_dir=str(self._tool_context.cwd),
            permission_mode=self._get_permission_mode(),
            tools=tool_names,
            deny_rules=deny_patterns,
        )

        try:
            session: PromptSession[str] = PromptSession(
                completer=self._completer,
                key_bindings=self._key_bindings,
                multiline=True,
            )
        except Exception as exc:
            # prompt-toolkit fails in non-TTY contexts (piped input, CI, etc.)
            _output.console.print(
                f"\n[{ERROR}]  Cannot create interactive session: {exc}[/{ERROR}]\n"
                f"  [{DIM}]Godspeed requires a real terminal. Run it directly in your"
                f" terminal, not through a pipe or non-interactive shell.[/{DIM}]"
            )
            return

        # Install SIGINT handler once before the main loop. First Ctrl+C
        # sets cancel_event (clean unwind via AgentCancelledError). Second
        # press within 1s raises KeyboardInterrupt for hard exit.
        running_loop = asyncio.get_running_loop()
        _sigint_installed = False
        try:
            running_loop.add_signal_handler(signal.SIGINT, self._on_sigint)
            _sigint_installed = True
        except (NotImplementedError, RuntimeError):
            # Windows: ProactorEventLoop does not support add_signal_handler
            pass

        while True:
            # Compute context percentage for the prompt
            context_pct = (
                self._conversation.token_count / self._conversation.max_tokens * 100
                if self._conversation.max_tokens > 0
                else 0.0
            )

            def _get_short_model() -> str:
                m = self._llm_client.model
                return m.split("/", 1)[-1] if "/" in m else m

            try:
                user_input = await session.prompt_async(
                    HTML(
                        icon_prompt(
                            self._get_prompt_state(),
                            turn=self._turn_count,
                            context_pct=context_pct,
                            compact=is_compact_mode(),
                            model=_get_short_model(),
                            cost=self._llm_client.total_cost_usd,
                        )
                    ),
                )
            except KeyboardInterrupt:
                _output.console.print(f"\n  [{DIM}]Interrupted. Type /quit to exit.[/{DIM}]")
                continue
            except EOFError:
                break

            if not user_input.strip():
                continue

            # Check for slash commands
            cmd_result = self._commands.dispatch(user_input)
            if cmd_result is not None:
                if cmd_result.should_quit:
                    break
                continue

            # Memory: detect and store user corrections
            if self._correction_tracker is not None:
                self._correction_tracker.check_for_correction(user_input)

            # Echo user message with turn marker
            self._turn_count += 1
            _output.console.print()
            _output.console.print(
                f"  {styled(str(self._turn_count), NEUTRAL)}"
                f" {styled(PROMPT_ICON, BOLD_PRIMARY)}"
                f" {user_input.strip()}"
            )

            # Parse @-mentions from input and resolve to content blocks
            effective_input = user_input

            cleaned_text, mentions = parse_mentions(user_input)
            if mentions:
                try:
                    mention_blocks = await resolve_mentions(mentions, self._tool_context.cwd)
                    if mention_blocks:
                        # Build multimodal message: cleaned text + resolved content

                        content_blocks = [{"type": "text", "text": cleaned_text}]
                        content_blocks.extend(mention_blocks)
                        self._conversation.add_user_message(content_blocks)
                        effective_input = ""  # Already added to conversation
                except Exception as exc:
                    logger.warning("Mention resolution failed: %s", exc)
                    # Fall through with original input

            # Run agent loop with context-aware thinking indicator
            spinner = _ThinkingSpinner()

            # Track per-tool-call timing: tool_name -> start_monotonic
            _tool_timings: dict[str, float] = {}

            def _track_tool_call(
                tool_name: str,
                args: dict[str, Any],
                _s: _ThinkingSpinner = spinner,
                _timings: dict[str, float] = _tool_timings,
            ) -> None:
                nonlocal tool_calls
                tool_calls += 1
                _timings[tool_name] = time.monotonic()
                _s.update(tool_name, args)
                _s.stop()
                format_tool_call(tool_name, args)

            def _track_tool_result(
                tool_name: str,
                result: Any,
                _s: _ThinkingSpinner = spinner,
                _timings: dict[str, float] = _tool_timings,
            ) -> None:
                nonlocal tool_errors
                is_error = getattr(result, "is_error", False)
                if is_error:
                    tool_errors += 1
                _s.start()
                output = getattr(result, "output", str(result))
                error = getattr(result, "error", None)
                display_text = str(error) if is_error and error else str(output)
                # Calculate elapsed time for this tool call
                start = _timings.pop(tool_name, None)
                duration_ms = (time.monotonic() - start) * 1000 if start is not None else 0.0
                _s.stop()
                format_tool_result(
                    tool_name, display_text, is_error=is_error, duration_ms=duration_ms
                )

            def _track_permission_denied(
                tool_name: str, reason: str, _s: _ThinkingSpinner = spinner
            ) -> None:
                nonlocal tool_denied
                tool_denied += 1
                _s.stop()
                format_permission_denied(tool_name, reason)

            def _track_parallel_start(
                calls: list[tuple[str, dict[str, Any]]],
                _s: _ThinkingSpinner = spinner,
            ) -> None:
                _s.stop()
                format_parallel_tool_calls(calls)

            def _track_parallel_complete(
                results: list[tuple[str, str, bool]],
                _s: _ThinkingSpinner = spinner,
            ) -> None:
                format_parallel_results(results)
                _s.start()

            def _on_thinking(
                text: str,
                _s: _ThinkingSpinner = spinner,
            ) -> None:
                _s.stop()
                format_thinking(text)
                _s.start()

            # Fresh cancel state per turn
            self._cancel_event.clear()
            # Reset SIGINT debounce timer so the "press twice" pattern
            # starts fresh each turn.
            self._last_sigint_monotonic = 0.0

            try:
                spinner.start()
                await agent_loop(
                    user_input=effective_input if effective_input else user_input,
                    conversation=self._conversation,
                    llm_client=self._llm_client,
                    tool_registry=self._tool_registry,
                    tool_context=self._tool_context,
                    on_assistant_text=spinner.wrap(_on_assistant_text),
                    on_tool_call=_track_tool_call,
                    on_tool_result=_track_tool_result,
                    on_permission_denied=_track_permission_denied,
                    on_assistant_chunk=spinner.wrap(_on_assistant_chunk),
                    max_iterations=self._commands.max_iterations,
                    pause_event=self._pause_event,
                    cancel_event=self._cancel_event,
                    hook_executor=self._hook_executor,
                    skip_user_message=not effective_input,
                    on_parallel_start=_track_parallel_start,
                    on_parallel_complete=_track_parallel_complete,
                    on_thinking=_on_thinking,
                )
                _output.console.print()  # End streaming output with newline
            except AgentCancelledError:
                _output.console.print(
                    f"\n  [{DIM}]Agent cancelled. Send another prompt or /quit.[/{DIM}]"
                )
            except KeyboardInterrupt:
                # Hard interrupt: user pressed Ctrl+C twice (or the loop-level
                # signal handler wasn't installed on this platform). Treat
                # same as cancel for display, but surface the distinct reason.
                _output.console.print(f"\n  [{DIM}]Agent interrupted.[/{DIM}]")
            except Exception as exc:
                logger.error("Agent loop error: %s", exc, exc_info=True)
                format_error(f"Agent error: {exc}")
            finally:
                spinner.stop()
                if _sigint_installed:
                    # Restore default SIGINT handling while we're waiting for
                    # the next prompt ΓÇö otherwise a Ctrl+C at the prompt would
                    # silently set an unused cancel_event and swallow the key.
                    try:
                        running_loop.remove_signal_handler(signal.SIGINT)
                    except (NotImplementedError, RuntimeError, ValueError):
                        logger.debug("Could not remove SIGINT handler")

                # Per-turn status HUD: compact one-line summary of tokens,
                # cost, model, and turn count. Prints after spinner + output
                # so it appears as the last line of the turn before the
                # next prompt. Uses LLMClient's own accumulators so no
                # session-state plumbing needed.
                context_pct = (
                    self._conversation.token_count / self._conversation.max_tokens * 100
                    if self._conversation.max_tokens > 0
                    else 0
                )
                preset_tag = ""
                for pname, pmodel in GodspeedSettings.MODEL_PRESETS.items():
                    if pmodel == self._llm_client.model:
                        preset_tag = pname
                        break
                perm_mode = ""
                if self._permission_engine is not None:
                    if getattr(self._permission_engine, "plan_mode", False):
                        perm_mode = "plan"
                    else:
                        perm_mode = getattr(self._permission_engine, "_mode", "normal")
                max_iters = self._commands.max_iterations or 0
                format_status_hud(
                    input_tokens=self._llm_client.total_input_tokens,
                    output_tokens=self._llm_client.total_output_tokens,
                    cost_usd=self._llm_client.total_cost_usd,
                    model=self._llm_client.model,
                    turns=self._turn_count,
                    budget_usd=getattr(self._llm_client, "max_cost_usd", 0.0),
                    max_iterations=max_iters,
                    context_pct=context_pct,
                    permission_mode=perm_mode,
                    preset=preset_tag,
                )

                # Visual separator before the next prompt
                if not is_compact_mode():
                    format_turn_separator(turn=self._turn_count)

        # Session summary on exit
        duration = time.monotonic() - start_time
        format_session_summary(
            duration_secs=duration,
            input_tokens=self._llm_client.total_input_tokens,
            output_tokens=self._llm_client.total_output_tokens,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            tool_denied=tool_denied,
            model=self._llm_client.model,
            session_id=self._session_id,
        )

        if self._session_memory is not None:
            self._session_memory.end_session(
                self._session_id,
                summary=f"turns={self._turn_count} tools={tool_calls} errors={tool_errors}",
            )
        if self._audit_trail is not None:
            self._audit_trail.record(
                event_type="session_end",
                detail={"reason": "user_quit"},
            )


# -- Thinking spinner -------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "file_read": "Reading",
    "file_write": "Writing",
    "file_edit": "Editing",
    "shell": "Running",
    "grep_search": "Searching",
    "glob_search": "Searching",
    "git": "Git",
    "repo_map": "Mapping",
}


class _ThinkingSpinner:
    """Context-aware Rich Status spinner with elapsed-time display.

    Shows what the agent is doing — "Thinking..." when waiting for LLM,
    tool-specific labels during tool execution. After 10s of thinking,
    shows elapsed seconds so the user knows the agent isn't frozen.
    """

    def __init__(self) -> None:
        self._status: Any | None = None
        self._started = False
        self._start_time: float = 0.0
        self._update_task: asyncio.Task | None = None
        self._tool_label: str = ""

    def _make_label(self, text: str) -> str:
        return f"[{NEUTRAL}]{PROMPT_ICON} {text}[/{NEUTRAL}]"

    def start(self) -> None:
        if self._started:
            return

        self._start_time = time.monotonic()
        self._tool_label = ""
        self._status = Status(
            self._make_label("Thinking..."),
            console=_output.console,
            spinner="dots",
            spinner_style=NEUTRAL,
        )
        self._status.start()
        self._started = True
        # Start elapsed-time updater
        self._update_task = asyncio.ensure_future(self._update_elapsed())

    async def _update_elapsed(self) -> None:
        """Update the spinner label with elapsed time after 10s."""
        while self._started:
            await asyncio.sleep(1)
            elapsed = int(time.monotonic() - self._start_time)
            if elapsed >= 10 and self._status is not None:
                label = self._tool_label or "Thinking..."
                self._status.update(self._make_label(f"{label} ({elapsed}s)"))

    def update(self, tool_name: str, args: dict[str, Any]) -> None:
        """Update spinner text based on current tool call."""
        if not self._started or self._status is None:
            return
        label = _TOOL_LABELS.get(tool_name, tool_name)
        primary_arg = args.get("file_path") or args.get("command") or args.get("pattern") or ""
        if primary_arg:
            if len(primary_arg) > 50:
                primary_arg = "..." + primary_arg[-47:]
            self._tool_label = f"{label} {primary_arg}"
        else:
            self._tool_label = f"{label}..."
        self._status.update(self._make_label(self._tool_label))

    def stop(self) -> None:
        if self._started and self._status is not None:
            self._status.stop()
            self._started = False
        if self._update_task is not None:
            self._update_task.cancel()
            self._update_task = None

    def wrap(self, fn: Any) -> Any:
        """Return a wrapper that stops the spinner before calling *fn*."""
        spinner = self

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            spinner.stop()
            return fn(*args, **kwargs)

        return _wrapped


# -- Callbacks for the agent loop -------------------------------------------------


def _on_assistant_chunk(text: str) -> None:
    """Callback: display streaming text chunk as it arrives."""
    _output.console.print(text, end="")


def _on_assistant_text(text: str) -> None:
    """Callback: render assistant text as Markdown."""
    format_assistant_text(text)


class _InteractivePermissionProxy:
    """Wraps PermissionEngine to intercept ASK decisions with an interactive prompt.

    When the permission engine returns ASK, this proxy prompts the user
    via the terminal and either grants, denies, or creates a session-scoped grant.

    Optionally tracks repeated approvals via ApprovalTracker and suggests
    adding patterns as permanent allow rules after a configurable threshold.
    """

    def __init__(
        self,
        engine: PermissionEngine,
        approval_tracker: Any | None = None,
    ) -> None:
        self._engine = engine
        self._tracker = approval_tracker

    def evaluate(self, tool_call: Any) -> PermissionDecision:
        """Evaluate permissions, prompting the user for ASK decisions."""
        decision = self._engine.evaluate(tool_call)
        if decision != ASK:
            return decision

        # Show the permission prompt with contextual detail
        args = getattr(tool_call, "arguments", None) or {}
        format_permission_prompt(tool_call.tool_name, decision.reason, arguments=args)
        try:
            answer = _output.console.input(f"[{BOLD_WARNING}]  > [/{BOLD_WARNING}]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer in ("y", "yes"):
            # Track approval for auto-permission suggestion
            pattern = tool_call.format_for_permission
            if self._tracker is not None:
                self._tracker.record_approval(pattern)
                if self._tracker.should_suggest(pattern):
                    self._suggest_auto_permission(pattern)
            return PermissionDecision(ALLOW, "user approved")

        if answer in ("a", "always"):
            pattern = tool_call.format_for_permission
            # LOW-risk tools get tool-level session grant (WebSearch(*) covers all queries)
            risk = self._engine._tool_risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
            if risk == RiskLevel.LOW:
                self._engine.grant_tool_session_permission(tool_call.tool_name)
                # Also track for auto-permission suggestion
                if self._tracker is not None:
                    self._tracker.record_approval(f"{tool_call.tool_name}(*)")
                return PermissionDecision(ALLOW, f"session grant: {tool_call.tool_name}(*)")
            self._engine.grant_session_permission(pattern)
            return PermissionDecision(ALLOW, f"session grant: {pattern}")

        return PermissionDecision("deny", "user denied")

    def _suggest_auto_permission(self, pattern: str) -> None:
        """Suggest adding a pattern as a permanent allow rule."""
        # Skip if already in allow rules
        for rule in self._engine.allow_rules:
            if rule == pattern:
                return

        from godspeed.tui.theme import NEUTRAL, SUCCESS

        _output.console.print(
            f"\n  [{NEUTRAL}]You've approved [{SUCCESS}]{pattern}"
            f"[/{SUCCESS}] multiple times.[/{NEUTRAL}]"
        )
        _output.console.print(f"  [{NEUTRAL}]Add to permanent allow rules? (y/n)[/{NEUTRAL}]")
        try:
            answer = _output.console.input(f"[{BOLD_WARNING}]  > [/{BOLD_WARNING}]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer in ("y", "yes"):
            from godspeed.config import append_allow_rule

            success = append_allow_rule(pattern)
            if success:
                # Also update engine in-memory
                self._engine.add_rule(pattern, "allow")
                _output.console.print(f"  [{SUCCESS}]Added to allow rules.[/{SUCCESS}]")
            else:
                from godspeed.tui.theme import WARNING

                _output.console.print(
                    f"  [{WARNING}]Could not persist rule. Added for this session only.[/{WARNING}]"
                )
                self._engine.grant_session_permission(pattern)


class _InteractiveDiffReviewer:
    """Implements `ToolContext.DiffReviewer` by prompting the human via the TUI.

    Distinct from `_InteractivePermissionProxy` ΓÇö permission answers
    "should this tool run?" once; the reviewer answers "should THIS
    specific diff be applied?" per pending write.

    Current decision vocabulary: `"accept"` / `"reject"`. Future:
    `"edit"` (open the patch in $EDITOR before apply). Unknown values
    degrade to reject by the calling tool.
    """

    def __init__(self) -> None:
        # Session-scoped "accept all" bypass. Set by the user answering "a".
        self._always_accept = False

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

        # Render the diff (Rich `_output.console.input` is sync; run in a thread so
        # we don't block the asyncio loop while waiting for keystrokes).
        format_diff_review_prompt(tool_name, path, before, after)
        try:
            answer = await asyncio.to_thread(
                lambda: (
                    _output.console.input(f"[{BOLD_WARNING}]  > [/{BOLD_WARNING}]").strip().lower()
                )
            )
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer in ("y", "yes", ""):
            return "accept"
        if answer in ("a", "always"):
            self._always_accept = True
            return "accept"
        return "reject"
