"""Main chat screen — conversation history, prompt input, and status bar."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DirectoryTree, Footer, Input

from godspeed.tui.commands import Commands
from godspeed.tui.theme import (
    BOLD_PRIMARY,
    DIM,
    NEUTRAL,
    PROMPT_ICON,
    styled,
)
from godspeed.tui.widgets.chat_view import ChatView
from godspeed.tui.widgets.file_picker import FilePicker

logger = logging.getLogger(__name__)


class ChatScreen(Screen):
    """Main interaction screen — chat history, multi-line input, status footer."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str, str]]] = [
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+s", "sessions", "Sessions"),
        Binding("ctrl+r", "shell", "Shell"),
        Binding("tab", "toggle_files", "Files", show=False),
        Binding("f1", "show_help", "Help"),
        Binding("escape", "focus_input", "Focus", show=False),
        Binding("ctrl+c", "cancel_agent", "Cancel", show=False),
    ]

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Any,
        tool_context: Any,
        conversation: Any,
        permission_engine: Any,
        audit_trail: Any,
        session_id: str,
        commands: Commands,
        hook_executor: Any = None,
        correction_tracker: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._conversation = conversation
        self._permission_engine = permission_engine
        self._audit_trail = audit_trail
        self._session_id = session_id
        self._commands = commands
        self._hook_executor = hook_executor
        self._correction_tracker = correction_tracker

        self._turn_count = 0
        self._tool_calls = 0
        self._tool_errors = 0
        self._tool_denied = 0
        self._start_time = time.monotonic()
        self._running = False

        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()

    def compose(self: Any) -> Any:
        yield DirectoryTree(self._tool_context.cwd, id="file-tree")
        yield ChatView(id="chat-log")
        with Horizontal(id="input-area"):
            yield Input(
                placeholder="Type your message or /command...",
                id="prompt-input",
            )
        yield FilePicker(self._tool_context.cwd)
        yield Footer()

    def on_mount(self: Any) -> None:
        self._display_welcome()
        self.query_one("#prompt-input", Input).focus()

    def _display_welcome(self: Any) -> None:
        chat_log = self.query_one("#chat-log", ChatView)
        import hashlib

        from godspeed import __version__

        verses = [
            (
                "Proverbs 3:5-6",
                "Trust in the Lord with all your heart and lean not on your own "
                "understanding; in all your ways submit to him, and he will make "
                "your paths straight.",
            ),
            (
                "Joshua 1:9",
                "Have I not commanded you? Be strong and courageous. Do not be "
                "afraid; do not be discouraged, for the Lord your God will be with "
                "you wherever you go.",
            ),
            ("Philippians 4:13", "I can do all this through him who gives me strength."),
            ("Psalm 23:1", "The Lord is my shepherd, I lack nothing."),
            (
                "Jeremiah 29:11",
                "For I know the plans I have for you, declares the Lord, plans to "
                "prosper you and not to harm you, plans to give you hope and a "
                "future.",
            ),
        ]

        seed_str = str(time.localtime().tm_yday).encode()
        day_seed = int(hashlib.sha256(seed_str).hexdigest()[:4], 16)
        verse_ref, verse_text = verses[day_seed % len(verses)]

        model_short = (
            self._llm_client.model.split("/", 1)[-1]
            if "/" in self._llm_client.model
            else self._llm_client.model
        )
        mode = self._get_permission_mode()

        chat_log.write()
        chat_log.write(
            f"  {styled(PROMPT_ICON, BOLD_PRIMARY)} {styled('godspeed', BOLD_PRIMARY)}"
            f"  {styled(f'v{__version__}', DIM)}"
        )
        chat_log.write(
            f"  {styled(model_short, NEUTRAL)}  {styled(mode, DIM)}"
        )
        chat_log.write()
        chat_log.write(f"  [{DIM}]{verse_text}[/{DIM}]")
        chat_log.write(f"  [{DIM}]— {verse_ref}[/{DIM}]")
        chat_log.write()

    def _get_permission_mode(self: Any) -> str:
        if self._permission_engine is None:
            return "normal"
        if getattr(self._permission_engine, "plan_mode", False):
            return "plan"
        deny_count = len(getattr(self._permission_engine, "deny_rules", []))
        ask_count = len(getattr(self._permission_engine, "ask_rules", []))
        if ask_count == 0 and deny_count == 0:
            return "yolo"
        return "normal"

    def action_focus_input(self: Any) -> None:
        self.query_one("#prompt-input", Input).focus()

    def action_show_help(self: Any) -> None:
        from godspeed.tui.screens.help_screen import HelpScreen

        self.app.push_screen(HelpScreen(self._commands))

    def action_sessions(self: Any) -> None:
        from godspeed.tui.screens.session_list import SessionListScreen

        def _resume(session_name: str) -> None:
            self._commands.dispatch(f"/restore {session_name}")
            chat_log = self.query_one("#chat-log", ChatView)
            chat_log.write()
            chat_log.write(
                f"  [{DIM}]Resumed session: {session_name}[/{DIM}]"
            )

        self.app.push_screen(
            SessionListScreen(self._tool_context.cwd, on_resume=_resume)
        )

    def action_shell(self: Any) -> None:
        from godspeed.tui.screens.shell_screen import ShellScreen

        self.app.push_screen(ShellScreen(cwd=str(self._tool_context.cwd)))

    def action_toggle_files(self: Any) -> None:
        tree = self.query_one("#file-tree", DirectoryTree)
        if tree.display:
            tree.display = False
            self.query_one("#prompt-input", Input).focus()
        else:
            tree.display = True
            tree.focus()

    def on_directory_tree_file_selected(
        self: Any, event: DirectoryTree.FileSelected
    ) -> None:
        tree = self.query_one("#file-tree", DirectoryTree)
        tree.display = False
        inp = self.query_one("#prompt-input", Input)
        current = inp.value or ""
        spacer = " " if current and not current.endswith(" ") else ""
        inp.value = f"{current}{spacer}@file:{event.path}"
        inp.focus()
        inp = self.query_one("#prompt-input", Input)
        inp.value = "/"
        inp.focus()

    def action_cancel_agent(self: Any) -> None:
        if not self._running:
            return
        self._cancel_event.set()
        chat_log = self.query_one("#chat-log", ChatView)
        chat_log.write(f"  [{DIM}]Cancelling... press Ctrl+C again for hard stop.[/{DIM}]")

    def on_input_changed(self: Any, event: Input.Changed) -> None:
        text = event.value or ""
        idx = text.rfind("@")
        if idx >= 0:
            after_at = text[idx:]
            if " " not in after_at:
                query = after_at[1:]
                picker = self.query_one("#file-picker", FilePicker)
                picker.filter_for(query)
                return
        picker = self.query_one("#file-picker", FilePicker)
        picker.display = False

    def on_file_picker_selected(self: Any, event: FilePicker.Selected) -> None:
        inp = self.query_one("#prompt-input", Input)
        text = inp.value or ""
        idx = text.rfind("@")
        if idx >= 0:
            inp.value = text[:idx] + f"@file:{event.item}"
        picker = self.query_one("#file-picker", FilePicker)
        picker.display = False
        inp.focus()

    def _get_status_text(self: Any) -> str:
        short = (
            self._llm_client.model.split("/", 1)[-1]
            if "/" in self._llm_client.model
            else self._llm_client.model
        )
        total = self._llm_client.total_input_tokens + self._llm_client.total_output_tokens
        context_pct = (
            self._conversation.token_count / self._conversation.max_tokens * 100
            if self._conversation.max_tokens > 0
            else 0.0
        )
        parts = [
            short,
            f"{total:,} tok",
        ]
        if self._llm_client.total_cost_usd > 0:
            parts.append(f"${self._llm_client.total_cost_usd:.4f}")
        parts.append(f"t{self._turn_count}")
        if context_pct:
            parts.append(f"ctx{context_pct:.0f}%")
        mode = self._get_permission_mode()
        if mode != "normal":
            parts.append(mode)
        return " | ".join(parts)

    async def on_input_submitted(self: Any, message: Any) -> None:
        user_input = message.value.strip()
        if not user_input:
            return

        self.query_one("#prompt-input", Input).value = ""

        cmd_result = self._commands.dispatch(user_input)
        if cmd_result is not None:
            if cmd_result.should_quit:
                self._show_session_summary()
                self.app.exit()
            return

        chat_log = self.query_one("#chat-log", ChatView)
        self._turn_count += 1
        self._cancel_event.clear()

        # Echo user message
        chat_log.write()
        chat_log.write(
            f"  {styled(str(self._turn_count), NEUTRAL)}"
            f" {styled(PROMPT_ICON, BOLD_PRIMARY)}"
            f" {user_input}"
        )

        # Memory: detect corrections
        if self._correction_tracker is not None:
            self._correction_tracker.check_for_correction(user_input)

        self._running = True
        chat_log.start_turn()
        try:
            from godspeed.agent.loop import agent_loop

            self._tool_calls = 0
            self._tool_errors = 0
            self._tool_denied = 0

            def _on_chunk(text: str) -> None:
                chat_log.write_chunk(text)

            def _on_text(text: str) -> None:
                chat_log.end_turn()
                chat_log.write_markdown(text)

            def _on_tool_call(name: str, args: dict[str, Any]) -> None:
                self._tool_calls += 1
                if not self._commands.whisper_mode:
                    chat_log.write_tool_call(name, args)

            def _on_tool_result(name: str, result: Any) -> None:
                is_error = getattr(result, "is_error", False)
                if is_error:
                    self._tool_errors += 1
                if not self._commands.whisper_mode:
                    output = getattr(result, "output", str(result))
                    error = getattr(result, "error", None)
                    display = str(error) if is_error and error else str(output)
                    chat_log.write_tool_result(name, display, is_error=is_error)

            def _on_denied(name: str, reason: str) -> None:
                self._tool_denied += 1
                if not self._commands.whisper_mode:
                    chat_log.write_permission_denied(name, reason)

            def _on_thinking(text: str) -> None:
                if not self._commands.whisper_mode:
                    chat_log.write_thinking(text)

            _final_text = await agent_loop(
                user_input=user_input,
                conversation=self._conversation,
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
                tool_context=self._tool_context,
                on_assistant_text=_on_text,
                on_tool_call=_on_tool_call,
                on_tool_result=_on_tool_result,
                on_permission_denied=_on_denied,
                on_assistant_chunk=_on_chunk,
                max_iterations=self._commands.max_iterations,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
                hook_executor=self._hook_executor,
                on_thinking=_on_thinking,
            )
            chat_log.end_turn()
            chat_log.write()
        except Exception as exc:
            logger.error("Agent loop error: %s", exc, exc_info=True)
            from godspeed.tui.theme import ERROR as ERR_COLOR

            chat_log.write(f"  [{ERR_COLOR}]Agent error: {exc}[/{ERR_COLOR}]")
        finally:
            self._running = False
            chat_log.write(
                f"  [{DIM}]{self._get_status_text()}[/{DIM}]"
            )
            self.query_one("#prompt-input", Input).focus()

    def _show_session_summary(self: Any) -> None:
        chat_log = self.query_one("#chat-log", ChatView)
        duration = time.monotonic() - self._start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        dur = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        total = self._llm_client.total_input_tokens + self._llm_client.total_output_tokens

        chat_log.write()
        chat_log.write(f"  [{DIM}]{dur}  {total:,} tokens[/{DIM}]")
        if self._llm_client.total_cost_usd > 0:
            chat_log.write(f"  [{DIM}]${self._llm_client.total_cost_usd:.4f}[/{DIM}]")
        if self._tool_calls > 0:
            success = self._tool_calls - self._tool_errors - self._tool_denied
            parts = [f"{success} ok"]
            if self._tool_errors > 0:
                parts.append(f"{self._tool_errors} x")
            if self._tool_denied > 0:
                parts.append(f"{self._tool_denied} denied")
            chat_log.write(
                f"  [{DIM}]{self._tool_calls} calls ({' | '.join(parts)})[/{DIM}]"
            )
        chat_log.write(
            f"  {styled(PROMPT_ICON, BOLD_PRIMARY)} {styled('Godspeed', BOLD_PRIMARY)}"
        )
