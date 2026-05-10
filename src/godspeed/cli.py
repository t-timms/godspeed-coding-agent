"""Click CLI entry point for Godspeed."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import click

from godspeed import __version__
from godspeed._bootstrap import _build_tool_registry, _load_env_files
from godspeed.config import DEFAULT_GLOBAL_DIR


def _force_utf8_stdio() -> None:
    """Make stdout/stderr survive non-ASCII output on legacy consoles.

    Default stdout encoding on Windows is often cp1252, which cannot encode
    common agent output characters (arrows, em-dashes, smart quotes) and
    crashes with UnicodeEncodeError after the agent already finished real
    work — masking success with a nonzero exit. We rewrap stdout/stderr in
    UTF-8 with errors='replace' so stray high-bit chars become '?' instead
    of crashing. Idempotent; only rewraps when the current encoding is not
    already utf-8.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "buffer"):
            continue
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc == "utf8":
            continue
        try:
            setattr(
                sys,
                name,
                io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                ),
            )
        except (AttributeError, OSError):
            continue


_force_utf8_stdio()

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/tags"
OLLAMA_STARTUP_TIMEOUT = 15  # seconds to wait for ollama to come up


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level.

    In verbose mode, only godspeed.* loggers get DEBUG — all third-party
    libraries stay at WARNING so they don't drown the TUI output.
    Logs go to stderr to avoid interleaving with Rich's stdout streaming.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )

    # Root logger: WARNING always (catches all third-party noise)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)

    # Godspeed loggers: DEBUG in verbose mode, WARNING otherwise
    godspeed_logger = logging.getLogger("godspeed")
    godspeed_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def _is_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(OLLAMA_URL, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
            return True
    except Exception:
        return False


def _ensure_ollama(console: Any | None = None) -> bool:
    """Start Ollama if it's not running. Returns True if Ollama is available.

    Args:
        console: Optional Rich Console for status output.
    """
    if _is_ollama_running():
        return True

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        if console is not None:
            from godspeed.tui.theme import WARNING

            console.print(
                f"[{WARNING}]  Ollama is not installed. "
                "Install from https://ollama.com or use a cloud model: "
                f"godspeed -m claude-sonnet-4-20250514[/{WARNING}]"
            )
        return False

    # Start ollama serve as a detached background process
    if console is not None:
        from godspeed.tui.theme import DIM

        console.print(f"[{DIM}]  Starting Ollama...[/{DIM}]", end="")
    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        logger.warning("Failed to start Ollama: %s", exc)
        if console is not None:
            from godspeed.tui.theme import ERROR

            console.print(f" [{ERROR}]failed: {exc}[/{ERROR}]")
        return False

    # Poll until it's up
    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if _is_ollama_running():
            if console is not None:
                from godspeed.tui.theme import SUCCESS

                console.print(f" [{SUCCESS}]ready[/{SUCCESS}]")
            return True

    if console is not None:
        from godspeed.tui.theme import WARNING

        console.print(f" [{WARNING}]timed out. Ollama may still be starting.[/{WARNING}]")
    return False


LLAMACPP_STARTUP_TIMEOUT = 60  # seconds to wait for llama.cpp server


def _ensure_llamacpp(console: Any | None = None) -> bool:
    """Start llama.cpp server if it's not running. Returns True if available.

    Configures LiteLLM env vars to route openai/ models to the local server.
    This MUST be called before LiteLLM is imported to prevent data leaks.

    Args:
        console: Optional Rich Console for status output.
    """
    from godspeed.tools.llamacpp_manager import (
        configure_litellm_env,
        is_server_running,
        start_server,
    )

    # Always configure LiteLLM env before ANY import — critical for security.
    # Without this, openai/ models would route to the real OpenAI API.
    configure_litellm_env()

    if is_server_running():
        return True

    if console is not None:
        from godspeed.tui.theme import DIM

        console.print(f"[{DIM}]  Starting llama.cpp server...[/{DIM}]", end="")

    proc = start_server(timeout=LLAMACPP_STARTUP_TIMEOUT)
    if proc is not None or is_server_running():
        if console is not None:
            from godspeed.tui.theme import SUCCESS

            console.print(f" [{SUCCESS}]ready[/{SUCCESS}]")
        return True

    if console is not None:
        from godspeed.tui.theme import WARNING

        console.print(
            f" [{WARNING}]timed out. Build with: "
            f"python scripts/setup_qwen36_local.py --build-only[/{WARNING}]"
        )
    return False


async def _run_app(
    model: str,
    project_dir: Path,
    _verbose: bool,
    audit_dir: Path | None,
    permission_mode: str | None = None,
    execution_mode: str = "tool",
) -> None:
    """Wire up all components and launch the Textual TUI."""
    from godspeed.config import GodspeedSettings

    overrides: dict = {}
    if model:
        overrides["model"] = model
    if permission_mode:
        overrides["permission_mode"] = permission_mode
    if execution_mode:
        overrides["execution_mode"] = execution_mode
    settings = GodspeedSettings(**overrides)

    effective_model = model or settings.model
    effective_project_dir = project_dir.resolve()
    session_id = str(uuid4())

    registry, risk_levels = _build_tool_registry()

    from godspeed.tui.textual_app import GodspeedTextualApp

    app = GodspeedTextualApp(
        settings=settings,
        registry=registry,
        risk_levels=risk_levels,
        effective_model=effective_model,
        effective_project_dir=effective_project_dir,
        session_id=session_id,
        permission_mode=permission_mode,
        execution_mode=execution_mode or "tool",
        audit_dir=audit_dir,
    )
    await app.run_async()


@click.group(invoke_without_command=True)
@click.option("--model", "-m", default="", help="Model to use (e.g. claude-sonnet-4-20250514).")
@click.option(
    "--project-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Project directory (default: current directory).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--audit-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for audit logs (default: ~/.godspeed/audit).",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["strict", "normal", "yolo"]),
    default=None,
    help=(
        "Permission mode: 'strict' (deny most, ask for everything), "
        "'normal' (default deny-first with allow rules), "
        "'yolo' (no permission checks, maximum speed)."
    ),
)
@click.option(
    "--execution-mode",
    type=click.Choice(["tool", "codeact"]),
    default=None,
    help="Execution mode: 'tool' (default, use tool calls), 'codeact' (write code blocks).",
)
@click.pass_context
def main(
    ctx: click.Context,
    model: str,
    project_dir: Path,
    verbose: bool,
    audit_dir: Path | None,
    permission_mode: str | None,
    execution_mode: str | None,
) -> None:
    """Godspeed -- Trusted production coding agent."""
    _setup_logging(verbose)

    # Auto-load env files so API keys in ~/.godspeed/.env.local (and the
    # project's .godspeed/.env.local) reach LiteLLM without requiring
    # per-shell env configuration. Shell env still wins — see
    # _load_env_files for the precedence rules.
    _load_env_files(project_dir=project_dir)

    # Store params for subcommands
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["project_dir"] = project_dir
    ctx.obj["verbose"] = verbose
    ctx.obj["audit_dir"] = audit_dir
    ctx.obj["permission_mode"] = permission_mode

    # If no subcommand, launch the TUI
    if ctx.invoked_subcommand is None:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(
                _run_app(
                    model,
                    project_dir,
                    verbose,
                    audit_dir,
                    permission_mode,
                    execution_mode or "tool",
                )
            )


@main.command()
def version() -> None:
    """Show Godspeed version."""
    from rich.console import Console as RichConsole

    from godspeed.tui.theme import brand

    c = RichConsole()
    c.print(brand(__version__))


@main.group()
def audit() -> None:
    """Audit trail commands."""


@audit.command("verify")
@click.argument("session_id", required=False, default=None)
@click.option(
    "--audit-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing audit logs.",
)
def audit_verify(session_id: str | None, audit_dir: Path | None) -> None:
    """Verify the hash chain integrity of an audit session.

    If SESSION_ID is not provided, verifies all sessions in the audit directory.
    """
    from rich.console import Console as RichConsole

    from godspeed.audit.trail import AuditTrail
    from godspeed.tui.theme import DIM, ERROR, SUCCESS

    c = RichConsole()
    effective_dir = audit_dir or (DEFAULT_GLOBAL_DIR / "audit")

    if not effective_dir.exists():
        c.print(f"[{ERROR}]Audit directory not found: {effective_dir}[/{ERROR}]")
        sys.exit(1)

    if session_id:
        # Verify single session
        trail = AuditTrail(log_dir=effective_dir, session_id=session_id)
        if not trail.log_path.exists():
            c.print(f"[{ERROR}]No audit log found for session: {session_id}[/{ERROR}]")
            sys.exit(1)
        is_valid, message = trail.verify_chain()
        if is_valid:
            c.print(f"[{SUCCESS}]VALID[/{SUCCESS}] -- {message}")
        else:
            c.print(f"[{ERROR}]BROKEN[/{ERROR}] -- {message}")
            sys.exit(1)
    else:
        # Verify all sessions
        found = False
        for log_file in sorted(effective_dir.glob("*.audit.jsonl")):
            found = True
            sid = log_file.stem.replace(".audit", "")
            trail = AuditTrail(log_dir=effective_dir, session_id=sid)
            is_valid, message = trail.verify_chain()
            status = f"[{SUCCESS}]VALID[/{SUCCESS}]" if is_valid else f"[{ERROR}]BROKEN[/{ERROR}]"
            c.print(f"  {status}  {sid[:12]}...  {message}")

        if not found:
            c.print(f"[{DIM}]No audit logs found.[/{DIM}]")


@main.command()
def init() -> None:
    """Set up Godspeed — create ~/.godspeed/ and default settings.yaml."""

    from rich.console import Console as RichConsole

    from godspeed.tui.theme import BOLD_PRIMARY, DIM, PRIMARY

    c = RichConsole()
    global_dir = DEFAULT_GLOBAL_DIR
    settings_path = global_dir / "settings.yaml"
    c.print(f"\n  [{BOLD_PRIMARY}]Welcome to Godspeed![/{BOLD_PRIMARY}]")
    c.print(f"  [{DIM}]The security-first coding agent[/{DIM}]")
    c.print()
    c.print(f"  [{DIM}]First-time setup:[/{DIM}]")
    c.print(f"    1. Install a local model: [{PRIMARY}]ollama pull qwen3:4b[/{PRIMARY}]")
    c.print(f"    2. Or set an API key:     [{PRIMARY}]export ANTHROPIC_API_KEY=sk-...[/{PRIMARY}]")
    c.print(f"    3. Edit your settings:    [{PRIMARY}]{settings_path}[/{PRIMARY}]")
    c.print(f"    4. Launch Godspeed:        [{PRIMARY}]godspeed[/{PRIMARY}]")


@main.command("run")
@click.argument("task", required=False, default="")
@click.option("--model", "-m", default="", help="Model to use.")
@click.option(
    "--project-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Project directory.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--auto-approve",
    type=click.Choice(["reads", "all", "none"]),
    default="reads",
    help="Auto-approve permission level (default: reads).",
)
@click.option("--max-iterations", type=int, default=50, help="Max agent loop iterations.")
@click.option(
    "--timeout",
    type=int,
    default=0,
    help="Wall-clock session timeout in seconds (0 = no limit).",
)
@click.option("--json-output", is_flag=True, help="Output result as JSON.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read the task from a file instead of the positional argument.",
)
@click.option(
    "--competition-mode",
    is_flag=True,
    help=(
        "Strip non-essential features for benchmark runs "
        "(no compaction, auto-stash, auto-commit, must-fix)."
    ),
)
def headless_run(
    task: str,
    model: str,
    project_dir: Path,
    verbose: bool,
    auto_approve: str,
    max_iterations: int,
    timeout: int,
    json_output: bool,
    prompt_file: Path | None,
    competition_mode: bool,
) -> None:
    """Run a task non-interactively (headless/CI mode).

    Executes the agent loop without a TUI. Permissions are auto-approved
    based on --auto-approve level. Outputs the final response to stdout.

    Task input precedence:
        1. --prompt-file FILE          (read task from file)
        2. TASK positional argument    (passed as argument)
        3. stdin                       (when TASK is '-' or omitted and stdin is a pipe)

    Exit codes:
        0   success — model stopped with a final text response
        1   tool error — final response starts with "Error:"
        2   max iterations reached without the model stopping
        3   cost budget exceeded
        4   LLM provider failure (all fallbacks exhausted)
        5   invalid input (no task provided)
        6   wall-clock timeout (--timeout exceeded)
        130 keyboard interrupt (SIGINT)

    Examples:
        godspeed run "Fix the failing test in test_auth.py"
        godspeed run --prompt-file experiment.prompt --json-output
        cat task.md | godspeed run -
        godspeed run "Long running task" --timeout 1800
    """
    from godspeed.agent.result import ExitCode

    _setup_logging(verbose)

    resolved_task = _resolve_task_input(task, prompt_file)
    if not resolved_task:
        sys.stderr.write(
            "Error: No task provided. Pass a positional argument, use --prompt-file, "
            "or pipe via stdin with `godspeed run -`.\n"
        )
        sys.exit(ExitCode.INVALID_INPUT)

    try:
        exit_code = asyncio.run(
            _headless_run(
                resolved_task,
                model,
                project_dir,
                auto_approve,
                max_iterations,
                timeout,
                json_output,
                competition_mode,
            )
        )
        sys.exit(int(exit_code))
    except KeyboardInterrupt:
        sys.exit(ExitCode.INTERRUPTED)


@main.command("serve")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Alternate settings.yaml path.",
)
def serve(config: Path | None) -> None:
    """Run Godspeed as an MCP stdio server."""
    from godspeed.mcp_server.server import run_server

    try:
        exit_code = run_server(config_path=config)
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)


@main.command("web")
@click.option(
    "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)."
)
@click.option(
    "--port", default=8000, type=int, help="Port to listen on (default: 8000)."
)
@click.option(
    "--model", "-m", default="", help="Model override."
)
@click.option(
    "--project-dir", "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Project directory.",
)
def web_serve(host: str, port: int, model: str, project_dir: Path) -> None:
    """Run Godspeed TUI as a web app accessible from any browser."""
    import os

    from rich.console import Console as RichConsole

    from godspeed.tui.theme import BOLD_PRIMARY, DIM

    c = RichConsole()
    c.print()
    c.print(f"  [{BOLD_PRIMARY}]Godspeed Web[/{BOLD_PRIMARY}]")
    c.print(f"  [{DIM}]http://{host}:{port}[/{DIM}]")
    c.print(f"  [{DIM}]Press Ctrl+C to stop[/{DIM}]")
    c.print()

    os.environ["TEXTUAL_DRIVER"] = "web"
    os.environ["TEXTUAL_WEB_HOST"] = host
    os.environ["TEXTUAL_WEB_PORT"] = str(port)

    try:
        asyncio.run(
            _run_app(
                model,
                project_dir,
                _verbose=False,
                audit_dir=None,
            )
        )
    except KeyboardInterrupt:
        c.print(f"\n  [{DIM}]Server stopped.[/{DIM}]")


def _resolve_task_input(task_arg: str, prompt_file: Path | None) -> str:
    """Resolve the task text from the available sources (in precedence order).

    Returns an empty string when no task is available — the caller treats
    that as INVALID_INPUT.
    """
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    if task_arg == "-":
        return sys.stdin.read().strip()
    if task_arg:
        return task_arg
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


async def _headless_run(
    task: str,
    model: str,
    project_dir: Path,
    auto_approve: str,
    max_iterations: int,
    timeout: int,
    json_output: bool,
    competition_mode: bool = False,
) -> int:
    """Execute the headless agent loop.

    Returns an ExitCode-compatible integer:
        0 SUCCESS, 1 TOOL_ERROR, 2 MAX_ITERATIONS, 3 BUDGET_EXCEEDED,
        4 LLM_ERROR, 6 TIMEOUT. Never returns INTERRUPTED (130) — the
        caller translates KeyboardInterrupt.
    """
    import json as json_module

    from godspeed.agent.conversation import Conversation
    from godspeed.agent.loop import agent_loop
    from godspeed.agent.result import AgentMetrics, ExitCode, ExitReason
    from godspeed.agent.system_prompt import build_system_prompt
    from godspeed.audit.trail import AuditTrail
    from godspeed.config import GodspeedSettings
    from godspeed.context.project_instructions import load_project_instructions
    from godspeed.llm.client import LLMClient, ModelRouter
    from godspeed.security.permissions import ALLOW, PermissionDecision, PermissionEngine
    from godspeed.tools.base import RiskLevel, ToolContext

    overrides: dict = {}
    if model:
        overrides["model"] = model
    settings = GodspeedSettings(**overrides)

    effective_model = model or settings.model
    effective_project_dir = project_dir.resolve()
    session_id = str(uuid4())

    # Audit trail — headless must have an audit trail by default. A
    # security-first agent without a tamper-evident log in unattended mode
    # is the worst of both worlds.
    audit_dir = settings.global_dir / "audit"
    audit_trail = AuditTrail(log_dir=audit_dir, session_id=session_id)
    audit_trail.record(
        event_type="session_start",
        detail={
            "mode": "headless",
            "task": task[:500],  # clipped; full task goes to training logger
            "model": effective_model,
            "auto_approve": auto_approve,
        },
    )

    # Tools
    registry, risk_levels = _build_tool_registry()

    # Permission engine with headless auto-approve
    permission_engine = PermissionEngine(
        deny_patterns=settings.permissions.deny,
        allow_patterns=settings.permissions.allow,
        ask_patterns=settings.permissions.ask,
        tool_risk_levels=risk_levels,
    )

    # Wrap with headless auto-approve proxy
    class _HeadlessPermissionProxy:
        """Auto-approve permissions based on configured level."""

        def __init__(self, engine: PermissionEngine, level: str) -> None:
            self._engine = engine
            self._level = level

        def evaluate(self, tool_call: Any) -> PermissionDecision:
            decision = self._engine.evaluate(tool_call)
            if decision == "deny":
                return decision
            if self._level == "all":
                return PermissionDecision(ALLOW, "headless: auto-approved (all)")
            if self._level == "reads":
                tool_risk = risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
                if tool_risk == RiskLevel.READ_ONLY:
                    return PermissionDecision(ALLOW, "headless: auto-approved (reads)")
                return PermissionDecision(ALLOW, "headless: auto-approved (reads+low)")
            return decision

    # Auto-start local inference servers if needed
    if effective_model.lower().startswith("ollama"):
        _ensure_ollama()
    elif effective_model.lower().startswith(("llamacpp/", "openai/")):
        _ensure_llamacpp()

    project_instructions = load_project_instructions(
        effective_project_dir,
        settings.context.project_instructions,
    )
    system_prompt = build_system_prompt(
        tools=registry.list_tools(),
        project_instructions=project_instructions,
        cwd=effective_project_dir,
    )

    router = ModelRouter(routing=settings.routing) if settings.routing else None
    llm_client = LLMClient(
        model=effective_model,
        fallback_models=settings.fallback_models,
        router=router,
        thinking_budget=settings.thinking_budget,
        max_cost_usd=settings.max_cost_usd,
        reasoning_effort=settings.reasoning_effort,
    )

    # Tool context — constructed after the LLM client so tools that need an
    # LLM (e.g. generate_tests) can invoke it via the context.
    tool_context = ToolContext(
        cwd=effective_project_dir,
        session_id=session_id,
        permissions=_HeadlessPermissionProxy(permission_engine, auto_approve),
        audit=audit_trail,
        llm_client=llm_client,  # type: ignore[arg-type]
    )

    # Kick off background codebase index if needed (non-blocking).
    from godspeed.context.auto_index import maybe_start_auto_index

    maybe_start_auto_index(effective_project_dir, settings.auto_index)

    # Conversation logger (training data collection)
    conversation_logger = None
    if settings.log_conversations and not competition_mode:
        from godspeed.training.conversation_logger import ConversationLogger

        training_dir = settings.global_dir / "training"
        conversation_logger = ConversationLogger(
            session_id=session_id,
            output_dir=training_dir,
        )

    conversation = Conversation(
        system_prompt=system_prompt,
        model=effective_model,
        max_tokens=settings.max_context_tokens,
        compaction_threshold=settings.compaction_threshold,
        conversation_logger=conversation_logger,
    )

    # Callbacks — write to stderr for tool activity, keep stdout for result
    def on_tool_call(name: str, args: dict) -> None:  # noqa: ARG001
        logger.info("Tool call: %s", name)
        if not json_output:
            sys.stderr.write(f"[tool] {name}\n")

    def on_tool_result(name: str, result: Any) -> None:
        is_error = getattr(result, "is_error", False)
        if is_error:
            logger.warning("Tool error: %s", name)

    metrics = AgentMetrics()
    timed_out = False
    final_text: str

    # Run agent loop, optionally under a wall-clock timeout.
    loop_coro = agent_loop(
        user_input=task,
        conversation=conversation,
        llm_client=llm_client,
        tool_registry=registry,
        tool_context=tool_context,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        max_iterations=max_iterations,
        metrics=metrics,
        competition_mode=competition_mode,
    )
    try:
        if timeout > 0:
            final_text = await asyncio.wait_for(loop_coro, timeout=timeout)
        else:
            final_text = await loop_coro
    except TimeoutError:
        timed_out = True
        final_text = f"Error: Session exceeded wall-clock timeout of {timeout}s."
        metrics.finalize(ExitReason.TIMEOUT)

    # Resolve final exit code. TOOL_ERROR overrides STOPPED when the model's
    # final message starts with "Error:" (common pattern for tool failures
    # that the model surfaces rather than silently ignoring).
    exit_code = int(metrics.exit_code)
    if (
        not timed_out
        and metrics.exit_reason == ExitReason.STOPPED
        and final_text.startswith("Error:")
    ):
        exit_code = int(ExitCode.TOOL_ERROR)
        metrics.exit_reason = ExitReason.TOOL_ERROR

    audit_trail.record(
        event_type="session_end",
        detail={
            "exit_reason": metrics.exit_reason.value,
            "exit_code": exit_code,
            "iterations_used": metrics.iterations_used,
            "tool_call_count": metrics.tool_call_count,
            "tool_error_count": metrics.tool_error_count,
            "must_fix_injections": metrics.must_fix_injections,
            "duration_seconds": round(metrics.duration_seconds, 3),
            "cost_usd": round(llm_client.total_cost_usd, 6),
        },
        outcome="success" if exit_code == 0 else "error",
    )

    # Mirror the audit session_end into the training log so RL pipelines
    # can read exit_reason/exit_code without parsing the audit trail.
    if conversation_logger is not None:
        conversation_logger.log_session_end(
            exit_reason=metrics.exit_reason.value,
            exit_code=exit_code,
            iterations_used=metrics.iterations_used,
            tool_call_count=metrics.tool_call_count,
            tool_error_count=metrics.tool_error_count,
            must_fix_injections=metrics.must_fix_injections,
            duration_seconds=metrics.duration_seconds,
            cost_usd=llm_client.total_cost_usd,
        )
        conversation_logger.close()

    # Output result to stdout
    if json_output:
        output = {
            "task": task,
            "model": effective_model,
            "session_id": session_id,
            "response": final_text,
            "exit_reason": metrics.exit_reason.value,
            "exit_code": exit_code,
            "iterations_used": metrics.iterations_used,
            "tool_calls": [{"name": tc.name, "is_error": tc.is_error} for tc in metrics.tool_calls],
            "tool_call_count": metrics.tool_call_count,
            "tool_error_count": metrics.tool_error_count,
            "must_fix_injections": metrics.must_fix_injections,
            "duration_seconds": round(metrics.duration_seconds, 3),
            "input_tokens": llm_client.total_input_tokens,
            "output_tokens": llm_client.total_output_tokens,
            "cost_usd": round(llm_client.total_cost_usd, 6),
            "audit_log_path": str(audit_trail.log_path),
        }
        sys.stdout.write(json_module.dumps(output, indent=2) + "\n")
    else:
        sys.stdout.write(final_text + "\n")

    return exit_code


@main.command()
def models() -> None:
    """Show popular model options, presets, and how to configure them."""
    from rich.console import Console as RichConsole
    from rich.table import Table

    from godspeed.config import GodspeedSettings
    from godspeed.tui.theme import BOLD_PRIMARY, DIM, NEUTRAL, SUCCESS, TABLE_BORDER

    c = RichConsole()

    presets = GodspeedSettings.MODEL_PRESETS

    c.print(f"\n  [{BOLD_PRIMARY}]Model Presets[/{BOLD_PRIMARY}]")
    c.print(f"  [{DIM}]Use --preset or /model <preset> to switch.[/{DIM}]\n")
    preset_table = Table(border_style=TABLE_BORDER, expand=False)
    preset_table.add_column("Preset", style=BOLD_PRIMARY)
    preset_table.add_column("Model", style=NEUTRAL)
    preset_table.add_column("Description")
    preset_descriptions = {
        "local": "Qwen2.5-Coder 14B + GPU spec dec, ~750 tok/s (default)",
        "zaya": "ZAYA1-8B NF4 (thinking), 0.7B active/8B total, 7.2GB VRAM",
    }
    for name, model in presets.items():
        desc = preset_descriptions.get(name, "")
        preset_table.add_row(name, model, desc)
    c.print(preset_table)

    c.print()
    table = Table(title="Popular Models", border_style=TABLE_BORDER, expand=False)
    table.add_column("Model", style=BOLD_PRIMARY)
    table.add_column("Provider", style=NEUTRAL)
    table.add_column("Cost")
    table.add_column("API Key Env Var", style=NEUTRAL)

    free = f"[{SUCCESS}]Free[/{SUCCESS}]"
    table.add_row("ollama/rnj-1:8b", "Ollama", free, "None (local, 5.1GB)")
    table.add_row("ollama/qwen2.5-coder:14b", "Ollama", free, "None (local, 9GB)")
    table.add_row("ollama/devstral-small-2:24b", "Ollama", free, "None (local, 15GB)")
    table.add_row("ollama/qwen3:4b", "Ollama", free, "None (local)")
    table.add_row("ollama/deepseek-r1:8b", "Ollama", free, "None (local)")
    table.add_row("ollama/mistral:7b", "Ollama", free, "None (local)")

    table.add_row("claude-sonnet-4-20250514", "Anthropic", "Paid", "ANTHROPIC_API_KEY")
    table.add_row("claude-opus-4-20250514", "Anthropic", "Paid", "ANTHROPIC_API_KEY")
    table.add_row("gpt-4o", "OpenAI", "Paid", "OPENAI_API_KEY")
    table.add_row("gpt-4o-mini", "OpenAI", "Paid", "OPENAI_API_KEY")
    table.add_row("gemini/gemini-2.0-flash", "Google", "Paid", "GEMINI_API_KEY")
    table.add_row("deepseek/deepseek-chat", "DeepSeek", "Paid", "DEEPSEEK_API_KEY")

    c.print(table)
    c.print()
    c.print(f"  [{BOLD_PRIMARY}]Switch models:[/{BOLD_PRIMARY}]")
    c.print(f"    [{DIM}]CLI flag:[/{DIM}]     godspeed -m claude-sonnet-4-20250514")
    c.print(
        f"    [{DIM}]Preset:[/{DIM}]       godspeed -m fast (or balanced/quality/cloud/frontier)"
    )
    c.print(f"    [{DIM}]Env var:[/{DIM}]      GODSPEED_MODEL=gpt-4o godspeed")
    c.print(f"    [{DIM}]Settings:[/{DIM}]     Edit ~/.godspeed/settings.yaml")
    c.print(f"    [{DIM}]At runtime:[/{DIM}]   /model fast  or  /model ollama/rnj-1:8b")


@main.group()
def ollama() -> None:
    """Manage local Ollama models — list, pull, show, delete, scan."""


@ollama.command("list")
def ollama_list() -> None:
    """List locally installed Ollama models."""
    from rich.console import Console as RichConsole

    from godspeed.tools.ollama_manager import list_models
    from godspeed.tui.theme import BOLD_PRIMARY, DIM, ERROR

    c = RichConsole()
    models_list = list_models()
    if not models_list:
        best = "rnj-1:8b"
        c.print(f"[{ERROR}]No local models found.[/{ERROR}]")
        c.print(f"  [{DIM}]Install Ollama from https://ollama.com, then:[/{DIM}]")
        c.print(f"  [{BOLD_PRIMARY}]godspeed ollama pull {best}[/{BOLD_PRIMARY}]")
        return

    c.print(f"\n  [{BOLD_PRIMARY}]Installed Ollama Models[/{BOLD_PRIMARY}] ({len(models_list)})\n")
    for m in sorted(models_list, key=lambda x: x.name):
        c.print(f"  {m.name:40s} {m.size_gb:5.1f} GB")


@ollama.command("pull")
@click.argument("model")
def ollama_pull(model: str) -> None:
    """Pull a model from Ollama (e.g. godspeed ollama pull rnj-1:8b)."""
    from rich.console import Console as RichConsole

    from godspeed.tools.ollama_manager import pull_model
    from godspeed.tui.theme import BOLD_PRIMARY, ERROR, SUCCESS

    c = RichConsole()
    c.print(f"  Pulling [{BOLD_PRIMARY}]{model}[/{BOLD_PRIMARY}]...")
    success = pull_model(model)
    if success:
        c.print(f"  [{SUCCESS}]Successfully pulled {model}[/{SUCCESS}]")
    else:
        c.print(
            f"  [{ERROR}]Failed to pull {model}. Check the model name and Ollama status.[/{ERROR}]"
        )


@ollama.command("show")
@click.argument("model")
def ollama_show(model: str) -> None:
    """Show detailed info about an installed Ollama model."""
    from rich.console import Console as RichConsole

    from godspeed.tools.ollama_manager import show_model
    from godspeed.tui.theme import BOLD_PRIMARY, DIM, ERROR

    c = RichConsole()
    info = show_model(model)
    if info is None:
        c.print(f"[{ERROR}]Model {model!r} not found locally.[/{ERROR}]")
        c.print(f"  [{DIM}]Pull it first: godspeed ollama pull {model}[/{DIM}]")
        return

    c.print(f"\n  [{BOLD_PRIMARY}]{model}[/{BOLD_PRIMARY}]")
    for key, value in sorted(info.items()):
        if key != "name":
            c.print(f"  {key:20s} {value}")


@ollama.command("delete")
@click.argument("model")
def ollama_delete(model: str) -> None:
    """Delete a local Ollama model to free disk space."""
    from rich.console import Console as RichConsole

    from godspeed.tools.ollama_manager import delete_model
    from godspeed.tui.theme import ERROR, SUCCESS

    c = RichConsole()
    success, message = delete_model(model)
    if success:
        c.print(f"  [{SUCCESS}]{message}[/{SUCCESS}]")
    else:
        c.print(f"  [{ERROR}]Failed: {message}[/{ERROR}]")


@main.command("scan")
def scan_machine() -> None:
    """Scan your machine hardware and recommend optimal models per preset tier."""
    from rich.console import Console as RichConsole

    from godspeed.evolution.hardware import format_machine_report
    from godspeed.tui.theme import BOLD_PRIMARY, DIM

    c = RichConsole()
    report = format_machine_report()
    c.print(report)
    c.print()
    c.print(
        f"  [{DIM}]Run"
        f" [{BOLD_PRIMARY}]godspeed ollama pull <model>[/{BOLD_PRIMARY}]"
        f"{DIM}] to install a model.[/{DIM}]"
    )
    c.print(
        f"  [{DIM}]Use"
        f" [{BOLD_PRIMARY}]/model <preset>[/{BOLD_PRIMARY}]"
        f"{DIM}] or"
        f" [{BOLD_PRIMARY}]godspeed -m <model>[/{BOLD_PRIMARY}]"
        f"{DIM}] to switch.[/{DIM}]"
    )


@main.command("doctor")
@click.option(
    "--fix",
    is_flag=True,
    help="Attempt to fix writable issues (create missing dirs).",
)
def doctor(fix: bool) -> None:
    """Diagnose setup issues — Ollama, API keys, audit dir, permissions.

    Checks:
      - Ollama running and responsive
      - API keys present and valid (lightweight probe)
      - Audit directory writable
      - Permission mode sanity
    """
    from rich.console import Console as RichConsole
    from rich.table import Table

    from godspeed.config import DEFAULT_GLOBAL_DIR
    from godspeed.tui.theme import BOLD_ERROR, BOLD_SUCCESS, ERROR, SUCCESS, WARNING

    c = RichConsole()
    c.print("\n  [bold]Godspeed Doctor — System Check[/bold]\n")

    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Check", style="dim", width=30)
    table.add_column("Status", width=10)
    table.add_column("Detail", style="dim")

    all_ok = True

    # ── 1. Ollama ──────────────────────────────────────────────────────
    ollama_ok = _is_ollama_running()
    if ollama_ok:
        # Try to list models as a deeper check
        try:
            from godspeed.tools.ollama_manager import list_models

            models = list_models()
            table.add_row(
                "Ollama server",
                f"[{SUCCESS}]ok[/{SUCCESS}]",
                f"running, {len(models)} model(s) installed",
            )
        except Exception as exc:
            table.add_row(
                "Ollama server",
                f"[{WARNING}]warn[/{WARNING}]",
                f"running but model list failed: {exc}",
            )
            all_ok = False
    else:
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            detail = "not running — start with 'ollama serve' or 'godspeed' will auto-start"
        else:
            detail = "not installed — install from https://ollama.com"
        table.add_row("Ollama server", f"[{ERROR}]x[/{ERROR}]", detail)
        all_ok = False

    # ── 2. API keys ────────────────────────────────────────────────────
    import yaml

    catalog_path = Path(__file__).parent / "llm" / "driver_catalog.yaml"
    required_env_vars: dict[str, str] = {}
    if catalog_path.exists():
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        for driver_cfg in (catalog.get("drivers") or {}).values():
            env_var = driver_cfg.get("requires_env")
            if env_var and env_var not in required_env_vars:
                required_env_vars[env_var] = driver_cfg.get("provider", "unknown")

    if required_env_vars:
        for env_var, provider in sorted(required_env_vars.items()):
            value = os.environ.get(env_var)
            if not value:
                table.add_row(
                    f"API key: {env_var}",
                    f"[{WARNING}]![/{WARNING}]",
                    f"not set — {provider} cloud models unavailable",
                )
                all_ok = False
            else:
                # Lightweight validation: try a minimal LiteLLM call
                try:
                    import litellm

                    # Map env var to provider for a quick models-list probe
                    if provider == "anthropic":
                        litellm.validate_environment("anthropic/claude-3-haiku-20240307")
                        table.add_row(
                            f"API key: {env_var}",
                            f"[{SUCCESS}]ok[/{SUCCESS}]",
                            f"{provider} — key present and accepted",
                        )
                    elif provider == "nvidia_nim":
                        table.add_row(
                            f"API key: {env_var}",
                            f"[{SUCCESS}]ok[/{SUCCESS}]",
                            f"{provider} — key present (NIM free-tier)",
                        )
                    elif provider == "moonshot":
                        table.add_row(
                            f"API key: {env_var}",
                            f"[{SUCCESS}]ok[/{SUCCESS}]",
                            f"{provider} — key present",
                        )
                    else:
                        table.add_row(
                            f"API key: {env_var}",
                            f"[{SUCCESS}]ok[/{SUCCESS}]",
                            f"{provider} — key present",
                        )
                except Exception as exc:
                    table.add_row(
                        f"API key: {env_var}",
                        f"[{ERROR}]x[/{ERROR}]",
                        f"{provider} — validation failed: {exc}",
                    )
                    all_ok = False
    else:
        table.add_row("API keys", f"[{WARNING}]![/{WARNING}]", "driver catalog not found")

    # ── 3. Audit directory ─────────────────────────────────────────────
    audit_dir = DEFAULT_GLOBAL_DIR / "audit"
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
        probe = audit_dir / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        table.add_row(
            "Audit directory",
            f"[{SUCCESS}]ok[/{SUCCESS}]",
            f"{audit_dir} — writable",
        )
    except (OSError, PermissionError) as exc:
        table.add_row(
            "Audit directory",
            f"[{ERROR}]x[/{ERROR}]",
            f"{audit_dir} — not writable: {exc}",
        )
        if fix:
            try:
                audit_dir.mkdir(parents=True, exist_ok=True)
                audit_dir.chmod(0o700)
                table.add_row(
                    "Audit directory (fix)",
                    f"[{SUCCESS}]ok[/{SUCCESS}]",
                    f"{audit_dir} — permissions fixed",
                )
            except Exception as fix_exc:
                table.add_row(
                    "Audit directory (fix)",
                    f"[{ERROR}]x[/{ERROR}]",
                    f"could not fix: {fix_exc}",
                )
        all_ok = False

    # ── 4. Permission mode sanity ──────────────────────────────────────
    from godspeed.config import GodspeedSettings

    try:
        settings = GodspeedSettings()
        mode = settings.permission_mode
        if mode == "yolo":
            table.add_row(
                "Permission mode",
                f"[{WARNING}]![/{WARNING}]",
                f"'{mode}' — all permission checks disabled",
            )
        else:
            table.add_row(
                "Permission mode",
                f"[{SUCCESS}]ok[/{SUCCESS}]",
                f"'{mode}' — secure",
            )
    except Exception as exc:
        table.add_row("Permission mode", f"[{ERROR}]x[/{ERROR}]", f"could not read config: {exc}")
        all_ok = False

    c.print(table)
    c.print()

    if all_ok:
        c.print(f"  [{BOLD_SUCCESS}]All checks passed — system ready.[{BOLD_SUCCESS}]")
    else:
        c.print(f"  [{BOLD_ERROR}]Some checks failed — review details above.[{BOLD_ERROR}]")
    c.print()


@main.command("export-training")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["openai", "chatml", "sharegpt"]),
    default="openai",
    help="Output format (default: openai).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("training_data.jsonl"),
    help="Output file path (default: training_data.jsonl).",
)
@click.option("--success-only", is_flag=True, help="Only export sessions with no tool errors.")
@click.option("--min-tools", type=int, default=1, help="Minimum tool calls per session.")
@click.option("--min-turns", type=int, default=2, help="Minimum user turns per session.")
@click.option("--max-sessions", type=int, default=0, help="Max sessions to export (0=all).")
@click.option(
    "--tools",
    type=str,
    default=None,
    help="Comma-separated tool names to filter by (e.g. 'file_read,file_edit').",
)
@click.option(
    "--max-tool-output",
    type=int,
    default=2000,
    help="Max chars per tool output (default: 2000).",
)
def export_training(
    fmt: str,
    output: Path,
    success_only: bool,
    min_tools: int,
    min_turns: int,
    max_sessions: int,
    tools: str | None,
    max_tool_output: int,
) -> None:
    """Export conversation logs to fine-tuning JSONL.

    Reads conversation logs from ~/.godspeed/training/ and converts them
    to the specified format for LLM fine-tuning.

    Examples:
        godspeed export-training --format openai --output training.jsonl
        godspeed export-training --format sharegpt --success-only --min-tools 3
        godspeed export-training --format chatml --tools file_read,file_edit
    """
    from rich.console import Console as RichConsole

    from godspeed.training.exporter import ExportFilters, TrainingExporter
    from godspeed.tui.theme import DIM, ERROR, SUCCESS

    c = RichConsole()
    training_dir = DEFAULT_GLOBAL_DIR / "training"

    if not training_dir.exists():
        c.print(f"[{ERROR}]No training data found at {training_dir}[/{ERROR}]")
        c.print(f"[{DIM}]Enable conversation logging in settings (log_conversations: true)[/{DIM}]")
        sys.exit(1)

    tool_list = [t.strip() for t in tools.split(",")] if tools else None

    filters = ExportFilters(
        min_tool_calls=min_tools,
        success_only=success_only,
        min_turns=min_turns,
        tools=tool_list,
        max_sessions=max_sessions,
    )

    exporter = TrainingExporter()
    stats = exporter.export_all(
        training_dir=training_dir,
        output_path=output,
        fmt=fmt,
        filters=filters,
        max_tool_output=max_tool_output,
    )

    c.print(f"[{SUCCESS}]Export complete[/{SUCCESS}]")
    c.print(f"  Format:   {fmt}")
    c.print(f"  Output:   {output}")
    c.print(f"  Sessions: {stats.sessions_exported}/{stats.sessions_scanned} exported")
    c.print(f"  Filtered: {stats.sessions_filtered}")
    c.print(f"  Messages: {stats.total_messages}")
    c.print(f"  Tool calls: {stats.total_tool_calls}")
    if stats.errors:
        c.print(f"  [{ERROR}]Errors: {len(stats.errors)}[/{ERROR}]")
        for err in stats.errors[:5]:
            c.print(f"    {err}")


# ── SWE-bench CLI group ────────────────────────────────────────────────


@main.group()
def swebench() -> None:
    """SWE-bench evaluation — run Godspeed on SWE-bench Lite and produce benchmark scores."""


@swebench.command("run")
@click.option(
    "--model", "-m",
    required=True,
    help="Model to evaluate (e.g. deepseek/deepseek-v4-pro).",
)
@click.option(
    "--run-id",
    default="",
    help="Identifier for this run (auto-generated if empty).",
)
@click.option(
    "--max-instances", "-n",
    type=int,
    default=300,
    help="Max number of instances to evaluate (default: 300 — full SWE-bench Lite).",
)
@click.option(
    "--max-workers", "-w",
    type=int,
    default=4,
    help="Max concurrent agent runs (default: 4).",
)
@click.option(
    "--timeout",
    type=int,
    default=900,
    help="Wall-clock timeout per instance in seconds (default: 900 = 15min).",
)
@click.option(
    "--agent-max-iterations",
    type=int,
    default=30,
    help="Max agent loop iterations per instance (default: 30).",
)
@click.option(
    "--tool-set",
    type=click.Choice(["swebench", "full", "local", "web"]),
    default="swebench",
    help="Tool set: swebench (12 tools), full (all 30+), local, web.",
)
@click.option(
    "--simple",
    "is_simple",
    is_flag=True,
    help="Simple mode: swebench tool set only (bash-heavy, matching mini-swe-agent v2).",
)
@click.option(
    "--complex",
    "is_complex",
    is_flag=True,
    help="Complex mode: full 30-tool Godspeed for A/B comparison.",
)
@click.option(
    "--competition/--no-competition",
    default=True,
    help="Strip non-essential features for benchmark runs (default: on).",
)
@click.option(
    "--docker/--no-docker",
    default=False,
    help="Run each instance in a Docker container (default: off).",
)
@click.option(
    "--skip-existing/--no-skip",
    default=True,
    help="Skip instances with existing predictions (default: on).",
)
@click.option(
    "--evaluate/--no-evaluate",
    default=False,
    help="Run Docker-based test evaluation after predictions (default: off).",
)
@click.option(
    "--instance-ids",
    default=None,
    help="Comma-separated instance IDs to run (overrides --max-instances).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for predictions (default: experiments/swebench_lite/predictions/).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def swebench_run(
    ctx: click.Context,
    model: str,
    run_id: str,
    max_instances: int,
    max_workers: int,
    timeout: int,
    agent_max_iterations: int,
    tool_set: str,
    is_simple: bool,
    is_complex: bool,
    competition: bool,
    docker: bool,
    skip_existing: bool,
    evaluate: bool,
    instance_ids: str | None,
    output_dir: Path | None,
    verbose: bool,
) -> None:
    """Run Godspeed on SWE-bench Lite and produce predictions.

    Runs the Godspeed agent on each SWE-bench Lite instance, captures the
    resulting git diff as a patch, and saves predictions in the standard
    SWE-bench JSONL format.

    Examples:
        godspeed swebench run -m deepseek/deepseek-v4-pro
        godspeed swebench run -m claude-sonnet-4-20250514 -n 10 -w 2 --simple
        godspeed swebench run -m gpt-4o --max-instances 50 --docker --evaluate
    """
    from godspeed._bootstrap import _load_env_files
    from godspeed.evaluation.swebench_harness import run_swebench_evaluation

    _setup_logging(verbose)

    # Auto-load env files so API keys reach LiteLLM
    project_dir = Path(ctx.obj.get("project_dir", ".")) if ctx.obj else Path(".")
    _load_env_files(project_dir=project_dir)

    # Resolve tool_set from flags
    if is_simple:
        tool_set = "swebench"
    elif is_complex:
        tool_set = "full"

    # Parse instance IDs
    ids_list: list[str] = []
    if instance_ids:
        ids_list = [i.strip() for i in instance_ids.split(",") if i.strip()]

    try:
        asyncio.run(
            run_swebench_evaluation(
                model=model,
                run_id=run_id,
                max_instances=max_instances,
                max_workers=max_workers,
                timeout_per_instance=timeout,
                agent_max_iterations=agent_max_iterations,
                tool_set=tool_set,
                competition_mode=competition,
                use_docker=docker,
                skip_existing=skip_existing,
                evaluate_after=evaluate,
                instance_ids=ids_list or None,
                output_dir=output_dir,
            )
        )
    except KeyboardInterrupt:
        from rich.console import Console as RichConsole

        from godspeed.tui.theme import WARNING

        RichConsole().print(
            f"[{WARNING}]Evaluation interrupted. "
            f"Predictions saved up to this point.[/{WARNING}]"
        )
        sys.exit(130)
    except ImportError as exc:
        from rich.console import Console as RichConsole

        from godspeed.tui.theme import ERROR

        RichConsole().print(f"[{ERROR}]Import error: {exc}[/{ERROR}]")
        RichConsole().print("  Ensure 'datasets' is installed: pip install datasets")
        sys.exit(1)


@swebench.command("list-instances")
@click.option(
    "--max", "-n",
    "max_count",
    type=int,
    default=20,
    help="Max instances to display (default: 20).",
)
@click.option(
    "--repo",
    default=None,
    help="Filter by repo (e.g. 'django/django').",
)
def swebench_list_instances(max_count: int, repo: str | None) -> None:
    """List available SWE-bench Lite instances."""
    from rich.console import Console as RichConsole
    from rich.table import Table

    from godspeed.evaluation.swebench_harness import load_swebench_lite
    from godspeed.tui.theme import BOLD_PRIMARY, NEUTRAL, TABLE_BORDER

    c = RichConsole()

    try:
        instances = load_swebench_lite(max_instances=max_count * 2)
    except ImportError as exc:
        c.print(f"[red]{exc}[/red]")
        return

    if repo:
        instances = [i for i in instances if i.repo == repo]

    instances = instances[:max_count]

    c.print(
        f"\n  [{BOLD_PRIMARY}]SWE-bench Lite Instances[/{BOLD_PRIMARY}]"
        f" ({len(instances)} shown)\n"
    )

    table = Table(border_style=TABLE_BORDER, expand=False)
    table.add_column("Instance ID", style=BOLD_PRIMARY)
    table.add_column("Repo", style=NEUTRAL)
    table.add_column("Base Commit")

    for inst in instances:
        table.add_row(inst.instance_id, inst.repo, inst.base_commit[:8])

    c.print(table)
    c.print()
    c.print(f"  [{NEUTRAL}]Total: {len(instances)} instances[/{NEUTRAL}]")


@swebench.command("eval")
@click.argument("predictions_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--max-workers", "-w",
    type=int,
    default=2,
    help="Max concurrent evaluation containers (default: 2).",
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    help="Timeout per evaluation container in seconds (default: 600).",
)
def swebench_eval(
    predictions_file: Path,
    max_workers: int,
    timeout: int,
) -> None:
    """Evaluate predictions against SWE-bench tests using Docker.

    Reads predictions from a JSONL file, applies each patch in a Docker
    container, runs the test suite, and reports resolved rates.

    PREDICTIONS_FILE: Path to the predictions JSONL file from `godspeed swebench run`.
    """
    import tempfile

    from rich.console import Console as RichConsole

    from godspeed.evaluation.swebench_harness import (
        SWEBenchInstance,
        SWEBenchPrediction,
        _docker_available,
        load_predictions,
        load_swebench_lite,
    )
    from godspeed.tui.theme import ERROR, SUCCESS, WARNING

    c = RichConsole()

    if not _docker_available():
        c.print(f"[{ERROR}]Docker is not available. Install Docker Desktop.[/{ERROR}]")
        sys.exit(1)

    predictions = load_predictions(predictions_file)
    if not predictions:
        c.print(f"[{ERROR}]No predictions found in {predictions_file}[/{ERROR}]")
        sys.exit(1)

    c.print(f"  Evaluating {len(predictions)} predictions...")

    # Load instances for repo/base_commit info
    all_instances = load_swebench_lite()
    instance_map: dict[str, SWEBenchInstance] = {
        i.instance_id: i for i in all_instances
    }

    async def _eval_predictions_async() -> None:
        from godspeed.evaluation.swebench_harness import _run_tests_in_docker

        semaphore = asyncio.Semaphore(max_workers)
        resolved = 0
        total = 0

        async def _eval_one(pred: SWEBenchPrediction) -> bool:
            nonlocal total
            async with semaphore:
                total += 1
                inst = instance_map.get(pred.instance_id)
                if inst is None:
                    c.print(
                        f"  [{WARNING}]Instance {pred.instance_id}"
                        f" not found in dataset[/{WARNING}]"
                    )
                    return False

                with tempfile.TemporaryDirectory() as tmpdir:
                    work_dir = Path(tmpdir)
                    is_resolved, report = _run_tests_in_docker(
                        inst, pred.model_patch, work_dir, timeout=timeout
                    )
                    status = (
                        f"[{SUCCESS}]RESOLVED[/{SUCCESS}]"
                        if is_resolved
                        else "[red]FAILED[/red]"
                    )
                    c.print(f"  [{total}/{len(predictions)}] {pred.instance_id}: {status}")
                    if report.get("total"):
                        c.print(
                            f"    Tests: {report.get('passed', 0)}"
                            f"/{report.get('total', 0)} passed"
                        )
                    return is_resolved

        results = await asyncio.gather(
            *[_eval_one(p) for p in predictions],
            return_exceptions=True,
        )

        for r in results:
            if r is True:
                resolved += 1

        if total > 0:
            rate = resolved / total * 100
            c.print()
            c.print(f"  [{SUCCESS}]Resolved: {resolved}/{total} ({rate:.1f}%)[/{SUCCESS}]")

    try:
        asyncio.run(_eval_predictions_async())
    except KeyboardInterrupt:
        c.print(f"\n[{WARNING}]Evaluation interrupted.[/{WARNING}]")
        sys.exit(130)
