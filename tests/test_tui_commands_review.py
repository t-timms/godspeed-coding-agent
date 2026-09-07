"""Tests for the diff review commands (/code-review, /security-review, /simplify, /effort)."""

from __future__ import annotations

import asyncio
import re
from io import StringIO
from pathlib import Path
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import godspeed.agent.review as review_module
from godspeed.agent.review import (
    ReviewDiff,
    collect_diff,
    findings_from_response,
    parse_review_args,
    review_prompt,
    security_scan_files,
)
from godspeed.agent.conversation import Conversation
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(fn, *args, **kwargs) -> str:
    """Run a sync callable and capture its Rich console output (ANSI stripped)."""
    buf = StringIO()
    original = _output.console
    _output.console = _output.Console(file=buf, force_terminal=True, width=120)
    try:
        fn(*args, **kwargs)
    finally:
        _output.console = original
    return _ANSI_RE.sub("", buf.getvalue())


def _git_runner(
    *,
    status: str = " M src/a.py",
    diff: str = "+ line",
    stat: str = "src/a.py | 1 +",
    repo_ok: bool = True,
) -> Callable:
    def runner(args, cwd, timeout):
        if args[1] == "rev-parse":
            return MagicMock(returncode=0 if repo_ok else 1, stdout="", stderr="")
        if args[1] == "status":
            return MagicMock(returncode=0, stdout=status, stderr="")
        if args[1] == "diff" and "--stat" in args:
            return MagicMock(returncode=0, stdout=stat, stderr="")
        return MagicMock(returncode=0, stdout=diff, stderr="")

    return runner


@pytest.fixture
def conversation() -> Conversation:
    return Conversation("You are a coding agent.", max_tokens=100_000)


@pytest.fixture
def commands(conversation: Conversation, tmp_path: Path) -> Commands:
    llm_client = MagicMock()
    llm_client.model = "test-model"
    llm_client.fallback_models = []
    llm_client.total_input_tokens = 0
    llm_client.total_output_tokens = 0
    llm_client.reasoning_effort = ""
    llm_client.chat = AsyncMock(return_value=MagicMock(content="NO ISSUES"))
    return Commands(
        conversation=conversation,
        llm_client=llm_client,
        permission_engine=None,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
        tool_registry=None,
    )


class TestPureHelpers:
    def test_collect_diff_not_repo(self, tmp_path: Path) -> None:
        result = collect_diff(
            tmp_path, runner=_git_runner(repo_ok=False, status="", diff="", stat="")
        )
        assert result.error == "Not a git repository."

    def test_collect_diff_truncation(self) -> None:
        big_diff = "\n".join(f"+line{i}" for i in range(review_module.MAX_DIFF_LINES + 50))
        result = collect_diff(Path("."), runner=_git_runner(diff=big_diff))
        assert result.ok
        assert result.truncated
        assert len(result.diff_text.splitlines()) == review_module.MAX_DIFF_LINES

    def test_collect_diff_changed_files(self) -> None:
        result = collect_diff(
            Path("."),
            runner=_git_runner(status=" M src/a.py\n M b.py\n?? c.txt"),
        )
        assert result.changed_files == ["src/a.py", "b.py", "c.txt"]

    def test_review_prompt_modes(self) -> None:
        base = ReviewDiff(diff_text="+ changed")
        for mode in ("code", "security", "simplify"):
            assert review_prompt(mode, base)
        with pytest.raises(ValueError):
            review_prompt("bogus", base)
        with pytest.raises(ValueError):
            review_prompt("code", ReviewDiff(error="Not a git repository."))

    def test_findings_from_response(self) -> None:
        assert findings_from_response("NO ISSUES") == []
        assert findings_from_response("no issues found") == []
        assert findings_from_response("- bug a\n- bug b") == ["bug a", "bug b"]
        assert findings_from_response("") == []
        assert findings_from_response("plain prose finding") == ["plain prose finding"]

    def test_parse_review_args(self) -> None:
        flags, positional = parse_review_args("--fix extra text")
        assert flags == ["--fix"]
        assert positional == "extra text"
        flags, positional = parse_review_args("")
        assert flags == []
        assert positional == ""


class TestSecurityScan:
    def test_flagged_file(self, tmp_path: Path) -> None:
        (tmp_path / "cfg.py").write_text('aws_key = "AKIAABCDEFGHIJKLMNOP"\n')
        findings = security_scan_files(["cfg.py"], tmp_path)
        assert findings
        assert all(f.startswith("[secrets]") for f in findings)

    def test_clean_file_no_findings(self, tmp_path: Path) -> None:
        (tmp_path / "clean.py").write_text("x = 1\n")
        assert security_scan_files(["clean.py"], tmp_path) == []

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        assert security_scan_files(["missing.py"], tmp_path) == []


class TestEffortCommand:
    def test_show_default(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/effort")
        assert "Reasoning effort" in output

    def test_set_and_clear(self, commands: Commands) -> None:
        commands.dispatch("/effort high")
        assert commands._llm_client.reasoning_effort == "high"
        commands.dispatch("/effort clear")
        assert commands._llm_client.reasoning_effort == ""

    def test_invalid(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/effort turbo")
        assert "Usage" in output
        assert commands._llm_client.reasoning_effort == ""


class TestReviewCommands:
    @pytest.mark.asyncio
    async def test_review_requires_repo(self, commands: Commands) -> None:
        with patch.object(review_module, "collect_diff", return_value=ReviewDiff(error="no")):
            output = _capture(commands.dispatch, "/code-review")
        assert "no" in output

    @pytest.mark.asyncio
    async def test_code_review_dispatches_single_shot(
        self, commands: Commands, tmp_path: Path
    ) -> None:
        collect = MagicMock(return_value=ReviewDiff(diff_text="+ changed"))
        with (
            patch.object(review_module, "collect_diff", collect),
            patch("godspeed.agent.review.collect_diff", collect, create=True),
        ):
            commands.dispatch("/code-review")
            await asyncio.sleep(0.05)

        assert commands._llm_client.chat.await_count == 1
        assert commands._llm_client.chat.await_args.kwargs.get("task_type") == "verification"

    @pytest.mark.asyncio
    async def test_code_review_fix_queues_guidance(self, commands: Commands) -> None:
        collect = MagicMock(return_value=ReviewDiff(diff_text="+ changed"))
        with (
            patch.object(review_module, "collect_diff", collect),
        ):
            output = _capture(commands.dispatch, "/code-review --fix")
        assert "queued" in output.lower()
        contents = [m.get("content", "") for m in commands._conversation.messages]
        assert any(
            content.__contains__("Fix the findings from the code review") for content in contents
        )

    @pytest.mark.asyncio
    async def test_simplify_mode_prompt(self, commands: Commands, tmp_path: Path) -> None:
        spy_prompt = MagicMock(return_value="prompt")
        collect = MagicMock(return_value=ReviewDiff(diff_text="+ x"))
        with (
            patch.object(review_module, "collect_diff", collect),
            patch.object(review_module, "review_prompt", spy_prompt),
            patch("godspeed.agent.review.review_prompt", spy_prompt, create=True),
        ):
            commands.dispatch("/simplify")
            await asyncio.sleep(0.05)
        assert spy_prompt.call_args and spy_prompt.call_args[0][0] == "simplify"

    @pytest.mark.asyncio
    async def test_security_review_merges_secrets_finding(
        self, commands: Commands, tmp_path: Path
    ) -> None:
        secret_file = tmp_path / "cfg.py"
        secret_file.write_text('secret_value = "c2f7cd9e1b8a4d2e6f3b8c7a4e5d6f2a"\n')

        collect = MagicMock(return_value=ReviewDiff(diff_text="+key", changed_files=["cfg.py"]))
        with (
            patch.object(review_module, "collect_diff", collect),
            patch.object(review_module, "security_scan_files", security_scan_files, create=True),
        ):
            commands.dispatch("/security-review")
            await asyncio.sleep(0.05)

        assert commands._llm_client.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_llm_failure_graceful(self, commands: Commands) -> None:
        commands._llm_client.chat = AsyncMock(side_effect=RuntimeError("no key"))
        collect = MagicMock(return_value=ReviewDiff(diff_text="+ x"))
        with patch.object(review_module, "collect_diff", collect):
            commands.dispatch("/code-review")
            await asyncio.sleep(0.05)
