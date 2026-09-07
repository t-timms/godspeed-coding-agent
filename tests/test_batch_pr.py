"""Tests for /batch GitHub PR automation (gh CLI integration).

Covers the ``create_pr`` helper (gh runner injection, guards, stderr
capping) and the ``WorktreeBatchRunner`` PR flow (per-patchable-unit
creation, per-unit failure isolation, gh-missing fallback) plus the CLI
``--open-pr`` flag / ``batch.open_pr`` config plumbing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from click.testing import CliRunner

from godspeed.agent.batch import (
    BatchPlan,
    BatchUnit,
    WorktreeBatchRunner,
    create_pr,
)
from godspeed.agent.coordinator import AgentCoordinator
from godspeed.cli import batch_cmd
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolContext
from godspeed.tools.registry import ToolRegistry

PR_URL = "https://github.com/acme/repo/pull/1"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A clean git repo with one committed file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_coordinator(tool_context: ToolContext) -> AgentCoordinator:
    client = LLMClient(model="test")
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    return AgentCoordinator(
        llm_client=client,
        tool_registry=ToolRegistry(),
        tool_context=tool_context,
    )


class _PatchExecutor:
    """Executor that writes a file so every unit produces a captured patch."""

    async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
        (worktree_path / f"{unit.id}.txt").write_text("new\n", encoding="utf-8")
        return f"patched {unit.id}"


class _FakeGhRunner:
    """Injected gh runner: records invocations, returns a configurable result."""

    def __init__(
        self,
        returncode: int = 0,
        stdout: str = PR_URL,
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args: list[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((args, cwd))
        return self.returncode, self.stdout, self.stderr


class _FlakyGhRunner:
    """Fails the first PR, succeeds the rest — for per-unit isolation."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args: list[str], cwd: Path) -> tuple[int, str, str]:
        self.calls.append((args, cwd))
        if len(self.calls) == 1:
            return (
                1,
                "",
                "error: no commits found on branch 'batch/abc/u1'\n"
                "hint: the branch is local-only; push it first\n"
                "line3\nline4\nline5\nline6\nline7",
            )
        return 0, "https://github.com/acme/repo/pull/2", ""


def _patch_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_path / ".gs-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")


class TestCreatePr:
    """The pure ``create_pr`` helper with gh runner injection."""

    def test_success_with_fake_runner(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        fake = _FakeGhRunner()
        result = create_pr(
            "u1",
            git_repo,
            branch="master",
            title="batch: u1",
            body="Unit: u1\n\nsummary",
            gh_runner=fake,
        )
        assert result.ok
        assert result.url_or_error == PR_URL
        assert len(fake.calls) == 1
        args, cwd = fake.calls[0]
        assert args == [
            "pr",
            "create",
            "--head",
            "master",
            "--title",
            "batch: u1",
            "--body",
            "Unit: u1\n\nsummary",
        ]
        assert cwd == git_repo

    def test_branch_derived_from_worktree(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        fake = _FakeGhRunner()
        result = create_pr("u1", git_repo, gh_runner=fake)
        assert result.ok
        args, _ = fake.calls[0]
        assert args[0:3] == ["pr", "create", "--head"]
        assert args[3] == _git(git_repo, "branch", "--show-current").strip()

    def test_gh_missing_returns_note(self, git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: None)
        fake = _FakeGhRunner()
        result = create_pr("u1", git_repo, branch="master", gh_runner=fake)
        assert not result.ok
        assert "gh CLI not available" in result.url_or_error
        assert fake.calls == []

    def test_missing_branch_returns_note(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        fake = _FakeGhRunner()
        result = create_pr("u1", git_repo, branch="does-not-exist", gh_runner=fake)
        assert not result.ok
        assert "not found in worktree" in result.url_or_error
        assert fake.calls == []

    def test_gh_error_caps_stderr(self, git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        fake = _FakeGhRunner(returncode=1, stderr="\n".join(f"line{i}" for i in range(1, 9)))
        result = create_pr("u1", git_repo, branch="master", gh_runner=fake)
        assert not result.ok
        lines = result.url_or_error.splitlines()
        assert len(lines) == 6  # 5 capped lines + "... (3 more lines)"
        assert lines[-1] == "... (3 more lines)"

    def test_empty_stdout_returns_note(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        fake = _FakeGhRunner(returncode=0, stdout="")
        result = create_pr("u1", git_repo, branch="master", gh_runner=fake)
        assert not result.ok
        assert "returned no URL" in result.url_or_error


class TestRunnerPrFlow:
    """PR creation through WorktreeBatchRunner with an injected gh runner."""

    async def test_pr_created_per_patchable_unit(
        self, git_repo: Path, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        coordinator = _make_coordinator(tool_context)
        fake = _FakeGhRunner()
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_PatchExecutor(),
            worktree_root=git_repo.parent / "batch-root",
        )
        runner._gh_runner = fake
        plan = BatchPlan(
            goal="g",
            units=[
                BatchUnit(id="u1", title="a", instructions="i"),
                BatchUnit(id="u2", title="b", instructions="j"),
            ],
        )
        results = await runner.run(plan, open_prs=True)
        assert all(r.ok for r in results)
        assert len(fake.calls) == 2
        assert all(r.pr_url == PR_URL for r in results)
        for args, _ in fake.calls:
            assert args[0:3] == ["pr", "create", "--head"]

    async def test_pr_failure_isolated_per_unit(
        self, git_repo: Path, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        coordinator = _make_coordinator(tool_context)
        fake = _FlakyGhRunner()
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_PatchExecutor(),
            worktree_root=git_repo.parent / "batch-root",
        )
        runner._gh_runner = fake
        plan = BatchPlan(
            goal="g",
            units=[
                BatchUnit(id="u1", title="a", instructions="i"),
                BatchUnit(id="u2", title="b", instructions="j"),
            ],
        )
        results = await runner.run(plan, open_prs=True)
        assert all(r.ok for r in results)
        assert results[0].pr_url is None
        assert "[pr]" in results[0].summary
        assert "no commits found" in results[0].summary
        assert results[1].pr_url == "https://github.com/acme/repo/pull/2"
        assert len(fake.calls) == 2

    async def test_gh_missing_notes_not_crash(
        self, git_repo: Path, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: None)
        coordinator = _make_coordinator(tool_context)
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_PatchExecutor(),
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan, open_prs=True)
        assert results[0].ok
        assert results[0].pr_url is None
        assert len(runner.pr_messages) == 1
        assert "gh CLI not available" in runner.pr_messages[0]
        assert "[pr]" in results[0].summary

    async def test_constructor_open_pr_enables_prs(
        self, git_repo: Path, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        coordinator = _make_coordinator(tool_context)
        fake = _FakeGhRunner()
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_PatchExecutor(),
            worktree_root=git_repo.parent / "batch-root",
            open_pr=True,
        )
        runner._gh_runner = fake
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan)
        assert results[0].ok
        assert results[0].pr_url == PR_URL
        assert len(fake.calls) == 1

    async def test_no_patch_no_pr(
        self, git_repo: Path, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: "/usr/bin/gh")
        coordinator = _make_coordinator(tool_context)

        class _NoopExecutor:
            async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
                return "no changes"

        fake = _FakeGhRunner()
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_NoopExecutor(),
            worktree_root=git_repo.parent / "batch-root",
        )
        runner._gh_runner = fake
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan, open_prs=True)
        assert results[0].ok
        assert not results[0].patch_available
        assert results[0].pr_url is None
        assert fake.calls == []


class TestCliPrFlag:
    """CLI --open-pr flag and batch.open_pr config plumbing."""

    def test_open_pr_flag_flows_to_runner(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(git_repo, monkeypatch)
        with patch("godspeed.agent.batch.WorktreeBatchRunner") as mock_runner:
            mock_runner.return_value.run = AsyncMock(return_value=[])
            runner = CliRunner()
            result = runner.invoke(
                batch_cmd,
                ["1. a\n2. b", "--open-pr", "--project-dir", str(git_repo)],
                standalone_mode=False,
            )
        assert result.exit_code == 0
        assert mock_runner.call_args.kwargs["open_pr"] is True

    def test_config_open_pr_flows_to_runner(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(git_repo, monkeypatch)
        global_dir = git_repo / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(
            yaml.dump({"batch": {"open_pr": True}}), encoding="utf-8"
        )
        with patch("godspeed.agent.batch.WorktreeBatchRunner") as mock_runner:
            mock_runner.return_value.run = AsyncMock(return_value=[])
            runner = CliRunner()
            result = runner.invoke(
                batch_cmd,
                ["1. a\n2. b", "--project-dir", str(git_repo)],
                standalone_mode=False,
            )
        assert result.exit_code == 0
        assert mock_runner.call_args.kwargs["open_pr"] is True

    def test_config_false_default_no_pr(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(git_repo, monkeypatch)
        with patch("godspeed.agent.batch.WorktreeBatchRunner") as mock_runner:
            mock_runner.return_value.run = AsyncMock(return_value=[])
            runner = CliRunner()
            result = runner.invoke(
                batch_cmd,
                ["1. a\n2. b", "--project-dir", str(git_repo)],
                standalone_mode=False,
            )
        assert result.exit_code == 0
        assert mock_runner.call_args.kwargs["open_pr"] is False

    def test_dry_run_with_open_pr_notifies_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            batch_cmd,
            ["1. a\n2. b", "--dry-run", "--open-pr", "--project-dir", str(tmp_path)],
            standalone_mode=False,
        )
        assert result.exit_code == 0
        assert "Batch plan: 2 unit(s)" in result.output
        assert "PR step skipped (dry run)" in result.output

    def test_dry_run_with_config_open_pr_notifies_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        global_dir = tmp_path / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(
            yaml.dump({"batch": {"open_pr": True}}), encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(
            batch_cmd,
            ["1. a\n2. b", "--dry-run", "--project-dir", str(tmp_path)],
            standalone_mode=False,
        )
        assert result.exit_code == 0
        assert "PR step skipped (dry run)" in result.output

    def test_help_documents_open_pr(self) -> None:
        runner = CliRunner()
        result = runner.invoke(batch_cmd, ["--help"], standalone_mode=False)
        assert result.exit_code == 0
        assert "--open-pr" in result.output
