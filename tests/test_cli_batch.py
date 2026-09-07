"""Tests for the `godspeed batch` CLI subcommand and batch config section."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from godspeed.agent.result import ExitCode
from godspeed.cli import batch_cmd
from godspeed.config import GodspeedSettings


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


def _patch_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_path / ".gs-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")


class TestBatchDryRun:
    """--dry-run decomposes and prints the plan without touching git."""

    def test_dry_run_prints_units(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        goal = "Do these:\n1. Write the parser\n2. Write the tests\n3. Update docs"
        result = runner.invoke(
            batch_cmd,
            [goal, "--dry-run", "--project-dir", str(tmp_path)],
            standalone_mode=False,
        )
        assert result.exit_code == 0
        assert "Batch plan: 3 unit(s)" in result.output
        assert "u1" in result.output
        assert "Write the parser" in result.output
        assert "u2" in result.output
        assert "u3" in result.output
        assert "Dry run" in result.output

    def test_dry_run_does_not_construct_coordinator_or_touch_git(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry-run must not build a coordinator or run git even on a dirty repo."""
        (git_repo / "dirty.txt").write_text("x\n", encoding="utf-8")

        def _boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("dry-run must not construct a coordinator")

        with (
            patch(
                "godspeed.agent.coordinator.AgentCoordinator.__init__",
                _boom,
            ),
            patch(
                "godspeed.agent.batch.WorktreeBatchRunner.__init__",
                _boom,
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                batch_cmd,
                ["1. a\n2. b", "--dry-run", "--project-dir", str(git_repo)],
                standalone_mode=False,
            )
        assert result.exit_code == 0
        assert "Batch plan: 2 unit(s)" in result.output


class TestBatchInvalidUnits:
    """--units outside 1-30 must reject with a nonzero exit."""

    def test_units_zero_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            batch_cmd,
            ["goal", "--units", "0", "--project-dir", str(tmp_path)],
            standalone_mode=False,
        )
        assert result.exit_code == int(ExitCode.INVALID_INPUT)
        assert "must be between 1 and 30" in result.output

    def test_units_above_cap_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            batch_cmd,
            ["goal", "--units", "31", "--project-dir", str(tmp_path)],
            standalone_mode=False,
        )
        assert result.exit_code == int(ExitCode.INVALID_INPUT)
        assert "must be between 1 and 30" in result.output

    def test_no_goal_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            batch_cmd, ["", "--project-dir", str(tmp_path)], standalone_mode=False
        )
        assert result.exit_code == int(ExitCode.INVALID_INPUT)
        assert "No goal provided" in result.output


class TestBatchDirtyRepo:
    """Dirty-repo refusal and --allow-dirty pass-through."""

    def test_dirty_repo_refused(self, git_repo: Path) -> None:
        (git_repo / "dirty.txt").write_text("x\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            batch_cmd,
            ["1. a\n2. b", "--project-dir", str(git_repo)],
            standalone_mode=False,
        )
        assert result.exit_code == int(ExitCode.TOOL_ERROR)
        assert "uncommitted changes" in result.output

    def test_allow_dirty_passes_through(self, git_repo: Path) -> None:
        (git_repo / "dirty.txt").write_text("x\n", encoding="utf-8")

        async def _fake_spawn(self, *args: object, **kwargs: object) -> str:
            return "unit done"

        with patch(
            "godspeed.agent.coordinator.AgentCoordinator.spawn",
            new=_fake_spawn,
        ):
            runner = CliRunner()
            result = runner.invoke(
                batch_cmd,
                ["1. a\n2. b", "--allow-dirty", "--project-dir", str(git_repo)],
                standalone_mode=False,
            )
        assert result.exit_code == 0
        assert "Batch finished: 2/2 units succeeded" in result.output


class TestBatchHelp:
    """`godspeed batch --help` documents the flags."""

    def test_help_documents_flags(self) -> None:
        runner = CliRunner()
        result = runner.invoke(batch_cmd, ["--help"], standalone_mode=False)
        assert result.exit_code == 0
        for flag in ["--units", "--parallelism", "--allow-dirty", "--dry-run", "--project-dir"]:
            assert flag in result.output


class TestBatchConfig:
    """Batch settings section defaults and custom values."""

    def test_batch_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(project_dir=tmp_path)
        assert s.batch.parallelism == 5
        assert s.batch.open_pr is False
        assert s.batch.worktree_dir == ""

    def test_batch_custom_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(
            project_dir=tmp_path,
            batch={"parallelism": 8, "open_pr": True, "worktree_dir": "/tmp/wt"},
        )
        assert s.batch.parallelism == 8
        assert s.batch.open_pr is True
        assert s.batch.worktree_dir == "/tmp/wt"

    def test_batch_yaml_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        global_dir = tmp_path / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(
            yaml.dump({"batch": {"parallelism": 12, "open_pr": True, "worktree_dir": "/wt"}})
        )
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")
        s = GodspeedSettings(project_dir=tmp_path)
        assert s.batch.parallelism == 12
        assert s.batch.open_pr is True
        assert s.batch.worktree_dir == "/wt"

    def test_batch_known_top_level_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'batch' must be a recognized top-level key (no unknown-key warning)."""
        from godspeed.config import _KNOWN_TOP_LEVEL_KEYS

        assert "batch" in _KNOWN_TOP_LEVEL_KEYS
