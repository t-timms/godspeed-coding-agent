"""Tests for the /batch worktree decomposition and dispatch module."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.batch import (
    MAX_BATCH_UNITS,
    BatchGitError,
    BatchPlan,
    BatchUnit,
    WorktreeBatchRunner,
    _extract_json_object,
    _parse_decompose_response,
    chunk_plan,
    decompose_task,
    default_worktree_root,
    validate_plan,
)
from godspeed.agent.coordinator import AgentCoordinator
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolContext
from godspeed.tools.registry import ToolRegistry


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


class _FakeExecutor:
    """Deterministic executor that records worktree paths and returns a summary."""

    def __init__(self, summary: str = "unit done", fail: bool = False) -> None:
        self.summary = summary
        self.fail = fail
        self.calls: list[tuple[str, Path]] = []

    async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
        self.calls.append((unit.id, worktree_path))
        if self.fail:
            raise RuntimeError("unit crashed")
        return self.summary


class _CountingExecutor:
    """Executor that tracks peak concurrency to verify the parallelism cap."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls: list[str] = []

    async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append(unit.id)
        await asyncio.sleep(0.01)
        self.active -= 1
        return f"done {unit.id}"


def _make_coordinator(tool_context: ToolContext) -> AgentCoordinator:
    client = LLMClient(model="test")
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    return AgentCoordinator(
        llm_client=client,
        tool_registry=ToolRegistry(),
        tool_context=tool_context,
    )


class TestChunkPlan:
    def test_numbered_list_splits_into_units(self) -> None:
        goal = "Do these:\n1. Write the parser\n2. Write the tests\n3. Update docs"
        plan = chunk_plan(goal)
        assert len(plan.units) == 3
        assert [u.id for u in plan.units] == ["u1", "u2", "u3"]
        assert plan.units[0].instructions == "Write the parser"

    def test_no_list_falls_back_to_single_unit(self) -> None:
        plan = chunk_plan("Just do the whole thing")
        assert len(plan.units) == 1
        assert plan.units[0].id == "u1"
        assert plan.units[0].instructions == "Just do the whole thing"

    def test_empty_goal_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_plan("   ")

    def test_hint_above_cap_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_plan("goal", num_units_hint=MAX_BATCH_UNITS + 1)


class TestValidatePlan:
    def test_valid_plan_passes(self) -> None:
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="t", instructions="i")],
        )
        validate_plan(plan)

    def test_empty_units_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_plan(BatchPlan(goal="g", units=[]))

    def test_duplicate_ids_raise(self) -> None:
        plan = BatchPlan(
            goal="g",
            units=[
                BatchUnit(id="u1", title="a", instructions="i"),
                BatchUnit(id="u1", title="b", instructions="j"),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate"):
            validate_plan(plan)

    def test_unknown_dependency_raises(self) -> None:
        plan = BatchPlan(
            goal="g",
            units=[
                BatchUnit(id="u1", title="a", instructions="i", depends_on=["nope"]),
            ],
        )
        with pytest.raises(ValueError, match="unknown unit"):
            validate_plan(plan)

    def test_over_cap_raises(self) -> None:
        units = [
            BatchUnit(id=f"u{i}", title=f"t{i}", instructions="i")
            for i in range(MAX_BATCH_UNITS + 1)
        ]
        with pytest.raises(ValueError, match="cap"):
            validate_plan(BatchPlan(goal="g", units=units))


class TestExtractJsonObject:
    def test_fenced_json(self) -> None:
        content = 'Here you go:\n```json\n{"units": [{"id": "u1"}]}\n```'
        assert _extract_json_object(content) == {"units": [{"id": "u1"}]}

    def test_bare_json(self) -> None:
        content = '{"units": []}'
        assert _extract_json_object(content) == {"units": []}

    def test_no_json_returns_none(self) -> None:
        assert _extract_json_object("no json here") is None

    def test_empty_returns_none(self) -> None:
        assert _extract_json_object("") is None


class TestParseDecomposeResponse:
    def test_valid_response(self) -> None:
        content = (
            '{"units": [{"id": "u1", "title": "Parser", '
            '"instructions": "Write parser", "depends_on": []}]}'
        )
        plan = _parse_decompose_response(content, "goal")
        assert plan is not None
        assert plan.units[0].id == "u1"

    def test_invalid_schema_returns_none(self) -> None:
        content = '{"units": [{"id": "u1", "title": "", "instructions": "i"}]}'
        assert _parse_decompose_response(content, "goal") is None

    def test_not_json_returns_none(self) -> None:
        assert _parse_decompose_response("garbage", "goal") is None


class TestDecomposeTask:
    async def test_no_llm_uses_chunk_plan(self) -> None:
        plan = await decompose_task("1. a\n2. b")
        assert len(plan.units) == 2

    async def test_llm_success(self) -> None:
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=_make_text_response(
                '{"units": [{"id": "u1", "title": "T", "instructions": "I", "depends_on": []}]}'
            )
        )
        plan = await decompose_task("goal", llm_client=client)
        assert len(plan.units) == 1
        assert plan.units[0].id == "u1"

    async def test_llm_failure_falls_back(self) -> None:
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("boom"))
        plan = await decompose_task("1. a\n2. b", llm_client=client)
        assert len(plan.units) == 2

    async def test_llm_invalid_schema_falls_back(self) -> None:
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("not json"))
        plan = await decompose_task("1. a", llm_client=client)
        assert len(plan.units) == 1

    async def test_empty_goal_raises(self) -> None:
        with pytest.raises(ValueError):
            await decompose_task("   ")


class TestDefaultWorktreeRoot:
    def test_sibling_of_working_dir(self) -> None:
        root = default_worktree_root(Path("/work/myrepo"))
        assert root == Path("/work/myrepo.godspeed-batch")


class TestWorktreeBatchRunner:
    async def test_run_success_removes_worktrees(
        self, git_repo: Path, tool_context: ToolContext
    ) -> None:
        coordinator = _make_coordinator(tool_context)
        executor = _FakeExecutor(summary="all good")
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=executor,
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[
                BatchUnit(id="u1", title="a", instructions="i"),
                BatchUnit(id="u2", title="b", instructions="j"),
            ],
        )
        results = await runner.run(plan)
        assert len(results) == 2
        assert all(r.ok for r in results)
        assert all(r.summary == "all good" for r in results)
        assert len(executor.calls) == 2
        for _, worktree_path in executor.calls:
            assert worktree_path.exists() is False
        leftover = runner.cleanup()
        assert leftover == []

    async def test_run_captures_patch(self, git_repo: Path, tool_context: ToolContext) -> None:
        coordinator = _make_coordinator(tool_context)

        class _PatchExecutor:
            async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
                (worktree_path / "new.txt").write_text("new\n", encoding="utf-8")
                return "patched"

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
        results = await runner.run(plan)
        assert results[0].ok
        assert results[0].patch_available
        assert results[0].patch_path is not None
        assert results[0].patch_path.exists()
        assert "new.txt" in results[0].patch_path.read_text(encoding="utf-8")

    async def test_parallelism_cap(self, git_repo: Path, tool_context: ToolContext) -> None:
        coordinator = _make_coordinator(tool_context)
        executor = _CountingExecutor()
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=executor,
            parallelism=2,
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id=f"u{i}", title=f"t{i}", instructions="i") for i in range(1, 6)],
        )
        results = await runner.run(plan)
        assert all(r.ok for r in results)
        assert executor.peak <= 2
        assert len(executor.calls) == 5

    async def test_dirty_repo_refused(self, git_repo: Path, tool_context: ToolContext) -> None:
        (git_repo / "dirty.txt").write_text("x\n", encoding="utf-8")
        coordinator = _make_coordinator(tool_context)
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_FakeExecutor(),
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        with pytest.raises(BatchGitError, match="uncommitted"):
            await runner.run(plan)

    async def test_allow_dirty_runs(self, git_repo: Path, tool_context: ToolContext) -> None:
        (git_repo / "dirty.txt").write_text("x\n", encoding="utf-8")
        coordinator = _make_coordinator(tool_context)
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_FakeExecutor(),
            allow_dirty=True,
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan)
        assert results[0].ok

    async def test_non_repo_refused(self, tmp_path: Path, tool_context: ToolContext) -> None:
        not_a_repo = tmp_path / "not-a-repo"
        not_a_repo.mkdir()
        coordinator = _make_coordinator(tool_context)
        runner = WorktreeBatchRunner(
            working_dir=not_a_repo,
            coordinator=coordinator,
            executor=_FakeExecutor(),
            worktree_root=tmp_path / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        with pytest.raises(BatchGitError, match="git repository"):
            await runner.run(plan)

    async def test_unit_crash_is_isolated(self, git_repo: Path, tool_context: ToolContext) -> None:
        coordinator = _make_coordinator(tool_context)
        executor = _FakeExecutor(fail=True)
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=executor,
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan)
        assert not results[0].ok
        assert "failed" in results[0].summary.lower()
        assert results[0].worktree_path is not None
        assert results[0].worktree_path.exists()
        leftover = runner.cleanup()
        assert leftover == []

    async def test_timeout_marks_unit(self, git_repo: Path, tool_context: ToolContext) -> None:
        coordinator = _make_coordinator(tool_context)

        class _SlowExecutor:
            async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
                await asyncio.sleep(10)
                return "late"

        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_SlowExecutor(),
            unit_timeout=0.05,
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan)
        assert not results[0].ok
        assert results[0].timed_out
        assert "timed out" in results[0].summary.lower()

    async def test_no_gh_fallback(
        self, git_repo: Path, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("godspeed.agent.batch.shutil.which", lambda _: None)
        coordinator = _make_coordinator(tool_context)

        class _PatchExecutor:
            async def execute(self, unit: BatchUnit, worktree_path: Path) -> str:
                (worktree_path / "new.txt").write_text("new\n", encoding="utf-8")
                return "patched"

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
        assert len(runner.pr_messages) == 1
        assert "gh CLI not available" in runner.pr_messages[0]

    async def test_keep_worktrees_preserves_path(
        self, git_repo: Path, tool_context: ToolContext
    ) -> None:
        coordinator = _make_coordinator(tool_context)
        runner = WorktreeBatchRunner(
            working_dir=git_repo,
            coordinator=coordinator,
            executor=_FakeExecutor(),
            keep_worktrees=True,
            worktree_root=git_repo.parent / "batch-root",
        )
        plan = BatchPlan(
            goal="g",
            units=[BatchUnit(id="u1", title="a", instructions="i")],
        )
        results = await runner.run(plan)
        assert results[0].ok
        assert results[0].worktree_path is not None
        assert results[0].worktree_path.exists()
        leftover = runner.cleanup()
        assert leftover == []
