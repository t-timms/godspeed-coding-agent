"""Tests for before-edit file checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.edit_checkpoints import (
    checkpoints_dir,
    list_checkpoints,
    restore_latest,
    snapshot_file,
)
from godspeed.tools.file_edit import FileEditTool
from godspeed.tools.file_write import FileWriteTool
from godspeed.tools.base import ToolContext


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="ses_test_001")


class TestSnapshotFile:
    def test_snapshots_current_content(self, tmp_path: Path, tool_context: ToolContext) -> None:
        target = tmp_path / "code.py"
        target.write_text("original", encoding="utf-8")
        snap = snapshot_file(target, tmp_path, tool_context.session_id)
        assert snap is not None
        assert snap.read_text(encoding="utf-8") == "original"

    def test_missing_file_returns_none(self, tmp_path: Path, tool_context: ToolContext) -> None:
        assert snapshot_file(tmp_path / "nope.py", tmp_path, tool_context.session_id) is None

    def test_same_second_snapshots_are_ordered(
        self, tmp_path: Path, tool_context: ToolContext
    ) -> None:
        target = tmp_path / "code.py"
        target.write_text("v1", encoding="utf-8")
        first = snapshot_file(target, tmp_path, tool_context.session_id)
        target.write_text("v2", encoding="utf-8")
        second = snapshot_file(target, tmp_path, tool_context.session_id)
        assert first is not None and second is not None
        assert sorted([first.name, second.name])[1] == second.name
        assert second.read_text(encoding="utf-8") == "v2"


class TestRestoreLatest:
    def test_restores_newest_snapshot(self, tmp_path: Path, tool_context: ToolContext) -> None:
        target = tmp_path / "code.py"
        target.write_text("original", encoding="utf-8")
        snapshot_file(target, tmp_path, tool_context.session_id)
        target.write_text("mutated", encoding="utf-8")
        restored_from = restore_latest(target, tmp_path, tool_context.session_id)
        assert restored_from is not None
        assert target.read_text(encoding="utf-8") == "original"

    def test_no_snapshots_returns_none(self, tmp_path: Path, tool_context: ToolContext) -> None:
        assert restore_latest(tmp_path / "x.py", tmp_path, tool_context.session_id) is None

    def test_snapshots_are_session_scoped(self, tmp_path: Path, tool_context: ToolContext) -> None:
        target = tmp_path / "code.py"
        target.write_text("v", encoding="utf-8")
        snapshot_file(target, tmp_path, tool_context.session_id)
        other_dir = checkpoints_dir(tmp_path, "ses_other")
        assert not other_dir.exists()


class TestListCheckpoints:
    def test_lists_oldest_first(self, tmp_path: Path, tool_context: ToolContext) -> None:
        target = tmp_path / "code.py"
        for content in ("a", "b", "c"):
            target.write_text(content, encoding="utf-8")
            snapshot_file(target, tmp_path, tool_context.session_id)
        snaps = list_checkpoints(target, tmp_path, tool_context.session_id)
        assert [s.read_text(encoding="utf-8") for s in snaps] == ["a", "b", "c"]


class TestToolIntegration:
    @pytest.mark.asyncio
    async def test_file_write_snapshots_before_overwrite(
        self, tmp_path: Path, tool_context: ToolContext
    ) -> None:
        target = tmp_path / "note.txt"
        target.write_text("before", encoding="utf-8")
        tool = FileWriteTool()
        result = await tool.execute({"file_path": str(target), "content": "after"}, tool_context)
        assert not result.is_error
        snaps = list_checkpoints(target, tmp_path, tool_context.session_id)
        assert len(snaps) == 1
        assert snaps[0].read_text(encoding="utf-8") == "before"
        assert restore_latest(target, tmp_path, tool_context.session_id) is not None
        assert target.read_text(encoding="utf-8") == "before"

    @pytest.mark.asyncio
    async def test_file_edit_snapshots_original(
        self, tmp_path: Path, tool_context: ToolContext
    ) -> None:
        target = tmp_path / "code.py"
        target.write_text("value = 1\n", encoding="utf-8")
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": str(target),
                "old_string": "value = 1",
                "new_string": "value = 2",
            },
            tool_context,
        )
        assert not result.is_error
        snaps = list_checkpoints(target, tmp_path, tool_context.session_id)
        assert len(snaps) == 1
        assert snaps[0].read_text(encoding="utf-8") == "value = 1\n"

    @pytest.mark.asyncio
    async def test_rejected_write_leaves_no_extra_snapshot(
        self, tmp_path: Path, tool_context: ToolContext
    ) -> None:
        target = tmp_path / "note.txt"
        target.write_text("before", encoding="utf-8")
        tool = FileWriteTool()
        result = await tool.execute({"file_path": "", "content": "x"}, tool_context)
        assert result.is_error
        assert list_checkpoints(target, tmp_path, tool_context.session_id) == []
