"""Tests for FilePicker widget — scanning, filtering, selection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_tree(tmp_path):
    """Create a realistic project tree for scanning."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "src" / "utils.py").write_text("")
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("")
    (tmp_path / "README.md").write_text("")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("")
    (tmp_path / ".git" / "HEAD").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-313.pyc").write_text("")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("")
    return tmp_path


class TestFilePickerScan:
    """Verify file scanning logic."""

    def test_scan_finds_py_files(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(project_tree)
        picker._scan_files()
        files = picker._all_files
        py_files = [f for f in files if f.endswith('.py')]
        assert len(py_files) >= 3
        assert "README.md" in files

    def test_scan_excludes_git(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(project_tree)
        picker._scan_files()
        for f in picker._all_files:
            assert not f.startswith(".git")

    def test_scan_excludes_pycache(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(project_tree)
        picker._scan_files()
        for f in picker._all_files:
            assert "__pycache__" not in f

    def test_scan_excludes_node_modules(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(project_tree)
        picker._scan_files()
        for f in picker._all_files:
            assert "node_modules" not in f

    def test_scan_excludes_venv(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(project_tree)
        picker._scan_files()
        for f in picker._all_files:
            assert ".venv" not in f

    def test_scan_sorted(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(project_tree)
        picker._scan_files()
        lower_files = [f.lower() for f in picker._all_files]
        assert lower_files == sorted(lower_files)

    def test_empty_directory(self, tmp_path):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(tmp_path)
        picker._scan_files()
        assert picker._all_files == []


class TestFilePickerFilter:
    """Verify file filtering logic."""

    @pytest.fixture
    def picker(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        p = FilePicker(project_tree)
        p._scan_files()
        return p

    def test_filter_by_substring(self, picker):
        results = picker._find_matches("main")
        assert any("main" in r for r in results)

    def test_filter_case_insensitive(self, picker):
        results = picker._find_matches("MAIN")
        assert any("main" in r for r in results)

    def test_filter_no_matches(self, picker):
        results = picker._find_matches("nonexistent")
        assert results == []

    def test_filter_empty_query_shows_all(self, picker):
        results = picker._find_matches("")
        assert len(results) >= 4

    def test_filter_exact_filename(self, picker):
        results = picker._find_matches("pyproject")
        assert any("pyproject" in r for r in results)

    def test_filter_max_items(self, project_tree):
        from godspeed.tui.widgets.file_picker import FilePicker

        for i in range(30):
            (project_tree / f"file_{i:03d}.txt").write_text("")

        picker = FilePicker(project_tree, max_items=5)
        picker._scan_files()
        results = picker._find_matches("")
        assert len(results) == 5

    def test_filter_better_matches_first(self, picker):
        """Earlier substring matches should appear before later ones."""
        results = picker._find_matches("py")
        if len(results) >= 2:
            first = results[0].lower()
            second = results[1].lower()
            assert first.find("py") <= second.find("py")


class TestFilePickerInit:
    """Verify FilePicker constructor."""

    def test_initial_display_false(self, tmp_path):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(tmp_path)
        assert picker.display is False

    def test_project_dir_resolved(self, tmp_path):
        from godspeed.tui.widgets.file_picker import FilePicker

        sub = tmp_path / "sub"
        sub.mkdir()
        picker = FilePicker(sub)
        assert picker._project_dir.is_absolute()


class TestFilePickerEmpty:
    """Verify behavior with no matching files."""

    def test_filter_without_files(self, tmp_path):
        from godspeed.tui.widgets.file_picker import FilePicker

        picker = FilePicker(tmp_path)
        picker._scan_files = lambda: None  # simulate empty
        picker._all_files = []
        picker.filter_for("test")
        assert picker.display is False
