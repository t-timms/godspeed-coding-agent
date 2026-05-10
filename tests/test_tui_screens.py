"""Tests for TUI screens — splash, permission, diff review, help, sessions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


class TestSplashScreen:
    """Verify splash screen creation and status updates."""

    def test_initial_status(self):
        from godspeed.tui.screens.splash import SplashScreen

        screen = SplashScreen()
        assert screen._status == "Starting..."

    def test_update_status(self):
        from godspeed.tui.screens.splash import SplashScreen

        screen = SplashScreen()
        screen.update_status("Loading...")
        assert screen._status == "Loading..."

    def test_update_status_multiple(self):
        from godspeed.tui.screens.splash import SplashScreen

        screen = SplashScreen()
        screen.update_status("Step 1")
        screen.update_status("Step 2")
        assert screen._status == "Step 2"

    def test_compose_yields_static(self):
        from godspeed.tui.screens.splash import SplashScreen
        from textual.widgets import Static

        screen = SplashScreen()
        widgets = list(screen.compose())
        assert len(widgets) == 1
        assert isinstance(widgets[0], Static)

    def test_compose_contains_version(self):
        from godspeed.tui.screens.splash import SplashScreen

        screen = SplashScreen()
        widgets = list(screen.compose())
        content = str(widgets[0].render())
        assert "godspeed" in content

    def test_compose_contains_status(self):
        from godspeed.tui.screens.splash import SplashScreen

        screen = SplashScreen()
        screen._status = "Testing..."
        widgets = list(screen.compose())
        content = str(widgets[0].render())
        assert "Testing..." in content

    def test_update_status_safe_when_not_mounted(self):
        from godspeed.tui.screens.splash import SplashScreen

        screen = SplashScreen()
        screen.update_status("Should not crash")


class TestPermissionDialog:
    """Verify permission dialog creation and actions."""

    def test_init_stores_args(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("shell", "needs approval", {"command": "rm -rf /"})
        assert dialog._tool_name == "shell"
        assert dialog._reason == "needs approval"
        assert dialog._arguments["command"] == "rm -rf /"

    def test_init_no_args(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("file_read", "needs approval")
        assert dialog._arguments == {}

    def test_compose_contains_tool_name(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("shell", "confirmation needed")
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "shell" in content
        assert "confirmation needed" in content

    def test_compose_shows_file_path(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("file_edit", "needs approval", {"file_path": "src/main.py"})
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "src/main.py" in content

    def test_compose_shows_command(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("shell", "needs approval", {"command": "pytest"})
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "pytest" in content

    def test_compose_shows_pattern(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("grep_search", "needs approval", {"pattern": "def test"})
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "def test" in content

    def test_bindings_include_approve(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        bindings = {b[1] for b in PermissionDialog.BINDINGS}
        assert "approve" in bindings
        assert "deny" in bindings
        assert "always_allow" in bindings

    def test_action_approve_dismisses_yes(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("shell", "test")
        result = []

        def capture(result_str):
            result.append(result_str)

        dialog.dismiss = capture
        dialog.action_approve()
        assert result == ["yes"]

    def test_action_deny_dismisses_no(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("shell", "test")
        result = []

        def capture(result_str):
            result.append(result_str)

        dialog.dismiss = capture
        dialog.action_deny()
        assert result == ["no"]

    def test_action_always_dismisses_always(self):
        from godspeed.tui.screens.permission_dialog import PermissionDialog

        dialog = PermissionDialog("shell", "test")
        result = []

        def capture(result_str):
            result.append(result_str)

        dialog.dismiss = capture
        dialog.action_always_allow()
        assert result == ["always"]


class TestDiffReviewDialog:
    """Verify diff review dialog creation and actions."""

    def test_init_stores_args(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog("file_edit", "src/main.py", "before", "after")
        assert dialog._tool_name == "file_edit"
        assert dialog._path == "src/main.py"
        assert dialog._before == "before"
        assert dialog._after == "after"

    def test_compose_contains_path(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog("file_write", "src/main.py", "old", "new")
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "src/main.py" in content

    def test_compose_contains_tool_name(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog("file_edit", "src/main.py", "old", "new")
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "file_edit" in content

    def test_compose_shows_diff_stats(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog(
            "file_edit", "src/main.py", "line1\nline2\nline3", "line1\nline2\nline4"
        )
        widgets = list(dialog.compose())
        content = str(widgets[0].render())
        assert "+" in content
        assert "-" in content

    def test_bindings_include_accept(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        bindings = {b[1] for b in DiffReviewDialog.BINDINGS}
        assert "accept" in bindings
        assert "reject" in bindings
        assert "always" in bindings

    def test_action_accept(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog("tool", "path", "before", "after")
        result = []

        def capture(r):
            result.append(r)

        dialog.dismiss = capture
        dialog.action_accept()
        assert result == ["accept"]

    def test_action_reject(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog("tool", "path", "before", "after")
        result = []

        def capture(r):
            result.append(r)

        dialog.dismiss = capture
        dialog.action_reject()
        assert result == ["reject"]

    def test_action_always(self):
        from godspeed.tui.screens.diff_review import DiffReviewDialog

        dialog = DiffReviewDialog("tool", "path", "before", "after")
        result = []

        def capture(r):
            result.append(r)

        dialog.dismiss = capture
        dialog.action_always()
        assert result == ["always"]


class TestHelpScreen:
    """Verify help screen content."""

    def test_compose_contains_content(self):
        from godspeed.tui.screens.help_screen import HelpScreen

        mock_commands = Mock()
        screen = HelpScreen(mock_commands)
        widgets = list(screen.compose())
        assert len(widgets) == 1
        content = str(widgets[0].render())
        assert "Commands" in content or "Keyboard" in content or "Godspeed" in content

    def test_help_bindings(self):
        from godspeed.tui.screens.help_screen import HelpScreen

        bindings = {b[1] for b in HelpScreen.BINDINGS}
        assert "dismiss" in bindings


class TestSessionListScreen:
    """Verify session list screen."""

    def test_compose_text(self, tmp_path):
        from godspeed.tui.screens.session_list import SessionListScreen

        screen = SessionListScreen(tmp_path)
        widgets = list(screen.compose())
        assert len(widgets) == 3

    def test_session_bindings(self):
        from godspeed.tui.screens.session_list import SessionListScreen

        bindings = {b.action for b in SessionListScreen.BINDINGS}
        assert "dismiss" in bindings
