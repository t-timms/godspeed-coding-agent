"""Tests for ShellWidget and ShellScreen."""

from __future__ import annotations


class TestShellWidget:
    """Verify shell widget creation and lifecycle."""

    def test_creation(self):
        from godspeed.tui.widgets.shell_widget import ShellWidget

        widget = ShellWidget()
        assert widget._running is False
        assert widget._proc is None

    def test_creation_with_cwd(self, tmp_path):
        from godspeed.tui.widgets.shell_widget import ShellWidget

        widget = ShellWidget(cwd=str(tmp_path))
        assert widget._cwd == str(tmp_path)

    def test_stop_idempotent(self):
        from godspeed.tui.widgets.shell_widget import ShellWidget

        widget = ShellWidget()
        widget.stop()
        widget.stop()

    def test_send_command_before_start(self):
        from godspeed.tui.widgets.shell_widget import ShellWidget

        widget = ShellWidget()
        widget.send_command("echo hello")

    def test_start_shell_running_idempotent(self):
        from godspeed.tui.widgets.shell_widget import ShellWidget

        widget = ShellWidget()
        widget._start_shell()
        assert widget._running is True
        widget.stop()
        widget._start_shell()
        assert widget._running is True
        widget.stop()


class TestShellScreen:
    """Verify shell screen creation and composition."""

    def test_compose_yields_expected_widgets(self):
        from godspeed.tui.screens.shell_screen import ShellScreen

        screen = ShellScreen(cwd=".")
        assert screen._cwd == "."
        bindings = {b.key: b.action for b in ShellScreen.BINDINGS}
        assert bindings.get("ctrl+r") == "dismiss"
