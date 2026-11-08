"""GODSPEED_SHELL_WRAPPER: benchmark runs route every agent command through a wrapper script."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import godspeed.tools.shell as shell_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    shell_mod._shell_cache = None
    yield
    shell_mod._shell_cache = None


def test_wrapper_replaces_bash_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(shell_mod.SHELL_WRAPPER_ENV, "/opt/isolate.sh")
    with patch("godspeed.tools.shell.platform.system", return_value="Linux"):
        assert shell_mod._detect_shell() == ["/opt/isolate.sh", "-c"]


def test_blank_wrapper_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(shell_mod.SHELL_WRAPPER_ENV, "   ")
    with patch("godspeed.tools.shell.platform.system", return_value="Linux"):
        assert shell_mod._detect_shell() == ["/bin/bash", "-c"]


def test_unset_wrapper_keeps_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(shell_mod.SHELL_WRAPPER_ENV, raising=False)
    with patch("godspeed.tools.shell.platform.system", return_value="Linux"):
        assert shell_mod._detect_shell() == ["/bin/bash", "-c"]


def test_wrapper_is_ignored_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(shell_mod.SHELL_WRAPPER_ENV, "/opt/isolate.sh")
    with (
        patch("godspeed.tools.shell.platform.system", return_value="Windows"),
        patch("godspeed.tools.shell._detect_windows_shell", return_value=["cmd.exe", "/c"]),
    ):
        assert shell_mod._detect_shell() == ["cmd.exe", "/c"]


def test_result_is_cached_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(shell_mod.SHELL_WRAPPER_ENV, "/opt/a.sh")
    with patch("godspeed.tools.shell.platform.system", return_value="Linux"):
        first = shell_mod._detect_shell()
        monkeypatch.setenv(shell_mod.SHELL_WRAPPER_ENV, "/opt/b.sh")
        assert shell_mod._detect_shell() is first
