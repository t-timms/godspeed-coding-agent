"""Tests for shell tool.

After commit 38e7a9d (force-kill process tree fix), ShellTool uses
``subprocess.Popen`` + ``.communicate(timeout=...)`` instead of
``subprocess.run``. Mocks here patch Popen accordingly.

Live timeout / force-kill behavior is covered in tests/test_shell_tool.py
(spawns real subprocesses); this file uses mocks for fast unit-level
coverage of stdout/stderr handling, exit-code dispatch, timeout error
shape, and timeout clamping to MAX_TIMEOUT.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import godspeed.tools.shell as shell_mod
from godspeed.tools.base import ToolContext
from godspeed.tools.shell import (
    MAX_OUTPUT_CHARS,
    ShellTool,
    _detect_shell,
    _kill_process_tree,
)


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


def _mock_popen(*, stdout: str, stderr: str, returncode: int) -> MagicMock:
    """Build a MagicMock that quacks like a subprocess.Popen object.

    ``communicate(timeout=...)`` returns ``(stdout, stderr)`` and the
    mock's ``returncode`` attribute is set so callers reading
    ``proc.returncode`` after communicate() see the expected value.
    """
    mock = MagicMock()
    mock.communicate.return_value = (stdout, stderr)
    mock.returncode = returncode
    mock.pid = 12345
    return mock


class TestKillProcessTree:
    """Unit tests for _kill_process_tree."""

    def test_psutil_not_available(self) -> None:
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                _kill_process_tree(12345)

    def test_no_such_process(self) -> None:
        import psutil as real_psutil

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied

        def _fake_process(pid):
            raise real_psutil.NoSuchProcess(pid)

        mock_psutil.Process = _fake_process

        with patch("godspeed.tools.shell.psutil", mock_psutil, create=True):
            _kill_process_tree(12345)

    def test_kill_process_tree_with_children(self) -> None:
        import builtins

        import psutil as real_psutil

        child_mock = MagicMock()
        parent_mock = MagicMock()
        parent_mock.children.return_value = [child_mock]

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.Process.return_value = parent_mock

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "psutil":
                return mock_psutil
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            _kill_process_tree(12345)

        child_mock.kill.assert_called_once()
        parent_mock.kill.assert_called_once()


class TestDetectShell:
    """Unit tests for _detect_shell."""

    def teardown_method(self) -> None:
        shell_mod._shell_cache = None

    @patch("godspeed.tools.shell._shell_cache", new=None)
    def test_detect_shell_caching(self) -> None:
        shell_mod._shell_cache = None
        result1 = _detect_shell()
        result2 = _detect_shell()
        assert result1 == result2

    def test_detect_shell_returns_cached(self) -> None:
        shell_mod._shell_cache = None
        result = _detect_shell()
        assert result == shell_mod._shell_cache
        result2 = _detect_shell()
        assert result2 is result  # Same list object (cached)

    def test_detect_shell_double_checked_lock(self) -> None:
        import threading

        shell_mod._shell_cache = None
        evt = threading.Event()
        pre_set_shell = ["/bin/bash", "-c"]

        def _set_cache() -> None:
            shell_mod._shell_cache = pre_set_shell
            evt.set()

        t = threading.Thread(target=_set_cache)
        t.start()
        evt.wait()
        result = _detect_shell()
        t.join()
        assert result == pre_set_shell
        assert shell_mod._shell_cache == pre_set_shell

    def test_detect_shell_on_windows(self) -> None:
        shell_mod._shell_cache = None
        with patch("godspeed.tools.shell.platform.system", return_value="Windows"):
            with patch(
                "godspeed.tools.shell.shutil.which",
                return_value="C:\\Program Files\\Git\\bin\\bash.exe",
            ):
                result = _detect_shell()
        assert result[0] == "C:\\Program Files\\Git\\bin\\bash.exe"
        assert result[1] == "-c"

    def test_detect_shell_windows_no_bash(self) -> None:
        shell_mod._shell_cache = None
        with patch("godspeed.tools.shell.platform.system", return_value="Windows"):
            with patch("godspeed.tools.shell.shutil.which", return_value=None):
                with patch.object(shell_mod, "_WINDOWS_GIT_BASH_CANDIDATES", new=()):
                    result = _detect_shell()
        assert result == ["cmd.exe", "/c"]

    def test_detect_shell_unix(self) -> None:
        shell_mod._shell_cache = None
        with patch("godspeed.tools.shell.platform.system", return_value="Linux"):
            result = _detect_shell()
        assert result == ["/bin/bash", "-c"]


class TestShellTool:
    """Test shell command execution."""

    def test_metadata(self, tool: ShellTool) -> None:
        assert tool.name == "shell"
        assert tool.risk_level == "high"
        assert "Run a shell command" in tool.description

    def test_schema_has_required_command(self, tool: ShellTool) -> None:
        schema = tool.get_schema()
        assert "command" in schema["properties"]
        assert schema["required"] == ["command"]

    @pytest.mark.asyncio
    async def test_successful_command(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="hello world\n", stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "echo hello world"}, tool_context)

        assert not result.is_error
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_command_with_stderr(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """A command exiting 0 with stderr warnings is still success; stderr is surfaced."""
        mock_proc = _mock_popen(stdout="output\n", stderr="warning: something\n", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "some-cmd"}, tool_context)

        assert not result.is_error
        assert "output" in result.output
        assert "STDERR" in result.output
        assert "warning" in result.output

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="", stderr="error: not found\n", returncode=1)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "bad-cmd"}, tool_context)

        assert result.is_error
        assert "Exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """TimeoutExpired from communicate() must surface as a timeout failure."""
        mock_proc = MagicMock()
        # First communicate(timeout=...) raises; the post-kill drain returns empty
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="sleep 999", timeout=5),
            ("", ""),
        ]
        mock_proc.pid = 99999

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree") as mock_kill,
        ):
            result = await tool.execute({"command": "sleep 999", "timeout": 5}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        # Force-kill must be triggered with the proc's pid.
        mock_kill.assert_called_once_with(99999)

    @pytest.mark.asyncio
    async def test_timeout_clamped_to_max(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """Timeout exceeding MAX_TIMEOUT is clamped before being passed to communicate()."""
        mock_proc = _mock_popen(stdout="ok", stderr="", returncode=0)

        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            await tool.execute({"command": "echo ok", "timeout": 9999}, tool_context)

        # communicate(timeout=...) should be called with the clamped value (600).
        mock_proc.communicate.assert_called_once()
        assert mock_proc.communicate.call_args.kwargs["timeout"] == 600

    @pytest.mark.asyncio
    async def test_empty_command(self, tool: ShellTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"command": ""}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    @pytest.mark.asyncio
    async def test_negative_timeout(self, tool: ShellTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"command": "echo hi", "timeout": -1}, tool_context)
        assert result.is_error
        assert "positive" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_output(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="", stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "true"}, tool_context)

        assert not result.is_error
        assert "no output" in result.output.lower()

    @pytest.mark.asyncio
    async def test_command_length_limit(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """Test that commands exceeding MAX_COMMAND_LENGTH are rejected."""
        long_command = "echo " + "x" * 10001  # Exceeds 10000 char limit
        result = await tool.execute({"command": long_command}, tool_context)
        assert result.is_error
        assert "exceeds maximum length" in result.error.lower()

    @pytest.mark.asyncio
    async def test_command_whitespace_only(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({"command": "   \t\n  "}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_float_converted(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = _mock_popen(stdout="ok\n", stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "echo ok", "timeout": 5.7}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_timeout_invalid_string(self, tool: ShellTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"command": "echo hi", "timeout": "fast"}, tool_context)
        assert result.is_error
        assert "timeout must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_timeout_zero_fails(self, tool: ShellTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"command": "echo hi", "timeout": 0}, tool_context)
        assert result.is_error
        assert "positive" in result.error.lower()

    @pytest.mark.asyncio
    async def test_shell_not_found(self, tool: ShellTool, tool_context: ToolContext) -> None:

        with patch("godspeed.tools.shell.subprocess.Popen", side_effect=FileNotFoundError("bash")):
            result = await tool.execute({"command": "echo hi"}, tool_context)
        assert result.is_error
        assert "Shell not found" in result.error

    @pytest.mark.asyncio
    async def test_timeout_with_partial_output(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
            ("partial stdout data", "partial stderr data"),
        ]
        mock_proc.pid = 11111
        mock_proc.returncode = None

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree"),
        ):
            result = await tool.execute({"command": "slow-cmd", "timeout": 2}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        assert "partial stdout data" in result.error
        assert "partial stderr data" in result.error

    @pytest.mark.asyncio
    async def test_timeout_drain_also_timeouts(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
        ]
        mock_proc.pid = 11111
        mock_proc.returncode = None

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree"),
        ):
            result = await tool.execute({"command": "stuck-cmd", "timeout": 2}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        assert mock_proc.kill.call_count >= 1

    @pytest.mark.asyncio
    async def test_proc_killed_in_finally(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("out", "")
        mock_proc.returncode = None  # Still running after communicate
        mock_proc.pid = 55555

        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            _discard = await tool.execute({"command": "echo hi"}, tool_context)

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_timeout_used(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="out\n", stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "echo hi"}, tool_context)
        assert not result.is_error
        assert mock_proc.communicate.call_args.kwargs["timeout"] == 120

    def test_get_schema_includes_background(self, tool: ShellTool) -> None:
        schema = tool.get_schema()
        assert "background" in schema["properties"]
        assert schema["properties"]["background"]["type"] == "boolean"

    @pytest.mark.asyncio
    async def test_background_execution(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = MagicMock()

        with patch("godspeed.tools.shell.asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("godspeed.tools.background.BackgroundRegistry") as mock_registry_cls:
                mock_registry = MagicMock()
                mock_registry.active_count = 0
                mock_registry.next_id.return_value = 42
                mock_registry_cls.get.return_value = mock_registry
                result = await tool.execute(
                    {"command": "npm run build", "background": True}, tool_context
                )

        assert not result.is_error
        assert "42" in result.output
        assert "background" in result.output.lower()

    @pytest.mark.asyncio
    async def test_background_max_concurrent(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        from godspeed.tools.background import MAX_CONCURRENT

        with patch("godspeed.tools.background.BackgroundRegistry") as mock_registry_cls:
            mock_registry = MagicMock()
            mock_registry.active_count = MAX_CONCURRENT
            mock_registry_cls.get.return_value = mock_registry
            result = await tool.execute({"command": "npm start", "background": True}, tool_context)

        assert result.is_error
        assert "Too many background processes" in result.error

    # --- Additional timeout / kill-process-tree coverage ---

    @pytest.mark.asyncio
    async def test_timeout_with_stdout_only_tail(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()
        stdout_data = "x" * 3000
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
            (stdout_data, ""),
        ]
        mock_proc.pid = 11111
        mock_proc.returncode = None

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree"),
        ):
            result = await tool.execute({"command": "slow-cmd", "timeout": 2}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        assert "STDOUT tail" in result.error

    @pytest.mark.asyncio
    async def test_timeout_with_stderr_only_tail(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()
        stderr_data = "E" * 3000
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
            ("", stderr_data),
        ]
        mock_proc.pid = 22222
        mock_proc.returncode = None

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree"),
        ):
            result = await tool.execute({"command": "err-cmd", "timeout": 2}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        assert "STDERR tail" in result.error

    @pytest.mark.asyncio
    async def test_timeout_with_both_tails(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
            ("out_tail", "err_tail"),
        ]
        mock_proc.pid = 33333
        mock_proc.returncode = None

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree"),
        ):
            result = await tool.execute({"command": "mixed-cmd", "timeout": 2}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        assert "STDOUT tail" in result.error
        assert "STDERR tail" in result.error

    @pytest.mark.asyncio
    async def test_stderr_only_no_stdout(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="", stderr="warning only\n", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "warn-cmd"}, tool_context)
        assert not result.is_error
        assert "STDERR" in result.output
        assert "warning only" in result.output

    @pytest.mark.asyncio
    async def test_finally_kill_proc_raises(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("out", "")
        mock_proc.returncode = None
        mock_proc.pid = 77777
        mock_proc.kill.side_effect = OSError("already dead")

        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            _discard = await tool.execute({"command": "echo hi"}, tool_context)
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_shell_not_found_error_message(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        with patch(
            "godspeed.tools.shell.subprocess.Popen", side_effect=FileNotFoundError("no-bash")
        ):
            result = await tool.execute({"command": "echo test"}, tool_context)
        assert result.is_error
        assert "Shell not found" in result.error

    @pytest.mark.asyncio
    async def test_background_output_collection_launched(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        mock_proc = MagicMock()

        with patch("godspeed.tools.shell.asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("godspeed.tools.background.BackgroundRegistry") as mock_registry_cls:
                mock_registry = MagicMock()
                mock_registry.active_count = 0
                mock_registry.next_id.return_value = 7
                mock_registry_cls.get.return_value = mock_registry
                with patch("godspeed.tools.background.asyncio.create_task") as mock_create_task:
                    result = await tool.execute(
                        {"command": "long-running", "background": True}, tool_context
                    )

        assert not result.is_error
        assert "7" in result.output
        mock_create_task.assert_called_once()
        mock_registry.add.assert_called_once()

    # --- kill process tree exhaustive coverage ---

    def test_kill_process_tree_with_grandchildren(self) -> None:
        import builtins

        import psutil as real_psutil

        grandchild = MagicMock()
        child = MagicMock()
        parent = MagicMock()
        parent.children.return_value = [child, grandchild]

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.Process.return_value = parent

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "psutil":
                return mock_psutil
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            _kill_process_tree(12345)

        child.kill.assert_called_once()
        parent.kill.assert_called_once()
        grandchild.kill.assert_called_once()

    def test_kill_process_tree_no_children(self) -> None:
        import builtins

        import psutil as real_psutil

        parent = MagicMock()
        parent.children.return_value = []

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.Process.return_value = parent

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "psutil":
                return mock_psutil
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            _kill_process_tree(12345)

        parent.kill.assert_called_once()

    def test_kill_process_tree_child_access_denied(self) -> None:
        import builtins

        import psutil as real_psutil

        bad_child = MagicMock()
        bad_child.kill.side_effect = real_psutil.AccessDenied("denied")
        good_child = MagicMock()
        parent = MagicMock()
        parent.children.return_value = [bad_child, good_child]

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.Process.return_value = parent

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "psutil":
                return mock_psutil
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            _kill_process_tree(12345)

        good_child.kill.assert_called_once()
        parent.kill.assert_called_once()

    def test_kill_process_tree_parent_access_denied(self) -> None:
        import builtins

        import psutil as real_psutil

        parent = MagicMock()
        parent.kill.side_effect = real_psutil.AccessDenied("denied")
        parent.children.return_value = []

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess
        mock_psutil.AccessDenied = real_psutil.AccessDenied
        mock_psutil.Process.return_value = parent

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "psutil":
                return mock_psutil
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            _kill_process_tree(12345)

        parent.kill.assert_called_once()


class TestOutputTruncation:
    """Output cap behavior for completed (non-timeout) commands."""

    @pytest.mark.asyncio
    async def test_output_at_cap_passes_through_exactly(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        stdout_data = "y" * MAX_OUTPUT_CHARS
        mock_proc = _mock_popen(stdout=stdout_data, stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "big-cmd"}, tool_context)

        assert not result.is_error
        assert result.output == stdout_data

    @pytest.mark.asyncio
    async def test_output_below_cap_passes_through_exactly(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        stdout_data = "z" * (MAX_OUTPUT_CHARS - 100)
        mock_proc = _mock_popen(stdout=stdout_data, stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "small-cmd"}, tool_context)

        assert not result.is_error
        assert result.output == stdout_data

    @pytest.mark.asyncio
    async def test_output_over_cap_truncated_with_marker(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        stdout_data = "a" * MAX_OUTPUT_CHARS + "b" * 5000
        mock_proc = _mock_popen(stdout=stdout_data, stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "huge-cmd"}, tool_context)

        assert not result.is_error
        assert result.output.startswith("a" * MAX_OUTPUT_CHARS)
        assert "... (5000 chars truncated)" in result.output
        assert "b" not in result.output

    @pytest.mark.asyncio
    async def test_failure_path_truncated_too(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        stderr_data = "e" * (MAX_OUTPUT_CHARS + 2500)
        mock_proc = _mock_popen(stdout="", stderr=stderr_data, returncode=1)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "failing-cmd"}, tool_context)

        combined_len = len("STDERR:\n") + len(stderr_data)
        expected_truncated = combined_len - MAX_OUTPUT_CHARS
        assert result.is_error
        assert "Exit code 1" in result.error
        assert f"... ({expected_truncated} chars truncated)" in result.error

    @pytest.mark.asyncio
    async def test_timeout_path_unchanged_by_output_cap(
        self, tool: ShellTool, tool_context: ToolContext
    ) -> None:
        """Timeout path still tails 2000 chars — MAX_OUTPUT_CHARS does not apply."""
        stdout_data = "t" * (MAX_OUTPUT_CHARS + 1000)
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="cmd", timeout=5),
            (stdout_data, ""),
        ]
        mock_proc.pid = 44444
        mock_proc.returncode = None

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree"),
        ):
            result = await tool.execute({"command": "slow-cmd", "timeout": 2}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        assert "STDOUT tail" in result.error
        assert "t" * 2000 in result.error
        assert "chars truncated" not in result.error
