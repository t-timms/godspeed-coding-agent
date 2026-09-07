"""Tests for hook executor."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from godspeed.hooks import HookEvent
from godspeed.hooks.config import HookDefinition
from godspeed.hooks.executor import HookExecutor


def _make_executor(
    hooks: list[HookDefinition],
    tmp_path: Path,
) -> HookExecutor:
    return HookExecutor(hooks=hooks, cwd=tmp_path, session_id="test-session-123")


def _py_cmd(script: str) -> str:
    """Build a shell command that runs a Python snippet.

    Uses ``-c`` with properly quoted executable path for cmd.exe.
    """
    return f'"{sys.executable}" -c "{script}"'


class TestPreToolHooks:
    """Test pre_tool_call hook execution."""

    def test_allows_when_no_hooks(self, tmp_path: Path) -> None:
        executor = _make_executor([], tmp_path)
        assert executor.run_pre_tool("shell") is True

    def test_allows_on_success(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("print('ok')"),
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is True

    def test_blocks_on_failure(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("import sys; sys.exit(1)"),
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is False

    def test_tool_filter_matches(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("import sys; sys.exit(1)"),
            tools=["shell"],
        )
        executor = _make_executor([hook], tmp_path)
        # Should block shell
        assert executor.run_pre_tool("shell") is False
        # Should allow file_read (not in tool list)
        assert executor.run_pre_tool("file_read") is True

    def test_tool_filter_none_matches_all(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("import sys; sys.exit(1)"),
            tools=None,
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is False
        assert executor.run_pre_tool("file_read") is False

    def test_template_variable_expansion(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd(f"open(r'{marker}', 'w').write('{{tool_name}}')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_pre_tool("shell")
        assert marker.exists()
        assert marker.read_text() == "shell"


class TestPostToolHooks:
    """Test post_tool_call hook execution."""

    def test_runs_post_hook(self, tmp_path: Path) -> None:
        marker = tmp_path / "post_marker.txt"
        hook = HookDefinition(
            event="post_tool_call",
            command=_py_cmd(f"open(r'{marker}', 'w').write('done')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_post_tool("shell")
        assert marker.exists()
        assert marker.read_text() == "done"

    def test_post_hook_tool_filter(self, tmp_path: Path) -> None:
        marker = tmp_path / "post_marker.txt"
        hook = HookDefinition(
            event="post_tool_call",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('done')\"",
            tools=["shell"],
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_post_tool("file_read")
        assert not marker.exists()


class TestSessionHooks:
    """Test session lifecycle hooks."""

    def test_pre_session(self, tmp_path: Path) -> None:
        marker = tmp_path / "pre_session.txt"
        hook = HookDefinition(
            event="session_start",
            command=_py_cmd(f"open(r'{marker}', 'w').write('started')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_pre_session()
        assert marker.exists()

    def test_post_session(self, tmp_path: Path) -> None:
        marker = tmp_path / "post_session.txt"
        hook = HookDefinition(
            event="session_end",
            command=_py_cmd(f"open(r'{marker}', 'w').write('ended')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_post_session()
        assert marker.exists()

    def test_only_session_hooks_run(self, tmp_path: Path) -> None:
        marker = tmp_path / "wrong.txt"
        hook = HookDefinition(
            event="pre_tool_call",
            command=f"{sys.executable} -c \"open(r'{marker}', 'w').write('wrong')\"",
        )
        executor = _make_executor([hook], tmp_path)
        executor.run_pre_session()
        executor.run_post_session()
        assert not marker.exists()


class TestHookTimeout:
    """Test hook timeout handling."""

    def test_timeout_returns_nonzero(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "import time; time.sleep(10)"',
            timeout=1,
        )
        executor = _make_executor([hook], tmp_path)
        assert executor.run_pre_tool("shell") is False


class TestHookErrors:
    """Test hook error handling."""

    def test_bad_template_variable(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command="echo {nonexistent_var}",
        )
        executor = _make_executor([hook], tmp_path)
        # Bad template returns exit code 1, blocking the tool
        assert executor.run_pre_tool("shell") is False

    def test_multiple_hooks_all_must_pass(self, tmp_path: Path) -> None:
        hook1 = HookDefinition(
            event="pre_tool_call",
            command=f"{sys.executable} -c \"print('ok')\"",
        )
        hook2 = HookDefinition(
            event="pre_tool_call",
            command=f'{sys.executable} -c "raise SystemExit(1)"',
        )
        executor = _make_executor([hook1, hook2], tmp_path)
        assert executor.run_pre_tool("shell") is False


class TestFireMethod:
    """Test the generic fire() method for all hook events."""

    def test_fire_permission_denied(self, tmp_path: Path) -> None:
        """Test firing permission_denied event."""
        marker = tmp_path / "permission_denied.txt"
        hook = HookDefinition(
            event="permission_denied",
            command=_py_cmd(f"open(r'{marker}', 'w').write('denied')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.PERMISSION_DENIED, tool="shell", pattern="dangerous")
        assert marker.exists()

    def test_fire_notification(self, tmp_path: Path) -> None:
        marker = tmp_path / "notification.txt"
        hook = HookDefinition(
            event="notification",
            command=_py_cmd(f"open(r'{marker}', 'w').write('notified')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.NOTIFICATION, message="hello")
        assert marker.read_text() == "notified"

    def test_fire_stop(self, tmp_path: Path) -> None:
        marker = tmp_path / "stop.txt"
        hook = HookDefinition(
            event="stop",
            command=_py_cmd(f"open(r'{marker}', 'w').write('stopped')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.STOP)
        assert marker.read_text() == "stopped"

    def test_fire_stuck_loop_detected(self, tmp_path: Path) -> None:
        """Test firing stuck_loop_detected event."""
        marker = tmp_path / "stuck_loop.txt"
        hook = HookDefinition(
            event="stuck_loop_detected",
            command=_py_cmd(f"open(r'{marker}', 'w').write('stuck')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.STUCK_LOOP_DETECTED, iterations=3, last_error="error")
        assert marker.exists()

    def test_fire_budget_exceeded(self, tmp_path: Path) -> None:
        """Test firing budget_exceeded event."""
        marker = tmp_path / "budget.txt"
        hook = HookDefinition(
            event="budget_exceeded",
            command=_py_cmd(f"open(r'{marker}', 'w').write('over')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.BUDGET_EXCEEDED, cost_usd="10.50")
        assert marker.exists()

    def test_fire_gs_environment_variables(self, tmp_path: Path) -> None:
        """Test that fire with context passes through to hook execution."""
        marker = tmp_path / "env_check.txt"
        hook = HookDefinition(
            event="pre_permission_check",
            command=_py_cmd(f"open(r'{marker}', 'w').write('ok')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.PRE_PERMISSION_CHECK, tool_name="shell")
        assert marker.exists()

    def test_fire_no_matching_hooks_returns_none(self, tmp_path: Path) -> None:
        """Test that fire returns None when no hooks match."""
        hook = HookDefinition(
            event="permission_denied",
            command=f"{sys.executable} -c \"print('ok')\"",
        )
        executor = _make_executor([hook], tmp_path)
        # Fire a different event - should return None (no hooks for this event)
        result = executor.fire(HookEvent.SESSION_START)
        assert result is None

    def test_fire_pre_event_blocks_execution(self, tmp_path: Path) -> None:
        """Test that pre_ events can block by returning False."""
        hook = HookDefinition(
            event="pre_permission_check",
            command=f'{sys.executable} -c "import sys; sys.exit(1)"',
        )
        executor = _make_executor([hook], tmp_path)
        result = executor.fire(HookEvent.PRE_PERMISSION_CHECK, tool_name="shell")
        assert result is False

    def test_fire_post_event_does_not_block(self, tmp_path: Path) -> None:
        """Test that post_ events don't block even on failure."""
        hook = HookDefinition(
            event="post_tool_call",
            command=f'{sys.executable} -c "import sys; sys.exit(1)"',
        )
        executor = _make_executor([hook], tmp_path)
        result = executor.fire(HookEvent.POST_TOOL_CALL, tool_name="shell")
        # Post events return None (advisory, don't block)
        assert result is None

    def test_all_25_events_are_valid(self) -> None:
        events = [
            HookEvent.SESSION_START,
            HookEvent.SESSION_END,
            HookEvent.PRE_PERMISSION_CHECK,
            HookEvent.POST_PERMISSION_CHECK,
            HookEvent.PERMISSION_DENIED,
            HookEvent.PRE_TOOL_CALL,
            HookEvent.POST_TOOL_CALL,
            HookEvent.PRE_COMPACTION,
            HookEvent.POST_COMPACTION,
            HookEvent.CONTEXT_THRESHOLD_75,
            HookEvent.CONTEXT_THRESHOLD_50,
            HookEvent.CONTEXT_THRESHOLD_25,
            HookEvent.PRE_SUBAGENT_SPAWN,
            HookEvent.POST_SUBAGENT_COMPLETE,
            HookEvent.POST_SUBAGENT_STOP,
            HookEvent.SUBAGENT_ERROR,
            HookEvent.SECRET_DETECTED,
            HookEvent.DANGEROUS_COMMAND,
            HookEvent.STUCK_LOOP_DETECTED,
            HookEvent.BUDGET_EXCEEDED,
            HookEvent.WORKFLOW_PHASE_COMPLETE,
            HookEvent.WORKFLOW_COMPLETE,
            HookEvent.WORKFLOW_REJECTED,
            HookEvent.STOP,
            HookEvent.NOTIFICATION,
        ]
        assert len(events) == 25
        assert len(set(e.value for e in HookEvent)) == 25


class TestNeedsShell:
    """Tests for _needs_shell shell metacharacter detection."""

    def test_pipe_needs_shell(self) -> None:
        from godspeed.hooks.executor import _needs_shell

        assert _needs_shell("echo hello | grep world") is True

    def test_no_metachars_does_not_need_shell(self) -> None:
        from godspeed.hooks.executor import _needs_shell

        assert _needs_shell("echo hello world") is False

    def test_semicolon_needs_shell(self) -> None:
        from godspeed.hooks.executor import _needs_shell

        assert _needs_shell("ls; pwd") is True

    def test_ampersand_needs_shell(self) -> None:
        from godspeed.hooks.executor import _needs_shell

        assert _needs_shell("ls &") is True

    def test_redirect_needs_shell(self) -> None:
        from godspeed.hooks.executor import _needs_shell

        assert _needs_shell("echo hi > out.txt") is True
        assert _needs_shell("echo hi < in.txt") is True


class TestExecuteNonWindows:
    """Tests for _execute on non-Windows platform (shlex.split path)."""

    def test_execute_on_posix_uses_shlex(self, tmp_path: Path) -> None:
        with patch("sys.platform", "linux"):
            hook = HookDefinition(
                event="pre_tool_call",
                command=_py_cmd("print('hello from linux')"),
            )
            executor = _make_executor([hook], tmp_path)
            result = executor.fire(HookEvent.PRE_TOOL_CALL, tool_name="shell")
            assert result is None


class TestExecuteValueError:
    """Tests for ValueError handling in _execute (e.g., bad command args)."""

    def test_valueerror_returns_1(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("print('ok')"),
        )
        executor = _make_executor([hook], tmp_path)
        with patch("subprocess.run", side_effect=ValueError("Bad args")):
            result = executor.fire(HookEvent.PRE_TOOL_CALL, tool_name="shell")
            assert result is False


class TestExecuteOSError:
    """Tests for OSError handling in _execute."""

    def test_oserror_returns_1(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("print('ok')"),
        )
        executor = _make_executor([hook], tmp_path)
        with patch("subprocess.run", side_effect=OSError("Exec not found")):
            result = executor.fire(HookEvent.PRE_TOOL_CALL, tool_name="shell")
            assert result is False


class TestExecuteStdoutStderr:
    """Tests for stdout/stderr logging in _execute."""

    def test_stdout_logged_on_success(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("print('hello stdout')"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.PRE_TOOL_CALL, tool_name="shell")

    def test_stderr_logged_on_error(self, tmp_path: Path) -> None:
        hook = HookDefinition(
            event="pre_tool_call",
            command=_py_cmd("import sys; print('error text', file=sys.stderr)"),
        )
        executor = _make_executor([hook], tmp_path)
        executor.fire(HookEvent.PRE_TOOL_CALL, tool_name="shell")
