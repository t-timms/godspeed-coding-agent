"""Additional tests for test_runner tool to increase coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.test_runner import (
    TestRunnerTool,
    _run_cargo_test,
    _run_go_test,
    _run_jest,
    _run_pytest,
    _run_tests,
    _run_vitest,
    detect_framework,
)


@pytest.fixture
def runner() -> TestRunnerTool:
    return TestRunnerTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


class TestDetectFramework:
    def test_detects_pytest_from_setup_py(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text("# setup.py", encoding="utf-8")
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_pytest_default_for_python(self, tmp_path: Path) -> None:
        """Test that pytest is default for Python projects."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"', encoding="utf-8")
        assert detect_framework(tmp_path) == "pytest"

    def test_pyproject_read_error(self, tmp_path: Path) -> None:
        """Test pyproject.toml read error."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]", encoding="utf-8")
        # Make the file unreadable after creation (simulate read error)
        with patch("pathlib.Path.read_text", side_effect=OSError("Cannot read")):
            result = detect_framework(tmp_path)
            # Should still return pytest as default for Python projects
            assert result == "pytest"

    def test_package_json_read_error(self, tmp_path: Path) -> None:
        """Test package.json read error."""
        package_json = tmp_path / "package.json"
        package_json.write_text('{"scripts": {}}', encoding="utf-8")
        with patch("pathlib.Path.read_text", side_effect=OSError("Cannot read")):
            result = detect_framework(tmp_path)
            assert result == ""

    def test_package_json_invalid_json(self, tmp_path: Path) -> None:
        """Test package.json with invalid JSON."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{invalid json}", encoding="utf-8")
        result = detect_framework(tmp_path)
        assert result == ""


class TestExecute:
    @pytest.mark.asyncio
    async def test_with_framework_argument(
        self,
        runner: TestRunnerTool,
        ctx: ToolContext,
    ) -> None:
        """Test execute with framework argument."""
        with patch("godspeed.tools.test_runner._RUNNERS") as mock_runners:
            mock_runners.get.return_value = lambda cwd, target, repeat: "mocked result"
            await runner.execute({"framework": "pytest"}, ctx)
            mock_runners.get.assert_called_once_with("pytest")

    @pytest.mark.asyncio
    async def test_runner_returns_none(self, runner: TestRunnerTool, ctx: ToolContext) -> None:
        """Test when runner returns None."""
        with patch("godspeed.tools.test_runner._RUNNERS") as mock_runners:
            mock_runners.get.return_value = None
            result = await runner.execute({"framework": "invalid"}, ctx)
            assert result.is_error
            assert "Unknown test framework" in result.error


class TestRunTests:
    def test_binary_not_found(self, tmp_path: Path) -> None:
        """Test when test binary is not found."""
        with patch("shutil.which", return_value=None):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest")
            assert "not found" in result.output

    def test_timeout(self, tmp_path: Path) -> None:
        """Test test timeout."""
        from subprocess import TimeoutExpired

        with patch("subprocess.run", side_effect=TimeoutExpired(cmd="pytest", timeout=60)):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest")
            assert "timed out" in result.error

    def test_os_error(self, tmp_path: Path) -> None:
        """Test OS error during test run."""
        with patch("subprocess.run", side_effect=OSError("Mocked OS error")):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest")
            assert "Failed to run" in result.error

    def test_output_truncated(self, tmp_path: Path) -> None:
        """Test output truncation."""
        from godspeed.tools.test_runner import MAX_OUTPUT_CHARS

        long_output = "x" * (MAX_OUTPUT_CHARS + 100)
        mock_result = MagicMock()
        mock_result.stdout = long_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = _run_tests(["pytest"], tmp_path, "pytest")
            assert "truncated" in result.output

    def test_stderr_only(self, tmp_path: Path) -> None:
        """Test when only stderr has content."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error output"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = _run_tests(["pytest"], tmp_path, "pytest")
            assert "error output" in result.output

    def test_both_stdout_stderr(self, tmp_path: Path) -> None:
        """Test when both stdout and stderr have content."""
        mock_result = MagicMock()
        mock_result.stdout = "stdout output"
        mock_result.stderr = "stderr output"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = _run_tests(["pytest"], tmp_path, "pytest")
            assert "stdout output" in result.output
            assert "STDERR" in result.output


class TestRunPytest:
    def test_with_target(self, tmp_path: Path) -> None:
        """Test pytest with target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_pytest(tmp_path, "tests/test_foo.py")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "tests/test_foo.py" in args[0]

    def test_without_target(self, tmp_path: Path) -> None:
        """Test pytest without target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_pytest(tmp_path, "")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "tests/test_foo.py" not in args[0]


class TestRunJest:
    def test_with_target(self, tmp_path: Path) -> None:
        """Test jest with target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_jest(tmp_path, "src/foo.test.js")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "src/foo.test.js" in args[0]

    def test_without_target(self, tmp_path: Path) -> None:
        """Test jest without target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_jest(tmp_path, "")
            mock_run.assert_called_once()


class TestRunVitest:
    def test_with_target(self, tmp_path: Path) -> None:
        """Test vitest with target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_vitest(tmp_path, "src/foo.test.ts")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "src/foo.test.ts" in args[0]


class TestRunGoTest:
    def test_with_target(self, tmp_path: Path) -> None:
        """Test go test with target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_go_test(tmp_path, "./...")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "./..." in args[0]

    def test_without_target(self, tmp_path: Path) -> None:
        """Test go test without target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_go_test(tmp_path, "")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "./..." in args[0]


class TestRunCargoTest:
    def test_with_target(self, tmp_path: Path) -> None:
        """Test cargo test with target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_cargo_test(tmp_path, "my_test")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "--" in args[0]
            assert "my_test" in args[0]

    def test_without_target(self, tmp_path: Path) -> None:
        """Test cargo test without target."""
        with patch("godspeed.tools.test_runner._run_tests") as mock_run:
            _run_cargo_test(tmp_path, "")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "--" not in args[0]
