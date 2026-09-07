"""Tests for the test runner tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.test_runner import (
    MAX_REPEAT,
    TestRunnerTool,
    _run_tests,
    detect_framework,
)


@pytest.fixture
def runner() -> TestRunnerTool:
    return TestRunnerTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


class TestDetectFramework:
    """Test framework auto-detection."""

    def test_detects_pytest_from_conftest(self, tmp_path: Path) -> None:
        (tmp_path / "conftest.py").write_text("", encoding="utf-8")
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_pytest_from_tests_conftest(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "conftest.py").write_text("", encoding="utf-8")
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_pytest_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]',
            encoding="utf-8",
        )
        assert detect_framework(tmp_path) == "pytest"

    def test_detects_jest(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"scripts": {"test": "jest"}}',
            encoding="utf-8",
        )
        assert detect_framework(tmp_path) == "jest"

    def test_detects_vitest(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"scripts": {"test": "vitest run"}}',
            encoding="utf-8",
        )
        assert detect_framework(tmp_path) == "vitest"

    def test_detects_go(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module example.com/foo", encoding="utf-8")
        assert detect_framework(tmp_path) == "go"

    def test_detects_cargo(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "foo"', encoding="utf-8")
        assert detect_framework(tmp_path) == "cargo"

    def test_no_framework(self, tmp_path: Path) -> None:
        assert detect_framework(tmp_path) == ""


class TestTestRunnerTool:
    """Test the test runner tool."""

    def test_name(self, runner: TestRunnerTool) -> None:
        assert runner.name == "test_runner"

    def test_schema(self, runner: TestRunnerTool) -> None:
        schema = runner.get_schema()
        assert "target" in schema["properties"]
        assert "framework" in schema["properties"]

    @pytest.mark.asyncio
    async def test_no_framework_detected(self, runner: TestRunnerTool, ctx: ToolContext) -> None:
        result = await runner.execute({}, ctx)
        assert not result.is_error
        assert "No test framework detected" in result.output

    @pytest.mark.asyncio
    async def test_unknown_framework(self, runner: TestRunnerTool, ctx: ToolContext) -> None:
        result = await runner.execute({"framework": "foobar"}, ctx)
        assert result.is_error
        assert "Unknown test framework" in result.error


class TestPassK:
    """pass^k semantics: repeat>1 requires k-of-k passes."""

    @staticmethod
    def _mock_result(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.stderr = stderr
        mock_result.returncode = returncode
        return mock_result

    def test_schema_includes_repeat(self, runner: TestRunnerTool) -> None:
        schema = runner.get_schema()
        assert "repeat" in schema["properties"]
        assert schema["properties"]["repeat"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_execute_forwards_repeat(self, runner: TestRunnerTool, ctx: ToolContext) -> None:
        """execute forwards repeat to the framework runner."""
        with patch("godspeed.tools.test_runner._RUNNERS") as mock_runners:
            seen: dict[str, int] = {}
            mock_runners.get.return_value = lambda cwd, target, repeat: seen.setdefault(
                "repeat", repeat
            )
            await runner.execute({"framework": "pytest", "repeat": 3}, ctx)
            assert seen["repeat"] == 3

    def test_repeat_all_pass(self, tmp_path: Path) -> None:
        """repeat=3, all runs pass -> PASS with 3 run lines."""
        results = [self._mock_result(0, "ok") for _ in range(3)]
        with patch("subprocess.run", side_effect=results):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest", repeat=3)
        assert not result.is_error
        assert "Tests PASSED (pytest, 3/3 runs)" in result.output
        assert "run 1: exit=0" in result.output
        assert "run 2: exit=0" in result.output
        assert "run 3: exit=0" in result.output

    def test_repeat_first_failure_early_stop(self, tmp_path: Path) -> None:
        """repeat=3, first run fails -> early stop, runs_attempted=1."""
        with patch("subprocess.run", side_effect=[self._mock_result(1, "boom")]) as mock_run:
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest", repeat=3)
        assert mock_run.call_count == 1
        assert "Tests FAILED (pytest, 0/3 runs passed" in result.output
        assert "runs_attempted: 1" in result.output
        assert "runs_passed: 0" in result.output
        assert "boom" in result.output

    def test_repeat_second_failure_early_stop(self, tmp_path: Path) -> None:
        """repeat=3, second run fails -> early stop, runs_attempted=2."""
        results = [self._mock_result(0, "ok"), self._mock_result(1, "boom")]
        with patch("subprocess.run", side_effect=results) as mock_run:
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest", repeat=3)
        assert mock_run.call_count == 2
        assert "Tests FAILED (pytest, 1/3 runs passed" in result.output
        assert "runs_attempted: 2" in result.output
        assert "runs_passed: 1" in result.output
        assert "first_failure_output:" in result.output
        assert "boom" in result.output

    def test_repeat_zero_treated_as_one(self, tmp_path: Path) -> None:
        """repeat=0 -> treated as 1 with a clamped note."""
        with patch("subprocess.run", return_value=self._mock_result(0, "ok")):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest", repeat=0)
        assert "note: repeat clamped to 1" in result.output
        assert "Tests PASSED (pytest):" in result.output

    def test_repeat_negative_treated_as_one(self, tmp_path: Path) -> None:
        """repeat=-2 -> treated as 1 with a clamped note."""
        with patch("subprocess.run", return_value=self._mock_result(0, "ok")):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest", repeat=-2)
        assert "note: repeat clamped to 1" in result.output
        assert "Tests PASSED (pytest):" in result.output

    def test_repeat_above_max_clamped(self, tmp_path: Path) -> None:
        """repeat > MAX_REPEAT -> clamped with note, runs MAX_REPEAT times."""
        results = [self._mock_result(0, "ok") for _ in range(MAX_REPEAT)]
        with patch("subprocess.run", side_effect=results) as mock_run:
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest", repeat=99)
        assert mock_run.call_count == MAX_REPEAT
        assert f"note: repeat clamped to {MAX_REPEAT}" in result.output
        assert f"Tests PASSED (pytest, {MAX_REPEAT}/{MAX_REPEAT} runs)" in result.output

    def test_single_run_output_unchanged(self, tmp_path: Path) -> None:
        """repeat=1 (default) keeps the original single-run output format."""
        with patch("subprocess.run", return_value=self._mock_result(0, "all good")):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest")
        assert result.output == "Tests PASSED (pytest):\nall good"

        with patch("subprocess.run", return_value=self._mock_result(1, "bad")):
            result = _run_tests(["pytest", "-x"], tmp_path, "pytest")
        assert result.output == "Tests FAILED (pytest, exit=1):\nbad"
