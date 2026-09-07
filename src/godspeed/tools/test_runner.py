"""Test runner tool — detect and run project test suites.

Auto-detects the project's test framework and runs targeted tests after edits.
Supports pytest, jest/vitest, go test, and cargo test.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Timeout for test execution (generous — tests can be slow)
TEST_TIMEOUT = 60

# Max output to capture from test runner
MAX_OUTPUT_CHARS = 5000

# Max times a test command may be repeated (pass^k). Values above this are
# clamped with a note in the output.
MAX_REPEAT = 5


class TestRunnerTool(Tool):
    """Run project tests to validate changes.

    Auto-detects the test framework based on project files:
    - pytest (pyproject.toml/setup.py/conftest.py)
    - jest/vitest (package.json with test script)
    - go test (go.mod)
    - cargo test (Cargo.toml)

    Can run all tests or target specific files/directories.
    """

    @property
    def name(self) -> str:
        return "test_runner"

    @property
    def description(self) -> str:
        return (
            "Run project tests to validate changes. Auto-detects the test framework. "
            "Optionally target a specific file or directory. Returns pass/fail with output. "
            "Supports pass^k: set repeat>1 to run the same command multiple times and "
            "require ALL runs to pass — detects flaky tests and counters reward hacking "
            "by requiring k-of-k passes."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Optional: specific test file or directory to run. "
                        "If empty, runs the full test suite."
                    ),
                },
                "framework": {
                    "type": "string",
                    "description": (
                        "Optional: force a specific framework (pytest, jest, vitest, "
                        "go, cargo). Auto-detected if omitted."
                    ),
                },
                "repeat": {
                    "type": "integer",
                    "description": (
                        "Optional: run the same test command this many times (pass^k). "
                        "Verdict is PASS only if ALL runs pass; stops early on the first "
                        "failure. Values above MAX_REPEAT (5) are clamped. Defaults to 1."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        target = arguments.get("target", "")
        framework = arguments.get("framework", "")
        repeat = arguments.get("repeat", 1)

        if not framework:
            framework = detect_framework(context.cwd)
            if not framework:
                return ToolResult.success(
                    "No test framework detected. Looked for: "
                    "pytest (conftest.py/pyproject.toml), jest/vitest (package.json), "
                    "go test (go.mod), cargo test (Cargo.toml)."
                )

        runner = _RUNNERS.get(framework)
        if runner is None:
            return ToolResult.failure(
                f"Unknown test framework: {framework}. Supported: {', '.join(_RUNNERS.keys())}"
            )

        return runner(context.cwd, target, repeat)


def detect_framework(cwd: Path) -> str:
    """Detect the project's test framework from project files."""
    # Python: pytest
    if (cwd / "conftest.py").exists() or (cwd / "tests" / "conftest.py").exists():
        return "pytest"
    if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
        # Check if pytest is in pyproject.toml
        pyproject = cwd / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "pytest" in content:
                    return "pytest"
            except OSError:
                logger.debug("Could not read pyproject.toml for test detection")
        return "pytest"  # Default for Python projects

    # JavaScript/TypeScript: jest or vitest
    package_json = cwd / "package.json"
    if package_json.exists():
        try:
            import json

            data = json.loads(package_json.read_text(encoding="utf-8"))
            scripts = data.get("scripts", {})
            if "test" in scripts:
                test_cmd = scripts["test"]
                if "vitest" in test_cmd:
                    return "vitest"
                return "jest"
        except (OSError, json.JSONDecodeError):
            logger.debug("Could not read package.json for test detection")

    # Go
    if (cwd / "go.mod").exists():
        return "go"

    # Rust
    if (cwd / "Cargo.toml").exists():
        return "cargo"

    return ""


def _run_tests(
    cmd: list[str],
    cwd: Path,
    framework: str,
    *,
    repeat: int = 1,
) -> ToolResult:
    """Run a test command and return structured results.

    With repeat > 1 (pass^k), the command runs up to ``repeat`` times and the
    verdict is PASS only if every run passes. Stops early on the first failure.
    """
    bin_name = cmd[0]
    resolved_bin = shutil.which(bin_name)
    if resolved_bin is None:
        return ToolResult.success(f"{bin_name} not found — cannot run tests.")

    cmd[0] = resolved_bin

    repeat, clamp_note = _normalize_repeat(repeat)

    if repeat == 1:
        result = _run_once(cmd, cwd, framework)
        if clamp_note:
            return ToolResult(
                output=f"{clamp_note}\n{result.output}",
                error=result.error,
                is_error=result.is_error,
            )
        return result

    run_lines: list[str] = []
    runs_attempted = 0
    runs_passed = 0
    first_failure_output = ""
    first_failure_exit: int | None = None

    for attempt in range(1, repeat + 1):
        runs_attempted = attempt
        result, duration, run_error = _run_command(cmd, cwd, framework)
        if run_error:
            run_lines.append(f"run {attempt}: error, {duration:.2f}s")
            first_failure_output = run_error
            break
        run_lines.append(f"run {attempt}: exit={result.returncode}, {duration:.2f}s")
        if result.returncode != 0:
            first_failure_output = _format_combined(result)
            first_failure_exit = result.returncode
            break
        runs_passed = attempt

    body = "\n".join(run_lines)
    if clamp_note:
        body = f"{clamp_note}\n{body}"

    if runs_passed == repeat:
        return ToolResult.success(
            f"Tests PASSED ({framework}, {runs_passed}/{repeat} runs):\n{body}"
        )

    exit_desc = f", exit={first_failure_exit}" if first_failure_exit is not None else ""
    return ToolResult.success(
        f"Tests FAILED ({framework}, {runs_passed}/{repeat} runs passed{exit_desc}):\n"
        f"{body}\n"
        f"runs_attempted: {runs_attempted}\n"
        f"runs_passed: {runs_passed}\n"
        f"first_failure_output:\n{first_failure_output}"
    )


def _normalize_repeat(repeat: int) -> tuple[int, str]:
    """Clamp repeat into [1, MAX_REPEAT]; returns (repeat, note)."""
    if repeat <= 0:
        return 1, "note: repeat clamped to 1 (min allowed)"
    if repeat > MAX_REPEAT:
        return MAX_REPEAT, f"note: repeat clamped to {MAX_REPEAT} (max allowed)"
    return repeat, ""


def _run_command(
    cmd: list[str],
    cwd: Path,
    framework: str,
) -> tuple[subprocess.CompletedProcess[str] | None, float, str]:
    """Run a command once; returns (result, duration_seconds, error)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        return (
            None,
            time.monotonic() - start,
            f"Tests timed out after {TEST_TIMEOUT}s ({framework})",
        )
    except OSError as exc:
        return None, time.monotonic() - start, f"Failed to run {framework}: {exc}"
    return result, time.monotonic() - start, ""


def _format_combined(result: subprocess.CompletedProcess[str]) -> str:
    """Combine stdout/stderr and truncate to MAX_OUTPUT_CHARS."""
    output = result.stdout.strip()
    errors = result.stderr.strip()
    combined = output
    if errors:
        combined = f"{output}\n\nSTDERR:\n{errors}" if output else errors

    if len(combined) > MAX_OUTPUT_CHARS:
        truncated = len(combined) - MAX_OUTPUT_CHARS
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n... ({truncated} chars truncated)"
    return combined


def _run_once(cmd: list[str], cwd: Path, framework: str) -> ToolResult:
    """Run a test command exactly once (the repeat==1 path)."""
    result, _duration, run_error = _run_command(cmd, cwd, framework)
    if run_error:
        return ToolResult.failure(run_error)

    combined = _format_combined(result)
    if result.returncode == 0:
        return ToolResult.success(f"Tests PASSED ({framework}):\n{combined}")

    return ToolResult.success(f"Tests FAILED ({framework}, exit={result.returncode}):\n{combined}")


def _run_pytest(cwd: Path, target: str, repeat: int = 1) -> ToolResult:
    """Run pytest."""
    cmd = ["pytest", "-x", "--tb=short", "-q"]
    if target:
        cmd.append(target)
    return _run_tests(cmd, cwd, "pytest", repeat=repeat)


def _run_jest(cwd: Path, target: str, repeat: int = 1) -> ToolResult:
    """Run jest via npx."""
    cmd = ["npx", "jest", "--no-coverage", "--bail"]
    if target:
        cmd.append(target)
    return _run_tests(cmd, cwd, "jest", repeat=repeat)


def _run_vitest(cwd: Path, target: str, repeat: int = 1) -> ToolResult:
    """Run vitest via npx."""
    cmd = ["npx", "vitest", "run", "--reporter=verbose"]
    if target:
        cmd.append(target)
    return _run_tests(cmd, cwd, "vitest", repeat=repeat)


def _run_go_test(cwd: Path, target: str, repeat: int = 1) -> ToolResult:
    """Run go test."""
    cmd = ["go", "test", "-v"]
    if target:
        cmd.append(target)
    else:
        cmd.append("./...")
    return _run_tests(cmd, cwd, "go", repeat=repeat)


def _run_cargo_test(cwd: Path, target: str, repeat: int = 1) -> ToolResult:
    """Run cargo test."""
    cmd = ["cargo", "test"]
    if target:
        cmd.extend(["--", target])
    return _run_tests(cmd, cwd, "cargo", repeat=repeat)


_RUNNERS: dict[str, Any] = {
    "pytest": _run_pytest,
    "jest": _run_jest,
    "vitest": _run_vitest,
    "go": _run_go_test,
    "cargo": _run_cargo_test,
}
