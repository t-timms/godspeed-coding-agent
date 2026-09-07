"""Tests for godspeed.tools.runtime_verify."""

from __future__ import annotations

import sys
from pathlib import Path


from godspeed.tools.runtime_verify import (
    BUILD_TIMEOUT_SECONDS,
    LAUNCH_TIMEOUT_SECONDS,
    PROBE_WINDOW_SECONDS,
    MAX_EVIDENCE_LINES,
    MAX_OUTPUT_CHARS,
    RuntimeVerifier,
    Verdict,
    detect_build_command,
    detect_launch_command,
    _truncate,
)


# --- Verdict dataclass tests ------------------------------------------------


class TestVerdict:
    def test_passed_all_true(self) -> None:
        v = Verdict(build_ok=True, launch_ok=True, alive_after_probe=True)
        assert v.passed is True

    def test_passed_build_fail(self) -> None:
        v = Verdict(build_ok=False, launch_ok=True, alive_after_probe=True)
        assert v.passed is False

    def test_passed_launch_fail(self) -> None:
        v = Verdict(build_ok=True, launch_ok=False, alive_after_probe=True)
        assert v.passed is False

    def test_passed_alive_fail(self) -> None:
        v = Verdict(build_ok=True, launch_ok=True, alive_after_probe=False)
        assert v.passed is False

    def test_add_evidence_respects_cap(self) -> None:
        v = Verdict(build_ok=True, launch_ok=True, alive_after_probe=True)
        for i in range(MAX_EVIDENCE_LINES + 5):
            v.add_evidence(f"line {i}")
        assert len(v.evidence) == MAX_EVIDENCE_LINES

    def test_add_evidence_preserves_order(self) -> None:
        v = Verdict(build_ok=True, launch_ok=True, alive_after_probe=True)
        v.add_evidence("first")
        v.add_evidence("second")
        assert v.evidence == ["first", "second"]


# --- Constants tests --------------------------------------------------------


class TestConstants:
    def test_build_timeout_is_positive(self) -> None:
        assert BUILD_TIMEOUT_SECONDS > 0

    def test_launch_timeout_is_positive(self) -> None:
        assert LAUNCH_TIMEOUT_SECONDS > 0

    def test_probe_window_is_positive(self) -> None:
        assert PROBE_WINDOW_SECONDS > 0

    def test_max_evidence_lines_is_positive(self) -> None:
        assert MAX_EVIDENCE_LINES > 0

    def test_max_output_chars_is_positive(self) -> None:
        assert MAX_OUTPUT_CHARS > 0


# --- _truncate tests -------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello", limit=10) == "hello"

    def test_long_text_truncated(self) -> None:
        result = _truncate("a" * 100, limit=50)
        assert len(result) < 100
        assert "truncated" in result

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 50
        assert _truncate(text, limit=50) == text


# --- Detect build command tests (real fixtures) -----------------------------


class TestDetectBuildCommand:
    def test_python_project_returns_compileall(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        cmd = detect_build_command(tmp_path)
        assert cmd is not None
        assert "compileall" in cmd

    def test_python_setup_py_returns_compileall(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text("# setup", encoding="utf-8")
        cmd = detect_build_command(tmp_path)
        assert cmd is not None
        assert "compileall" in cmd

    def test_no_project_returns_none(self, tmp_path: Path) -> None:
        cmd = detect_build_command(tmp_path)
        assert cmd is None


# --- Detect launch command tests (real fixtures) ----------------------------


class TestDetectLaunchCommand:
    def test_python_project_with_main(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
        cmd = detect_launch_command(tmp_path)
        assert cmd is not None
        assert "main.py" in " ".join(cmd)

    def test_python_project_with_app(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
        cmd = detect_launch_command(tmp_path)
        assert cmd is not None
        assert "app.py" in " ".join(cmd)

    def test_python_project_no_entrypoint(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        cmd = detect_launch_command(tmp_path)
        assert cmd is None

    def test_no_project_returns_none(self, tmp_path: Path) -> None:
        cmd = detect_launch_command(tmp_path)
        assert cmd is None


# --- RuntimeVerifier tests (real subprocesses, no mocking) -------------------


class TestRuntimeVerifier:
    def test_build_pass_python_project(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
        verifier = RuntimeVerifier(tmp_path)
        verdict = verifier.verify()
        assert verdict.build_ok is True
        assert any("Build PASSED" in e for e in verdict.evidence)

    def test_build_fail_syntax_error(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        (tmp_path / "bad.py").write_text("def \n", encoding="utf-8")
        verifier = RuntimeVerifier(tmp_path)
        verdict = verifier.verify()
        assert verdict.build_ok is False
        assert verdict.build_ok is False

    def test_build_timeout_returns_structured_fail(self, tmp_path: Path) -> None:
        """Overriding build_command with a slow command and patching timeout."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        slow_cmd = [sys.executable, "-c", "import time; time.sleep(100)"]
        verifier = RuntimeVerifier(tmp_path, build_command=slow_cmd)
        import godspeed.tools.runtime_verify as rv_mod

        old_timeout = rv_mod.BUILD_TIMEOUT_SECONDS
        try:
            rv_mod.BUILD_TIMEOUT_SECONDS = 1
            verdict = verifier.verify()
        finally:
            rv_mod.BUILD_TIMEOUT_SECONDS = old_timeout

        assert verdict.build_ok is False
        assert any("TIMED OUT" in e or "timed out" in e.lower() for e in verdict.evidence)

    def test_launch_crash_exits_immediately(self, tmp_path: Path) -> None:
        crasher = tmp_path / "crasher.py"
        crasher.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, str(crasher)],
        )
        verdict = verifier.verify()
        assert verdict.build_ok is True
        assert verdict.launch_ok is False
        assert any("exited immediately" in e for e in verdict.evidence)

    def test_healthy_process_alive_after_probe(self, tmp_path: Path) -> None:
        sleeper = tmp_path / "sleeper.py"
        sleeper.write_text("import time; time.sleep(60)\n", encoding="utf-8")
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, str(sleeper)],
            probe_window=2,
        )
        verdict = verifier.verify()
        assert verdict.build_ok is True
        assert verdict.launch_ok is True
        assert verdict.alive_after_probe is True
        assert verdict.passed is True
        assert any("Probe OK" in e for e in verdict.evidence)
        verifier._kill()

    def test_evidence_messages_collected(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
        verifier = RuntimeVerifier(tmp_path)
        verdict = verifier.verify()
        assert len(verdict.evidence) > 0
        for line in verdict.evidence:
            assert isinstance(line, str)
            assert len(line) > 0

    def test_no_build_no_launch_returns_evidence(self, tmp_path: Path) -> None:
        verifier = RuntimeVerifier(tmp_path)
        verdict = verifier.verify()
        assert verdict.build_ok is True
        assert verdict.launch_ok is False
        assert any("No build step" in e for e in verdict.evidence)
        assert any("No launch command" in e for e in verdict.evidence)

    def test_launch_survives_probe_window(self, tmp_path: Path) -> None:
        sleeper = tmp_path / "sleeper.py"
        sleeper.write_text("import time; time.sleep(30)\n", encoding="utf-8")
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, str(sleeper)],
            probe_window=2,
        )
        verdict = verifier.verify()
        assert verdict.launch_ok is True
        assert any("alive" in e.lower() for e in verdict.evidence)
        verifier._kill()

    def test_kill_cleans_up_process(self, tmp_path: Path) -> None:
        sleeper = tmp_path / "sleeper.py"
        sleeper.write_text("import time; time.sleep(60)\n", encoding="utf-8")
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, str(sleeper)],
            probe_window=1,
        )
        verdict = verifier.verify()
        assert verdict.passed is True
        assert verifier._proc is None

    def test_http_probe_extracted_from_launch_args(self, tmp_path: Path) -> None:
        """A URL in the launch command args becomes the HTTP probe target."""
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, "server.py", "http://127.0.0.1:8000"],
        )
        assert verifier._resolved_http_probe() == "http://127.0.0.1:8000"

    def test_http_probe_explicit_overrides_launch_args(self, tmp_path: Path) -> None:
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, "server.py", "http://127.0.0.1:8000"],
            http_probe="http://127.0.0.1:9000",
        )
        assert verifier._resolved_http_probe() == "http://127.0.0.1:9000"

    def test_http_probe_none_when_no_url(self, tmp_path: Path) -> None:
        verifier = RuntimeVerifier(
            tmp_path,
            build_command=[sys.executable, "-c", "pass"],
            launch_command=[sys.executable, "server.py"],
        )
        assert verifier._resolved_http_probe() is None
