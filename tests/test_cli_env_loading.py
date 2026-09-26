"""Tests for the CLI's .env / .env.local auto-loading on startup."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed._bootstrap import _load_env_files, _parse_env_file


class TestParseEnvFile:
    """Parser for ``KEY=value`` env files."""

    def test_parses_basic_key_value(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=bar\nBAZ=qux\n")
        result = _parse_env_file(env)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_skips_comments(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("# this is a comment\nFOO=bar\n# another\nBAZ=qux\n")
        assert _parse_env_file(env) == {"FOO": "bar", "BAZ": "qux"}

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("\n\nFOO=bar\n\n\nBAZ=qux\n\n")
        assert _parse_env_file(env) == {"FOO": "bar", "BAZ": "qux"}

    def test_strips_double_quotes(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text('FOO="value with spaces"\n')
        assert _parse_env_file(env) == {"FOO": "value with spaces"}

    def test_strips_single_quotes(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO='value with spaces'\n")
        assert _parse_env_file(env) == {"FOO": "value with spaces"}

    def test_leaves_mismatched_quotes_alone(self, tmp_path: Path) -> None:
        # ``"foo'`` (mismatched) — don't strip.
        env = tmp_path / ".env"
        env.write_text("FOO=\"foo'\n")
        assert _parse_env_file(env) == {"FOO": "\"foo'"}

    def test_preserves_equals_in_value(self, tmp_path: Path) -> None:
        # partition() splits on the FIRST = so key=a=b=c gives value="a=b=c".
        env = tmp_path / ".env"
        env.write_text("NVIDIA_NIM_API_KEY=nvapi-abc=def=ghi\n")
        assert _parse_env_file(env) == {"NVIDIA_NIM_API_KEY": "nvapi-abc=def=ghi"}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=bar\nthis line has no equals sign\n=value_with_no_key\nBAZ=qux\n")
        assert _parse_env_file(env) == {"FOO": "bar", "BAZ": "qux"}

    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert _parse_env_file(tmp_path / "does_not_exist") == {}

    def test_trims_whitespace_around_key_and_value(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("  FOO  =  bar  \n")
        assert _parse_env_file(env) == {"FOO": "bar"}


class TestLoadEnvFiles:
    """End-to-end loader that injects parsed values into os.environ."""

    def _clean_env(self, keys: list[str]) -> None:
        for k in keys:
            os.environ.pop(k, None)

    def test_loads_global_env_local(self, tmp_path: Path) -> None:
        # Simulate ~/.godspeed/.env.local by pointing DEFAULT_GLOBAL_DIR.
        global_dir = tmp_path / "home_godspeed"
        global_dir.mkdir()
        (global_dir / ".env.local").write_text("NVIDIA_NIM_API_KEY=nvapi-from-global\n")

        self._clean_env(["NVIDIA_NIM_API_KEY"])
        try:
            with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
                loaded = _load_env_files(project_dir=None)
            assert os.environ["NVIDIA_NIM_API_KEY"] == "nvapi-from-global"
            # One file loaded, one key injected.
            assert len(loaded) == 1
            assert loaded[0][1] == ["NVIDIA_NIM_API_KEY"]
        finally:
            self._clean_env(["NVIDIA_NIM_API_KEY"])

    def test_shell_env_wins_over_file(self, tmp_path: Path) -> None:
        global_dir = tmp_path / "home_godspeed"
        global_dir.mkdir()
        (global_dir / ".env.local").write_text("NVIDIA_NIM_API_KEY=from-file\n")

        self._clean_env(["NVIDIA_NIM_API_KEY"])
        os.environ["NVIDIA_NIM_API_KEY"] = "from-shell"
        try:
            with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
                _load_env_files(project_dir=None)
            # Shell value must still be present — file never overwrites it.
            assert os.environ["NVIDIA_NIM_API_KEY"] == "from-shell"
        finally:
            self._clean_env(["NVIDIA_NIM_API_KEY"])

    def test_project_env_overrides_global(self, tmp_path: Path) -> None:
        # Precedence: project/.env.local > global/.env.local for the
        # same key. Matches Vite/Next/dotenv convention.
        global_dir = tmp_path / "home_godspeed"
        global_dir.mkdir()
        (global_dir / ".env.local").write_text("SHARED_KEY=global-value\n")

        project_dir = tmp_path / "project"
        (project_dir / ".godspeed").mkdir(parents=True)
        (project_dir / ".godspeed" / ".env.local").write_text(
            "SHARED_KEY=project-value\nPROJECT_ONLY=x\n"
        )

        self._clean_env(["SHARED_KEY", "PROJECT_ONLY"])
        try:
            with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
                _load_env_files(project_dir=project_dir)
            # Project's value wins over global's for the same key.
            assert os.environ["SHARED_KEY"] == "project-value"
            # And keys only in project are injected too.
            assert os.environ["PROJECT_ONLY"] == "x"
        finally:
            self._clean_env(["SHARED_KEY", "PROJECT_ONLY"])

    def test_env_precedence_local_wins_over_env(self, tmp_path: Path) -> None:
        # Within the same directory: .env.local overrides .env.
        # Matches dotenv convention — .env.local is the gitignored override.
        global_dir = tmp_path / "home_godspeed"
        global_dir.mkdir()
        (global_dir / ".env").write_text("KEY_ONLY_IN_ENV=env-value\nSHARED=from-env\n")
        (global_dir / ".env.local").write_text("KEY_ONLY_IN_LOCAL=local-value\nSHARED=from-local\n")

        self._clean_env(["KEY_ONLY_IN_ENV", "KEY_ONLY_IN_LOCAL", "SHARED"])
        try:
            with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
                _load_env_files(project_dir=None)
            assert os.environ["KEY_ONLY_IN_ENV"] == "env-value"
            assert os.environ["KEY_ONLY_IN_LOCAL"] == "local-value"
            # .env.local wins for overlapping keys.
            assert os.environ["SHARED"] == "from-local"
        finally:
            self._clean_env(["KEY_ONLY_IN_ENV", "KEY_ONLY_IN_LOCAL", "SHARED"])

    def test_missing_global_dir_is_no_op(self, tmp_path: Path) -> None:
        # Fresh install with no ~/.godspeed yet — must not crash.
        global_dir = tmp_path / "does_not_exist"
        loaded = None
        with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
            loaded = _load_env_files(project_dir=None)
        assert loaded == []

    def test_malformed_file_does_not_raise(self, tmp_path: Path) -> None:
        # A binary-looking .env.local shouldn't crash startup.
        global_dir = tmp_path / "home_godspeed"
        global_dir.mkdir()
        (global_dir / ".env.local").write_bytes(b"\x00\x01\x02\xff\xfe")
        with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
            # Must not raise; caller gets either empty or whatever the
            # parser salvaged.
            _load_env_files(project_dir=None)


class TestLoadEnvFilesNoValueLeakInLogs:
    """Guardrail: values must never appear in the info-level log message."""

    def test_log_message_contains_key_not_value(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        global_dir = tmp_path / "home_godspeed"
        global_dir.mkdir()
        secret_value = "nvapi-super-secret-do-not-leak-me"
        (global_dir / ".env.local").write_text(f"NVIDIA_NIM_API_KEY={secret_value}\n")

        os.environ.pop("NVIDIA_NIM_API_KEY", None)
        try:
            import logging as _logging

            with (
                caplog.at_level(_logging.INFO, logger="godspeed._bootstrap"),
                patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir),
            ):
                _load_env_files(project_dir=None)

            full_log = caplog.text
            assert "NVIDIA_NIM_API_KEY" in full_log  # key logged
            assert secret_value not in full_log  # value NEVER logged
        finally:
            os.environ.pop("NVIDIA_NIM_API_KEY", None)


class TestProjectEnvCannotSetExecutionVariables:
    """A cloned repo's .godspeed/.env is untrusted; it may not change what code runs.

    The shell tool passes os.environ to every agent command and (on POSIX) resolves the shell
    from GODSPEED_SHELL_WRAPPER, so these keys would otherwise give the repo code execution
    outside the permission engine.
    """

    HOSTILE = {
        "BASH_ENV": "./.godspeed/x.sh",
        "LD_PRELOAD": "/nonexistent/x.so",
        "GODSPEED_SHELL_WRAPPER": "./.godspeed/wrapper.sh",
        "PROMPT_COMMAND": "./.godspeed/x.sh",
        "PYTHONPATH": "./.godspeed/py",
        "NODE_OPTIONS": "--require ./x.js",
        "GIT_SSH_COMMAND": "./x.sh",
    }

    @staticmethod
    def _isolate(monkeypatch: pytest.MonkeyPatch, keys: list[str]) -> None:
        # set-then-delete registers the original (absent) state for restoration at teardown
        for key in keys:
            monkeypatch.setenv(key, "placeholder")
            monkeypatch.delenv(key)

    def _project(self, tmp_path: Path, body: str, name: str = ".env") -> Path:
        project_dir = tmp_path / "project"
        (project_dir / ".godspeed").mkdir(parents=True)
        (project_dir / ".godspeed" / name).write_text(body)
        return project_dir

    @pytest.mark.parametrize("filename", [".env", ".env.local"])
    def test_exec_variables_from_a_project_file_are_dropped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        filename: str,
    ) -> None:
        global_dir = tmp_path / "g"
        global_dir.mkdir()
        keys = [*self.HOSTILE, "SAFE_API_KEY"]
        self._isolate(monkeypatch, keys)
        body = "".join(f"{k}={v}\n" for k, v in self.HOSTILE.items()) + "SAFE_API_KEY=sk-1\n"
        project_dir = self._project(tmp_path, body, filename)
        with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
            loaded = _load_env_files(project_dir=project_dir)
        for key in self.HOSTILE:
            assert key not in os.environ, key
        assert os.environ["SAFE_API_KEY"] == "sk-1"  # ordinary keys still load
        assert [k for _, ks in loaded for k in ks] == ["SAFE_API_KEY"]
        assert "BASH_ENV" in caplog.text  # the user is told which names were ignored
        assert "./.godspeed/x.sh" not in caplog.text  # ... but never the values

    def test_matching_is_case_insensitive_and_covers_loader_prefixes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_path / "g"
        global_dir.mkdir()
        keys = ["bash_env", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "BASH_FUNC_evil%%"]
        self._isolate(monkeypatch, keys)
        project_dir = self._project(tmp_path, "".join(f"{k}=x\n" for k in keys))
        with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
            _load_env_files(project_dir=project_dir)
        for key in keys:
            assert key not in os.environ, key

    def test_the_users_own_global_file_is_not_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_path / "g"
        global_dir.mkdir()
        (global_dir / ".env").write_text("PYTHONPATH=/my/libs\n")
        self._isolate(monkeypatch, ["PYTHONPATH"])
        with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
            _load_env_files(project_dir=None)
        assert os.environ["PYTHONPATH"] == "/my/libs"

    def test_a_project_file_cannot_override_the_users_global_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_path / "g"
        global_dir.mkdir()
        (global_dir / ".env").write_text("PYTHONPATH=/my/libs\n")
        self._isolate(monkeypatch, ["PYTHONPATH"])
        project_dir = self._project(tmp_path, "PYTHONPATH=/attacker\n")
        with patch("godspeed._bootstrap.DEFAULT_GLOBAL_DIR", global_dir):
            _load_env_files(project_dir=project_dir)
        assert os.environ["PYTHONPATH"] == "/my/libs"
