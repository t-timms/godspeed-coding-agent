"""Tests for typed config: StrEnum fields, schema version, unknown-key warnings."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from godspeed.config import (
    CONFIG_SCHEMA_VERSION,
    ExecutionMode,
    GodspeedSettings,
    PermissionMode,
    SandboxMode,
    SandboxSettings,
    _warn_unknown_keys,
)


def _patch_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_path / ".gs-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")


class TestTypedEnums:
    def test_permission_mode_is_streum(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(permission_mode="strict", project_dir=tmp_path)
        assert isinstance(s.permission_mode, PermissionMode)
        assert s.permission_mode == PermissionMode.STRICT
        assert s.permission_mode == "strict"

    def test_execution_mode_is_streum(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(execution_mode="codeact", project_dir=tmp_path)
        assert isinstance(s.execution_mode, ExecutionMode)
        assert s.execution_mode == ExecutionMode.CODEACT

    def test_sandbox_is_streum(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(sandbox="docker", project_dir=tmp_path)
        assert isinstance(s.sandbox, SandboxMode)
        assert s.sandbox == SandboxMode.DOCKER

    def test_defaults(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(project_dir=tmp_path)
        assert s.permission_mode == PermissionMode.NORMAL
        assert s.execution_mode == ExecutionMode.TOOL
        assert s.sandbox == SandboxMode.NONE


class TestSandboxSettings:
    def test_defaults(self) -> None:
        sb = SandboxSettings()
        assert sb.mode == SandboxMode.NONE
        assert sb.image == "python:3.12-slim"
        assert sb.timeout_seconds == 120
        assert sb.network_enabled is False

    def test_nested_in_godspeed_settings(self, tmp_path: Path, monkeypatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(
            sandbox_settings={"mode": "docker", "timeout_seconds": 300},
            project_dir=tmp_path,
        )
        assert s.sandbox_settings.mode == SandboxMode.DOCKER
        assert s.sandbox_settings.timeout_seconds == 300


class TestSchemaVersion:
    def test_schema_version_defined(self) -> None:
        assert isinstance(CONFIG_SCHEMA_VERSION, int)
        assert CONFIG_SCHEMA_VERSION >= 1


class TestWarnUnknownKeys:
    def test_unknown_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        _warn_unknown_keys({"model": "x", "typo_key": 1})
        assert any("Unknown settings key 'typo_key'" in r.message for r in caplog.records)

    def test_known_keys_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        _warn_unknown_keys({"model": "x", "sandbox": "none"})
        assert not any("Unknown settings key" in r.message for r in caplog.records)

    def test_yaml_unknown_key_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        global_dir = tmp_path / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(yaml.dump({"model": "x", "bogus": 1}))
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")
        caplog.set_level(logging.WARNING)
        GodspeedSettings(project_dir=tmp_path)
        assert any("Unknown settings key 'bogus'" in r.message for r in caplog.records)
