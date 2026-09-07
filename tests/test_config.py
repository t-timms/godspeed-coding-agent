"""Tests for configuration system."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from godspeed.config import (
    GodspeedSettings,
    PermissionSettings,
    _MAX_YAML_CACHE_SIZE,
    _load_yaml_cached,
    _merge_configs,
    _yaml_cache,
    append_allow_rule,
    append_permission_rule,
    get_model_context_window,
)


def _patch_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_path / ".gs-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")


class TestGodspeedSettings:
    """Test root configuration."""

    def test_defaults(self, settings: GodspeedSettings) -> None:
        assert settings.model == "openai/qwen2.5-coder-14b"
        assert settings.permission_mode == "normal"
        assert settings.max_context_tokens == 100_000
        assert settings.compaction_threshold == 0.8
        assert settings.audit.enabled is True

    def test_env_override(self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_project, monkeypatch)
        monkeypatch.setenv("GODSPEED_MODEL", "gpt-4o")
        monkeypatch.setenv("GODSPEED_PERMISSION_MODE", "strict")
        s = GodspeedSettings(project_dir=tmp_project)
        assert s.model == "gpt-4o"
        assert s.permission_mode == "strict"

    def test_yaml_global_config(self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        global_dir = tmp_project / ".gs-global"
        global_dir.mkdir()
        config = {"model": "ollama/llama3", "permission_mode": "strict"}
        (global_dir / "settings.yaml").write_text(yaml.dump(config))
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_project / ".godspeed")
        s = GodspeedSettings(project_dir=tmp_project)
        assert s.model == "ollama/llama3"
        assert s.permission_mode == "strict"

    def test_yaml_project_overrides_global(
        self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_project / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(yaml.dump({"model": "gpt-4o"}))
        project_dir = tmp_project / ".godspeed"
        project_dir.mkdir(exist_ok=True)
        (project_dir / "settings.yaml").write_text(yaml.dump({"model": "claude-sonnet-4-20250514"}))
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", project_dir)
        s = GodspeedSettings(project_dir=tmp_project)
        assert s.model == "claude-sonnet-4-20250514"

    # ── field-validator error paths ──

    def test_permission_mode_raises_on_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_PERMISSION_MODE", "unsafe")
        with pytest.raises(ValueError, match="permission_mode must be one of"):
            GodspeedSettings(project_dir=tmp_path)

    def test_model_raises_on_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            GodspeedSettings(model="", project_dir=tmp_path)

    def test_sandbox_raises_on_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_SANDBOX", "virtualbox")
        with pytest.raises(ValueError, match="sandbox must be one of"):
            GodspeedSettings(project_dir=tmp_path)

    def test_execution_mode_raises_on_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_EXECUTION_MODE", "autonomous")
        with pytest.raises(ValueError, match="execution_mode must be one of"):
            GodspeedSettings(project_dir=tmp_path)

    def test_auto_fix_retries_raises_on_negative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_AUTO_FIX_RETRIES", "-1")
        with pytest.raises(ValueError, match="auto_fix_retries must be >= 0"):
            GodspeedSettings(project_dir=tmp_path)

    def test_thinking_budget_raises_on_negative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_THINKING_BUDGET", "-5")
        with pytest.raises(ValueError, match="thinking_budget must be >= 0"):
            GodspeedSettings(project_dir=tmp_path)

    def test_max_cost_usd_raises_on_negative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_MAX_COST_USD", "-0.01")
        with pytest.raises(ValueError, match="max_cost_usd must be >= 0"):
            GodspeedSettings(project_dir=tmp_path)

    def test_auto_commit_threshold_raises_on_lt_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_AUTO_COMMIT_THRESHOLD", "0")
        with pytest.raises(ValueError, match="auto_commit_threshold must be >= 1"):
            GodspeedSettings(project_dir=tmp_path)

    def test_compaction_threshold_raises_on_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_COMPACTION_THRESHOLD", "0")
        with pytest.raises(ValueError, match="compaction_threshold must be between 0 and 1"):
            GodspeedSettings(project_dir=tmp_path)

    def test_max_context_tokens_raises_on_lt_1000(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        monkeypatch.setenv("GODSPEED_MAX_CONTEXT_TOKENS", "500")
        with pytest.raises(ValueError, match="max_context_tokens must be at least 1000"):
            GodspeedSettings(project_dir=tmp_path)

    # ── collection validation (hooks / mcp_servers / routing) ──

    def test_hooks_non_dict_entry_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        s = GodspeedSettings.model_construct(
            hooks=["not_a_dict"],
            project_dir=Path("."),
            global_dir=Path("."),
        )
        s.validate_collections()
        assert any("Skipping non-dict hook entry" in r.message for r in caplog.records)

    def test_hooks_missing_command_key_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(hooks=[{"command": "lint"}, {"no_command": "here"}], project_dir=tmp_path)
        assert any("missing required 'command' key" in r.message for r in caplog.records)

    def test_mcp_servers_non_dict_entry_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        s = GodspeedSettings.model_construct(
            mcp_servers=["not_a_dict"],
            project_dir=Path("."),
            global_dir=Path("."),
        )
        s.validate_collections()
        assert any("Skipping non-dict MCP server entry" in r.message for r in caplog.records)

    def test_mcp_servers_missing_name_key_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(mcp_servers=[{"name": "ok"}, {"no_name": "here"}], project_dir=tmp_path)
        assert any("missing required 'name' key" in r.message for r in caplog.records)

    def test_routing_empty_model_name_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(routing={"plan": "  "}, project_dir=tmp_path)
        assert any("Empty model name in routing" in r.message for r in caplog.records)

    # ── insecure-settings warnings ──

    def test_yolo_mode_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(permission_mode="yolo", project_dir=tmp_path)
        assert any("INSECURE: permission_mode='yolo'" in r.message for r in caplog.records)

    def test_empty_deny_list_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(permissions={"deny": []}, project_dir=tmp_path)
        assert any("No deny rules configured" in r.message for r in caplog.records)

    def test_audit_disabled_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(audit={"enabled": False}, project_dir=tmp_path)
        assert any(
            "INSECURE: audit.enabled=False disables the audit trail" in r.message
            for r in caplog.records
        )

    def test_wildcard_allow_rule_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(permissions={"allow": ["*"]}, project_dir=tmp_path)
        assert any("permits ALL tools" in r.message for r in caplog.records)

    def test_sandbox_none_with_yolo_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(permission_mode="yolo", sandbox="none", project_dir=tmp_path)
        assert any(
            "sandbox='none' with permission_mode='yolo'" in r.message for r in caplog.records
        )

    def test_high_context_tokens_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING)
        GodspeedSettings(max_context_tokens=600_000, project_dir=tmp_path)
        assert any(
            "Very high max_context_tokens=600000 may cause memory issues" in r.message
            for r in caplog.records
        )

    # ── routing shortcuts ──

    def test_cheap_model_populates_routing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(cheap_model="gpt-4o", project_dir=tmp_path)
        assert s.routing["edit"] == "gpt-4o"
        assert s.routing["read"] == "gpt-4o"
        assert s.routing["shell"] == "gpt-4o"

    def test_strong_model_populates_routing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(strong_model="claude-sonnet-4-20250514", project_dir=tmp_path)
        assert s.routing["plan"] == "claude-sonnet-4-20250514"

    def test_architect_model_populates_routing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(architect_model="gemini-2-flash", project_dir=tmp_path)
        assert s.routing["architect"] == "gemini-2-flash"

    def test_explicit_routing_wins_over_shortcuts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        s = GodspeedSettings(
            cheap_model="gpt-4o",
            routing={"edit": "claude-sonnet-4-20250514"},
            project_dir=tmp_path,
        )
        assert s.routing["edit"] == "claude-sonnet-4-20250514"
        assert s.routing["read"] == "gpt-4o"

    # ── YAML error-handling in load_yaml_configs ──

    def test_malformed_global_yaml_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        global_dir = tmp_path / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(":: malformed yaml ::")
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")
        caplog.set_level(logging.WARNING)
        s = GodspeedSettings(project_dir=tmp_path)
        assert s.model == "openai/qwen2.5-coder-14b"
        assert any("Malformed global settings.yaml" in r.message for r in caplog.records)

    def test_malformed_project_yaml_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        global_dir = tmp_path / ".gs-global"
        global_dir.mkdir()
        project_dir = tmp_path / ".godspeed"
        project_dir.mkdir(exist_ok=True)
        (project_dir / "settings.yaml").write_text("} broken yaml {")
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", project_dir)
        caplog.set_level(logging.WARNING)
        s = GodspeedSettings(project_dir=tmp_path)
        assert s.model == "openai/qwen2.5-coder-14b"
        assert any("Malformed project settings.yaml" in r.message for r in caplog.records)


class TestPermissionSettings:
    """Test permission defaults."""

    def test_default_deny_rules(self) -> None:
        p = PermissionSettings()
        assert "FileRead(.env)" in p.deny
        assert "FileRead(*.pem)" in p.deny

    def test_default_allow_rules(self) -> None:
        p = PermissionSettings()
        assert "shell(git *)" in p.allow
        assert "shell(pytest *)" in p.allow

    def test_default_ask_rules(self) -> None:
        p = PermissionSettings()
        assert "shell(*)" in p.ask


class TestMergeConfigs:
    """Test config merging logic."""

    def test_deny_rules_are_additive(self) -> None:
        base = {"permissions": {"deny": ["FileRead(.env)"]}}
        override = {"permissions": {"deny": ["FileRead(*.key)"]}}
        _merge_configs(base, override)
        assert "FileRead(.env)" in base["permissions"]["deny"]
        assert "FileRead(*.key)" in base["permissions"]["deny"]

    def test_project_cannot_remove_global_deny(self) -> None:
        base = {"permissions": {"deny": ["FileRead(.env)", "Bash(rm -rf *)"]}}
        override = {"permissions": {"deny": []}}
        _merge_configs(base, override)
        assert "FileRead(.env)" in base["permissions"]["deny"]
        assert "Bash(rm -rf *)" in base["permissions"]["deny"]

    def test_allow_rules_override(self) -> None:
        base = {"permissions": {"allow": ["Bash(git *)"]}}
        override = {"permissions": {"allow": ["Bash(npm *)"]}}
        _merge_configs(base, override)
        assert base["permissions"]["allow"] == ["Bash(npm *)"]

    def test_non_permission_keys_override(self) -> None:
        base = {"model": "gpt-4o"}
        override = {"model": "claude-sonnet-4-20250514"}
        _merge_configs(base, override)
        assert base["model"] == "claude-sonnet-4-20250514"

    def test_ask_rules_override(self) -> None:
        base = {"permissions": {"ask": ["Bash(echo *)"]}}
        override = {"permissions": {"ask": ["Bash(curl *)"]}}
        _merge_configs(base, override)
        assert base["permissions"]["ask"] == ["Bash(curl *)"]

    def test_nested_non_permissions_dict_merge(self) -> None:
        base: dict = {"audit": {"enabled": True, "retention_days": 30}}
        override: dict = {"audit": {"enabled": False}}
        _merge_configs(base, override)
        assert base["audit"]["enabled"] is False
        assert base["audit"]["retention_days"] == 30


class TestLoadYamlCached:
    """Test the _load_yaml_cached LRU cache behaviour."""

    def test_returns_none_for_missing_file(self) -> None:
        assert _load_yaml_cached(Path("/nonexistent/path.yaml")) is None

    def test_non_dict_root_becomes_empty_dict(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "list_config.yaml"
        cache_path.write_text("- item1\n- item2\n")
        result = _load_yaml_cached(cache_path)
        assert result == {}

    def test_lru_eviction(self, tmp_path: Path) -> None:
        _yaml_cache.clear()
        try:
            files = []
            for i in range(_MAX_YAML_CACHE_SIZE + 5):
                f = tmp_path / f"config_{i}.yaml"
                f.write_text(f"key: {i}\n")
                files.append(f)

            results = []
            for f in files:
                results.append(_load_yaml_cached(f))

            assert len(_yaml_cache) <= _MAX_YAML_CACHE_SIZE
            for i, r in enumerate(results):
                assert r == {"key": i}
        finally:
            _yaml_cache.clear()

    def test_mtime_cache_hit(self, tmp_path: Path) -> None:
        _yaml_cache.clear()
        try:
            f = tmp_path / "config.yaml"
            f.write_text("key: 42\n")
            r1 = _load_yaml_cached(f)
            r2 = _load_yaml_cached(f)
            assert r1 == r2 == {"key": 42}
            assert len(_yaml_cache) == 1
        finally:
            _yaml_cache.clear()

    def test_mtime_invalidation(self, tmp_path: Path) -> None:
        import time

        _yaml_cache.clear()
        try:
            f = tmp_path / "config.yaml"
            f.write_text("key: 1\n")
            r1 = _load_yaml_cached(f)
            assert r1 == {"key": 1}

            time.sleep(0.02)
            f.write_text("key: 2\n")
            r2 = _load_yaml_cached(f)
            assert r2 == {"key": 2}
            assert len(_yaml_cache) == 1
        finally:
            _yaml_cache.clear()

    def test_lru_promotes_on_hit(self, tmp_path: Path) -> None:
        _yaml_cache.clear()
        try:
            first = tmp_path / "first.yaml"
            first.write_text("key: first\n")
            _load_yaml_cached(first)

            for i in range(_MAX_YAML_CACHE_SIZE):
                f = tmp_path / f"fill_{i}.yaml"
                f.write_text(f"key: {i}\n")
                _load_yaml_cached(f)

            assert len(_yaml_cache) == _MAX_YAML_CACHE_SIZE
            # Accessing `first` should promote it to MRU
            _load_yaml_cached(first)

            # Add one more to force eviction — `first` should survive because it was promoted
            extra = tmp_path / "extra.yaml"
            extra.write_text("key: extra\n")
            _load_yaml_cached(extra)

            assert _yaml_cache.get(first) is not None
        finally:
            _yaml_cache.clear()


class TestGetModelContextWindow:
    """Test context window prefix matching."""

    def test_exact_match(self) -> None:
        assert get_model_context_window("claude-sonnet-4-20250514") == 200_000

    def test_prefix_match_gpt4(self) -> None:
        assert get_model_context_window("gpt-4-turbo") == 128_000

    def test_longest_prefix_wins(self) -> None:
        assert get_model_context_window("gpt-4o") == 128_000

    def test_case_insensitive(self) -> None:
        assert get_model_context_window("Claude-Sonnet") == 200_000

    def test_default_for_unknown_model(self) -> None:
        assert get_model_context_window("unknown-model-v42") == 32_768

    def test_ollama_model_match(self) -> None:
        assert get_model_context_window("ollama/qwen3.6") == 32_768


class TestAppendPermissionRule:
    """Test append_permission_rule edge cases."""

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError, match=r"action must be 'allow' \| 'deny' \| 'ask'"):
            append_permission_rule("Bash(git *)", "invalid")

    def test_creates_new_settings_file(self, tmp_path: Path) -> None:
        result = append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        assert result is not None
        settings_path = tmp_path / ".godspeed" / "settings.yaml"
        assert settings_path.exists()
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["shell(git *)"]

    def test_appends_to_existing_rule_list(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text(yaml.dump({"permissions": {"allow": ["shell(git *)"]}}))

        append_permission_rule("shell(pytest *)", "allow", project_dir=tmp_path)
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["shell(git *)", "shell(pytest *)"]

    def test_idempotent_duplicate_rule(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text(yaml.dump({"permissions": {"allow": ["shell(git *)"]}}))

        append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["shell(git *)"]

    def test_malformed_existing_yaml_rebuilt(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text(":: broken ::")
        caplog.set_level(logging.WARNING)
        result = append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        assert result is not None
        assert any("rebuilding" in r.message for r in caplog.records)
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["shell(git *)"]

    def test_non_dict_data_becomes_empty_dict(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text("- item1\n- item2\n")
        result = append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        assert result is not None
        data = yaml.safe_load(settings_path.read_text())
        assert isinstance(data, dict)
        assert data["permissions"]["allow"] == ["shell(git *)"]

    def test_non_dict_permissions_field(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text(yaml.dump({"permissions": "not_a_dict"}))
        result = append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        assert result is not None
        data = yaml.safe_load(settings_path.read_text())
        assert isinstance(data["permissions"], dict)
        assert data["permissions"]["allow"] == ["shell(git *)"]

    def test_corrupted_rule_list_rebuilt(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text(yaml.dump({"permissions": {"allow": "not_a_list"}}))
        caplog.set_level(logging.WARNING)
        result = append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        assert result is not None
        assert any("Corrupted rule list" in r.message for r in caplog.records)
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["shell(git *)"]

    def test_oserror_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _mock_dump(*args: object, **kwargs: object) -> None:
            raise OSError("Disk full")

        monkeypatch.setattr(yaml, "safe_dump", _mock_dump)
        caplog.set_level(logging.WARNING)
        result = append_permission_rule("shell(git *)", "allow", project_dir=tmp_path)
        assert result is None
        assert any("Failed to write allow rule" in r.message for r in caplog.records)

    def test_append_allow_rule_returns_true(self, tmp_path: Path) -> None:
        result = append_allow_rule("shell(git *)", project_dir=tmp_path)
        assert result is True

    def test_writes_to_global_when_no_project_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_path / ".gs-global"
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        result = append_permission_rule("Bash(git *)", "deny")
        assert result == global_dir / "settings.yaml"
        assert result.exists()

class TestSandboxReconcile:
    """sandbox field syncs into sandbox_settings.mode (single source of truth)."""

    def test_sandbox_docker_promotes_to_settings(self, tmp_path: Path) -> None:
        s = GodspeedSettings(project_dir=tmp_path, sandbox="docker")
        assert s.sandbox_settings.mode == "docker"

    def test_explicit_settings_mode_wins(self, tmp_path: Path) -> None:
        s = GodspeedSettings(
            project_dir=tmp_path,
            sandbox="none",
            sandbox_settings={"mode": "docker"},
        )
        assert s.sandbox_settings.mode == "docker"


class TestDenyMergeOrder:
    def test_deny_merge_preserves_order_and_dedupes(self) -> None:
        base: dict = {"permissions": {"deny": ["B", "A"]}}
        _merge_configs(base, {"permissions": {"deny": ["C", "A"]}})
        assert base["permissions"]["deny"] == ["B", "A", "C"]
