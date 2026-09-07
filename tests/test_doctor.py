"""Tests for the `godspeed doctor` diagnostic command."""

from __future__ import annotations

import os

import godspeed.cli as cli

# ─── Helpers ──────────────────────────────────────────────────────────


def _run_doctor(
    monkeypatch,
    tmp_path,
    *,
    ollama_running=True,
    env_vars=None,
    audit_writable=True,
    fix=False,
):
    """Patch out external deps and invoke the doctor command via Click's CliRunner."""
    from click.testing import CliRunner

    runner = CliRunner()

    monkeypatch.setattr(cli, "_is_ollama_running", lambda: ollama_running)

    if env_vars is not None:
        # Overlay test keys on the real environment: replacing os.environ
        # wholesale drops USERPROFILE, and any Path.home() on the doctor's
        # call path then raises "Could not determine home directory."
        monkeypatch.setattr(os, "environ", {**os.environ, **env_vars})

    if audit_writable:
        monkeypatch.setattr(cli, "DEFAULT_GLOBAL_DIR", tmp_path / ".godspeed")
    else:
        # Point to a read-only directory
        read_only = tmp_path / "readonly"
        read_only.mkdir()
        monkeypatch.setattr(cli, "DEFAULT_GLOBAL_DIR", read_only)
        if os.name != "nt":
            os.chmod(str(read_only), 0o500)

    # Patch out LiteLLM validation so tests don't need real keys
    def _fake_validate(*args, **kwargs):
        return None

    monkeypatch.setattr("litellm.validate_environment", _fake_validate)

    # Patch driver catalog to avoid file-system dependency
    mock_catalog = {
        "drivers": {
            "anthropic/claude-sonnet-4-6": {
                "provider": "anthropic",
                "requires_env": "ANTHROPIC_API_KEY",
            },
            "nvidia_nim/qwen3.5-397b-a17b": {
                "provider": "nvidia_nim",
                "requires_env": "NVIDIA_NIM_API_KEY",
            },
        }
    }

    import yaml

    monkeypatch.setattr(yaml, "safe_load", lambda text: mock_catalog)

    result = runner.invoke(cli.main, ["doctor"] + (["--fix"] if fix else []))
    return result


# ─── Ollama checks ───────────────────────────────────────────────────


class TestOllamaCheck:
    def test_ollama_running(self, monkeypatch, tmp_path):
        result = _run_doctor(monkeypatch, tmp_path, ollama_running=True, env_vars={})
        assert "Ollama server" in result.output
        assert "ok" in result.output

    def test_ollama_not_running_no_binary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_is_ollama_running", lambda: False)
        monkeypatch.setattr("shutil.which", lambda x: None)
        result = _run_doctor(monkeypatch, tmp_path, ollama_running=False, env_vars={})
        assert "Ollama server" in result.output
        assert "x" in result.output or "not installed" in result.output

    def test_ollama_not_running_binary_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_is_ollama_running", lambda: False)
        monkeypatch.setattr("shutil.which", lambda x: "C:\\ollama.exe")
        result = _run_doctor(monkeypatch, tmp_path, ollama_running=False, env_vars={})
        assert "Ollama server" in result.output
        assert "not running" in result.output


# ─── API key checks ──────────────────────────────────────────────────


class TestApiKeyCheck:
    def test_key_present(self, monkeypatch, tmp_path):
        env = {"ANTHROPIC_API_KEY": "sk-ant-test", "NVIDIA_NIM_API_KEY": "nvapi-test"}
        result = _run_doctor(monkeypatch, tmp_path, env_vars=env)
        assert "ANTHROPIC_API_KEY" in result.output
        assert "ok" in result.output

    def test_key_missing(self, monkeypatch, tmp_path):
        result = _run_doctor(monkeypatch, tmp_path, env_vars={})
        assert "ANTHROPIC_API_KEY" in result.output
        assert "!" in result.output or "not set" in result.output

    def test_partial_keys(self, monkeypatch, tmp_path):
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        result = _run_doctor(monkeypatch, tmp_path, env_vars=env)
        assert "ANTHROPIC_API_KEY" in result.output
        assert "NVIDIA_NIM_API_KEY" in result.output


# ─── Audit directory checks ───────────────────────────────────────────


class TestAuditDirCheck:
    def test_audit_dir_writable(self, monkeypatch, tmp_path):
        result = _run_doctor(monkeypatch, tmp_path, audit_writable=True, env_vars={})
        assert "Audit directory" in result.output
        assert "ok" in result.output or "writable" in result.output

    def test_audit_dir_not_writable(self, monkeypatch, tmp_path):
        # Mock Path.write_text to raise OSError for the probe file
        from pathlib import Path as PathCls

        original_write_text = PathCls.write_text

        def _mock_write_text(self, data, *args, **kwargs):
            if ".doctor_probe" in str(self):
                raise OSError("mocked: read-only filesystem")
            return original_write_text(self, data, *args, **kwargs)

        monkeypatch.setattr(PathCls, "write_text", _mock_write_text)
        result = _run_doctor(monkeypatch, tmp_path, audit_writable=True, env_vars={})
        assert "Audit directory" in result.output
        assert "x" in result.output or "not writable" in result.output

    def test_audit_dir_fix_flag(self, monkeypatch, tmp_path):
        result = _run_doctor(monkeypatch, tmp_path, audit_writable=False, env_vars={}, fix=True)
        assert "Audit directory" in result.output


# ─── Exit / output ───────────────────────────────────────────────────


class TestDoctorOutput:
    def test_all_ok_exit_zero(self, monkeypatch, tmp_path):
        env = {"ANTHROPIC_API_KEY": "sk-ant-test", "NVIDIA_NIM_API_KEY": "nvapi-test"}
        result = _run_doctor(monkeypatch, tmp_path, ollama_running=True, env_vars=env)
        assert result.exit_code == 0
        assert "All checks passed" in result.output

    def test_table_rendered(self, monkeypatch, tmp_path):
        result = _run_doctor(monkeypatch, tmp_path, env_vars={})
        assert "Check" in result.output
        assert "Status" in result.output
        assert "Detail" in result.output
