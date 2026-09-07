"""Tests for the `godspeed mcp` CLI commands (add/list/remove).

Covers the Claude Code `claude mcp` parity surface: adding stdio and SSE
servers into the persistent settings.yaml, duplicate handling with --force,
listing with scope, removal with unknown-name errors, and round-tripping
through the real GodspeedSettings loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from godspeed.cli import main
from godspeed.config import GodspeedSettings


def _patch_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_path / ".gs-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_path / ".godspeed")


def _project_settings(tmp_path: Path) -> Path:
    return tmp_path / ".godspeed" / "settings.yaml"


class TestMCPAdd:
    def test_add_stdio_writes_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--project-dir",
                str(tmp_path),
                "mcp",
                "add",
                "github",
                "npx",
                "-y",
                "@modelcontextprotocol/server-github",
            ],
        )
        assert result.exit_code == 0
        assert "Added MCP server 'github'" in result.output

        settings_path = _project_settings(tmp_path)
        assert settings_path.exists()
        data = yaml.safe_load(settings_path.read_text())
        assert data["mcp_servers"] == [
            {
                "name": "github",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            }
        ]

        # Round-trip through the real loader.
        s = GodspeedSettings(project_dir=tmp_path)
        assert len(s.mcp_servers) == 1
        assert s.mcp_servers[0]["name"] == "github"
        assert s.mcp_servers[0]["command"] == "npx"
        assert s.mcp_servers[0]["args"] == ["-y", "@modelcontextprotocol/server-github"]

    def test_add_with_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--project-dir",
                str(tmp_path),
                "mcp",
                "add",
                "srv",
                "cmd",
                "--env",
                "TOKEN=abc",
                "--env",
                "FOO=bar",
            ],
        )
        assert result.exit_code == 0
        data = yaml.safe_load(_project_settings(tmp_path).read_text())
        assert data["mcp_servers"][0]["env"] == {"TOKEN": "abc", "FOO": "bar"}

        s = GodspeedSettings(project_dir=tmp_path)
        assert s.mcp_servers[0]["env"] == {"TOKEN": "abc", "FOO": "bar"}

    def test_add_sse_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--project-dir",
                str(tmp_path),
                "mcp",
                "add",
                "remote",
                "https://example.com/mcp",
            ],
        )
        assert result.exit_code == 0
        data = yaml.safe_load(_project_settings(tmp_path).read_text())
        assert data["mcp_servers"] == [
            {"name": "remote", "transport": "sse", "url": "https://example.com/mcp"}
        ]

        s = GodspeedSettings(project_dir=tmp_path)
        assert s.mcp_servers[0]["transport"] == "sse"
        assert s.mcp_servers[0]["url"] == "https://example.com/mcp"

    def test_add_user_scope_writes_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "add", "--scope", "user", "srv", "echo"])
        assert result.exit_code == 0
        global_path = tmp_path / ".gs-global" / "settings.yaml"
        assert global_path.exists()
        data = yaml.safe_load(global_path.read_text())
        assert data["mcp_servers"][0]["name"] == "srv"

    def test_add_invalid_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--project-dir", str(tmp_path), "mcp", "add", "srv", "cmd", "--env", "NOEQUALS"],
        )
        assert result.exit_code != 0
        assert "expected KEY=VALUE" in result.output

    def test_add_duplicate_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        first = runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "add", "srv", "echo"])
        assert first.exit_code == 0
        second = runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "add", "srv", "echo"])
        assert second.exit_code != 0
        assert "already exists" in second.output

    def test_add_duplicate_force_replaces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "add", "srv", "echo", "old"])
        result = runner.invoke(
            main,
            ["--project-dir", str(tmp_path), "mcp", "add", "--force", "srv", "echo", "new"],
        )
        assert result.exit_code == 0
        data = yaml.safe_load(_project_settings(tmp_path).read_text())
        assert data["mcp_servers"] == [
            {"name": "srv", "transport": "stdio", "command": "echo", "args": ["new"]}
        ]

        s = GodspeedSettings(project_dir=tmp_path)
        assert s.mcp_servers[0]["args"] == ["new"]


class TestMCPList:
    def test_list_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "list"])
        assert result.exit_code == 0
        assert "No MCP servers configured" in result.output

    def test_list_shows_servers_and_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        runner.invoke(
            main,
            ["--project-dir", str(tmp_path), "mcp", "add", "github", "npx", "-y", "server"],
        )
        runner.invoke(
            main,
            [
                "--project-dir",
                str(tmp_path),
                "mcp",
                "add",
                "--scope",
                "user",
                "remote",
                "https://example.com/mcp",
            ],
        )
        result = runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "list"])
        assert result.exit_code == 0
        assert "github" in result.output
        assert "remote" in result.output
        assert "project" in result.output
        assert "user" in result.output
        assert "npx" in result.output
        assert "sse" in result.output


class TestMCPRemove:
    def test_remove_deletes_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "add", "srv", "echo"])
        result = runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "remove", "srv"])
        assert result.exit_code == 0
        assert "Removed MCP server 'srv'" in result.output

        data = yaml.safe_load(_project_settings(tmp_path).read_text())
        assert "mcp_servers" not in data or data["mcp_servers"] == []

        # Round-trip through the real loader.
        s = GodspeedSettings(project_dir=tmp_path)
        assert s.mcp_servers == []

    def test_remove_unknown_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        runner = CliRunner()
        result = runner.invoke(main, ["--project-dir", str(tmp_path), "mcp", "remove", "ghost"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestMCPConfigHelpers:
    """Direct tests for the config.py MCP helpers (edge cases)."""

    def test_append_mcp_server_requires_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        from godspeed.config import append_mcp_server

        with pytest.raises(ValueError, match="requires a 'name' key"):
            append_mcp_server({"command": "echo"}, project_dir=tmp_path)

    def test_append_mcp_server_malformed_yaml_rebuilt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        from godspeed.config import append_mcp_server

        settings_dir = tmp_path / ".godspeed"
        settings_dir.mkdir()
        (settings_dir / "settings.yaml").write_text(":: broken ::")
        result = append_mcp_server({"name": "srv", "command": "echo"}, project_dir=tmp_path)
        assert result is not None
        data = yaml.safe_load((settings_dir / "settings.yaml").read_text())
        assert data["mcp_servers"][0]["name"] == "srv"

    def test_remove_mcp_server_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dirs(tmp_path, monkeypatch)
        from godspeed.config import remove_mcp_server

        found, path = remove_mcp_server("ghost", project_dir=tmp_path)
        assert found is False
        assert path is None
