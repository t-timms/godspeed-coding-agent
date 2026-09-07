"""Tests for the VS Code extension scaffold (ide/vscode/).

Validates the extension manifest, source wiring, documentation coverage,
and dependency hygiene without requiring a JS toolchain.  Runs in CI
alongside the Python test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXTENSION_DIR = Path(__file__).resolve().parent.parent / "ide" / "vscode"
PACKAGE_JSON = EXTENSION_DIR / "package.json"
EXTENSION_TS = EXTENSION_DIR / "extension.ts"
README_MD = EXTENSION_DIR / "README.md"


# ---------------------------------------------------------------------------
# package.json integrity
# ---------------------------------------------------------------------------


class TestPackageJson:
    """package.json must be valid JSON with the declared structure."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))

    def test_valid_json(self) -> None:
        """package.json must be parseable JSON."""
        assert isinstance(self.data, dict)

    def test_has_engines_vscode(self) -> None:
        """engines.vscode must be declared for marketplace compatibility."""
        assert "engines" in self.data
        assert "vscode" in self.data["engines"]

    def test_declares_four_commands(self) -> None:
        """Exactly 4 commands must be registered in contributes.commands."""
        commands = self.data.get("contributes", {}).get("commands", [])
        assert len(commands) == 4

    def test_command_ids_match_expected(self) -> None:
        """Command IDs must match the canonical set."""
        expected = {
            "godspeed.runTask",
            "godspeed.explainSelection",
            "godspeed.reviewDiff",
            "godspeed.resume",
        }
        commands = self.data.get("contributes", {}).get("commands", [])
        actual = {cmd["command"] for cmd in commands}
        assert actual == expected

    def test_activation_events_cover_all_commands(self) -> None:
        """Every declared command must have a matching onCommand activation event."""
        commands = self.data.get("contributes", {}).get("commands", [])
        activation = self.data.get("activationEvents", [])
        for cmd in commands:
            event = f"onCommand:{cmd['command']}"
            assert event in activation, f"Missing activation event for {cmd['command']}"

    def test_has_settings(self) -> None:
        """Three configuration settings must be declared."""
        settings = self.data.get("contributes", {}).get("configuration", {}).get("properties", {})
        expected_keys = {"godspeed.executablePath", "godspeed.defaultTimeout", "godspeed.model"}
        assert set(settings.keys()) == expected_keys

    def test_no_runtime_dependencies(self) -> None:
        """dependencies must be empty — zero runtime deps."""
        deps = self.data.get("dependencies", {})
        assert deps == {}, f"Unexpected runtime dependencies: {deps}"

    def test_dev_dependencies_are_typescript_only(self) -> None:
        """devDependencies must contain only @types/vscode and typescript."""
        dev_deps = self.data.get("devDependencies", {})
        assert set(dev_deps.keys()) == {"@types/vscode", "typescript"}

    def test_main_points_to_out_extension_js(self) -> None:
        """main entry must point to compiled output."""
        assert self.data.get("main") == "./out/extension.js"


# ---------------------------------------------------------------------------
# extension.ts ↔ package.json wiring
# ---------------------------------------------------------------------------


class TestExtensionSource:
    """extension.ts must register every command declared in package.json."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.pkg = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
        self.ts_source = EXTENSION_TS.read_text(encoding="utf-8")

    def test_source_file_exists(self) -> None:
        """extension.ts must exist."""
        assert EXTENSION_TS.is_file()

    def test_all_command_ids_in_source(self) -> None:
        """Every command ID from package.json must appear in extension.ts."""
        commands = self.pkg.get("contributes", {}).get("commands", [])
        for cmd in commands:
            cmd_id = cmd["command"]
            assert cmd_id in self.ts_source, (
                f"Command '{cmd_id}' declared in package.json but not found in extension.ts"
            )

    def test_all_settings_in_source(self) -> None:
        """Every setting key from package.json must appear in extension.ts.

        The extension reads settings through a scoped configuration object
        (``getConfiguration("godspeed")``), so the ``godspeed.`` prefix never
        appears literally in source — check the setting name suffix instead.
        """
        settings = self.pkg.get("contributes", {}).get("configuration", {}).get("properties", {})
        for key in settings:
            suffix = key.split(".", 1)[1]
            assert suffix in self.ts_source, (
                f"Setting '{key}' declared in package.json but not found in extension.ts"
            )

    def test_source_has_activate_export(self) -> None:
        """extension.ts must export an activate function."""
        assert "export function activate" in self.ts_source

    def test_source_has_deactivate_export(self) -> None:
        """extension.ts must export a deactivate function."""
        assert "export function deactivate" in self.ts_source

    def test_source_uses_vscode_import(self) -> None:
        """extension.ts must import vscode module."""
        assert 'from "vscode"' in self.ts_source or "from 'vscode'" in self.ts_source

    def test_source_uses_child_process(self) -> None:
        """extension.ts must import child_process for spawning godspeed."""
        assert "child_process" in self.ts_source


# ---------------------------------------------------------------------------
# README coverage
# ---------------------------------------------------------------------------


class TestReadme:
    """README.md must document all commands and settings."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.pkg = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
        self.readme = README_MD.read_text(encoding="utf-8")

    def test_readme_exists(self) -> None:
        """README.md must exist."""
        assert README_MD.is_file()

    def test_all_commands_documented(self) -> None:
        """Every command title from package.json must appear in README."""
        commands = self.pkg.get("contributes", {}).get("commands", [])
        for cmd in commands:
            title = cmd["title"]
            assert title in self.readme, f"Command title '{title}' not documented in README.md"

    def test_all_settings_documented(self) -> None:
        """Every setting key must appear in README."""
        settings = self.pkg.get("contributes", {}).get("configuration", {}).get("properties", {})
        for key in settings:
            assert key in self.readme, f"Setting '{key}' not documented in README.md"

    def test_readme_mentions_install_from_source(self) -> None:
        """README must mention building from source."""
        assert "install from source" in self.readme.lower() or "from source" in self.readme.lower()


# ---------------------------------------------------------------------------
# File completeness
# ---------------------------------------------------------------------------


class TestFileCompleteness:
    """All required scaffold files must exist."""

    @pytest.mark.parametrize(
        "filename",
        ["package.json", "extension.ts", "tsconfig.json", "README.md", ".vscodeignore"],
    )
    def test_file_exists(self, filename: str) -> None:
        path = EXTENSION_DIR / filename
        assert path.is_file(), f"Missing required file: ide/vscode/{filename}"
