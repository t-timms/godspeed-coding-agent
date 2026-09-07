"""Tests for output styles: built-ins, custom loader, /style command, system-prompt swap."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.output_styles import (
    BUILT_IN_STYLES,
    STYLE_NAME_MAX_CHARS,
    STYLE_NAME_RE,
    load_custom_styles,
    resolve_style,
)
from godspeed.tui import output as _output
from godspeed.tui.commands import Commands

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(fn, *args, **kwargs) -> str:
    """Run a function and capture its Rich console output (ANSI stripped)."""
    buf = StringIO()
    original = _output.console
    _output.console = _output.Console(file=buf, force_terminal=True, width=120)
    try:
        fn(*args, **kwargs)
    finally:
        _output.console = original
    return _ANSI_RE.sub("", buf.getvalue())


@pytest.fixture
def conversation() -> Conversation:
    return Conversation("You are a coding agent.", max_tokens=100_000)


@pytest.fixture
def commands(conversation: Conversation, tmp_path: Path) -> Commands:
    llm_client = MagicMock()
    llm_client.model = "test-model"
    llm_client.fallback_models = []
    llm_client.total_input_tokens = 0
    llm_client.total_output_tokens = 0
    return Commands(
        conversation=conversation,
        llm_client=llm_client,
        permission_engine=None,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
        tool_registry=None,
    )


class TestBuiltInStyles:
    def test_default_has_no_suffix(self) -> None:
        assert BUILT_IN_STYLES["default"] is None

    def test_explanatory_and_learning_have_suffixes(self) -> None:
        assert BUILT_IN_STYLES["explanatory"]
        assert BUILT_IN_STYLES["learning"]
        assert BUILT_IN_STYLES["explanatory"] != BUILT_IN_STYLES["learning"]


class TestStyleNameValidation:
    def test_valid_names_match(self) -> None:
        assert STYLE_NAME_RE.match("default")
        assert STYLE_NAME_RE.match("explanatory")
        assert STYLE_NAME_RE.match("my-style")

    def test_invalid_names_rejected(self) -> None:
        assert not STYLE_NAME_RE.match("Bad Name")
        assert not STYLE_NAME_RE.match("UPPER")
        assert not STYLE_NAME_RE.match("with space")
        assert not STYLE_NAME_RE.match("under_score")

    def test_max_chars(self) -> None:
        assert STYLE_NAME_MAX_CHARS == 32
        assert len("a" * STYLE_NAME_MAX_CHARS) <= STYLE_NAME_MAX_CHARS


class TestLoadCustomStyles:
    def test_no_styles_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_custom_styles(tmp_path) == {}

    def test_loads_valid_files(self, tmp_path: Path) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "concise.md").write_text("Be concise.", encoding="utf-8")
        (styles_dir / "senior.md").write_text("Think like a senior engineer.", encoding="utf-8")
        assert load_custom_styles(tmp_path) == {
            "concise": "Be concise.",
            "senior": "Think like a senior engineer.",
        }

    def test_skips_invalid_names(self, tmp_path: Path) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "Bad Name.md").write_text("x", encoding="utf-8")
        (styles_dir / "UPPER.md").write_text("x", encoding="utf-8")
        (styles_dir / f"{'a' * (STYLE_NAME_MAX_CHARS + 1)}.md").write_text("x", encoding="utf-8")
        (styles_dir / "good.md").write_text("ok", encoding="utf-8")
        assert load_custom_styles(tmp_path) == {"good": "ok"}

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "empty.md").write_text("   \n", encoding="utf-8")
        assert load_custom_styles(tmp_path) == {}

    def test_ignores_non_markdown(self, tmp_path: Path) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "notes.txt").write_text("not a style", encoding="utf-8")
        assert load_custom_styles(tmp_path) == {}


class TestResolveStyle:
    def test_default_resolves_to_none(self, tmp_path: Path) -> None:
        assert resolve_style("default", tmp_path) is None

    def test_builtin_resolves(self, tmp_path: Path) -> None:
        assert resolve_style("explanatory", tmp_path) == BUILT_IN_STYLES["explanatory"]

    def test_custom_resolves(self, tmp_path: Path) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "concise.md").write_text("Be concise.", encoding="utf-8")
        assert resolve_style("concise", tmp_path) == "Be concise."

    def test_builtin_wins_over_custom(self, tmp_path: Path) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "explanatory.md").write_text("custom override", encoding="utf-8")
        assert resolve_style("explanatory", tmp_path) == BUILT_IN_STYLES["explanatory"]

    def test_unknown_returns_none(self, tmp_path: Path) -> None:
        assert resolve_style("bogus", tmp_path) is None


class TestSetSystemPrompt:
    def test_replaces_system_message(self, conversation: Conversation) -> None:
        conversation.set_system_prompt("New prompt.")
        assert conversation.messages[0] == {"role": "system", "content": "New prompt."}

    def test_invalidates_caches(self, conversation: Conversation) -> None:
        conversation.set_system_prompt("New prompt.")
        assert conversation.messages[0]["content"] == "New prompt."


class TestStyleCommand:
    def test_no_args_shows_current_and_available(self, commands: Commands) -> None:
        out = _capture(commands.dispatch, "/style")
        assert "Output Style" in out
        assert "Current" in out
        assert "default" in out
        assert "explanatory" in out
        assert "learning" in out

    def test_applies_builtin_style(self, commands: Commands, conversation: Conversation) -> None:
        result = commands.dispatch("/style explanatory")
        assert result is not None and result.handled
        assert conversation.messages[0]["content"] == (
            "You are a coding agent.\n\n" + BUILT_IN_STYLES["explanatory"]
        )
        assert commands._current_style == "explanatory"

    def test_persists_style_to_settings(self, commands: Commands, tmp_path: Path) -> None:
        commands.dispatch("/style explanatory")
        settings_path = tmp_path / ".godspeed" / "settings.yaml"
        assert settings_path.exists()
        assert "output_style: explanatory" in settings_path.read_text(encoding="utf-8")

    def test_default_restores_base_prompt(
        self, commands: Commands, conversation: Conversation
    ) -> None:
        commands.dispatch("/style explanatory")
        commands.dispatch("/style default")
        assert conversation.messages[0]["content"] == "You are a coding agent."
        assert commands._current_style == "default"

    def test_unknown_style_errors(self, commands: Commands, conversation: Conversation) -> None:
        out = _capture(commands.dispatch, "/style bogus")
        assert "Unknown style: bogus" in out
        assert conversation.messages[0]["content"] == "You are a coding agent."

    def test_applies_custom_style(
        self, commands: Commands, conversation: Conversation, tmp_path: Path
    ) -> None:
        styles_dir = tmp_path / ".godspeed" / "styles"
        styles_dir.mkdir(parents=True)
        (styles_dir / "concise.md").write_text("Be concise.", encoding="utf-8")
        commands.dispatch("/style concise")
        assert conversation.messages[0]["content"] == "You are a coding agent.\n\nBe concise."

    def test_persistence_failure_degrades_to_session_only(
        self, commands: Commands, conversation: Conversation
    ) -> None:
        with patch("godspeed.tui.commands.set_output_style", return_value=None):
            out = _capture(commands.dispatch, "/style explanatory")
        assert "for this session only" in out
        assert conversation.messages[0]["content"] == (
            "You are a coding agent.\n\n" + BUILT_IN_STYLES["explanatory"]
        )
        assert commands._current_style == "explanatory"
