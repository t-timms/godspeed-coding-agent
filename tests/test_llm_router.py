"""Tests for task-aware model routing — classifier + config shortcuts."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from godspeed.config import GodspeedSettings
from godspeed.llm.client import ChatResponse, LLMClient, ModelRouter
from godspeed.llm.router import (
    TASK_ARCHITECT,
    TASK_COMPACTION,
    TASK_EDIT,
    TASK_PLAN,
    TASK_READ,
    TASK_SHELL,
    TASK_TYPES,
    classify_task_type,
)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Prevent GodspeedSettings from loading the user's real global config."""
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_path)


def _assistant(*tool_names: str) -> dict[str, object]:
    """Build an assistant message with the given tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for i, name in enumerate(tool_names)
        ],
    }


def _user(text: str = "do the thing") -> dict[str, object]:
    return {"role": "user", "content": text}


def _tool_result(call_id: str = "call_0", content: str = "ok") -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestClassifyTaskType:
    """The rule-based classifier that picks plan / edit / read / shell."""

    def test_empty_conversation_is_plan(self) -> None:
        assert classify_task_type([]) == TASK_PLAN

    def test_only_user_message_is_plan(self) -> None:
        # Fresh user input, no assistant turn yet — model needs to reason
        # about what to do next.
        assert classify_task_type([_user("add a feature")]) == TASK_PLAN

    def test_assistant_text_only_is_plan(self) -> None:
        # Model previously stopped (text-only response). Next turn is
        # another fresh planning step.
        msgs = [
            _user("hi"),
            {"role": "assistant", "content": "Hello! How can I help?"},
            _user("now do the thing"),
        ]
        assert classify_task_type(msgs) == TASK_PLAN

    def test_after_file_edit_is_edit(self) -> None:
        msgs = [_user(), _assistant("file_edit"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_after_file_write_is_edit(self) -> None:
        msgs = [_user(), _assistant("file_write"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_after_diff_apply_is_edit(self) -> None:
        msgs = [_user(), _assistant("diff_apply"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_after_shell_is_shell(self) -> None:
        msgs = [_user(), _assistant("shell"), _tool_result()]
        assert classify_task_type(msgs) == TASK_SHELL

    def test_after_test_runner_is_shell(self) -> None:
        msgs = [_user(), _assistant("test_runner"), _tool_result()]
        assert classify_task_type(msgs) == TASK_SHELL

    def test_after_only_reads_is_read(self) -> None:
        msgs = [
            _user(),
            _assistant("file_read", "grep_search", "glob_search"),
            _tool_result(),
        ]
        assert classify_task_type(msgs) == TASK_READ

    def test_edit_wins_over_read_in_same_batch(self) -> None:
        # Highest-stakes tool wins — an edit-and-also-read batch is
        # still an edit-phase continuation.
        msgs = [_user(), _assistant("file_read", "file_edit"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_shell_wins_over_read_in_same_batch(self) -> None:
        msgs = [_user(), _assistant("file_read", "shell"), _tool_result()]
        assert classify_task_type(msgs) == TASK_SHELL

    def test_edit_wins_over_shell_in_same_batch(self) -> None:
        msgs = [_user(), _assistant("shell", "file_edit"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_unknown_tool_falls_back_to_plan(self) -> None:
        # Unclassified tool (e.g. an MCP tool we don't know about) —
        # safer to keep the strong model than silently downgrade.
        msgs = [_user(), _assistant("some_mcp_tool"), _tool_result()]
        assert classify_task_type(msgs) == TASK_PLAN

    def test_uses_most_recent_assistant_turn(self) -> None:
        # Should look at the LAST assistant turn, not the first.
        msgs = [
            _user("first"),
            _assistant("file_edit"),
            _tool_result(),
            _user("now read"),
            _assistant("file_read"),
            _tool_result(),
        ]
        assert classify_task_type(msgs) == TASK_READ

    def test_malformed_tool_call_is_ignored(self) -> None:
        # Defensive: a tool_call entry without a usable function.name
        # shouldn't crash the classifier or be treated as an edit.
        msgs = [
            _user(),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "x", "function": {"arguments": "{}"}}],
            },
        ]
        # No usable tool names → treated as text-only assistant turn → plan.
        assert classify_task_type(msgs) == TASK_PLAN

    def test_non_assistant_message_returns_empty_list(self) -> None:
        from godspeed.llm.router import _extract_tool_names

        result = _extract_tool_names({"role": "user", "content": "hello"})
        assert result == []

    def test_non_dict_tool_calls_ignored(self) -> None:
        from godspeed.llm.router import _extract_tool_names

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                "not_a_dict",
                {"id": "x", "function": {"name": "file_read", "arguments": "{}"}},
            ],
        }
        result = _extract_tool_names(msg)
        assert result == ["file_read"]

    def test_function_not_a_dict(self) -> None:
        from godspeed.llm.router import _extract_tool_names

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": "not_a_dict"},
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": "{}"},
                },
            ],
        }
        result = _extract_tool_names(msg)
        assert result == ["file_read"]

    def test_function_name_not_a_string(self) -> None:
        from godspeed.llm.router import _extract_tool_names

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": 123, "arguments": "{}"}},
                {"id": "b", "function": {"name": "good_tool", "arguments": "{}"}},
            ],
        }
        result = _extract_tool_names(msg)
        assert result == ["good_tool"]

    def test_tool_call_with_empty_name(self) -> None:
        from godspeed.llm.router import _extract_tool_names

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "", "arguments": "{}"}},
                {"id": "b", "function": {"name": "valid_tool", "arguments": "{}"}},
            ],
        }
        result = _extract_tool_names(msg)
        assert result == ["valid_tool"]

    def test_message_with_no_tool_calls(self) -> None:
        from godspeed.llm.router import _extract_tool_names

        result = _extract_tool_names({"role": "assistant", "content": "just text"})
        assert result == []

    def test_notebook_edit_triggers_edit(self) -> None:
        msgs = [_user(), _assistant("notebook_edit"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_generate_tests_triggers_edit(self) -> None:
        msgs = [_user(), _assistant("generate_tests"), _tool_result()]
        assert classify_task_type(msgs) == TASK_EDIT

    def test_git_triggers_shell(self) -> None:
        msgs = [_user(), _assistant("git"), _tool_result()]
        assert classify_task_type(msgs) == TASK_SHELL

    def test_github_triggers_shell(self) -> None:
        msgs = [_user(), _assistant("github"), _tool_result()]
        assert classify_task_type(msgs) == TASK_SHELL

    def test_web_search_is_read(self) -> None:
        msgs = [_user(), _assistant("web_search"), _tool_result()]
        assert classify_task_type(msgs) == TASK_READ

    def test_no_assistant_turn_is_plan(self) -> None:
        msgs = [_user("first"), _tool_result(), _user("second")]
        assert classify_task_type(msgs) == TASK_PLAN

    def test_canonical_task_types_constant(self) -> None:
        # The TASK_TYPES tuple is the canonical surface area exposed
        # to settings YAML / docs — pin it so accidental drift fails
        # loudly in review.
        assert TASK_PLAN in TASK_TYPES
        assert TASK_EDIT in TASK_TYPES
        assert TASK_READ in TASK_TYPES
        assert TASK_SHELL in TASK_TYPES
        assert TASK_COMPACTION in TASK_TYPES
        assert TASK_ARCHITECT in TASK_TYPES


class TestSettingsAutoRouting:
    """The cheap_model / strong_model / architect_model shortcuts auto-fill
    routing[<task_type>] without users having to learn the dict syntax."""

    def test_no_shortcuts_leaves_routing_empty(self) -> None:
        s = GodspeedSettings()
        assert s.routing == {}

    def test_cheap_model_populates_three_task_types(self) -> None:
        s = GodspeedSettings(cheap_model="ollama/qwen3:4b")
        assert s.routing["edit"] == "ollama/qwen3:4b"
        assert s.routing["read"] == "ollama/qwen3:4b"
        assert s.routing["shell"] == "ollama/qwen3:4b"
        # Strong-model task type isn't populated by cheap shortcut.
        assert "plan" not in s.routing

    def test_strong_model_populates_plan(self) -> None:
        s = GodspeedSettings(strong_model="claude-sonnet-4")
        assert s.routing["plan"] == "claude-sonnet-4"
        assert "edit" not in s.routing

    def test_architect_model_populates_architect(self) -> None:
        s = GodspeedSettings(architect_model="claude-opus-4")
        assert s.routing["architect"] == "claude-opus-4"

    def test_combined_shortcuts_populate_all_tiers(self) -> None:
        s = GodspeedSettings(
            cheap_model="ollama/qwen3:4b",
            strong_model="claude-sonnet-4",
            architect_model="claude-opus-4",
        )
        assert s.routing == {
            "edit": "ollama/qwen3:4b",
            "read": "ollama/qwen3:4b",
            "shell": "ollama/qwen3:4b",
            "plan": "claude-sonnet-4",
            "architect": "claude-opus-4",
        }

    def test_explicit_routing_wins_over_cheap_shortcut(self) -> None:
        # User wrote `routing.edit: gpt-4o` — must override `cheap_model`.
        s = GodspeedSettings(
            cheap_model="ollama/qwen3:4b",
            routing={"edit": "gpt-4o"},
        )
        assert s.routing["edit"] == "gpt-4o"
        # Other cheap-tier task types still get the shortcut.
        assert s.routing["read"] == "ollama/qwen3:4b"
        assert s.routing["shell"] == "ollama/qwen3:4b"

    def test_explicit_routing_wins_over_strong_shortcut(self) -> None:
        s = GodspeedSettings(
            strong_model="claude-sonnet-4",
            routing={"plan": "gpt-4o"},
        )
        assert s.routing["plan"] == "gpt-4o"

    def test_empty_string_shortcuts_are_ignored(self) -> None:
        # Default field value is "" — must NOT populate routing with
        # an empty model string.
        s = GodspeedSettings(cheap_model="", strong_model="", architect_model="")
        assert s.routing == {}


class TestRoutingEndToEnd:
    """The classifier + ModelRouter cooperate so chat() picks the right model."""

    @pytest.mark.asyncio
    async def test_classifier_routes_edit_phase_to_cheap_model(self) -> None:
        # Settings with a cheap model populated for edit/read/shell.
        s = GodspeedSettings(
            model="claude-sonnet-4",
            cheap_model="ollama/qwen3:4b",
        )
        router = ModelRouter(routing=s.routing)
        client = LLMClient(model=s.model, router=router)

        # Simulate the loop: classify against an "edit just happened" state.
        msgs = [_user(), _assistant("file_edit"), _tool_result()]
        task_type = classify_task_type(msgs)
        assert task_type == TASK_EDIT

        mock_fallback = AsyncMock(
            return_value=ChatResponse(content="ok", finish_reason="stop"),
        )
        client._chat_with_fallback = mock_fallback
        await client.chat(messages=msgs, task_type=task_type)

        # The resolved model is passed as _model kwarg, never mutating self.
        call_kwargs = mock_fallback.call_args
        assert call_kwargs.kwargs.get("_model") == "ollama/qwen3:4b"
        assert client.model == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_classifier_routes_plan_to_strong_model(self) -> None:
        s = GodspeedSettings(
            model="ollama/qwen3:4b",
            strong_model="claude-sonnet-4",
        )
        router = ModelRouter(routing=s.routing)
        client = LLMClient(model=s.model, router=router)

        # Fresh user input → plan task type → strong model.
        msgs = [_user("add a new feature")]
        task_type = classify_task_type(msgs)
        assert task_type == TASK_PLAN

        mock_fallback = AsyncMock(
            return_value=ChatResponse(content="plan", finish_reason="stop"),
        )
        client._chat_with_fallback = mock_fallback
        await client.chat(messages=msgs, task_type=task_type)

        call_kwargs = mock_fallback.call_args
        assert call_kwargs.kwargs.get("_model") == "claude-sonnet-4"
        assert client.model == "ollama/qwen3:4b"
