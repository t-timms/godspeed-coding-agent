"""Regression tests for Qwen3.5+ chat-template failure modes.

Strict Qwen3.5+ templates (documented in Ternary-Bonsai-2-27B KNOWN_ISSUES and
visible in the Qwen3.8 chat_template.jinja) return HTTP 500/400 for:

* any ``system`` message that is not the single first message,
* a tool call whose ``arguments`` are empty / not a JSON object string.

and ``json.loads("")`` in the tool-call parser would drop a valid zero-argument
call as malformed. These tests pin the guards.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import _parse_tool_call
from godspeed.llm.client import LLMClient


def _system_indices(conv: Conversation) -> list[int]:
    return [i for i, m in enumerate(conv.messages) if m.get("role") == "system"]


class TestEmptyToolArgumentsAreNormalized:
    @pytest.mark.parametrize("empty", ["", "   ", "\n", None])
    def test_empty_arguments_become_empty_object(self, empty: str | None) -> None:
        conv = Conversation("sys")
        conv.add_assistant_message(
            tool_calls=[{"id": "c1", "function": {"name": "git_status", "arguments": empty}}]
        )
        assert conv.messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_valid_arguments_are_untouched(self) -> None:
        conv = Conversation("sys")
        conv.add_assistant_message(
            tool_calls=[{"id": "c1", "function": {"name": "file_read", "arguments": '{"a": 1}'}}]
        )
        assert conv.messages[-1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

    def test_malformed_arguments_are_left_for_the_parser_to_reject(self) -> None:
        conv = Conversation("sys")
        conv.add_assistant_message(
            tool_calls=[{"id": "c1", "function": {"name": "shell", "arguments": "broken{"}}]
        )
        assert conv.messages[-1]["tool_calls"][0]["function"]["arguments"] == "broken{"

    def test_caller_dicts_are_not_mutated(self) -> None:
        original = {"id": "c1", "function": {"name": "t", "arguments": ""}}
        conv = Conversation("sys")
        conv.add_assistant_message(tool_calls=[original])
        assert original["function"]["arguments"] == ""

    def test_multiple_calls_each_normalized(self) -> None:
        conv = Conversation("sys")
        conv.add_assistant_message(
            tool_calls=[
                {"id": "a", "function": {"name": "x", "arguments": ""}},
                {"id": "b", "function": {"name": "y", "arguments": '{"k": "v"}'}},
            ]
        )
        args = [tc["function"]["arguments"] for tc in conv.messages[-1]["tool_calls"]]
        assert args == ["{}", '{"k": "v"}']


class TestParseToolCallEmptyArguments:
    @pytest.mark.parametrize("empty", ["", "  ", None])
    def test_empty_arguments_parse_as_no_arguments(self, empty: str | None) -> None:
        tc = _parse_tool_call({"id": "c1", "function": {"name": "git_status", "arguments": empty}})
        assert tc is not None
        assert tc.arguments == {}

    def test_missing_arguments_key_still_works(self) -> None:
        tc = _parse_tool_call({"id": "c1", "function": {"name": "git_status"}})
        assert tc is not None
        assert tc.arguments == {}

    def test_malformed_json_is_still_rejected(self) -> None:
        assert (
            _parse_tool_call({"id": "c1", "function": {"name": "shell", "arguments": "{oops"}})
            is None
        )


class TestSystemMessageStaysFirstAndSingle:
    def test_bootstrap_on_empty_conversation_merges_into_system_prompt(self) -> None:
        conv = Conversation("base prompt")
        conv.add_system_message("[resumed: abc]\n\nPrevious session summary:\nfixed the bug")
        assert _system_indices(conv) == [0]
        content = conv.messages[0]["content"]
        assert content.startswith("base prompt")
        assert "[resumed: abc]" in content
        assert len(conv.messages) == 1  # no extra message was appended

    def test_merge_is_logged_like_a_system_prompt_change(self) -> None:
        logger = MagicMock()
        conv = Conversation("base", conversation_logger=logger)
        logger.reset_mock()
        conv.add_system_message("extra")
        logger.log_system.assert_called_once_with("base\n\nextra")

    def test_after_messages_exist_it_becomes_a_labelled_user_note(self) -> None:
        conv = Conversation("base")
        conv.add_user_message("hello")
        conv.add_assistant_message(content="hi")
        conv.add_system_message("late context")
        assert _system_indices(conv) == [0]
        last = conv.messages[-1]
        assert last["role"] == "user"
        assert "late context" in last["content"]
        assert last["content"].startswith("[system note]")

    def test_empty_base_prompt_is_replaced_not_prefixed(self) -> None:
        conv = Conversation("")
        conv.add_system_message("only context")
        assert conv.messages[0]["content"] == "only context"

    def test_token_and_message_caches_are_invalidated(self) -> None:
        conv = Conversation("base")
        before = conv.token_count
        conv.add_system_message("a fairly long piece of additional system context " * 5)
        assert conv.token_count > before


class TestEffortTiersNeverDeriveLow:
    @pytest.mark.parametrize("budget", [1, 100, 2048, 8192])
    def test_small_budgets_derive_medium(self, budget: int) -> None:
        client = LLMClient(model="openai/qwen3.8-27b", thinking_budget=budget)
        assert client._qwen_template_kwargs() == {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }

    @pytest.mark.parametrize("budget", [8193, 10_000, 100_000])
    def test_large_budgets_derive_xhigh(self, budget: int) -> None:
        client = LLMClient(model="openai/qwen3.8-27b", thinking_budget=budget)
        assert client._qwen_template_kwargs() == {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
        }

    def test_explicit_low_is_still_honoured(self) -> None:
        client = LLMClient(model="openai/qwen3.8-27b", reasoning_effort="low")
        assert client._qwen_template_kwargs() == {
            "enable_thinking": True,
            "reasoning_effort": "low",
        }

    def test_minimal_alias_maps_to_medium(self) -> None:
        client = LLMClient(model="openai/qwen3.8-27b", reasoning_effort="minimal")
        assert client._qwen_template_kwargs() == {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
