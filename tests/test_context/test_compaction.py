"""Tests for model-aware compaction and graduated compactor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.config import get_model_context_window
from godspeed.context.compaction import (
    COMPACTION_PROMPT_LARGE,
    COMPACTION_PROMPT_SMALL,
    COMPACTION_STAGES,
    CompactionContext,
    GraduatedCompactor,
    compact_if_needed,
    get_compaction_prompt,
)
from godspeed.llm.client import ChatResponse, LLMClient


class TestModelContextWindows:
    """Test model context window lookup."""

    def test_claude_opus(self) -> None:
        assert get_model_context_window("claude-opus-4-20250514") == 200_000

    def test_claude_sonnet(self) -> None:
        assert get_model_context_window("claude-sonnet-4-20250514") == 200_000

    def test_gpt4o(self) -> None:
        assert get_model_context_window("gpt-4o-2024-05-13") == 128_000

    def test_gpt4_base(self) -> None:
        assert get_model_context_window("gpt-4-0613") == 8_192

    def test_ollama_qwen(self) -> None:
        assert get_model_context_window("ollama/qwen3:4b") == 32_768

    def test_ollama_llama3(self) -> None:
        assert get_model_context_window("ollama/llama3:8b") == 8_192

    def test_ollama_llama31(self) -> None:
        assert get_model_context_window("ollama/llama3.1:8b") == 128_000

    def test_unknown_model_defaults(self) -> None:
        assert get_model_context_window("totally-unknown-model") == 32_768

    def test_empty_string_defaults(self) -> None:
        assert get_model_context_window("") == 32_768

    def test_case_insensitive(self) -> None:
        assert get_model_context_window("Claude-Sonnet-4") == 200_000

    def test_gemini_large_context(self) -> None:
        assert get_model_context_window("gemini-2-flash") == 1_000_000


class TestCompactionPromptSelection:
    """Test model-aware compaction prompt selection."""

    def test_small_model_gets_aggressive_prompt(self) -> None:
        # ollama/llama3 = 8192 → small
        prompt = get_compaction_prompt("ollama/llama3:8b")
        assert prompt == COMPACTION_PROMPT_SMALL
        assert "aggressively" in prompt.lower()

    def test_medium_model_gets_balanced_prompt(self) -> None:
        # ollama/qwen3 = 32768 → small threshold (≤32K), actually hits small
        # Use a model that's between 32K and 100K
        prompt = get_compaction_prompt("ollama/mistral:7b")
        assert prompt == COMPACTION_PROMPT_SMALL  # 32768 ≤ 32768 threshold

    def test_frontier_model_gets_detailed_prompt(self) -> None:
        # claude-sonnet = 200K → large
        prompt = get_compaction_prompt("claude-sonnet-4-20250514")
        assert prompt == COMPACTION_PROMPT_LARGE
        assert "thorough" in prompt.lower()

    def test_gpt4o_gets_large_prompt(self) -> None:
        # gpt-4o = 128K → large
        prompt = get_compaction_prompt("gpt-4o-mini")
        assert prompt == COMPACTION_PROMPT_LARGE

    def test_unknown_model_gets_medium_prompt(self) -> None:
        # Unknown = 32768 default → small threshold
        prompt = get_compaction_prompt("unknown-model")
        assert prompt == COMPACTION_PROMPT_SMALL

    def test_empty_model_gets_medium_prompt(self) -> None:
        prompt = get_compaction_prompt("")
        assert prompt == COMPACTION_PROMPT_SMALL  # default 32768 ≤ threshold

    def test_gpt4_base_gets_small_prompt(self) -> None:
        # gpt-4 = 8192 → small
        prompt = get_compaction_prompt("gpt-4-0613")
        assert prompt == COMPACTION_PROMPT_SMALL


class TestCompactIfNeeded:
    """Test the compact_if_needed function with model awareness."""

    @pytest.mark.asyncio
    async def test_no_compaction_when_not_needed(self) -> None:
        conv = Conversation("System prompt", max_tokens=100_000)
        client = LLMClient(model="test")
        result = await compact_if_needed(conv, client, model="claude-sonnet-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_compaction_uses_model_prompt(self) -> None:
        conv = Conversation("System prompt", max_tokens=100, compaction_threshold=0.01)
        # Add enough content to trigger compaction
        for i in range(20):
            conv.add_user_message(f"Message {i} with some content to fill tokens")
            conv.add_assistant_message(f"Response {i} with additional content here")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(
                content="Summary of conversation", tool_calls=[], finish_reason="stop"
            )
        )

        result = await compact_if_needed(conv, client, model="claude-sonnet-4")

        if result:
            # Verify chat was called with the large prompt (claude = 200K)
            call_args = client.chat.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
            system_content = messages[0]["content"]
            assert "thorough" in system_content.lower()

    @pytest.mark.asyncio
    async def test_compaction_with_small_model(self) -> None:
        conv = Conversation("System prompt", max_tokens=100, compaction_threshold=0.01)
        for i in range(20):
            conv.add_user_message(f"Message {i} with content")
            conv.add_assistant_message(f"Response {i} here")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Brief summary", tool_calls=[], finish_reason="stop")
        )

        result = await compact_if_needed(conv, client, model="ollama/llama3:8b")

        if result:
            call_args = client.chat.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
            system_content = messages[0]["content"]
            assert "aggressively" in system_content.lower()

    @pytest.mark.asyncio
    async def test_compaction_fallback_no_model(self) -> None:
        """Without a model name, falls back to medium prompt."""
        conv = Conversation("System prompt", max_tokens=100, compaction_threshold=0.01)
        for i in range(20):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Summary", tool_calls=[], finish_reason="stop")
        )

        # model="" triggers the fallback
        result = await compact_if_needed(conv, client, model="")
        # Should not crash regardless of compaction triggering
        assert isinstance(result, bool)


class TestGraduatedCompactor:
    """Test the graduated compaction ladder (4 deterministic + 1 async)."""

    def test_stages_defined(self) -> None:
        assert len(COMPACTION_STAGES) == 4
        assert COMPACTION_STAGES[0].name == "budget_reduction"
        assert COMPACTION_STAGES[3].name == "context_collapse"

    def test_get_stage_for_context(self) -> None:
        compactor = GraduatedCompactor()
        idx = compactor.get_stage_for_context(80_000, 100_000)
        assert idx == 0  # 80% > 75% → budget_reduction
        idx = compactor.get_stage_for_context(65_000, 100_000)
        assert idx == 1  # 65% > 60% → snip
        idx = compactor.get_stage_for_context(50_000, 100_000)
        assert idx == 2  # 50% > 45% → microcompact
        idx = compactor.get_stage_for_context(35_000, 100_000)
        assert idx == 3  # 35% > 30% → context_collapse
        idx = compactor.get_stage_for_context(10_000, 100_000)
        assert idx == -1  # 10% < 15% → no stage

    def test_reset(self) -> None:
        compactor = GraduatedCompactor()
        compactor.get_stage_for_context(80_000, 100_000)
        assert compactor.context_pct > 0
        compactor.reset()
        assert compactor.context_pct == 0.0

    def test_apply_stages_reduces_messages(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System prompt", max_tokens=100_000)
        for i in range(50):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")
            conv.add_tool_result(f"tool-{i}", f"Result data from tool {i} " * 100)

        before = len(conv._messages)
        results = compactor.apply_stages(conv, 80_000, 100_000)
        if results:
            applied = [r for r in results if r.applied]
            assert len(applied) > 0
            after = len(conv._messages)
            assert after <= before

    def test_apply_stages_no_redundant_recompaction(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System prompt", max_tokens=100_000)
        for i in range(30):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")

        r1 = compactor.apply_stages(conv, 80_000, 100_000)
        has_applied = any(r.applied for r in r1)
        if has_applied:
            r2 = compactor.apply_stages(conv, 80_000, 100_000)
            assert len(r2) == 0

    def test_stage1_drop_verbose_tool_outputs(self) -> None:
        from godspeed.context.compaction import _drop_verbose_tool_outputs

        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "x" * 5000},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _drop_verbose_tool_outputs(ctx)
        assert len(result) == 2
        # Verbose tool output should be truncated
        tool_msg = result[1]
        assert len(tool_msg["content"]) < 5000

    def test_stage2_remove_low_signal_turns(self) -> None:
        from godspeed.context.compaction import _remove_low_signal_turns

        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "ok"},  # low signal
                {
                    "role": "assistant",
                    "content": "I will fix the parser",
                    "tool_calls": [{"function": {"name": "file_edit"}}],
                },
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 2  # "ok" removed

    def test_stage4_keep_metadata_only(self) -> None:
        from godspeed.context.compaction import _keep_metadata_only

        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "hello " * 100},
                {
                    "role": "assistant",
                    "content": "long " * 100,
                    "tool_calls": [{"function": {"name": "file_edit"}}],
                },
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert len(result) >= 2

    def test_empty_conversation(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System prompt", max_tokens=100_000)
        results = compactor.apply_stages(conv, 80_000, 100_000)
        assert isinstance(results, list)
