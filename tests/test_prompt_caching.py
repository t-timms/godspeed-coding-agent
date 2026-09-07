"""Tests for prompt caching support."""

from __future__ import annotations

from godspeed.llm.client import LLMClient


class TestPromptCaching:
    """Test _apply_prompt_caching static method."""

    def test_adds_cache_control_for_claude(self) -> None:
        """Cache all but the last 2 messages for Anthropic models."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        # system_and_3: up to 3 breakpoints on the last stable messages.
        # 4 total - 1 newest = 3 cached (indices 0, 1, 2)
        for i in range(3):
            assert isinstance(result[i]["content"], list)
            assert result[i]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Newest message is never cached (changes every turn)
        assert result[3]["content"] == "Second question"

    def test_short_conversation_no_cache_for_claude(self) -> None:
        """With 2 messages, only the stable system prefix gets a breakpoint."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        # System message (stable prefix) gets the single breakpoint
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Newest message unchanged
        assert result[1]["content"] == "Hello"

    def test_no_cache_for_openai(self) -> None:
        """OpenAI caches automatically — no explicit cache_control."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Query"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = LLMClient._apply_prompt_caching("gpt-4o", messages)
        # All messages should be unchanged
        assert result[0]["content"] == "System prompt"
        assert result[1]["content"] == "Query"
        assert result[2]["content"] == "Reply"

    def test_no_cache_for_ollama(self) -> None:
        messages = [{"role": "system", "content": "System prompt"}]
        result = LLMClient._apply_prompt_caching("ollama/qwen3:4b", messages)
        # Should be unchanged
        assert result[0]["content"] == "System prompt"

    def test_no_cache_for_unknown_model(self) -> None:
        messages = [{"role": "system", "content": "System prompt"}]
        result = LLMClient._apply_prompt_caching("some-random-model", messages)
        assert result[0]["content"] == "System prompt"

    def test_preserves_non_system_messages(self) -> None:
        """User and assistant messages also get cache_control (prefix caching)."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User input"},
            {"role": "assistant", "content": "Response"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        # system_and_3: 3 total - 1 newest = 2 cached (system + user)
        assert isinstance(result[0]["content"], list)
        assert "cache_control" in result[0]["content"][0]
        assert isinstance(result[1]["content"], list)
        assert "cache_control" in result[1]["content"][0]
        # Newest message unchanged
        assert result[2] == messages[2]

    def test_handles_already_structured_content(self) -> None:
        """System message with list content should not be double-wrapped."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Already structured"}]},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        # Should pass through unchanged since content is already a list
        assert result[0]["content"] == [{"type": "text", "text": "Already structured"}]

    def test_empty_messages(self) -> None:
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", [])
        assert result == []
