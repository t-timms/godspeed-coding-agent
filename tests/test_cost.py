"""Tests for cost estimation module."""

from __future__ import annotations

from godspeed.llm.cost import estimate_cost, format_cost


class TestEstimateCost:
    """Test cost estimation for various models."""

    def test_ollama_is_free(self) -> None:
        cost = estimate_cost("ollama/qwen3:4b", 1000, 500)
        assert cost == 0.0

    def test_ollama_chat_is_free(self) -> None:
        cost = estimate_cost("ollama_chat/llama3:8b", 5000, 2000)
        assert cost == 0.0

    def test_lm_studio_is_free(self) -> None:
        cost = estimate_cost("lm_studio/local-model", 10000, 5000)
        assert cost == 0.0

    def test_claude_sonnet_pricing(self) -> None:
        # 1M input tokens at $3/M + 1M output tokens at $15/M = $18
        cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert cost == 18.0

    def test_claude_opus_pricing(self) -> None:
        # 1M input at $15/M + 1M output at $75/M = $90
        cost = estimate_cost("claude-opus-4-20250514", 1_000_000, 1_000_000)
        assert cost == 90.0

    def test_gpt4o_pricing(self) -> None:
        # 1M input at $2.50/M + 1M output at $10/M = $12.50
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 12.5

    def test_unknown_model_is_free(self) -> None:
        cost = estimate_cost("some-unknown-model/v1", 1000, 500)
        assert cost == 0.0

    def test_provider_prefix_stripped(self) -> None:
        # anthropic/claude-sonnet should match claude-sonnet pricing
        cost = estimate_cost("anthropic/claude-sonnet-4-20250514", 1_000_000, 0)
        assert cost == 3.0

    def test_small_usage(self) -> None:
        # 1000 input tokens of claude-sonnet: $3/M * 0.001M = $0.003
        cost = estimate_cost("claude-sonnet-4-20250514", 1000, 0)
        assert abs(cost - 0.003) < 0.0001

    def test_zero_tokens(self) -> None:
        cost = estimate_cost("claude-sonnet-4-20250514", 0, 0)
        assert cost == 0.0

    def test_cached_tokens_discount(self) -> None:
        # 1M input: 600k cached at 10% + 400k normal = 0.6*0.3 + 0.4*3.0 = 0.18 + 1.2 = 1.38
        cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 0, cached_input_tokens=600_000)
        assert abs(cost - 1.38) < 0.001

    def test_cached_tokens_all_cached(self) -> None:
        # All 1M tokens cached: 1M * $3/M * 0.10 = $0.30
        cost = estimate_cost(
            "claude-sonnet-4-20250514", 1_000_000, 0, cached_input_tokens=1_000_000
        )
        assert abs(cost - 0.30) < 0.001

    def test_cached_tokens_capped_to_input(self) -> None:
        # cached > input should be clamped
        cost = estimate_cost("claude-sonnet-4-20250514", 500_000, 0, cached_input_tokens=900_000)
        cost_all_cached = estimate_cost(
            "claude-sonnet-4-20250514", 500_000, 0, cached_input_tokens=500_000
        )
        assert abs(cost - cost_all_cached) < 0.0001

    def test_cached_tokens_negative_clamped(self) -> None:
        cost = estimate_cost("claude-sonnet-4-20250514", 1000, 0, cached_input_tokens=-500)
        cost_no_cache = estimate_cost("claude-sonnet-4-20250514", 1000, 0)
        assert abs(cost - cost_no_cache) < 0.0001


class TestFormatCost:
    """Test cost formatting."""

    def test_free(self) -> None:
        assert format_cost(0.0) == "free"

    def test_small_cost(self) -> None:
        assert format_cost(0.003) == "$0.0030"

    def test_larger_cost(self) -> None:
        assert format_cost(1.50) == "$1.50"

    def test_very_small_cost(self) -> None:
        result = format_cost(0.0001)
        assert result.startswith("$")
