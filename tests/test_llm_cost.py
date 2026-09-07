"""Tests for LLM cost estimation and model selection."""

from __future__ import annotations

from godspeed.llm.cost import estimate_cost, format_cost, get_cheapest_model


class TestEstimateCost:
    def test_ollama_free(self) -> None:
        assert estimate_cost("ollama/qwen3:4b", 100_000, 50_000) == 0.0

    def test_ollama_chat_free(self) -> None:
        assert estimate_cost("ollama_chat/qwen3:4b", 100_000, 50_000) == 0.0

    def test_lm_studio_free(self) -> None:
        assert estimate_cost("lm_studio/local-model", 100_000, 50_000) == 0.0

    def test_llamacpp_free(self) -> None:
        assert estimate_cost("llamacpp/qwen3:4b", 100_000, 50_000) == 0.0

    def test_claude_sonnet(self) -> None:
        cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        # $3/M input + $15/M output = $18
        assert cost == 18.0

    def test_gpt_4o(self) -> None:
        cost = estimate_cost("gpt-4o-2024-08-06", 1_000_000, 1_000_000)
        # $2.50/M input + $10/M output = $12.50
        assert cost == 12.5

    def test_unknown_model_free(self) -> None:
        assert estimate_cost("some-unknown-model", 100_000, 50_000) == 0.0

    def test_with_provider_prefix(self) -> None:
        cost = estimate_cost("anthropic/claude-sonnet-4-20250514", 1_000_000, 0)
        assert cost == 3.0  # $3/M input

    def test_zero_tokens(self) -> None:
        assert estimate_cost("claude-sonnet", 0, 0) == 0.0

    def test_small_token_count(self) -> None:
        cost = estimate_cost("claude-sonnet", 1000, 500)
        assert cost > 0
        assert cost < 0.02  # ~$0.0105 for 1K input + 500 output

    def test_negative_input_tokens_clamped_to_zero(self) -> None:
        assert estimate_cost("claude-sonnet-4-20250514", -500, 1000) == estimate_cost(
            "claude-sonnet-4-20250514", 0, 1000
        )

    def test_negative_output_tokens_clamped_to_zero(self) -> None:
        assert estimate_cost("claude-sonnet-4-20250514", 1000, -500) == estimate_cost(
            "claude-sonnet-4-20250514", 1000, 0
        )

    def test_both_negative_tokens(self) -> None:
        assert estimate_cost("claude-sonnet-4-20250514", -100, -200) == 0.0

    def test_negative_tokens_on_unknown_model(self) -> None:
        assert estimate_cost("unknown-model", -1000, -500) == 0.0

    def test_deepseek_chat_pricing(self) -> None:
        cost = estimate_cost("deepseek-chat", 1_000_000, 1_000_000)
        assert cost == 1.37  # $0.27/M + $1.10/M = $1.37

    def test_deepseek_reasoner_pricing(self) -> None:
        cost = estimate_cost("deepseek-reasoner", 1_000_000, 0)
        assert cost == 0.55

    def test_deepseek_v4_pro_pricing(self) -> None:
        cost = estimate_cost("deepseek-v4-pro", 1_000_000, 0)
        assert cost == 1.74

    def test_gemini_25_pro_pricing(self) -> None:
        cost = estimate_cost("gemini-2.5-pro", 1_000_000, 1_000_000)
        assert cost == 11.25  # $1.25/M + $10/M = $11.25

    def test_gemini_20_flash_pricing(self) -> None:
        cost = estimate_cost("gemini-2.0-flash", 1_000_000, 0)
        assert cost == 0.10

    def test_mistral_large_pricing(self) -> None:
        cost = estimate_cost("mistral-large", 1_000_000, 1_000_000)
        assert cost == 8.0  # $2/M + $6/M = $8

    def test_mistral_small_pricing(self) -> None:
        cost = estimate_cost("mistral-small", 1_000_000, 0)
        assert cost == 0.10

    def test_claude_opus_pricing(self) -> None:
        cost = estimate_cost("claude-opus-4-20250514", 1_000_000, 1_000_000)
        assert cost == 90.0  # $15/M + $75/M = $90

    def test_claude_haiku_pricing(self) -> None:
        cost = estimate_cost("claude-haiku-3.5", 1_000_000, 0)
        assert cost == 0.80

    def test_gpt_4_pricing(self) -> None:
        cost = estimate_cost("gpt-4", 1_000_000, 1_000_000)
        assert cost == 90.0  # $30/M + $60/M = $90

    def test_o3_pricing(self) -> None:
        cost = estimate_cost("o3", 1_000_000, 1_000_000)
        assert cost == 50.0  # $10/M + $40/M = $50

    def test_o3_mini_pricing(self) -> None:
        cost = estimate_cost("o3-mini", 1_000_000, 0)
        assert cost == 1.10

    def test_best_match_prefix(self) -> None:
        assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == 0.15
        assert estimate_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15

    def test_very_large_token_counts(self) -> None:
        cost = estimate_cost("claude-sonnet-4-20250514", 50_000_000, 25_000_000)
        assert cost > 500  # Rough sanity check
        assert isinstance(cost, float)

    def test_gpt4_turbo_pricing(self) -> None:
        cost = estimate_cost("gpt-4-turbo", 1_000_000, 1_000_000)
        assert cost == 40.0  # $10/M + $30/M = $40

    def test_codex_pricing(self) -> None:
        cost = estimate_cost("codex-1", 1_000_000, 0)
        assert cost == 10.0


class TestGetCheapestModel:
    def test_empty_list(self) -> None:
        assert get_cheapest_model([]) == ""

    def test_ollama_is_cheapest(self) -> None:
        models = ["claude-sonnet-4-20250514", "ollama/qwen3:4b", "gpt-4o"]
        assert get_cheapest_model(models) == "ollama/qwen3:4b"

    def test_all_api_models(self) -> None:
        models = ["claude-opus-4-20250514", "claude-haiku-3.5", "gpt-4o"]
        result = get_cheapest_model(models)
        assert result == "claude-haiku-3.5"  # Cheapest API model

    def test_single_model(self) -> None:
        assert get_cheapest_model(["gpt-4o"]) == "gpt-4o"

    def test_all_ollama(self) -> None:
        models = ["ollama/qwen3:4b", "ollama/gemma3:12b"]
        # First free model wins
        assert get_cheapest_model(models) == "ollama/qwen3:4b"

    def test_llamacpp_is_cheapest(self) -> None:
        models = ["claude-sonnet-4-20250514", "llamacpp/local-model", "gpt-4o"]
        assert get_cheapest_model(models) == "llamacpp/local-model"

    def test_all_unknown_returns_first(self) -> None:
        models = ["unknown-a", "unknown-b", "unknown-c"]
        assert get_cheapest_model(models) == "unknown-a"

    def test_unknown_models_with_known(self) -> None:
        models = ["unknown-model", "claude-haiku-3.5", "gpt-4o"]
        assert get_cheapest_model(models) == "claude-haiku-3.5"

    def test_free_model_not_first(self) -> None:
        models = ["claude-sonnet-4-20250514", "gpt-4o", "ollama/qwen3:4b"]
        assert get_cheapest_model(models) == "ollama/qwen3:4b"

    def test_deepseek_vs_claude(self) -> None:
        models = ["claude-opus-4-20250514", "deepseek-chat"]
        assert get_cheapest_model(models) == "deepseek-chat"


class TestFormatCost:
    def test_free(self) -> None:
        assert format_cost(0.0) == "free"

    def test_small_cost(self) -> None:
        assert format_cost(0.0042) == "$0.0042"

    def test_normal_cost(self) -> None:
        assert format_cost(1.50) == "$1.50"

    def test_very_small_cost(self) -> None:
        assert format_cost(0.0001) == "$0.0001"

    def test_large_cost(self) -> None:
        assert format_cost(12345.67) == "$12345.67"

    def test_near_zero_but_not_exactly(self) -> None:
        result = format_cost(1e-12)
        assert result == "free"

    def test_just_below_cent(self) -> None:
        assert format_cost(0.0099) == "$0.0099"

    def test_just_above_cent(self) -> None:
        assert format_cost(0.01) == "$0.01"
