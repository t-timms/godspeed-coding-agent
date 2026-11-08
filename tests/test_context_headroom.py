"""Context headroom on hard windows: calibration, reply reserve, truncated tool calls.

Observed on a local llama-server with a 32K window: Godspeed compacted at an estimate of ~27K
tokens while the server already held more than the window, the reply was cut off in the middle of a
tool call, and llama-server answered HTTP 500 "Failed to parse tool call arguments" on every retry.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import (
    _check_context_and_compact,
    _is_truncated_tool_call,
    _observe_reported_prompt_tokens,
)
from godspeed.config import GodspeedSettings


def _conv(max_tokens: int = 1000, reserve: int = 0, threshold: float = 0.8) -> Conversation:
    return Conversation(
        system_prompt="sys",
        model="openai/qwen3.8-27b",
        max_tokens=max_tokens,
        compaction_threshold=threshold,
        completion_reserve_tokens=reserve,
    )


class TestCalibration:
    def test_gap_becomes_fixed_overhead(self) -> None:
        conv = _conv(max_tokens=100_000)
        conv.add_user_message("hello " * 2000)
        raw = conv.raw_token_count
        conv.observe_prompt_tokens(reported=raw + 700, estimated_at_call=raw)
        assert conv.token_count == raw + 700

    def test_overhead_survives_growth_and_is_not_multiplied(self) -> None:
        conv = _conv(max_tokens=100_000)
        conv.add_user_message("hello " * 2000)
        raw = conv.raw_token_count
        conv.observe_prompt_tokens(reported=raw + 500, estimated_at_call=raw)
        conv.add_user_message("more " * 400)
        assert conv.token_count == conv.raw_token_count + 500

    def test_over_counting_estimate_clears_overhead(self) -> None:
        conv = _conv(max_tokens=100_000)
        conv.add_user_message("hello " * 200)
        raw = conv.raw_token_count
        conv.observe_prompt_tokens(reported=raw + 500, estimated_at_call=raw)
        conv.observe_prompt_tokens(reported=raw - 20, estimated_at_call=raw)
        assert conv.token_count == raw

    def test_reported_size_far_above_the_window_is_ignored(self) -> None:
        conv = _conv(max_tokens=1_000)
        conv.add_user_message("hello " * 20)
        raw = conv.raw_token_count
        conv.observe_prompt_tokens(reported=2_500, estimated_at_call=raw)
        assert conv.token_count == raw

    def test_large_early_overhead_is_kept(self) -> None:
        """Tool schemas can dwarf a tiny first estimate; that overhead is real and constant."""
        conv = _conv(max_tokens=32_768)
        conv.add_user_message("fix the bug")
        raw = conv.raw_token_count
        conv.observe_prompt_tokens(reported=raw + 7_000, estimated_at_call=raw)
        assert conv.token_count == raw + 7_000

    def test_zero_or_missing_values_are_ignored(self) -> None:
        conv = _conv()
        conv.observe_prompt_tokens(reported=0, estimated_at_call=100)
        conv.observe_prompt_tokens(reported=100, estimated_at_call=0)
        assert conv.token_count == conv.raw_token_count

    def test_overhead_is_capped_at_half_the_window(self) -> None:
        conv = _conv(max_tokens=1000)
        conv.add_user_message("hi")
        raw = conv.raw_token_count
        conv.observe_prompt_tokens(reported=raw * 3, estimated_at_call=raw)
        assert conv.token_count - raw <= 500

    def test_loop_helper_reads_both_usage_shapes(self) -> None:
        for usage in ({"input_tokens": 3_000}, {"prompt_tokens": 3_000}):
            conv = _conv(max_tokens=100_000)
            conv.add_user_message("hello " * 1000)
            raw = conv.raw_token_count
            assert raw < 3_000
            _observe_reported_prompt_tokens(conv, SimpleNamespace(usage=usage), raw)  # type: ignore[arg-type]
            assert conv.token_count == 3_000

    def test_loop_helper_tolerates_missing_usage(self) -> None:
        conv = _conv()
        _observe_reported_prompt_tokens(conv, SimpleNamespace(usage={}), 10)  # type: ignore[arg-type]
        _observe_reported_prompt_tokens(conv, SimpleNamespace(usage=None), 10)  # type: ignore[arg-type]
        assert conv.token_count == conv.raw_token_count


class TestReplyReserve:
    def test_reserve_lowers_the_compaction_point(self) -> None:
        plain = _conv(max_tokens=10_000, reserve=0)
        reserved = _conv(max_tokens=10_000, reserve=2_000)
        assert plain.usable_tokens == 10_000
        assert reserved.usable_tokens == 8_000

    def test_reserve_is_capped_at_a_quarter_of_the_window(self) -> None:
        conv = _conv(max_tokens=1_000, reserve=900)
        assert conv.usable_tokens == 750

    def test_is_near_limit_uses_the_usable_budget(self) -> None:
        text = "word " * 620  # between 0.6 and 0.8 of a 1000-token window
        plain = _conv(max_tokens=1_000, reserve=0, threshold=0.8)
        plain.add_user_message(text)
        reserved = _conv(max_tokens=1_000, reserve=250, threshold=0.8)
        reserved.add_user_message(text)
        assert not plain.is_near_limit
        assert reserved.is_near_limit

    @pytest.mark.asyncio
    async def test_compaction_fires_earlier_with_a_reserve(self) -> None:
        text = "word " * 620  # between 0.6 and 0.8 of a 1000-token window
        for reserve, expect_compaction in ((0, False), (250, True)):
            conv = _conv(max_tokens=1_000, reserve=reserve, threshold=0.8)
            conv.add_user_message(text)
            frac = conv.token_count / 1_000
            assert 0.6 < frac < 0.8, frac  # sanity: the fixture sits between the two limits
            with patch("godspeed.agent.loop._compact_conversation", new=AsyncMock()) as compact:
                await _check_context_and_compact(conv, MagicMock(), None, None, MagicMock())
            assert compact.called is expect_compaction, reserve


class TestTruncatedToolCall:
    ERR = Exception(
        "litellm.InternalServerError: OpenAIException - Failed to parse tool call arguments as "
        "JSON: [json.exception.parse_error.101] parse error ... missing closing quote; last read: '\"python -c'"
    )

    def test_near_the_limit_it_is_treated_as_overflow(self) -> None:
        conv = _conv(max_tokens=1_000)
        conv.add_user_message("word " * 700)
        assert conv.token_count >= 600
        assert _is_truncated_tool_call(self.ERR, conv)

    def test_far_from_the_limit_it_is_an_ordinary_error(self) -> None:
        conv = _conv(max_tokens=100_000)
        conv.add_user_message("short")
        assert not _is_truncated_tool_call(self.ERR, conv)

    def test_unrelated_errors_are_never_matched(self) -> None:
        conv = _conv(max_tokens=1_000)
        conv.add_user_message("word " * 700)
        assert not _is_truncated_tool_call(Exception("connection reset"), conv)


class TestSetting:
    def test_default_and_validation(self) -> None:
        assert GodspeedSettings().completion_reserve_tokens == 4096
        with pytest.raises(ValueError, match="completion_reserve_tokens"):
            GodspeedSettings(completion_reserve_tokens=-1)

    def test_yaml_key_is_known(self) -> None:
        from godspeed.config import _KNOWN_TOP_LEVEL_KEYS

        assert "completion_reserve_tokens" in _KNOWN_TOP_LEVEL_KEYS
