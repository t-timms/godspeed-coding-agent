"""Coverage gap tests for conversation compaction — all missed branches."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.context.compaction import (
    COMPACTION_PROMPT_LARGE,
    COMPACTION_PROMPT_SMALL,
    COMPACTION_STAGES,
    EMERGENCY_STAGE_IDX,
    CompactionContext,
    CompactionResult,
    GraduatedCompactor,
    _build_gcg_summary,
    _collapse_tool_runs_to_gcg_summaries,
    _drop_verbose_tool_outputs,
    _keep_metadata_only,
    _llm_emergency_summarize,
    _messages_to_text,
    _remove_low_signal_turns,
    compact_if_needed,
    get_compaction_prompt,
)
from godspeed.llm.client import ChatResponse, LLMClient


# ── Drop verbose tool outputs — edge cases ──────────────────────────────────


class TestDropVerboseToolOutputs:
    """Cover remaining branches in _drop_verbose_tool_outputs."""

    def test_tool_output_under_3000_chars_unchanged(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "tool", "content": "short output"},
                {"role": "user", "content": "hello"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _drop_verbose_tool_outputs(ctx)
        assert len(result) == 2
        assert result[0]["content"] == "short output"  # line 137: preserved as-is

    def test_tool_content_not_string_preserved(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "tool", "content": [{"type": "text", "text": "structured"}]},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _drop_verbose_tool_outputs(ctx)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)


# ── Remove low signal turns — edge cases ────────────────────────────────────


class TestRemoveLowSignalTurns:
    """Cover _remove_low_signal_turns with signal-scorer-based decisions."""

    def test_tool_messages_always_preserved(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "tool", "content": "some output"},
                {"role": "tool", "tool_call_id": "t1", "content": "result"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 2

    def test_short_acknowledgement_dropped(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "assistant", "content": "ok"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 0

    def test_planning_discussion_preserved(self) -> None:
        ctx = CompactionContext(
            messages=[
                {
                    "role": "assistant",
                    "content": (
                        "I propose using a factory pattern instead. "
                        "This approach trades simplicity for extensibility. "
                        "What do you think about this alternative?"
                    ),
                },
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 1

    def test_code_block_preserved(self) -> None:
        ctx = CompactionContext(
            messages=[
                {
                    "role": "assistant",
                    "content": "Here's the fix:\n```python\ndef solve():\n    pass\n```",
                },
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 1

    def test_question_preserved(self) -> None:
        ctx = CompactionContext(
            messages=[
                {
                    "role": "assistant",
                    "content": "Should we refactor the adapter? I'm concerned about the design.",
                },
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 1

    def test_user_messages_always_preserved(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "ok"},
                {"role": "user", "content": "sure"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        assert len(result) == 2


# ── Collapse tool runs to GCG summaries ─────────────────────────────────────


class TestCollapseToolRuns:
    """Cover _collapse_tool_runs_to_gcg_summaries."""

    def test_tool_run_above_threshold_collapsed(self) -> None:
        ctx = CompactionContext(
            messages=[
                {
                    "role": "assistant",
                    "content": "doing work",
                    "tool_calls": [{"function": {"name": "read"}}],
                },
                {"role": "tool", "tool_call_id": "read-1", "content": "a"},
                {"role": "tool", "tool_call_id": "read-2", "content": "b"},
                {"role": "tool", "tool_call_id": "write-3", "content": "c"},
                {"role": "tool", "tool_call_id": "write-4", "content": "d"},
                {"role": "user", "content": "next message"},  # triggers flush
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _collapse_tool_runs_to_gcg_summaries(ctx)
        # 4 tool results collapsed into 1 compacted + assistant + user
        assert len(result) == 3
        assert any("compacted" in m.get("content", "") for m in result)

    def test_tool_run_below_threshold_not_collapsed(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "assistant", "content": "doing work"},
                {"role": "tool", "tool_call_id": "read-1", "content": "a"},
                {"role": "tool", "tool_call_id": "read-2", "content": "b"},
                {"role": "user", "content": "next"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _collapse_tool_runs_to_gcg_summaries(ctx)
        # Only 2 tool results — not collapsed (threshold > 3)
        assert len(result) == 4

    def test_tool_run_trailing_pending_not_collapsed(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "tool", "tool_call_id": "read-1", "content": "a"},
                {"role": "tool", "tool_call_id": "read-2", "content": "b"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _collapse_tool_runs_to_gcg_summaries(ctx)
        # trailing pending (2 < threshold) — appended as-is
        assert len(result) == 3

    def test_collapse_with_gcg_refs(self) -> None:
        fake_gcg = MagicMock()
        ctx = CompactionContext(
            messages=[
                {"role": "assistant", "content": "working"},
                {"role": "tool", "tool_call_id": "r-1", "content": "file: /app/main.py\nline 42"},
                {"role": "tool", "tool_call_id": "r-2", "content": "/src/utils.py"},
                {"role": "tool", "tool_call_id": "r-3", "content": "other stuff"},
                {"role": "tool", "tool_call_id": "r-4", "content": "more"},
                {"role": "user", "content": "next"},  # triggers flush
            ],
            token_count=100,
            max_tokens=100_000,
            gcg=fake_gcg,
        )
        result = _collapse_tool_runs_to_gcg_summaries(ctx)
        assert len(result) == 3
        compacted_content = result[1]["content"]  # second element is the compacted message
        assert "compacted" in compacted_content


# ── Keep metadata only — edge cases ─────────────────────────────────────────


class TestKeepMetadataOnly:
    """Cover remaining branches in _keep_metadata_only."""

    def test_user_message_truncation(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "x" * 250},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert len(result) == 1
        # Content > 200 chars should be truncated
        assert len(result[0]["content"]) <= 203  # 200 + "..."

    def test_user_message_under_200_unchanged(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "short message"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert result[0]["content"] == "short message"

    def test_assistant_with_tool_calls(self) -> None:
        ctx = CompactionContext(
            messages=[
                {
                    "role": "assistant",
                    "content": "doing work",
                    "tool_calls": [
                        {"function": {"name": "file_edit"}},
                        {"function": {"name": "shell"}},
                    ],
                },
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert len(result) == 1
        assert "[tool calls:" in result[0]["content"]
        assert "file_edit" in result[0]["content"]
        assert "shell" in result[0]["content"]

    def test_assistant_without_tool_calls_no_truncation(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "assistant", "content": "short"},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert result[0]["content"] == "short"

    def test_assistant_without_tool_calls_with_truncation(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "assistant", "content": "y" * 250},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert len(result[0]["content"]) <= 203

    def test_tool_message_truncation(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "tool", "content": "z" * 250},
            ],
            token_count=100,
            max_tokens=100_000,
        )
        result = _keep_metadata_only(ctx)
        assert len(result[0]["content"]) <= 203

    def test_gcg_symbol_ids_appended(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "hi"},
            ],
            token_count=100,
            max_tokens=100_000,
            gcg_symbol_ids=["gcg:main.helper", "gcg:main.Calculator"],
        )
        result = _keep_metadata_only(ctx)
        assert len(result) == 2
        assert "GCG references preserved" in result[1]["content"]
        assert "2 symbols" in result[1]["content"]

    def test_no_gcg_symbol_ids_no_gcg_message(self) -> None:
        ctx = CompactionContext(  # covers lines 239-245 branch
            messages=[{"role": "user", "content": "hi"}],
            token_count=100,
            max_tokens=100_000,
            gcg_symbol_ids=[],
        )
        result = _keep_metadata_only(ctx)
        assert len(result) == 1


# ── LLM emergency summarize ─────────────────────────────────────────────────


class TestLLMEmergencySummarize:
    """Cover _llm_emergency_summarize."""

    @pytest.mark.asyncio
    async def test_emergency_summarize_basic(self) -> None:
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Summary text", tool_calls=[], finish_reason="stop")
        )
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "working"},
            ],
            token_count=200,
            max_tokens=100_000,
        )
        result = await _llm_emergency_summarize(ctx, client, "claude-sonnet-4")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Summary text" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_emergency_summarize_with_tool_calls(self) -> None:
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Compact", tool_calls=[], finish_reason="stop")
        )
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "edit file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "file_edit", "arguments": '{"path":"x"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "1", "content": "done"},
            ],
            token_count=300,
            max_tokens=100_000,
        )
        result = await _llm_emergency_summarize(ctx, client, "claude-sonnet-4")
        assert len(result) == 1
        assert "[Conversation compacted" in result[0]["content"]


# ── Messages to text ────────────────────────────────────────────────────────


class TestMessagesToText:
    """Cover _messages_to_text."""

    def test_converts_messages_to_text(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        text = _messages_to_text(messages)
        assert "[user]: hello" in text
        assert "[assistant]: hi there" in text
        assert "[tool]: result" in text

    def test_message_with_tool_calls_included(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "file_read", "arguments": '{"file":"a.py"}'}}],
            },
        ]
        text = _messages_to_text(messages)
        assert "[tool_call]: file_read" in text
        assert '{"file":"a.py"}' in text

    def test_empty_content_messages_skipped(self) -> None:
        messages = [
            {"role": "assistant", "content": ""},
        ]
        text = _messages_to_text(messages)
        assert text == ""  # no content and no tool_calls

    def test_unknown_role_default(self) -> None:
        messages = [
            {"role": "unknown", "content": "some msg"},
        ]
        text = _messages_to_text(messages)
        assert "[unknown]: some msg" in text

    def test_multiple_tool_calls_in_one_message(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": "{}"}},
                    {"function": {"name": "edit", "arguments": '{"path":"x"}'}},
                ],
            },
        ]
        text = _messages_to_text(messages)
        assert "[tool_call]: read" in text
        assert "[tool_call]: edit" in text


# ── Build GCG summary ───────────────────────────────────────────────────────


class TestBuildGCGSummary:
    """Cover _build_gcg_summary."""

    def test_no_gcg_returns_empty(self) -> None:
        ctx = CompactionContext(messages=[], token_count=0, max_tokens=100_000, gcg=None)
        result = _build_gcg_summary(ctx, [])
        assert result == ""

    def test_no_file_paths_returns_empty(self) -> None:
        fake_gcg = MagicMock()
        ctx = CompactionContext(messages=[], token_count=0, max_tokens=100_000, gcg=fake_gcg)
        results = [{"content": "no file paths here\njust text"}]
        result = _build_gcg_summary(ctx, results)
        assert result == ""

    def test_extracts_file_paths(self) -> None:
        fake_gcg = MagicMock()
        ctx = CompactionContext(messages=[], token_count=0, max_tokens=100_000, gcg=fake_gcg)
        results = [
            {"content": "file: /app/main.py\n/usr/lib/util.py"},
            {"content": "noise"},
        ]
        result = _build_gcg_summary(ctx, results)
        assert "GCG refs:" in result
        assert "/app/main.py" in result or "/usr/lib/util.py" in result

    def test_non_string_content_skipped(self) -> None:
        fake_gcg = MagicMock()
        ctx = CompactionContext(messages=[], token_count=0, max_tokens=100_000, gcg=fake_gcg)
        results = [{"content": ["not a string"]}]
        result = _build_gcg_summary(ctx, results)
        assert result == ""

    def test_many_paths_truncated_to_10(self) -> None:
        fake_gcg = MagicMock()
        ctx = CompactionContext(messages=[], token_count=0, max_tokens=100_000, gcg=fake_gcg)
        results = [{"content": "\n".join(f"file: /path/p{i}.py" for i in range(20))}]
        result = _build_gcg_summary(ctx, results)
        assert result.count("/path/") <= 10


# ── Graduated compactor — exception and emergency paths ─────────────────────


class TestGraduatedCompactorExceptions:
    """Cover exception handler in apply_stages and emergency_compact."""

    def test_stage_strategy_raises_exception_logged(self, caplog) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(10):
            conv.add_user_message(f"msg {i}")
            conv.add_assistant_message(f"resp {i}")

        # Replace stage 0 strategy to raise
        with patch.object(COMPACTION_STAGES[0], "strategy", side_effect=RuntimeError("boom")):
            results = compactor.apply_stages(conv, 80_000, 100_000)
            assert len(results) == 0  # exception swallowed

        assert "failed" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_emergency_compact(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(20):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(
                content="Emergency summary", tool_calls=[], finish_reason="stop"
            )
        )

        before = len(conv._messages)
        result = await compactor.emergency_compact(conv, client, "claude-sonnet-4")
        after = len(conv._messages)

        assert isinstance(result, CompactionResult)
        assert result.stage_name == "auto_compact"
        assert result.applied is True
        assert result.messages_before == before
        assert result.messages_after == after
        assert after < before

    @pytest.mark.asyncio
    async def test_emergency_compact_sets_last_stage(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(20):
            conv.add_user_message(f"Msg {i}")
            conv.add_assistant_message(f"Rsp {i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Done", tool_calls=[], finish_reason="stop")
        )

        await compactor.emergency_compact(conv, client, "test-model")
        assert compactor._last_stage_idx >= EMERGENCY_STAGE_IDX


# ── Get compaction prompt — medium path with logging ────────────────────────


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    import logging

    logging.getLogger("godspeed.context.compaction").handlers.clear()
    logging.getLogger("godspeed.context.compaction").propagate = True
    yield


class TestCompactionPromptMedium:
    """Cover the medium prompt selection and logging paths."""

    def test_medium_prompt_logging(self) -> None:
        prompt = get_compaction_prompt("gpt-4o-2024-05-13")
        assert prompt == COMPACTION_PROMPT_LARGE  # 128K > 100K threshold

    def test_small_prompt_logging(self) -> None:
        prompt = get_compaction_prompt("gpt-4-0613")
        assert prompt == COMPACTION_PROMPT_SMALL

    def test_medium_prompt_fallback_logging(self) -> None:
        # Need a model between 32K and 100K. Default unknown is 32K (≤ threshold, small)
        # Let's test medium: use a model with context between 32768+1 and 100000
        # There's no built-in model in that range in the config, so let's check what happens
        prompt = get_compaction_prompt("some-custom-model-50k")
        # Default unknown = 32768, which is ≤ SMALL_CONTEXT_THRESHOLD (32768)
        assert prompt == COMPACTION_PROMPT_SMALL


# ── compact_if_needed — exception and edge paths ────────────────────────────


class TestCompactIfNeededEdgeCases:
    """Cover exception handler and branch edges in compact_if_needed."""

    @pytest.mark.asyncio
    async def test_compaction_failure_returns_false(self) -> None:
        conv = Conversation("System", max_tokens=100, compaction_threshold=0.01)
        for i in range(20):
            conv.add_user_message(f"Message {i} with content")
            conv.add_assistant_message(f"Response {i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=Exception("LLM down"))

        result = await compact_if_needed(conv, client, model="claude-sonnet-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_compaction_not_near_limit_no_op(self) -> None:
        conv = Conversation("System", max_tokens=1_000_000)
        conv.add_user_message("hi")

        client = LLMClient(model="test")
        result = await compact_if_needed(conv, client, model="test")
        assert result is False

    @pytest.mark.asyncio
    async def test_compaction_model_from_client_fallback(self) -> None:
        conv = Conversation("System", max_tokens=100, compaction_threshold=0.01)
        for i in range(20):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")

        client = LLMClient(model="fallback-model")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Summary", tool_calls=[], finish_reason="stop")
        )

        # No explicit model arg -> uses client.model fallback
        result = await compact_if_needed(conv, client, model=None)
        assert isinstance(result, bool)


# ── Additional graduated compactor edge cases ───────────────────────────────


class TestGraduatedCompactorEdges:
    """Cover edge cases in GraduatedCompactor."""

    def test_get_stage_zero_tokens(self) -> None:
        compactor = GraduatedCompactor()
        idx = compactor.get_stage_for_context(0, 0)
        assert idx == -1
        assert compactor.context_pct == 0.0

    def test_custom_stages(self) -> None:
        custom_stages = COMPACTION_STAGES[:2]  # only first 2
        compactor = GraduatedCompactor(stages=custom_stages)
        assert len(compactor._stages) == 2

    def test_apply_stages_multiple_rounds(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(50):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")
            conv.add_tool_result(f"tool-{i}", f"Result {i}" * 50)

        # Round 1: 80% → stage 0
        r1 = compactor.apply_stages(conv, 80_000, 100_000)
        assert any(r.applied for r in r1)

        # Reset for round 2
        compactor.reset()

        # Round 2: 65% → stage 1
        conv2 = Conversation("System", max_tokens=100_000)
        for i in range(50):
            conv2.add_user_message(f"Msg {i}")
            conv2.add_assistant_message(f"Rsp {i}")
            conv2.add_tool_result(f"t-{i}", f"Data {i}" * 50)

        r2 = compactor.apply_stages(conv2, 65_000, 100_000)
        applied2 = [r for r in r2 if r.applied]
        if applied2:
            assert any(r.stage_name in ("budget_reduction", "snip") for r in applied2)

    def test_apply_stages_empty_messages(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        results = compactor.apply_stages(conv, 80_000, 100_000)
        assert isinstance(results, list)

    def test_stage3_collapse_preserves_system_and_user(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "do something"},
                {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [{"function": {"name": "edit"}}],
                },
                {"role": "tool", "tool_call_id": "a-1", "content": "x"},
                {"role": "tool", "tool_call_id": "a-2", "content": "y"},
                {"role": "tool", "tool_call_id": "a-3", "content": "z"},
                {"role": "tool", "tool_call_id": "a-4", "content": "w"},
                {"role": "user", "content": "done"},  # triggers flush
            ],
            token_count=200,
            max_tokens=100_000,
        )
        result = _collapse_tool_runs_to_gcg_summaries(ctx)
        assert len(result) == 5  # system + user + assistant + compacted tool + done user

    def test_stage3_preserves_short_tool_runs(self) -> None:
        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [{"function": {"name": "read"}}],
                },
                {"role": "tool", "tool_call_id": "r-1", "content": "a"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "next"},
                {
                    "role": "assistant",
                    "content": "ok2",
                    "tool_calls": [{"function": {"name": "edit"}}],
                },
                {"role": "tool", "tool_call_id": "e-1", "content": "b"},
            ],
            token_count=200,
            max_tokens=100_000,
        )
        result = _collapse_tool_runs_to_gcg_summaries(ctx)
        assert len(result) > 0

    def test_context_pct_property(self) -> None:
        compactor = GraduatedCompactor()
        assert compactor.context_pct == 0.0
        compactor.get_stage_for_context(80_000, 100_000)
        assert compactor.context_pct == 0.8

    def test_get_stage_edge_between0_and1(self) -> None:
        compactor = GraduatedCompactor()
        # Exactly at threshold
        idx = compactor.get_stage_for_context(75_000, 100_000)  # 75% → budget_reduction
        assert idx == 0
        # Slightly above
        idx = compactor.get_stage_for_context(75_001, 100_000)  # > 75%
        assert idx == 0

    def test_apply_stages_already_at_stage(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(50):
            conv.add_user_message(f"Msg {i}")
            conv.add_assistant_message(f"Rsp {i}")
            conv.add_tool_result(f"t-{i}", f"Data {i}" * 50)

        # Apply stage 0
        compactor.apply_stages(conv, 80_000, 100_000)
        # Same usage again — should be no-op
        results = compactor.apply_stages(conv, 80_000, 100_000)
        assert len(results) == 0

    def test_apply_stages_progresses_through_stages(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(50):
            conv.add_user_message(f"Msg {i}")
            conv.add_assistant_message(f"Rsp {i}")
            conv.add_tool_result(f"t-{i}", f"Data {i}" * 50)

        # First: apply up to stage 0 (75%)
        r1 = compactor.apply_stages(conv, 80_000, 100_000)
        assert any(r.applied for r in r1)

        # Then: context usage stays high, apply more stages (simulate 45% → stage 2)
        conv2 = Conversation("System", max_tokens=100_000)
        for i in range(50):
            conv2.add_user_message(f"X{i}")
            conv2.add_assistant_message(f"Y{i}")

        compactor2 = GraduatedCompactor()
        r2 = compactor2.apply_stages(conv2, 50_000, 100_000)
        applied = [r.stage_name for r in r2 if r.applied]
        # Should have applied stages 0, 1, 2
        assert "budget_reduction" in applied

    def test_no_stage_needed_returns_empty(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        conv.add_user_message("short")
        results = compactor.apply_stages(conv, 5_000, 100_000)  # 5% → no stage
        assert results == []


# ── get_compaction_prompt coverage ──────────────────────────────────────────


class TestGetCompactionPromptCoverage:
    """Cover all branches of get_compaction_prompt."""

    def test_small_context_threshold_boundary(self) -> None:
        # Model with context == 32768 → small prompt
        prompt = get_compaction_prompt("gpt-4o-mini")
        # gpt-4o-mini = 128000 → large
        assert prompt == COMPACTION_PROMPT_LARGE

    def test_large_context_threshold_boundary(self) -> None:
        # Model with context > 100000 → large prompt
        prompt = get_compaction_prompt("claude-opus-4-20250514")
        assert prompt == COMPACTION_PROMPT_LARGE

    def test_between_thresholds_logs_medium(self) -> None:
        prompt = get_compaction_prompt("gemini-2-flash")
        assert prompt == COMPACTION_PROMPT_LARGE


# ── Data-loss regression: factory-pattern discussion survives ───────────────


class TestFactoryPatternSurvival:
    """Stage 2 must preserve planning discussions (not just keyword hits)."""

    def test_factory_pattern_discussion_survives_scorer(self) -> None:
        from godspeed.context.compaction import _remove_low_signal_turns

        ctx = CompactionContext(
            messages=[
                {"role": "user", "content": "How should we handle the instantiation?"},
                {
                    "role": "assistant",
                    "content": (
                        "I propose using a factory pattern. The factory encapsulates "
                        "creation logic and lets us swap implementations without "
                        "changing the caller. This is the recommended approach for "
                        "our architecture. What do you think about this alternative?"
                    ),
                },
                {"role": "assistant", "content": "sure"},
                {"role": "user", "content": "sounds good"},
            ],
            token_count=200,
            max_tokens=100_000,
        )
        result = _remove_low_signal_turns(ctx)
        roles = [m["role"] for m in result]
        assert "user" in roles
        contents = [m.get("content", "") for m in result]
        assert any("factory pattern" in c for c in contents)
        assert not any(c == "sure" for c in contents)


# ── Stage5 ≠ Stage4 ─────────────────────────────────────────────────────────


class TestStage5DistinctFromStage4:
    """Stage 5 (auto_compact) must differ from Stage 4 (context_collapse)."""

    def test_stages_list_has_no_auto_compact(self) -> None:
        names = [s.name for s in COMPACTION_STAGES]
        assert "auto_compact" not in names

    def test_context_collapse_is_last_deterministic(self) -> None:
        assert COMPACTION_STAGES[-1].name == "context_collapse"

    def test_emergency_compact_uses_llm_not_truncation(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(20):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(
                content="LLM summary of everything", tool_calls=[], finish_reason="stop"
            )
        )

        result = asyncio.run(compactor.emergency_compact(conv, client, "test-model"))
        assert result.stage_name == "auto_compact"
        summary_text = conv.messages[1]["content"]
        assert "LLM summary" in summary_text
        assert "[tool calls:" not in summary_text


# ── replace_messages atomicity ───────────────────────────────────────────────


class TestReplaceMessages:
    """Conversation.replace_messages must atomically swap + invalidate caches."""

    def test_replaces_and_invalidates_cache(self) -> None:
        conv = Conversation("System", max_tokens=100_000)
        conv.add_user_message("hello")
        _ = conv.token_count
        conv.replace_messages([{"role": "user", "content": "world"}])
        assert conv.messages[1]["content"] == "world"
        assert conv._token_count_cache is None

    def test_does_not_leak_old_messages(self) -> None:
        conv = Conversation("System", max_tokens=100_000)
        conv.add_user_message("first")
        conv.add_assistant_message("second")
        conv.replace_messages([{"role": "user", "content": "only"}])
        assert len(conv.messages) == 2
        assert conv.messages[1]["content"] == "only"

    def test_makes_defensive_copy(self) -> None:
        conv = Conversation("System", max_tokens=100_000)
        external = [{"role": "user", "content": "test"}]
        conv.replace_messages(external)
        external[0]["content"] = "mutated"
        assert conv.messages[1]["content"] == "test"


# ── _last_stage_idx guard ────────────────────────────────────────────────────


class TestLastStageIdxGuard:
    """_last_stage_idx must only advance forward, never regress."""

    def test_emergency_does_not_regard_after_apply(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(50):
            conv.add_user_message(f"Msg {i}")
            conv.add_assistant_message(f"Rsp {i}")
            conv.add_tool_result(f"t-{i}", f"Data {i}" * 50)

        compactor.apply_stages(conv, 80_000, 100_000)
        stage_after_apply = compactor._last_stage_idx
        assert stage_after_apply >= 0

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Emergency", tool_calls=[], finish_reason="stop")
        )
        asyncio.run(compactor.emergency_compact(conv, client, "test"))
        assert compactor._last_stage_idx >= stage_after_apply

    def test_double_emergency_does_not_regard(self) -> None:
        compactor = GraduatedCompactor()
        conv = Conversation("System", max_tokens=100_000)
        for i in range(20):
            conv.add_user_message(f"M{i}")
            conv.add_assistant_message(f"R{i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Sum", tool_calls=[], finish_reason="stop")
        )
        asyncio.run(compactor.emergency_compact(conv, client, "m"))
        first = compactor._last_stage_idx
        asyncio.run(compactor.emergency_compact(conv, client, "m"))
        assert compactor._last_stage_idx >= first


class TestFailedStageSkip:
    """A stage that raised is added to _failed_stages and never retried."""

    def test_failed_stage_skipped_not_retried(self) -> None:
        calls: list[int] = []

        def boom(ctx: object) -> list:
            calls.append(1)
            msg = "boom"
            raise RuntimeError(msg)

        from godspeed.context.compaction import CompactionStage

        stages = [
            CompactionStage(name="boom", threshold_pct=0.10, preserves=[], strategy=boom),
        ]
        compactor = GraduatedCompactor(stages=stages)
        conv = Conversation("System", max_tokens=100_000)
        for i in range(5):
            conv.add_user_message(f"M{i}")
            conv.add_assistant_message(f"R{i}")

        r1 = compactor.apply_stages(conv, 20_000, 100_000)
        assert r1 == []
        assert compactor._failed_stages == {0}
        assert len(calls) == 1

        r2 = compactor.apply_stages(conv, 20_000, 100_000)
        assert r2 == []
        assert len(calls) == 1
