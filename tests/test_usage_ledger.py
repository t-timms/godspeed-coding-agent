"""Tests for the usage ledger and LLM-client / coordinator attribution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


import pytest

from godspeed.llm.client import LLMClient
from godspeed.llm.usage_ledger import (
    LedgerEntry,
    LedgerRow,
    PARENT_KEY,
    UsageLedger,
    subagent_context,
)
from godspeed.observability.usage_report import TokenRow


def _mock_response(
    content: str = "Hello",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> SimpleNamespace:
    """Mock litellm response mirroring tests/test_llm_client.py's helper."""
    msg = SimpleNamespace(content=content, tool_calls=[], thinking=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestLedgerPure:
    def test_zero_state(self) -> None:
        ledger = UsageLedger()
        assert ledger.by_task_type() == {}
        assert ledger.by_subagent() == {}
        assert ledger.totals() == LedgerRow()

    def test_mixed_task_types_accumulate(self) -> None:
        ledger = UsageLedger()
        ledger.record(task_type="chat", input_tokens=10, output_tokens=5, cost_usd=0.10)
        ledger.record(task_type="edit", input_tokens=20, output_tokens=8, cost_usd=0.20)
        ledger.record(task_type="chat", input_tokens=5, output_tokens=2, cost_usd=0.05)
        rows = ledger.by_task_type()
        assert set(rows) == {"chat", "edit"}
        chat_row = rows["chat"]
        assert chat_row.calls == 2
        assert chat_row.input_tokens == 15
        assert chat_row.output_tokens == 7
        assert chat_row.cost_usd == pytest.approx(0.15)
        edit_row = rows["edit"]
        assert edit_row.calls == 1
        assert edit_row.input_tokens == 20
        assert edit_row.output_tokens == 8
        assert edit_row.cost_usd == pytest.approx(0.2)

    def test_totals(self) -> None:
        ledger = UsageLedger()
        ledger.record(task_type="chat", input_tokens=10, output_tokens=5, cost_usd=0.1)
        ledger.record(task_type="edit", input_tokens=20, output_tokens=8, cost_usd=0.2)
        assert ledger.totals() == LedgerRow(
            calls=2, input_tokens=30, output_tokens=13, cost_usd=pytest.approx(0.3)
        )

    def test_explicit_subagent_beats_default(self) -> None:
        ledger = UsageLedger(default_subagent_id="def00001")
        ledger.record(
            task_type="chat", input_tokens=1, output_tokens=1, cost_usd=0.01, subagent_id="exp0002"
        )
        assert set(ledger.by_subagent()) == {"exp0002"}

    def test_default_subagent_id_tags_rows(self) -> None:
        ledger = UsageLedger(default_subagent_id="child123")
        ledger.record(task_type="chat", input_tokens=1, output_tokens=1, cost_usd=0.01)
        assert ledger.by_subagent()["child123"].calls == 1

    def test_untagged_keyed_parent(self) -> None:
        ledger = UsageLedger()
        ledger.record(task_type="chat", input_tokens=1, output_tokens=1, cost_usd=0.01)
        assert list(ledger.by_subagent()) == [PARENT_KEY]

    def test_contextvar_scope(self) -> None:
        ledger = UsageLedger()
        with subagent_context("ctxid999"):
            ledger.record(task_type="chat", input_tokens=1, output_tokens=1, cost_usd=0.01)
        ledger.record(task_type="chat", input_tokens=2, output_tokens=2, cost_usd=0.02)
        by_sa = ledger.by_subagent()
        assert by_sa["ctxid999"] == LedgerRow(
            calls=1, input_tokens=1, output_tokens=1, cost_usd=0.01
        )
        assert by_sa[PARENT_KEY] == LedgerRow(
            calls=1, input_tokens=2, output_tokens=2, cost_usd=0.02
        )

    def test_contextvar_reset_after_exit(self) -> None:
        ledger = UsageLedger()
        with subagent_context("gone123"):
            pass
        ledger.record(task_type="chat", input_tokens=0, output_tokens=0, cost_usd=0.0)
        assert list(ledger.by_subagent()) == [PARENT_KEY]

    def test_merge_from(self) -> None:
        parent = UsageLedger()
        parent.record(task_type="chat", input_tokens=1, cost_usd=1.0)
        child = UsageLedger(default_subagent_id="sid1234")
        child.record(task_type="chat", input_tokens=7, output_tokens=3, cost_usd=0.02)
        parent.merge_from(child)
        assert parent.by_subagent()["sid1234"].input_tokens == 7
        assert parent.totals().calls == 2

    def test_merge_self_noop(self) -> None:
        ledger = UsageLedger()
        ledger.record(task_type="chat", input_tokens=1, cost_usd=0.01)
        ledger.merge_from(ledger)
        assert ledger.totals().calls == 1

    def test_row_add(self) -> None:
        row = LedgerRow(calls=1, input_tokens=5, output_tokens=5, cost_usd=0.5)
        merged = row.add(LedgerEntry(task_type="x", input_tokens=2, output_tokens=1, cost_usd=0.25))
        assert merged == LedgerRow(calls=2, input_tokens=7, output_tokens=6, cost_usd=0.75)

    def test_from_rows_roundtrip(self) -> None:
        ledger = UsageLedger()
        ledger.record(task_type="chat", input_tokens=3, output_tokens=4, cost_usd=0.03)
        ledger.record(
            task_type="edit", input_tokens=9, output_tokens=2, cost_usd=0.01, subagent_id="aa12345"
        )
        rows: list[dict[str, object]] = [
            {
                "task_type": e.task_type,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "cost_usd": e.cost_usd,
                "subagent_id": e.subagent_id,
            }
            for e in ledger.entries
        ]
        rebuilt = UsageLedger.from_rows(rows)
        assert rebuilt.by_task_type() == ledger.by_task_type()
        assert rebuilt.by_subagent() == ledger.by_subagent()


class TestClientIntegration:
    @pytest.mark.asyncio
    async def test_chat_records_row_with_task_type(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = _mock_response(content="Hi", input_tokens=100, output_tokens=20)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            await client.chat([{"role": "user", "content": "hello"}], task_type="edit")

        rows = client.usage_ledger.by_task_type()
        assert rows["edit"] == LedgerRow(calls=1, input_tokens=100, output_tokens=20)
        assert rows["edit"].cost_usd >= 0
        assert client.usage_ledger.totals().calls == 1
        assert list(client.usage_ledger.by_subagent()) == [PARENT_KEY]

    def test_derive_shares_parent_ledger(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        child = client.derive("gpt-4o-mini")
        assert child.usage_ledger is client.usage_ledger

    def test_explicit_ledger_param_used(self) -> None:
        shared = UsageLedger()
        client = LLMClient(model="ollama/qwen3:4b", usage_ledger=shared)
        assert client.usage_ledger is shared


class TestCoordinatorAttribution:
    @pytest.mark.asyncio
    async def test_shared_client_scope_tags_subagent_rows(self) -> None:
        """Shared-parent path: rows recorded during a spawn carry its spawn id."""
        from godspeed.agent.coordinator import AgentCoordinator

        client = LLMClient(model="ollama/qwen3:4b")

        async def fake_loop(**kwargs: object) -> str:
            kwargs["llm_client"]._record_usage(  # type: ignore[union-attr]
                task_type="chat", input_tokens=10, output_tokens=5, cost_usd=0.01
            )
            return "done"

        mock_registry = SimpleNamespace(list_tools=lambda: [], _sandbox=None)
        tool_context = SimpleNamespace(audit=None, session_id="sess1", cwd=".")
        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=mock_registry,  # type: ignore[arg-type]
            tool_context=tool_context,  # type: ignore[arg-type]
        )
        with patch("godspeed.agent.coordinator.agent_loop", fake_loop):
            await coordinator.spawn("do things", depth=0)

        by_sa = client.usage_ledger.by_subagent()
        assert set(by_sa) != {PARENT_KEY}
        assert len(by_sa) == 1
        row = next(iter(by_sa.values()))
        assert row.input_tokens == 10
        assert row.output_tokens == 5


class TestUsageReportLedger:
    def test_from_client_with_ledger_fills_rows(self) -> None:
        from godspeed.observability.usage_report import from_client

        ledger = UsageLedger()
        ledger.record(task_type="chat", input_tokens=10, output_tokens=5, cost_usd=0.1)
        ledger.record(
            task_type="edit", input_tokens=20, output_tokens=8, cost_usd=0.2, subagent_id="sub0001"
        )
        client = SimpleNamespace(total_input_tokens=30, total_output_tokens=13, total_cost_usd=0.3)
        report = from_client(client, ledger)

        chat_row = report.by_task_type["chat"]
        assert isinstance(chat_row, TokenRow)
        assert (chat_row.calls, chat_row.input_tokens, chat_row.output_tokens) == (1, 10, 5)
        assert report.by_subagent["sub0001"].calls == 1
        assert set(report.by_subagent) == {"parent", "sub0001"}

    def test_from_client_without_ledger_unchanged(self) -> None:
        from godspeed.observability.usage_report import UsageReport, from_client

        client = SimpleNamespace(total_input_tokens=1, total_output_tokens=2, total_cost_usd=3.0)
        report = from_client(client)
        assert report == UsageReport(
            total_input_tokens=1, total_output_tokens=2, total_cost_usd=3.0
        )
