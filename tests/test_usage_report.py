"""Tests for the /usage command and its pure usage-report builders."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.audit.trail import AuditTrail
from godspeed.observability.usage_report import (
    SubagentRow,
    TokenRow,
    ToolRow,
    UsageReport,
    from_audit,
    from_client,
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
    llm_client.total_cost_usd = 0.0
    return Commands(
        conversation=conversation,
        llm_client=llm_client,
        permission_engine=None,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
        tool_registry=None,
    )


def _make_client(input_tokens: int, output_tokens: int, cost_usd: float) -> MagicMock:
    client = MagicMock()
    client.total_input_tokens = input_tokens
    client.total_output_tokens = output_tokens
    client.total_cost_usd = cost_usd
    return client


class TestFromClient:
    """Pure builder: session totals from an LLM client."""

    def test_zero_state(self) -> None:
        report = from_client(_make_client(0, 0, 0.0))
        assert report.total_input_tokens == 0
        assert report.total_output_tokens == 0
        assert report.total_cost_usd == 0.0
        assert report.total_tokens == 0
        assert report.by_task_type == {}
        assert report.by_tool == {}
        assert report.by_subagent == {}

    def test_mixed_rows(self) -> None:
        report = from_client(_make_client(12_345, 6_789, 0.42))
        assert report.total_input_tokens == 12_345
        assert report.total_output_tokens == 6_789
        assert report.total_cost_usd == 0.42
        assert report.total_tokens == 19_134

    def test_missing_attributes_default_to_zero(self) -> None:
        report = from_client(object())
        assert report.total_input_tokens == 0
        assert report.total_output_tokens == 0
        assert report.total_cost_usd == 0.0

    def test_aggregation_math_exact(self) -> None:
        report = from_client(_make_client(1_000_000, 500_000, 3.75))
        assert report.total_input_tokens == 1_000_000
        assert report.total_output_tokens == 500_000
        assert report.total_cost_usd == 3.75
        assert report.total_tokens == 1_500_000


class TestFromAudit:
    """Pure builder: tool-call and sub-agent counts from an audit trail."""

    def _trail(self, tmp_path: Path, session_id: str = "sess-1") -> AuditTrail:
        return AuditTrail(log_dir=tmp_path, session_id=session_id)

    def test_zero_state(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        report = from_audit(trail)
        assert report.by_tool == {}
        assert report.by_subagent == {}
        assert report.total_input_tokens == 0

    def test_tool_call_counts_aggregate(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.record("tool_call", detail={"tool": "file_read"})
        trail.record("tool_call", detail={"tool": "file_read"})
        trail.record("tool_call", detail={"tool": "shell"})
        trail.record("tool_call", detail={"tool": "shell"})
        trail.record("tool_call", detail={"tool": "shell"})
        trail.close()

        report = from_audit(trail)
        assert report.by_tool["file_read"].calls == 2
        assert report.by_tool["shell"].calls == 3
        assert len(report.by_tool) == 2
        # No token/cost attribution in the audit trail — stays zero.
        assert report.by_tool["file_read"].input_tokens == 0
        assert report.by_tool["file_read"].cost_usd == 0.0

    def test_non_tool_events_ignored(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.record("llm_request", detail={"model": "x"})
        trail.record("permission_check", detail={"tool": "shell"})
        trail.close()

        report = from_audit(trail)
        assert report.by_tool == {}

    def test_missing_tool_key_counts_as_unknown(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.record("tool_call", detail={"arguments": {}})
        trail.close()

        report = from_audit(trail)
        assert report.by_tool["unknown"].calls == 1

    def test_sidechain_records_count_subagents(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.record("session_start", detail={}, is_sidechain=True)
        trail.record("session_start", detail={}, is_sidechain=True)
        trail.record("session_start", detail={})
        trail.close()

        report = from_audit(trail)
        assert report.by_subagent["sess-1"].calls == 2
        assert len(report.by_subagent) == 1

    def test_missing_log_file_returns_empty(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        report = from_audit(trail)
        assert report.by_tool == {}
        assert report.by_subagent == {}

    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.record("tool_call", detail={"tool": "file_read"})
        trail.close()
        trail.log_path.write_text(
            trail.log_path.read_text() + "{not valid json}\n",
            encoding="utf-8",
        )

        report = from_audit(trail)
        assert report.by_tool["file_read"].calls == 1


class TestUsageCommand:
    """/usage dispatch behavior."""

    def test_usage_registered(self, commands: Commands) -> None:
        result = commands.dispatch("/usage")
        assert result is not None
        assert result.handled

    def test_usage_empty_session_zero_state(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/usage")
        assert "Session Usage" in output
        assert "Input tokens" in output
        assert "Output tokens" in output
        assert "Total tokens" in output
        assert "Estimated cost" in output
        assert "No task-type calls recorded yet" in output  # honest gap note, not a crash

    def test_usage_shows_client_totals(self, commands: Commands) -> None:
        commands._llm_client.total_input_tokens = 1_000
        commands._llm_client.total_output_tokens = 500
        commands._llm_client.total_cost_usd = 0.012
        output = _capture(commands.dispatch, "/usage")
        assert "1,000" in output
        assert "500" in output
        assert "1,500" in output
        assert "$0.01" in output  # format_cost: 0.012 >= $0.01 → 2 decimals

    def test_usage_unknown_scope_graceful(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/usage bogus")
        assert "Unknown scope: bogus" in output

    def test_usage_tools_without_audit(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/usage tools")
        assert "Audit trail is disabled" in output

    def test_usage_tools_with_audit(self, commands: Commands, tmp_path: Path) -> None:
        trail = AuditTrail(log_dir=tmp_path, session_id="test-session")
        trail.record("tool_call", detail={"tool": "file_read"})
        trail.record("tool_call", detail={"tool": "file_read"})
        trail.record("tool_call", detail={"tool": "shell"})
        trail.close()
        commands._audit_trail = trail

        output = _capture(commands.dispatch, "/usage tools")
        assert "Tool Usage" in output
        assert "file_read" in output
        assert "shell" in output
        assert "2" in output
        assert "1" in output
        assert "only call counts are measurable" in output

    def test_usage_tools_empty_audit(self, commands: Commands, tmp_path: Path) -> None:
        trail = AuditTrail(log_dir=tmp_path, session_id="test-session")
        trail.close()
        commands._audit_trail = trail

        output = _capture(commands.dispatch, "/usage tools")
        assert "No tool calls recorded" in output

    def test_usage_agents_shows_gap_note(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/usage agents")
        assert "Sub-Agent Usage" in output
        assert "No per-subagent usage recorded" in output

    def test_usage_agents_shows_aggregate_cost(self, commands: Commands) -> None:
        commands._llm_client.total_sub_agent_cost = 0.25
        output = _capture(commands.dispatch, "/usage agents")
        assert "Aggregate sub-agent cost" in output
        assert "$0.25" in output

    def test_usage_in_help(self, commands: Commands) -> None:
        output = _capture(commands.dispatch, "/help")
        assert "/usage [scope]" in output

    def test_stats_untouched(self, commands: Commands) -> None:
        commands._llm_client.total_input_tokens = 100
        commands._llm_client.total_output_tokens = 50
        result = commands.dispatch("/stats")
        assert result is not None
        assert result.handled


class TestRowDataclasses:
    """Row dataclasses carry the expected fields with zero defaults."""

    def test_token_row_defaults(self) -> None:
        row = TokenRow()
        assert row.calls == 0
        assert row.input_tokens == 0
        assert row.output_tokens == 0
        assert row.cost_usd == 0.0

    def test_tool_row_defaults(self) -> None:
        row = ToolRow()
        assert row.calls == 0
        assert row.input_tokens == 0
        assert row.output_tokens == 0
        assert row.cost_usd == 0.0

    def test_subagent_row_defaults(self) -> None:
        row = SubagentRow()
        assert row.calls == 0
        assert row.input_tokens == 0
        assert row.output_tokens == 0
        assert row.cost_usd == 0.0

    def test_usage_report_defaults(self) -> None:
        report = UsageReport()
        assert report.total_input_tokens == 0
        assert report.total_output_tokens == 0
        assert report.total_cost_usd == 0.0
        assert report.by_task_type == {}
        assert report.by_tool == {}
        assert report.by_subagent == {}
