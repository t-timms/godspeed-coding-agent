"""Tests for OTLP/HTTP-JSON trace export and the /metrics command."""

from __future__ import annotations

import json
import re
import urllib.error
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.observability.metrics import Span, SpanStatus
from godspeed.observability.otlp import (
    ExportResult,
    export_otlp,
    spans_to_otlp_json,
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


def _span() -> Span:
    return Span(
        name="session",
        trace_id="0xabc123",
        span_id="DEF456",
        parent_span_id=None,
        start_time=1_700_000_000.0,
        end_time=1_700_000_001.5,
        status=SpanStatus.OK,
        attributes={
            "session.id": "sess-1",
            "input_tokens": 100,
            "cost_usd": 0.5,
            "is_active": True,
        },
    )


class TestSpansToOtlpJson:
    def test_exact_payload_shape(self) -> None:
        payload = spans_to_otlp_json([_span()])
        assert set(payload) == {"resourceSpans"}
        resource_spans = payload["resourceSpans"]
        assert len(resource_spans) == 1
        rs = resource_spans[0]
        assert rs["resource"]["attributes"] == [
            {"key": "service.name", "value": {"stringValue": "godspeed"}}
        ]
        scope_spans = rs["scopeSpans"]
        assert len(scope_spans) == 1
        assert scope_spans[0]["scope"] == {"name": "godspeed"}
        spans = scope_spans[0]["spans"]
        assert len(spans) == 1
        span = spans[0]
        assert span["traceId"] == "abc123"  # 0x stripped, lowercased
        assert span["spanId"] == "def456"  # lowercased
        assert span["parentSpanId"] == ""  # None -> empty string
        assert span["name"] == "session"
        assert span["kind"] == 1  # internal
        assert span["startTimeUnixNano"] == "1700000000000000000"
        assert span["endTimeUnixNano"] == "1700000001500000000"
        assert span["status"] == {"code": 1}  # OK
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        assert attrs["session.id"] == {"stringValue": "sess-1"}
        assert attrs["input_tokens"] == {"intValue": "100"}
        assert attrs["cost_usd"] == {"doubleValue": 0.5}
        assert attrs["is_active"] == {"boolValue": True}

    def test_error_status_code(self) -> None:
        span = _span()
        span = Span(
            name=span.name,
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            start_time=span.start_time,
            end_time=span.end_time,
            status=SpanStatus.ERROR,
            attributes=span.attributes,
        )
        payload = spans_to_otlp_json([span])
        assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"] == {"code": 2}

    def test_parent_span_id_preserved(self) -> None:
        span = _span()
        span = Span(
            name=span.name,
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id="0xPARENT",
            start_time=span.start_time,
            end_time=span.end_time,
            status=span.status,
            attributes=span.attributes,
        )
        payload = spans_to_otlp_json([span])
        assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["parentSpanId"] == "parent"


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TestExportOtlp:
    def test_empty_endpoint_is_noop(self) -> None:
        result = export_otlp([_span()], "")
        assert result == ExportResult(ok=False, error="no endpoint configured")

    def test_success_posts_to_v1_traces(self) -> None:
        with patch(
            "godspeed.observability.otlp.urllib.request.urlopen",
            return_value=_FakeResponse(200),
        ) as mock_urlopen:
            result = export_otlp([_span()], "http://localhost:4318")
        assert result.ok is True
        assert result.status == 200
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:4318/v1/traces"
        assert request.method == "POST"
        assert request.get_header("Content-type") == "application/json"
        body = json.loads(request.data)
        assert "resourceSpans" in body

    def test_trailing_slash_endpoint(self) -> None:
        with patch(
            "godspeed.observability.otlp.urllib.request.urlopen",
            return_value=_FakeResponse(204),
        ) as mock_urlopen:
            result = export_otlp([_span()], "http://localhost:4318/")
        assert result.ok is True
        assert mock_urlopen.call_args.args[0].full_url == "http://localhost:4318/v1/traces"

    def test_http_error_returns_result(self) -> None:
        exc = urllib.error.HTTPError("http://x/v1/traces", 500, "boom", {}, None)
        with patch("godspeed.observability.otlp.urllib.request.urlopen", side_effect=exc):
            result = export_otlp([_span()], "http://localhost:4318")
        assert result.ok is False
        assert result.status == 500

    def test_url_error_returns_result(self) -> None:
        exc = urllib.error.URLError("connection refused")
        with patch("godspeed.observability.otlp.urllib.request.urlopen", side_effect=exc):
            result = export_otlp([_span()], "http://localhost:4318")
        assert result.ok is False
        assert result.error is not None

    def test_timeout_returns_result(self) -> None:
        with patch(
            "godspeed.observability.otlp.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = export_otlp([_span()], "http://localhost:4318")
        assert result.ok is False
        assert "timed out" in (result.error or "")


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
    return Commands(
        conversation=conversation,
        llm_client=llm_client,
        permission_engine=None,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
        tool_registry=None,
    )


class TestMetricsCommand:
    def test_snapshot_renders(self, commands: Commands) -> None:
        out = _capture(commands.dispatch, "/metrics")
        assert "Session Metrics" in out
        assert "Tool Calls Total" in out
        assert "0" in out

    def test_export_without_endpoint_shows_info(self, commands: Commands) -> None:
        out = _capture(commands.dispatch, "/metrics export")
        assert "No export endpoint configured" in out

    def test_export_success(self, commands: Commands) -> None:
        with patch(
            "godspeed.tui.commands.export_otlp",
            return_value=ExportResult(ok=True, status=200),
        ):
            out = _capture(commands.dispatch, "/metrics export http://localhost:4318")
        assert "Exported session span" in out
        assert "HTTP 200" in out

    def test_export_failure(self, commands: Commands) -> None:
        with patch(
            "godspeed.tui.commands.export_otlp",
            return_value=ExportResult(ok=False, error="boom"),
        ):
            out = _capture(commands.dispatch, "/metrics export http://localhost:4318")
        assert "Export failed: boom" in out

    def test_unknown_subcommand_errors(self, commands: Commands) -> None:
        out = _capture(commands.dispatch, "/metrics bogus")
        assert "Unknown subcommand: bogus" in out
