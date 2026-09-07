"""Tests for OTel-style spans and trace_id emission to MetricsSink."""

from __future__ import annotations

import json
from pathlib import Path

from godspeed.observability.metrics import MetricsSink, SpanStatus, Tracer


class TestTracer:
    def test_span_emitted_with_trace_id(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsSink(path=path) as sink:
            tracer = Tracer(sink)
            with tracer.start_span("test.op", {"key": "value"}):
                pass

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "span"
        assert record["name"] == "test.op"
        assert record["trace_id"]
        assert record["span_id"]
        assert record["parent_span_id"] is None
        assert record["status"] == "unset"
        assert record["attributes"] == {"key": "value"}
        assert record["duration_ms"] >= 0

    def test_nested_spans_share_trace_id(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsSink(path=path) as sink:
            tracer = Tracer(sink)
            with tracer.start_span("root"):
                with tracer.start_span("child"):
                    pass

        lines = path.read_text().strip().splitlines()
        records = [json.loads(l) for l in lines]
        root = next(r for r in records if r["name"] == "root")
        child = next(r for r in records if r["name"] == "child")
        assert root["trace_id"] == child["trace_id"]
        assert child["parent_span_id"] == root["span_id"]

    def test_error_sets_status(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsSink(path=path) as sink:
            tracer = Tracer(sink)
            try:
                with tracer.start_span("failing"):
                    raise ValueError("boom")
            except ValueError:
                pass

        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["status"] == "error"

    def test_explicit_status(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsSink(path=path) as sink:
            tracer = Tracer(sink)
            with tracer.start_span("ok") as span:
                span.set_status(SpanStatus.OK)

        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["status"] == "ok"


class TestMetricsSinkTraceId:
    def test_emit_with_trace_id(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsSink(path=path) as sink:
            sink.emit("loop", {"iterations": 1}, trace_id="abc123")

        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["trace_id"] == "abc123"
        assert record["iterations"] == 1

    def test_emit_without_trace_id(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsSink(path=path) as sink:
            sink.emit("loop", {"iterations": 1})

        record = json.loads(path.read_text().strip().splitlines()[0])
        assert "trace_id" not in record
