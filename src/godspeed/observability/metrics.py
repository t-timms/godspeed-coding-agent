"""Lightweight structured metrics for the agent loop.

Emits JSONL lines that can be scraped by external collectors. No external
dependencies — everything is stdlib.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTOGRAM_BUCKETS = 100

_span_stack: contextvars.ContextVar[tuple[tuple[str, str], ...]] = contextvars.ContextVar(
    "godspeed_span_stack",
    default=(),
)


class SpanStatus(enum.StrEnum):
    """OTel-compatible span status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Span:
    """A single OTel-style span record.

    Stdlib-only: no opentelemetry dependency. Emitted to the MetricsSink
    as a JSONL line with a ``trace_id`` so spans from one logical run can
    be correlated across events.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time: float
    end_time: float
    status: SpanStatus
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Span duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON emission."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status.value,
            "attributes": self.attributes,
        }


class Tracer:
    """Minimal stdlib-only tracer that emits OTel-style spans.

    Provides a context-manager ``start_span`` that records start/end times
    and writes the completed span to a :class:`MetricsSink`. A ``trace_id``
    is generated per root span and propagated to children so a logical run
    is traceable across all emitted events.
    """

    def __init__(self, sink: MetricsSink) -> None:
        self._sink = sink

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> _SpanContext:
        """Start a new span, returning a context manager."""
        stack = _span_stack.get()
        if stack:
            trace_id, parent_span_id = stack[-1]
        else:
            trace_id = uuid.uuid4().hex
            parent_span_id = None
        span_id = uuid.uuid4().hex
        return _SpanContext(
            tracer=self,
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )

    def _enter(self, trace_id: str, span_id: str) -> None:
        _span_stack.set((*_span_stack.get(), (trace_id, span_id)))

    def _exit(self, trace_id: str, span_id: str) -> None:
        stack = _span_stack.get()
        if stack and stack[-1] == (trace_id, span_id):
            _span_stack.set(stack[:-1])

    def _emit(self, span: Span) -> None:
        self._sink.emit("span", span.to_dict())


class _SpanContext:
    """Context manager returned by :meth:`Tracer.start_span`."""

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        attributes: dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._trace_id = trace_id
        self._span_id = span_id
        self._parent_span_id = parent_span_id
        self._attributes = attributes
        self._start_time: float | None = None
        self._status = SpanStatus.UNSET

    def set_status(self, status: SpanStatus) -> None:
        """Set the span status before exiting."""
        self._status = status

    def set_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the span."""
        self._attributes[key] = value

    def __enter__(self) -> _SpanContext:
        self._start_time = time.time()
        self._tracer._enter(self._trace_id, self._span_id)
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        if exc_type is not None:
            self._status = SpanStatus.ERROR
        self._tracer._exit(self._trace_id, self._span_id)
        span = Span(
            name=self._name,
            trace_id=self._trace_id,
            span_id=self._span_id,
            parent_span_id=self._parent_span_id,
            start_time=self._start_time or time.time(),
            end_time=time.time(),
            status=self._status,
            attributes=self._attributes,
        )
        self._tracer._emit(span)


class AlertSeverity(enum.StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Alert:
    """A single threshold breach alert."""

    severity: AlertSeverity
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MetricsThresholds:
    """Configurable threshold levels for LoopMetrics.

    All thresholds are optional — None disables the check.
    """

    token_velocity_warning: float | None = 10_000.0  # tokens/min
    token_velocity_critical: float | None = 50_000.0
    error_rate_warning: float | None = 0.10  # 10% of tool calls failing
    error_rate_critical: float | None = 0.25
    llm_latency_p99_warning_ms: float | None = 30_000.0
    llm_latency_p99_critical_ms: float | None = 60_000.0
    loop_duration_p99_warning_ms: float | None = 5_000.0
    loop_duration_p99_critical_ms: float | None = 15_000.0
    speculative_hit_rate_warning: float | None = 0.30  # below 30% is wasteful


def check_thresholds(
    metrics: LoopMetrics,
    thresholds: MetricsThresholds | None = None,
) -> list[Alert]:
    """Evaluate *metrics* against *thresholds* and return triggered alerts.

    Alerts are ordered by severity (critical first) then metric name.
    """
    if thresholds is None:
        thresholds = MetricsThresholds()

    alerts: list[Alert] = []

    # Token velocity
    velocity = metrics.token_velocity_per_min
    if velocity is not None:
        tv_crit = thresholds.token_velocity_critical
        tv_warn = thresholds.token_velocity_warning
        if tv_crit is not None and velocity > tv_crit:
            alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric="token_velocity_per_min",
                    value=velocity,
                    threshold=tv_crit,
                    message=(
                        f"Token velocity {velocity:,.0f}/min exceeds "
                        f"critical threshold {tv_crit:,.0f}/min"
                    ),
                )
            )
        elif tv_warn is not None and velocity > tv_warn:
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    metric="token_velocity_per_min",
                    value=velocity,
                    threshold=tv_warn,
                    message=(
                        f"Token velocity {velocity:,.0f}/min exceeds "
                        f"warning threshold {tv_warn:,.0f}/min"
                    ),
                )
            )

    # Error rate
    if metrics.tool_calls_total > 0:
        error_rate = metrics.tool_errors_total / metrics.tool_calls_total
        er_crit = thresholds.error_rate_critical
        er_warn = thresholds.error_rate_warning
        if er_crit is not None and error_rate > er_crit:
            alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric="error_rate",
                    value=error_rate,
                    threshold=er_crit,
                    message=(
                        f"Tool error rate {error_rate:.1%} exceeds critical threshold {er_crit:.1%}"
                    ),
                )
            )
        elif er_warn is not None and error_rate > er_warn:
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    metric="error_rate",
                    value=error_rate,
                    threshold=er_warn,
                    message=(
                        f"Tool error rate {error_rate:.1%} exceeds warning threshold {er_warn:.1%}"
                    ),
                )
            )

    # LLM latency p99
    llm_p99 = metrics._llm_latency_ms.p99
    if llm_p99 is not None:
        if (
            thresholds.llm_latency_p99_critical_ms is not None
            and llm_p99 > thresholds.llm_latency_p99_critical_ms
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric="llm_latency_p99_ms",
                    value=llm_p99,
                    threshold=thresholds.llm_latency_p99_critical_ms,
                    message=(
                        f"LLM p99 latency {llm_p99:,.0f}ms exceeds "
                        f"critical threshold {thresholds.llm_latency_p99_critical_ms:,.0f}ms"
                    ),
                )
            )
        elif (
            thresholds.llm_latency_p99_warning_ms is not None
            and llm_p99 > thresholds.llm_latency_p99_warning_ms
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    metric="llm_latency_p99_ms",
                    value=llm_p99,
                    threshold=thresholds.llm_latency_p99_warning_ms,
                    message=(
                        f"LLM p99 latency {llm_p99:,.0f}ms exceeds "
                        f"warning threshold {thresholds.llm_latency_p99_warning_ms:,.0f}ms"
                    ),
                )
            )

    # Loop duration p99
    loop_p99 = metrics._loop_duration_ms.p99
    if loop_p99 is not None:
        if (
            thresholds.loop_duration_p99_critical_ms is not None
            and loop_p99 > thresholds.loop_duration_p99_critical_ms
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric="loop_duration_p99_ms",
                    value=loop_p99,
                    threshold=thresholds.loop_duration_p99_critical_ms,
                    message=(
                        f"Loop p99 duration {loop_p99:,.0f}ms exceeds "
                        f"critical threshold {thresholds.loop_duration_p99_critical_ms:,.0f}ms"
                    ),
                )
            )
        elif (
            thresholds.loop_duration_p99_warning_ms is not None
            and loop_p99 > thresholds.loop_duration_p99_warning_ms
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    metric="loop_duration_p99_ms",
                    value=loop_p99,
                    threshold=thresholds.loop_duration_p99_warning_ms,
                    message=(
                        f"Loop p99 duration {loop_p99:,.0f}ms exceeds "
                        f"warning threshold {thresholds.loop_duration_p99_warning_ms:,.0f}ms"
                    ),
                )
            )

    # Speculative hit rate (low is bad)
    if metrics.speculative_hits + metrics.speculative_misses > 0:
        hit_rate = metrics.speculative_hit_rate
        if (
            thresholds.speculative_hit_rate_warning is not None
            and hit_rate < thresholds.speculative_hit_rate_warning
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    metric="speculative_hit_rate",
                    value=hit_rate,
                    threshold=thresholds.speculative_hit_rate_warning,
                    message=(
                        f"Speculative hit rate {hit_rate:.1%} below "
                        f"warning threshold {thresholds.speculative_hit_rate_warning:.1%}"
                    ),
                )
            )

    # Sort: critical first, then by metric name
    alerts.sort(key=lambda a: (0 if a.severity == AlertSeverity.CRITICAL else 1, a.metric))
    return alerts


@dataclass(slots=True)
class _Histogram:
    """Rolling histogram with fixed-size buckets."""

    buckets: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_HISTOGRAM_BUCKETS))

    def observe(self, value: float) -> None:
        self.buckets.append(value)

    @property
    def count(self) -> int:
        return len(self.buckets)

    @property
    def p50(self) -> float | None:
        if not self.buckets:
            return None
        return sorted(self.buckets)[len(self.buckets) // 2]

    @property
    def p99(self) -> float | None:
        if not self.buckets:
            return None
        s = sorted(self.buckets)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def mean(self) -> float | None:
        if not self.buckets:
            return None
        return sum(self.buckets) / len(self.buckets)


@dataclass
class LoopMetrics:
    """Session-level metrics accumulator with histograms and counters.

    Designed to be cheap to update on every loop iteration — all operations
    are O(1) except percentile reads (which sort at most 100 elements).
    """

    # Counters
    iterations: int = 0
    tool_calls_total: int = 0
    tool_errors_total: int = 0
    tool_denials_total: int = 0
    speculative_hits: int = 0
    speculative_misses: int = 0
    must_fix_injections: int = 0
    compactions: int = 0

    # Timing histograms (milliseconds)
    _loop_duration_ms: _Histogram = field(default_factory=_Histogram)
    _tool_latency_ms: _Histogram = field(default_factory=_Histogram)
    _llm_latency_ms: _Histogram = field(default_factory=_Histogram)

    # Token tracking
    _token_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=60))
    _last_token_time: float = field(default_factory=time.monotonic)

    # State
    start_time: float = field(default_factory=time.monotonic)

    def record_iteration(self, duration_sec: float) -> None:
        self.iterations += 1
        self._loop_duration_ms.observe(duration_sec * 1000)

    def record_tool_call(self, name: str, duration_sec: float, is_error: bool = False) -> None:
        self.tool_calls_total += 1
        self._tool_latency_ms.observe(duration_sec * 1000)
        if is_error:
            self.tool_errors_total += 1

    def record_tool_denial(self) -> None:
        self.tool_denials_total += 1

    def record_speculative_hit(self) -> None:
        self.speculative_hits += 1

    def record_speculative_miss(self) -> None:
        self.speculative_misses += 1

    def record_llm_call(self, duration_sec: float) -> None:
        self._llm_latency_ms.observe(duration_sec * 1000)

    def record_token_count(self, tokens: int) -> None:
        now = time.monotonic()
        self._token_samples.append((now, tokens))
        self._last_token_time = now

    def record_compaction(self) -> None:
        self.compactions += 1

    def record_must_fix(self) -> None:
        self.must_fix_injections += 1

    @property
    def token_velocity_per_min(self) -> float | None:
        """Tokens per minute based on the last 60 samples."""
        if len(self._token_samples) < 2:
            return None
        first_t, first_n = self._token_samples[0]
        last_t, last_n = self._token_samples[-1]
        delta_t = last_t - first_t
        delta_n = last_n - first_n
        if delta_t <= 0:
            return None
        return (delta_n / delta_t) * 60

    @property
    def speculative_hit_rate(self) -> float:
        total = self.speculative_hits + self.speculative_misses
        if total == 0:
            return 0.0
        return self.speculative_hits / total

    @property
    def duration_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON emission."""
        return {
            "iterations": self.iterations,
            "tool_calls_total": self.tool_calls_total,
            "tool_errors_total": self.tool_errors_total,
            "tool_denials_total": self.tool_denials_total,
            "speculative_hits": self.speculative_hits,
            "speculative_misses": self.speculative_misses,
            "speculative_hit_rate": round(self.speculative_hit_rate, 3),
            "must_fix_injections": self.must_fix_injections,
            "compactions": self.compactions,
            "duration_seconds": round(self.duration_seconds, 2),
            "token_velocity_per_min": (
                round(v, 1) if (v := self.token_velocity_per_min) is not None else None
            ),
            "loop_duration_ms": {
                "count": self._loop_duration_ms.count,
                "p50": round(p, 2) if (p := self._loop_duration_ms.p50) is not None else None,
                "p99": round(p, 2) if (p := self._loop_duration_ms.p99) is not None else None,
                "mean": round(m, 2) if (m := self._loop_duration_ms.mean) is not None else None,
            },
            "tool_latency_ms": {
                "count": self._tool_latency_ms.count,
                "p50": round(p, 2) if (p := self._tool_latency_ms.p50) is not None else None,
                "p99": round(p, 2) if (p := self._tool_latency_ms.p99) is not None else None,
                "mean": round(m, 2) if (m := self._tool_latency_ms.mean) is not None else None,
            },
            "llm_latency_ms": {
                "count": self._llm_latency_ms.count,
                "p50": round(p, 2) if (p := self._llm_latency_ms.p50) is not None else None,
                "p99": round(p, 2) if (p := self._llm_latency_ms.p99) is not None else None,
                "mean": round(m, 2) if (m := self._llm_latency_ms.mean) is not None else None,
            },
        }


class MetricsSink:
    """JSONL metrics sink that writes to a file or stdout.

    Lightweight — no threading, no buffering beyond Python's file IO.
    Each call to emit() writes a single JSON line.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._file: Any | None = None

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        trace_id: str | None = None,
    ) -> None:
        record: dict[str, Any] = {"ts": time.time(), "event": event_type}
        if trace_id is not None:
            record["trace_id"] = trace_id
        record.update(payload)
        line = json.dumps(record, default=str)
        if self._path is not None:
            try:
                if self._file is None:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
                self._file.write(line + "\n")
            except OSError as exc:
                logger.debug("Metrics emit failed: %s", exc)
        else:
            logger.debug("metrics %s", line)

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    def __enter__(self) -> MetricsSink:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
