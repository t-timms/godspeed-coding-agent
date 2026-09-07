"""OTLP/HTTP-JSON trace export for Godspeed spans.

Stdlib-only: builds the OTLP ``/v1/traces`` JSON payload shape and posts
it with ``urllib``. No opentelemetry dependency. Every failure path
returns an :class:`ExportResult` — this module never raises.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from godspeed.observability.metrics import Span, SpanStatus

logger = logging.getLogger(__name__)

_SERVICE_NAME = "godspeed"
_SCOPE_NAME = "godspeed"
_SPAN_KIND_INTERNAL = 1

_STATUS_CODES: dict[SpanStatus, int] = {
    SpanStatus.UNSET: 0,
    SpanStatus.OK: 1,
    SpanStatus.ERROR: 2,
}


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Outcome of an OTLP export attempt."""

    ok: bool
    status: int | None = None
    error: str | None = None


def _normalize_hex(value: str) -> str:
    """Normalize a hex id: lowercase, no ``0x`` prefix."""
    return value.lower().removeprefix("0x")


def _attribute_to_otlp(key: str, value: Any) -> dict[str, Any]:
    """Convert one attribute to the OTLP ``{key, value}`` shape."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _span_to_otlp(span: Span) -> dict[str, Any]:
    """Convert one :class:`Span` to the OTLP span shape."""
    return {
        "traceId": _normalize_hex(span.trace_id),
        "spanId": _normalize_hex(span.span_id),
        "parentSpanId": _normalize_hex(span.parent_span_id) if span.parent_span_id else "",
        "name": span.name,
        "kind": _SPAN_KIND_INTERNAL,
        "startTimeUnixNano": str(int(span.start_time * 1_000_000_000)),
        "endTimeUnixNano": str(int(span.end_time * 1_000_000_000)),
        "attributes": [_attribute_to_otlp(k, v) for k, v in span.attributes.items()],
        "status": {"code": _STATUS_CODES[span.status]},
    }


def spans_to_otlp_json(spans: list[Span]) -> dict[str, Any]:
    """Build the OTLP/HTTP-JSON ``/v1/traces`` request payload.

    Shape: ``resourceSpans -> scopeSpans -> spans`` with hex-encoded
    trace/span ids, nanosecond timestamps as strings, and attributes as
    ``[{key, value}]`` pairs.
    """
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": _SERVICE_NAME}}]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": _SCOPE_NAME},
                        "spans": [_span_to_otlp(s) for s in spans],
                    }
                ],
            }
        ]
    }


def export_otlp(
    spans: list[Span],
    endpoint: str,
    timeout: float = 5,
) -> ExportResult:
    """POST spans to ``{endpoint}/v1/traces`` as OTLP/HTTP-JSON.

    Never raises: every failure (empty endpoint, network error, HTTP
    error, timeout) is returned as an :class:`ExportResult`.
    """
    if not endpoint:
        return ExportResult(ok=False, error="no endpoint configured")

    payload = spans_to_otlp_json(spans)
    url = f"{endpoint.rstrip('/')}/v1/traces"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = response.status
            return ExportResult(ok=200 <= status < 300, status=status)
    except urllib.error.HTTPError as exc:
        logger.warning("OTLP export HTTP error status=%s url=%s", exc.code, url)
        return ExportResult(ok=False, status=exc.code, error=str(exc))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("OTLP export failed url=%s error=%s", url, exc)
        return ExportResult(ok=False, error=str(exc))
