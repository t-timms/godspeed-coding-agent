"""Pure, unit-testable builders for session usage reports.

Aggregates what the codebase actually measures — nothing more. The LLM
client tracks session totals (``total_input_tokens``,
``total_output_tokens``, ``total_cost_usd``); when a ``UsageLedger``,
is attached, per-task-type and per-subagent rows are filled from real
tagged records. The audit trail records ``tool_call`` events with a
``tool`` name, giving per-tool call counts.

No data is ever fabricated: dimensions without a measurable source are
left empty and callers are expected to state that gap honestly.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from godspeed.audit.events import AuditEventType, AuditRecord

logger = logging.getLogger(__name__)

# Audit detail key that carries the tool name on ``tool_call`` records.
_TOOL_DETAIL_KEY = "tool"


@dataclass(frozen=True, slots=True)
class TokenRow:
    """Token/cost totals for a single task type."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class ToolRow:
    """Call count and (where measurable) token attribution for one tool."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class SubagentRow:
    """Invocation count and (where measurable) token/cost share for one subagent."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class UsageReport:
    """Aggregated session usage across all measurable dimensions."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_task_type: dict[str, TokenRow] = field(default_factory=dict)
    by_tool: dict[str, ToolRow] = field(default_factory=dict)
    by_subagent: dict[str, SubagentRow] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across the session."""
        return self.total_input_tokens + self.total_output_tokens


def from_client(llm_client: Any, ledger: Any | None = None) -> UsageReport:
    """Build a report from an LLM client's session totals.

    Reads only the usage attributes the client actually exposes
    (``total_input_tokens``, ``total_output_tokens``, ``total_cost_usd``).
    When a ``UsageLedger`` is available, the per-task-type and
    per-sub-agent dimensions are filled from real recorded rows; ledger
    dimensions without recorded calls stay empty rather than fabricated.
    """
    report = UsageReport(
        total_input_tokens=int(getattr(llm_client, "total_input_tokens", 0) or 0),
        total_output_tokens=int(getattr(llm_client, "total_output_tokens", 0) or 0),
        total_cost_usd=float(getattr(llm_client, "total_cost_usd", 0.0) or 0.0),
    )
    if ledger is None:
        return report
    if hasattr(ledger, "by_task_type"):
        for name, row in ledger.by_task_type().items():
            report.by_task_type[name] = TokenRow(
                calls=row.calls,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cost_usd=row.cost_usd,
            )
    if hasattr(ledger, "by_subagent"):
        for name, row in ledger.by_subagent().items():
            report.by_subagent[name] = SubagentRow(
                calls=row.calls,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cost_usd=row.cost_usd,
            )
    return report


def _read_audit_records(log_path: Path) -> list[AuditRecord]:
    """Read and parse all audit records from a (possibly gzipped) JSONL file.

    Malformed lines are skipped with a debug log — a single corrupt line
    must not abort the whole report.
    """
    records: list[AuditRecord] = []
    try:
        if str(log_path).endswith(".gz"):
            _open = gzip.open(log_path, "rt", encoding="utf-8")  # noqa: SIM115
        else:
            _open = open(log_path, encoding="utf-8")  # noqa: SIM115
        with _open as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(AuditRecord.model_validate(data))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.debug(
                        "Skipping malformed audit record path=%s error=%s",
                        log_path,
                        exc,
                    )
    except (OSError, gzip.BadGzipFile) as exc:
        logger.warning("Failed to read audit file path=%s error=%s", log_path, exc)
    return records


def from_audit(trail: Any) -> UsageReport:
    """Build a report from an audit trail's recorded events.

    Counts ``tool_call`` events per tool (from the ``tool`` detail key) and
    sub-agent sidechain records (``is_sidechain=True``) per session. The
    audit trail does not carry token or cost attribution, so those fields
    remain zero — only call counts are measurable here.

    Returns an empty report when the trail is unavailable or its log file
    cannot be read.
    """
    log_path = getattr(trail, "log_path", None)
    if log_path is None or not Path(log_path).exists():
        return UsageReport()

    by_tool: dict[str, ToolRow] = {}
    by_subagent: dict[str, SubagentRow] = {}

    for rec in _read_audit_records(Path(log_path)):
        if rec.action_type == AuditEventType.TOOL_CALL:
            tool_name = rec.action_detail.get(_TOOL_DETAIL_KEY, "unknown")
            tool_row = by_tool.setdefault(tool_name, ToolRow())
            by_tool[tool_name] = ToolRow(
                calls=tool_row.calls + 1,
                input_tokens=tool_row.input_tokens,
                output_tokens=tool_row.output_tokens,
                cost_usd=tool_row.cost_usd,
            )
        elif rec.is_sidechain:
            agent_name = rec.session_id or "unknown"
            agent_row = by_subagent.setdefault(agent_name, SubagentRow())
            by_subagent[agent_name] = SubagentRow(
                calls=agent_row.calls + 1,
                input_tokens=agent_row.input_tokens,
                output_tokens=agent_row.output_tokens,
                cost_usd=agent_row.cost_usd,
            )

    return UsageReport(by_tool=by_tool, by_subagent=by_subagent)
