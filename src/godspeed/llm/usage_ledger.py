"""Per-call LLM usage ledger for token/cost attribution.

Records one row per completed LLM call, tagged with the task type used
for model routing and (when known) the sub-agent that made the call.
Attribution never fabricates data: rows without a sub-agent identity
aggregate under the ``"parent"`` key.

Sub-agent resolution order when ``record()`` has no explicit
``subagent_id``:

1. The ledger's ``default_subagent_id`` (set when a coordinator builds a
   dedicated client for a spawn).
2. The ``subagent_context`` contextvar (set when a coordinator shares the
   parent client and tags the spawn scope instead).

No locks: rows are only appended from coroutines on the single asyncio
event loop they belong to.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass

# Contextvar tagging the sub-agent id of the spawn currently in scope.
# ``None`` means the call is not inside a sub-agent spawn (i.e. parent).
_SUBAGENT_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "godspeed_usage_subagent_id", default=None
)

# Aggregation key for rows recorded without a sub-agent identity.
PARENT_KEY = "parent"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """A single completed LLM call."""

    task_type: str = "default"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    subagent_id: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """Aggregated totals for one bucket (task type or sub-agent)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, entry: LedgerEntry) -> LedgerRow:
        """Return a merged row including the given entry."""
        return LedgerRow(
            calls=self.calls + 1,
            input_tokens=self.input_tokens + entry.input_tokens,
            output_tokens=self.output_tokens + entry.output_tokens,
            cost_usd=self.cost_usd + entry.cost_usd,
        )


@contextlib.contextmanager
def subagent_context(subagent_id: str):
    """Tag all ledger ``record()`` calls inside the block with *subagent_id*."""
    token = _SUBAGENT_ID.set(subagent_id)
    try:
        yield
    finally:
        _SUBAGENT_ID.reset(token)


def _resolve_subagent(explicit: str | None, default: str | None) -> str:
    if explicit is not None:
        return explicit
    resolved = default if default is not None else _SUBAGENT_ID.get()
    return PARENT_KEY if resolved is None else resolved


class UsageLedger:
    """Accumulates per-call usage rows for later reporting."""

    def __init__(self, default_subagent_id: str | None = None) -> None:
        self.default_subagent_id = default_subagent_id
        self._entries: list[LedgerEntry] = []
        self._by_task_type: dict[str, LedgerRow] = {}
        self._by_subagent: dict[str, LedgerRow] = {}

    def record(
        self,
        *,
        task_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        subagent_id: str | None = None,
    ) -> None:
        """Record one completed LLM call."""
        entry = LedgerEntry(
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            subagent_id=_resolve_subagent(subagent_id, self.default_subagent_id),
        )
        self._entries.append(entry)
        name = entry.task_type or "default"
        self._by_task_type[name] = self._by_task_type.get(name, LedgerRow()).add(entry)
        key = entry.subagent_id or PARENT_KEY
        self._by_subagent[key] = self._by_subagent.get(key, LedgerRow()).add(entry)

    @property
    def entries(self) -> list[LedgerEntry]:
        """All recorded entries in order (read-only copy)."""
        return list(self._entries)

    def by_task_type(self) -> dict[str, LedgerRow]:
        """Per-task-type aggregated rows."""
        return dict(self._by_task_type)

    def by_subagent(self) -> dict[str, LedgerRow]:
        """Per-sub-agent aggregated rows; untagged calls keyed ``"parent"``."""
        return dict(self._by_subagent)

    def totals(self) -> LedgerRow:
        """Totals across all recorded calls."""
        row = LedgerRow()
        for entry in self._entries:
            row = row.add(entry)
        return row

    def merge_from(self, other: UsageLedger) -> None:
        """Absorb another ledger's entries (e.g. a spawned child's usage)."""
        if other is self:
            return
        for entry in other._entries:
            self._entries.append(entry)
            name = entry.task_type or "default"
            self._by_task_type[name] = self._by_task_type.get(name, LedgerRow()).add(entry)
            key = entry.subagent_id or PARENT_KEY
            self._by_subagent[key] = self._by_subagent.get(key, LedgerRow()).add(entry)

    @classmethod
    def from_rows(cls, rows: list[dict[str, object]]) -> UsageLedger:
        """Rebuild a ledger from serialized entry dicts (tests/persistence)."""

        def _as_int(value: object) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        def _as_float(value: object) -> float:
            return float(value) if isinstance(value, (int, float)) else 0.0

        ledger = cls()
        for row in rows:
            raw_subagent = row.get("subagent_id")
            raw_task = row.get("task_type")
            ledger.record(
                task_type=raw_task if isinstance(raw_task, str) else "default",
                input_tokens=_as_int(row.get("input_tokens")),
                output_tokens=_as_int(row.get("output_tokens")),
                cost_usd=_as_float(row.get("cost_usd")),
                subagent_id=raw_subagent if isinstance(raw_subagent, str) else None,
            )
        return ledger
