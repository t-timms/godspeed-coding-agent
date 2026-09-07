"""Acceptance-criteria tracker — file-backed contract of passing/failing items.

The agent declares acceptance criteria up front (``acceptance_init``), then
marks each item passing with cited evidence (``acceptance_update``). The
pre-completion gate treats items still ``failing`` exactly like open tasks,
so the model cannot stop while acceptance criteria remain unmet.

Persistence: JSON at ``.godspeed/acceptance.json`` under the tool context's
cwd. All writes stay inside ``.godspeed/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

ACCEPTANCE_FILENAME = "acceptance.json"
ACCEPTANCE_DIRNAME = ".godspeed"

_PASSING_STATUS = "passing"
_FAILING_STATUS = "failing"
_VALID_STATUSES: frozenset[str] = frozenset({_PASSING_STATUS, _FAILING_STATUS})


@dataclass
class AcceptanceItem:
    """A single acceptance criterion."""

    id: int
    title: str
    status: str = _FAILING_STATUS
    evidence: str | None = None


@dataclass
class AcceptanceContract:
    """File-backed acceptance-criteria tracker.

    Items start ``failing`` with no evidence. ``update`` requires a non-empty
    evidence string to flip an item to ``passing`` — the model must cite the
    test/verify output that proves the criterion.
    """

    items: list[AcceptanceItem] = field(default_factory=list)
    _next_id: int = 1

    @classmethod
    def from_titles(cls, titles: list[str]) -> AcceptanceContract:
        """Build a fresh contract with all items starting ``failing``."""
        contract = cls()
        for title in titles:
            contract.items.append(
                AcceptanceItem(id=contract._next_id, title=title, status=_FAILING_STATUS)
            )
            contract._next_id += 1
        return contract

    @classmethod
    def load(cls, path: Path) -> AcceptanceContract:
        """Load a contract from JSON. Returns an empty contract on missing/corrupt file."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Acceptance contract unreadable path=%s error=%s", path, exc)
            return cls()
        contract = cls()
        for raw in data.get("items", []):
            item = AcceptanceItem(
                id=int(raw["id"]),
                title=str(raw["title"]),
                status=str(raw.get("status", _FAILING_STATUS)),
                evidence=raw.get("evidence"),
            )
            contract.items.append(item)
            contract._next_id = max(contract._next_id, item.id + 1)
        return contract

    def save(self, path: Path) -> None:
        """Persist the contract as JSON, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "status": item.status,
                    "evidence": item.evidence,
                }
                for item in self.items
            ]
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def get(self, item_id: int) -> AcceptanceItem | None:
        """Get an item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def update(
        self, item_id: int, status: str, evidence: str | None = None
    ) -> AcceptanceItem | None:
        """Set an item's status. Returns None if the item is not found.

        Passing requires a non-empty evidence string; failing is always
        allowed (evidence is cleared).
        """
        item = self.get(item_id)
        if item is None:
            return None
        if status == _PASSING_STATUS and not (evidence or "").strip():
            raise ValueError(
                "Passing an acceptance item requires evidence. "
                "Cite the test/verify output that proves the criterion."
            )
        item.status = status
        item.evidence = evidence.strip() if evidence else None
        return item

    def failing_items(self) -> list[AcceptanceItem]:
        """Items still failing (not yet accepted)."""
        return [item for item in self.items if item.status != _PASSING_STATUS]

    def format_active(self) -> str | None:
        """Format failing items for system-prompt injection.

        Returns None when every item passes (or the contract is empty).
        """
        failing = self.failing_items()
        if not failing:
            return None
        lines = []
        for item in failing:
            lines.append(f"  ❌ [{item.id}] {item.title} (failing)")
        return "\n".join(lines)

    def format_status(self) -> str:
        """Render a compact table of items + statuses + evidence presence."""
        if not self.items:
            return "No acceptance criteria defined."
        lines = []
        for item in self.items:
            evidence = "evidence ✓" if item.evidence else "no evidence"
            lines.append(f"[{item.id}] {item.title} — {item.status} ({evidence})")
        return "\n".join(lines)


def _contract_path(context: ToolContext) -> Path:
    """Resolve the contract file path from a tool context."""
    return context.cwd / ACCEPTANCE_DIRNAME / ACCEPTANCE_FILENAME


class AcceptanceInitTool(Tool):
    """Create or overwrite the acceptance contract from a list of titles."""

    @property
    def name(self) -> str:
        return "acceptance_init"

    @property
    def description(self) -> str:
        return (
            "Create or overwrite the acceptance-criteria contract. "
            "Provide the list of criteria that must be met before the task "
            "is considered done. All items start as failing."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Acceptance criteria titles.",
                },
            },
            "required": ["titles"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        titles = arguments.get("titles", [])
        if not isinstance(titles, list) or not titles:
            return ToolResult.failure("titles must be a non-empty list of strings")
        cleaned = [str(t).strip() for t in titles if str(t).strip()]
        if not cleaned:
            return ToolResult.failure("titles must be a non-empty list of strings")
        contract = AcceptanceContract.from_titles(cleaned)
        contract.save(_contract_path(context))
        logger.info("Acceptance contract initialized items=%d", len(cleaned))
        return ToolResult.ok(
            f"Acceptance contract initialized with {len(cleaned)} criteria, all failing."
        )


class AcceptanceUpdateTool(Tool):
    """Set an acceptance item's status. Passing requires cited evidence."""

    def __init__(self) -> None:
        self._contract: AcceptanceContract | None = None

    @property
    def name(self) -> str:
        return "acceptance_update"

    @property
    def description(self) -> str:
        return (
            "Update an acceptance item's status. To mark an item passing you "
            "MUST provide the evidence string citing the test/verify output "
            "that proves the criterion. Failing is always allowed."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "integer",
                    "description": "Acceptance item ID.",
                },
                "status": {
                    "type": "string",
                    "enum": ["failing", "passing"],
                    "description": "New status.",
                },
                "evidence": {
                    "type": "string",
                    "description": "Required for passing: cite the test/verify output.",
                },
            },
            "required": ["item_id", "status"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        item_id = arguments.get("item_id")
        status = arguments.get("status")
        if item_id is None or status not in _VALID_STATUSES:
            return ToolResult.failure("item_id and status (failing|passing) are required")
        evidence = arguments.get("evidence")
        contract = self._load_contract(context)
        try:
            item = contract.update(int(item_id), status, evidence)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if item is None:
            return ToolResult.failure(f"Acceptance item {item_id} not found")
        contract.save(_contract_path(context))
        return ToolResult.ok(f"Updated acceptance item [{item.id}]: {item.title} → {item.status}")

    def _load_contract(self, context: ToolContext) -> AcceptanceContract:
        """Load the contract, caching it for the session."""
        if self._contract is None:
            self._contract = AcceptanceContract.load(_contract_path(context))
        return self._contract


class AcceptanceStatusTool(Tool):
    """Render the current acceptance contract as a compact table."""

    @property
    def name(self) -> str:
        return "acceptance_status"

    @property
    def description(self) -> str:
        return (
            "Show the current acceptance contract: each item, its status, "
            "and whether evidence has been cited."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        contract = AcceptanceContract.load(_contract_path(context))
        return ToolResult.ok(contract.format_status())
