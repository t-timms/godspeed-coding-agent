"""Typed audit event models following the IETF Agent Audit Trail pattern."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AuditEventType(StrEnum):
    """Types of auditable events."""

    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    PERMISSION_CHECK = "permission_check"
    PERMISSION_GRANT = "permission_grant"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    COMPACTION = "compaction"
    ERROR = "error"
    SECRET_REDACTED = "secret_redacted"  # noqa: S105


class AuditRecord(BaseModel):
    """A single audit event record.

    Hash-chained: each record includes the SHA-256 hash of the previous
    record's canonical JSON, creating a tamper-evident chain.
    """

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    session_id: str
    sequence: int = 0
    action_type: AuditEventType
    action_detail: dict[str, Any] = Field(default_factory=dict)
    outcome: str = "success"  # "success" | "denied" | "error" | "timeout"
    prev_hash: str = ""  # SHA-256 of previous record (empty for first record)
    record_hash: str = ""  # Computed after creation
    parent_session_id: str | None = None  # set when session forked from another
    is_sidechain: bool = False  # True for sub-agent transcripts
    cost_usd: float | None = None
    lines_added: int | None = None
    lines_removed: int | None = None
