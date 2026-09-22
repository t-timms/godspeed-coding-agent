"""4-tier permission engine — the core security differentiator.

Evaluation order: deny > dangerous > session > allow > ask > default (risk level).
Deny rules always win. Dangerous command detection runs before session grants
so that user-approved patterns cannot bypass destructive command blocking.
Fail-closed: any ambiguity results in denial.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from godspeed.security.dangerous import detect_dangerous_command
from godspeed.security.plan_gate import PLAN_GATE_TOOL_NAME
from godspeed.security.rules import RuleAction, parse_rules
from godspeed.tools.base import RiskLevel, ToolCall

logger = logging.getLogger(__name__)


class PermissionDecision:
    """Result of a permission evaluation."""

    def __init__(
        self, action: str, reason: str = "", metadata: dict[str, object] | None = None
    ) -> None:
        self.action = action
        self.reason = reason
        # Purely additive, read-only context (e.g. a Laya advisory) attached
        # after the decision is made — never consulted by evaluate() itself.
        # __eq__ below still only compares .action, so nothing that does
        # `decision == "ask"` is affected by what's in here.
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.action == other
        if isinstance(other, PermissionDecision):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"PermissionDecision({self.action!r}, {self.reason!r})"


ALLOW = "allow"
DENY = "deny"
ASK = "ask"


def _extract_tool_prefix(pattern: str) -> str | None:
    """Extract the tool-name prefix from a rule pattern.

    Returns ``None`` for wildcard patterns like ``*(*)``. The prefix is
    normalized to snake_case via ``normalize_tool_name`` so documented
    CamelCase patterns (``FileWrite(*.env*)``) index under the registered
    name (``file_write``) and are found by the indexed lookup.
    """
    if pattern.startswith("*("):
        return None
    idx = pattern.find("(")
    if idx <= 0:
        return None
    from godspeed.tools.base import normalize_tool_name

    return normalize_tool_name(pattern[:idx])


MAX_PENDING = 50


def approval_fingerprint(tool_name: str, arguments: dict | None) -> str:
    """Stable sha256 fingerprint of a tool call for durable approvals.

    Includes the tool name and canonical (sorted-key) JSON of the
    arguments so an identical request after a restart matches.
    """
    canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
    return hashlib.sha256(f"{tool_name}\n{canonical}".encode()).hexdigest()


class PermissionEngine:
    """4-tier permission engine with deny-first evaluation.

    Tiers (by tool risk level):
    - READ_ONLY: auto-allowed, no prompt
    - LOW: ask once, then session-scoped allow
    - HIGH: ask every time (unless pattern-matched to allow)
    - DESTRUCTIVE: blocked by default, requires explicit allow rule

    Rule evaluation:
    1. Check deny rules — if any match, DENY
    2. Check dangerous command patterns — if detected, DENY
    3. Check session grants — if granted, ALLOW
    4. Check allow rules — if any match, ALLOW
    5. Check ask rules — if any match, ASK
    6. Fall back to risk-level default
    """

    def __init__(
        self,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        ask_patterns: list[str] | None = None,
        tool_risk_levels: dict[str, RiskLevel] | None = None,
        pending_dir: Path | None = None,
    ) -> None:
        self._deny_rules = parse_rules(deny_patterns or [], RuleAction.DENY)
        self._allow_rules = parse_rules(allow_patterns or [], RuleAction.ALLOW)
        self._ask_rules = parse_rules(ask_patterns or [], RuleAction.ASK)
        self._tool_risk_levels = tool_risk_levels or {}
        self._session_grants: dict[str, float] = {}
        self._grant_ttl: float = 3600.0  # 1 hour default
        self._grants_lock = threading.Lock()
        self.plan_mode: bool = False
        # Crash-durable pending approvals; None disables persistence (default).
        self._pending_dir = pending_dir

        # Build tool-name indexes for O(1) rule lookup. Wildcard rules
        # (e.g., "*(*)") are kept in a separate list checked every time.
        self._deny_index: dict[str, list] = {}
        self._deny_wildcards: list = []
        self._allow_index: dict[str, list] = {}
        self._allow_wildcards: list = []
        self._ask_index: dict[str, list] = {}
        self._ask_wildcards: list = []
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild rule indexes after rules change."""
        self._deny_index, self._deny_wildcards = self._build_index(self._deny_rules)
        self._allow_index, self._allow_wildcards = self._build_index(self._allow_rules)
        self._ask_index, self._ask_wildcards = self._build_index(self._ask_rules)

    @staticmethod
    def _build_index(rules: list) -> tuple[dict[str, list], list]:
        """Partition rules into per-tool-name index + wildcard list."""
        index: dict[str, list] = {}
        wildcards: list = []
        for rule in rules:
            prefix = _extract_tool_prefix(rule.pattern)
            if prefix is None:
                wildcards.append(rule)
            else:
                index.setdefault(prefix, []).append(rule)
        return index, wildcards

    def _rules_for_tool(self, index: dict[str, list], wildcards: list, tool_name: str) -> list:
        """Return rules that might match the given tool name."""
        from godspeed.tools.base import normalize_tool_name

        return index.get(normalize_tool_name(tool_name), []) + wildcards

    def evaluate(self, tool_call: ToolCall) -> PermissionDecision:
        """Evaluate a tool call against all rules.

        Returns a PermissionDecision with action and reason.
        """
        # Plan mode: block everything except READ_ONLY tools. The plan-exit
        # gate is explicitly exempt so the agent can always request approval
        # to leave plan mode, regardless of its declared risk level.
        if self.plan_mode and tool_call.tool_name != PLAN_GATE_TOOL_NAME:
            risk = self._tool_risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
            if risk != RiskLevel.READ_ONLY:
                return PermissionDecision(DENY, "Plan mode active — read-only tools only")

        formatted = tool_call.format_for_permission
        tool_name = tool_call.tool_name

        # 1. Deny rules first — always win (indexed lookup)
        for rule in self._rules_for_tool(self._deny_index, self._deny_wildcards, tool_name):
            if rule.matches(formatted):
                return PermissionDecision(DENY, f"Matched deny rule: {rule.pattern}")

        # 2. Dangerous command detection (for shell commands) — BEFORE session grants
        #    so that a session grant like "shell(npm *)" cannot bypass dangerous detection
        if tool_name.lower() in ("bash", "shell"):
            command = ""
            if isinstance(tool_call.arguments, dict):
                # Prefer the 'command' key — do NOT use first string value,
                # which could be a benign 'description' field
                command = tool_call.arguments.get("command", "")
                if not isinstance(command, str):
                    command = ""
            if command:
                try:
                    dangers = detect_dangerous_command(command)
                except Exception as exc:
                    # Security gate must fail closed — any error in dangerous
                    # command detection blocks the tool rather than allowing it
                    logger.error(
                        "Dangerous command detection failed — fail closed: %s",
                        exc,
                        exc_info=True,
                    )
                    return PermissionDecision(
                        DENY,
                        "Dangerous command check failed — fail closed",
                    )
                if dangers:
                    return PermissionDecision(
                        DENY,
                        f"Dangerous command detected: {', '.join(dangers)}",
                    )

        # 3. Session grants (user already approved this pattern)
        if self._check_session_grant(formatted):
            return PermissionDecision(ALLOW, "Session grant (time-limited)")

        # 4. Allow rules (indexed lookup)
        for rule in self._rules_for_tool(self._allow_index, self._allow_wildcards, tool_name):
            if rule.matches(formatted):
                return PermissionDecision(ALLOW, f"Matched allow rule: {rule.pattern}")

        # 5. Ask rules (indexed lookup)
        for rule in self._rules_for_tool(self._ask_index, self._ask_wildcards, tool_name):
            if rule.matches(formatted):
                return self._ask_or_replay(tool_call, f"Matched ask rule: {rule.pattern}")

        # 6. Default based on risk level
        risk = self._tool_risk_levels.get(tool_call.tool_name, RiskLevel.HIGH)
        decision = self._default_for_risk(risk)
        if decision == ASK:
            return self._ask_or_replay(tool_call, decision.reason)
        return decision

    def add_rule(self, pattern: str, action: str) -> None:
        """Add a pattern to the in-memory rule list at runtime.

        Used by the ``/remember`` slash command so a persisted rule
        takes effect immediately in the current session, not just on
        next restart.

        Args:
            pattern: ``Tool(glob)`` style pattern.
            action: ``"allow" | "deny" | "ask"``.
        """
        action_lc = action.lower()
        if action_lc == "allow":
            self._allow_rules.extend(parse_rules([pattern], RuleAction.ALLOW))
        elif action_lc == "deny":
            self._deny_rules.extend(parse_rules([pattern], RuleAction.DENY))
        elif action_lc == "ask":
            self._ask_rules.extend(parse_rules([pattern], RuleAction.ASK))
        else:
            msg = f"action must be 'allow' | 'deny' | 'ask', got {action!r}"
            raise ValueError(msg)
        self._rebuild_indexes()
        logger.info("Runtime rule added action=%s pattern=%s", action_lc, pattern)

    def grant_session_permission(self, pattern: str) -> None:
        """Grant a session-scoped permission for a pattern.

        Called when the user approves an ASK prompt. Thread-safe.
        """
        with self._grants_lock:
            self._session_grants[pattern] = time.monotonic()
        logger.info("Session permission granted pattern=%s ttl=%ds", pattern, int(self._grant_ttl))

    def grant_tool_session_permission(self, tool_name: str) -> None:
        """Grant a session-scoped permission for ALL calls to a tool.

        Stores ``ToolName(*)`` so any argument pattern matches.
        Used for LOW-risk tools where the user wants to approve once per session.
        Thread-safe.
        """
        pattern = f"{tool_name}(*)"
        with self._grants_lock:
            self._session_grants[pattern] = time.monotonic()
        logger.info(
            "Tool-level session grant tool=%s pattern=%s ttl=%ds",
            tool_name,
            pattern,
            int(self._grant_ttl),
        )

    def revoke_session_permission(self, pattern: str) -> None:
        """Revoke a single session-scoped permission. Thread-safe."""
        with self._grants_lock:
            self._session_grants.pop(pattern, None)

    def revoke_session_permissions(self) -> None:
        """Revoke all session-scoped permissions. Thread-safe."""
        with self._grants_lock:
            self._session_grants.clear()

    @property
    def deny_rules(self) -> list:
        """Read-only access to deny rules."""
        return list(self._deny_rules)

    @property
    def allow_rules(self) -> list:
        """Read-only access to allow rules."""
        return list(self._allow_rules)

    @property
    def ask_rules(self) -> list:
        """Read-only access to ask rules."""
        return list(self._ask_rules)

    @property
    def session_grants(self) -> dict[str, float]:
        """Read-only copy of active session grants. Thread-safe."""
        with self._grants_lock:
            return dict(self._session_grants)

    def _check_session_grant(self, tool_call_str: str) -> bool:
        """Check session grants, removing expired ones. Thread-safe."""
        now = time.monotonic()
        with self._grants_lock:
            expired = [p for p, t in self._session_grants.items() if now - t > self._grant_ttl]
            for p in expired:
                del self._session_grants[p]
                logger.info("Session grant expired pattern=%s", p)

            # Snapshot grants under lock, then match outside
            grants = list(self._session_grants.keys())

        # Normalize the grant's tool-name token so a CamelCase grant
        # ("Bash(npm *)") matches the canonical formatted call
        # ("bash(npm install)") on every platform: fnmatch.fnmatch is
        # case-sensitive on POSIX but case-folds on Windows, and matching
        # must not depend on the OS.
        from godspeed.security.rules import normalize_pattern_tool_name

        return any(
            fnmatch.fnmatch(tool_call_str, normalize_pattern_tool_name(pattern))
            for pattern in grants
        )

    def _ask_or_replay(self, tool_call: ToolCall, reason: str) -> PermissionDecision:
        """Return a replayed decision for a pending approval, else ASK.

        When a pending record exists for this exact tool call (same
        fingerprint) with a recorded decision, that decision is replayed
        and the record is consumed. Otherwise the decision is persisted as
        pending so a crash before the user answers can be reconciled next
        session.
        """
        if self._pending_dir is None:
            return PermissionDecision(ASK, reason)
        fingerprint = approval_fingerprint(tool_call.tool_name, tool_call.arguments)
        replayed = self._replay_pending(fingerprint)
        if replayed is not None:
            return replayed
        self._persist_pending(fingerprint, tool_call, reason)
        return PermissionDecision(ASK, reason)

    def _pending_path(self, fingerprint: str) -> Path:
        if self._pending_dir is None:
            raise RuntimeError("Pending-approvals directory is not configured")
        return self._pending_dir / f"{fingerprint}.json"

    def _persist_pending(self, fingerprint: str, tool_call: ToolCall, reason: str) -> None:
        """Persist a pending approval record BEFORE prompting. Fail-safe."""
        try:
            self._pending_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "fingerprint": fingerprint,
                "tool_name": tool_call.tool_name,
                "arguments_json": json.dumps(
                    tool_call.arguments or {}, sort_keys=True, default=str
                ),
                "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rule_suggestion": reason,
            }
            self._pending_path(fingerprint).write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            self._evict_pending()
        except OSError as exc:
            logger.warning(
                "Pending approval persist failed fingerprint=%s error=%s", fingerprint, exc
            )

    def _evict_pending(self) -> None:
        """Cap the pending directory at MAX_PENDING, evicting oldest by mtime."""
        try:
            entries = sorted(
                self._pending_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            for stale in entries[: max(0, len(entries) - MAX_PENDING)]:
                stale.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Pending approval eviction failed error=%s", exc)

    def record_decision(self, fingerprint: str, approved: bool) -> None:
        """Record the user's decision on a pending approval. Fail-safe.

        Called by the permission proxy after the user answers an ASK
        prompt. If the process dies before the tool executes, the next
        session replays this decision instead of re-prompting.
        """
        if self._pending_dir is None:
            return
        path = self._pending_path(fingerprint)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.debug("No pending record for fingerprint=%s", fingerprint)
            return
        record["decision"] = "approved" if approved else "denied"
        record["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Pending approval decision write failed fingerprint=%s error=%s", fingerprint, exc
            )

    def _replay_pending(self, fingerprint: str) -> PermissionDecision | None:
        """Replay a recorded decision for a pending approval, if any.

        Returns None when there is no record or no recorded decision (the
        user never answered before the crash — re-ask). Consumes the
        record on replay so a decision is applied exactly once.
        """
        path = self._pending_path(fingerprint)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        decision = record.get("decision")
        if decision not in ("approved", "denied"):
            return None
        path.unlink(missing_ok=True)
        if decision == "approved":
            return PermissionDecision(ALLOW, "replayed from pending record")
        return PermissionDecision(DENY, "replayed from pending record (denied)")

    @staticmethod
    def _default_for_risk(risk: RiskLevel) -> PermissionDecision:
        """Get the default permission decision for a risk level."""
        if risk == RiskLevel.READ_ONLY:
            return PermissionDecision(ALLOW, "read-only tool")
        if risk == RiskLevel.LOW:
            return PermissionDecision(ASK, "low-risk write tool")
        if risk == RiskLevel.DESTRUCTIVE:
            return PermissionDecision(DENY, "destructive tool blocked by default")
        return PermissionDecision(ASK, "high-risk tool")
