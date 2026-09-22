"""Client-agnostic MCP server that wraps existing Godspeed tools."""

from __future__ import annotations

import json
import logging
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import yaml
from mcp import types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from godspeed._bootstrap import _build_tool_registry, _load_env_files
from godspeed.audit.trail import AuditTrail
from godspeed.config import GodspeedSettings, load_settings
from godspeed.mcp_server.schemas import build_mcp_tools
from godspeed.security.permissions import ALLOW, PermissionEngine
from godspeed.security.secrets import redact_secrets
from godspeed.tools.base import ToolCall, ToolContext
from godspeed.tools.tasks import TaskStore, TaskTool

logger = logging.getLogger(__name__)


def _load_settings_with_optional_config(config_path: Path | None) -> GodspeedSettings:
    """Load settings using standard CLI behavior, with optional file override."""
    _load_env_files(project_dir=Path("."))
    if config_path is None:
        return load_settings()
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        config_data = {}
    return load_settings(**config_data)


def _extract_caller(arguments: dict[str, Any] | None) -> str:
    """Extract MCP caller identity from metadata when available."""
    if not isinstance(arguments, dict):
        return "mcp_client:unknown"

    metadata = arguments.get("_meta")
    client_name = ""
    if isinstance(metadata, dict):
        raw_name = metadata.get("client_name") or metadata.get("clientName")
        if isinstance(raw_name, str):
            client_name = raw_name.strip()
    if not client_name:
        return "mcp_client:unknown"
    return f"mcp_client:{client_name}"


def _sanitize_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Remove protocol-only fields before dispatching to Godspeed tools."""
    if not isinstance(arguments, dict):
        return {}
    sanitized = dict(arguments)
    sanitized.pop("_meta", None)
    return sanitized


def _primary_argument(arguments: dict[str, Any]) -> str:
    """Return a best-effort primary argument string for audit context."""
    for key in ("command", "file_path", "path", "url", "pattern", "action"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    for value in arguments.values():
        if isinstance(value, str):
            return value
    return "*"


class GodspeedMCPServer:
    """MCP stdio server that wraps Godspeed tool dispatch."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        self.settings = _load_settings_with_optional_config(config_path)
        self.project_dir = Path(self.settings.project_dir).resolve()
        self.session_id = str(uuid4())

        registry, risk_levels = _build_tool_registry()
        task_store = TaskStore()
        task_tool = TaskTool(task_store)
        registry.register(task_tool)
        risk_levels[task_tool.name] = task_tool.risk_level
        self.registry = registry

        self.permission_engine = PermissionEngine(
            deny_patterns=self.settings.permissions.deny,
            allow_patterns=self.settings.permissions.allow,
            ask_patterns=self.settings.permissions.ask,
            tool_risk_levels=risk_levels,
        )

        self.audit_trail: AuditTrail | None = None
        if self.settings.audit.enabled:
            audit_dir = self.settings.global_dir / "audit"
            self.audit_trail = AuditTrail(log_dir=audit_dir, session_id=self.session_id)
            self.audit_trail.record(
                event_type="session_start",
                detail={
                    "mode": "mcp_server",
                    "project_dir": str(self.project_dir),
                },
            )
            self.audit_trail.cleanup_expired(self.settings.audit.retention_days)

        self.tool_context = ToolContext(
            cwd=self.project_dir,
            session_id=self.session_id,
            permissions=self.permission_engine,
            audit=self.audit_trail,
        )

        self.server = Server("godspeed-mcp")
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return self.list_mcp_tools()

        @self.server.call_tool(validate_input=True)
        async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            return await self.handle_tool_call(name=name, arguments=arguments)

    def list_mcp_tools(self) -> list[types.Tool]:
        """Return all registered Godspeed tools as MCP definitions."""
        return build_mcp_tools(self.registry.list_tools())

    async def handle_tool_call(
        self, *, name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        """Permission-check, execute, redact, and audit an MCP tool call."""
        caller = _extract_caller(arguments)
        clean_arguments = _sanitize_arguments(arguments)
        tool_call = ToolCall(tool_name=name, arguments=clean_arguments)
        primary_arg = _primary_argument(clean_arguments)
        decision = self.permission_engine.evaluate(tool_call)

        if decision.action != ALLOW:
            deny_payload = {
                "tool": name,
                "argument": primary_arg,
                "reason": decision.reason or "Permission denied",
                "timestamp": datetime.now(UTC).isoformat(),
                "caller": caller,
            }
            await self._audit(
                outcome="error",
                detail={
                    "caller": caller,
                    "tool": name,
                    "argument": primary_arg,
                    "denied": True,
                    "reason": decision.reason or "Permission denied",
                    "decision": decision.action,
                },
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(deny_payload))],
                structuredContent=deny_payload,
                isError=True,
            )

        result = await self.registry.dispatch(tool_call, self.tool_context)
        output_text = result.error if result.is_error else result.output
        redacted_output = redact_secrets(output_text or "")
        outcome = "error" if result.is_error else "success"
        await self._audit(
            outcome=outcome,
            detail={
                "caller": caller,
                "tool": name,
                "argument": primary_arg,
                "denied": False,
                "is_error": result.is_error,
                "output_length": len(redacted_output),
            },
        )

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=redacted_output)],
            structuredContent={
                "tool": name,
                "caller": caller,
                "is_error": result.is_error,
            },
            isError=result.is_error,
        )

    async def _audit(self, *, outcome: str, detail: dict[str, Any]) -> None:
        if self.audit_trail is None:
            return
        await self.audit_trail.arecord(
            event_type="tool_call",
            detail=detail,
            outcome=outcome,
        )

    async def run_stdio(self) -> None:
        """Run the MCP server over stdio transport."""
        sys.stderr.write("Godspeed MCP server ready\n")
        sys.stderr.flush()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="godspeed",
                    server_version="0.4.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    async def shutdown(self) -> None:
        """Flush and close audit trail resources."""
        if self.audit_trail is not None:
            await self.audit_trail.aclose()


def run_server(config_path: Path | None = None) -> int:
    """Run MCP server until interrupted."""
    server = GodspeedMCPServer(config_path=config_path)

    def _handle_signal(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
    try:
        anyio.run(server.run_stdio)
    except KeyboardInterrupt:
        logger.info("MCP server interrupted, shutting down")
    finally:
        anyio.run(server.shutdown)
        sys.stderr.write("Godspeed MCP server shutdown\n")
        sys.stderr.flush()
    return 0
