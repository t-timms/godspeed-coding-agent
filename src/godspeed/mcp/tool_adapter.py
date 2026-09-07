"""MCP tool adapter — maps MCP tool definitions to Godspeed Tool ABC.

Trust tiering: tools from servers listed in ``settings.mcp_trusted_servers``
are LOW risk (ask once, then session-scoped allow); untrusted servers' tools
stay HIGH risk (gate every call). Lazy tool defs: the full input schema is not
materialized into the LLM's context until the tool is first executed — MCP
servers validate their own arguments, and after the first call the real
schema flows into context on the next turn.
"""

from __future__ import annotations

import logging
from typing import Any

from godspeed.mcp.client import MCPClient, MCPToolDefinition
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA_STUB: dict[str, Any] = {"type": "object", "properties": {}}


class MCPToolAdapter(Tool):
    """Adapts an MCP tool definition into a Godspeed Tool.

    Tools from untrusted MCP servers default to HIGH risk since they execute
    external code from third-party servers; servers on the trusted list get
    LOW risk (ask once, then session-scoped allow). When ``lazy_schema`` is
    set, the advertised schema is a compact stub until first execution
    (lazy tool definitions).
    """

    def __init__(
        self,
        definition: MCPToolDefinition,
        mcp_client: MCPClient,
        *,
        trusted: bool = False,
        lazy_schema: bool = True,
    ) -> None:
        self._definition = definition
        self._mcp_client = mcp_client
        self._trusted = trusted
        self._lazy_schema = lazy_schema
        self._schema_loaded = not lazy_schema
        # Strip mcp_{server}_ prefix to get the original tool name
        prefix = f"mcp_{definition.server_name}_"
        self._original_name = definition.name.removeprefix(prefix)

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW if self._trusted else RiskLevel.HIGH

    @property
    def trusted(self) -> bool:
        """Whether the originating MCP server is on the trusted list."""
        return self._trusted

    def get_schema(self) -> dict[str, Any]:
        if not self._schema_loaded:
            return dict(_EMPTY_SCHEMA_STUB)
        schema = self._definition.input_schema
        if not schema or not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        return schema

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the MCP tool by calling the remote server.

        First execution materializes the real input schema so subsequent
        turns advertise it to the LLM (lazy tool definitions).
        """
        if not self._schema_loaded:
            self._schema_loaded = True
            logger.info("MCP tool schema materialized tool=%s", self.name)
        try:
            result = await self._mcp_client.call_tool(
                server_name=self._definition.server_name,
                tool_name=self._original_name,
                arguments=arguments,
            )
            if result.startswith("Error:"):
                return ToolResult.failure(result)
            return ToolResult.success(result)
        except Exception as exc:
            logger.error(
                "MCP tool execution failed tool=%s error=%s",
                self.name,
                exc,
                exc_info=True,
            )
            return ToolResult.failure(f"MCP tool '{self.name}' failed: {exc}")


def adapt_mcp_tools(
    definitions: list[MCPToolDefinition],
    mcp_client: MCPClient,
    *,
    trusted_servers: frozenset[str] | set[str] = frozenset(),
    lazy_schema: bool = True,
) -> list[MCPToolAdapter]:
    """Convert a list of MCP tool definitions into Godspeed tools.

    Args:
        definitions: Tool definitions discovered from MCP servers.
        mcp_client: Connected MCP client used for tool execution.
        trusted_servers: Server names whose tools get LOW risk instead
            of HIGH (from ``settings.mcp_trusted_servers``).
        lazy_schema: Advertise a stub schema until first execution.
    """
    return [
        MCPToolAdapter(
            defn,
            mcp_client,
            trusted=defn.server_name in trusted_servers,
            lazy_schema=lazy_schema,
        )
        for defn in definitions
    ]
