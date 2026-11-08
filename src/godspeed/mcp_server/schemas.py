"""MCP schema helpers for exposing Godspeed tools."""

from __future__ import annotations

from mcp import types

from godspeed.tools.base import Tool


def tool_to_mcp_tool(tool: Tool) -> types.Tool:
    """Convert a Godspeed Tool into an MCP Tool definition."""
    return types.Tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.get_schema(),
    )


def build_mcp_tools(tools: list[Tool]) -> list[types.Tool]:
    """Build MCP tool definitions from registered tools."""
    return [tool_to_mcp_tool(tool) for tool in tools]
