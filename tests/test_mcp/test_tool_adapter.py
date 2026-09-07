"""Tests for MCP tool adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from godspeed.mcp.client import MCPClient, MCPToolDefinition
from godspeed.mcp.tool_adapter import MCPToolAdapter, adapt_mcp_tools
from godspeed.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool_defn() -> MCPToolDefinition:
    return MCPToolDefinition(
        name="mcp_github_search_repos",
        description="Search GitHub repositories",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
        server_name="github",
    )


@pytest.fixture
def mock_mcp_client() -> MCPClient:
    client = MCPClient()
    client.call_tool = AsyncMock(return_value="Found 5 repos matching 'godspeed'")
    return client


class TestMCPToolAdapter:
    """Test adapting MCP tools to Godspeed Tool ABC."""

    def test_name(self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        assert adapter.name == "mcp_github_search_repos"

    def test_description(self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        assert adapter.description == "Search GitHub repositories"

    def test_risk_level_is_high(
        self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient
    ) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        assert adapter.risk_level == "high"

    def test_schema(self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client, lazy_schema=False)
        schema = adapter.get_schema()
        assert schema["type"] == "object"
        assert "query" in schema["properties"]

    def test_schema_empty_fallback(self, mock_mcp_client: MCPClient) -> None:
        defn = MCPToolDefinition(
            name="mcp_test_empty",
            description="Empty schema",
            input_schema={},
            server_name="test",
        )
        adapter = MCPToolAdapter(defn, mock_mcp_client, lazy_schema=False)
        schema = adapter.get_schema()
        assert schema["type"] == "object"

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        tool_defn: MCPToolDefinition,
        mock_mcp_client: MCPClient,
        tmp_path: Path,
    ) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        context = ToolContext(cwd=tmp_path, session_id="test")
        result = await adapter.execute({"query": "godspeed"}, context)
        assert not result.is_error
        assert "5 repos" in result.output
        mock_mcp_client.call_tool.assert_called_once_with(
            server_name="github",
            tool_name="search_repos",
            arguments={"query": "godspeed"},
        )

    @pytest.mark.asyncio
    async def test_execute_error_from_server(
        self,
        tool_defn: MCPToolDefinition,
        tmp_path: Path,
    ) -> None:
        client = MCPClient()
        client.call_tool = AsyncMock(return_value="Error: Server disconnected")
        adapter = MCPToolAdapter(tool_defn, client)
        context = ToolContext(cwd=tmp_path, session_id="test")
        result = await adapter.execute({"query": "test"}, context)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_execute_exception(
        self,
        tool_defn: MCPToolDefinition,
        tmp_path: Path,
    ) -> None:
        client = MCPClient()
        client.call_tool = AsyncMock(side_effect=RuntimeError("Connection lost"))
        adapter = MCPToolAdapter(tool_defn, client)
        context = ToolContext(cwd=tmp_path, session_id="test")
        result = await adapter.execute({"query": "test"}, context)
        assert result.is_error
        assert "failed" in result.error.lower()


class TestAdaptMCPTools:
    """Test batch tool adaptation."""

    def test_adapt_multiple(self, mock_mcp_client: MCPClient) -> None:
        definitions = [
            MCPToolDefinition("mcp_gh_search", "Search", {}, "gh"),
            MCPToolDefinition("mcp_gh_create", "Create", {}, "gh"),
        ]
        tools = adapt_mcp_tools(definitions, mock_mcp_client)
        assert len(tools) == 2
        assert all(isinstance(t, MCPToolAdapter) for t in tools)
        assert tools[0].name == "mcp_gh_search"
        assert tools[1].name == "mcp_gh_create"

    def test_adapt_empty(self, mock_mcp_client: MCPClient) -> None:
        tools = adapt_mcp_tools([], mock_mcp_client)
        assert tools == []


class TestTrustTiering:
    """Per-server trust tiering: trusted servers' tools get LOW risk."""

    def test_untrusted_default_high(
        self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient
    ) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        assert adapter.risk_level == RiskLevel.HIGH
        assert adapter.trusted is False

    def test_trusted_low(self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client, trusted=True)
        assert adapter.risk_level == RiskLevel.LOW
        assert adapter.trusted is True

    def test_adapt_maps_trusted_servers(self, mock_mcp_client: MCPClient) -> None:
        definitions = [
            MCPToolDefinition("mcp_gh_search", "Search", {}, "gh"),
            MCPToolDefinition("mcp_ext_query", "Query", {}, "ext"),
        ]
        tools = adapt_mcp_tools(definitions, mock_mcp_client, trusted_servers={"gh"})
        assert tools[0].risk_level == RiskLevel.LOW
        assert tools[1].risk_level == RiskLevel.HIGH


class TestLazyToolDefs:
    """Lazy tool definitions: stub schema until first execution."""

    def test_stub_schema_before_first_execute(
        self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient, tmp_path: Path
    ) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        schema = adapter.get_schema()
        assert schema == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_real_schema_after_first_execute(
        self,
        tool_defn: MCPToolDefinition,
        mock_mcp_client: MCPClient,
        tmp_path: Path,
    ) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client)
        context = ToolContext(cwd=tmp_path, session_id="test")
        await adapter.execute({"query": "godspeed"}, context)
        schema = adapter.get_schema()
        assert "query" in schema["properties"]

    def test_eager_schema_opt_out(
        self, tool_defn: MCPToolDefinition, mock_mcp_client: MCPClient
    ) -> None:
        adapter = MCPToolAdapter(tool_defn, mock_mcp_client, lazy_schema=False)
        assert "query" in adapter.get_schema()["properties"]

    def test_adapt_lazy_by_default(self, mock_mcp_client: MCPClient) -> None:
        definitions = [MCPToolDefinition("mcp_gh_search", "Search", {}, "gh")]
        tools = adapt_mcp_tools(definitions, mock_mcp_client)
        assert tools[0].get_schema() == {"type": "object", "properties": {}}


# ============================================================================
# ACPToolAdapter.aclose()
# ============================================================================


class TestACPToolAdapterAclose:
    """Test that aclose() tears down session + client."""

    @pytest.mark.asyncio
    async def test_aclose_closes_session_and_client(self) -> None:
        from godspeed.mcp.acp_adapter import ACPClient, ACPToolAdapter, ACPToolDefinition

        client = ACPClient()
        client.close_session = AsyncMock()
        client.disconnect_all = AsyncMock()

        defn = ACPToolDefinition(
            name="acp_test",
            description="test",
            models=[],
            capabilities=[],
            server_name="test",
        )
        adapter = ACPToolAdapter(defn, client)
        adapter._session_id = "abc-123"

        await adapter.aclose()

        client.close_session.assert_awaited_once_with("abc-123")
        client.disconnect_all.assert_awaited_once()
        assert adapter._session_id is None

    @pytest.mark.asyncio
    async def test_aclose_no_active_session(self) -> None:
        from godspeed.mcp.acp_adapter import ACPClient, ACPToolAdapter, ACPToolDefinition

        client = ACPClient()
        client.close_session = AsyncMock()
        client.disconnect_all = AsyncMock()

        defn = ACPToolDefinition(
            name="acp_test",
            description="test",
            models=[],
            capabilities=[],
            server_name="test",
        )
        adapter = ACPToolAdapter(defn, client)

        await adapter.aclose()

        client.close_session.assert_not_awaited()
        client.disconnect_all.assert_awaited_once()
