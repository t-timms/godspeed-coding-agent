"""Tests for MCP client."""

from __future__ import annotations

import asyncio
import builtins
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from godspeed.mcp.client import MCPClient, MCPServerConfig, MCPToolDefinition


class TestMCPServerConfig:
    """Test MCP server configuration."""

    def test_defaults(self) -> None:
        config = MCPServerConfig(name="test", command="echo")
        assert config.name == "test"
        assert config.command == "echo"
        assert config.args == []
        assert config.env == {}
        assert config.transport == "stdio"

    def test_with_args(self) -> None:
        config = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "test"},
        )
        assert config.args == ["-y", "@modelcontextprotocol/server-github"]
        assert config.env == {"GITHUB_TOKEN": "test"}

    def test_sse_config(self) -> None:
        """SSE transport config stores url and headers."""
        config = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:3001",
            headers={"Authorization": "Bearer tok123"},
        )
        assert config.transport == "sse"
        assert config.url == "http://localhost:3001"
        assert config.headers == {"Authorization": "Bearer tok123"}
        assert config.command == ""
        assert config.args == []

    def test_transport_defaults_to_stdio(self) -> None:
        """Omitting transport field defaults to stdio (backward compat)."""
        config = MCPServerConfig(name="legacy", command="/usr/bin/server")
        assert config.transport == "stdio"
        assert config.url is None
        assert config.headers is None

    def test_empty_args_and_env(self) -> None:
        """None args/env get replaced by empty list/dict."""
        config = MCPServerConfig(name="t", command="c", args=None, env=None)
        assert config.args == []
        assert config.env == {}


class TestMCPToolDefinition:
    """Test tool definition data class."""

    def test_basic(self) -> None:
        defn = MCPToolDefinition(
            name="mcp_github_search",
            description="Search GitHub repos",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            server_name="github",
        )
        assert defn.name == "mcp_github_search"
        assert defn.server_name == "github"
        assert "query" in defn.input_schema["properties"]


# ---------------------------------------------------------------------------
# Helpers for mocking the mcp SDK (stdio transport)
# ---------------------------------------------------------------------------


def _make_mock_stdio_session(tools_list: list[MagicMock]) -> MagicMock:
    """Return a mock ClientSession with list_tools and call_tool."""
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock()
    tools_response = MagicMock()
    tools_response.tools = tools_list
    session.list_tools.return_value = tools_response
    return session


def _make_mock_sse_client(tools_list: list[dict], call_tool_result: str = "ok") -> MagicMock:
    """Return a mock MCPSSEClient."""
    client = MagicMock()
    client.connect = AsyncMock()
    client.list_tools = AsyncMock(return_value=tools_list)
    client.call_tool = AsyncMock(return_value=call_tool_result)
    client.disconnect = AsyncMock()
    return client


def _mock_tool(name: str = "search", description: str = "desc") -> MagicMock:
    """Return a mock stdio tool object."""
    t = MagicMock()
    t.name = name
    t.description = description
    type(t).inputSchema = PropertyMock(return_value={"type": "object"})
    return t


def _mock_call_tool_result(content_texts: list[str]) -> MagicMock:
    """Return a mock CallToolResult with text content items."""
    result = MagicMock()
    items = []
    for text in content_texts:
        item = MagicMock()
        item.text = text
        items.append(item)
    result.content = items
    return result


# ---------------------------------------------------------------------------
# Helper: force stdio check to False by faking an ImportError
# ---------------------------------------------------------------------------


def _patch_import_check(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """Make `import <module_name>` raise ImportError within _check_*_available."""
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"No module named '{module_name}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


# ============================================================================
# MCPClient: availability checks
# ============================================================================


class TestMCPClientAvailability:
    """Test MCPClient availability reporting."""

    def test_both_available_by_default(self) -> None:
        client = MCPClient()
        assert client.stdio_available is True
        assert client.available is True

    def test_stdio_unavailable_when_mcp_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_import_check(monkeypatch, "mcp")
        client = MCPClient()
        assert client.stdio_available is False
        assert client.available is True  # sse still works

    def test_sse_unavailable_when_httpx_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_import_check(monkeypatch, "httpx")
        client = MCPClient()
        assert client._sse_available is False
        assert client.available is True  # stdio still works

    def test_nothing_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _fake(name, *a, **kw):
            if name in ("mcp", "httpx"):
                raise ImportError(name)
            return original_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fake)
        client = MCPClient()
        assert client.stdio_available is False
        assert client._sse_available is False
        assert client.available is False


# ============================================================================
# MCPClient: connect stdio
# ============================================================================


class TestMCPClientConnectStdio:
    """Test stdio transport connections."""

    def test_connect_stdio_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_import_check(monkeypatch, "mcp")
        client = MCPClient()
        config = MCPServerConfig(name="srv", command="echo")
        result = asyncio.run(client.connect(config))
        assert result == []

    def test_connect_stdio_success(self) -> None:
        mock_tool = _mock_tool("greet", "Say hello")
        mock_session = _make_mock_stdio_session([mock_tool])

        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters") as mock_params,
        ):
            client = MCPClient()
            config = MCPServerConfig(name="my-server", command="python", args=["-m", "mcp_server"])
            result = asyncio.run(client.connect(config))

            assert len(result) == 1
            assert result[0].name == "mcp_my-server_greet"
            assert result[0].server_name == "my-server"

    def test_connect_stdio_exception(self) -> None:
        with patch(
            "mcp.client.stdio.stdio_client",
            side_effect=RuntimeError("subprocess died"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="broken", command="nonexistent")
            result = asyncio.run(client.connect(config))
            assert result == []

    def test_connect_stdio_tool_without_description(self) -> None:
        t = _mock_tool("no_desc", description=None)
        mock_session = _make_mock_stdio_session([t])
        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)
        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="srv", command="cmd")
            result = asyncio.run(client.connect(config))
            assert len(result) == 1
            assert result[0].name == "mcp_srv_no_desc"

    def test_connect_stdio_tool_without_input_schema(self) -> None:
        t = MagicMock()
        t.name = "bare"
        t.description = "bare tool"
        del t.inputSchema
        mock_session = _make_mock_stdio_session([t])
        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)
        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="srv", command="cmd")
            result = asyncio.run(client.connect(config))
            assert result[0].input_schema == {}

    def test_connect_stdio_env_none(self) -> None:
        """When config.env is empty dict, StdioServerParameters gets None."""
        mock_tool = _mock_tool("t")
        mock_session = _make_mock_stdio_session([mock_tool])
        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)
        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters") as mock_params,
        ):
            client = MCPClient()
            config = MCPServerConfig(name="srv", command="cmd", env={})
            asyncio.run(client.connect(config))
            mock_params.assert_called_once_with(command="cmd", args=[], env=None)


# ============================================================================
# MCPClient: connect sse
# ============================================================================


class TestMCPClientConnectSSE:
    """Test SSE transport connections."""

    def test_connect_sse_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_import_check(monkeypatch, "httpx")
        client = MCPClient()
        config = MCPServerConfig(name="remote", transport="sse", url="http://localhost:3001")
        result = asyncio.run(client.connect(config))
        assert result == []

    def test_connect_sse_missing_url(self) -> None:
        client = MCPClient()
        config = MCPServerConfig(name="remote", transport="sse")
        result = asyncio.run(client.connect(config))
        assert result == []

    def test_connect_sse_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_sse = _make_mock_sse_client(
            [
                {"name": "echo", "description": "Echo back", "inputSchema": {"type": "object"}},
            ]
        )

        def _mock_mcp_sse_client(*a, **kw):
            return mock_sse

        monkeypatch.setattr("godspeed.mcp.sse_transport.MCPSSEClient", _mock_mcp_sse_client)

        client = MCPClient()
        config = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:3001",
            headers={"X-Token": "abc"},
        )
        result = asyncio.run(client.connect(config))

        assert len(result) == 1
        assert result[0].name == "mcp_remote_echo"
        assert result[0].server_name == "remote"

    def test_connect_sse_tool_missing_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_sse = _make_mock_sse_client([{"description": "No name tool", "inputSchema": {}}])

        monkeypatch.setattr(
            "godspeed.mcp.sse_transport.MCPSSEClient",
            lambda *a, **kw: mock_sse,
        )

        client = MCPClient()
        config = MCPServerConfig(name="remote", transport="sse", url="http://localhost:3001")
        result = asyncio.run(client.connect(config))
        assert result[0].name == "mcp_remote_unknown"

    def test_connect_sse_tool_missing_description(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_sse = _make_mock_sse_client([{"name": "no_desc"}])

        monkeypatch.setattr(
            "godspeed.mcp.sse_transport.MCPSSEClient",
            lambda *a, **kw: mock_sse,
        )

        client = MCPClient()
        config = MCPServerConfig(name="remote", transport="sse", url="http://localhost:3001")
        result = asyncio.run(client.connect(config))
        assert "MCP tool: no_desc" in result[0].description

    def test_connect_sse_connection_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_sse = MagicMock()
        mock_sse.connect = AsyncMock(side_effect=ConnectionError("refused"))
        monkeypatch.setattr(
            "godspeed.mcp.sse_transport.MCPSSEClient",
            lambda *a, **kw: mock_sse,
        )

        client = MCPClient()
        config = MCPServerConfig(name="remote", transport="sse", url="http://localhost:3001")
        result = asyncio.run(client.connect(config))
        assert result == []


# ============================================================================
# MCPClient: call_tool
# ============================================================================


class TestMCPClientCallTool:
    """Test tool invocation on connected servers."""

    def test_call_tool_via_sse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MCPClient()
        mock_sse = _make_mock_sse_client([], call_tool_result="SSE result text")
        client._sse_clients["remote"] = mock_sse

        result = asyncio.run(client.call_tool("remote", "echo", {"message": "hi"}))
        assert result == "SSE result text"
        mock_sse.call_tool.assert_called_once_with("echo", {"message": "hi"})

    def test_call_tool_via_stdio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MCPClient()
        mock_result = _mock_call_tool_result(["stdio output"])
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        client._connections["local"] = mock_session

        result = asyncio.run(client.call_tool("local", "search", {"query": "x"}))
        assert result == "stdio output"
        mock_session.call_tool.assert_called_once_with("search", {"query": "x"})

    def test_call_tool_stdio_result_no_content(self) -> None:
        client = MCPClient()
        mock_result = MagicMock()
        del mock_result.content
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        client._connections["srv"] = mock_session

        result = asyncio.run(client.call_tool("srv", "t", {}))
        assert result == str(mock_result)

    def test_call_tool_stdio_result_content_without_text(self) -> None:
        client = MCPClient()
        item = MagicMock()
        del item.text
        mock_result = MagicMock()
        mock_result.content = [item]
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        client._connections["srv"] = mock_session

        result = asyncio.run(client.call_tool("srv", "t", {}))
        assert result == str(mock_result)

    def test_call_tool_server_not_connected(self) -> None:
        client = MCPClient()
        result = asyncio.run(client.call_tool("missing", "tool", {}))
        assert "not connected" in result

    def test_call_tool_stdio_exception(self) -> None:
        client = MCPClient()
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        client._connections["local"] = mock_session

        result = asyncio.run(client.call_tool("local", "broken", {}))
        assert "MCP tool call failed" in result


# ============================================================================
# MCPClient: disconnect_all
# ============================================================================


class TestMCPClientDisconnect:
    """Test disconnect and cleanup."""

    def test_disconnect_all_empty(self) -> None:
        client = MCPClient()
        asyncio.run(client.disconnect_all())

    def test_disconnect_all_with_sse_clients(self) -> None:
        client = MCPClient()
        mock_sse = _make_mock_sse_client([])
        client._sse_clients["a"] = mock_sse
        client._sse_clients["b"] = mock_sse
        asyncio.run(client.disconnect_all())
        assert client._sse_clients == {}
        assert client._connections == {}

    def test_disconnect_all_sse_exception(self) -> None:
        client = MCPClient()
        mock_bad = MagicMock()
        mock_bad.disconnect = AsyncMock(side_effect=ConnectionError("lost"))
        client._sse_clients["bad"] = mock_bad
        asyncio.run(client.disconnect_all())
        assert client._sse_clients == {}

    def test_disconnect_all_mixed_clients(self) -> None:
        client = MCPClient()
        mock_sse = _make_mock_sse_client([])
        mock_session = MagicMock()
        client._sse_clients["remote"] = mock_sse
        client._connections["local"] = mock_session
        asyncio.run(client.disconnect_all())
        assert client._sse_clients == {}
        assert client._connections == {}


# ============================================================================
# MCPClient: multiple server connections
# ============================================================================


class TestMCPClientMultipleServers:
    """Test connecting to multiple MCP servers."""

    def test_multiple_connections(self) -> None:
        mock_t = _mock_tool("t1")
        mock_session = _make_mock_stdio_session([mock_t])
        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)
        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            c1 = asyncio.run(client.connect(MCPServerConfig(name="srv1", command="a")))
            c2 = asyncio.run(client.connect(MCPServerConfig(name="srv2", command="b")))
            assert len(c1) == 1
            assert len(c2) == 1
            assert client._connections  # still holds references


# ============================================================================
# MCPClient: stdio CM leak protection
# ============================================================================


class TestMCPClientStdioLeakProtection:
    """Verify context managers are exited when initialization fails."""

    def test_initialize_failure_exits_cm(self) -> None:
        """If session.initialize() raises, both CMs must be exited."""
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock(side_effect=RuntimeError("init failed"))

        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="broken", command="cmd")
            result = asyncio.run(client.connect(config))

            assert result == []
            mock_stdlib_client.__aexit__.assert_awaited_once()
            mock_session_cls.__aexit__.assert_awaited_once()

    def test_list_tools_failure_exits_cm(self) -> None:
        """If session.list_tools() raises, both CMs must be exited."""
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=RuntimeError("list failed"))

        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="broken", command="cmd")
            result = asyncio.run(client.connect(config))

            assert result == []
            mock_stdlib_client.__aexit__.assert_awaited_once()
            mock_session_cls.__aexit__.assert_awaited_once()

    def test_failure_does_not_register_connections(self) -> None:
        """Failed init must not leave dangling entries in _connections / _stdio_transports."""
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock(side_effect=RuntimeError("boom"))

        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="leaky", command="cmd")
            asyncio.run(client.connect(config))

            assert "leaky" not in client._connections
            assert "leaky" not in client._stdio_transports


# ============================================================================
# MCPClient: timeout
# ============================================================================


class TestMCPClientTimeout:
    """Verify asyncio.wait_for wraps critical operations."""

    def test_connect_stdio_timeout(self) -> None:
        """Slow initialize() triggers TimeoutError and CM cleanup."""

        async def _slow_init() -> None:
            await asyncio.sleep(100)

        mock_session = MagicMock()
        mock_session.initialize = _slow_init

        mock_stdlib_client = MagicMock()
        mock_stdlib_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdlib_client.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls = MagicMock()
        mock_session_cls.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdlib_client),
            patch("mcp.ClientSession", return_value=mock_session_cls),
            patch("mcp.StdioServerParameters"),
        ):
            client = MCPClient()
            config = MCPServerConfig(name="slow", command="cmd")
            result = asyncio.run(client.connect(config, timeout=0.1))

            assert result == []
            mock_stdlib_client.__aexit__.assert_awaited_once()
            mock_session_cls.__aexit__.assert_awaited_once()

    def test_call_tool_timeout(self) -> None:
        """Slow call_tool triggers TimeoutError and returns error string."""

        async def _slow_call(*_a: object, **_kw: object) -> None:
            await asyncio.sleep(100)

        client = MCPClient()
        mock_session = MagicMock()
        mock_session.call_tool = _slow_call
        client._connections["slow"] = mock_session

        result = asyncio.run(client.call_tool("slow", "t", {}, timeout=0.1))
        assert "timed out" in result.lower()

    def test_connect_sse_timeout(self) -> None:
        """Slow SSE connect triggers TimeoutError and returns empty."""

        async def _slow_connect() -> None:
            await asyncio.sleep(100)

        mock_sse = MagicMock()
        mock_sse.connect = _slow_connect

        def _factory(*a: object, **kw: object) -> MagicMock:
            return mock_sse

        with patch("godspeed.mcp.sse_transport.MCPSSEClient", _factory):
            client = MCPClient()
            config = MCPServerConfig(name="slow_sse", transport="sse", url="http://localhost:9999")
            result = asyncio.run(client.connect(config, timeout=0.1))
            assert result == []
