"""Tests for ACP adapter (Agent Client Protocol)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from godspeed.mcp.acp_adapter import (
    ACPServerConfig,
    ACPSession,
    ACPClient,
    ACPToolAdapter,
    ACPToolDefinition,
    adapt_acp_agents,
)
from godspeed.tools.base import RiskLevel


class TestACPServerConfig:
    """Test ACPServerConfig construction."""

    def test_defaults(self) -> None:
        config = ACPServerConfig(name="test", base_url="http://localhost:8000")
        assert config.name == "test"
        assert config.base_url == "http://localhost:8000"
        assert config.headers == {}
        assert config.timeout == 120.0

    def test_strips_trailing_slash(self) -> None:
        config = ACPServerConfig(name="test", base_url="http://localhost:8000/")
        assert config.base_url == "http://localhost:8000"


class TestACPToolDefinition:
    """Test ACPToolDefinition construction."""

    def test_fields(self) -> None:
        defn = ACPToolDefinition(
            name="acp_claude_fix",
            description="Fix bugs using Claude",
            models=["claude-opus-4"],
            capabilities=["code_edit", "test_run"],
            server_name="claude",
        )
        assert defn.name == "acp_claude_fix"
        assert defn.models == ["claude-opus-4"]
        assert defn.server_name == "claude"


class TestACPSession:
    """Test ACPSession dataclass."""

    def test_fields(self) -> None:
        session = ACPSession(session_id="abc-123", agent_name="claude")
        assert session.session_id == "abc-123"
        assert len(session.messages) == 0

    def test_messages_capped_at_200(self) -> None:
        session = ACPSession(session_id="x", agent_name="srv")
        for i in range(250):
            session.messages.append({"role": "user", "content": f"msg-{i}"})
        assert len(session.messages) == 200
        assert session.messages[0]["content"] == "msg-50"
        assert session.messages[-1]["content"] == "msg-249"


class TestACPClient:
    """Test ACPClient connection, sessions, and messaging."""

    @pytest.mark.asyncio
    async def test_connect_discovers_agents(self) -> None:
        client = ACPClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "agents": [
                {
                    "name": "fixer",
                    "description": "Fix bugs",
                    "models": ["gpt-4o"],
                    "capabilities": ["code_edit"],
                }
            ]
        }

        with patch("godspeed.mcp.acp_adapter.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_http

            config = ACPServerConfig(name="claude", base_url="http://localhost:8000")
            defs = await client.connect(config)

            assert len(defs) == 1
            assert defs[0].name == "acp_claude_fixer"
            assert defs[0].server_name == "claude"

    @pytest.mark.asyncio
    async def test_connect_failure_returns_empty(self) -> None:
        client = ACPClient()

        with patch("godspeed.mcp.acp_adapter.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_http.aclose = AsyncMock()
            mock_cls.return_value = mock_http

            config = ACPServerConfig(name="claude", base_url="http://localhost:8000")
            defs = await client.connect(config)
            assert defs == []

    @pytest.mark.asyncio
    async def test_create_session(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_clients["claude"] = mock_http

        session_id = await client.create_session("claude")
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id in client._sessions

    @pytest.mark.asyncio
    async def test_create_session_no_server_raises(self) -> None:
        client = ACPClient()
        with pytest.raises(ConnectionError, match="not connected"):
            await client.create_session("nonexistent")

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"content": "I fixed the bug."}
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_clients["claude"] = mock_http

        session_id = await client.create_session("claude")
        result = await client.send_message(session_id, "Fix the auth bug")
        assert result == "I fixed the bug."

    @pytest.mark.asyncio
    async def test_send_message_unknown_session(self) -> None:
        client = ACPClient()
        result = await client.send_message("nonexistent", "hello")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_send_message_timeout(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        client._http_clients["claude"] = mock_http

        session_id = str(__import__("uuid").uuid4())
        client._sessions[session_id] = ACPSession(session_id=session_id, agent_name="claude")

        result = await client.send_message(session_id, "hello")
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_close_session(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_http.delete = AsyncMock(return_value=mock_response)
        client._http_clients["claude"] = mock_http

        session_id = str(__import__("uuid").uuid4())
        client._sessions[session_id] = ACPSession(session_id=session_id, agent_name="claude")

        await client.close_session(session_id)
        assert session_id not in client._sessions

    @pytest.mark.asyncio
    async def test_disconnect_all(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client._http_clients["claude"] = mock_http
        client._capabilities["claude"] = {}

        await client.disconnect_all()
        assert client._http_clients == {}
        assert client._capabilities == {}

    @pytest.mark.asyncio
    async def test_send_message_list_content(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Part 1"}, {"text": "Part 2"}]}
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_clients["claude"] = mock_http

        session_id = await client.create_session("claude")
        result = await client.send_message(session_id, "hello")
        assert "Part 1" in result
        assert "Part 2" in result


class TestACPToolAdapter:
    """Test ACPToolAdapter wrapping remote agents as Godspeed tools."""

    def test_metadata(self) -> None:
        client = ACPClient()
        defn = ACPToolDefinition(
            name="acp_claude_fix",
            description="Fix bugs via Claude",
            models=["claude-opus-4"],
            capabilities=[],
            server_name="claude",
        )
        adapter = ACPToolAdapter(defn, client)
        assert adapter.name == "acp_claude_fix"
        assert adapter.description == "Fix bugs via Claude"
        assert adapter.risk_level == RiskLevel.HIGH

    def test_schema(self) -> None:
        client = ACPClient()
        defn = ACPToolDefinition(
            name="acp_claude_fix",
            description="Fix bugs via Claude",
            models=[],
            capabilities=[],
            server_name="claude",
        )
        adapter = ACPToolAdapter(defn, client)
        schema = adapter.get_schema()
        assert "message" in schema["properties"]
        assert "new_session" in schema["properties"]
        assert schema["required"] == ["message"]

    @pytest.mark.asyncio
    async def test_execute_no_message(self) -> None:
        client = ACPClient()
        defn = ACPToolDefinition(
            name="acp_test",
            description="test",
            models=[],
            capabilities=[],
            server_name="test",
        )
        adapter = ACPToolAdapter(defn, client)
        from godspeed.tools.base import ToolContext

        result = await adapter.execute(
            {}, ToolContext(cwd=__import__("pathlib").Path("."), session_id="s")
        )
        assert result.is_error
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_creates_session_and_sends(self) -> None:
        mock_http = AsyncMock()

        session_resp = MagicMock()
        session_resp.raise_for_status = MagicMock()

        msg_resp = MagicMock()
        msg_resp.raise_for_status = MagicMock()
        msg_resp.json.return_value = {"content": "Fixed!"}

        mock_http.post = AsyncMock(side_effect=[session_resp, msg_resp])
        mock_http.delete = AsyncMock(return_value=MagicMock(raise_for_status=MagicMock()))

        client = ACPClient()
        client._http_clients["test_server"] = mock_http

        defn = ACPToolDefinition(
            name="acp_test_fixer",
            description="Fix",
            models=[],
            capabilities=[],
            server_name="test_server",
        )
        adapter = ACPToolAdapter(defn, client)

        from godspeed.tools.base import ToolContext

        ctx = ToolContext(cwd=__import__("pathlib").Path("."), session_id="s")
        result = await adapter.execute({"message": "Fix the bug"}, ctx)
        assert not result.is_error
        assert "Fixed!" in result.output

    @pytest.mark.asyncio
    async def test_close_session(self) -> None:
        client = ACPClient()
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_http.delete = AsyncMock(return_value=mock_resp)
        client._http_clients["test"] = mock_http

        defn = ACPToolDefinition(
            name="acp_test",
            description="test",
            models=[],
            capabilities=[],
            server_name="test",
        )
        adapter = ACPToolAdapter(defn, client)
        adapter._session_id = "abc-123"

        await adapter.close()
        assert adapter._session_id is None


class TestAdaptAcpAgents:
    """Test adapt_acp_agents utility."""

    def test_creates_adapters(self) -> None:
        client = ACPClient()
        defs = [
            ACPToolDefinition(
                name="acp_claude_1",
                description="Agent 1",
                models=[],
                capabilities=[],
                server_name="claude",
            ),
            ACPToolDefinition(
                name="acp_claude_2",
                description="Agent 2",
                models=[],
                capabilities=[],
                server_name="claude",
            ),
        ]
        adapters = adapt_acp_agents(defs, client)
        assert len(adapters) == 2
        assert adapters[0].name == "acp_claude_1"
        assert adapters[1].name == "acp_claude_2"
