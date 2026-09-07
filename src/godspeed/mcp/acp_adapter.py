"""Agent Client Protocol (ACP) adapter — drive external coding agents as sub-workers.

ACP is a lightweight HTTP+JSON protocol for agent-to-agent interop.  This
module provides:

- ``ACPClient``: discovers capabilities and drives sessions on an external
  ACP-compliant agent server.
- ``ACPToolAdapter``: wraps a remote agent as a Godspeed ``Tool`` so the
  LLM can delegate subtasks to it seamlessly.
- ``ACPToolDefinition``: metadata about a remote agent's capabilities.

The transport uses ``httpx`` (always bundled) for HTTP and follows the
same retry/timeout patterns as the SSE transport in ``godspeed.mcp.sse_transport``.
"""

from __future__ import annotations

import collections
import logging
import uuid
from typing import Any

import httpx

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 120.0
_MAX_RESPONSE_BYTES = 2_097_152  # 2 MB — agent responses can be large
_MAX_SESSION_MESSAGES = 200


class ACPServerConfig:
    """Configuration for an ACP server connection."""

    def __init__(
        self,
        name: str,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = _TIMEOUT_SECONDS,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout


class ACPToolDefinition:
    """Metadata about a remote agent discovered via ACP capabilities."""

    def __init__(
        self,
        name: str,
        description: str,
        models: list[str],
        capabilities: list[str],
        server_name: str,
    ) -> None:
        self.name = name
        self.description = description
        self.models = models
        self.capabilities = capabilities
        self.server_name = server_name


class ACPSession:
    """Represents an active session with a remote ACP agent."""

    def __init__(self, session_id: str, agent_name: str) -> None:
        self.session_id = session_id
        self.agent_name = agent_name
        self.messages: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_SESSION_MESSAGES,
        )


class ACPClient:
    """Client for discovering and driving external agents via ACP.

    ACP endpoints:

    - ``POST /initialize`` — handshake, returns agent capabilities.
    - ``POST /sessions`` — create a new agent session.
    - ``POST /sessions/{id}/messages`` — send a message, get a response.
    - ``DELETE /sessions/{id}`` — close a session.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ACPSession] = {}
        self._capabilities: dict[str, dict[str, Any]] = {}
        self._http_clients: dict[str, httpx.AsyncClient] = {}

    async def connect(self, config: ACPServerConfig) -> list[ACPToolDefinition]:
        """Connect to an ACP server and discover its capabilities."""
        client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=config.headers,
            timeout=httpx.Timeout(config.timeout),
        )
        self._http_clients[config.name] = client

        try:
            resp = await client.post("/initialize", json={})
            resp.raise_for_status()
            data = resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.error("ACP connect failed server=%s error=%s", config.name, exc)
            await client.aclose()
            self._http_clients.pop(config.name, None)
            return []
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "ACP initialize non-2xx server=%s status=%d",
                config.name,
                exc.response.status_code,
            )
            await client.aclose()
            self._http_clients.pop(config.name, None)
            return []

        self._capabilities[config.name] = data
        agents = data.get("agents", data.get("tools", []))

        definitions: list[ACPToolDefinition] = []
        for agent in agents:
            agent_name = agent.get("name", config.name)
            defn = ACPToolDefinition(
                name=f"acp_{config.name}_{agent_name}",
                description=agent.get(
                    "description",
                    f"External ACP agent: {agent_name} (via {config.name})",
                ),
                models=agent.get("models", []),
                capabilities=agent.get("capabilities", []),
                server_name=config.name,
            )
            definitions.append(defn)

        logger.info(
            "ACP connected server=%s agents=%d",
            config.name,
            len(definitions),
        )
        return definitions

    async def create_session(self, server_name: str) -> str:
        """Create a new session with a remote agent. Returns the session ID."""
        client = self._http_clients.get(server_name)
        if client is None:
            raise ConnectionError(f"ACP server '{server_name}' is not connected")

        session_id = str(uuid.uuid4())
        try:
            resp = await client.post(
                "/sessions",
                json={"session_id": session_id},
            )
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.error("ACP session creation failed server=%s error=%s", server_name, exc)
            raise ConnectionError(f"ACP session creation failed: {exc}") from exc

        session = ACPSession(session_id=session_id, agent_name=server_name)
        self._sessions[session_id] = session
        logger.info("ACP session created server=%s session=%s", server_name, session_id)
        return session_id

    async def send_message(
        self,
        session_id: str,
        message: str,
    ) -> str:
        """Send a message to a remote agent and return the response text."""
        session = self._sessions.get(session_id)
        if session is None:
            return f"Error: ACP session '{session_id}' not found"

        client = self._http_clients.get(session.agent_name)
        if client is None:
            return f"Error: ACP server '{session.agent_name}' not connected"

        session.messages.append({"role": "user", "content": message})

        try:
            resp = await client.post(
                f"/sessions/{session_id}/messages",
                json={"content": message},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            logger.error(
                "ACP message timed out server=%s session=%s",
                session.agent_name,
                session_id,
            )
            return f"Error: ACP message timed out for session '{session_id}'"
        except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.error(
                "ACP message failed server=%s session=%s error=%s",
                session.agent_name,
                session_id,
                exc,
            )
            return f"Error: ACP message failed — {exc}"

        content = data.get("content", data.get("response", ""))
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)

        session.messages.append({"role": "assistant", "content": content})
        return str(content)

    async def close_session(self, session_id: str) -> None:
        """Close an ACP session."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return

        client = self._http_clients.get(session.agent_name)
        if client is not None:
            try:
                resp = await client.delete(f"/sessions/{session_id}")
                resp.raise_for_status()
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning("ACP session close failed: %s", exc)

        logger.info("ACP session closed server=%s session=%s", session.agent_name, session_id)

    async def disconnect_all(self) -> None:
        """Close all sessions and HTTP clients."""
        for session_id in list(self._sessions):
            await self.close_session(session_id)
        for client in self._http_clients.values():
            try:
                await client.aclose()
            except Exception as exc:
                logger.error("ACP client close error: %s", exc)
        self._http_clients.clear()
        self._capabilities.clear()


class ACPToolAdapter(Tool):
    """Wraps a remote ACP agent as a Godspeed Tool.

    All ACP tools default to HIGH risk since they execute code on
    external agent servers.  The adapter manages sessions automatically:
    a session is created on first execute and reused until the tool
    is garbage-collected or explicitly closed.
    """

    def __init__(
        self,
        definition: ACPToolDefinition,
        acp_client: ACPClient,
    ) -> None:
        self._definition = definition
        self._acp_client = acp_client
        self._session_id: str | None = None

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The task/message to send to the external agent",
                },
                "new_session": {
                    "type": "boolean",
                    "description": "Force a new session instead of reusing the existing one",
                },
            },
            "required": ["message"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        message = arguments.get("message", "")
        if not message:
            return ToolResult.failure("message is required for ACP agent call")

        new_session = arguments.get("new_session", False)

        if self._session_id is None or new_session:
            if self._session_id is not None:
                await self._acp_client.close_session(self._session_id)
            try:
                self._session_id = await self._acp_client.create_session(
                    self._definition.server_name,
                )
            except ConnectionError as exc:
                return ToolResult.failure(str(exc))

        result = await self._acp_client.send_message(self._session_id, message)
        if result.startswith("Error:"):
            return ToolResult.failure(result)
        return ToolResult.success(result)

    async def close(self) -> None:
        """Close the underlying ACP session."""
        if self._session_id is not None:
            await self._acp_client.close_session(self._session_id)
            self._session_id = None

    async def aclose(self) -> None:
        """Close the session and disconnect the underlying ACP client."""
        await self.close()
        await self._acp_client.disconnect_all()


def adapt_acp_agents(
    definitions: list[ACPToolDefinition],
    acp_client: ACPClient,
) -> list[ACPToolAdapter]:
    """Convert ACP agent definitions into Godspeed tools."""
    return [ACPToolAdapter(defn, acp_client) for defn in definitions]
