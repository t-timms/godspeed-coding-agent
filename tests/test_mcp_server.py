from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from mcp import types

from godspeed.mcp_server.server import (
    GodspeedMCPServer,
    _extract_caller,
    _load_settings_with_optional_config,
    _primary_argument,
    _sanitize_arguments,
    run_server,
)
from godspeed.security.permissions import ALLOW, DENY, ASK, PermissionDecision
from godspeed.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings_for(tmp_path: Path, *, audit_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        project_dir=tmp_path,
        global_dir=tmp_path,
        permissions=SimpleNamespace(deny=[], allow=[], ask=[]),
        audit=SimpleNamespace(enabled=audit_enabled, retention_days=30),
    )


def _make_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> GodspeedMCPServer:
    monkeypatch.setattr(
        "godspeed.mcp_server.server._load_settings_with_optional_config",
        lambda _path: _settings_for(tmp_path),
    )
    return GodspeedMCPServer()


# ============================================================================
# _load_settings_with_optional_config
# ============================================================================


class TestLoadSettings:
    def test_with_config_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "godspeed.yaml"
        config_file.write_text("project_dir: " + str(tmp_path).replace("\\", "\\\\"))

        result = _load_settings_with_optional_config(config_file)
        assert str(result.project_dir) == str(tmp_path)

    def test_with_config_path_non_dict_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "godspeed.yaml"
        config_file.write_text("just_a_string")

        result = _load_settings_with_optional_config(config_file)
        assert result is not None

    def test_with_config_path_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "godspeed.yaml"
        config_file.write_text("")

        result = _load_settings_with_optional_config(config_file)
        assert result is not None

    def test_with_none_config_path(self) -> None:
        result = _load_settings_with_optional_config(None)
        assert result is not None


# ============================================================================
# _extract_caller
# ============================================================================


class TestExtractCaller:
    def test_non_dict_arguments(self) -> None:
        assert _extract_caller(None) == "mcp_client:unknown"
        assert _extract_caller("string") == "mcp_client:unknown"

    def test_no_meta_field(self) -> None:
        assert _extract_caller({"file_path": "x"}) == "mcp_client:unknown"

    def test_meta_not_dict(self) -> None:
        assert _extract_caller({"_meta": "not_a_dict"}) == "mcp_client:unknown"

    def test_client_name_in_meta(self) -> None:
        result = _extract_caller({"_meta": {"client_name": "vscode"}})
        assert result == "mcp_client:vscode"

    def test_client_name_camel_case(self) -> None:
        result = _extract_caller({"_meta": {"clientName": "Cursor"}})
        assert result == "mcp_client:Cursor"

    def test_client_name_prefers_snake_case(self) -> None:
        result = _extract_caller({"_meta": {"client_name": "snake", "clientName": "camel"}})
        assert result == "mcp_client:snake"

    def test_client_name_empty_string(self) -> None:
        result = _extract_caller({"_meta": {"client_name": ""}})
        assert result == "mcp_client:unknown"

    def test_client_name_stripped(self) -> None:
        result = _extract_caller({"_meta": {"client_name": "  claude  "}})
        assert result == "mcp_client:claude"


# ============================================================================
# _sanitize_arguments
# ============================================================================


class TestSanitizeArguments:
    def test_non_dict(self) -> None:
        assert _sanitize_arguments(None) == {}
        assert _sanitize_arguments([]) == {}

    def test_removes_meta(self) -> None:
        result = _sanitize_arguments({"file_path": "x", "_meta": "data"})
        assert "_meta" not in result
        assert result["file_path"] == "x"

    def test_no_meta_field(self) -> None:
        args = {"file_path": "x", "line": 42}
        result = _sanitize_arguments(args)
        assert result == {"file_path": "x", "line": 42}
        assert result is not args  # copy


# ============================================================================
# _primary_argument
# ============================================================================


class TestPrimaryArgument:
    def test_command_key_first(self) -> None:
        assert _primary_argument({"command": "ls -la", "file_path": "x"}) == "ls -la"

    def test_file_path_fallback(self) -> None:
        assert _primary_argument({"file_path": "README.md"}) == "README.md"

    def test_path_fallback(self) -> None:
        assert _primary_argument({"path": "/tmp"}) == "/tmp"

    def test_url_fallback(self) -> None:
        assert _primary_argument({"url": "http://localhost"}) == "http://localhost"

    def test_pattern_fallback(self) -> None:
        assert _primary_argument({"pattern": "*.py"}) == "*.py"

    def test_action_fallback(self) -> None:
        assert _primary_argument({"action": "delete"}) == "delete"

    def test_first_string_value_fallback(self) -> None:
        assert _primary_argument({"unknown_key": "hello"}) == "hello"

    def test_no_string_values(self) -> None:
        assert _primary_argument({"count": 42}) == "*"

    def test_empty_arguments(self) -> None:
        assert _primary_argument({}) == "*"


# ============================================================================
# GodspeedMCPServer: initialization
# ============================================================================


class TestServerInit:
    def test_audit_enabled(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        assert server.audit_trail is not None
        assert server.permission_engine is not None
        assert server.tool_context is not None
        assert server.server is not None

    def test_audit_disabled(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "godspeed.mcp_server.server._load_settings_with_optional_config",
            lambda _path: _settings_for(tmp_path, audit_enabled=False),
        )
        server = GodspeedMCPServer()
        assert server.audit_trail is None

    def test_list_mcp_tools(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        listed = server.list_mcp_tools()
        registered = server.registry.list_tools()
        assert len(listed) == len(registered)
        assert len(listed) > 0


# ============================================================================
# GodspeedMCPServer: handle_tool_call
# ============================================================================


class TestHandleToolCall:
    @pytest.mark.asyncio
    async def test_file_read_allowed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("file-content"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        result = await server.handle_tool_call(
            name="file_read", arguments={"file_path": "README.md"}
        )

        assert result.is_error is False
        assert result.content[0].text == "file-content"
        assert "file_read" in result.structured_content["tool"]
        server.audit_trail.arecord.assert_awaited_once()
        audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
        assert audit_detail["denied"] is False

    @pytest.mark.asyncio
    async def test_file_read_denied(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(
            return_value=PermissionDecision(DENY, "blocked path")
        )
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("should-not-run"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        result = await server.handle_tool_call(name="file_read", arguments={"file_path": ".env"})

        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["tool"] == "file_read"
        assert payload["reason"] == "blocked path"
        server.registry.dispatch.assert_not_awaited()
        audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
        assert audit_detail["denied"] is True

    @pytest.mark.asyncio
    async def test_ask_decision_denied(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(
            return_value=PermissionDecision(ASK, "Please confirm")
        )
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("should-not-run"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        result = await server.handle_tool_call(name="shell", arguments={"command": "rm file"})

        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["tool"] == "shell"

    @pytest.mark.asyncio
    async def test_deny_without_explicit_reason(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(DENY, ""))
        server.registry.dispatch = AsyncMock()
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        result = await server.handle_tool_call(name="shell", arguments={"command": "rm -rf /"})

        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["reason"] == "Permission denied"

    @pytest.mark.asyncio
    async def test_tool_error_result(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(
            return_value=ToolResult.failure("something went wrong")
        )
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        result = await server.handle_tool_call(name="risky", arguments={"action": "test"})

        assert result.is_error is True
        assert result.content[0].text == "something went wrong"
        audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
        assert audit_detail["is_error"] is True

    @pytest.mark.asyncio
    async def test_secret_redaction(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(
            return_value=ToolResult.ok("token=sk-ant-api03-fakeabcdefghijklmnopqrstu")
        )
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        result = await server.handle_tool_call(
            name="file_read", arguments={"file_path": "secret.txt"}
        )

        assert result.is_error is False
        assert "sk-ant" not in result.content[0].text
        assert "[REDACTED]" in result.content[0].text

    @pytest.mark.asyncio
    async def test_unknown_caller(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("ok"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        await server.handle_tool_call(name="file_read", arguments={"file_path": "README.md"})

        audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
        assert audit_detail["caller"] == "mcp_client:unknown"

    @pytest.mark.asyncio
    async def test_caller_with_meta(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("ok"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        await server.handle_tool_call(
            name="file_read",
            arguments={
                "file_path": "x",
                "_meta": {"client_name": "claude-code"},
            },
        )

        audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
        assert audit_detail["caller"] == "mcp_client:claude-code"

    @pytest.mark.asyncio
    async def test_primary_argument_in_audit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("ok"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        await server.handle_tool_call(
            name="shell", arguments={"command": "echo hello", "_meta": {}}
        )

        audit_detail = server.audit_trail.arecord.await_args.kwargs["detail"]
        assert audit_detail["argument"] == "echo hello"


# ============================================================================
# GodspeedMCPServer: _audit
# ============================================================================


class TestAuditMethod:
    @pytest.mark.asyncio
    async def test_audit_when_trail_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "godspeed.mcp_server.server._load_settings_with_optional_config",
            lambda _path: _settings_for(tmp_path, audit_enabled=False),
        )
        server = GodspeedMCPServer()
        await server._audit(outcome="success", detail={"tool": "t"})
        # should not raise

    @pytest.mark.asyncio
    async def test_audit_when_trail_enabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        await server._audit(outcome="success", detail={"tool": "t"})

        server.audit_trail.arecord.assert_awaited_once_with(
            event_type="tool_call",
            detail={"tool": "t"},
            outcome="success",
        )


# ============================================================================
# GodspeedMCPServer: shutdown
# ============================================================================


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_with_audit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        assert server.audit_trail is not None
        await server.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_without_audit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "godspeed.mcp_server.server._load_settings_with_optional_config",
            lambda _path: _settings_for(tmp_path, audit_enabled=False),
        )
        server = GodspeedMCPServer()
        assert server.audit_trail is None
        await server.shutdown()
        # should not raise


# ============================================================================
# GodspeedMCPServer: run_stdio
# ============================================================================


class TestRunStdio:
    @pytest.mark.asyncio
    async def test_run_stdio_writes_ready_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        mock_server_run = AsyncMock()
        server.server.run = mock_server_run

        mock_stdio = MagicMock()
        mock_stdio.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "godspeed.mcp_server.server.stdio_server", return_value=mock_stdio
            ) as mock_stdio_server,
            patch("sys.stderr") as mock_stderr,
        ):
            await server.run_stdio()

            mock_stderr.write.assert_called_with("Godspeed MCP server ready\n")
            mock_stderr.flush.assert_called()
            mock_stdio_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_server_top_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "godspeed.mcp_server.server._load_settings_with_optional_config",
            lambda _path: _settings_for(tmp_path),
        )

        mock_shutdown = AsyncMock()
        mock_run_stdio = AsyncMock()

        with (
            patch("godspeed.mcp_server.server.GodspeedMCPServer.shutdown", mock_shutdown),
            patch("godspeed.mcp_server.server.GodspeedMCPServer.run_stdio", mock_run_stdio),
            patch("godspeed.mcp_server.server.anyio.run") as mock_anyio,
            patch("sys.stderr") as mock_stderr,
        ):
            result = run_server()
            assert result == 0
            mock_stderr.write.assert_any_call("Godspeed MCP server shutdown\n")

    def test_run_server_keyboard_interrupt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "godspeed.mcp_server.server._load_settings_with_optional_config",
            lambda _path: _settings_for(tmp_path),
        )

        call_count = 0
        signal_handler = None

        def _capture_signal(signum, handler):
            nonlocal signal_handler
            signal_handler = handler

        def _anyio_run(fn):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt

        with (
            patch("godspeed.mcp_server.server.anyio.run", side_effect=_anyio_run),
            patch("signal.signal", side_effect=_capture_signal),
            patch("sys.stderr") as mock_stderr,
        ):
            result = run_server()
            assert result == 0
            assert signal_handler is not None
            stderr_writes = [c[0][0] for c in mock_stderr.write.call_args_list if c[0]]
            assert "Godspeed MCP server shutdown\n" in stderr_writes

        assert signal_handler is not None
        with pytest.raises(KeyboardInterrupt):
            signal_handler(signal.SIGINT, None)


# ============================================================================
# Registered handler tests (exercising the adapter closures)
# ============================================================================


class TestRegisteredHandlers:
    def test_handlers_registered_on_server(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        assert server.server.get_request_handler("tools/list") is not None
        assert server.server.get_request_handler("tools/call") is not None

    @pytest.mark.asyncio
    async def test_list_tools_handler(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        result = await server._on_list_tools(ANY, None)
        assert isinstance(result, types.ListToolsResult)
        assert any(t.name == "file_read" for t in result.tools)

    @pytest.mark.asyncio
    async def test_call_tool_handler_dispatches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock(return_value=PermissionDecision(ALLOW, "ok"))
        server.registry.dispatch = AsyncMock(return_value=ToolResult.ok("done"))
        server.audit_trail = Mock()
        server.audit_trail.arecord = AsyncMock()

        params = types.CallToolRequestParams(name="mock_tool", arguments={"key": "val"})
        result = await server._on_call_tool(ANY, params)
        assert isinstance(result, types.CallToolResult)
        assert result.content[0].text == "done"

    @pytest.mark.asyncio
    async def test_call_tool_handler_rejects_invalid_input(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """mcp 2.x dropped validate_input; the server must still refuse bad arguments
        before the permission engine or the tool ever runs."""
        server = _make_server(monkeypatch, tmp_path)
        server.permission_engine.evaluate = Mock()
        server.registry.dispatch = AsyncMock()

        params = types.CallToolRequestParams(name="file_read", arguments={})
        result = await server._on_call_tool(ANY, params)
        assert result.is_error is True
        assert result.content[0].text.startswith("Input validation error:")
        assert "file_path" in result.content[0].text
        server.permission_engine.evaluate.assert_not_called()
        server.registry.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_tool_handler_none_arguments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """mcp 2.x passes arguments=None through; it must not crash or skip validation."""
        server = _make_server(monkeypatch, tmp_path)
        server.registry.dispatch = AsyncMock()

        params = types.CallToolRequestParams(name="file_read", arguments=None)
        result = await server._on_call_tool(ANY, params)
        assert result.is_error is True
        server.registry.dispatch.assert_not_called()

    def test_validate_arguments_unknown_tool_skips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        server = _make_server(monkeypatch, tmp_path)
        assert server.validate_arguments("no_such_tool", {"x": 1}) is None
