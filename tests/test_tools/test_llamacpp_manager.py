"""Tests for llama.cpp server management tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.llamacpp_manager import (
    DEFAULT_CONTEXT,
    DRAFT_MODE_MAX_CONTEXT,
    LlamaCppTool,
    configure_litellm_env,
    get_server_status,
    is_server_running,
    start_server,
    stop_server,
)


def _all_flag_values(cmd: list[str], flag: str) -> list[str]:
    """Every value passed for *flag* in *cmd*, in order — lets a test
    assert a flag appears exactly once (regression guard for the bug where
    a second -c 24576 silently overrode whatever the first -c decided,
    since llama-server takes the last occurrence of a repeated flag)."""
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == flag and i + 1 < len(cmd)]


class TestFindServerBinary:
    """Test binary discovery."""

    @patch.object(Path, "exists", return_value=True)
    def test_default_path_found(self, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_server_binary

        result = _find_server_binary()
        assert result is not None

    @patch.object(Path, "exists", return_value=False)
    @patch("shutil.which", return_value="/usr/bin/llama-server")
    def test_path_found(self, mock_which: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_server_binary

        result = _find_server_binary()
        assert result == Path("/usr/bin/llama-server")

    @patch.object(Path, "exists", return_value=False)
    @patch("shutil.which", return_value=None)
    def test_not_found(self, mock_which: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_server_binary

        result = _find_server_binary()
        assert result is None


class TestFindModel:
    """Test model file discovery."""

    @patch.object(Path, "exists", return_value=True)
    def test_default_model_found(self, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_model

        result = _find_model()
        assert result is not None

    @patch.object(Path, "exists", side_effect=[False, True])
    @patch("pathlib.Path.glob", return_value=[Path("/models/other.gguf")])
    def test_any_gguf_found(self, mock_glob: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_model

        result = _find_model()
        assert result == Path("/models/other.gguf")

    @patch.object(Path, "exists", return_value=False)
    @patch("pathlib.Path.glob", return_value=[])
    def test_no_model_found(self, mock_glob: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_model

        result = _find_model()
        assert result is None


class TestIsServerRunning:
    """Test server health check."""

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_server_running(self, mock_request: MagicMock, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__.return_value = MagicMock()
        assert is_server_running("http://127.0.0.1:8080") is True

    @patch("urllib.request.urlopen", side_effect=Exception("Connection refused"))
    @patch("urllib.request.Request")
    def test_server_not_running(self, mock_request: MagicMock, mock_urlopen: MagicMock) -> None:
        assert is_server_running("http://127.0.0.1:8080") is False


class TestStartServer:
    """Test server startup logic."""

    def test_already_running(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True):
            result = start_server()
        assert result is None

    def test_binary_not_found(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch("godspeed.tools.llamacpp_manager._find_server_binary", return_value=None):
                result = start_server()
        assert result is None

    def test_model_not_found(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch("godspeed.tools.llamacpp_manager._find_model", return_value=None):
                    result = start_server()
        assert result is None

    def test_start_success(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(timeout=1)
        assert result is mock_proc

    def test_start_os_error(self) -> None:
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", side_effect=OSError("Permission denied")):
                        result = start_server()
        assert result is None

    def test_server_exits_early(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.communicate.return_value = ("stdout", "stderr")
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(timeout=1)
        assert result is None


class TestStopServer:
    """Test server shutdown."""

    def test_none_process(self) -> None:
        assert stop_server(None) is True

    def test_already_terminated(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        assert stop_server(mock_proc) is True

    def test_graceful_stop(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        assert stop_server(mock_proc) is True
        mock_proc.terminate.assert_called_once()

    def test_force_kill(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
        assert stop_server(mock_proc) is True
        mock_proc.kill.assert_called_once()

    def test_error_handling(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = Exception("boom")
        assert stop_server(mock_proc) is False


class TestGetServerStatus:
    """Test status querying."""

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False)
    def test_not_running(self, mock_is_running: MagicMock) -> None:
        status = get_server_status()
        assert status["running"] is False
        assert status["model"] is None

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_running_with_model(self, mock_request, mock_urlopen, mock_is_running) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": [{"id": "test-model"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        status = get_server_status()
        assert status["running"] is True
        assert status["model"] == "test-model"

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_running_list_format(self, mock_request, mock_urlopen, mock_is_running) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'[{"id": "list-model"}]'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        status = get_server_status()
        assert status["model"] == "list-model"


class TestConfigureLitellmEnv:
    """Test environment configuration."""

    @patch.dict("os.environ", {}, clear=True)
    def test_sets_defaults(self) -> None:
        configure_litellm_env()
        import os

        assert os.environ.get("LLAMACPP_API_BASE") == "http://127.0.0.1:8080/v1"
        assert os.environ.get("OPENAI_BASE_URL") == "http://127.0.0.1:8080/v1"
        assert os.environ.get("OPENAI_API_KEY") == "none"

    @patch.dict("os.environ", {"LLAMACPP_API_BASE": "existing"}, clear=True)
    def test_preserves_existing(self) -> None:
        configure_litellm_env()
        import os

        assert os.environ.get("LLAMACPP_API_BASE") == "existing"


class TestLlamaCppTool:
    """Test the LlamaCpp tool integration."""

    @pytest.mark.asyncio
    async def test_status_not_running(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            result = await tool.execute({"action": "status"}, tool_context)
            assert "NOT RUNNING" in result.output

    @pytest.mark.asyncio
    async def test_status_running(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch(
            "godspeed.tools.llamacpp_manager.get_server_status",
            return_value={
                "running": True,
                "url": "http://127.0.0.1:8080",
                "model": "test-model",
                "version": "1.0",
            },
        ):
            result = await tool.execute({"action": "status"}, tool_context)
            assert "RUNNING" in result.output
            assert "test-model" in result.output

    @pytest.mark.asyncio
    async def test_start_success(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.start_server", return_value=MagicMock()):
            result = await tool.execute({"action": "start"}, tool_context)
            assert "started successfully" in result.output

    @pytest.mark.asyncio
    async def test_start_already_running(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.start_server", return_value=None):
            with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True):
                result = await tool.execute({"action": "start"}, tool_context)
                assert "already running" in result.output

    @pytest.mark.asyncio
    async def test_start_failure(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch("godspeed.tools.llamacpp_manager.start_server", return_value=None):
            with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
                result = await tool.execute({"action": "start"}, tool_context)
                assert "Failed to start" in result.error

    @pytest.mark.asyncio
    async def test_stop(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        result = await tool.execute({"action": "stop"}, tool_context)
        assert "stop requested" in result.output

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        result = await tool.execute({"action": "fly"}, tool_context)
        assert result.is_error is True
        assert "Unknown action" in result.error

    def test_schema(self) -> None:
        tool = LlamaCppTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["status", "start", "stop"]

    def test_name(self) -> None:
        tool = LlamaCppTool()
        assert tool.name == "llamacpp"

    def test_risk_level(self) -> None:
        tool = LlamaCppTool()
        from godspeed.tools.base import RiskLevel

        assert tool.risk_level == RiskLevel.LOW


class TestFindDraftModel:
    """Test draft model auto-detection."""

    @patch.object(Path, "exists", return_value=False)
    def test_no_models_dir(self, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_draft_model

        result = _find_draft_model()
        assert result is None

    @patch.object(Path, "exists", return_value=True)
    @patch.object(Path, "stat")
    def test_large_main_model_skips_draft(
        self, mock_stat: MagicMock, mock_exists: MagicMock
    ) -> None:
        from godspeed.tools.llamacpp_manager import _find_draft_model

        mock_stat.return_value.st_size = 14 * 1024**3
        result = _find_draft_model()
        assert result is None

    @patch.object(Path, "exists")
    @patch.object(Path, "stat")
    @patch("pathlib.Path.glob")
    def test_qwen25_coder_1_5b_found(
        self, mock_glob: MagicMock, mock_stat: MagicMock, mock_exists: MagicMock
    ) -> None:
        from godspeed.tools.llamacpp_manager import _find_draft_model

        mock_stat.return_value.st_size = 8 * 1024**3  # small enough
        mock_exists.return_value = True
        mock_glob.return_value = []

        result = _find_draft_model()
        assert result is not None

    @patch.object(Path, "exists")
    @patch.object(Path, "stat")
    @patch("pathlib.Path.glob")
    def test_qwen25_coder_0_5b_found(
        self, mock_glob: MagicMock, mock_stat: MagicMock, mock_exists: MagicMock
    ) -> None:
        from godspeed.tools.llamacpp_manager import _find_draft_model

        mock_stat.return_value.st_size = 8 * 1024**3
        mock_exists.return_value = True
        mock_glob.return_value = []

        result = _find_draft_model()
        assert result is not None

    @patch.object(Path, "exists")
    @patch.object(Path, "stat")
    @patch("pathlib.Path.glob")
    def test_1_5b_in_name_found(
        self, mock_glob: MagicMock, mock_stat: MagicMock, mock_exists: MagicMock
    ) -> None:

        from godspeed.tools.llamacpp_manager import _find_draft_model

        mock_stat.return_value.st_size = 8 * 1024**3
        mock_exists.return_value = True
        dummy = Path("/models/dummy-1.5b.gguf")
        mock_glob.return_value = [dummy]

        result = _find_draft_model()
        assert result is not None

    @patch.object(Path, "exists", return_value=False)
    @patch("pathlib.Path.glob", return_value=[])
    def test_no_draft_found(self, mock_glob: MagicMock, mock_exists: MagicMock) -> None:
        from godspeed.tools.llamacpp_manager import _find_draft_model

        result = _find_draft_model()
        assert result is None


class TestStartServerExtended:
    """Extended start_server tests with various configurations."""

    def test_start_with_draft_model_disabled(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(
                            draft_model_path=False,
                            timeout=1,
                        )
        assert result is mock_proc

    def test_start_no_flash_attn(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(
                            flash_attn=False,
                            timeout=1,
                        )
        assert result is mock_proc

    @pytest.mark.skipif(
        not (Path.home() / ".llamacpp" / "models" / "qwen2.5-coder-14b-q4_K_M.gguf").exists(),
        reason="Model file not available on CI",
    )
    def test_start_with_cuda_bin_path(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cuda_bin = Path.home() / "miniconda3" / "envs" / "cuda-build" / "Library" / "bin"
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch.object(Path, "exists", return_value=True):
                        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                            result = start_server(timeout=1)
        assert result is mock_proc
        env = mock_popen.call_args[1]["env"]
        assert str(cuda_bin) in env["PATH"]

    def test_start_cuda_bin_not_exists(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch.object(Path, "exists", return_value=False):
                        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                            result = start_server(timeout=1)
        assert result is mock_proc
        env = mock_popen.call_args[1]["env"]
        assert "cuda-build" not in env.get("PATH", "")

    def test_start_timeout(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        with patch(
                            "godspeed.tools.llamacpp_manager.time.monotonic", side_effect=[0, 999]
                        ):
                            result = start_server(timeout=1)
        assert result is mock_proc

    def test_start_explicit_draft_model_path(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(
                            draft_model_path=Path("/fake/draft.gguf"),
                            timeout=1,
                        )
        assert result is mock_proc

    def test_start_no_kv_offload_disabled(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(
                            no_kv_offload=False,
                            timeout=1,
                        )
        assert result is mock_proc

    def test_start_server_early_exit_with_stdout_stderr(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.communicate.return_value = ("stdout text", "stderr text")
        with patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc):
                        result = start_server(timeout=1)
        assert result is None


class TestDraftModeContextCapping:
    """Regression coverage for the double -c flag bug: a second, hardcoded
    -c 24576 used to be appended whenever a draft model was active, always
    silently overriding whatever the caller actually requested — raising a
    smaller request, lowering a larger one, always landing on exactly
    24576 regardless of intent. Fixed to cap (never raise) via
    effective_context, with only one -c flag ever emitted."""

    def _start_and_capture_cmd(
        self, *, context: int, draft_model_path: Path | None | bool
    ) -> list[str]:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]):
            with patch(
                "godspeed.tools.llamacpp_manager._find_server_binary",
                return_value=Path("/fake/llama-server"),
            ):
                with patch(
                    "godspeed.tools.llamacpp_manager._find_model",
                    return_value=Path("/fake/model.gguf"),
                ):
                    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                        result = start_server(
                            context=context,
                            draft_model_path=draft_model_path,
                            timeout=1,
                        )
        assert result is mock_proc
        cmd: list[str] = mock_popen.call_args[0][0]
        return cmd

    def test_no_draft_model_context_passed_through_unchanged(self) -> None:
        cmd = self._start_and_capture_cmd(context=65536, draft_model_path=False)
        assert _all_flag_values(cmd, "-c") == ["65536"]

    def test_draft_model_default_context_gets_capped(self) -> None:
        cmd = self._start_and_capture_cmd(
            context=DEFAULT_CONTEXT, draft_model_path=Path("/fake/draft.gguf")
        )
        assert DEFAULT_CONTEXT > DRAFT_MODE_MAX_CONTEXT
        assert _all_flag_values(cmd, "-c") == [str(DRAFT_MODE_MAX_CONTEXT)]

    def test_draft_model_small_explicit_context_not_raised(self) -> None:
        # The exact inverse of the old bug: a request smaller than the cap
        # used to get silently bumped UP to 24576. Must stay as requested.
        cmd = self._start_and_capture_cmd(context=8192, draft_model_path=Path("/fake/draft.gguf"))
        assert _all_flag_values(cmd, "-c") == ["8192"]

    def test_draft_model_context_at_cap_boundary_unchanged(self) -> None:
        cmd = self._start_and_capture_cmd(
            context=DRAFT_MODE_MAX_CONTEXT, draft_model_path=Path("/fake/draft.gguf")
        )
        assert _all_flag_values(cmd, "-c") == [str(DRAFT_MODE_MAX_CONTEXT)]

    def test_draft_model_large_explicit_context_capped(self) -> None:
        cmd = self._start_and_capture_cmd(context=131072, draft_model_path=Path("/fake/draft.gguf"))
        assert _all_flag_values(cmd, "-c") == [str(DRAFT_MODE_MAX_CONTEXT)]

    def test_c_flag_never_appears_more_than_once(self) -> None:
        # Direct regression guard for the actual bug shape, independent of
        # which value wins.
        cmd = self._start_and_capture_cmd(
            context=DEFAULT_CONTEXT, draft_model_path=Path("/fake/draft.gguf")
        )
        assert cmd.count("-c") == 1


class TestGetServerStatusExtended:
    """Extended get_server_status tests."""

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_running_with_version_from_props(
        self, mock_request, mock_urlopen, mock_is_running
    ) -> None:
        models_resp = MagicMock()
        models_resp.read.return_value = b'{"data": [{"id": "test-model"}]}'
        props_resp = MagicMock()
        props_resp.read.return_value = (
            b'{"version": "b9066", "default_generation_settings": {"n_predict": 256}}'
        )

        mock_urlopen.return_value.__enter__.side_effect = [models_resp, props_resp]

        status = get_server_status()
        assert status["running"] is True
        assert status["model"] == "test-model"
        assert status["version"] == "b9066"
        assert status["default_generation_settings"] == {"n_predict": 256}

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_models_error_graceful(self, mock_request, mock_urlopen, mock_is_running) -> None:
        mock_urlopen.return_value.__enter__.side_effect = Exception("failed to read models")

        status = get_server_status()
        assert status["running"] is True
        assert status["model"] is None

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_empty_models_list(self, mock_request, mock_urlopen, mock_is_running) -> None:
        models_resp = MagicMock()
        models_resp.read.return_value = b'{"data": []}'

        mock_urlopen.return_value.__enter__.return_value = models_resp

        status = get_server_status()
        assert status["running"] is True
        assert status["model"] is None

    @patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=True)
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_models_dict_without_data_key(
        self, mock_request, mock_urlopen, mock_is_running
    ) -> None:
        models_resp = MagicMock()
        models_resp.read.return_value = b'{"object": "list"}'

        mock_urlopen.return_value.__enter__.return_value = models_resp

        status = get_server_status()
        assert status["running"] is True
        assert status["model"] is None


class TestLlamaCppToolExtended:
    """Extended tests for LlamaCpp tool."""

    @pytest.mark.asyncio
    async def test_status_running_no_model(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch(
            "godspeed.tools.llamacpp_manager.get_server_status",
            return_value={
                "running": True,
                "url": "http://127.0.0.1:8080",
                "model": None,
                "version": None,
            },
        ):
            result = await tool.execute({"action": "status"}, tool_context)
            assert "RUNNING" in result.output
            assert "Model:" not in result.output

    @pytest.mark.asyncio
    async def test_status_not_running_verbose(self, tool_context: Any) -> None:
        tool = LlamaCppTool()
        with patch(
            "godspeed.tools.llamacpp_manager.get_server_status",
            return_value={
                "running": False,
                "url": "http://127.0.0.1:8080",
                "model": None,
                "version": None,
            },
        ):
            result = await tool.execute({"action": "status"}, tool_context)
            assert "NOT RUNNING" in result.output
            assert "start" in result.output.lower()
