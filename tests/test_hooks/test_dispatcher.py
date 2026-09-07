"""Tests for the hook dispatcher (auto-approve/deny, adapters)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from godspeed.hooks import HookEvent
from godspeed.hooks.config import HookDefinition
from godspeed.hooks.dispatcher import (
    AutoApproveRule,
    AutoDenyRule,
    AgentHooksAdapter,
    DispatcherConfig,
    HookDispatcher,
    FlatListHooksAdapter,
    filter_trusted_hooks,
    is_trusted_hook_source,
)
from godspeed.hooks.executor import HookExecutor
from godspeed.sandbox.policy import SandboxPolicy
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import RiskLevel, ToolCall


def _make_dispatcher(
    tmp_path: Path,
    *,
    auto_approve: list[AutoApproveRule] | None = None,
    auto_deny: list[AutoDenyRule] | None = None,
    sandbox: SandboxPolicy | None = None,
    permission_engine: PermissionEngine | None = None,
) -> HookDispatcher:
    executor = HookExecutor(hooks=[], cwd=tmp_path, session_id="test-session")
    config = DispatcherConfig(
        auto_approve=auto_approve or [],
        auto_deny=auto_deny or [],
        sandbox=sandbox or SandboxPolicy(),
    )
    return HookDispatcher(
        executor=executor,
        permission_engine=permission_engine,
        config=config,
    )


class TestAutoApproveRule:
    """Test AutoApproveRule matching."""

    def test_wildcard_matches_all(self) -> None:
        rule = AutoApproveRule(event="pre_tool_call")
        assert rule.matches("pre_tool_call")
        assert rule.matches("pre_tool_call", "shell")

    def test_tool_pattern(self) -> None:
        rule = AutoApproveRule(event="pre_tool_call", tool_pattern="file_*")
        assert rule.matches("pre_tool_call", "file_read")
        assert rule.matches("pre_tool_call", "file_write")
        assert rule.matches("pre_tool_call", "shell") is False

    def test_event_mismatch(self) -> None:
        rule = AutoApproveRule(event="pre_tool_call")
        assert rule.matches("post_tool_call") is False


class TestAutoDenyRule:
    """Test AutoDenyRule matching."""

    def test_wildcard_matches_all(self) -> None:
        rule = AutoDenyRule(event="pre_tool_call")
        assert rule.matches("pre_tool_call", "shell")

    def test_tool_pattern(self) -> None:
        rule = AutoDenyRule(event="pre_tool_call", tool_pattern="shell")
        assert rule.matches("pre_tool_call", "shell")
        assert rule.matches("pre_tool_call", "file_read") is False


class TestDispatcherConfig:
    """Test DispatcherConfig loading."""

    def test_from_json(self, tmp_path: Path) -> None:
        path = tmp_path / "dispatcher.json"
        path.write_text(
            json.dumps(
                {
                    "auto_approve": [
                        {
                            "event": "pre_tool_call",
                            "tool_pattern": "file_read",
                            "reason": "read-only",
                        }
                    ],
                    "auto_deny": [
                        {
                            "event": "pre_tool_call",
                            "tool_pattern": "shell",
                            "reason": "blocked",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        config = DispatcherConfig.from_json(path)
        assert len(config.auto_approve) == 1
        assert len(config.auto_deny) == 1
        assert config.auto_approve[0].tool_pattern == "file_read"
        assert config.auto_deny[0].tool_pattern == "shell"

    def test_from_json_missing_file(self, tmp_path: Path) -> None:
        config = DispatcherConfig.from_json(tmp_path / "nope.json")
        assert config.auto_approve == []
        assert config.auto_deny == []

    def test_from_json_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        config = DispatcherConfig.from_json(path)
        assert config.auto_approve == []

    def test_from_dict(self) -> None:
        config = DispatcherConfig.from_dict(
            {
                "auto_approve": [{"event": "pre_tool_call", "tool_pattern": "*"}],
                "auto_deny": [{"event": "pre_tool_call", "tool_pattern": "shell"}],
            }
        )
        assert len(config.auto_approve) == 1
        assert len(config.auto_deny) == 1


class TestHookDispatcher:
    """Test the HookDispatcher orchestration."""

    def test_auto_deny_blocks(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(
            tmp_path,
            auto_deny=[AutoDenyRule(event="pre_tool_call", tool_pattern="shell")],
        )
        assert dispatcher.run_pre_tool("shell") is False
        assert dispatcher.run_pre_tool("file_read") is True

    def test_auto_approve_skips_hooks(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(
            tmp_path,
            auto_approve=[AutoApproveRule(event="pre_tool_call", tool_pattern="file_read")],
        )
        assert dispatcher.run_pre_tool("file_read") is True

    def test_evaluate_tool_call_auto_deny(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(
            tmp_path,
            auto_deny=[AutoDenyRule(event="pre_tool_call", tool_pattern="shell")],
        )
        tool_call = ToolCall(tool_name="shell", arguments={"command": "ls"})
        result = dispatcher.evaluate_tool_call(tool_call)
        assert result.allowed is False
        assert "Auto-deny" in result.reason

    def test_evaluate_tool_call_auto_approve(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(
            tmp_path,
            auto_approve=[AutoApproveRule(event="pre_tool_call", tool_pattern="file_read")],
        )
        tool_call = ToolCall(tool_name="file_read", arguments={"file_path": "a.py"})
        result = dispatcher.evaluate_tool_call(tool_call)
        assert result.allowed is True
        assert "Auto-approve" in result.reason

    def test_evaluate_tool_call_sandbox_deny(self, tmp_path: Path) -> None:
        sandbox = SandboxPolicy(writable_paths=[str(tmp_path)])
        dispatcher = _make_dispatcher(tmp_path, sandbox=sandbox)
        tool_call = ToolCall(
            tool_name="file_write",
            arguments={"file_path": str(tmp_path.parent / "out.txt")},
        )
        result = dispatcher.evaluate_tool_call(tool_call)
        assert result.allowed is False
        assert result.sandbox_ok is False

    def test_evaluate_tool_call_permission_deny(self, tmp_path: Path) -> None:
        engine = PermissionEngine(
            deny_patterns=["shell(rm *)"],
            tool_risk_levels={"shell": RiskLevel.HIGH},
        )
        dispatcher = _make_dispatcher(tmp_path, permission_engine=engine)
        tool_call = ToolCall(tool_name="shell", arguments={"command": "rm -rf /"})
        result = dispatcher.evaluate_tool_call(tool_call)
        assert result.allowed is False
        assert result.permission_decision is not None

    def test_fire_respects_auto_deny(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(
            tmp_path,
            auto_deny=[AutoDenyRule(event="pre_tool_call", tool_pattern="shell")],
        )
        result = dispatcher.fire(HookEvent.PRE_TOOL_CALL, tool_name="shell")
        assert result is False

    def test_fire_respects_auto_approve(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(
            tmp_path,
            auto_approve=[AutoApproveRule(event="pre_tool_call", tool_pattern="file_read")],
        )
        result = dispatcher.fire(HookEvent.PRE_TOOL_CALL, tool_name="file_read")
        assert result is None

    def test_run_stop(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(tmp_path)
        dispatcher.run_stop()  # should not raise

    def test_run_notification(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(tmp_path)
        dispatcher.run_notification("hello")  # should not raise

    def test_run_session(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(tmp_path)
        dispatcher.run_pre_session()
        dispatcher.run_post_session()

    def test_run_post_tool(self, tmp_path: Path) -> None:
        dispatcher = _make_dispatcher(tmp_path)
        dispatcher.run_post_tool("shell", result="ok")


class TestAgentHooksAdapter:
    """Test hook format adapter translation."""

    def test_translate_basic(self) -> None:
        config = {
            "hooks": {
                "PreToolUse": [{"type": "command", "command": "echo pre"}],
                "PostToolUse": [{"type": "command", "command": "echo post"}],
                "Stop": [{"type": "command", "command": "echo stop"}],
            }
        }
        hooks = AgentHooksAdapter.translate(config)
        assert len(hooks) == 3
        assert hooks[0].event == "pre_tool_call"
        assert hooks[1].event == "post_tool_call"
        assert hooks[2].event == "stop"

    def test_translate_unknown_event_skipped(self) -> None:
        config = {"hooks": {"UnknownEvent": [{"type": "command", "command": "echo x"}]}}
        hooks = AgentHooksAdapter.translate(config)
        assert hooks == []

    def test_translate_empty_command_skipped(self) -> None:
        config = {"hooks": {"PreToolUse": [{"type": "command", "command": ""}]}}
        hooks = AgentHooksAdapter.translate(config)
        assert hooks == []

    def test_from_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The pre-trust gate only accepts user-owned config locations, so
        # simulate one by pointing Path.home at the tmp_path.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        path = tmp_path / "hooks-dict.json"
        path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [{"type": "command", "command": "echo pre"}],
                        "SessionStart": [{"type": "command", "command": "echo start"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        hooks = AgentHooksAdapter.from_json(path)
        assert len(hooks) == 2
        assert hooks[0].event == "pre_tool_call"
        assert hooks[1].event == "session_start"

    def test_from_json_missing(self, tmp_path: Path) -> None:
        hooks = AgentHooksAdapter.from_json(tmp_path / "nope.json")
        assert hooks == []


class TestFlatListHooksAdapter:
    """Test flat-list hooks adapter translation."""

    def test_translate_basic(self) -> None:
        config = [
            {"event": "pre_tool_use", "command": "echo pre"},
            {"event": "post_tool_use", "command": "echo post"},
            {"event": "session_start", "command": "echo start"},
        ]
        hooks = FlatListHooksAdapter.translate(config)
        assert len(hooks) == 3
        assert hooks[0].event == "pre_tool_call"
        assert hooks[1].event == "post_tool_call"
        assert hooks[2].event == "session_start"

    def test_translate_unknown_event_skipped(self) -> None:
        config = [{"event": "bogus_event", "command": "echo x"}]
        hooks = FlatListHooksAdapter.translate(config)
        assert hooks == []

    def test_from_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The pre-trust gate only accepts user-owned config locations, so
        # simulate one by pointing Path.home at the tmp_path.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        path = tmp_path / "hooks-list.json"
        path.write_text(
            json.dumps([{"event": "stop", "command": "echo stop"}]),
            encoding="utf-8",
        )
        hooks = FlatListHooksAdapter.from_json(path)
        assert len(hooks) == 1
        assert hooks[0].event == "stop"

    def test_from_json_not_list(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"hooks": []}), encoding="utf-8")
        hooks = FlatListHooksAdapter.from_json(path)
        assert hooks == []


class TestPreTrustGate:
    """Pre-trust gate: hooks from untrusted sources fail closed."""

    def test_empty_source_is_trusted(self) -> None:
        assert is_trusted_hook_source("") is True

    def test_home_config_source_is_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        source = str(fake_home / ".agent-hooks" / "hooks.json")
        assert is_trusted_hook_source(source) is True

    def test_project_source_is_untrusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        project = tmp_path / "repo"
        project.mkdir()
        hook_file = project / ".agent-hooks" / "hooks.json"
        hook_file.parent.mkdir(parents=True)
        hook_file.write_text("{}", encoding="utf-8")
        assert is_trusted_hook_source(str(hook_file)) is False

    def test_unknown_absolute_source_is_untrusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        assert is_trusted_hook_source(str(tmp_path / "elsewhere" / "h.json")) is False

    def test_filter_drops_untrusted_keeps_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        project = tmp_path / "repo"
        project.mkdir()
        hooks = [
            HookDefinition(event=HookEvent.PRE_TOOL_CALL, command="echo settings"),
            HookDefinition(
                event=HookEvent.PRE_TOOL_CALL,
                command="evil",
                source=str(project / "hooks.json"),
            ),
        ]
        kept = filter_trusted_hooks(hooks)
        assert len(kept) == 1
        assert kept[0].command == "echo settings"

    def test_dict_adapter_stamps_source(self, tmp_path: Path) -> None:
        config = {"hooks": {"PreToolUse": [{"type": "command", "command": "echo hi"}]}}
        defs = AgentHooksAdapter.translate(config, source=str(tmp_path / "h.json"))
        assert all(d.source == str(tmp_path / "h.json") for d in defs)

    def test_dict_from_json_drops_project_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "repo"
        hook_file = project / "hooks.json"
        hook_file.parent.mkdir(parents=True)
        hook_file.write_text(
            json.dumps({"hooks": {"PreToolUse": [{"command": "evil"}]}}), encoding="utf-8"
        )
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        defs = AgentHooksAdapter.from_json(hook_file)
        assert defs == []

    def test_list_from_json_keeps_home_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        hook_file = home / "hooks.json"
        hook_file.parent.mkdir(parents=True)
        hook_file.write_text(
            json.dumps([{"event": "pre_tool_use", "command": "echo ok"}]), encoding="utf-8"
        )
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        defs = FlatListHooksAdapter.from_json(hook_file)
        assert len(defs) == 1
        assert defs[0].command == "echo ok"
