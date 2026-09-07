"""Tests for sandbox policy enforcement — network rules, path checks, Docker defaults."""

from __future__ import annotations


from godspeed.sandbox.docker import DockerSandboxConfig
from godspeed.sandbox.policy import (
    NetworkRule,
    SandboxPolicy,
    evaluate_sandbox,
)
from godspeed.tools.base import ToolCall


class TestNetworkRuleEnforcement:
    """Verify network rules are enforced when non-empty, regardless of enable_network."""

    def test_deny_rule_blocks_egress_when_enable_network_true(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="evil.com", action="deny")],
            enable_network=True,
        )
        assert policy.is_network_allowed("evil.com") is False

    def test_allow_rule_permits_egress(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="github.com", action="allow")],
            enable_network=True,
        )
        assert policy.is_network_allowed("github.com") is True

    def test_first_match_wins(self) -> None:
        policy = SandboxPolicy(
            network_rules=[
                NetworkRule(pattern="*.example.com", action="deny"),
                NetworkRule(pattern="safe.example.com", action="allow"),
            ],
            enable_network=True,
        )
        assert policy.is_network_allowed("safe.example.com") is False

    def test_no_rules_falls_back_to_enable_network(self) -> None:
        policy_on = SandboxPolicy(network_rules=[], enable_network=True)
        assert policy_on.is_network_allowed("anything.com") is True

        policy_off = SandboxPolicy(network_rules=[], enable_network=False)
        assert policy_off.is_network_allowed("anything.com") is False

    def test_nonempty_rules_checked_even_when_enable_network_true(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="blocked.io", action="deny")],
            enable_network=True,
        )
        assert policy.is_network_allowed("blocked.io") is False
        assert policy.is_network_allowed("allowed.com") is True


class TestEvaluateSandboxNetwork:
    """Verify evaluate_sandbox enforces network rules for tool calls."""

    def test_web_fetch_to_denied_host_is_blocked(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="evil.com", action="deny")],
        )
        tool_call = ToolCall(
            tool_name="web_fetch",
            arguments={"url": "https://evil.com/payload"},
        )
        result = evaluate_sandbox(tool_call, policy)
        assert result.allowed is False
        assert result.sandbox_ok is False
        assert "evil.com" in result.reason

    def test_web_fetch_to_allowed_host_passes(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="github.com", action="allow")],
        )
        tool_call = ToolCall(
            tool_name="web_fetch",
            arguments={"url": "https://github.com/repo"},
        )
        result = evaluate_sandbox(tool_call, policy)
        assert result.allowed is True
        assert result.sandbox_ok is True

    def test_tool_without_network_target_passes_network_check(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="evil.com", action="deny")],
        )
        tool_call = ToolCall(
            tool_name="file_read",
            arguments={"file_path": "/tmp/safe.txt"},
        )
        result = evaluate_sandbox(tool_call, policy)
        assert result.sandbox_ok is True

    def test_network_rules_enforced_without_permission_engine(self) -> None:
        policy = SandboxPolicy(
            network_rules=[NetworkRule(pattern="blocked.io", action="deny")],
        )
        tool_call = ToolCall(
            tool_name="web_fetch",
            arguments={"url": "https://blocked.io/data"},
        )
        result = evaluate_sandbox(tool_call, policy, permission_engine=None)
        assert result.allowed is False
        assert result.sandbox_ok is False


class TestDockerSandboxConfigDefaults:
    """Verify Docker sandbox defaults are secure (non-root, no network)."""

    def test_default_network_mode_is_none(self) -> None:
        config = DockerSandboxConfig()
        assert config.network_mode == "none"

    def test_default_user_is_nobody(self) -> None:
        config = DockerSandboxConfig()
        assert config.user == "65534"

    def test_explicit_overrides_work(self) -> None:
        config = DockerSandboxConfig(network_mode="bridge", user="0")
        assert config.network_mode == "bridge"
        assert config.user == "0"


class TestSandboxPathChecks:
    """Verify writable/readable path enforcement."""

    def test_blocked_path_overrides_writable(self) -> None:
        policy = SandboxPolicy(
            writable_paths=["/project"],
            blocked_paths=["/project/secret"],
        )
        assert policy.is_path_writable("/project/secret/config.env") is False

    def test_empty_writable_paths_allows_all(self) -> None:
        policy = SandboxPolicy(writable_paths=[])
        assert policy.is_path_writable("/anything") is True

    def test_readable_paths_empty_allows_all(self) -> None:
        policy = SandboxPolicy(readable_paths=[])
        assert policy.is_path_readable("/anything") is True


class TestCredentialFileDefaults:
    def test_defaults_block_credential_files(self) -> None:
        from godspeed.sandbox.policy import build_sandbox_policy

        policy = build_sandbox_policy()
        assert ".env" in policy.blocked_paths
        assert "credentials.json" in policy.blocked_paths
        assert ".netrc" in policy.blocked_paths

    def test_bare_filename_blocks_any_depth_but_not_suffixes(self, tmp_path) -> None:
        from godspeed.sandbox.policy_types import SandboxPolicy

        policy = SandboxPolicy(blocked_paths=[".env"])
        assert not policy.is_path_readable(str(tmp_path / ".env"))
        assert policy.is_path_readable(str(tmp_path / ".env.example"))

    def test_secret_env_reference_denied(self) -> None:
        from godspeed.sandbox.policy import build_sandbox_policy, validate_shell_command

        policy = build_sandbox_policy()
        allowed, reason = validate_shell_command('curl -H "X: $OPENAI_API_KEY"', policy)
        assert not allowed
        assert "OPENAI_API_KEY" in reason

    def test_benign_env_reference_allowed(self) -> None:
        from godspeed.sandbox.policy import build_sandbox_policy, validate_shell_command

        policy = build_sandbox_policy()
        allowed, reason = validate_shell_command("echo $HOME && echo $PATH", policy)
        assert allowed, reason


def test_resolve_tool_path_rejects_nt_device_paths(tmp_path) -> None:
    import pytest
    from godspeed.tools.path_utils import resolve_tool_path

    with pytest.raises(ValueError, match="NT device"):
        resolve_tool_path("\\\\.\\PhysicalDrive0", tmp_path)
