"""Tests for kernel-level sandbox enforcement (src/godspeed/sandbox/kernel.py).

Core contract under test: **never overclaim enforcement**.  Every report
must be honest about what the current platform actually enforces, and the
shell tool must fail CLOSED when kernel sandboxing was requested but cannot
be enforced.

All tests run on Windows CI: preexec callables are never invoked here, and
platform-specific code paths are exercised via monkeypatched ``sys.platform``
plus stubbed libc/job-object surfaces.  One Linux-only smoke test spawns a
real Landlock-sandboxed process and skips when Landlock is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from godspeed.sandbox import kernel as kernel_module
from godspeed.sandbox.kernel import (
    EnforcementReport,
    ExecutionPlan,
    KernelEnforcement,
    SandboxConstructionError,
    _read_network_status,
    _sb_escape,
    format_enforcement_header,
    generate_seatbelt_profile,
    plan_execution,
    probe_landlock_abi,
)
from godspeed.sandbox.policy_types import SandboxPolicy
from godspeed.tools.base import ToolContext
from godspeed.tools.shell import ShellTool


def _policy(**overrides: Any) -> SandboxPolicy:
    """SandboxPolicy with kernel_enforced=True and sane test defaults."""
    defaults: dict[str, Any] = {"kernel_enforced": True, "enable_network": False}
    defaults.update(overrides)
    return SandboxPolicy(**defaults)


class _FakeJobObject:
    """Job-object stand-in for non-Windows runs (creationflags is ignored
    there, so no resume is needed)."""

    def __init__(self) -> None:
        self.closed = False
        self.assigned = False

    def assign(self, proc: Any) -> None:
        self.assigned = True

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Honesty invariants — the core contract
# ---------------------------------------------------------------------------


def test_windows_report_is_honest_about_lifetime_only() -> None:
    report = EnforcementReport(
        strategy=KernelEnforcement.JOB_OBJECT,
        enforced=False,
        verified=False,
        network_restricted=False,
        details=[
            "process-tree lifetime bounded via Job Object (KILL_ON_JOB_CLOSE); "
            "filesystem/network access NOT restricted",
        ],
    )
    assert report.enforced is False
    assert report.verified is False
    assert report.network_restricted is False
    assert "NOT restricted" in report.details[0]


def test_linux_report_never_claims_enforcement_when_abi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "probe_landlock_abi", lambda: None)
    plan = plan_execution(["sh", "-c", "true"], cwd=Path(tempfile.gettempdir()), policy=_policy())
    assert plan.report.strategy == KernelEnforcement.LANDLOCK_SECCOMP
    assert plan.report.enforced is False
    assert plan.report.verified is False
    assert any("unavailable" in d for d in plan.report.details)


def test_unsupported_platform_report_is_unenforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "plan9")
    plan = plan_execution(["cmd"], cwd=Path(tempfile.gettempdir()), policy=_policy())
    assert plan.report.strategy == KernelEnforcement.NONE
    assert plan.report.enforced is False
    assert plan.report.verified is False


# ---------------------------------------------------------------------------
# format_enforcement_header
# ---------------------------------------------------------------------------


def test_header_job_object_is_explicit_about_lifetime_only() -> None:
    report = EnforcementReport(
        strategy=KernelEnforcement.JOB_OBJECT,
        enforced=False,
        verified=False,
        network_restricted=False,
    )
    assert format_enforcement_header(report) == (
        "sandbox: job-object (lifetime-only, fs/network NOT enforced)"
    )


def test_header_none_is_unenforced() -> None:
    report = EnforcementReport(
        strategy=KernelEnforcement.NONE,
        enforced=False,
        verified=False,
    )
    assert format_enforcement_header(report) == "sandbox: unenforced"


def test_header_landlock_network_enforced() -> None:
    report = EnforcementReport(
        strategy=KernelEnforcement.LANDLOCK_SECCOMP,
        enforced=True,
        verified=True,
        network_restricted=True,
    )
    assert format_enforcement_header(report) == (
        "sandbox: landlock (fs enforced, network enforced)"
    )


def test_header_landlock_network_not_enforced() -> None:
    report = EnforcementReport(
        strategy=KernelEnforcement.LANDLOCK_SECCOMP,
        enforced=True,
        verified=True,
        network_restricted=False,
    )
    assert format_enforcement_header(report) == (
        "sandbox: landlock (fs enforced, network NOT enforced)"
    )


def test_header_seatbelt() -> None:
    report = EnforcementReport(
        strategy=KernelEnforcement.SEATBELT,
        enforced=True,
        verified=False,
        network_restricted=True,
    )
    assert format_enforcement_header(report) == (
        "sandbox: seatbelt (fs enforced, network enforced)"
    )


# ---------------------------------------------------------------------------
# _read_network_status — honest network reporting
# ---------------------------------------------------------------------------


def _pipe_with(data: bytes) -> tuple[int, int]:
    read_fd, write_fd = os.pipe()
    if data:
        os.write(write_fd, data)
    os.close(write_fd)
    return read_fd, write_fd


def test_network_status_reports_restricted() -> None:
    base = EnforcementReport(
        strategy=KernelEnforcement.LANDLOCK_SECCOMP,
        enforced=True,
        verified=True,
    )
    read_fd, _ = _pipe_with(b"1")
    updated = _read_network_status(read_fd, base)
    assert updated.network_restricted is True
    assert any("unshare(CLONE_NEWNET)" in d for d in updated.details)


def test_network_status_reports_not_restricted() -> None:
    base = EnforcementReport(
        strategy=KernelEnforcement.LANDLOCK_SECCOMP,
        enforced=True,
        verified=True,
    )
    read_fd, _ = _pipe_with(b"0")
    updated = _read_network_status(read_fd, base)
    assert updated.network_restricted is False
    assert any("NOT restricted" in d for d in updated.details)


def test_network_status_reports_unverified_on_eof() -> None:
    base = EnforcementReport(
        strategy=KernelEnforcement.LANDLOCK_SECCOMP,
        enforced=True,
        verified=True,
    )
    read_fd, _ = _pipe_with(b"")
    updated = _read_network_status(read_fd, base)
    assert updated.network_restricted is False
    assert any("unverified" in d for d in updated.details)


# ---------------------------------------------------------------------------
# Seatbelt profile generation
# ---------------------------------------------------------------------------


def test_seatbelt_profile_denies_network_when_restricted(tmp_path: Path) -> None:
    profile = generate_seatbelt_profile(tmp_path, tmp_path, network_restricted=True)
    assert "(deny network*)" in profile
    assert "(allow network*)" not in profile
    assert f'(allow file-write* (subpath "{_sb_escape(str(tmp_path))}"))' in profile


def test_seatbelt_profile_allows_network_when_not_restricted(tmp_path: Path) -> None:
    profile = generate_seatbelt_profile(tmp_path, tmp_path, network_restricted=False)
    assert "(allow network*)" in profile
    assert "(deny network*)" not in profile


def test_seatbelt_profile_escapes_quotes_and_backslashes(tmp_path: Path) -> None:
    weird = tmp_path / 'we"ird\\dir'
    profile = generate_seatbelt_profile(weird, tmp_path, network_restricted=True)
    assert 'we\\"ird\\\\dir' in profile


def test_macos_plan_uses_sandbox_exec_and_cleans_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "darwin")
    monkeypatch.setattr(kernel_module.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    plan = plan_execution(["echo", "hi"], cwd=tmp_path, policy=_policy())
    assert plan.argv[0] == "/usr/bin/sandbox-exec"
    assert plan.argv[1] == "-f"
    profile_path = Path(plan.argv[2])
    assert profile_path.suffix == ".sb"
    assert profile_path.exists()
    assert plan.report.strategy == KernelEnforcement.SEATBELT
    assert plan.report.enforced is True
    assert plan.report.verified is False
    assert plan.report.network_restricted is True
    assert plan.cleanup is not None
    plan.cleanup()
    assert not profile_path.exists()


def test_macos_plan_fails_openly_when_sandbox_exec_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "darwin")
    monkeypatch.setattr(kernel_module.shutil, "which", lambda name: None)
    plan = plan_execution(["echo", "hi"], cwd=tmp_path, policy=_policy())
    assert plan.report.strategy == KernelEnforcement.SEATBELT
    assert plan.report.enforced is False
    assert any("sandbox-exec not found" in d for d in plan.report.details)


# ---------------------------------------------------------------------------
# Execution-plan construction (preexec never invoked)
# ---------------------------------------------------------------------------


def test_windows_plan_is_lifetime_only_and_suspended(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "win32")
    if sys.platform != "win32":
        monkeypatch.setattr(kernel_module, "_JobObject", _FakeJobObject)
    cmd = ["cmd", "/c", "echo hi"]
    plan = plan_execution(cmd, cwd=tmp_path, policy=_policy())
    assert plan.argv == cmd
    assert plan.preexec is None
    assert plan.creationflags == 0x00000004  # CREATE_SUSPENDED
    assert plan.post_start is not None
    assert plan.cleanup is not None
    assert plan.report.strategy == KernelEnforcement.JOB_OBJECT
    assert plan.report.enforced is False
    assert plan.report.network_restricted is False
    assert any("NOT restricted" in d for d in plan.report.details)


def test_linux_plan_uses_landlock_when_abi_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "probe_landlock_abi", lambda: 3)
    monkeypatch.setattr(kernel_module, "_create_ruleset", lambda handled: 7)
    monkeypatch.setattr(kernel_module, "_add_path_beneath_rule", lambda fd, path, allowed: None)
    cmd = ["sh", "-c", "echo hi"]
    plan = plan_execution(cmd, cwd=tmp_path, policy=_policy())
    assert plan.argv == cmd
    assert plan.preexec is not None
    assert plan.report.strategy == KernelEnforcement.LANDLOCK_SECCOMP
    assert plan.report.enforced is True
    assert plan.report.verified is True
    assert plan.post_start is not None
    assert plan.cleanup is not None


def test_linux_plan_network_allowed_when_policy_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "probe_landlock_abi", lambda: 1)
    monkeypatch.setattr(kernel_module, "_create_ruleset", lambda handled: 7)
    monkeypatch.setattr(kernel_module, "_add_path_beneath_rule", lambda fd, path, allowed: None)
    plan = plan_execution(["sh", "-c", "true"], cwd=tmp_path, policy=_policy(enable_network=True))
    assert plan.report.enforced is True
    assert any("network allowed by policy" in d for d in plan.report.details)
    # No status pipe when network is allowed → post_start returns None.
    assert plan.post_start is not None
    assert plan.post_start(None) is None


# ---------------------------------------------------------------------------
# probe_landlock_abi
# ---------------------------------------------------------------------------


class _FakeLibc:
    def __init__(self, result: int) -> None:
        self.result = result

    def syscall(self, number: int, *args: Any) -> int:
        return self.result


def test_abi_probe_returns_none_when_syscall_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "_LANDLOCK_SYSFS_PATH", Path("Z:/nonexistent/landlock"))
    monkeypatch.setattr(kernel_module, "_LIBC", _FakeLibc(-1))
    assert probe_landlock_abi() is None


def test_abi_probe_returns_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "_LANDLOCK_SYSFS_PATH", Path("Z:/nonexistent/landlock"))
    monkeypatch.setattr(kernel_module, "_LIBC", _FakeLibc(3))
    assert probe_landlock_abi() == 3


def test_abi_probe_treats_zero_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "_LANDLOCK_SYSFS_PATH", Path("Z:/nonexistent/landlock"))
    monkeypatch.setattr(kernel_module, "_LIBC", _FakeLibc(0))
    assert probe_landlock_abi() is None


def test_abi_probe_non_linux_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "win32")
    assert probe_landlock_abi() is None


# ---------------------------------------------------------------------------
# Shell-tool integration — fail-closed + honest header
# ---------------------------------------------------------------------------


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-kernel")


@pytest.mark.asyncio
async def test_windows_kernel_mode_produces_honest_header(
    monkeypatch: pytest.MonkeyPatch, tool: ShellTool, ctx: ToolContext
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "win32")
    if sys.platform != "win32":
        monkeypatch.setattr(kernel_module, "_JobObject", _FakeJobObject)
    ctx.sandbox = _policy()
    result = await tool.execute({"command": "echo hello"}, ctx)
    assert result.is_error is False
    assert result.output.startswith("sandbox: job-object (lifetime-only, fs/network NOT enforced)")


@pytest.mark.asyncio
async def test_sandbox_construction_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tool: ShellTool, ctx: ToolContext
) -> None:
    def _boom(cmd: list[str], cwd: Path, policy: SandboxPolicy) -> ExecutionPlan:
        raise SandboxConstructionError("boom")

    monkeypatch.setattr("godspeed.tools.shell.plan_execution", _boom)
    ctx.sandbox = _policy()
    result = await tool.execute({"command": "echo hello"}, ctx)
    assert result.is_error is True
    assert "Sandbox construction failed" in (result.error or "")


@pytest.mark.asyncio
async def test_kernel_mode_unavailable_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tool: ShellTool, ctx: ToolContext
) -> None:
    monkeypatch.setattr(kernel_module.sys, "platform", "linux")
    monkeypatch.setattr(kernel_module, "probe_landlock_abi", lambda: None)
    ctx.sandbox = _policy()
    result = await tool.execute({"command": "echo hello"}, ctx)
    assert result.is_error is True
    assert "Sandbox unavailable" in (result.error or "")


@pytest.mark.asyncio
async def test_kernel_mode_disabled_runs_unsandboxed(tool: ShellTool, ctx: ToolContext) -> None:
    ctx.sandbox = SandboxPolicy(kernel_enforced=False)
    result = await tool.execute({"command": "echo hello"}, ctx)
    assert result.is_error is False
    assert "sandbox:" not in result.output


# ---------------------------------------------------------------------------
# Linux-only real smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only Landlock smoke test")
def test_linux_landlock_real_execution() -> None:
    if not probe_landlock_abi():
        pytest.skip("Landlock unavailable on this kernel")
    plan = plan_execution(
        ["sh", "-c", "echo ok"],
        cwd=Path(tempfile.gettempdir()),
        policy=_policy(),
    )
    assert plan.report.enforced is True
    proc = subprocess.Popen(
        plan.argv,
        preexec_fn=plan.preexec,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if plan.post_start is not None:
            plan.post_start(proc)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if plan.cleanup is not None:
            plan.cleanup()
    assert proc.returncode == 0, f"stderr: {stderr}"
    assert "ok" in stdout
