"""Kernel-level sandbox enforcement with honest per-platform reporting.

Strategy per platform:

* **Linux / WSL2** — Landlock (filesystem write restriction via ctypes
  syscalls) plus best-effort network-namespace isolation via
  ``unshare(CLONE_NEWNET)``.  A full seccomp BPF filter is explicitly out
  of scope for v1 and reported as such.
* **macOS** — Seatbelt profile executed through ``sandbox-exec -f``
  (deprecated but still present on current macOS).
* **Windows** — Job Object lifetime bounding only (``KILL_ON_JOB_CLOSE``).
  This bounds the process-tree lifetime; it does NOT restrict filesystem
  or network access.  Reports are explicit about that.

The core contract of this module: **never overclaim enforcement**.
``EnforcementReport.enforced`` is only ``True`` when the platform strategy
actually restricts access; every limitation is surfaced in ``details`` and
in the one-line header produced by :func:`format_enforcement_header`.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from godspeed.sandbox.policy_types import SandboxPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landlock constants.
#
# Syscall numbers 444-446 come from arch/x86/entry/syscalls/syscall_64.tbl
# and are identical in the generic syscall table used by aarch64
# (include/uapi/asm-generic/unistd.h).  Access-right bit values come from
# include/uapi/linux/landlock.h (Linux 5.13+).
# ---------------------------------------------------------------------------

# landlock_create_ruleset(2) — syscall number 444
_SYS_LANDLOCK_CREATE_RULESET = 444
# landlock_add_rule(2) — syscall number 445
_SYS_LANDLOCK_ADD_RULE = 445
# landlock_restrict_self(2) — syscall number 446
_SYS_LANDLOCK_RESTRICT_SELF = 446

# LANDLOCK_CREATE_RULESET_VERSION — flag to query the ABI version
_LANDLOCK_CREATE_RULESET_VERSION = 1
# LANDLOCK_RULE_PATH_BENEATH — rule type for path-hierarchy rules
_LANDLOCK_RULE_PATH_BENEATH = 1


# struct landlock_ruleset_attr { __u64 handled_access_fs; }  (8 bytes)
class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


# struct landlock_path_beneath_attr {
#     __u64 allowed_access;
#     __s32 parent_fd;
# }  (16 bytes with natural alignment)
class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


# LANDLOCK_ACCESS_FS_* rights (ABI 1) — include/uapi/linux/landlock.h
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
# ABI 2: LANDLOCK_ACCESS_FS_REFER
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
# ABI 3: LANDLOCK_ACCESS_FS_TRUNCATE
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

# CLONE_NEWNET — include/uapi/linux/sched.h
_CLONE_NEWNET = 0x40000000

# sysfs ABI version file (kernel >= 5.13 exposes Landlock here)
_LANDLOCK_SYSFS_PATH = Path("/sys/kernel/security/landlock")


class _LibcHandle:
    """Minimal typed surface over the libc functions the sandbox needs.

    The handle is created once at import time so the preexec_fn (which runs
    in the forked child before exec) only performs direct function calls and
    never triggers lazy library loading in a potentially multithreaded
    process.
    """

    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.syscall.restype = ctypes.c_long
        self._libc.syscall.argtypes = [ctypes.c_long]
        self._libc.unshare.restype = ctypes.c_int
        self._libc.unshare.argtypes = [ctypes.c_int]

    def syscall(self, number: int, *args: Any) -> int:
        """Invoke ``syscall(2)``; *args* must be explicit ctypes objects."""
        return int(self._libc.syscall(ctypes.c_long(number), *args))

    def unshare(self, flags: int) -> int:
        """Invoke ``unshare(2)``; returns 0 on success, -1 on failure."""
        return int(self._libc.unshare(flags))


if sys.platform.startswith("linux"):
    _LIBC: _LibcHandle | None = _LibcHandle()
else:
    _LIBC = None

# ---------------------------------------------------------------------------
# Windows Job Object constants — winnt.h / winbase.h
# ---------------------------------------------------------------------------

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_CREATE_SUSPENDED = 0x00000004
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
# Required by NtResumeProcess to resume a CREATE_SUSPENDED process.
_PROCESS_SUSPEND_RESUME = 0x0800


class _LargeInteger(ctypes.Structure):
    _fields_ = [("QuadPart", ctypes.c_longlong)]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", _LargeInteger),
        ("PerJobUserTimeLimit", _LargeInteger),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


if sys.platform == "win32":
    _kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll: Any = ctypes.WinDLL("ntdll", use_last_error=True)

    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = ctypes.c_long
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
else:
    _kernel32 = None
    _ntdll = None


class KernelEnforcement(StrEnum):
    """Kernel sandboxing strategy selected for the current platform.

    ``JOB_OBJECT`` (Windows) is lifetime-only — it bounds the process-tree
    lifetime but does NOT restrict filesystem or network access.
    """

    NONE = "none"
    LANDLOCK_SECCOMP = "landlock"
    SEATBELT = "seatbelt"
    JOB_OBJECT = "job-object"


@dataclass(frozen=True)
class EnforcementReport:
    """Honest report of what a kernel sandbox actually enforces.

    Attributes:
        strategy: The platform strategy in use.
        enforced: True only when the strategy actually restricts access.
        verified: True when enforcement was positively verified.
        details: Human-readable notes; every limitation is listed here.
        network_restricted: True when network access is actually blocked.
    """

    strategy: KernelEnforcement
    enforced: bool
    verified: bool
    details: list[str] = field(default_factory=list)
    network_restricted: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    """How to execute a command under kernel sandboxing.

    Attributes:
        argv: Transformed argv (e.g. ``["sandbox-exec", "-f", profile, *cmd]``).
        preexec: POSIX ``preexec_fn`` callable (runs in the child before exec).
        env: Environment variable overrides (merged over the parent env).
        report: Honest enforcement report for this plan.
        pass_fds: Extra fds to keep open across exec (rarely needed).
        creationflags: Windows creation flags (e.g. ``CREATE_SUSPENDED``).
        post_start: Called once after Popen returns; may return an updated
            report (e.g. after the preexec status pipe is drained).
        cleanup: Called once after the process exits (closes fds/handles,
            deletes temp files).
    """

    argv: list[str]
    preexec: Callable[[], None] | None
    env: dict[str, str]
    report: EnforcementReport
    pass_fds: tuple[int, ...] = ()
    creationflags: int = 0
    post_start: Callable[[Any], EnforcementReport | None] | None = None
    cleanup: Callable[[], None] | None = None


class SandboxConstructionError(RuntimeError):
    """Raised when a kernel sandbox cannot be constructed (fail-closed)."""


def _syscall(number: int, *args: Any) -> int:
    """Invoke libc ``syscall(2)`` with explicit ctypes argument types."""
    if _LIBC is None:
        raise OSError("libc unavailable")
    return _LIBC.syscall(number, *args)


def probe_landlock_abi() -> int | None:
    """Probe the Landlock ABI version exposed by the running kernel.

    Returns the ABI version (>= 1) when Landlock is available, or ``None``
    when it is not (kernel < 5.13, sysfs not mounted, or the syscall is not
    present).  ABI 0 is treated as unsupported.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        text = _LANDLOCK_SYSFS_PATH.read_text(encoding="ascii").strip()
        abi = int(text)
        return abi if abi > 0 else None
    except (OSError, ValueError):
        pass
    if _LIBC is None:
        return None
    try:
        version = _LIBC.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except Exception as exc:
        logger.debug("landlock syscall probe failed: %s", exc)
        return None
    if version <= 0:
        return None
    return int(version)


def _landlock_write_rights(abi: int) -> int:
    """Write-related Landlock access rights supported by the given ABI."""
    rights = (
        _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_REMOVE_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_FILE
        | _LANDLOCK_ACCESS_FS_MAKE_CHAR
        | _LANDLOCK_ACCESS_FS_MAKE_DIR
        | _LANDLOCK_ACCESS_FS_MAKE_REG
        | _LANDLOCK_ACCESS_FS_MAKE_SOCK
        | _LANDLOCK_ACCESS_FS_MAKE_FIFO
        | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | _LANDLOCK_ACCESS_FS_MAKE_SYM
    )
    if abi >= 2:
        rights |= _LANDLOCK_ACCESS_FS_REFER
    if abi >= 3:
        rights |= _LANDLOCK_ACCESS_FS_TRUNCATE
    return rights


def _create_ruleset(handled_access_fs: int) -> int:
    """Create a Landlock ruleset restricting the given filesystem rights."""
    attr = _LandlockRulesetAttr(handled_access_fs)
    fd = _syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    )
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    return fd


def _add_path_beneath_rule(ruleset_fd: int, path: Path, allowed_access: int) -> None:
    """Allow *allowed_access* rights beneath *path* in the ruleset."""
    parent_fd = os.open(str(path), os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _LandlockPathBeneathAttr(allowed_access, parent_fd)
        rc = _syscall(
            _SYS_LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )
        if rc != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
    finally:
        os.close(parent_fd)


def _read_network_status(status_fd: int, report: EnforcementReport) -> EnforcementReport:
    """Drain the preexec status pipe and return an updated, honest report."""
    try:
        data = os.read(status_fd, 1)
    except OSError as exc:
        logger.warning("sandbox.network-status read failed: %s", exc)
        data = b""
    finally:
        with contextlib.suppress(OSError):
            os.close(status_fd)
    if data == b"1":
        return replace(
            report,
            network_restricted=True,
            details=[*report.details, "network namespace isolated via unshare(CLONE_NEWNET)"],
        )
    if data == b"0":
        return replace(
            report,
            network_restricted=False,
            details=[
                *report.details,
                "network NOT restricted: unshare(CLONE_NEWNET) unavailable "
                "(WSL2 quirk or missing CAP_SYS_ADMIN)",
            ],
        )
    return replace(
        report,
        network_restricted=False,
        details=[*report.details, "network status unverified (preexec failed before reporting)"],
    )


def _build_linux_plan(cmd: list[str], cwd: Path, policy: SandboxPolicy) -> ExecutionPlan:
    """Build a Landlock (+ best-effort network namespace) execution plan."""
    abi = probe_landlock_abi()
    if not abi:
        return ExecutionPlan(
            argv=cmd,
            preexec=None,
            env={},
            report=EnforcementReport(
                strategy=KernelEnforcement.LANDLOCK_SECCOMP,
                enforced=False,
                verified=False,
                details=[
                    "Landlock unavailable (kernel < 5.13 or sysfs not exposed); "
                    "seccomp BPF filter out of scope v1",
                ],
            ),
        )

    handled = _landlock_write_rights(abi)
    ruleset_fd = _create_ruleset(handled)
    allow_paths = [cwd, Path(tempfile.gettempdir())]
    allow_paths.extend(Path(p).expanduser().resolve() for p in policy.writable_paths)
    try:
        for path in allow_paths:
            _add_path_beneath_rule(ruleset_fd, path, handled)
    except OSError:
        with contextlib.suppress(OSError):
            os.close(ruleset_fd)
        raise

    want_network = not policy.enable_network
    details = [
        f"Landlock ABI {abi}: filesystem writes restricted to working dir + session temp",
        "seccomp BPF filter out of scope v1",
    ]
    if not want_network:
        details.append("network allowed by policy (enable_network=True)")
    base_report = EnforcementReport(
        strategy=KernelEnforcement.LANDLOCK_SECCOMP,
        enforced=True,
        verified=True,
        details=details,
    )

    status_fd: int | None = None
    write_fd: int | None = None
    if want_network:
        status_fd, write_fd = os.pipe()

    def preexec() -> None:
        net_ok = True
        if want_network and (_LIBC is None or _LIBC.unshare(_CLONE_NEWNET) != 0):
            net_ok = False
        if _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
        if write_fd is not None:
            os.write(write_fd, b"1" if net_ok else b"0")

    def post_start(_proc: Any) -> EnforcementReport | None:
        if write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        if status_fd is None:
            return None
        return _read_network_status(status_fd, base_report)

    def cleanup() -> None:
        with contextlib.suppress(OSError):
            os.close(ruleset_fd)
        if write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        if status_fd is not None:
            with contextlib.suppress(OSError):
                os.close(status_fd)

    return ExecutionPlan(
        argv=cmd,
        preexec=preexec,
        env={},
        report=base_report,
        post_start=post_start,
        cleanup=cleanup,
    )


def _sb_escape(value: str) -> str:
    """Escape a path for embedding in a Seatbelt profile string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def generate_seatbelt_profile(cwd: Path, temp_dir: Path, *, network_restricted: bool) -> str:
    """Generate a Seatbelt profile: read-only filesystem, writes in cwd/temp.

    The profile denies everything by default, allows file reads, allows
    writes only under *cwd* and *temp_dir*, and (when ``network_restricted``)
    denies all network access.
    """
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow file-read*)",
        f'(allow file-write* (subpath "{_sb_escape(str(cwd))}"))',
        f'(allow file-write* (subpath "{_sb_escape(str(temp_dir))}"))',
    ]
    if network_restricted:
        lines.append("(deny network*)")
    else:
        lines.append("(allow network*)")
    return "\n".join(lines) + "\n"


def _build_macos_plan(cmd: list[str], cwd: Path, policy: SandboxPolicy) -> ExecutionPlan:
    """Build a Seatbelt (``sandbox-exec -f``) execution plan."""
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return ExecutionPlan(
            argv=cmd,
            preexec=None,
            env={},
            report=EnforcementReport(
                strategy=KernelEnforcement.SEATBELT,
                enforced=False,
                verified=False,
                details=[
                    "sandbox-exec not found (deprecated since macOS 10.15); "
                    "kernel sandboxing unavailable",
                ],
            ),
        )

    network_restricted = not policy.enable_network
    profile = generate_seatbelt_profile(
        cwd, Path(tempfile.gettempdir()), network_restricted=network_restricted
    )
    fd, profile_path = tempfile.mkstemp(prefix="godspeed-seatbelt-", suffix=".sb")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(profile)
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(profile_path)
        raise

    def cleanup() -> None:
        with contextlib.suppress(OSError):
            os.unlink(profile_path)

    return ExecutionPlan(
        argv=[sandbox_exec, "-f", profile_path, *cmd],
        preexec=None,
        env={},
        cleanup=cleanup,
        report=EnforcementReport(
            strategy=KernelEnforcement.SEATBELT,
            enforced=True,
            verified=False,
            network_restricted=network_restricted,
            details=[
                "Seatbelt profile via sandbox-exec (deprecated since macOS 10.15; "
                "enforcement not verified)",
                "seccomp BPF filter out of scope v1",
            ],
        ),
    )


class _JobObject:
    """Windows Job Object wrapper — lifetime bounding only.

    Binds a process tree to a Job Object with ``KILL_ON_JOB_CLOSE`` so that
    closing the job handle terminates every process still in the job.  This
    bounds the *lifetime* of the process tree; it does NOT restrict
    filesystem or network access (reported honestly).
    """

    def __init__(self) -> None:
        if _kernel32 is None:
            raise OSError("kernel32 unavailable (not Windows)")
        self._handle = _kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            self._handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            _kernel32.CloseHandle(self._handle)
            self._handle = None
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(self, proc: Any) -> None:
        """Assign *proc* (created suspended) to the job and resume it."""
        if _kernel32 is None or _ntdll is None:
            raise OSError("kernel32/ntdll unavailable (not Windows)")
        hproc = _kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_SUSPEND_RESUME,
            False,
            proc.pid,
        )
        if not hproc:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            if not _kernel32.AssignProcessToJobObject(self._handle, hproc):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
            status = _ntdll.NtResumeProcess(hproc)
            if status != 0:
                raise OSError(f"NtResumeProcess failed with status 0x{status & 0xFFFFFFFF:08X}")
        finally:
            _kernel32.CloseHandle(hproc)

    def close(self) -> None:
        """Close the job handle, terminating any processes still in the job."""
        if self._handle and _kernel32 is not None:
            _kernel32.CloseHandle(self._handle)
            self._handle = None


def _build_windows_plan(cmd: list[str], _cwd: Path, _policy: SandboxPolicy) -> ExecutionPlan:
    """Build a Job Object execution plan (lifetime bounding only)."""
    job = _JobObject()

    def post_start(proc: Any) -> EnforcementReport | None:
        job.assign(proc)
        return None

    def cleanup() -> None:
        job.close()

    return ExecutionPlan(
        argv=cmd,
        preexec=None,
        env={},
        creationflags=_CREATE_SUSPENDED,
        post_start=post_start,
        cleanup=cleanup,
        report=EnforcementReport(
            strategy=KernelEnforcement.JOB_OBJECT,
            enforced=False,
            verified=False,
            network_restricted=False,
            details=[
                "process-tree lifetime bounded via Job Object (KILL_ON_JOB_CLOSE); "
                "filesystem/network access NOT restricted",
            ],
        ),
    )


def plan_execution(cmd: list[str], cwd: Path, policy: SandboxPolicy) -> ExecutionPlan:
    """Build an execution plan that kernel-sandboxes *cmd*.

    Raises:
        SandboxConstructionError: when the sandbox cannot be constructed
            (fail-closed — callers must not run the command unsandboxed).
    """
    try:
        if sys.platform == "win32":
            return _build_windows_plan(cmd, cwd, policy)
        if sys.platform == "darwin":
            return _build_macos_plan(cmd, cwd, policy)
        if sys.platform.startswith("linux"):
            return _build_linux_plan(cmd, cwd, policy)
    except OSError as exc:
        raise SandboxConstructionError(f"{exc}") from exc
    return ExecutionPlan(
        argv=cmd,
        preexec=None,
        env={},
        report=EnforcementReport(
            strategy=KernelEnforcement.NONE,
            enforced=False,
            verified=False,
            details=[f"no kernel sandbox strategy for platform {sys.platform!r}"],
        ),
    )


def enforcement_report_for_current_platform() -> EnforcementReport:
    """Report what the current platform's kernel sandbox can enforce.

    This is the honest-status helper: it never claims enforcement that the
    current platform cannot provide.
    """
    if sys.platform == "win32":
        return EnforcementReport(
            strategy=KernelEnforcement.JOB_OBJECT,
            enforced=False,
            verified=False,
            network_restricted=False,
            details=[
                "process-tree lifetime bounded via Job Object (KILL_ON_JOB_CLOSE); "
                "filesystem/network access NOT restricted",
            ],
        )
    if sys.platform == "darwin":
        if shutil.which("sandbox-exec"):
            return EnforcementReport(
                strategy=KernelEnforcement.SEATBELT,
                enforced=True,
                verified=False,
                network_restricted=True,
                details=[
                    "Seatbelt profile via sandbox-exec (deprecated since macOS 10.15; "
                    "enforcement not verified)",
                ],
            )
        return EnforcementReport(
            strategy=KernelEnforcement.SEATBELT,
            enforced=False,
            verified=False,
            details=["sandbox-exec not found (deprecated since macOS 10.15)"],
        )
    if sys.platform.startswith("linux"):
        abi = probe_landlock_abi()
        if abi:
            return EnforcementReport(
                strategy=KernelEnforcement.LANDLOCK_SECCOMP,
                enforced=True,
                verified=True,
                details=[
                    "Landlock ABI "
                    f"{abi}: filesystem writes restricted to working dir + session temp",
                    "seccomp BPF filter out of scope v1",
                ],
            )
        return EnforcementReport(
            strategy=KernelEnforcement.LANDLOCK_SECCOMP,
            enforced=False,
            verified=False,
            details=[
                "Landlock unavailable (kernel < 5.13 or sysfs not exposed); "
                "seccomp BPF filter out of scope v1",
            ],
        )
    return EnforcementReport(
        strategy=KernelEnforcement.NONE,
        enforced=False,
        verified=False,
        details=[f"no kernel sandbox strategy for platform {sys.platform!r}"],
    )


def format_enforcement_header(report: EnforcementReport) -> str:
    """One-line honest sandbox status for tool-result headers.

    Examples:
        ``sandbox: job-object (lifetime-only, fs/network NOT enforced)``
        ``sandbox: landlock (fs enforced, network enforced)``
        ``sandbox: seatbelt (fs enforced, network NOT enforced)``
    """
    if report.strategy == KernelEnforcement.JOB_OBJECT:
        return "sandbox: job-object (lifetime-only, fs/network NOT enforced)"
    if report.strategy == KernelEnforcement.NONE:
        return "sandbox: unenforced"
    fs = "fs enforced" if report.enforced else "fs NOT enforced"
    net = "network enforced" if report.network_restricted else "network NOT enforced"
    return f"sandbox: {report.strategy.value} ({fs}, {net})"
