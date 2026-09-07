"""Pre-flight checks for benchmark runs.

Verifies the execution environment before starting a long benchmark run:
- NVIDIA NIM API key connectivity
- Docker availability (required for SWE-bench verification)
- WSL availability (Windows hosts)
- Disk space sufficiency
- Python environment (swebench package, sb-cli)
- Network connectivity to NIM endpoint

Run with:
    python -m godspeed.benchmarks.preflight --all
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NIM_ENDPOINT = "https://api.nvidia.com/v1"
MIN_DISK_GB = 20
MIN_DOCKER_VERSION = (20, 0)
POOR_CONNECTIVITY_MS = 5000


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    fatal: bool = False


@dataclass
class PreFlightReport:
    results: list[CheckResult] = field(default_factory=list)
    all_passed: bool = True

    def add(self, name: str, passed: bool, detail: str = "", fatal: bool = False) -> None:
        self.results.append(CheckResult(name=name, passed=passed, detail=detail, fatal=fatal))
        if not passed:
            self.all_passed = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the report."""
        return {
            "all_passed": self.all_passed,
            "checks": [
                {"name": r.name, "passed": r.passed, "detail": r.detail, "fatal": r.fatal}
                for r in self.results
            ],
        }


def _default_fetch(url: str, *, headers: dict[str, str] | None = None, timeout: int = 10) -> Any:
    """Production fetcher — hits the real network."""
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {})  # noqa: S310
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


FetchFn = Callable[[str, dict[str, str] | None, int], Any]


def check_nim_connectivity(
    report: PreFlightReport,
    *,
    keys_env: str = "NVIDIA_NIM_API_KEYS",
    fetcher: FetchFn | None = None,
) -> None:
    """Verify NIM API keys can authenticate against the NVIDIA endpoint.

    Parameters
    ----------
    report:
        Accumulator for check results.
    keys_env:
        Primary environment variable name to read keys from.
    fetcher:
        Injectable HTTP fetcher ``(url, headers, timeout) -> response``.
        ``None`` (default) uses the real network.  Pass a stub for offline
        testing — the stub may raise ``OSError`` to simulate offline mode.
    """
    _fetch = fetcher if fetcher is not None else _default_fetch

    raw = os.environ.get(keys_env, os.environ.get("NVIDIA_NIM_API_KEY", ""))
    if not raw:
        report.add(
            "NIM keys", False, f"Neither {keys_env} nor NVIDIA_NIM_API_KEY is set", fatal=True
        )
        return

    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        report.add("NIM keys", False, "No non-empty keys found", fatal=True)
        return

    report.add("NIM key count", True, f"{len(keys)} key(s) configured")

    healthy = 0
    for i, key in enumerate(keys, 1):
        try:
            _fetch(
                f"{NIM_ENDPOINT}/models",
                {"Authorization": f"Bearer {key}"},
                10,
            )
            healthy += 1
        except Exception as e:  # noqa: BLE001
            error = str(e)[:120]
            report.add(f"NIM key #{i}", False, f"Key ending ...{key[-8:]} failed: {error}")

    if healthy == 0:
        report.add("NIM connectivity", False, "All keys failed to authenticate", fatal=True)
    elif healthy < len(keys):
        report.add("NIM connectivity", True, f"{healthy}/{len(keys)} keys healthy (some degraded)")
    else:
        report.add("NIM connectivity", True, f"All {len(keys)} keys authenticated")


def _evict_partial_module(pkg: str) -> None:
    """Remove a partially initialized package from sys.modules.

    A crashed first import leaves the half-built module cached, so every
    later ``import`` in the process returns the broken object. Evicting
    the package and its submodules lets the next import retry cleanly.
    """
    for mod_name in [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]:
        sys.modules.pop(mod_name, None)


def _prebind_litellm_deps() -> None:
    """Import aiohttp submodules litellm touches at its import time.

    litellm's ``aiohttp_transport`` reads ``aiohttp.client_exceptions`` at
    module level. aiohttp uses lazy attribute loading, so if that
    submodule is not pre-imported, litellm's first import crashes with a
    confusing AttributeError that then leaves litellm half-initialized in
    sys.modules. Importing it here is cheap and makes the litellm import
    deterministic regardless of environment.
    """
    try:
        import aiohttp.client_exceptions  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        logger.debug("aiohttp.client_exceptions pre-import failed: %s", exc)


def check_python_env(report: PreFlightReport) -> None:
    """Verify required Python packages are importable."""
    packages = {
        "godspeed": "godspeed package",
        "litellm": "LiteLLM",
        "datasets": "HuggingFace datasets (SWE-bench)",
    }
    for pkg, label in packages.items():
        if pkg == "litellm":
            _prebind_litellm_deps()
        try:
            __import__(pkg)
            report.add(f"Python: {label}", True, "installed")
        except Exception as e:  # noqa: BLE001
            # Import can fail with more than ImportError (e.g. litellm's
            # circular-import AttributeError). Treat any import failure as a
            # failed check rather than crashing the whole pre-flight.
            #
            # A failed first import can leave a third-party package PARTIALLY
            # initialized in sys.modules, poisoning every later import in the
            # process (litellm does this). Evict third-party packages so a
            # later clean import attempt gets a fresh start — but never evict
            # the godspeed package itself: mid-session eviction splits the
            # module universe (live references keep the old objects while
            # fresh imports build new ones), which breaks monkeypatching and
            # isinstance checks for the rest of the process.
            if pkg != "godspeed":
                _evict_partial_module(pkg)
            report.add(
                f"Python: {label}",
                False,
                f"{pkg} import failed: {type(e).__name__}: {str(e)[:120]}",
                fatal=pkg == "godspeed",
            )

    # swebench CLI (optional but recommended)
    if shutil.which("swebench") or shutil.which("sb-cli"):
        report.add("Python: sb-cli", True, "available")
    else:
        report.add(
            "Python: sb-cli",
            True,
            "not found — needed to submit results. Install: pip install sb-cli",
        )


def check_docker(report: PreFlightReport) -> None:
    """Verify Docker daemon is running and functional."""
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        report.add("Docker", False, "docker command not found on PATH", fatal=False)
        report.add(
            "Docker note",
            True,
            "SWE-bench verification requires Docker; run without --agent-in-loop to skip",
        )
        return

    try:
        result = subprocess.run(
            [docker_bin, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            report.add("Docker", True, f"v{version} running")
        else:
            stderr_tail = result.stderr.strip()[-200:] if result.stderr else "unknown error"
            report.add("Docker", False, f"daemon not accessible: {stderr_tail}")
    except subprocess.TimeoutExpired:
        report.add("Docker", False, "command timed out — daemon unresponsive")
    except FileNotFoundError:
        report.add("Docker", False, "docker executable vanished mid-check")


def check_wsl(report: PreFlightReport) -> None:
    """Verify WSL availability on Windows hosts."""
    if sys.platform != "win32":
        return  # Not Windows — nothing to check

    wsl_bin = shutil.which("wsl.exe") or shutil.which("wsl")
    if wsl_bin is None:
        report.add(
            "WSL", False, "WSL not found; SWE-bench Docker verification unavailable on Windows"
        )
        return

    try:
        result = subprocess.run(
            [str(wsl_bin), "-d", "Ubuntu", "-e", "bash", "-c", "echo ok"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            report.add("WSL", True, "Ubuntu available")
        else:
            report.add("WSL", False, f"Ubuntu WSL not responding: {result.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        report.add("WSL", False, "WSL command timed out")
    except FileNotFoundError:
        report.add("WSL", False, "WSL binary not found")


def check_disk_space(report: PreFlightReport, min_gb: int = MIN_DISK_GB) -> None:
    """Verify sufficient free disk space."""
    try:
        usage = shutil.disk_usage(Path.cwd())
        free_gb = usage.free / (1024**3)
        if free_gb >= min_gb:
            report.add("Disk space", True, f"{free_gb:.1f} GB free (>= {min_gb} GB required)")
        else:
            report.add(
                "Disk space",
                False,
                f"Only {free_gb:.1f} GB free ({min_gb} GB required). "
                f"SWE-bench clones repositories — free up space before running.",
                fatal=True,
            )
    except OSError as e:
        report.add("Disk space", False, f"Unable to check: {e}")


def check_nim_rpm(report: PreFlightReport, keys_env: str = "NVIDIA_NIM_API_KEYS") -> None:
    """Estimate effective RPM capacity with configured keys."""
    raw = os.environ.get(keys_env, os.environ.get("NVIDIA_NIM_API_KEY", ""))
    if not raw:
        return
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    rpm = len(keys) * 30
    report.add("NIM RPM capacity", True, f"{len(keys)} keys x 30 RPM = {rpm} RPM effective")


def run_all_checks(
    *,
    fetcher: FetchFn | None = None,
    skip_network: bool = False,
) -> PreFlightReport:
    """Run all pre-flight checks. Returns a report with pass/fail per check.

    Parameters
    ----------
    fetcher:
        Injectable HTTP fetcher for NIM connectivity (see
        ``check_nim_connectivity``).  ``None`` uses the real network.
    skip_network:
        When ``True``, skip the network-dependent NIM connectivity check and
        record it as a non-fatal informational result.  Used for offline
        dry-runs where no endpoint is reachable.
    """
    report = PreFlightReport()
    t0 = time.monotonic()

    if skip_network:
        report.add(
            "NIM connectivity",
            True,
            "skipped (offline dry-run) — real run will verify against NIM endpoint",
        )
    else:
        check_nim_connectivity(report, fetcher=fetcher)
    check_nim_rpm(report)
    check_python_env(report)
    check_docker(report)
    check_wsl(report)
    check_disk_space(report)

    ms = (time.monotonic() - t0) * 1000
    report.add("Pre-flight total time", True, f"{ms:.0f}ms")

    return report


def print_report(report: PreFlightReport) -> int:
    """Pretty-print the pre-flight report. Returns 0 if all pass, 1 otherwise."""
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        if r.fatal:
            status = "FATAL"
        detail_str = f" — {r.detail}" if r.detail else ""
        print(f"  [{status:>5}] {r.name}{detail_str}")  # noqa: T201

    fatal_count = sum(1 for r in report.results if r.fatal and not r.passed)
    fail_count = sum(1 for r in report.results if not r.fatal and not r.passed)

    if report.all_passed:
        return 0

    summary_parts = []
    if fatal_count:
        summary_parts.append(f"{fatal_count} FATAL")
    if fail_count:
        summary_parts.append(f"{fail_count} failed")
    return 1


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Godspeed benchmark pre-flight checks")
    parser.add_argument("--all", action="store_true", default=True, help="Run all checks")
    parser.add_argument("--check-nim", action="store_true", help="Check NIM connectivity only")
    parser.add_argument("--check-docker", action="store_true", help="Check Docker only")
    parser.add_argument("--check-disk", action="store_true", help="Check disk space only")
    parser.add_argument("--dry-run", action="store_true", help="Skip network checks (offline)")
    parser.add_argument("--quiet", action="store_true", help="Exit code only, no output")
    args = parser.parse_args()

    report = run_all_checks(skip_network=args.dry_run)

    if args.quiet:
        return 0 if report.all_passed else 1

    return print_report(report)


if __name__ == "__main__":
    sys.exit(main())
