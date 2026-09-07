"""Runtime verification — build, launch, and observe a project.

This module provides a runtime gate that chains the classic static
verification (lint/type-check) with a dynamic check: does the project
actually build, launch, and stay alive long enough to be considered
healthy? The running app is observed, not just compiled.

The module is intentionally pure and unit-testable: every step
(build, launch, probe, kill) is a small function with a bounded
timeout, and the result is a structured :class:`Verdict` dataclass.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from godspeed.tools.test_runner import detect_framework

logger = logging.getLogger(__name__)

# --- Module constants (caps for every subprocess) ---------------------------

#: Max seconds a build step may run before it is killed and reported as FAIL.
BUILD_TIMEOUT_SECONDS = 120

#: Max seconds a launch step may take to spawn the process.
LAUNCH_TIMEOUT_SECONDS = 30

#: How long the launched process must survive before the probe is considered
#: "alive". A process that exits within this window is a launch failure.
PROBE_WINDOW_SECONDS = 5

#: Max seconds to wait for an optional HTTP GET probe to respond.
HTTP_PROBE_TIMEOUT_SECONDS = 5

#: Cap on the number of evidence lines collected per verdict.
MAX_EVIDENCE_LINES = 20

#: Cap on the number of characters captured from a single subprocess output.
MAX_OUTPUT_CHARS = 2000

#: Windows flag to avoid spawning a console window for console apps.
_CREATE_NO_WINDOW = 0x08000000 if platform.system() == "Windows" else 0


@dataclass
class Verdict:
    """Structured result of a runtime verification run.

    Attributes:
        build_ok: True if the build step completed with exit code 0.
        launch_ok: True if the process spawned and survived the probe window.
        alive_after_probe: True if the process was still alive after the
            probe window elapsed (i.e. it did not crash immediately).
        evidence: Human-readable lines describing what happened at each step.
    """

    build_ok: bool
    launch_ok: bool
    alive_after_probe: bool
    evidence: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Overall pass/fail: build, launch, and survival must all hold."""
        return self.build_ok and self.launch_ok and self.alive_after_probe

    def add_evidence(self, line: str) -> None:
        """Append an evidence line, respecting the cap."""
        if len(self.evidence) < MAX_EVIDENCE_LINES:
            self.evidence.append(line)


def _windows_flags() -> int:
    """Return subprocess creation flags that avoid a console window on Windows."""
    return _CREATE_NO_WINDOW


def _run_bounded(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
    capture: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    """Run a command with a bounded timeout, Windows-safe.

    Returns the CompletedProcess on success, or None if the command could
    not be started (binary missing, OSError). Timeouts are surfaced by
    raising ``subprocess.TimeoutExpired`` — callers decide how to report.
    """
    creationflags = _windows_flags()
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "timeout": timeout,
        "creationflags": creationflags,
    }
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)  # type: ignore[arg-type]


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate output to a bounded length for evidence lines."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({len(text) - limit} chars truncated)"


def detect_build_command(project_root: Path) -> list[str] | None:
    """Detect the build command for a project, or None if none is needed.

    Reuses the project-type detection from ``test_runner.detect_framework``
    and extends it with build-specific markers. Returns a command list
    suitable for ``subprocess.run`` (argv form, no shell).

    Detection order:
      1. Python (pytest/pyproject/setup.py) -> ``python -m compileall``
         (a cheap, dependency-free syntax/build check).
      2. Node (package.json with a ``build`` script) -> ``npm run build``.
      3. Go (go.mod) -> ``go build ./...``.
      4. Rust (Cargo.toml) -> ``cargo check``.
      5. Unknown -> None (no build step).
    """
    framework = detect_framework(project_root)

    if framework == "pytest":
        # Python: compileall is a fast, dependency-free syntax check.
        python_bin = shutil.which("python") or shutil.which("python3")
        if python_bin is None:
            return None
        return [python_bin, "-m", "compileall", "-q", str(project_root)]

    if framework in ("jest", "vitest"):
        # Node: prefer the package.json "build" script if present.
        package_json = project_root / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})
                if "build" in scripts:
                    npm_bin = shutil.which("npm")
                    if npm_bin is None:
                        return None
                    return [npm_bin, "run", "build"]
            except (OSError, json.JSONDecodeError):
                logger.debug("Could not read package.json for build detection")
        # No build script — no build step.
        return None

    if framework == "go":
        go_bin = shutil.which("go")
        if go_bin is None:
            return None
        return [go_bin, "build", "./..."]

    if framework == "cargo":
        cargo_bin = shutil.which("cargo")
        if cargo_bin is None:
            return None
        return [cargo_bin, "check"]

    return None


def detect_launch_command(project_root: Path) -> list[str] | None:
    """Detect the launch command for a project, or None if none is known.

    Returns a command list (argv form) that starts the project's main
    entrypoint. Detection order:

      1. Python: ``main.py`` / ``app.py`` / ``__main__.py`` in the root.
      2. Node: package.json ``start`` script.
      3. Go: ``go run .`` (module root).
      4. Rust: ``cargo run``.
      5. Unknown -> None.
    """
    framework = detect_framework(project_root)

    if framework == "pytest":
        # Python entrypoints.
        for entry in ("main.py", "app.py", "__main__.py"):
            if (project_root / entry).exists():
                python_bin = shutil.which("python") or shutil.which("python3")
                if python_bin is None:
                    return None
                return [python_bin, str(project_root / entry)]
        return None

    if framework in ("jest", "vitest"):
        package_json = project_root / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})
                if "start" in scripts:
                    npm_bin = shutil.which("npm")
                    if npm_bin is None:
                        return None
                    return [npm_bin, "run", "start"]
            except (OSError, json.JSONDecodeError):
                logger.debug("Could not read package.json for launch detection")
        return None

    if framework == "go":
        go_bin = shutil.which("go")
        if go_bin is None:
            return None
        return [go_bin, "run", "."]

    if framework == "cargo":
        cargo_bin = shutil.which("cargo")
        if cargo_bin is None:
            return None
        return [cargo_bin, "run"]

    return None


class RuntimeVerifier:
    """Build, launch, and observe a project to confirm it actually runs.

    The verifier is deliberately synchronous and pure: each step is a
    small method with a bounded timeout, and the whole flow is driven by
    :meth:`verify`, which returns a :class:`Verdict`. No mocking is needed
    in tests — real tiny fixtures (sleep-based Python scripts) are used.

    Args:
        project_root: Absolute path to the project to verify.
        build_command: Optional override; if None, auto-detected.
        launch_command: Optional override; if None, auto-detected.
        probe_window: Seconds the process must survive before being
            considered alive (defaults to :data:`PROBE_WINDOW_SECONDS`).
        http_probe: Optional URL to GET after launch (e.g. ``http://127.0.0.1:8000``).
            If provided, the probe also requires a successful HTTP response.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        build_command: list[str] | None = None,
        launch_command: list[str] | None = None,
        probe_window: int = PROBE_WINDOW_SECONDS,
        http_probe: str | None = None,
    ) -> None:
        self.project_root = project_root
        self.build_command = build_command
        self.launch_command = launch_command
        self.probe_window = probe_window
        self.http_probe = http_probe
        self._proc: subprocess.Popen[str] | None = None

    def _resolved_launch_command(self) -> list[str] | None:
        """Return the effective launch command (override or auto-detected)."""
        if self.launch_command is not None:
            return self.launch_command
        return detect_launch_command(self.project_root)

    def _resolved_http_probe(self) -> str | None:
        """Return the effective HTTP probe URL.

        Uses the explicit ``http_probe`` if set; otherwise scans the launch
        command args for a URL (e.g. ``uvicorn app:app --port 8000`` or
        ``python main.py http://127.0.0.1:8000``) and probes that.
        """
        if self.http_probe is not None:
            return self.http_probe
        cmd = self._resolved_launch_command()
        if cmd is None:
            return None
        for arg in cmd:
            if arg.startswith(("http://", "https://")):
                return arg
        return None

    # -- Public API ----------------------------------------------------------

    def verify(self) -> Verdict:
        """Run the full build -> launch -> probe -> kill pipeline.

        Returns a :class:`Verdict` with evidence lines describing each step.
        The launched process is always killed on completion (success or
        failure) so no orphan processes leak.
        """
        verdict = Verdict(build_ok=False, launch_ok=False, alive_after_probe=False)

        # 1. Build
        build_ok, build_evidence = self._build()
        verdict.build_ok = build_ok
        for line in build_evidence:
            verdict.add_evidence(line)

        if not build_ok:
            verdict.add_evidence("Build FAILED — skipping launch.")
            return verdict

        # 2. Launch
        launch_ok, launch_evidence = self._launch()
        verdict.launch_ok = launch_ok
        for line in launch_evidence:
            verdict.add_evidence(line)

        if not launch_ok:
            verdict.add_evidence("Launch FAILED — no process to probe.")
            return verdict

        # 3. Probe (survival window + optional HTTP GET)
        alive, probe_evidence = self._probe()
        verdict.alive_after_probe = alive
        for line in probe_evidence:
            verdict.add_evidence(line)

        # 4. Kill on completion (always)
        self._kill()

        return verdict

    # -- Steps ---------------------------------------------------------------

    def _build(self) -> tuple[bool, list[str]]:
        """Run the build step. Returns (ok, evidence_lines)."""
        cmd = self.build_command
        if cmd is None:
            cmd = detect_build_command(self.project_root)

        if cmd is None:
            return True, ["No build step detected — skipping build."]

        evidence: list[str] = []
        evidence.append(f"Build: {' '.join(cmd)}")

        try:
            result = _run_bounded(cmd, cwd=self.project_root, timeout=BUILD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            evidence.append(f"Build TIMED OUT after {BUILD_TIMEOUT_SECONDS}s — killed.")
            return False, evidence
        except OSError as exc:
            evidence.append(f"Build could not start: {exc}")
            return False, evidence

        if result is None:
            evidence.append("Build could not start (binary missing).")
            return False, evidence

        if result.returncode == 0:
            evidence.append("Build PASSED (exit 0).")
            return True, evidence

        output = _truncate(result.stdout.strip() or result.stderr.strip())
        evidence.append(f"Build FAILED (exit {result.returncode}):\n{output}")
        return False, evidence

    def _launch(self) -> tuple[bool, list[str]]:
        """Launch the process. Returns (ok, evidence_lines)."""
        cmd = self._resolved_launch_command()

        if cmd is None:
            return False, ["No launch command detected — cannot launch."]

        evidence: list[str] = []
        evidence.append(f"Launch: {' '.join(cmd)}")

        creationflags = _windows_flags()
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
            )
        except OSError as exc:
            evidence.append(f"Launch could not start: {exc}")
            return False, evidence

        # Wait briefly for the process to either exit (crash) or stay alive.
        # Use a bounded wait so we never block indefinitely.
        deadline = time.monotonic() + LAUNCH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # Process exited immediately — launch failure.
                returncode = self._proc.returncode
                stderr = self._read_stderr()
                evidence.append(f"Launch FAILED — process exited immediately (exit {returncode}).")
                if stderr:
                    evidence.append(f"stderr:\n{_truncate(stderr)}")
                return False, evidence
            time.sleep(0.1)

        evidence.append(f"Launch OK — process alive (pid {self._proc.pid}).")
        return True, evidence

    def _probe(self) -> tuple[bool, list[str]]:
        """Probe the running process. Returns (alive, evidence_lines)."""
        if self._proc is None:
            return False, ["No process to probe."]

        evidence: list[str] = []

        # Survival window: the process must still be alive after probe_window.
        deadline = time.monotonic() + self.probe_window
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                returncode = self._proc.returncode
                stderr = self._read_stderr()
                evidence.append(
                    f"Probe FAILED — process died during probe window (exit {returncode})."
                )
                if stderr:
                    evidence.append(f"stderr:\n{_truncate(stderr)}")
                return False, evidence
            time.sleep(0.1)

        evidence.append(f"Probe OK — process alive after {self.probe_window}s window.")

        # Optional HTTP GET probe (explicit URL or extracted from launch args).
        http_url = self._resolved_http_probe()
        if http_url:
            http_ok, http_evidence = self._http_get(http_url)
            evidence.extend(http_evidence)
            if not http_ok:
                return False, evidence

        return True, evidence

    def _http_get(self, url: str) -> tuple[bool, list[str]]:
        """Perform an optional HTTP GET probe. Returns (ok, evidence_lines)."""
        import urllib.request
        from urllib.parse import urlparse

        evidence: list[str] = []
        evidence.append(f"HTTP probe: GET {url}")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            evidence.append(f"HTTP probe FAILED: unsupported scheme {parsed.scheme!r}.")
            return False, evidence

        try:
            # Scheme validated above (http/https only) — safe to open.
            with urllib.request.urlopen(  # noqa: S310
                url, timeout=HTTP_PROBE_TIMEOUT_SECONDS
            ) as resp:
                status = resp.status
                evidence.append(f"HTTP probe OK — status {status}.")
                return True, evidence
        except Exception as exc:
            evidence.append(f"HTTP probe FAILED: {exc}")
            return False, evidence

    def _kill(self) -> None:
        """Kill the launched process if it is still running."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                logger.warning("Failed to cleanly kill process pid=%s", self._proc.pid)
        self._proc = None

    def _read_stderr(self) -> str:
        """Best-effort read of the process stderr (non-blocking)."""
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read()
        except (OSError, ValueError):
            return ""
