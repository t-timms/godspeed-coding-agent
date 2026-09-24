"""Per-task isolation for benchmark runs: a throwaway venv and a confined agent shell.

Two problems this exists for (both observed, see docs/benchmark_hygiene.md):

* The agent runs ``pip install -e .`` and ``pip install "setuptools<81"``. With Godspeed's own venv first
  on PATH those installs landed IN Godspeed's venv and leaked from one task into the next.
* The agent's shell shares the host filesystem, and a model searched it for hidden tests and gold data.

``task_isolation`` gives each task its own Python venv (first on PATH for the agent's commands) and,
optionally, routes every agent command through ``scripts/agent_shell_isolate.sh``. The runner process
itself is not confined and keeps using its own interpreter.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATE_SCRIPT = _REPO_ROOT / "scripts" / "agent_shell_isolate.sh"
_ENV_KEYS = ("PATH", "GODSPEED_SHELL_WRAPPER", "AGENT_PRIV", "AGENT_VENV", "AGENT_REAL_HOME")


class IsolationUnavailableError(RuntimeError):
    """Isolation was requested but this host cannot provide it."""


def check_isolation_available() -> None:
    """Fail early (not per command) when the confined shell cannot work here."""
    if os.name != "posix" or not shutil.which("unshare") or not shutil.which("setpriv"):
        raise IsolationUnavailableError("needs Linux with util-linux 'unshare' and 'setpriv'")
    if not ISOLATE_SCRIPT.is_file():
        raise IsolationUnavailableError(f"missing {ISOLATE_SCRIPT}")
    probe = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--mount", "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise IsolationUnavailableError(
            "unprivileged user+mount namespaces are unavailable "
            f"({probe.stderr.strip() or 'unshare failed'}); on Ubuntu 24.04 check "
            "kernel.apparmor_restrict_unprivileged_userns"
        )


def make_task_venv(dest: Path, python: str, extra_packages: tuple[str, ...] = ("pytest",)) -> None:
    """Create a seeded venv at ``dest`` with a specific interpreter (needs ``uv``)."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for --task-python (https://docs.astral.sh/uv/)")
    subprocess.run(
        [uv, "venv", "--seed", "--python", python, "-q", str(dest)], check=True, capture_output=True
    )
    if extra_packages:
        subprocess.run(
            [str(dest / "bin" / "pip"), "install", "-q", *extra_packages],
            check=True,
            capture_output=True,
        )


@contextlib.contextmanager
def task_isolation(
    *,
    tag: str,
    instance_id: str,
    python: str | None = None,
    isolate_shell: bool = False,
    base_dir: Path | None = None,
) -> Iterator[dict[str, str]]:
    """Apply per-task isolation to ``os.environ`` for the duration of one task.

    ``python`` (e.g. ``"3.9"``) creates a throwaway venv whose ``bin`` comes first on PATH.
    ``isolate_shell`` sets ``GODSPEED_SHELL_WRAPPER`` to the confinement script and gives the task a
    private scratch directory that persists across its commands. Everything is removed afterwards.
    Yields the variables that were set.
    """
    base = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
    safe = f"{tag}__{instance_id}".replace("/", "_")
    venv = base / "agent_venvs" / safe
    priv = base / "agent_priv" / safe
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    applied: dict[str, str] = {}
    try:
        if python:
            shutil.rmtree(venv, ignore_errors=True)
            venv.parent.mkdir(parents=True, exist_ok=True)
            make_task_venv(venv, python)
            applied["PATH"] = f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            applied["AGENT_VENV"] = str(venv)
        if isolate_shell:
            shutil.rmtree(priv, ignore_errors=True)
            priv.mkdir(parents=True, exist_ok=True)
            applied["GODSPEED_SHELL_WRAPPER"] = str(ISOLATE_SCRIPT)
            applied["AGENT_PRIV"] = str(priv)
            applied["AGENT_REAL_HOME"] = str(Path.home())
        os.environ.update(applied)
        yield applied
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        shutil.rmtree(venv, ignore_errors=True)
        shutil.rmtree(priv, ignore_errors=True)
