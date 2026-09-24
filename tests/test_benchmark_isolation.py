"""Per-task isolation for benchmark runs (experiments/swebench_lite/isolation.py) and the shell script.

The integration tests run the real ``scripts/agent_shell_isolate.sh``; they are skipped where
unprivileged user+mount namespaces are unavailable (some CI runners, macOS, Windows).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_EXPERIMENTS = (Path(__file__).parent.parent / "experiments" / "swebench_lite").resolve()
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import isolation  # noqa: E402

SCRIPT = Path(__file__).parent.parent / "scripts" / "agent_shell_isolate.sh"


def _userns_works() -> bool:
    if os.name != "posix" or not shutil.which("unshare") or not shutil.which("setpriv"):
        return False
    if not Path("/dev/shm").is_dir():
        return False
    probe = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--mount", "true"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


needs_userns = pytest.mark.skipif(not _userns_works(), reason="user+mount namespaces unavailable")


class TestTaskIsolationEnv:
    def test_sets_and_restores_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GODSPEED_SHELL_WRAPPER", raising=False)
        monkeypatch.delenv("AGENT_PRIV", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin")
        with isolation.task_isolation(
            tag="run", instance_id="a__b-1", isolate_shell=True, base_dir=tmp_path
        ) as applied:
            assert os.environ["GODSPEED_SHELL_WRAPPER"] == str(isolation.ISOLATE_SCRIPT)
            priv = Path(os.environ["AGENT_PRIV"])
            assert priv.is_dir()
            assert applied["AGENT_PRIV"] == str(priv)
        assert "GODSPEED_SHELL_WRAPPER" not in os.environ
        assert "AGENT_PRIV" not in os.environ
        assert not priv.exists()

    def test_task_venv_goes_first_on_path_and_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "/usr/bin")

        def fake_make(dest: Path, python: str, extra_packages: tuple[str, ...] = ()) -> None:
            (dest / "bin").mkdir(parents=True)

        with (
            patch.object(isolation, "make_task_venv", side_effect=fake_make) as make,
            isolation.task_isolation(
                tag="run", instance_id="a__b-1", python="3.9", base_dir=tmp_path
            ),
        ):
            venv = Path(os.environ["AGENT_VENV"])
            assert os.environ["PATH"].split(os.pathsep)[0] == str(venv / "bin")
            assert make.call_args.args[1] == "3.9"
        assert os.environ["PATH"] == "/usr/bin"
        assert "AGENT_VENV" not in os.environ
        assert not venv.exists()

    def test_environment_is_restored_when_the_task_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENT_PRIV", raising=False)
        with (
            pytest.raises(RuntimeError, match="boom"),
            isolation.task_isolation(
                tag="run", instance_id="x", isolate_shell=True, base_dir=tmp_path
            ),
        ):
            raise RuntimeError("boom")
        assert "AGENT_PRIV" not in os.environ

    def test_nothing_requested_changes_nothing(self, tmp_path: Path) -> None:
        before = dict(os.environ)
        with isolation.task_isolation(tag="r", instance_id="x", base_dir=tmp_path) as applied:
            assert applied == {}
        assert dict(os.environ) == before


class TestPreflight:
    def test_missing_unshare_is_reported(self) -> None:
        with (
            patch.object(isolation.shutil, "which", return_value=None),
            pytest.raises(isolation.IsolationUnavailableError, match="unshare"),
        ):
            isolation.check_isolation_available()

    def test_failing_namespace_probe_is_reported(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", "Operation not permitted")
        with (
            patch.object(isolation.shutil, "which", return_value="/usr/bin/x"),
            patch.object(isolation.subprocess, "run", return_value=failed),
            pytest.raises(isolation.IsolationUnavailableError, match="Operation not permitted"),
        ):
            isolation.check_isolation_available()


def _run(script_cmd: str, cwd: Path, env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **env_extra}
    return subprocess.run(
        ["bash", str(SCRIPT), "-c", script_cmd],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@needs_userns
class TestIsolateScript:
    @pytest.fixture
    def world(self, tmp_path: Path) -> dict[str, Path]:
        home = tmp_path / "home"
        (home / "gold").mkdir(parents=True)
        (home / "gold" / "patch.diff").write_text("FAIL_TO_PASS secret_test_name\n")
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "a.py").write_text("print(1)\n")
        return {"home": home, "ws": ws, "priv": tmp_path / "priv"}

    def _env(self, w: dict[str, Path]) -> dict[str, str]:
        return {
            "AGENT_REAL_HOME": str(w["home"]),
            "AGENT_PRIV": str(w["priv"]),
            "AGENT_KEEP_RO": "",
        }

    def test_home_is_empty_and_gold_is_not_reachable(self, world: dict[str, Path]) -> None:
        out = _run(
            f"ls -A {world['home']}; cat {world['home']}/gold/patch.diff 2>&1; "
            f"grep -rl secret_test_name {world['home']} 2>/dev/null; echo done",
            world["ws"],
            self._env(world),
        )
        assert out.returncode == 0, out.stderr
        assert "secret_test_name" not in out.stdout
        assert "No such file" in out.stdout
        assert out.stdout.strip().endswith("done")

    def test_workspace_edits_reach_the_host(self, world: dict[str, Path]) -> None:
        out = _run("echo edited >> a.py; pwd", world["ws"], self._env(world))
        assert out.returncode == 0, out.stderr
        assert (world["ws"] / "a.py").read_text().endswith("edited\n")

    def test_private_tmp_persists_across_commands_of_one_task(self, world: dict[str, Path]) -> None:
        env = self._env(world)
        assert _run("echo kept > /tmp/scratch.txt", world["ws"], env).returncode == 0
        again = _run("cat /tmp/scratch.txt", world["ws"], env)
        assert again.stdout.strip() == "kept"
        assert (world["priv"] / "scratch.txt").exists()

    def test_capabilities_are_dropped_so_mounts_cannot_be_undone(
        self, world: dict[str, Path]
    ) -> None:
        out = _run(
            f"umount {world['home']} >/dev/null 2>&1; umount -l {world['home']} >/dev/null 2>&1; "
            f"mount -t tmpfs tmpfs /mnt >/dev/null 2>&1; "
            f"cat {world['home']}/gold/patch.diff 2>&1; grep CapEff /proc/self/status",
            world["ws"],
            self._env(world),
        )
        assert "secret_test_name" not in out.stdout  # the hidden data is still hidden
        assert out.stdout.strip().endswith("0" * 16)  # and no capability is left to undo the mount

    def test_fails_closed_when_namespaces_cannot_be_created(
        self, world: dict[str, Path], tmp_path: Path
    ) -> None:
        fake = tmp_path / "bin"
        fake.mkdir()
        unshare = fake / "unshare"
        unshare.write_text("#!/bin/sh\necho 'unshare: denied' >&2\nexit 1\n")
        unshare.chmod(0o755)
        env = {**self._env(world), "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"}
        out = _run("echo SHOULD-NOT-RUN", world["ws"], env)
        assert out.returncode != 0
        assert "SHOULD-NOT-RUN" not in out.stdout

    def test_usage_error_without_dash_c(self, world: dict[str, Path]) -> None:
        out = subprocess.run(
            ["bash", str(SCRIPT), "echo hi"], cwd=world["ws"], capture_output=True, text=True
        )
        assert out.returncode == 2


@needs_userns
def test_task_venv_is_visible_at_its_own_path_even_under_tmp(tmp_path: Path) -> None:
    """The venv usually lives under /tmp, which the sandbox replaces; it must be bound back."""
    home = tmp_path / "home"
    home.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    venv = tmp_path / "venvs" / "task1"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "marker").write_text("#!/bin/sh\necho venv-visible\n")
    (venv / "bin" / "marker").chmod(0o755)
    env = {
        "AGENT_REAL_HOME": str(home),
        "AGENT_PRIV": str(tmp_path / "priv"),
        "AGENT_VENV": str(venv),
        "AGENT_KEEP_RO": "",
    }
    out = _run(f"{venv}/bin/marker", ws, env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "venv-visible"
