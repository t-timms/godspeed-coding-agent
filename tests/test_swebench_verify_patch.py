"""Tests for experiments/swebench_lite/verify_patch.py against the swebench 5.x harness.

Three things were broken on a fresh Linux box with swebench 5.x installed:
  * the interpreter path was hard-coded to ``/home/swebench_venv/bin/python3``;
  * ``--cache_level`` was always passed, but swebench 5.x removed it (argparse error);
  * the one-row dataset came only from ``benchmarks/swebench_lite_test.jsonl``, which is not in
    the repo, so an empty dataset was handed to the harness and every verification "failed".
No Docker/WSL/network here: the harness subprocess and ``datasets`` are faked.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

_EXPERIMENTS_DIR = (Path(__file__).parent.parent / "experiments" / "swebench_lite").resolve()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import verify_patch as vp  # noqa: E402

ROW = {"instance_id": "acme__widget-1", "repo": "acme/widget", "base_commit": "abc123"}


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    vp._supports_cache_level.cache_clear()
    monkeypatch.delenv(vp.PYTHON_ENV, raising=False)


def _completed(stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args="x", returncode=rc, stdout=stdout, stderr=stderr)


def _fake_datasets(rows_by_split: dict[str, list[dict]]) -> types.SimpleNamespace:
    def load_dataset(name: str, split: str) -> list[dict]:
        return rows_by_split[split]

    return types.SimpleNamespace(load_dataset=load_dataset)


class TestSwebenchPython:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(vp.PYTHON_ENV, "/opt/py/bin/python")
        assert vp._swebench_python(use_wsl=True) == "/opt/py/bin/python"
        assert vp._swebench_python(use_wsl=False) == "/opt/py/bin/python"

    def test_wsl_uses_legacy_path(self) -> None:
        assert vp._swebench_python(use_wsl=True) == vp.LEGACY_WSL_PYTHON

    def test_native_falls_back_to_current_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(vp, "LEGACY_WSL_PYTHON", str(tmp_path / "missing" / "python3"))
        assert vp._swebench_python(use_wsl=False) == sys.executable

    def test_native_prefers_existing_legacy_venv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        legacy = tmp_path / "python3"
        legacy.write_text("")
        monkeypatch.setattr(vp, "LEGACY_WSL_PYTHON", str(legacy))
        assert vp._swebench_python(use_wsl=False) == str(legacy)


class TestHarnessCmd:
    def test_cache_flag_included_only_when_supported(self) -> None:
        on = vp._harness_cmd("/w", "/p.jsonl", "i-1", "r1", "/d.jsonl", "py", True)
        off = vp._harness_cmd("/w", "/p.jsonl", "i-1", "r1", "/d.jsonl", "py", False)
        assert "--cache_level instance" in on
        assert "--cache_level" not in off
        assert off.startswith("cd '/w' && py -m swebench.harness.run_evaluation ")

    def test_hf_fallback_uses_requested_split(self) -> None:
        cmd = vp._harness_cmd("/w", "/p", "i-1", "r1", None, "py", False, split="test")
        assert f"--dataset_name {vp.LITE_DATASET}" in cmd
        assert "--split test" in cmd


class TestSupportsCacheLevel:
    def test_detects_flag_in_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            vp, "_native_run", lambda cmd, timeout=900: _completed("--cache_level X")
        )
        assert vp._supports_cache_level("py", False) is True

    def test_swebench5_help_has_no_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            vp, "_native_run", lambda cmd, timeout=900: _completed("--max_workers N")
        )
        assert vp._supports_cache_level("py", False) is False

    def test_timeout_means_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(cmd: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd, timeout)

        monkeypatch.setattr(vp, "_native_run", boom)
        assert vp._supports_cache_level("py", False) is False


class TestFindInstanceRow:
    def test_local_jsonl_hit(self, tmp_path: Path) -> None:
        (tmp_path / "benchmarks").mkdir()
        (tmp_path / "benchmarks" / "swebench_lite_test.jsonl").write_text(
            json.dumps(ROW) + "\n", encoding="utf-8"
        )
        assert vp._find_instance_row("acme__widget-1", "dev", tmp_path) == ROW

    def test_falls_back_to_hf_requested_split_then_other(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake = _fake_datasets({"dev": [], "test": [ROW]})
        monkeypatch.setitem(sys.modules, "datasets", fake)
        assert vp._find_instance_row("acme__widget-1", "dev", tmp_path) == ROW

    def test_missing_everywhere_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets({"dev": [], "test": []}))
        assert vp._find_instance_row("nope__nope-9", "dev", tmp_path) is None


class TestVerifyPatchEndToEnd:
    def _wire(self, monkeypatch: pytest.MonkeyPatch, resolved: bool, calls: list[str]) -> None:
        monkeypatch.setattr(vp, "_use_wsl", lambda: False)
        monkeypatch.setattr(vp, "_supports_cache_level", lambda python_path, use_wsl: False)

        def fake_native(cmd: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            run_id = re.search(r"--run_id (\S+)", cmd).group(1)  # type: ignore[union-attr]
            workdir = Path(re.match(r"cd '([^']+)'", cmd).group(1))  # type: ignore[union-attr]
            model_norm = "openai__test"
            report = {"resolved_ids": [ROW["instance_id"]] if resolved else []}
            (workdir / f"{model_norm}.{run_id}.json").write_text(json.dumps(report))
            return _completed()

        monkeypatch.setattr(vp, "_native_run", fake_native)

    def test_resolves_without_removed_flag_and_without_local_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        self._wire(monkeypatch, True, calls)
        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets({"dev": [ROW], "test": []}))
        workdir = tmp_path / "experiments" / "swebench_lite"
        resolved, _ = vp.verify_patch(ROW["instance_id"], "openai/test", "diff --git a b", workdir)
        assert resolved is True
        assert len(calls) == 1
        assert "--cache_level" not in calls[0]
        dataset = next(workdir.glob(".dataset_*.jsonl"))
        assert json.loads(dataset.read_text())["instance_id"] == ROW["instance_id"]

    def test_stale_empty_dataset_file_is_regenerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        self._wire(monkeypatch, True, calls)
        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets({"dev": [ROW], "test": []}))
        workdir = tmp_path / "experiments" / "swebench_lite"
        workdir.mkdir(parents=True)
        patch_text = "diff --git a b"
        import hashlib

        digest = hashlib.sha1(
            (ROW["instance_id"] + "::" + patch_text).encode(), usedforsecurity=False
        ).hexdigest()[:12]
        (workdir / f".dataset_{digest}.jsonl").write_text("")  # what the old code left behind
        resolved, _ = vp.verify_patch(ROW["instance_id"], "openai/test", patch_text, workdir)
        assert resolved is True
        assert (workdir / f".dataset_{digest}.jsonl").stat().st_size > 0

    def test_unknown_instance_fails_fast_without_running_harness(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        self._wire(monkeypatch, True, calls)
        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets({"dev": [], "test": []}))
        resolved, output = vp.verify_patch(
            "nope__nope-9", "openai/test", "diff --git a b", tmp_path / "e" / "s"
        )
        assert resolved is False
        assert "not found" in output
        assert calls == []
