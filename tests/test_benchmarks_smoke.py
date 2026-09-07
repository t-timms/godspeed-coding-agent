"""Offline smoke tests for the SWE-bench harness pre-flight and dry-run paths.

These tests exercise the benchmark harness's pre-flight checks, config
validation, instance selection, and dry-run planning WITHOUT any network
access, LLM calls, or dataset downloads.  They prove the offline path works
end-to-end and that failure modes return structured, actionable errors
rather than crashing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.benchmarks.preflight import (
    PreFlightReport,
    check_nim_connectivity,
    check_python_env,
    print_report,
    run_all_checks,
)
from godspeed.benchmarks.swebench_runner import load_instances_from_file, run_swebench


def _pop_nim_keys() -> None:
    """Remove NIM key env vars for a test.

    Never clear the whole environment: on Windows an empty environment
    breaks Winsock initialization (WinError 10106) for the entire pytest
    process, poisoning every later network-dependent test.
    """
    os.environ.pop("NVIDIA_NIM_API_KEYS", None)
    os.environ.pop("NVIDIA_NIM_API_KEY", None)


# ---------------------------------------------------------------------------
# Pre-flight: missing API key
# ---------------------------------------------------------------------------


class TestPreflightMissingKey:
    def test_missing_key_is_structured_fatal(self) -> None:
        """Missing NIM key yields a structured fatal check, not a crash."""
        report = PreFlightReport()
        _pop_nim_keys()
        with patch.dict(os.environ, {}, clear=False):
            check_nim_connectivity(report)
        assert not report.all_passed
        nim = next(r for r in report.results if r.name == "NIM keys")
        assert nim.passed is False
        assert nim.fatal is True
        assert "NVIDIA_NIM_API_KEYS" in nim.detail
        assert "NVIDIA_NIM_API_KEY" in nim.detail

    def test_run_all_checks_missing_key_no_crash(self) -> None:
        """run_all_checks with no keys returns a report, never raises."""
        _pop_nim_keys()
        with patch.dict(os.environ, {}, clear=False):
            report = run_all_checks(skip_network=False, fetcher=_raise_offline)
        assert isinstance(report, PreFlightReport)
        assert not report.all_passed
        assert any(r.fatal and not r.passed for r in report.results)


# ---------------------------------------------------------------------------
# Pre-flight: bad key format
# ---------------------------------------------------------------------------


class TestPreflightBadKeyFormat:
    def test_blank_key_after_split_is_fatal(self) -> None:
        """A key string that splits to only whitespace is a fatal error."""
        report = PreFlightReport()
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEYS": "   , ,  "}):
            check_nim_connectivity(report)
        assert not report.all_passed
        nim = next(r for r in report.results if r.name == "NIM keys")
        assert nim.passed is False
        assert nim.fatal is True
        assert "No non-empty keys" in nim.detail


# ---------------------------------------------------------------------------
# Pre-flight: package import failure must not crash the report
# ---------------------------------------------------------------------------


class TestPreflightImportFailure:
    def test_non_import_error_import_failure_is_reported_not_crashed(self) -> None:
        """A package whose import raises AttributeError (circular import) is
        reported as a failed check, not propagated as a crash."""
        report = PreFlightReport()

        def _boom_import(name: str, *args: object, **kwargs: object) -> object:
            raise AttributeError("partially initialized module (circular import)")

        with patch("builtins.__import__", side_effect=_boom_import):
            check_python_env(report)
        assert not report.all_passed
        # Every package check should be present and failed, none fatal except godspeed
        failed = [
            r for r in report.results if r.name.startswith("Python: ") and "sb-cli" not in r.name
        ]
        assert len(failed) == 3
        assert all(not r.passed for r in failed)
        godspeed_check = next(r for r in failed if "godspeed" in r.name)
        assert godspeed_check.fatal is True
        assert "circular import" in godspeed_check.detail


# ---------------------------------------------------------------------------
# Pre-flight: present key but no network (offline)
# ---------------------------------------------------------------------------


def _raise_offline(url: str, headers: dict[str, str] | None = None, timeout: int = 10) -> None:
    raise OSError("offline: no route to host")


class TestPreflightOffline:
    def test_offline_key_fails_structured(self) -> None:
        """A key present but unreachable yields structured per-key + fatal."""
        report = PreFlightReport()
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEYS": "nvapi-secret-key-123"}):
            check_nim_connectivity(report, fetcher=_raise_offline)
        assert not report.all_passed
        key_check = next(r for r in report.results if r.name == "NIM key #1")
        assert key_check.passed is False
        assert "failed" in key_check.detail
        conn = next(r for r in report.results if r.name == "NIM connectivity")
        assert conn.passed is False
        assert conn.fatal is True

    def test_offline_dry_run_skips_network(self) -> None:
        """skip_network=True records a skipped check and never calls the fetcher."""
        report = PreFlightReport()
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEYS": "nvapi-secret-key-123"}):
            report = run_all_checks(skip_network=True)
        conn = next(r for r in report.results if r.name == "NIM connectivity")
        assert conn.passed is True
        assert "skipped" in conn.detail
        # No per-key failure checks were recorded because no network was attempted
        assert not any(r.name.startswith("NIM key #") for r in report.results)


# ---------------------------------------------------------------------------
# Pre-flight: report printing must not crash on Windows cp1252
# ---------------------------------------------------------------------------


class TestPreflightPrintReport:
    def test_print_report_ascii_safe_on_cp1252(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_report must not raise UnicodeEncodeError when stdout is cp1252.

        Regression: the disk-space detail used '≥' (U+2265) which cp1252
        cannot encode, crashing the CLI on Windows consoles.
        """
        report = PreFlightReport()
        report.add("Disk space", True, "10.0 GB free (>= 20 GB required)")
        report.add(
            "NIM keys",
            False,
            "Neither NVIDIA_NIM_API_KEYS nor NVIDIA_NIM_API_KEY is set",
            fatal=True,
        )
        rc = print_report(report)
        captured = capsys.readouterr()
        assert rc == 1
        assert "Disk space" in captured.out
        assert "FATAL" in captured.out


# ---------------------------------------------------------------------------
# SWE-bench runner: dry-run / offline planning
# ---------------------------------------------------------------------------


def _fake_loader(split: str) -> list[dict]:
    return [
        {
            "instance_id": "repo__proj-1",
            "repo": "repo/proj",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
        },
        {
            "instance_id": "repo__proj-2",
            "repo": "repo/proj",
            "base_commit": "def456",
            "problem_statement": "Fix another bug",
        },
    ]


class TestLoadInstancesFromFile:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "instances.jsonl"
        f.write_text(
            '{"instance_id": "a", "repo": "r/a", "base_commit": "c1", "problem_statement": "p1"}\n'
            '{"instance_id": "b", "repo": "r/b", "base_commit": "c2", "problem_statement": "p2"}\n',
            encoding="utf-8",
        )
        loader = load_instances_from_file(f)
        instances = loader("test")
        assert [i["instance_id"] for i in instances] == ["a", "b"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        loader = load_instances_from_file(tmp_path / "nope.jsonl")
        with pytest.raises(FileNotFoundError):
            loader("test")

    def test_malformed_line_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n", encoding="utf-8")
        loader = load_instances_from_file(f)
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader("test")

    def test_utf8_bom_file_loads(self, tmp_path: Path) -> None:
        """Files written with a UTF-8 BOM (PowerShell 5.1 Set-Content) must load."""
        f = tmp_path / "bom.jsonl"
        f.write_bytes(
            b"\xef\xbb\xbf"
            b'{"instance_id": "a", "repo": "r/a", "base_commit": "c1", "problem_statement": "p1"}\n'
        )
        loader = load_instances_from_file(f)
        instances = loader("test")
        assert [i["instance_id"] for i in instances] == ["a"]


class TestSwebenchDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_returns_plan_without_network(self, tmp_path: Path) -> None:
        """Dry-run with an injected loader returns a valid plan, no network."""
        _pop_nim_keys()
        with patch.dict(os.environ, {}, clear=False):
            summary = await run_swebench(
                model="nvidia_nim/deepseek-ai/deepseek-v4-pro",
                split="test",
                out=tmp_path / "pred.jsonl",
                dry_run=True,
                instance_loader=_fake_loader,
            )
        assert summary["dry_run"] is True
        assert summary["total"] == 2
        assert summary["instance_ids"] == ["repo__proj-1", "repo__proj-2"]
        assert summary["errors"] == 0
        assert summary["nim_keys_configured"] == 0
        assert summary["model"] == "nvidia_nim/deepseek-ai/deepseek-v4-pro"

    @pytest.mark.asyncio
    async def test_dry_run_respects_instance_ids(self, tmp_path: Path) -> None:
        """Dry-run instance selection honors the instance_ids filter."""
        _pop_nim_keys()
        with patch.dict(os.environ, {}, clear=False):
            summary = await run_swebench(
                model="m",
                split="test",
                instance_ids=["repo__proj-2"],
                out=tmp_path / "pred.jsonl",
                dry_run=True,
                instance_loader=_fake_loader,
            )
        assert summary["total"] == 1
        assert summary["instance_ids"] == ["repo__proj-2"]

    @pytest.mark.asyncio
    async def test_dry_run_respects_instances_cap(self, tmp_path: Path) -> None:
        """Dry-run instance selection honors the instances cap."""
        _pop_nim_keys()
        with patch.dict(os.environ, {}, clear=False):
            summary = await run_swebench(
                model="m",
                split="test",
                instances=1,
                out=tmp_path / "pred.jsonl",
                dry_run=True,
                instance_loader=_fake_loader,
            )
        assert summary["total"] == 1
        assert summary["instance_ids"] == ["repo__proj-1"]

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write_predictions(self, tmp_path: Path) -> None:
        """Dry-run must not write prediction output files."""
        out = tmp_path / "pred.jsonl"
        _pop_nim_keys()
        with patch.dict(os.environ, {}, clear=False):
            await run_swebench(
                model="m",
                split="test",
                out=out,
                dry_run=True,
                instance_loader=_fake_loader,
            )
        assert not out.exists()

    @pytest.mark.asyncio
    async def test_dry_run_with_nim_keys_reports_count(self, tmp_path: Path) -> None:
        """Dry-run reports the configured NIM key count without network."""
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEYS": "k1,k2"}):
            summary = await run_swebench(
                model="m",
                split="test",
                out=tmp_path / "pred.jsonl",
                dry_run=True,
                instance_loader=_fake_loader,
            )
        assert summary["nim_keys_configured"] == 2
