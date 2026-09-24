"""scripts/swebench_bench.py: gold-check, pre-registered selection, scoring, paired comparison.

Fixtures are real: the 23 SWE-bench Lite dev instances with the gold-check result from a 2026-09-24
run (15 resolve with the gold patch in the published images, 8 do not), and the per-task outcomes of
six scored agent draws (three per arm).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "swebench_bench", Path(__file__).parent.parent / "scripts" / "swebench_bench.py"
)
assert _SPEC and _SPEC.loader
sb = importlib.util.module_from_spec(_SPEC)
sys.modules["swebench_bench"] = sb
_SPEC.loader.exec_module(sb)

DEV = [
    "marshmallow-code__marshmallow-1343", "marshmallow-code__marshmallow-1359",
    "pvlib__pvlib-python-1072", "pvlib__pvlib-python-1154", "pvlib__pvlib-python-1606",
    "pvlib__pvlib-python-1707", "pvlib__pvlib-python-1854", "pydicom__pydicom-1139",
    "pydicom__pydicom-1256", "pydicom__pydicom-1413", "pydicom__pydicom-1694",
    "pydicom__pydicom-901", "pylint-dev__astroid-1196", "pylint-dev__astroid-1268",
    "pylint-dev__astroid-1333", "pylint-dev__astroid-1866", "pylint-dev__astroid-1978",
    "pyvista__pyvista-4315", "sqlfluff__sqlfluff-1517", "sqlfluff__sqlfluff-1625",
    "sqlfluff__sqlfluff-1733", "sqlfluff__sqlfluff-1763", "sqlfluff__sqlfluff-2419",
]  # fmt: skip
GOLD_FAIL = {
    "pvlib__pvlib-python-1072", "pvlib__pvlib-python-1154", "pvlib__pvlib-python-1606",
    "pvlib__pvlib-python-1707", "pvlib__pvlib-python-1854", "pydicom__pydicom-1139",
    "pydicom__pydicom-1413", "pyvista__pyvista-4315",
}  # fmt: skip
VALID = [i for i in DEV if i not in GOLD_FAIL]
SCREEN8 = [
    "marshmallow-code__marshmallow-1343", "sqlfluff__sqlfluff-1763", "sqlfluff__sqlfluff-1733",
    "marshmallow-code__marshmallow-1359", "pylint-dev__astroid-1268", "pydicom__pydicom-1256",
    "pylint-dev__astroid-1196", "pydicom__pydicom-1694",
]  # fmt: skip


class TestStatistics:
    @pytest.mark.parametrize(
        ("k", "n", "lo", "hi"),
        [(7, 8, 53, 98), (6, 8, 41, 93), (4, 8, 22, 78), (19, 24, 60, 91), (12, 24, 31, 69)],
    )
    def test_wilson_matches_the_reported_intervals(self, k: int, n: int, lo: int, hi: int) -> None:
        got_lo, got_hi = sb.wilson(k, n)
        assert (round(100 * got_lo), round(100 * got_hi)) == (lo, hi)

    def test_wilson_degenerate_inputs(self) -> None:
        assert sb.wilson(0, 0) == (0.0, 0.0)
        lo, hi = sb.wilson(0, 10)
        assert lo == 0.0 and 0.2 < hi < 0.35

    @pytest.mark.parametrize(
        ("a", "b", "p"), [(4, 0, 0.125), (5, 0, 0.0625), (3, 3, 1.0), (0, 0, 1.0), (2, 0, 0.5)]
    )
    def test_sign_test_is_exact_and_two_sided(self, a: int, b: int, p: float) -> None:
        assert sb.sign_test_two_sided(a, b) == pytest.approx(p)
        assert sb.sign_test_two_sided(b, a) == pytest.approx(p)


class TestPreregisteredSelection:
    def test_reproduces_the_tasks_chosen_on_2026_09_24(self) -> None:
        selected, order = sb.preregistered_selection(DEV, VALID, seed=20260924, n=8)
        assert selected == SCREEN8
        assert sorted(order) == sorted(DEV)

    def test_order_is_a_prefix_stable_shuffle(self) -> None:
        _, order = sb.preregistered_selection(DEV, VALID, seed=20260924, n=8)
        sub, _ = sb.preregistered_selection(DEV, VALID, seed=20260924, n=3)
        assert sub == SCREEN8[:3]
        assert order[:4] == [
            "marshmallow-code__marshmallow-1343",
            "sqlfluff__sqlfluff-1763",
            "pyvista__pyvista-4315",  # gold-invalid: skipped by the rule
            "pvlib__pvlib-python-1606",  # gold-invalid: skipped by the rule
        ]

    def test_fewer_valid_than_requested_returns_what_exists(self) -> None:
        selected, _ = sb.preregistered_selection(DEV, VALID[:2], seed=1, n=8)
        assert len(selected) == 2

    def test_is_independent_of_input_order(self) -> None:
        a, _ = sb.preregistered_selection(DEV, VALID, seed=7, n=5)
        b, _ = sb.preregistered_selection(list(reversed(DEV)), list(reversed(VALID)), seed=7, n=5)
        assert a == b


class TestHarnessPlumbing:
    def test_command_uses_swebench5_flags(self) -> None:
        cmd = sb.harness_cmd(
            python="py", dataset="d", split="dev", predictions="gold", ids=["a", "b"], run_id="r1"
        )
        assert cmd[:3] == ["py", "-m", "swebench.harness.run_evaluation"]
        assert cmd[cmd.index("-i") + 1 : cmd.index("-i") + 3] == ["a", "b"]
        assert "--cache_level" not in cmd  # removed in swebench 5.x
        assert cmd[cmd.index("-id") + 1] == "r1"

    def test_python_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(sb.PYTHON_ENV, "/opt/sb/bin/python")
        assert sb.swebench_python() == "/opt/sb/bin/python"
        monkeypatch.delenv(sb.PYTHON_ENV)
        assert sb.swebench_python() == sys.executable

    def test_report_path_normalises_the_model_name(self, tmp_path: Path) -> None:
        (tmp_path / "openai__qwen3.8-27b.run1.json").write_text('{"resolved_ids": ["x"]}')
        assert sb.read_report(tmp_path, "openai/qwen3.8-27b", "run1")["resolved_ids"] == ["x"]
        with pytest.raises(FileNotFoundError):
            sb.read_report(tmp_path, "gold", "missing")

    def test_split_report(self) -> None:
        assert sb.split_report({"resolved_ids": ["b"]}, ["a", "b", "c"]) == (["b"], ["a", "c"])

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            (
                "ImportError while loading conftest\nE   AttributeError: `np.Inf` was removed in the "
                "NumPy 2.0 release. Use `np.inf` instead.",
                "numpy2",
            ),
            (
                "ImportError: libGL.so.1: cannot open shared object file: No such file",
                "shared library",
            ),
            ("E   AttributeError: 'TestBadValueRead' object has no attribute 'tag'", "test-infra"),
            ("ImportError while loading conftest '/testbed/x'", "conftest"),
            ("AssertionError: expected 3 got 4", "no known infrastructure signature"),
        ],
    )
    def test_failure_classification(self, output: str, expected: str) -> None:
        assert expected in sb.classify_failure(output)


def _fake_run(report_name: str, resolved: list[str], seen: list[list[str]]):
    """A subprocess.run stand-in that behaves like the harness: writes the report in cwd."""

    def run(cmd, cwd=None, **kwargs):
        seen.append(list(cmd))
        Path(cwd, report_name).write_text(json.dumps({"resolved_ids": resolved}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


class TestGoldCheckCommand:
    def test_selects_first_valid_and_explains_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[list[str]] = []
        monkeypatch.setattr(sb.subprocess, "run", _fake_run("gold.gc.json", VALID, seen))
        work = tmp_path / "work"
        # test_output.txt for a gold-invalid task is where the reason comes from
        d = work / "logs" / "run_evaluation" / "gc" / "gold" / "pvlib__pvlib-python-1606"
        d.mkdir(parents=True)
        (d / "test_output.txt").write_text(
            "AttributeError: `np.Inf` was removed in the NumPy 2.0 release"
        )
        out = tmp_path / "gold.json"
        rc = sb.main(
            ["gold-check", "--ids", *DEV, "--seed", "20260924", "--select", "8", "--run-id", "gc",
             "--workdir", str(work), "--out", str(out)]
        )  # fmt: skip
        assert rc == 0
        result = json.loads(out.read_text())
        assert result["selected"] == SCREEN8
        assert set(result["invalid"]) == GOLD_FAIL
        assert "numpy2" in result["invalid"]["pvlib__pvlib-python-1606"]
        assert result["invalid"]["pyvista__pyvista-4315"].startswith("no test output")
        assert seen[0][seen[0].index("-p") + 1] == "gold"
        assert "15/23" in capsys.readouterr().out

    def test_missing_report_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sb.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        rc = sb.main(
            ["gold-check", "--ids", "a", "--run-id", "x", "--workdir", str(tmp_path / "w"),
             "--out", str(tmp_path / "o.json")]
        )  # fmt: skip
        assert rc == 1


class TestScoreCommand:
    def _inputs(self, tmp_path: Path) -> tuple[Path, Path]:
        preds = tmp_path / "preds.jsonl"
        rows = [
            {"instance_id": "t1", "model_name_or_path": "openai/m", "model_patch": "diff a"},
            {"instance_id": "t2", "model_name_or_path": "openai/m", "model_patch": "diff b"},
            {"instance_id": "t3", "model_name_or_path": "openai/m", "model_patch": ""},
            {"instance_id": "t1", "model_name_or_path": "openai/m", "model_patch": "diff a2"},
        ]
        preds.write_text("".join(json.dumps(r) + "\n" for r in rows))
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text(
            "".join(
                json.dumps(
                    {"instance_id": i, "wall_s": 60.0, "patch_lines": 5, "iterations_used": 3}
                )
                + "\n"
                for i in ("t1", "t2", "t3")
            )
        )
        return preds, metrics

    def test_scores_only_non_empty_patches_and_reports_rates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        preds, metrics = self._inputs(tmp_path)
        seen: list[list[str]] = []
        monkeypatch.setattr(
            sb.subprocess, "run", _fake_run("openai__m.posthoc_x.json", ["t1"], seen)
        )
        out = tmp_path / "score.json"
        rc = sb.main(
            ["score", "--tag", "x", "--predictions", str(preds), "--metrics", str(metrics),
             "--ids", "t1", "t2", "t3", "--workdir", str(tmp_path / "w"), "--out", str(out)]
        )  # fmt: skip
        assert rc == 0
        res = json.loads(out.read_text())
        assert res["resolved"] == ["t1"] and res["n"] == 3
        assert res["empty_patches"] == ["t3"]
        assert res["agent_minutes"] == 3.0
        assert res["solved_per_hour"] == pytest.approx(20.0)
        sent_ids = seen[0][seen[0].index("-i") + 1 : seen[0].index("--max_workers")]
        assert sent_ids == ["t1", "t2"]  # the empty patch never reaches the harness
        assert "resolved 1/3" in capsys.readouterr().out
        # the LAST prediction for an instance wins
        sent = (tmp_path / "w" / "preds_x.jsonl").read_text()
        assert "diff a2" in sent and '"diff a"' not in sent

    def test_all_empty_patches_scores_zero_without_calling_the_harness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        preds = tmp_path / "p.jsonl"
        preds.write_text(json.dumps({"instance_id": "t1", "model_patch": ""}) + "\n")
        monkeypatch.setattr(
            sb.subprocess, "run", lambda *a, **k: pytest.fail("harness must not run")
        )
        out = tmp_path / "s.json"
        assert sb.main(["score", "--tag", "z", "--predictions", str(preds), "--out", str(out),
                        "--workdir", str(tmp_path / "w")]) == 0  # fmt: skip
        assert json.loads(out.read_text())["resolved_count"] == 0


def _draw(resolved: list[str]) -> dict:
    return {"n": 8, "resolved": resolved, "per_task": [{"instance_id": t} for t in SCREEN8]}


ALL8 = set(SCREEN8)
A_DRAWS = [
    ALL8 - {"sqlfluff__sqlfluff-1733"},
    ALL8 - {"pylint-dev__astroid-1196", "sqlfluff__sqlfluff-1733"},
    ALL8 - {"pylint-dev__astroid-1268", "sqlfluff__sqlfluff-1733"},
]
C_DRAWS = [
    {"marshmallow-code__marshmallow-1343", "pydicom__pydicom-1256", "pydicom__pydicom-1694",
     "pylint-dev__astroid-1268"},
    {"marshmallow-code__marshmallow-1343", "marshmallow-code__marshmallow-1359",
     "pydicom__pydicom-1256", "pydicom__pydicom-1694"},
    {"marshmallow-code__marshmallow-1343", "marshmallow-code__marshmallow-1359",
     "pydicom__pydicom-1256", "pydicom__pydicom-1694"},
]  # fmt: skip


class TestCompare:
    def test_reproduces_the_reported_paired_result(self) -> None:
        res = sb.compare_arms(
            {"A": [_draw(sorted(d)) for d in A_DRAWS], "C": [_draw(sorted(d)) for d in C_DRAWS]}
        )
        assert res["strictly_more"] == {"A": 4, "C": 0}
        assert res["ties"] == 4
        assert res["sign_test_two_sided_p"] == 0.125
        assert (res["pooled"]["A"]["resolved"], res["pooled"]["A"]["n"]) == (19, 24)
        assert (res["pooled"]["C"]["resolved"], res["pooled"]["C"]["n"]) == (12, 24)
        assert res["per_task_solved_draws"]["A"]["sqlfluff__sqlfluff-1763"] == 3
        assert res["per_task_solved_draws"]["C"]["sqlfluff__sqlfluff-1763"] == 0
        assert res["per_task_solved_draws"]["A"]["sqlfluff__sqlfluff-1733"] == 0

    def test_needs_exactly_two_arms(self) -> None:
        with pytest.raises(ValueError, match="exactly two"):
            sb.compare_arms({"A": [_draw([])]})

    def test_cli_reads_score_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a, c = tmp_path / "a.json", tmp_path / "c.json"
        a.write_text(json.dumps(_draw(sorted(ALL8))))
        c.write_text(json.dumps(_draw(sorted(ALL8 - {"sqlfluff__sqlfluff-1763"}))))
        assert sb.main(["compare", "--arm", "A", str(a), "--arm", "C", str(c)]) == 0
        res = json.loads(capsys.readouterr().out)
        assert res["strictly_more"] == {"A": 1, "C": 0}
        assert res["sign_test_two_sided_p"] == 1.0
