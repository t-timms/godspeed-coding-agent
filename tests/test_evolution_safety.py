"""Tests for evolution safety gate — including test-suite verification."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate
from godspeed.evolution.safety import SafetyGate


def _make_candidate(**overrides: object) -> MutationCandidate:
    defaults: dict[str, object] = {
        "artifact_type": "tool_description",
        "artifact_id": "file_read",
        "original": "Read a file from disk.",
        "mutated": "Read a file from disk safely.",
        "mutation_rationale": "improve clarity",
        "model_used": "test",
    }
    defaults.update(overrides)
    return MutationCandidate(**defaults)  # type: ignore[arg-type]


def _make_score(**overrides: object) -> FitnessScore:
    defaults: dict[str, object] = {
        "correctness": 0.8,
        "procedure_following": 0.7,
        "conciseness": 0.9,
        "overall": 0.8,
        "length_penalty": 0.0,
        "confidence": 0.6,
    }
    defaults.update(overrides)
    return FitnessScore(**defaults)  # type: ignore[arg-type]


# -- run_test_suite unit tests ------------------------------------------------


class TestRunTestSuite:
    """Unit tests for SafetyGate.run_test_suite()."""

    @pytest.mark.asyncio
    async def test_passing_suite_returns_true(self) -> None:
        gate = SafetyGate()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"5 passed", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            passed, msg = await gate.run_test_suite()

        assert passed is True
        assert "exit_code=0" in msg

    @pytest.mark.asyncio
    async def test_failing_suite_returns_false(self) -> None:
        gate = SafetyGate()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 failed", b"FAILED test_x"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            passed, msg = await gate.run_test_suite()

        assert passed is False
        assert "exit_code=1" in msg

    @pytest.mark.asyncio
    async def test_timeout_returns_false_and_kills_proc(self) -> None:
        gate = SafetyGate(test_suite_timeout=0.01)
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            passed, msg = await gate.run_test_suite()

        assert passed is False
        assert "timed out" in msg
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_error_returns_false(self) -> None:
        gate = SafetyGate()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("no such binary"),
        ):
            passed, msg = await gate.run_test_suite()

        assert passed is False
        assert "launch error" in msg


# -- gate() integration with test-suite check ---------------------------------


class TestSafetyGateTestSuiteIntegration:
    """Verify gate() includes test-suite results in verdict."""

    @pytest.mark.asyncio
    async def test_gate_includes_test_suite_check(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate()
        score = _make_score()

        with patch.object(
            gate,
            "run_test_suite",
            new_callable=AsyncMock,
            return_value=(True, "pytest exit_code=0"),
        ):
            verdict = await gate.gate(candidate, score)

        assert verdict.passed is True
        check_names = [name for name, _, _ in verdict.checks]
        assert "test_suite" in check_names

    @pytest.mark.asyncio
    async def test_gate_fails_when_suite_fails(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate()
        score = _make_score()

        with patch.object(
            gate,
            "run_test_suite",
            new_callable=AsyncMock,
            return_value=(False, "pytest exit_code=1"),
        ):
            verdict = await gate.gate(candidate, score)

        assert verdict.passed is False
        suite_check = [c for c in verdict.checks if c[0] == "test_suite"]
        assert len(suite_check) == 1
        assert suite_check[0][1] is False

    @pytest.mark.asyncio
    async def test_gate_fails_when_suite_errors(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate()
        score = _make_score()

        with patch.object(
            gate,
            "run_test_suite",
            new_callable=AsyncMock,
            return_value=(False, "test suite launch error: no pytest"),
        ):
            verdict = await gate.gate(candidate, score)

        assert verdict.passed is False

    @pytest.mark.asyncio
    async def test_gate_fails_when_other_checks_fail(self) -> None:
        gate = SafetyGate(min_fitness=0.99)
        candidate = _make_candidate()
        score = _make_score()

        with patch.object(
            gate, "run_test_suite", new_callable=AsyncMock, return_value=(True, "ok")
        ):
            verdict = await gate.gate(candidate, score)

        assert verdict.passed is False
        fitness_check = [c for c in verdict.checks if c[0] == "fitness_threshold"]
        assert len(fitness_check) == 1
        assert fitness_check[0][1] is False

    @pytest.mark.asyncio
    async def test_all_five_checks_present(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate()
        score = _make_score()

        with patch.object(
            gate, "run_test_suite", new_callable=AsyncMock, return_value=(True, "ok")
        ):
            verdict = await gate.gate(candidate, score)

        names = [name for name, _, _ in verdict.checks]
        assert names == [
            "size_limit",
            "semantic_drift",
            "fitness_threshold",
            "confidence",
            "test_suite",
        ]


# -- Existing synchronous checks (no test suite) -----------------------------


class TestSafetyGateSyncChecks:
    """Fast synchronous safety checks — unchanged behavior."""

    def test_size_limit_pass(self) -> None:
        gate = SafetyGate(max_growth=2.0)
        candidate = _make_candidate(
            original="a reasonably long description here",
            mutated="a reasonably long description here indeed",
        )
        passed, msg = gate.check_size_limit(candidate)
        assert passed is True
        assert "ratio" in msg

    def test_size_limit_fail(self) -> None:
        gate = SafetyGate(max_growth=1.5)
        candidate = _make_candidate(original="x", mutated="x" * 100)
        passed, _ = gate.check_size_limit(candidate)
        assert passed is False

    def test_semantic_drift_pass(self) -> None:
        gate = SafetyGate(min_similarity=0.1)
        candidate = _make_candidate(
            original="read the file carefully",
            mutated="read the file carefully please",
        )
        passed, _ = gate.check_semantic_drift(candidate)
        assert passed is True

    def test_semantic_drift_fail(self) -> None:
        gate = SafetyGate(min_similarity=0.9)
        candidate = _make_candidate(
            original="read the file",
            mutated="delete everything now",
        )
        passed, _ = gate.check_semantic_drift(candidate)
        assert passed is False

    def test_fitness_threshold_pass(self) -> None:
        gate = SafetyGate(min_fitness=0.5)
        score = _make_score(overall=0.8)
        passed, _ = gate.check_fitness_threshold(score)
        assert passed is True

    def test_fitness_threshold_fail(self) -> None:
        gate = SafetyGate(min_fitness=0.9)
        score = _make_score(overall=0.5)
        passed, _ = gate.check_fitness_threshold(score)
        assert passed is False

    def test_requires_human_review_high_impact(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate(artifact_type="prompt_section", artifact_id="core")
        assert gate.requires_human_review(candidate) is True

    def test_requires_human_review_security_tool(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate(
            artifact_type="tool_description",
            artifact_id="shell",
        )
        assert gate.requires_human_review(candidate) is True

    def test_requires_human_review_bypass_pattern(self) -> None:
        gate = SafetyGate()
        candidate = _make_candidate(mutated="always granted access to all files")
        assert gate.requires_human_review(candidate) is True
