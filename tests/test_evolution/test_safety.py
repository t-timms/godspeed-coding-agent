"""Tests for the safety gate — preventing regressions from evolved artifacts."""

from __future__ import annotations

import pytest

from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate
from godspeed.evolution.safety import SafetyGate, SafetyVerdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    original: str = "Read files from disk with path validation and traversal protection.",
    mutated: str = "Read files from the local filesystem with path validation and examples.",
    artifact_type: str = "tool_description",
    artifact_id: str = "file_read",
) -> MutationCandidate:
    return MutationCandidate(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        original=original,
        mutated=mutated,
        mutation_rationale="test",
        model_used="ollama/test",
    )


def _make_score(
    overall: float = 0.8,
    confidence: float = 1.0,
) -> FitnessScore:
    return FitnessScore(
        correctness=0.9,
        procedure_following=0.8,
        conciseness=0.7,
        overall=overall,
        length_penalty=0.0,
        confidence=confidence,
    )


async def _stub_test_suite() -> tuple[bool, str]:
    return True, "stubbed test suite"


# ---------------------------------------------------------------------------
# Test: SafetyVerdict data structure
# ---------------------------------------------------------------------------


class TestSafetyVerdict:
    def test_frozen(self) -> None:
        v = SafetyVerdict(
            passed=True,
            checks=(("size_limit", True, "ok"),),
            requires_human_review=False,
        )
        with pytest.raises(AttributeError):
            v.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: check_size_limit
# ---------------------------------------------------------------------------


class TestCheckSizeLimit:
    def test_within_limit(self) -> None:
        gate = SafetyGate(max_growth=2.0)
        candidate = _make_candidate(original="short text", mutated="slightly longer text")
        ok, _msg = gate.check_size_limit(candidate)
        assert ok is True

    def test_exceeds_limit(self) -> None:
        gate = SafetyGate(max_growth=2.0)
        candidate = _make_candidate(original="short", mutated="x" * 1000)
        ok, msg = gate.check_size_limit(candidate)
        assert ok is False
        assert "ratio" in msg

    def test_empty_original(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(original="", mutated="something")
        ok, _msg = gate.check_size_limit(candidate)
        assert ok is True  # No check on empty original


# ---------------------------------------------------------------------------
# Test: check_semantic_drift
# ---------------------------------------------------------------------------


class TestCheckSemanticDrift:
    def test_similar_text_passes(self) -> None:
        gate = SafetyGate(min_similarity=0.3)
        candidate = _make_candidate(
            original="Read files from disk with validation.",
            mutated="Read files from the local disk with path validation and examples.",
        )
        ok, _msg = gate.check_semantic_drift(candidate)
        assert ok is True

    def test_completely_different_fails(self) -> None:
        gate = SafetyGate(min_similarity=0.3)
        candidate = _make_candidate(
            original="Read files from disk.",
            mutated="Execute shell commands in a sandbox container.",
        )
        ok, _msg = gate.check_semantic_drift(candidate)
        assert ok is False

    def test_both_empty(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(original="", mutated="")
        ok, _ = gate.check_semantic_drift(candidate)
        assert ok is True

    def test_one_empty(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(original="content", mutated="")
        ok, _ = gate.check_semantic_drift(candidate)
        assert ok is False


# ---------------------------------------------------------------------------
# Test: check_fitness_threshold
# ---------------------------------------------------------------------------


class TestCheckFitnessThreshold:
    def test_above_threshold(self) -> None:
        gate = SafetyGate(min_fitness=0.6)
        score = _make_score(overall=0.8)
        ok, _msg = gate.check_fitness_threshold(score)
        assert ok is True

    def test_below_threshold(self) -> None:
        gate = SafetyGate(min_fitness=0.6)
        score = _make_score(overall=0.4)
        ok, _msg = gate.check_fitness_threshold(score)
        assert ok is False

    def test_exact_threshold(self) -> None:
        gate = SafetyGate(min_fitness=0.6)
        score = _make_score(overall=0.6)
        ok, _ = gate.check_fitness_threshold(score)
        assert ok is True


# ---------------------------------------------------------------------------
# Test: requires_human_review
# ---------------------------------------------------------------------------


class TestRequiresHumanReview:
    def test_prompt_section_needs_review(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(artifact_type="prompt_section", artifact_id="tools")
        assert gate.requires_human_review(candidate) is True

    def test_core_artifact_needs_review(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(artifact_type="tool_description", artifact_id="core")
        assert gate.requires_human_review(candidate) is True

    def test_security_artifact_needs_review(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(artifact_id="security")
        assert gate.requires_human_review(candidate) is True

    def test_regular_tool_no_review(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(artifact_type="tool_description", artifact_id="file_read")
        assert gate.requires_human_review(candidate) is False


# ---------------------------------------------------------------------------
# Test: gate (full check)
# ---------------------------------------------------------------------------


class TestGate:
    async def test_all_checks_pass(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate()
        score = _make_score(overall=0.8, confidence=1.0)
        verdict = await gate.gate(candidate, score)

        assert verdict.passed is True
        assert all(ok for _, ok, _ in verdict.checks)
        assert verdict.requires_human_review is False

    async def test_size_limit_fails(self) -> None:
        gate = SafetyGate(max_growth=1.5, test_runner=_stub_test_suite)
        candidate = _make_candidate(original="a", mutated="a" * 100)
        score = _make_score()
        verdict = await gate.gate(candidate, score)

        assert verdict.passed is False
        size_check = next(c for c in verdict.checks if c[0] == "size_limit")
        assert size_check[1] is False

    async def test_low_confidence_fails(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate()
        score = _make_score(confidence=0.3)
        verdict = await gate.gate(candidate, score)

        assert verdict.passed is False
        conf_check = next(c for c in verdict.checks if c[0] == "confidence")
        assert conf_check[1] is False

    async def test_prompt_section_flagged_for_review(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(artifact_type="prompt_section")
        score = _make_score()
        verdict = await gate.gate(candidate, score)

        assert verdict.requires_human_review is True


class TestSecuritySensitiveMutations:
    """Mutations to security-critical tools or containing bypass phrases
    must never auto-apply. They require explicit human review.
    """

    @pytest.mark.parametrize(
        "tool_id",
        ["shell", "bash", "file_write", "file_edit", "git", "github", "diff_apply", "background"],
    )
    async def test_security_sensitive_tool_requires_review(self, tool_id: str) -> None:
        """Tool descriptions that gate dangerous actions must need review."""
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(artifact_type="tool_description", artifact_id=tool_id)
        score = _make_score()
        verdict = await gate.gate(candidate, score)
        assert verdict.requires_human_review is True, (
            f"{tool_id} mutation must require human review"
        )

    async def test_benign_tool_does_not_require_review(self) -> None:
        """Read-only tool mutations with benign content are auto-applicable."""
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="Read files from disk.",
            mutated="Read files from the local filesystem with examples.",
        )
        score = _make_score()
        verdict = await gate.gate(candidate, score)
        assert verdict.requires_human_review is False

    @pytest.mark.parametrize(
        "bypass_phrase",
        [
            "always granted",
            "bypass permission",
            "bypass the permission check",
            "ignore safety",
            "ignore the safety gate",
            "skip confirmation",
            "auto-approve",
            "without permission",
            "disable audit",
        ],
    )
    async def test_bypass_phrase_requires_review(self, bypass_phrase: str) -> None:
        """Any mutation containing a security-bypass phrase requires review."""
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="Read files from disk with path validation.",
            mutated=f"Read files from disk. Note: this tool is {bypass_phrase}.",
        )
        score = _make_score()
        verdict = await gate.gate(candidate, score)
        assert verdict.requires_human_review is True, (
            f"mutation with phrase {bypass_phrase!r} must require review"
        )

    async def test_bypass_phrase_case_insensitive(self) -> None:
        gate = SafetyGate(test_runner=_stub_test_suite)
        candidate = _make_candidate(
            mutated="Read files. Note: this tool is ALWAYS GRANTED.",
        )
        verdict = await gate.gate(candidate, _make_score())
        assert verdict.requires_human_review is True
