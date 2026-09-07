"""Safety gate — prevent regressions from evolved artifacts.

All mutations must pass safety checks before being applied:
- Test suite still passes (100%)
- Size limits respected (no >2x growth)
- Semantic drift within bounds
- Fitness above threshold
- High-impact changes flagged for human review
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import sys
from collections.abc import Awaitable, Callable

from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """Result of running all safety checks on a mutation candidate."""

    passed: bool
    checks: tuple[tuple[str, bool, str], ...]  # (check_name, passed, reason)
    requires_human_review: bool


# Artifacts that always need human review
HIGH_IMPACT_ARTIFACT_TYPES = frozenset({"prompt_section"})
HIGH_IMPACT_ARTIFACT_IDS = frozenset({"core", "security", "permissions"})

# Tool IDs whose descriptions gate dangerous or destructive actions.
# Mutations to these tool descriptions must be reviewed by a human before
# taking effect — the LLM reads the description to decide when to use the
# tool, so an innocuous-looking reword can materially weaken safety.
SECURITY_SENSITIVE_TOOL_IDS = frozenset(
    {
        "shell",
        "bash",
        "file_write",
        "file_edit",
        "diff_apply",
        "git",
        "github",
        "background",
    }
)

# Phrases that indicate a mutation is trying to relax or disable safety,
# regardless of which artifact it targets. Matched case-insensitively as
# regular expressions (whitespace is flexible).
SECURITY_BYPASS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"always\s+granted", re.IGNORECASE),
    re.compile(r"bypass\s+(?:the\s+)?permission", re.IGNORECASE),
    re.compile(r"ignore\s+(?:the\s+)?safety", re.IGNORECASE),
    re.compile(r"skip\s+confirmation", re.IGNORECASE),
    re.compile(r"auto[-\s]?approve", re.IGNORECASE),
    re.compile(r"without\s+permission", re.IGNORECASE),
    re.compile(r"disable\s+audit", re.IGNORECASE),
    re.compile(r"no\s+(?:permission|audit|confirmation)\s+(?:needed|required)", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Safety Gate
# ---------------------------------------------------------------------------


class SafetyGate:
    """Run safety checks on mutation candidates before applying them."""

    def __init__(
        self,
        max_growth: float = 2.0,
        min_similarity: float = 0.3,
        min_fitness: float = 0.6,
        test_suite_timeout: float = 120.0,
        test_runner: Callable[[], Awaitable[tuple[bool, str]]] | None = None,
    ) -> None:
        self._max_growth = max_growth
        self._min_similarity = min_similarity
        self._min_fitness = min_fitness
        self._test_suite_timeout = test_suite_timeout
        self._test_runner = test_runner

    async def gate(
        self,
        candidate: MutationCandidate,
        score: FitnessScore,
    ) -> SafetyVerdict:
        """Run all safety checks and return a verdict.

        Runs synchronous checks (size, drift, fitness, confidence) plus the
        async test suite.  A test-suite error or timeout is treated as a
        failure (fail-closed) so that no mutation can bypass verification.
        """
        checks: list[tuple[str, bool, str]] = []

        size_ok, size_msg = self.check_size_limit(candidate)
        checks.append(("size_limit", size_ok, size_msg))

        drift_ok, drift_msg = self.check_semantic_drift(candidate)
        checks.append(("semantic_drift", drift_ok, drift_msg))

        fitness_ok, fitness_msg = self.check_fitness_threshold(score)
        checks.append(("fitness_threshold", fitness_ok, fitness_msg))

        conf_ok = score.confidence >= 0.5
        conf_msg = f"confidence={score.confidence:.2f} (min=0.50)"
        checks.append(("confidence", conf_ok, conf_msg))

        runner = self._test_runner or self.run_test_suite
        tests_ok, tests_msg = await runner()
        checks.append(("test_suite", tests_ok, tests_msg))

        all_passed = all(ok for _, ok, _ in checks)
        needs_review = self.requires_human_review(candidate)

        return SafetyVerdict(
            passed=all_passed,
            checks=tuple(checks),
            requires_human_review=needs_review,
        )

    async def run_test_suite(self) -> tuple[bool, str]:
        """Run the project test suite via ``pytest`` as a subprocess.

        Returns ``(passed, message)``.  ``passed`` is True only when pytest
        exits 0 within the configured timeout.  Errors, non-zero exits, and
        timeouts all return ``(False, …)`` (fail-closed).
        """
        python = sys.executable
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                python,
                "-m",
                "pytest",
                "--tb=short",
                "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._test_suite_timeout,
            )
        except TimeoutError:
            logger.warning("test suite timed out timeout=%.0fs", self._test_suite_timeout)
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return False, f"test suite timed out ({self._test_suite_timeout:.0f}s)"
        except OSError as exc:
            logger.warning("test suite failed to launch error=%s", exc)
            return False, f"test suite launch error: {exc}"

        exit_code = proc.returncode or 0
        stdout = (stdout_bytes or b"").decode(errors="replace")
        stderr = (stderr_bytes or b"").decode(errors="replace")
        detail = (stdout + stderr).strip()
        if exit_code != 0:
            tail = detail[-300:] if len(detail) > 300 else detail
            logger.warning(
                "test suite failed exit_code=%d tail=%s",
                exit_code,
                tail,
            )
        return exit_code == 0, f"pytest exit_code={exit_code}"

    def check_size_limit(self, candidate: MutationCandidate) -> tuple[bool, str]:
        """Check that mutated text is not excessively larger than original."""
        orig_len = len(candidate.original)
        mut_len = len(candidate.mutated)

        if orig_len == 0:
            return True, "original is empty — no size check"

        ratio = mut_len / orig_len
        passed = ratio <= self._max_growth
        msg = f"size ratio={ratio:.2f} (max={self._max_growth:.1f})"
        return passed, msg

    def check_semantic_drift(self, candidate: MutationCandidate) -> tuple[bool, str]:
        """Check that mutated text hasn't drifted too far from original.

        Uses word-overlap Jaccard similarity (no embeddings needed).
        """
        orig_words = self._tokenize(candidate.original)
        mut_words = self._tokenize(candidate.mutated)

        if not orig_words and not mut_words:
            return True, "both empty"
        if not orig_words or not mut_words:
            return False, "one side is empty"

        intersection = orig_words & mut_words
        union = orig_words | mut_words
        similarity = len(intersection) / len(union) if union else 0.0

        passed = similarity >= self._min_similarity
        msg = f"similarity={similarity:.3f} (min={self._min_similarity:.2f})"
        return passed, msg

    def check_fitness_threshold(self, score: FitnessScore) -> tuple[bool, str]:
        """Check that fitness score meets minimum threshold."""
        passed = score.overall >= self._min_fitness
        msg = f"overall={score.overall:.3f} (min={self._min_fitness:.2f})"
        return passed, msg

    def requires_human_review(self, candidate: MutationCandidate) -> bool:
        """Determine if this mutation needs human approval.

        A mutation requires review when any of the following is true:
        - artifact_type is in HIGH_IMPACT_ARTIFACT_TYPES (e.g. prompt_section)
        - artifact_id is in HIGH_IMPACT_ARTIFACT_IDS (core/security/permissions)
        - artifact_type is "tool_description" AND artifact_id names a
          security-sensitive tool whose description gates dangerous actions
        - the mutated text contains any SECURITY_BYPASS_PATTERNS phrase
          (matched case-insensitively), regardless of artifact
        """
        if candidate.artifact_type in HIGH_IMPACT_ARTIFACT_TYPES:
            return True
        if candidate.artifact_id in HIGH_IMPACT_ARTIFACT_IDS:
            return True
        if (
            candidate.artifact_type == "tool_description"
            and candidate.artifact_id in SECURITY_SENSITIVE_TOOL_IDS
        ):
            return True
        return any(pattern.search(candidate.mutated) for pattern in SECURITY_BYPASS_PATTERNS)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple word tokenization for similarity comparison."""
        return set(re.findall(r"\w+", text.lower()))
