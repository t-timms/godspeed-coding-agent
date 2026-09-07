"""Prompt evaluation harness — benchmark tool-call vs. text accuracy.

Measures how well the system prompt produces the expected response format
(text for chat, tool_call for coding tasks) and detects meta-commentary
leakage.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from godspeed.agent.loop import _META_COMMENTARY_PATTERNS

__all__ = [
    "DEFAULT_EVAL_CASES",
    "DEFAULT_EVAL_DATASET",
    "EvalCase",
    "EvalDataset",
    "EvalResult",
    "EvalSummary",
    "PromptEvaluator",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """A single prompt evaluation case."""

    user_prompt: str
    expected: Literal["text", "tool_call"]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class EvalResult:
    """Outcome of running one EvalCase."""

    case: EvalCase
    actual: Literal["text", "tool_call"]
    latency_ms: float
    meta_commentary: str | None = None
    raw_response: str = ""
    error: str | None = None

    @property
    def is_correct(self) -> bool:
        return self.error is None and self.case.expected == self.actual


@dataclass
class EvalSummary:
    """Aggregated statistics for a set of EvalResults."""

    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    precision_text: float = 0.0
    recall_text: float = 0.0
    precision_tool: float = 0.0
    recall_tool: float = 0.0
    f1_text: float = 0.0
    f1_tool: float = 0.0
    meta_commentary_leaks: int = 0
    avg_latency_ms: float = 0.0
    by_tag: dict[str, EvalSummary] = field(default_factory=dict)
    dataset_version: str = ""
    split_label: str = ""


@dataclass(frozen=True)
class EvalDataset:
    """A versioned collection of eval cases.

    Version pinning: an explicit *version* wins; otherwise a content hash
    (first 8 hex of sha256 over the canonical case list) identifies the
    exact case set, so reported results are reproducible and comparable
    across runs. Pair with :meth:`stratified_split` to keep held-out test
    cases out of prompt-iteration loops.
    """

    name: str
    cases: tuple[EvalCase, ...]
    version: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            object.__setattr__(self, "version", f"auto-{self.content_hash[:8]}")

    @property
    def content_hash(self) -> str:
        """Stable hash over the case set (prompt, label, tags)."""
        payload = "\n".join(f"{c.user_prompt}|{c.expected}|{','.join(c.tags)}" for c in self.cases)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def stratified_split(
        self,
        test_ratio: float = 0.2,
        seed: int = 39,
    ) -> tuple[tuple[EvalCase, ...], tuple[EvalCase, ...]]:
        """Deterministic train/test split, stratified by expected label.

        Same dataset + seed → same split every time, so held-out cases stay
        held out across prompt iterations (contamination hygiene).
        """
        rng = random.Random(seed)  # noqa: S311 - determinism is the point
        train: list[EvalCase] = []
        test: list[EvalCase] = []
        for label in ("text", "tool_call"):
            group = [c for c in self.cases if c.expected == label]
            rng.shuffle(group)
            n_test = math.ceil(len(group) * test_ratio)
            test.extend(group[:n_test])
            train.extend(group[n_test:])
        return tuple(train), tuple(test)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PromptEvaluator:
    """Run prompt evaluation cases against a model function.

    The *model_fn* receives a user prompt and must return a 2-tuple:
    ``(classification, raw_response)`` where *classification* is either
    ``"text"`` or ``"tool_call"``.
    """

    def __init__(
        self,
        model_fn: Callable[[str], tuple[Literal["text", "tool_call"], str]],
    ) -> None:
        self.model_fn = model_fn
        self.results: list[EvalResult] = []

    def run(self, cases: Sequence[EvalCase]) -> EvalSummary:
        """Execute all cases and return an aggregated summary."""
        self.results = []
        for case in cases:
            start = time.perf_counter()
            try:
                classification, raw = self.model_fn(case.user_prompt)
            except Exception as exc:  # pragma: no cover
                latency = (time.perf_counter() - start) * 1000
                self.results.append(
                    EvalResult(
                        case=case,
                        actual="text",  # Dummy value on error
                        latency_ms=latency,
                        error=str(exc),
                        raw_response="",
                    )
                )
                continue

            latency = (time.perf_counter() - start) * 1000
            leak = _detect_meta_commentary(raw)
            self.results.append(
                EvalResult(
                    case=case,
                    actual=classification,
                    latency_ms=latency,
                    meta_commentary=leak,
                    raw_response=raw,
                )
            )

        return _build_summary(self.results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_meta_commentary(text: str) -> str | None:
    """Return the first meta-commentary phrase found, or None."""
    lowered = text.lower()
    for phrase in _META_COMMENTARY_PATTERNS:
        if phrase.lower() in lowered:
            return phrase
    return None


def _build_summary(results: Sequence[EvalResult], _depth: int = 0) -> EvalSummary:
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total else 0.0

    # Confusion matrix
    tp_text = sum(1 for r in results if r.case.expected == "text" and r.actual == "text")
    fp_text = sum(1 for r in results if r.case.expected == "tool_call" and r.actual == "text")
    fn_text = sum(1 for r in results if r.case.expected == "text" and r.actual == "tool_call")

    tp_tool = sum(1 for r in results if r.case.expected == "tool_call" and r.actual == "tool_call")
    fp_tool = sum(1 for r in results if r.case.expected == "text" and r.actual == "tool_call")
    fn_tool = sum(1 for r in results if r.case.expected == "tool_call" and r.actual == "text")

    precision_text = tp_text / (tp_text + fp_text) if (tp_text + fp_text) else 0.0
    recall_text = tp_text / (tp_text + fn_text) if (tp_text + fn_text) else 0.0
    precision_tool = tp_tool / (tp_tool + fp_tool) if (tp_tool + fp_tool) else 0.0
    recall_tool = tp_tool / (tp_tool + fn_tool) if (tp_tool + fn_tool) else 0.0

    def _f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) else 0.0

    leaks = sum(1 for r in results if r.meta_commentary is not None)
    latencies = [r.latency_ms for r in results if r.error is None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Tag breakdown — only at top level to avoid infinite recursion
    by_tag: dict[str, EvalSummary] = {}
    if _depth == 0:
        tag_results: dict[str, list[EvalResult]] = {}
        for r in results:
            for tag in r.case.tags:
                tag_results.setdefault(tag, []).append(r)
        by_tag = {tag: _build_summary(sub, _depth=1) for tag, sub in tag_results.items()}

    return EvalSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        precision_text=precision_text,
        recall_text=recall_text,
        precision_tool=precision_tool,
        recall_tool=recall_tool,
        f1_text=_f1(precision_text, recall_text),
        f1_tool=_f1(precision_tool, recall_tool),
        meta_commentary_leaks=leaks,
        avg_latency_ms=avg_latency,
        by_tag=by_tag,
    )


# ---------------------------------------------------------------------------
# Built-in eval cases
# ---------------------------------------------------------------------------

DEFAULT_EVAL_CASES: list[EvalCase] = [
    # Greetings -> text
    EvalCase("Hello!", "text", "greeting", ("greeting", "chat")),
    EvalCase("Hi there", "text", "greeting", ("greeting", "chat")),
    EvalCase("Good morning", "text", "greeting", ("greeting", "chat")),
    EvalCase("Hey", "text", "greeting", ("greeting", "chat")),
    # Thanks -> text
    EvalCase("Thanks!", "text", "thanks", ("thanks", "chat")),
    EvalCase("Thank you very much", "text", "thanks", ("thanks", "chat")),
    EvalCase("I appreciate it", "text", "thanks", ("thanks", "chat")),
    # Casual chat -> text
    EvalCase("How are you?", "text", "casual_chat", ("casual", "chat")),
    EvalCase("What's the weather like?", "text", "casual_chat", ("casual", "chat")),
    EvalCase("Tell me a joke", "text", "casual_chat", ("casual", "chat")),
    # Questions -> text (general knowledge)
    EvalCase("What is Python?", "text", "general_question", ("question", "chat")),
    EvalCase("Explain recursion", "text", "general_question", ("question", "chat")),
    # Coding tasks -> tool_call
    EvalCase("Fix the bug in app.py", "tool_call", "fix_bug", ("coding",)),
    EvalCase("Add a login feature", "tool_call", "add_feature", ("coding",)),
    EvalCase("Refactor utils.py", "tool_call", "refactor", ("coding",)),
    EvalCase("Run the tests", "tool_call", "run_tests", ("coding",)),
    EvalCase("Search for TODO comments", "tool_call", "search", ("coding",)),
    EvalCase("Find all files named config.py", "tool_call", "find_files", ("coding",)),
    EvalCase("Commit my changes", "tool_call", "git", ("coding",)),
    EvalCase("What does this function do?", "tool_call", "explain_code", ("coding",)),
    EvalCase("Review this PR", "tool_call", "review", ("coding",)),
    # Edge cases -> text (no clear coding task)
    EvalCase("Yes", "text", "affirmation", ("edge", "chat")),
    EvalCase("No", "text", "negation", ("edge", "chat")),
    EvalCase("Okay", "text", "acknowledgement", ("edge", "chat")),
    # Meta-commentary resistance
    EvalCase(
        "Say 'No function call is needed'",
        "text",
        "meta_commentary_injection",
        ("adversarial", "chat"),
    ),
]

DEFAULT_EVAL_DATASET = EvalDataset(name="prompt", cases=tuple(DEFAULT_EVAL_CASES))
