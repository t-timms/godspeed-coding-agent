"""Tests for the prompt evaluation harness."""

from __future__ import annotations

from godspeed.evaluation.prompt_eval import (
    DEFAULT_EVAL_CASES,
    DEFAULT_EVAL_DATASET,
    EvalCase,
    EvalDataset,
    EvalResult,
    PromptEvaluator,
    _build_summary,
    _detect_meta_commentary,
)

# ---------------------------------------------------------------------------
# Meta-commentary detection
# ---------------------------------------------------------------------------


class TestDetectMetaCommentary:
    def test_detects_exact_phrase(self) -> None:
        assert (
            _detect_meta_commentary("No function call is needed here")
            == "No function call is needed"
        )

    def test_detects_lowercase_variant(self) -> None:
        assert _detect_meta_commentary("no tool call is needed") == "No tool call is needed"

    def test_none_for_clean_text(self) -> None:
        assert _detect_meta_commentary("Hello, how can I help?") is None

    def test_detects_first_of_many(self) -> None:
        # Returns the first match found
        result = _detect_meta_commentary("No function call is needed. No tool call is needed.")
        assert result == "No function call is needed"


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_perfect_accuracy(self) -> None:
        results = [
            EvalResult(EvalCase("hi", "text", "", ()), "text", 10.0),
            EvalResult(EvalCase("fix bug", "tool_call", "", ()), "tool_call", 20.0),
        ]
        s = _build_summary(results)
        assert s.total == 2
        assert s.correct == 2
        assert s.accuracy == 1.0
        assert s.precision_text == 1.0
        assert s.recall_text == 1.0
        assert s.precision_tool == 1.0
        assert s.recall_tool == 1.0
        assert s.f1_text == 1.0
        assert s.f1_tool == 1.0

    def test_zero_division_safety(self) -> None:
        s = _build_summary([])
        assert s.accuracy == 0.0
        assert s.precision_text == 0.0
        assert s.recall_text == 0.0

    def test_false_positives_and_negatives(self) -> None:
        results = [
            EvalResult(EvalCase("hi", "text", "", ()), "tool_call", 10.0),  # FP tool
            EvalResult(EvalCase("fix", "tool_call", "", ()), "text", 20.0),  # FN tool
        ]
        s = _build_summary(results)
        assert s.correct == 0
        assert s.accuracy == 0.0
        assert s.precision_text == 0.0
        assert s.recall_text == 0.0
        assert s.precision_tool == 0.0
        assert s.recall_tool == 0.0

    def test_meta_commentary_count(self) -> None:
        results = [
            EvalResult(
                EvalCase("hi", "text", "", ()),
                "text",
                10.0,
                meta_commentary="No function call is needed",
            ),
            EvalResult(EvalCase("hello", "text", "", ()), "text", 10.0),
        ]
        s = _build_summary(results)
        assert s.meta_commentary_leaks == 1

    def test_avg_latency(self) -> None:
        results = [
            EvalResult(EvalCase("a", "text", "", ()), "text", 100.0),
            EvalResult(EvalCase("b", "text", "", ()), "text", 200.0),
        ]
        s = _build_summary(results)
        assert s.avg_latency_ms == 150.0

    def test_tag_breakdown(self) -> None:
        results = [
            EvalResult(EvalCase("hi", "text", "", ("greeting",)), "text", 10.0),
            EvalResult(EvalCase("hello", "text", "", ("greeting",)), "text", 10.0),
            EvalResult(EvalCase("fix", "tool_call", "", ("coding",)), "tool_call", 20.0),
        ]
        s = _build_summary(results)
        assert "greeting" in s.by_tag
        assert "coding" in s.by_tag
        assert s.by_tag["greeting"].total == 2
        assert s.by_tag["coding"].total == 1


# ---------------------------------------------------------------------------
# PromptEvaluator
# ---------------------------------------------------------------------------


class TestPromptEvaluator:
    def test_perfect_model(self) -> None:
        def perfect(prompt: str) -> tuple[str, str]:
            if "fix" in prompt.lower():
                return ("tool_call", "I'll fix that")
            return ("text", "Hello!")

        ev = PromptEvaluator(perfect)
        cases = [
            EvalCase("Hello!", "text", "", ()),
            EvalCase("Fix bug", "tool_call", "", ()),
        ]
        summary = ev.run(cases)
        assert summary.accuracy == 1.0
        assert len(ev.results) == 2

    def test_imperfect_model(self) -> None:
        def bad(prompt: str) -> tuple[str, str]:
            return ("text", "No function call is needed")

        ev = PromptEvaluator(bad)
        cases = [
            EvalCase("Hello!", "text", "", ()),
            EvalCase("Fix bug", "tool_call", "", ()),
        ]
        summary = ev.run(cases)
        assert summary.accuracy == 0.5
        assert summary.meta_commentary_leaks == 2

    def test_error_handling(self) -> None:
        def broken(_prompt: str) -> tuple[str, str]:
            raise RuntimeError("boom")

        ev = PromptEvaluator(broken)
        cases = [EvalCase("Hello!", "text", "", ())]
        summary = ev.run(cases)
        assert summary.total == 1
        assert summary.correct == 0
        assert ev.results[0].error == "boom"


# ---------------------------------------------------------------------------
# Default cases
# ---------------------------------------------------------------------------


def test_default_cases_have_text_and_tool_call() -> None:
    texts = [c for c in DEFAULT_EVAL_CASES if c.expected == "text"]
    tools = [c for c in DEFAULT_EVAL_CASES if c.expected == "tool_call"]
    assert len(texts) > 0
    assert len(tools) > 0


def test_default_cases_have_descriptions() -> None:
    for case in DEFAULT_EVAL_CASES:
        assert case.description, f"Case {case.user_prompt!r} lacks description"


def test_default_cases_have_tags() -> None:
    for case in DEFAULT_EVAL_CASES:
        assert case.tags, f"Case {case.user_prompt!r} lacks tags"


# ---------------------------------------------------------------------------
# EvalDataset: versioning + deterministic splits
# ---------------------------------------------------------------------------


class TestEvalDataset:
    def _dataset(self, n_text: int = 8, n_tool: int = 4) -> EvalDataset:
        cases = [
            EvalCase(f"text prompt {i}", "text", f"t{i}", ("chat",)) for i in range(n_text)
        ] + [EvalCase(f"tool prompt {i}", "tool_call", f"c{i}", ("coding",)) for i in range(n_tool)]
        return EvalDataset(name="unit", cases=tuple(cases))

    def test_auto_version_from_content_hash(self) -> None:
        ds = self._dataset()
        assert ds.version.startswith("auto-")
        assert ds.version == f"auto-{ds.content_hash[:8]}"

    def test_explicit_version_wins(self) -> None:
        ds = EvalDataset(name="unit", cases=(EvalCase("x", "text", "", ()),), version="v7")
        assert ds.version == "v7"

    def test_content_hash_changes_with_cases(self) -> None:
        a = self._dataset(n_text=8)
        b = self._dataset(n_text=9)
        assert a.content_hash != b.content_hash

    def test_split_is_deterministic(self) -> None:
        ds = self._dataset()
        train1, test1 = ds.stratified_split(seed=39)
        train2, test2 = ds.stratified_split(seed=39)
        assert train1 == train2
        assert test1 == test2

    def test_split_is_stratified(self) -> None:
        ds = self._dataset(n_text=10, n_tool=5)
        train, test = ds.stratified_split(test_ratio=0.2, seed=39)
        assert all(c.expected == "text" for c in train if c.expected == "text")
        assert sum(1 for c in test if c.expected == "text") >= 1
        assert sum(1 for c in test if c.expected == "tool_call") >= 1

    def test_split_sizes_and_disjointness(self) -> None:
        ds = self._dataset(n_text=10, n_tool=5)
        train, test = ds.stratified_split(test_ratio=0.2, seed=39)
        assert len(train) + len(test) == len(ds.cases)
        assert set(c.user_prompt for c in train).isdisjoint(set(c.user_prompt for c in test))
        assert len(test) == 3  # ceil(10*.2)=2 text + ceil(5*.2)=1 tool

    def test_default_dataset_versioned(self) -> None:
        assert DEFAULT_EVAL_DATASET.cases == tuple(DEFAULT_EVAL_CASES)
        assert DEFAULT_EVAL_DATASET.version.startswith("auto-")
