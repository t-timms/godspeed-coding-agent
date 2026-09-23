"""Tests for the Laya advisory layer — must never change a decision's .action."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from godspeed.config import LayaSettings
from godspeed.security.laya_advisor import (
    LayaAdvisor,
    LayaAdvisory,
    annotate_ask_decision,
    extract_answer,
)
from godspeed.security.permissions import ALLOW, ASK, DENY, PermissionDecision
from godspeed.tools.base import ToolCall


def _ask_decision(reason: str = "matches ask rule: shell(*)") -> PermissionDecision:
    return PermissionDecision(ASK, reason)


def _shell_call(command: str = "rm somefile") -> ToolCall:
    return ToolCall(tool_name="shell", arguments={"command": command})


@pytest.fixture
def advisor() -> Iterator[LayaAdvisor]:
    """A LayaAdvisor with its ThreadPoolExecutor shut down on teardown.

    LayaAdvisor.__init__ creates a live single-worker executor
    (laya_advisor.py); constructing one directly (bypassing the mocked
    ``.get()`` singleton used elsewhere in this file) leaks that pool for
    the rest of the test session unless something shuts it down.
    """
    instance = LayaAdvisor()
    yield instance
    instance._executor.shutdown(wait=False)


class TestAnnotateAskDecisionGuards:
    """Every case where annotation must NOT happen — same object back, untouched."""

    def test_disabled_returns_same_object(self) -> None:
        decision = _ask_decision()
        settings = LayaSettings(enabled=False)
        result = annotate_ask_decision(decision, _shell_call(), settings)
        assert result is decision

    def test_non_ask_decision_untouched(self) -> None:
        for base in (PermissionDecision(ALLOW, "r"), PermissionDecision(DENY, "r")):
            result = annotate_ask_decision(base, _shell_call(), LayaSettings(enabled=True))
            assert result is base

    def test_non_shell_tool_untouched(self) -> None:
        decision = _ask_decision()
        call = ToolCall(tool_name="file_write", arguments={"file_path": "x.py"})
        result = annotate_ask_decision(decision, call, LayaSettings(enabled=True))
        assert result is decision

    def test_laya_not_installed_untouched(self) -> None:
        decision = _ask_decision()
        with patch("godspeed.security.laya_advisor._is_laya_available", return_value=False):
            result = annotate_ask_decision(decision, _shell_call(), LayaSettings(enabled=True))
        assert result is decision

    def test_empty_command_untouched(self) -> None:
        decision = _ask_decision()
        call = ToolCall(tool_name="shell", arguments={})
        with patch("godspeed.security.laya_advisor._is_laya_available", return_value=True):
            result = annotate_ask_decision(decision, call, LayaSettings(enabled=True))
        assert result is decision


class TestAnnotateAskDecisionFailsNeutral:
    """Any failure inside the advisor must degrade to 'no annotation', not raise."""

    def test_advisory_none_returns_same_object(self) -> None:
        decision = _ask_decision()
        with (
            patch("godspeed.security.laya_advisor._is_laya_available", return_value=True),
            patch.object(LayaAdvisor, "get") as mock_get,
        ):
            mock_get.return_value.get_advisory.return_value = None
            result = annotate_ask_decision(decision, _shell_call(), LayaSettings(enabled=True))
        assert result is decision


class TestAnnotateAskDecisionAttachesAdvisory:
    """The one success path — action must stay ASK, only reason/metadata change.

    No confidence gating: a validation run against the real checkpoint found
    confidence never approached the old 0.7 default (highest observed 0.64
    across 40 cases), so annotate_ask_decision always attaches the advisory
    when Laya succeeds rather than hard-gating on an unvalidated threshold.
    """

    def test_annotation_never_changes_action(self) -> None:
        decision = _ask_decision("matches ask rule: shell(*)")
        advisory = LayaAdvisory(is_destructive=0.88, raw={"answers": {}})
        with (
            patch("godspeed.security.laya_advisor._is_laya_available", return_value=True),
            patch.object(LayaAdvisor, "get") as mock_get,
        ):
            mock_get.return_value.get_advisory.return_value = advisory
            result = annotate_ask_decision(decision, _shell_call(), LayaSettings(enabled=True))
        assert result.action == ASK
        assert result == decision  # __eq__ compares .action only, still ASK
        assert "destructive" in result.reason
        assert "laya" in result.metadata
        assert result is not decision  # a new annotated object, original untouched
        assert decision.metadata == {}  # original object never mutated


class TestLayaAdvisorTimeout:
    """get_advisory must return None, never hang or raise, on a slow predict()."""

    def test_slow_predict_times_out_to_none(self, advisor: LayaAdvisor) -> None:
        slow_agent = MagicMock()
        slow_agent.predict.side_effect = lambda *_args, **_kwargs: time.sleep(2)
        advisor._agent = slow_agent
        result = advisor.get_advisory("rm -rf /tmp/x", [], timeout_ms=50)
        assert result is None

    def test_predict_exception_returns_none(self, advisor: LayaAdvisor) -> None:
        broken_agent = MagicMock()
        broken_agent.predict.side_effect = RuntimeError("model error")
        advisor._agent = broken_agent
        result = advisor.get_advisory("rm -rf /tmp/x", [], timeout_ms=200)
        assert result is None

    def test_load_failure_returns_none(self, advisor: LayaAdvisor) -> None:
        """Simulates a broken laya install via sys.modules — works whether or
        not the real package is actually installed in this environment."""
        fake_laya = MagicMock()
        fake_laya.load.side_effect = RuntimeError("checkpoint download failed")
        with patch.dict(sys.modules, {"laya": fake_laya}):
            result = advisor.get_advisory("ls", [], timeout_ms=200)
        assert result is None
        assert advisor._agent_load_failed is True


class TestAsk:
    """The generic ask() method — shared by get_advisory and (eventually)
    task-type routing. Same fail-neutral/timeout contract as get_advisory,
    tested once here directly rather than only indirectly through it."""

    def test_returns_raw_predict_result(self, advisor: LayaAdvisor) -> None:
        agent = MagicMock()
        agent.predict.return_value = {"answers": {"foo": {"type": "noul", "noul": 0.5}}}
        advisor._agent = agent
        result = advisor.ask({"request": "hi"}, {"foo": {"type": "noul"}})
        assert result == {"answers": {"foo": {"type": "noul", "noul": 0.5}}}

    def test_non_dict_predict_result_becomes_empty_dict(self, advisor: LayaAdvisor) -> None:
        agent = MagicMock()
        agent.predict.return_value = "not a dict"
        advisor._agent = agent
        result = advisor.ask({"request": "hi"}, {"foo": {"type": "noul"}})
        assert result == {}

    def test_timeout_returns_none(self, advisor: LayaAdvisor) -> None:
        agent = MagicMock()
        agent.predict.side_effect = lambda *_a, **_kw: time.sleep(2)
        advisor._agent = agent
        result = advisor.ask({"request": "hi"}, {"foo": {"type": "noul"}}, timeout_ms=50)
        assert result is None

    def test_agent_unavailable_returns_none(self, advisor: LayaAdvisor) -> None:
        advisor._agent_load_failed = True
        result = advisor.ask({"request": "hi"}, {"foo": {"type": "noul"}})
        assert result is None


class TestGetAdvisoryParsing:
    """Regression guard for get_advisory's response parsing, using response
    shapes matching the real checkpoint's actual output format (verified via
    a live inference call this session) — not the model's real accuracy,
    which needs the real network-downloaded checkpoint and isn't something a
    fast unit test should depend on."""

    def test_high_destructive_probability_parsed_correctly(self, advisor: LayaAdvisor) -> None:
        """Shape matches a real `rm -rf /` response: is_destructive.noul high."""
        agent = MagicMock()
        agent.predict.return_value = {
            "answers": {"is_destructive": {"type": "noul", "noul": 0.87, "confidence": 0.61}}
        }
        advisor._agent = agent
        advisory = advisor.get_advisory("rm -rf /", [])
        assert advisory is not None
        assert advisory.is_destructive == 0.87

    def test_low_destructive_probability_parsed_correctly(self, advisor: LayaAdvisor) -> None:
        """Shape matches a real `ls -la` response: is_destructive.noul low."""
        agent = MagicMock()
        agent.predict.return_value = {
            "answers": {"is_destructive": {"type": "noul", "noul": 0.18, "confidence": 0.17}}
        }
        advisor._agent = agent
        advisory = advisor.get_advisory("ls -la", [])
        assert advisory is not None
        assert advisory.is_destructive == 0.18

    def test_missing_answers_key_defaults_to_zero(self, advisor: LayaAdvisor) -> None:
        agent = MagicMock()
        agent.predict.return_value = {"answers": {}}
        advisor._agent = agent
        advisory = advisor.get_advisory("ls -la", [])
        assert advisory is not None
        assert advisory.is_destructive == 0.0

    def test_null_destructive_probability_does_not_raise(self, advisor: LayaAdvisor) -> None:
        """Regression guard: get_advisory's parsing used to call float()
        unguarded on whatever noul comes back. An explicit null (key
        present, value None — plausible if the model 'declines to answer')
        must degrade to the same safe default as a missing key, not raise
        and break the fail-neutral contract callers rely on."""
        agent = MagicMock()
        agent.predict.return_value = {"answers": {"is_destructive": {"type": "noul", "noul": None}}}
        advisor._agent = agent
        advisory = advisor.get_advisory("rm -rf /", [])
        assert advisory is not None
        assert advisory.is_destructive == 0.0


class TestExtractAnswer:
    """Shared response-parsing helper — used by both get_advisory and
    llm/router.py's task-type routing, so it's tested once here directly."""

    def test_extracts_matching_answer(self) -> None:
        result = {"answers": {"difficulty": {"type": "score", "score": 2.1}}}
        assert extract_answer(result, "difficulty") == {"type": "score", "score": 2.1}

    def test_missing_question_returns_none(self) -> None:
        result = {"answers": {"other": {"type": "score", "score": 1.0}}}
        assert extract_answer(result, "difficulty") is None

    def test_missing_answers_key_returns_none(self) -> None:
        assert extract_answer({}, "difficulty") is None

    def test_non_dict_result_returns_none(self) -> None:
        assert extract_answer(None, "difficulty") is None
        assert extract_answer("not a dict", "difficulty") is None  # type: ignore[arg-type]

    def test_non_dict_answers_returns_none(self) -> None:
        assert extract_answer({"answers": "not a dict"}, "difficulty") is None

    def test_non_dict_answer_value_returns_none(self) -> None:
        assert extract_answer({"answers": {"difficulty": "not a dict"}}, "difficulty") is None
