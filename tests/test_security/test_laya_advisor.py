"""Tests for the Laya advisory layer — must never change a decision's .action."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

from godspeed.config import LayaSettings
from godspeed.security.laya_advisor import (
    LayaAdvisor,
    LayaAdvisory,
    annotate_ask_decision,
)
from godspeed.security.permissions import ALLOW, ASK, DENY, PermissionDecision
from godspeed.tools.base import ToolCall


def _ask_decision(reason: str = "matches ask rule: shell(*)") -> PermissionDecision:
    return PermissionDecision(ASK, reason)


def _shell_call(command: str = "rm somefile") -> ToolCall:
    return ToolCall(tool_name="shell", arguments={"command": command})


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

    def test_below_confidence_threshold_returns_same_object(self) -> None:
        decision = _ask_decision()
        advisory = LayaAdvisory(
            risk_category="moderate",
            risk_category_confidence=0.3,
            is_destructive=0.4,
            raw={},
        )
        with (
            patch("godspeed.security.laya_advisor._is_laya_available", return_value=True),
            patch.object(LayaAdvisor, "get") as mock_get,
        ):
            mock_get.return_value.get_advisory.return_value = advisory
            result = annotate_ask_decision(
                decision, _shell_call(), LayaSettings(enabled=True, confidence_threshold=0.7)
            )
        assert result is decision


class TestAnnotateAskDecisionAttachesAdvisory:
    """The one success path — action must stay ASK, only reason/metadata change."""

    def test_annotation_never_changes_action(self) -> None:
        decision = _ask_decision("matches ask rule: shell(*)")
        advisory = LayaAdvisory(
            risk_category="destructive",
            risk_category_confidence=0.92,
            is_destructive=0.88,
            raw={"answers": {}},
        )
        with (
            patch("godspeed.security.laya_advisor._is_laya_available", return_value=True),
            patch.object(LayaAdvisor, "get") as mock_get,
        ):
            mock_get.return_value.get_advisory.return_value = advisory
            result = annotate_ask_decision(
                decision, _shell_call(), LayaSettings(enabled=True, confidence_threshold=0.7)
            )
        assert result.action == ASK
        assert result == decision  # __eq__ compares .action only, still ASK
        assert "destructive" in result.reason
        assert "laya" in result.metadata
        assert result is not decision  # a new annotated object, original untouched
        assert decision.metadata == {}  # original object never mutated


class TestLayaAdvisorTimeout:
    """get_advisory must return None, never hang or raise, on a slow predict()."""

    def test_slow_predict_times_out_to_none(self) -> None:
        advisor = LayaAdvisor()
        slow_agent = MagicMock()
        slow_agent.predict.side_effect = lambda *_args, **_kwargs: time.sleep(2)
        advisor._agent = slow_agent
        result = advisor.get_advisory("rm -rf /tmp/x", [], timeout_ms=50)
        assert result is None

    def test_predict_exception_returns_none(self) -> None:
        advisor = LayaAdvisor()
        broken_agent = MagicMock()
        broken_agent.predict.side_effect = RuntimeError("model error")
        advisor._agent = broken_agent
        result = advisor.get_advisory("rm -rf /tmp/x", [], timeout_ms=200)
        assert result is None

    def test_load_failure_returns_none(self) -> None:
        """Simulates a broken laya install via sys.modules — works whether or
        not the real package is actually installed in this environment."""
        advisor = LayaAdvisor()
        fake_laya = MagicMock()
        fake_laya.load.side_effect = RuntimeError("checkpoint download failed")
        with patch.dict(sys.modules, {"laya": fake_laya}):
            result = advisor.get_advisory("ls", [], timeout_ms=200)
        assert result is None
        assert advisor._agent_load_failed is True
