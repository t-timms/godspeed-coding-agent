"""Tests for evolution orchestrator and memory wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.agent.system_prompt import build_system_prompt
from godspeed.evolution.orchestrator import EvolutionOrchestrator
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

# -- Helpers ------------------------------------------------------------------


class _FakeTool(Tool):
    """Minimal tool stub for testing."""

    def __init__(self, name: str, risk: RiskLevel) -> None:
        self._name = name
        self._risk = risk

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk

    def get_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult.ok(f"{self._name} executed")


# -- Tests: EvolutionOrchestrator ---------------------------------------------


class TestEvolutionOrchestratorCanRun:
    """Tests for the _can_run scheduling logic."""

    def test_first_run_always_allowed(self, tmp_path: Path) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            min_sessions_between_runs=10,
            evo_dir=tmp_path,
        )
        can_run, reason = orch._can_run()
        assert can_run is True
        assert reason == "first run"

    def test_subsequent_run_blocked_until_threshold(self, tmp_path: Path) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            min_sessions_between_runs=10,
            evo_dir=tmp_path,
        )
        # Simulate 5 mutations already done
        with patch.object(orch._registry, "stats", return_value={"total_mutations": 5}):
            can_run, reason = orch._can_run()
            assert can_run is False
            assert "wait 5 more sessions" in reason

    def test_subsequent_run_allowed_after_threshold(self, tmp_path: Path) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            min_sessions_between_runs=10,
            evo_dir=tmp_path,
        )
        with patch.object(orch._registry, "stats", return_value={"total_mutations": 15}):
            can_run, reason = orch._can_run()
            assert can_run is True
            assert reason == "threshold met"


class TestEvolutionOrchestratorRunCycle:
    """Tests for run_cycle skip paths."""

    @pytest.mark.asyncio
    async def test_skipped_when_cant_run(self, tmp_path: Path) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            min_sessions_between_runs=10,
            evo_dir=tmp_path,
        )
        with patch.object(orch._registry, "stats", return_value={"total_mutations": 5}):
            report = await orch.run_cycle()
            assert report["skipped"] is True
            assert "wait" in report["reason"]

    @pytest.mark.asyncio
    async def test_skipped_when_cost_budget_exceeded(self, tmp_path: Path) -> None:
        llm_client = MagicMock()
        llm_client.total_cost_usd = 0.75
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            llm_client=llm_client,
            evo_dir=tmp_path,
        )
        report = await orch.run_cycle()
        assert report["skipped"] is True
        assert "cost budget exceeded" in report["reason"]

    @pytest.mark.asyncio
    async def test_no_sessions_returns_empty_report(self, tmp_path: Path) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )
        # Empty audit dir
        report = await orch.run_cycle()
        assert report["skipped"] is False
        assert report["reason"] == "no sessions in audit trail"


class TestEvolutionOrchestratorEdgeCases:
    """Edge case tests for run_cycle."""

    @pytest.mark.asyncio
    async def test_analyze_exception_returns_error(self, tmp_path: Path) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = []
        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )
        with patch.object(orch._analyzer, "load_sessions", side_effect=RuntimeError("disk full")):
            report = await orch.run_cycle()
        assert report["skipped"] is False
        assert "analyze: disk full" in report["errors"]

    @pytest.mark.asyncio
    async def test_full_cycle_with_mocks(self, tmp_path: Path) -> None:
        from godspeed.evolution.fitness import FitnessScore
        from godspeed.evolution.mutator import MutationCandidate
        from godspeed.evolution.safety import SafetyVerdict
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="file_read",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )
        fake_candidate = MutationCandidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="old",
            mutated="new",
            mutation_rationale="fix",
            model_used="test",
        )
        fake_score = FitnessScore(
            correctness=0.8,
            procedure_following=0.7,
            conciseness=0.9,
            overall=0.8,
            length_penalty=0.0,
            confidence=0.6,
        )
        fake_verdict = SafetyVerdict(
            passed=True,
            checks=(("size", True, "ok"),),
            requires_human_review=False,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    return_value=[fake_candidate],
                ):
                    with patch(
                        "godspeed.evolution.orchestrator.FitnessEvaluator.evaluate",
                        return_value=fake_score,
                    ):
                        with patch.object(
                            orch._safety, "gate", new_callable=AsyncMock, return_value=fake_verdict
                        ):
                            with patch.object(orch._registry, "register", return_value="rec-123"):
                                with patch.object(orch._applier, "apply_tool_description"):
                                    with patch.object(registry, "update_description"):
                                        report = await orch.run_cycle()

        assert report["skipped"] is False
        assert report["mutations_generated"] == 1
        assert report["mutations_applied"] == 1
        assert report["mutations_rejected"] == 0

    @pytest.mark.asyncio
    async def test_skips_security_sensitive_tools(self, tmp_path: Path) -> None:
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("shell", RiskLevel.DESTRUCTIVE)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("shell", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("shell", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="shell",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                report = await orch.run_cycle()

        assert report["mutations_generated"] == 0
        assert report["mutations_applied"] == 0
        assert report["mutations_rejected"] == 1

    @pytest.mark.asyncio
    async def test_empty_candidates_skips_tool(self, tmp_path: Path) -> None:
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="file_read",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    return_value=[],
                ):
                    report = await orch.run_cycle()

        assert report["mutations_generated"] == 0
        assert report["mutations_applied"] == 0
        assert report["mutations_rejected"] == 0

    @pytest.mark.asyncio
    async def test_safety_rejected_counts_correctly(self, tmp_path: Path) -> None:
        from godspeed.evolution.fitness import FitnessScore
        from godspeed.evolution.mutator import MutationCandidate
        from godspeed.evolution.safety import SafetyVerdict
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="file_read",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )
        fake_candidate = MutationCandidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="old",
            mutated="new",
            mutation_rationale="fix",
            model_used="test",
        )
        fake_score = FitnessScore(
            correctness=0.8,
            procedure_following=0.7,
            conciseness=0.9,
            overall=0.8,
            length_penalty=0.0,
            confidence=0.6,
        )
        fake_verdict = SafetyVerdict(
            passed=False,
            checks=(("size", False, "too big"),),
            requires_human_review=False,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    return_value=[fake_candidate],
                ):
                    with patch(
                        "godspeed.evolution.orchestrator.FitnessEvaluator.evaluate",
                        return_value=fake_score,
                    ):
                        with patch.object(
                            orch._safety, "gate", new_callable=AsyncMock, return_value=fake_verdict
                        ):
                            report = await orch.run_cycle()

        assert report["mutations_generated"] == 1
        assert report["mutations_applied"] == 0
        assert report["mutations_rejected"] == 1

    @pytest.mark.asyncio
    async def test_requires_human_review_counts_neither(self, tmp_path: Path) -> None:
        from godspeed.evolution.fitness import FitnessScore
        from godspeed.evolution.mutator import MutationCandidate
        from godspeed.evolution.safety import SafetyVerdict
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="file_read",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )
        fake_candidate = MutationCandidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="old",
            mutated="new",
            mutation_rationale="fix",
            model_used="test",
        )
        fake_score = FitnessScore(
            correctness=0.8,
            procedure_following=0.7,
            conciseness=0.9,
            overall=0.8,
            length_penalty=0.0,
            confidence=0.6,
        )
        fake_verdict = SafetyVerdict(
            passed=True,
            checks=(("size", True, "ok"),),
            requires_human_review=True,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    return_value=[fake_candidate],
                ):
                    with patch(
                        "godspeed.evolution.orchestrator.FitnessEvaluator.evaluate",
                        return_value=fake_score,
                    ):
                        with patch.object(
                            orch._safety, "gate", new_callable=AsyncMock, return_value=fake_verdict
                        ):
                            report = await orch.run_cycle()

        assert report["mutations_generated"] == 1
        assert report["mutations_applied"] == 0
        assert report["mutations_rejected"] == 0

    @pytest.mark.asyncio
    async def test_missing_tool_skips_failure(self, tmp_path: Path) -> None:
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        registry.get.return_value = None
        registry.list_tools.return_value = []

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("unknown_tool", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("unknown_tool", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="unknown_tool",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                report = await orch.run_cycle()

        assert report["mutations_generated"] == 0
        assert report["mutations_applied"] == 0

    @pytest.mark.asyncio
    async def test_max_mutations_caps_loop(self, tmp_path: Path) -> None:
        from godspeed.evolution.fitness import FitnessScore
        from godspeed.evolution.mutator import MutationCandidate
        from godspeed.evolution.safety import SafetyVerdict
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            max_mutations=2,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        failures = tuple(
            ToolFailurePattern(
                tool_name="file_read",
                error_category="invalid_args",
                frequency=i,
                example_args=({},),
                suggested_fix="n/a",
            )
            for i in range(5)
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=failures,
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )
        fake_candidate = MutationCandidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="old",
            mutated="new",
            mutation_rationale="fix",
            model_used="test",
        )
        fake_score = FitnessScore(
            correctness=0.8,
            procedure_following=0.7,
            conciseness=0.9,
            overall=0.8,
            length_penalty=0.0,
            confidence=0.6,
        )
        fake_verdict = SafetyVerdict(
            passed=True,
            checks=(("size", True, "ok"),),
            requires_human_review=False,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    return_value=[fake_candidate],
                ):
                    with patch(
                        "godspeed.evolution.orchestrator.FitnessEvaluator.evaluate",
                        return_value=fake_score,
                    ):
                        with patch.object(
                            orch._safety, "gate", new_callable=AsyncMock, return_value=fake_verdict
                        ):
                            with patch.object(orch._registry, "register", return_value="rec-123"):
                                with patch.object(orch._applier, "apply_tool_description"):
                                    with patch.object(registry, "update_description"):
                                        report = await orch.run_cycle()

        assert report["mutations_generated"] == 2
        assert report["mutations_applied"] == 2
        assert report["mutations_rejected"] == 0

    @pytest.mark.asyncio
    async def test_mutate_exception_logs_error(self, tmp_path: Path) -> None:
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="file_read",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    side_effect=RuntimeError("llm down"),
                ):
                    report = await orch.run_cycle()

        assert "mutate file_read: llm down" in report["errors"]

    @pytest.mark.asyncio
    async def test_test_suite_failure_blocks_mutation(self, tmp_path: Path) -> None:
        from godspeed.evolution.fitness import FitnessScore
        from godspeed.evolution.mutator import MutationCandidate
        from godspeed.evolution.safety import SafetyVerdict
        from godspeed.evolution.trace_analyzer import (
            EvolutionReport,
            SessionTrace,
            ToolCall,
            ToolFailurePattern,
        )

        registry = MagicMock()
        fake_tool = _FakeTool("file_read", RiskLevel.READ_ONLY)
        registry.get.return_value = fake_tool
        registry.list_tools.return_value = [fake_tool]

        orch = EvolutionOrchestrator(
            tool_registry=registry,
            audit_dir=tmp_path,
            max_cost_usd=0.50,
            evo_dir=tmp_path,
        )

        fake_trace = SessionTrace(
            session_id="s1",
            tool_calls=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            errors=(ToolCall("file_read", {}, 10, True, 100.0, "error"),),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=100.0,
            model="test",
        )
        fake_failure = ToolFailurePattern(
            tool_name="file_read",
            error_category="invalid_args",
            frequency=3,
            example_args=({},),
            suggested_fix="n/a",
        )
        fake_report = EvolutionReport(
            sessions_analyzed=1,
            tool_failures=(fake_failure,),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.5,
        )
        fake_candidate = MutationCandidate(
            artifact_type="tool_description",
            artifact_id="file_read",
            original="old",
            mutated="new",
            mutation_rationale="fix",
            model_used="test",
        )
        fake_score = FitnessScore(
            correctness=0.8,
            procedure_following=0.7,
            conciseness=0.9,
            overall=0.8,
            length_penalty=0.0,
            confidence=0.6,
        )
        failing_verdict = SafetyVerdict(
            passed=False,
            checks=(
                ("size_limit", True, "ok"),
                ("semantic_drift", True, "ok"),
                ("fitness_threshold", True, "ok"),
                ("confidence", True, "ok"),
                ("test_suite", False, "pytest exit_code=1"),
            ),
            requires_human_review=False,
        )

        with patch.object(orch._analyzer, "load_sessions", return_value=[fake_trace]):
            with patch.object(orch._analyzer, "generate_report", return_value=fake_report):
                with patch(
                    "godspeed.evolution.orchestrator.EvolutionEngine.mutate_tool_description",
                    return_value=[fake_candidate],
                ):
                    with patch(
                        "godspeed.evolution.orchestrator.FitnessEvaluator.evaluate",
                        return_value=fake_score,
                    ):
                        with patch.object(
                            orch._safety,
                            "gate",
                            new_callable=AsyncMock,
                            return_value=failing_verdict,
                        ):
                            with patch.object(orch._registry, "register", return_value="rec-456"):
                                report = await orch.run_cycle()

        assert report["mutations_generated"] == 1
        assert report["mutations_applied"] == 0
        assert report["mutations_rejected"] == 1


# -- Tests: build_system_prompt with memory -----------------------------------


class TestBuildSystemPromptMemory:
    """Tests for memory hint injection into system prompt."""

    def test_memory_hints_appended_when_present(self) -> None:
        tools = [_FakeTool("file_read", RiskLevel.READ_ONLY)]
        prompt = build_system_prompt(
            tools=tools,
            memory_hints="User prefers: snake_case over camelCase",
        )
        assert "## Memory" in prompt
        assert "User prefers: snake_case over camelCase" in prompt

    def test_memory_hints_omitted_when_none(self) -> None:
        tools = [_FakeTool("file_read", RiskLevel.READ_ONLY)]
        prompt = build_system_prompt(tools=tools, memory_hints=None)
        assert "## Memory" not in prompt

    def test_memory_hints_omitted_when_empty(self) -> None:
        tools = [_FakeTool("file_read", RiskLevel.READ_ONLY)]
        prompt = build_system_prompt(tools=tools, memory_hints="")
        assert "## Memory" not in prompt


# -- Tests: run_external_tool helper ------------------------------------------


class TestRunExternalTool:
    """Tests for the run_external_tool helper in tools/base.py."""

    def test_missing_binary_returns_failure(self, tmp_path: Path) -> None:
        from godspeed.tools.base import run_external_tool

        result = run_external_tool(
            ["nonexistent_binary_12345"],
            cwd=tmp_path,
            check_binary="nonexistent_binary_12345",
        )
        assert result.is_error is True
        assert "not installed" in result.error.lower()

    def test_successful_command(self, tmp_path: Path) -> None:
        from godspeed.tools.base import run_external_tool

        result = run_external_tool(
            ["python", "-c", "print('hello')"],
            cwd=tmp_path,
        )
        assert result.is_error is False
        assert "hello" in result.output

    def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        from godspeed.tools.base import run_external_tool

        result = run_external_tool(
            ["python", "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            timeout=1,
        )
        assert result.is_error is True
        assert "timed out" in result.error.lower()

    def test_output_truncation(self, tmp_path: Path) -> None:
        from godspeed.tools.base import run_external_tool

        result = run_external_tool(
            ["python", "-c", "print('x' * 10000)"],
            cwd=tmp_path,
            max_output_chars=100,
        )
        assert result.is_error is False
        assert len(result.output) <= 103  # 100 + "..."
        assert result.output.endswith("...")

    def test_failing_command_returns_error(self, tmp_path: Path) -> None:
        from godspeed.tools.base import run_external_tool

        result = run_external_tool(
            ["python", "-c", "import sys; sys.exit(1)"],
            cwd=tmp_path,
        )
        assert result.is_error is True
        assert result.output == ""

    def test_command_not_found(self, tmp_path: Path) -> None:
        from godspeed.tools.base import run_external_tool

        result = run_external_tool(
            ["nonexistent_cmd_xyz"],
            cwd=tmp_path,
        )
        assert result.is_error is True
        assert "Command not found" in result.error


class TestToolCallFormatForPermission:
    """Tests for ToolCall.format_for_permission coverage."""

    def test_git_action_argument(self) -> None:
        from godspeed.tools.base import ToolCall

        tc = ToolCall(tool_name="git", arguments={"action": "status"})
        assert tc.format_for_permission == "git(status)"

    def test_empty_dict_arguments(self) -> None:
        from godspeed.tools.base import ToolCall

        tc = ToolCall(tool_name="repo_map", arguments={})
        assert tc.format_for_permission == "repo_map()"

    def test_no_string_values_fallback(self) -> None:
        from godspeed.tools.base import ToolCall

        tc = ToolCall(tool_name="unknown", arguments={"count": 42})
        assert tc.format_for_permission == "unknown(*)"


class TestToolResult:
    """Tests for ToolResult helper methods."""

    def test_failure_factory(self) -> None:
        from godspeed.tools.base import ToolResult

        result = ToolResult.failure("something broke")
        assert result.is_error is True
        assert result.error == "something broke"
        assert result.output == ""

    def test_ok_factory(self) -> None:
        from godspeed.tools.base import ToolResult

        result = ToolResult.ok("output text")
        assert result.is_error is False
        assert result.output == "output text"

    def test_success_alias(self) -> None:
        from godspeed.tools.base import ToolResult

        result = ToolResult.success("output text")
        assert result.is_error is False
        assert result.output == "output text"


class TestCorrectCommand:
    """Tests for /correct TUI command."""

    def test_correct_records_correction(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from godspeed.tui.commands import Commands

        cmds = Commands(
            conversation=MagicMock(),
            llm_client=MagicMock(),
            permission_engine=MagicMock(),
            audit_trail=None,
            session_id="test",
            cwd=tmp_path,
            tool_registry=None,
        )
        result = cmds.dispatch("/correct always use Path objects")
        assert result is not None
        assert result.handled is True

    def test_correct_no_args_shows_error(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from godspeed.tui.commands import Commands

        cmds = Commands(
            conversation=MagicMock(),
            llm_client=MagicMock(),
            permission_engine=MagicMock(),
            audit_trail=None,
            session_id="test",
            cwd=tmp_path,
            tool_registry=None,
        )
        result = cmds.dispatch("/correct")
        assert result is not None
        assert result.handled is True


class TestPreferencesCommand:
    """Tests for /preferences TUI command."""

    def test_preferences_shows_empty_state(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from godspeed.tui.commands import Commands

        cmds = Commands(
            conversation=MagicMock(),
            llm_client=MagicMock(),
            permission_engine=MagicMock(),
            audit_trail=None,
            session_id="test",
            cwd=tmp_path,
            tool_registry=None,
        )
        result = cmds.dispatch("/preferences")
        assert result is not None
        assert result.handled is True
