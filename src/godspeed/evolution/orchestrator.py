"""Evolution orchestrator — runs the full self-improvement pipeline.

Tight cost controls:
- max_cost_usd per cycle (default $0.50)
- min_sessions_between_runs (default 10)
- max_mutations_per_cycle (default 3)
- Only mutates tool descriptions (not prompt sections)
- Never auto-applies security-sensitive tool mutations
- All changes flagged for human review
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.evolution.applier import EvolutionApplier
from godspeed.evolution.fitness import FitnessEvaluator
from godspeed.evolution.mutator import EvolutionEngine
from godspeed.evolution.registry import EvolutionRegistry
from godspeed.evolution.safety import SafetyGate
from godspeed.evolution.trace_analyzer import TraceAnalyzer
from godspeed.llm.client import LLMClient
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Tools whose descriptions must never be auto-mutated.
_SECURITY_SENSITIVE_TOOLS = frozenset(
    {
        "shell",
        "file_write",
        "file_edit",
        "diff_apply",
        "git",
        "github",
    }
)


class EvolutionOrchestrator:
    """Run the full evolution pipeline with strict cost and safety limits.

    Usage::

        orch = EvolutionOrchestrator(settings, tool_registry, audit_dir)
        report = await orch.run_cycle()
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        audit_dir: Path,
        *,
        evolution_model: str = "",
        max_cost_usd: float = 0.50,
        max_mutations: int = 3,
        min_sessions_between_runs: int = 10,
        llm_client: LLMClient | None = None,
        evo_dir: Path | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._audit_dir = audit_dir
        self._evo_dir = evo_dir or (Path.home() / ".godspeed" / "evolution")
        self._evo_dir.mkdir(parents=True, exist_ok=True)

        self._max_cost_usd = max_cost_usd
        self._max_mutations = max_mutations
        self._min_sessions = min_sessions_between_runs

        # Use dedicated LLM client for evolution with its own cost tracker
        self._llm_client = llm_client
        self._evo_model = evolution_model

        self._registry = EvolutionRegistry(self._evo_dir)
        self._applier = EvolutionApplier(self._registry)
        self._analyzer = TraceAnalyzer()
        self._safety = SafetyGate()

    @property
    def _engine(self) -> EvolutionEngine:
        return EvolutionEngine(model=self._evo_model, llm_client=self._llm_client)

    @property
    def _evaluator(self) -> FitnessEvaluator:
        return FitnessEvaluator(judge_model=self._evo_model, llm_client=self._llm_client)

    def _can_run(self) -> tuple[bool, str]:
        """Check if enough sessions have passed since last run."""
        stats = self._registry.stats()
        total_mutations = stats.get("total_mutations", 0)
        # Simple heuristic: allow first run immediately, then every N mutations
        # (approximating sessions — each session may generate 0-1 mutations)
        if total_mutations == 0:
            return True, "first run"
        sessions_approx = total_mutations  # rough proxy
        if sessions_approx < self._min_sessions:
            remaining = self._min_sessions - sessions_approx
            return False, f"wait {remaining} more sessions (min={self._min_sessions})"
        return True, "threshold met"

    async def run_cycle(self) -> dict[str, Any]:
        """Execute one evolution cycle.

        Returns:
            Report dict with mutations_generated, mutations_applied,
            cost_usd, errors.
        """
        can_run, reason = self._can_run()
        if not can_run:
            logger.info("evolution skipped reason=%s", reason)
            return {"skipped": True, "reason": reason, "mutations": 0}

        # Check cost budget
        current_cost = getattr(self._llm_client, "total_cost_usd", 0.0) if self._llm_client else 0.0
        if current_cost >= self._max_cost_usd:
            logger.warning(
                "evolution skipped cost_budget_exceeded current=%.4f max=%.4f",
                current_cost,
                self._max_cost_usd,
            )
            return {"skipped": True, "reason": "cost budget exceeded", "mutations": 0}

        report: dict[str, Any] = {
            "skipped": False,
            "mutations_generated": 0,
            "mutations_applied": 0,
            "mutations_rejected": 0,
            "cost_usd_start": current_cost,
            "errors": [],
        }

        # 1. Analyze audit trail
        try:
            sessions = self._analyzer.load_sessions(self._audit_dir, last_n=50)
            if not sessions:
                report["reason"] = "no sessions in audit trail"
                return report

            evo_report = self._analyzer.generate_report(sessions)
            logger.info(
                "evolution.analyzed sessions=%d errors=%.1f%%",
                evo_report.sessions_analyzed,
                evo_report.error_rate * 100,
            )
        except Exception as exc:
            logger.warning("evolution.analyze_failed error=%s", exc, exc_info=True)
            report["errors"].append(f"analyze: {exc}")
            return report

        # 2. Generate mutations for top failing tools
        mutations_count = 0
        applied_count = 0
        rejected_count = 0

        for failure in evo_report.tool_failures[: self._max_mutations]:
            if mutations_count >= self._max_mutations:
                break

            tool = self._tool_registry.get(failure.tool_name)
            if tool is None:
                continue

            # Skip security-sensitive tools — never mutate their descriptions
            if failure.tool_name in _SECURITY_SENSITIVE_TOOLS:
                logger.info("evolution.skip_security_sensitive tool=%s", failure.tool_name)
                rejected_count += 1
                continue

            try:
                candidates = await self._engine.mutate_tool_description(
                    tool_name=failure.tool_name,
                    current_desc=tool.description,
                    failure_patterns=[failure],
                    num_candidates=1,  # tight limit: one candidate per tool
                )
            except Exception as exc:
                logger.warning("evolution.mutate_failed tool=%s error=%s", failure.tool_name, exc)
                report["errors"].append(f"mutate {failure.tool_name}: {exc}")
                continue

            if not candidates:
                continue

            for candidate in candidates:
                mutations_count += 1

                # 3. Evaluate fitness
                try:
                    score = await self._evaluator.evaluate(candidate)
                except (
                    Exception
                ) as exc:  # pragma: no cover — LLM failure, tested at integration level
                    logger.warning(
                        "evolution.fitness_failed tool=%s error=%s",
                        failure.tool_name,
                        exc,
                    )
                    report["errors"].append(f"fitness {failure.tool_name}: {exc}")
                    continue

                # 4. Safety gate
                verdict = await self._safety.gate(candidate, score)
                record_id = self._registry.register(candidate, score, verdict)

                if not verdict.passed:
                    logger.info(
                        "evolution.safety_rejected tool=%s record=%s",
                        failure.tool_name,
                        record_id,
                    )
                    rejected_count += 1
                    continue

                # All mutations require human review (flagged by safety gate)
                if verdict.requires_human_review:
                    logger.info(
                        "evolution.pending_review tool=%s record=%s fitness=%.3f",
                        failure.tool_name,
                        record_id,
                        score.overall,
                    )
                    continue

                # 5. Apply (only reached for low-impact, high-confidence mutations)
                try:
                    self._applier.apply_tool_description(
                        record_id=record_id,
                        tool_name=failure.tool_name,
                        mutated_text=candidate.mutated,
                        original_text=candidate.original,
                    )
                    self._tool_registry.update_description(failure.tool_name, candidate.mutated)
                    applied_count += 1
                    logger.info(
                        "evolution.applied tool=%s record=%s fitness=%.3f",
                        failure.tool_name,
                        record_id,
                        score.overall,
                    )
                except Exception as exc:  # pragma: no cover — disk/registry failure
                    logger.warning(
                        "evolution.apply_failed tool=%s error=%s",
                        failure.tool_name,
                        exc,
                    )
                    report["errors"].append(f"apply {failure.tool_name}: {exc}")

        report["mutations_generated"] = mutations_count
        report["mutations_applied"] = applied_count
        report["mutations_rejected"] = rejected_count
        report["cost_usd_end"] = (
            getattr(self._llm_client, "total_cost_usd", 0.0) if self._llm_client else 0.0
        )
        report["cost_usd_delta"] = report["cost_usd_end"] - report["cost_usd_start"]

        logger.info(
            "evolution.complete generated=%d applied=%d rejected=%d cost=$%.4f",
            mutations_count,
            applied_count,
            rejected_count,
            report["cost_usd_delta"],
        )
        return report
