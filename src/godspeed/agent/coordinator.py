"""Sub-agent coordinator — spawn isolated agent loops for parallel sub-tasks."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.agent.retrieval_agent import RETRIEVAL_SYSTEM_PROMPT
from godspeed.llm.client import LLMClient
from godspeed.llm.usage_ledger import UsageLedger, subagent_context
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_SUB_AGENT_DEPTH = 3
SUB_AGENT_ITERATION_LIMIT = 25
MAX_PARALLEL_AGENTS = 5
_SPAWN_ID_LENGTH = 8


class _SidechainAudit:
    """Audit recorder proxy tagging every record as a sub-agent sidechain.

    Wraps the parent context's recorder and fills the existing (previously
    always-empty) AuditRecord fields ``is_sidechain`` and
    ``parent_session_id`` without widening the audit wire format.
    """

    def __init__(self, inner: Any, parent_session_id: str | None) -> None:
        self._inner = inner
        self._parent_session_id = parent_session_id

    def record(
        self,
        event_type: Any,
        detail: dict[str, Any] | None = None,
        outcome: str = "success",
        **extra: Any,
    ) -> Any:
        extra.setdefault("is_sidechain", True)
        extra.setdefault("parent_session_id", self._parent_session_id)
        return self._inner.record(event_type, detail=detail, outcome=outcome, **extra)

    async def arecord(
        self,
        event_type: Any,
        detail: dict[str, Any] | None = None,
        outcome: str = "success",
        **extra: Any,
    ) -> Any:
        extra.setdefault("is_sidechain", True)
        extra.setdefault("parent_session_id", self._parent_session_id)
        return await self._inner.arecord(event_type, detail=detail, outcome=outcome, **extra)


SUB_AGENT_SYSTEM_PROMPT = """\
You are a sub-agent of Godspeed, a security-first coding agent. You have been \
spawned to handle a specific sub-task. Complete the task and return a concise \
summary of what you accomplished.

## Guidelines
- Focus on the specific task assigned to you
- Use tools efficiently — minimize unnecessary reads
- Return a clear summary when done
- Do not spawn further sub-agents unless absolutely necessary
"""


class CapabilityBundle(StrEnum):
    CORE = "core"
    READONLY = "readonly"
    WRITE = "write"
    FULL = "full"


_BUNDLE_ALLOWED: dict[CapabilityBundle, set[str]] = {
    CapabilityBundle.READONLY: {
        "file_read",
        "directory_list",
        "grep_search",
        "web_fetch",
        "repo_map",
        "retrieval",
    },
    CapabilityBundle.CORE: {
        "file_read",
        "file_write",
        "file_edit",
        "directory_list",
        "grep_search",
        "shell",
    },
    CapabilityBundle.WRITE: {
        "file_read",
        "file_write",
        "file_edit",
        "directory_list",
        "grep_search",
        "shell",
        "git_diff",
        "git_log",
        "git_status",
    },
    CapabilityBundle.FULL: set(),
}


_EFFORT_ITERATIONS: dict[str, int] = {
    "low": 10,
    "normal": 25,
    "high": 40,
}


class CycleDetectedError(Exception):
    """Raised when a dependency cycle is found in a Kanban plan."""


@dataclass(frozen=True)
class SubAgentConfig:
    """Per-subagent overrides for model, effort, and tool access."""

    model: str | None = None
    effort: str = "normal"
    max_iterations: int = 0
    tool_bundle: CapabilityBundle | None = None
    system_prompt: str | None = None
    max_cost_usd: float | None = None

    @property
    def iteration_limit(self) -> int:
        if self.max_iterations > 0:
            return self.max_iterations
        return _EFFORT_ITERATIONS.get(self.effort, SUB_AGENT_ITERATION_LIMIT)


class AgentCoordinator:
    """Coordinates sub-agent spawning with depth limiting and isolation."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        max_depth: int = MAX_SUB_AGENT_DEPTH,
        iteration_limit: int = SUB_AGENT_ITERATION_LIMIT,
        max_parallel: int = MAX_PARALLEL_AGENTS,
        max_spawns: int = 200,
        stagger_seconds: float = 0.25,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._max_depth = max_depth
        self._iteration_limit = iteration_limit
        self._current_depth = 0
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._total_sub_agent_cost: float = 0.0
        self._max_spawns = max_spawns
        self._spawn_count = 0
        self._stagger_seconds = stagger_seconds

    @property
    def total_sub_agent_cost(self) -> float:
        return self._total_sub_agent_cost

    def _build_llm_client_for_config(self, config: SubAgentConfig) -> LLMClient:
        parent = self._llm_client
        child_max_cost = (
            config.max_cost_usd if config.max_cost_usd is not None else parent.max_cost_usd
        )
        if config.model is None:
            if child_max_cost == parent.max_cost_usd:
                return parent
            return LLMClient(
                model=parent.model,
                fallback_models=list(parent.fallback_models),
                timeout=parent.timeout,
                router=parent.router,
                thinking_budget=parent.thinking_budget,
                max_cost_usd=child_max_cost,
                reasoning_effort=parent.reasoning_effort,
            )

        return LLMClient(
            model=config.model,
            fallback_models=list(parent.fallback_models),
            timeout=parent.timeout,
            router=parent.router,
            thinking_budget=parent.thinking_budget,
            max_cost_usd=child_max_cost,
            reasoning_effort=parent.reasoning_effort,
        )

    def _filter_registry_for_bundle(self, bundle: CapabilityBundle | None) -> ToolRegistry:
        if bundle is None or bundle == CapabilityBundle.FULL:
            return self._tool_registry
        allowed = _BUNDLE_ALLOWED.get(bundle)
        if not allowed:
            return self._tool_registry
        filtered = ToolRegistry(sandbox=self._tool_registry._sandbox)
        for tool in self._tool_registry.list_tools():
            if tool.name in allowed:
                filtered.register(tool)
        return filtered

    async def spawn(
        self,
        task: str,
        depth: int = 0,
        config: SubAgentConfig | None = None,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Spawn an isolated sub-agent to handle a task.

        Args:
            task: The sub-task description.
            depth: Current nesting depth (0 = top-level spawn).
            config: Optional per-subagent model/effort overrides.
            tool_context: Optional execution context override (e.g. a
                worktree-scoped cwd for /batch units). Defaults to the
                coordinator's shared context.

        Returns:
            The sub-agent's final text response.
        """
        if depth >= self._max_depth:
            return (
                f"Error: Maximum sub-agent depth ({self._max_depth}) reached. "
                f"Cannot spawn further sub-agents."
            )

        self._spawn_count += 1
        if self._spawn_count > self._max_spawns:
            return (
                f"Error: Session spawn cap ({self._max_spawns}) reached. "
                f"Cannot spawn more sub-agents."
            )

        effective_config = config or SubAgentConfig(max_iterations=self._iteration_limit)
        llm_client = self._build_llm_client_for_config(effective_config)
        iteration_limit = effective_config.iteration_limit
        system_prompt = effective_config.system_prompt or SUB_AGENT_SYSTEM_PROMPT
        registry = self._filter_registry_for_bundle(effective_config.tool_bundle)
        effective_tool_context = tool_context or self._tool_context

        spawn_id = uuid.uuid4().hex[:_SPAWN_ID_LENGTH]
        is_shared_client = llm_client is self._llm_client
        if not is_shared_client:
            llm_client.usage_ledger = UsageLedger(default_subagent_id=spawn_id)
        if effective_tool_context.audit is not None:
            effective_tool_context = effective_tool_context.model_copy(
                update={
                    "audit": _SidechainAudit(
                        effective_tool_context.audit, effective_tool_context.session_id
                    )
                }
            )
        attribution_scope = subagent_context(spawn_id) if is_shared_client else nullcontext()

        logger.info(
            "Spawning sub-agent depth=%d task=%r model=%s",
            depth,
            task[:100],
            llm_client.model,
        )

        conversation = Conversation(
            system_prompt=system_prompt,
            model=llm_client.model,
            max_tokens=getattr(llm_client, "_max_tokens", 100_000),
        )

        try:
            async with self._semaphore:
                pre_cost = llm_client.total_cost_usd
                with attribution_scope:
                    result = await agent_loop(
                        user_input=task,
                        conversation=conversation,
                        llm_client=llm_client,
                        tool_registry=registry,
                        tool_context=effective_tool_context,
                        max_iterations=iteration_limit,
                    )
                cost_delta = llm_client.total_cost_usd - pre_cost
                self._total_sub_agent_cost += cost_delta
            logger.info("Sub-agent completed depth=%d result_len=%d", depth, len(result))
            return result
        except Exception as exc:
            logger.error("Sub-agent failed depth=%d error=%s", depth, exc, exc_info=True)
            return f"Sub-agent error: {exc}"
        finally:
            if not is_shared_client:
                self._llm_client.usage_ledger.merge_from(llm_client.usage_ledger)

    async def spawn_parallel(
        self,
        tasks: list[str],
        depth: int = 0,
        configs: list[SubAgentConfig] | None = None,
    ) -> list[str]:
        """Spawn multiple sub-agents in parallel with concurrency cap.

        Args:
            tasks: List of sub-task descriptions.
            depth: Current nesting depth.
            configs: Optional per-task config overrides.

        Returns:
            List of results (one per task, in order).
        """
        if depth >= self._max_depth:
            return [f"Error: Maximum sub-agent depth ({self._max_depth}) reached."] * len(tasks)

        logger.info("Spawning %d parallel sub-agents depth=%d", len(tasks), depth)

        cfg_list = configs or [None] * len(tasks)

        async def _staggered(idx: int, task_str: str, cfg: SubAgentConfig | None) -> str:
            if idx and self._stagger_seconds > 0:
                await asyncio.sleep(idx * self._stagger_seconds)
            return await self.spawn(task_str, depth=depth, config=cfg)

        coros = [_staggered(i, t, c) for i, (t, c) in enumerate(zip(tasks, cfg_list, strict=True))]
        return list(await asyncio.gather(*coros, return_exceptions=False))

    async def spawn_retrieval(
        self,
        query: str,
    ) -> str:
        """Spawn a retrieval sub-agent with read-only tools and structured output."""
        logger.info("Spawning retrieval agent query=%r", query[:100])

        conversation = Conversation(
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            model=self._llm_client.model,
            max_tokens=getattr(self._llm_client, "_max_tokens", 100_000),
        )

        retrieval_id = uuid.uuid4().hex[:_SPAWN_ID_LENGTH]
        try:
            with subagent_context(retrieval_id):
                result = await agent_loop(
                    user_input=query,
                    conversation=conversation,
                    llm_client=self._llm_client,
                    tool_registry=self._tool_registry,
                    tool_context=self._tool_context,
                    max_iterations=25,
                )
            logger.info("Retrieval agent completed result_len=%d", len(result))
            return result
        except Exception as exc:
            logger.error("Retrieval agent failed error=%s", exc, exc_info=True)
            return f"Retrieval error: {exc}"


@dataclass
class WorkItem:
    """A single work item in a Kanban plan."""

    id: str
    description: str
    files: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    acceptance: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class KanbanPlan:
    """An ordered list of work items with dependency tracking."""

    objective: str
    items: list[WorkItem] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return all(item.status == "verified" for item in self.items)


class KanbanSwarm:
    """Orchestrates parallel sub-agents on a Kanban plan."""

    def __init__(
        self,
        coordinator: AgentCoordinator,
        worker_model: str | None = None,
        verifier_model: str | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._worker_model = worker_model
        self._verifier_model = verifier_model

    @staticmethod
    def _topo_sort_items(items: list[WorkItem]) -> list[list[WorkItem]]:
        by_id: dict[str, WorkItem] = {item.id: item for item in items}
        in_degree: dict[str, int] = {item.id: 0 for item in items}
        dependents: dict[str, list[str]] = defaultdict(list)
        for item in items:
            for dep in item.deps:
                if dep not in by_id:
                    raise CycleDetectedError(
                        f"Work item '{item.id}' depends on unknown item '{dep}'"
                    )
                dependents[dep].append(item.id)
                in_degree[item.id] += 1
        queue: deque[str] = deque(k for k, v in in_degree.items() if v == 0)
        levels: list[list[WorkItem]] = []
        while queue:
            level_size = len(queue)
            level: list[WorkItem] = []
            for _ in range(level_size):
                nid = queue.popleft()
                level.append(by_id[nid])
                for dep_id in dependents[nid]:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        queue.append(dep_id)
            levels.append(level)
        visited = sum(len(lv) for lv in levels)
        if visited != len(items):
            cycle_ids = [i.id for i in items if in_degree[i.id] > 0]
            raise CycleDetectedError(f"Dependency cycle detected among: {cycle_ids}")
        return levels

    async def _verify_item(self, item: WorkItem, worker_result: str) -> bool:
        if not self._verifier_model:
            return True
        verifier_client = LLMClient(model=self._verifier_model)
        prompt = (
            f"You are a verification agent. Check if the following work result "
            f"satisfies the acceptance criteria.\n\n"
            f"Work item: {item.description}\n"
            f"Acceptance criteria: {item.acceptance}\n"
            f"Result:\n{worker_result}\n\n"
            f"Reply with PASS if criteria are met, FAIL otherwise. "
            f"Start your response with PASS or FAIL."
        )
        try:
            response = await verifier_client.chat(messages=[{"role": "user", "content": prompt}])
            content = response.content.strip().upper()
            return content.startswith("PASS")
        except Exception:
            logger.warning("Verifier failed for item %s, defaulting to PASS", item.id)
            return True

    async def execute(self, plan: KanbanPlan) -> dict[str, str]:
        if not plan.items:
            return {}
        levels = self._topo_sort_items(plan.items)
        results: dict[str, str] = {}
        for level in levels:
            independent = [
                item for item in level if not item.deps or all(d in results for d in item.deps)
            ]

            async def _run_item(item: WorkItem) -> tuple[str, str]:
                cfg = SubAgentConfig(model=self._worker_model) if self._worker_model else None
                worker_result = await self._coordinator.spawn(
                    f"Work on: {item.description}",
                    depth=1,
                    config=cfg,
                )
                return item.id, worker_result

            task_coros = [_run_item(item) for item in independent]
            worker_results = await asyncio.gather(*task_coros)
            for item_id, worker_result in worker_results:
                results[item_id] = worker_result
            verify_coros = []
            for item in independent:
                vr = results[item.id]
                verify_coros.append(self._verify_item(item, vr))
            verify_results = await asyncio.gather(*verify_coros)
            for item, passed in zip(independent, verify_results, strict=True):
                item.status = "verified" if passed else "failed"
        return results


class SpawnAgentTool(Tool):
    """Tool for the LLM to spawn sub-agents for complex sub-tasks."""

    def __init__(self, coordinator: AgentCoordinator) -> None:
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        return "spawn_agent"

    @property
    def description(self) -> str:
        return (
            "Spawn a sub-agent to handle a specific sub-task independently. "
            "The sub-agent has its own conversation but shares your tools. "
            "Use for tasks that can be delegated (e.g., 'search for all "
            "usages of function X', 'refactor module Y'). "
            "Optionally reference a file-defined agent by name "
            "(see .godspeed/agents/). "
            "Returns the sub-agent's final response."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The sub-task to delegate to the sub-agent",
                },
                "agent_name": {
                    "type": "string",
                    "description": (
                        "Optional name of a file-defined agent "
                        "(.godspeed/agents/<name>.md). When set, its config "
                        "overrides model/effort."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override for the sub-agent",
                },
                "effort": {
                    "type": "string",
                    "description": "Effort level: low, normal, high",
                },
            },
            "required": ["task"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        task = arguments.get("task", "")
        if not task:
            return ToolResult.failure("task is required for spawn_agent")

        agent_name = arguments.get("agent_name")
        if agent_name is not None:
            from godspeed.skills.agent_loader import load_agent_definitions

            definitions = load_agent_definitions(context.cwd)
            definition = definitions.get(agent_name)
            if definition is None:
                available = ", ".join(sorted(definitions)) or "none"
                return ToolResult.failure(
                    f"Unknown agent {agent_name!r}. Available agents: {available}"
                )
            config = definition.to_config()
        else:
            config = SubAgentConfig(
                model=arguments.get("model"),
                effort=arguments.get("effort", "normal"),
            )
        result = await self._coordinator.spawn(task, depth=0, config=config)
        return ToolResult.success(result)


class SpawnKanbanTool(Tool):
    """Tool for orchestrating a Kanban plan with parallel sub-agents."""

    def __init__(self, coordinator: AgentCoordinator) -> None:
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        return "kanban_swarm"

    @property
    def description(self) -> str:
        return (
            "Execute a Kanban plan by spawning sub-agents for each work item. "
            "Items with dependencies are resolved in order."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The overall objective of the plan",
                },
                "items": {
                    "type": "array",
                    "description": "Work items to execute",
                },
                "worker_model": {
                    "type": "string",
                    "description": "Optional model for workers",
                },
                "verifier_model": {
                    "type": "string",
                    "description": "Optional model for verifier",
                },
            },
            "required": ["objective", "items"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        objective = arguments.get("objective", "")
        items_raw = arguments.get("items", [])
        if not objective:
            return ToolResult.failure("objective is required for kanban_swarm")
        if not items_raw:
            return ToolResult.failure("at least one work item is required for kanban_swarm")

        items = []
        for item_data in items_raw:
            if isinstance(item_data, dict):
                items.append(
                    WorkItem(
                        id=item_data.get("id", f"w{len(items) + 1}"),
                        description=item_data.get("description", ""),
                        files=item_data.get("files", []),
                        deps=item_data.get("deps", []),
                        acceptance=item_data.get("acceptance", []),
                    )
                )

        plan = KanbanPlan(objective=objective, items=items)
        swarm = KanbanSwarm(
            self._coordinator,
            worker_model=arguments.get("worker_model"),
            verifier_model=arguments.get("verifier_model"),
        )
        results = await swarm.execute(plan)
        summary = "\n".join(f"{k}: {v[:200]}" for k, v in results.items())
        return ToolResult.success(summary)
