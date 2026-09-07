"""Context assembly — 5-layer prompt construction with memoization.

Assembles the full context window for each LLM call from 5 layers,
each independently memoized and lazily evaluated:

    Layer 1: Core prompt — role, security, tool guidelines (never changes)
    Layer 2: Project instructions — GODSPEED.md / AGENTS.md / CLAUDE.md (per-cwd)
    Layer 3: Memory hints — user profile + project memories (lazy, relevance-filtered)
    Layer 4: Codebase context — repo map / skill content (lazy, task-triggered)
    Layer 5: Tool descriptions — name, description, risk_level (already cached)

Design features:
- Memoization with LRU cache per layer (invalidated on cwd change)
- Prefetch-while-streaming: start next context assembly while LLM streams
- Lazy skill loading: only loads skills relevant to the current task
- Prompt-cache markings for Anthropic/OpenAI cache_control
- Integrates with existing Conversation compaction tiers (32K/100K)
- Integrates with cheapest-model compaction for cost efficiency

References:
- Claude Code layered cached context
- openJiuwen Context Management + Goal Mode
- SOTA AHE finding: tools/memory carry gains; prompt-only regresses
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Cache bounds (LRU eviction)
_MAX_LAYER_CACHE_SIZE: int = 16

# Context budget thresholds (token fractions)
_CORE_BUDGET_FRACTION: float = 0.10
_PROJECT_BUDGET_FRACTION: float = 0.15
_MEMORY_BUDGET_FRACTION: float = 0.15
_CODEBASE_BUDGET_FRACTION: float = 0.40
_TOOL_BUDGET_FRACTION: float = 0.20

# Skill loading keywords for lazy task detection
_SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "testing": ("test", "pytest", "unittest", "coverage", "assert"),
    "security": ("security", "vulnerability", "vulnerabilities", "secret", "credential", "auth"),
    "database": ("sql", "query", "database", "db", "migration"),
    "git": ("git", "commit", "branch", "merge", "rebase", "diff"),
    "web": ("web", "html", "css", "javascript", "react", "fetch"),
    "data": ("csv", "json", "dataframe", "pandas", "numpy", "dataset"),
}

# Prompt-cache control headers for Anthropic/OpenAI
_CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class LayerResult:
    """Result from assembling a single context layer."""

    layer: int
    name: str
    content: str
    token_estimate: int
    cached: bool
    cache_hit_key: str | None = None


@dataclass
class AssemblyResult:
    """Full context assembly result."""

    system_prompt: str
    layers: list[LayerResult]
    total_token_estimate: int
    cache_hits: int
    cache_misses: int
    prefetch_task: asyncio.Task[str] | None = None


@dataclass
class ContextBudget:
    """Token budget allocation for context assembly."""

    max_tokens: int
    core_budget: int
    project_budget: int
    memory_budget: int
    codebase_budget: int
    tool_budget: int

    @classmethod
    def from_max_tokens(cls, max_tokens: int) -> ContextBudget:
        """Create a budget from max token count."""
        return cls(
            max_tokens=max_tokens,
            core_budget=int(max_tokens * _CORE_BUDGET_FRACTION),
            project_budget=int(max_tokens * _PROJECT_BUDGET_FRACTION),
            memory_budget=int(max_tokens * _MEMORY_BUDGET_FRACTION),
            codebase_budget=int(max_tokens * _CODEBASE_BUDGET_FRACTION),
            tool_budget=int(max_tokens * _TOOL_BUDGET_FRACTION),
        )


# ── LRU Cache ─────────────────────────────────────────────────────────


class _LayerLRUCache:
    """LRU cache keyed by (cwd, layer_hash) → content.

    Bounded to _MAX_LAYER_CACHE_SIZE entries with LRU eviction.
    """

    def __init__(self, max_size: int = _MAX_LAYER_CACHE_SIZE) -> None:
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> str | None:
        """Get cached content by key. Returns None on miss."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][1]
        return None

    def put(self, key: str, content: str) -> None:
        """Store content with current timestamp."""
        import time

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (time.time(), content)
            return
        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), content)

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries matching a key prefix. Returns count removed."""
        to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)


# ── Context Assembly ───────────────────────────────────────────────────


class ContextAssembler:
    """5-layer context assembly with memoization and lazy loading.

    Assembles the system prompt from independently memoized layers.
    Supports prefetch-while-streaming and prompt-cache markings.

    Args:
        max_tokens: Maximum context window tokens.
        cwd: Current working directory for project instruction loading.
        model: Model name for provider-specific cache markings.
    """

    def __init__(
        self,
        max_tokens: int = 100_000,
        cwd: Path | None = None,
        model: str = "",
    ) -> None:
        self._budget = ContextBudget.from_max_tokens(max_tokens)
        self._cwd = cwd or Path.cwd()
        self._model = model
        self._cache = _LayerLRUCache()
        self._core_prompt: str = ""
        self._tool_descriptions: str = ""
        self._project_instructions: str | None = None
        self._project_instructions_loaded: bool = False
        self._repo_map: str = ""
        self._repo_map_loaded: bool = False

    @property
    def cwd(self) -> Path:
        """Current working directory."""
        return self._cwd

    @cwd.setter
    def cwd(self, value: Path) -> None:
        """Set cwd and invalidate project-scoped caches."""
        if value != self._cwd:
            self._cwd = value
            self._invalidate_project_caches()

    def _invalidate_project_caches(self) -> None:
        """Invalidate caches that depend on cwd."""
        self._project_instructions = None
        self._project_instructions_loaded = False
        self._repo_map = ""
        self._repo_map_loaded = False
        self._cache.invalidate_prefix("layer2:")
        self._cache.invalidate_prefix("layer4:")
        logger.debug("Project caches invalidated cwd=%s", self._cwd)

    def set_core_prompt(self, prompt: str) -> None:
        """Set the core prompt (Layer 1). Call once at session start."""
        self._core_prompt = prompt
        self._cache.invalidate_prefix("layer1:")

    def set_tool_descriptions(self, descriptions: str) -> None:
        """Set tool descriptions (Layer 5). Call when tools change."""
        self._tool_descriptions = descriptions

    # ── Layer assembly ─────────────────────────────────────────────────

    def _layer_key(self, layer: int, extra: str = "") -> str:
        """Generate a cache key for a layer."""
        cwd_str = str(self._cwd)
        content = f"{cwd_str}:{extra}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"layer{layer}:{content_hash}"

    def _assemble_layer1_core(self) -> LayerResult:
        """Layer 1: Core prompt (cache keyed by prompt content)."""
        key = self._layer_key(1, extra=self._core_prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return LayerResult(
                layer=1,
                name="core",
                content=cached,
                token_estimate=_estimate_tokens(cached),
                cached=True,
                cache_hit_key=key,
            )

        content = self._core_prompt
        token_est = _estimate_tokens(content)
        self._cache.put(key, content)
        logger.debug("Layer 1 (core) assembled tokens=%d", token_est)
        return LayerResult(
            layer=1,
            name="core",
            content=content,
            token_estimate=token_est,
            cached=False,
        )

    def _assemble_layer2_project(self) -> LayerResult:
        """Layer 2: Project instructions from GODSPEED.md / AGENTS.md / CLAUDE.md."""
        key = self._layer_key(2)
        cached = self._cache.get(key)
        if cached is not None:
            return LayerResult(
                layer=2,
                name="project",
                content=cached,
                token_estimate=_estimate_tokens(cached),
                cached=True,
                cache_hit_key=key,
            )

        if not self._project_instructions_loaded:
            self._project_instructions = self._load_project_instructions()
            self._project_instructions_loaded = True

        content = self._project_instructions or ""
        token_est = _estimate_tokens(content)
        self._cache.put(key, content)
        logger.debug("Layer 2 (project) assembled tokens=%d", token_est)
        return LayerResult(
            layer=2,
            name="project",
            content=content,
            token_estimate=token_est,
            cached=False,
        )

    def _load_project_instructions(self) -> str | None:
        """Load project instructions using existing project_instructions module."""
        try:
            from godspeed.context.project_instructions import load_project_instructions

            return load_project_instructions(self._cwd)
        except ImportError:
            logger.debug("project_instructions module unavailable")
            return None

    def _assemble_layer3_memory(
        self,
        memory_store: Any | None = None,
        recall_query: str | None = None,
    ) -> LayerResult:
        """Layer 3: Memory hints — user profile + project memories + semantic recall.

        Cache key is query-independent: the expensive memory subsystem
        (profile + project memories) is the same regardless of query.
        Semantic recall results may vary but the base context is shared.
        """
        key = self._layer_key(3)
        cached = self._cache.get(key)
        if cached is not None:
            return LayerResult(
                layer=3,
                name="memory",
                content=cached,
                token_estimate=_estimate_tokens(cached),
                cached=True,
                cache_hit_key=key,
            )

        content = ""
        if memory_store is not None:
            try:
                if hasattr(memory_store, "recall_all"):
                    content = memory_store.recall_all(
                        query=recall_query,
                        project_limit=10,
                        profile=True,
                    )
            except Exception as exc:
                logger.warning("Memory recall failed: %s", exc)

        token_est = _estimate_tokens(content)
        self._cache.put(key, content)
        logger.debug("Layer 3 (memory) assembled tokens=%d", token_est)
        return LayerResult(
            layer=3,
            name="memory",
            content=content,
            token_estimate=token_est,
            cached=False,
        )

    def _assemble_layer4_codebase(self) -> LayerResult:
        """Layer 4: Codebase context — repo map / skill content.

        Lazy-loaded. Only fetches when explicitly needed, or attempts
        a background load if the repo map is available via repo_map module.
        """
        key = self._layer_key(4)
        cached = self._cache.get(key)
        if cached is not None:
            return LayerResult(
                layer=4,
                name="codebase",
                content=cached,
                token_estimate=_estimate_tokens(cached),
                cached=True,
                cache_hit_key=key,
            )

        if not self._repo_map_loaded:
            self._repo_map = self._load_repo_map_lazy()
            self._repo_map_loaded = True

        content = self._repo_map
        if not content.strip():
            logger.debug("Layer 4 (codebase) degraded: repo map empty")
            return LayerResult(
                layer=4,
                name="codebase",
                content="",
                token_estimate=0,
                cached=False,
            )
        token_est = _estimate_tokens(content)
        self._cache.put(key, content)
        logger.debug("Layer 4 (codebase) assembled tokens=%d", token_est)
        return LayerResult(
            layer=4,
            name="codebase",
            content=content,
            token_estimate=token_est,
            cached=False,
        )

    def _load_repo_map_lazy(self) -> str:
        """Try to load repo map from the repo_map module if available."""
        try:
            from godspeed.context.repo_map import RepoMapper

            mapper = RepoMapper()
            if not mapper.available:
                return ""
            content = mapper.map_directory(self._cwd)
            if content == "No symbols found in directory.":
                return ""
            return content
        except (ImportError, AttributeError):
            logger.debug("repo_map module unavailable for lazy load")
            return ""

    def _assemble_layer5_tools(self) -> LayerResult:
        """Layer 5: Tool descriptions — name, description, risk_level."""
        key = self._layer_key(5)
        cached = self._cache.get(key)
        if cached is not None:
            return LayerResult(
                layer=5,
                name="tools",
                content=cached,
                token_estimate=_estimate_tokens(cached),
                cached=True,
                cache_hit_key=key,
            )

        content = self._tool_descriptions
        token_est = _estimate_tokens(content)
        self._cache.put(key, content)
        logger.debug("Layer 5 (tools) assembled tokens=%d", token_est)
        return LayerResult(
            layer=5,
            name="tools",
            content=content,
            token_estimate=token_est,
            cached=False,
        )

    # ── Full assembly ──────────────────────────────────────────────────

    def assemble(
        self,
        memory_store: Any | None = None,
        recall_query: str | None = None,
    ) -> AssemblyResult:
        """Assemble all 5 layers into a complete system prompt.

        Layers are assembled independently. Memoized layers return cached
        content. Budget-aware: stops adding layers when budget exhausted.

        Args:
            memory_store: Optional MemoryStore for Layer 3.
            recall_query: Optional query for semantic recall in Layer 3.

        Returns:
            AssemblyResult with the full prompt and metadata.
        """
        layers: list[LayerResult] = []
        total_tokens = 0
        cache_hits = 0
        cache_misses = 0

        # Assemble each layer
        layer_results = [
            self._assemble_layer1_core(),
            self._assemble_layer2_project(),
            self._assemble_layer3_memory(memory_store, recall_query),
            self._assemble_layer4_codebase(),
            self._assemble_layer5_tools(),
        ]

        for result in layer_results:
            if result.cached:
                cache_hits += 1
            else:
                cache_misses += 1

            # Budget check: skip layer if would exceed total budget
            if total_tokens + result.token_estimate > self._budget.max_tokens:
                logger.warning(
                    "Layer %d (%s) skipped: would exceed budget (%d + %d > %d)",
                    result.layer,
                    result.name,
                    total_tokens,
                    result.token_estimate,
                    self._budget.max_tokens,
                )
                continue

            layers.append(result)
            total_tokens += result.token_estimate

        # Build the final system prompt
        parts = [layer.content for layer in layers if layer.content]
        system_prompt = "\n\n".join(parts)

        logger.info(
            "Context assembled layers=%d tokens=%d cache_hits=%d cache_misses=%d",
            len(layers),
            total_tokens,
            cache_hits,
            cache_misses,
        )

        return AssemblyResult(
            system_prompt=system_prompt,
            layers=layers,
            total_token_estimate=total_tokens,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    # ── Prefetch (runs in background while LLM streams) ────────────────

    def prefetch_async(
        self,
        memory_store: Any | None = None,
        recall_query: str | None = None,
    ) -> asyncio.Task[AssemblyResult]:
        """Start a background prefetch of the full context assembly.

        Returns an asyncio.Task that resolves to AssemblyResult.
        Use this while the LLM is streaming to prepare the next turn's
        context assembly.

        Example::

            # While LLM streams the response:
            prefetch_task = assembler.prefetch_async(memory_store)
            # ... handle streaming chunks ...
            # When next turn starts:
            result = await prefetch_task  # already done!
        """
        loop = asyncio.get_event_loop()
        return loop.create_task(
            self._prefetch_coro(memory_store=memory_store, recall_query=recall_query)
        )

    async def _prefetch_coro(
        self,
        memory_store: Any | None = None,
        recall_query: str | None = None,
    ) -> AssemblyResult:
        """Run the blocking assembly in an executor and await its result."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.assemble(
                memory_store=memory_store,
                recall_query=recall_query,
            ),
        )

    # ── Prompt-cache markings ──────────────────────────────────────────

    def apply_cache_control(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply prompt-cache control markings for Anthropic/OpenAI.

        Marks all but the last 2 messages with cache_control. This
        enables ~75-80% input cost reduction on long conversations.
        Delegates to LLMClient._apply_prompt_caching for provider-specific
        formatting.

        Args:
            messages: Conversation messages.

        Returns:
            Messages with cache_control markings applied.
        """
        try:
            from godspeed.llm.client import LLMClient

            return LLMClient._apply_prompt_caching(self._model, messages)
        except ImportError:
            return messages

    # ── Lazy skill loading ─────────────────────────────────────────────

    @staticmethod
    def detect_relevant_skills(
        task_description: str,
    ) -> list[str]:
        """Detect which skill categories are relevant to the task.

        Uses keyword matching to determine which skills to lazy-load.
        Avoids loading all skills every turn.

        Args:
            task_description: The user's task or query.

        Returns:
            List of relevant skill category names.
        """
        desc_lower = task_description.lower()
        relevant: list[str] = []
        for category, keywords in _SKILL_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                relevant.append(category)
        return relevant

    @staticmethod
    def format_skill_hints(
        relevant_skills: list[str],
        max_tokens: int = 500,
    ) -> str:
        """Format skill category hints for the system prompt.

        Returns brief guidance for relevant skill categories without
        loading full skill definitions (those load on-demand when the
        tool is called).
        """
        if not relevant_skills:
            return ""

        lines = ["Relevant skill categories for this task:"]
        for skill in relevant_skills[:5]:  # Cap at 5 to stay within budget
            lines.append(f"  - {skill}")

        result = "\n".join(lines)
        # Truncate if exceeds token budget (rough: 4 chars per token)
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "..."
        return result

    # ── Invalidation ───────────────────────────────────────────────────

    def invalidate_memory_cache(self) -> None:
        """Invalidate only the memory layer cache. Call after memory updates."""
        self._cache.invalidate_prefix("layer3:")
        logger.debug("Memory layer cache invalidated")

    def invalidate_all(self) -> None:
        """Invalidate all cached layers."""
        self._cache.clear()
        self._project_instructions = None
        self._project_instructions_loaded = False
        self._repo_map = ""
        self._repo_map_loaded = False
        logger.debug("All context caches invalidated")

    def set_repo_map(self, repo_map: str) -> None:
        """Set the repo map content for Layer 4."""
        self._repo_map = repo_map
        self._repo_map_loaded = True
        self._cache.invalidate_prefix("layer4:")


# ── Token estimation ──────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars for English text).

    This is a fast heuristic for budget planning. For accurate counts,
    use count_message_tokens() from the token_counter module.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
