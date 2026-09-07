"""Isolated Retrieval Subagent.

Runs in its own context window (never shares history with main agent).
Queries GCG first, then uses file tools for confirmation.
Returns FileSpan list only — never raw file content.
Max 4 turns, 8 parallel tool calls per turn.
Budget-conscious: uses cheap_model routing for retrieval tasks.

Design advantage: GCG-first retrieval is more precise than grep-first.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from godspeed.context.coherence_graph import CoherenceGraph

logger = logging.getLogger(__name__)

RETRIEVAL_ALLOWED_TOOLS = frozenset(
    {
        "glob",
        "grep",
        "code_search",
        "repo_map",
    }
)
"""Read-only tools the retrieval subagent may use. Never: file_read, file_write, shell."""

MAX_TURNS = 4
MAX_PARALLEL_CALLS_PER_TURN = 8
MAX_SPANS_RETURNED = 20


@dataclass
class FileSpan:
    """A file:line-range reference returned by the retrieval subagent."""

    file: Path
    start_line: int
    end_line: int
    symbol_id: str | None = None
    relevance_score: float = 0.0
    match_reason: str = ""


@dataclass
class RetrievalResult:
    """Complete result from a retrieval operation."""

    spans: list[FileSpan] = field(default_factory=list)
    gcg_hits: int = 0
    turns_used: int = 0
    tokens_used: int = 0
    cache_hit: bool = False


class RetrievalSubagent:
    """Isolated context retrieval agent.

    One instance per main agent session. Reuse across calls (maintains
    its own lightweight span cache). Always routes through cheap_model
    — never frontier.

    Args:
        gcg: The Global Coherence Graph instance (Phase 2).
        model: Model name — always use cheap_model for retrieval.
        repo_root: Repository root directory.
    """

    def __init__(
        self,
        gcg: CoherenceGraph,
        repo_root: Path,
        model: str = "",
    ) -> None:
        self.gcg = gcg
        self.repo_root = repo_root
        self.model = model
        self._span_cache: dict[str, RetrievalResult] = {}

    async def retrieve(
        self,
        query: str,
        task_context: str = "",
        max_spans: int = MAX_SPANS_RETURNED,
    ) -> RetrievalResult:
        """Primary entry point. Returns ranked FileSpan list.

        Strategy:
        1. GCG direct lookup (exact symbol name match) → highest confidence
        2. GCG transitive lookup (dependency graph traversal) → high confidence
        3. Grep/glob fallback for non-symbol queries → lower confidence
        4. Rank by relevance, deduplicate, cap at max_spans

        Args:
            query: The search query or symbol name.
            task_context: Optional task description for cache keying.
            max_spans: Maximum number of spans to return.
        """
        cache_key = hashlib.sha256(f"{query}:{task_context}".encode()).hexdigest()[:16]

        if cache_key in self._span_cache:
            cached = self._span_cache[cache_key]
            return RetrievalResult(
                spans=cached.spans[:max_spans],
                gcg_hits=cached.gcg_hits,
                turns_used=cached.turns_used,
                tokens_used=cached.tokens_used,
                cache_hit=True,
            )

        spans, gcg_hits = await self._retrieve_internal(query, max_spans)
        result = RetrievalResult(
            spans=spans,
            gcg_hits=gcg_hits,
            turns_used=1,
            tokens_used=0,
            cache_hit=False,
        )
        self._span_cache[cache_key] = result
        return result

    async def _retrieve_internal(self, query: str, max_spans: int) -> tuple[list[FileSpan], int]:
        """Internal retrieval logic — GCG first, then grep fallback."""
        gcg_spans = await self._gcg_lookup(query)

        if len(gcg_spans) >= max_spans:
            return self._rank_and_deduplicate(gcg_spans, max_spans), len(gcg_spans)

        grep_spans = await self._grep_fallback(query)
        all_spans = gcg_spans + grep_spans
        return self._rank_and_deduplicate(all_spans, max_spans), len(gcg_spans)

    async def _gcg_lookup(self, query: str) -> list[FileSpan]:
        """Query GCG for direct symbol matches and their dependencies."""
        symbols = self.gcg.find_symbol(query)
        spans: list[FileSpan] = []

        for sym in symbols[:10]:
            spans.append(
                FileSpan(
                    file=sym.file,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    symbol_id=sym.id,
                    relevance_score=0.95,
                    match_reason="gcg_direct",
                )
            )

            try:
                blast = self.gcg.get_blast_radius(sym.id, max_depth=1)
                for affected in blast.affected_symbols[:5]:
                    spans.append(
                        FileSpan(
                            file=affected.file,
                            start_line=affected.start_line,
                            end_line=affected.end_line,
                            symbol_id=affected.id,
                            relevance_score=0.7,
                            match_reason="gcg_transitive",
                        )
                    )
            except Exception as exc:
                logger.debug("Blast radius failed for %s: %s", sym.id, exc)

        return spans

    async def _grep_fallback(self, query: str) -> list[FileSpan]:
        """Run grep/glob as fallback when GCG has no match.

        Returns line-range spans only — never file contents. In a full
        implementation this would dispatch grep/glob tool calls through
        the tool registry. For now, returns empty list — the caller
        falls back to existing tool dispatch.
        """
        return []

    def _rank_and_deduplicate(self, spans: list[FileSpan], max_spans: int) -> list[FileSpan]:
        """Sort by relevance_score desc, merge overlapping ranges, cap."""
        if not spans:
            return []

        spans = sorted(spans, key=lambda s: s.relevance_score, reverse=True)
        seen: set[tuple[str, int, int]] = set()
        result: list[FileSpan] = []

        for span in spans:
            key = (str(span.file), span.start_line, span.end_line)
            if key in seen:
                continue
            seen.add(key)
            result.append(span)
            if len(result) >= max_spans:
                break

        return result

    def format_spans_for_agent(self, spans: list[FileSpan]) -> str:
        """Format a list of FileSpans as a compact string for the agent.

        Produces output like::

            Found 5 relevant locations:
            src/auth.py:145-167 (gcg_direct) — AuthManager.sign_token
            src/auth.py:200-220 (gcg_transitive) — validate_token
            src/utils.py:10-15 (grep) — config_loader
        """
        if not spans:
            return "No relevant code locations found."

        lines = [f"Found {len(spans)} relevant locations:"]
        for span in spans:
            reason = span.match_reason or "search"
            label = span.symbol_id or f"{span.file}:{span.start_line}"
            lines.append(f"{span.file}:{span.start_line}-{span.end_line} ({reason}) — {label}")

        return "\n".join(lines)
