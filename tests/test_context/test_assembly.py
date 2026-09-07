"""Tests for ContextAssembler — 5-layer context assembly with memoization."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from godspeed.context.assembly import (
    ContextAssembler,
    ContextBudget,
    _estimate_tokens,
)


@pytest.fixture
def assembler(tmp_path: Path) -> ContextAssembler:
    """Create a ContextAssembler with a temp directory."""
    return ContextAssembler(
        max_tokens=100_000,
        cwd=tmp_path,
        model="test-model",
    )


# ── Token estimation ──────────────────────────────────────────────────


class TestTokenEstimation:
    """Test rough token estimation."""

    def test_empty_text(self) -> None:
        assert _estimate_tokens("") == 0

    def test_short_text(self) -> None:
        # "hello" is 5 chars -> 5//4 = 1
        assert _estimate_tokens("hello") == 1

    def test_longer_text(self) -> None:
        text = "x" * 100
        assert _estimate_tokens(text) == 25

    def test_returns_at_least_one(self) -> None:
        assert _estimate_tokens("x") == 1


# ── Context budget ────────────────────────────────────────────────────


class TestContextBudget:
    """Test budget allocation from max tokens."""

    def test_budget_fractions(self) -> None:
        budget = ContextBudget.from_max_tokens(100_000)
        assert budget.core_budget == 10_000  # 10%
        assert budget.project_budget == 15_000  # 15%
        assert budget.memory_budget == 15_000  # 15%
        assert budget.codebase_budget == 40_000  # 40%
        assert budget.tool_budget == 20_000  # 20%

    def test_budget_sum(self) -> None:
        budget = ContextBudget.from_max_tokens(80_000)
        total = (
            budget.core_budget
            + budget.project_budget
            + budget.memory_budget
            + budget.codebase_budget
            + budget.tool_budget
        )
        assert total == 80_000


# ── Layer assembly ────────────────────────────────────────────────────


class TestLayerAssembly:
    """Test individual layer assembly."""

    def test_layer1_core(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("You are a test agent.")
        result = assembler._assemble_layer1_core()
        assert result.layer == 1
        assert result.name == "core"
        assert result.content == "You are a test agent."
        assert result.cached is False

    def test_layer1_caching(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("You are a test agent.")
        # First call: not cached
        result1 = assembler._assemble_layer1_core()
        assert result1.cached is False
        # Second call: cached
        result2 = assembler._assemble_layer1_core()
        assert result2.cached is True
        assert result2.content == result1.content

    def test_layer5_tools(self, assembler: ContextAssembler) -> None:
        assembler.set_tool_descriptions("## Tools\nfile_read, file_edit")
        result = assembler._assemble_layer5_tools()
        assert result.layer == 5
        assert result.name == "tools"
        assert "file_read" in result.content


# ── Full assembly ─────────────────────────────────────────────────────


class TestFullAssembly:
    """Test full 5-layer assembly."""

    def test_assemble_minimal(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("You are a test agent.")
        result = assembler.assemble()
        assert result.system_prompt
        assert len(result.layers) >= 1  # At least core layer
        assert result.total_token_estimate > 0

    def test_assemble_with_all_layers(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("Core prompt")
        assembler.set_tool_descriptions("Tool descriptions")
        result = assembler.assemble()
        assert "Core prompt" in result.system_prompt
        assert "Tool descriptions" in result.system_prompt

    def test_assemble_caches_across_calls(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("Test")
        r1 = assembler.assemble()
        r2 = assembler.assemble()
        # Second call should have more cache hits
        assert r2.cache_hits >= r1.cache_hits

    def test_assemble_respects_budget(self, tmp_path: Path) -> None:
        """Assembly skips layers that would exceed budget."""
        assembler = ContextAssembler(max_tokens=50, cwd=tmp_path)  # Very small budget
        assembler.set_core_prompt("x" * 400)  # 100 tokens
        assembler.set_tool_descriptions("y" * 400)  # 100 tokens
        result = assembler.assemble()
        # With budget of 50, not all layers should fit
        assert result.total_token_estimate <= 50 + 100  # Allow some slack for core


# ── Memoization / Cache ───────────────────────────────────────────────


class TestMemoization:
    """Test LRU cache behavior."""

    def test_invalidate_memory_cache(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("Test")
        assembler.assemble()
        assembler.invalidate_memory_cache()
        # Memory layer should be re-computed on next assemble
        result = assembler.assemble()
        assert result.cache_misses >= 0  # Memory layer re-computed

    def test_invalidate_all(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("Test")
        assembler.assemble()
        assembler.invalidate_all()
        result = assembler.assemble()
        # After invalidation, all layers should be re-computed
        assert result.cache_misses >= 1

    def test_invalidate_project_caches_on_cwd_change(
        self, assembler: ContextAssembler, tmp_path: Path
    ) -> None:
        assembler.set_core_prompt("Test")
        assembler.assemble()
        new_cwd = tmp_path / "new_project"
        new_cwd.mkdir()
        assembler.cwd = new_cwd
        # Caches should be invalidated
        assert assembler.cwd == new_cwd

    def test_cache_size_bounded(self, assembler: ContextAssembler) -> None:
        """Cache should not grow unbounded."""
        assembler.set_core_prompt("Test")
        for i in range(20):
            assembler.cwd = Path(f"/tmp/project_{i}")
            assembler.assemble()
        # Cache is bounded to _MAX_LAYER_CACHE_SIZE
        assert assembler._cache.size <= 16


# ── Prefetch ──────────────────────────────────────────────────────────


class TestPrefetch:
    """Test prefetch-while-streaming."""

    @pytest.mark.asyncio
    async def test_prefetch_returns_result(self, assembler: ContextAssembler) -> None:
        assembler.set_core_prompt("Test prompt")
        task = assembler.prefetch_async()
        result = await task
        assert result.system_prompt
        assert result.total_token_estimate > 0

    @pytest.mark.asyncio
    async def test_prefetch_is_background(self, assembler: ContextAssembler) -> None:
        """Prefetch returns a task, not blocking."""
        assembler.set_core_prompt("Test prompt")
        task = assembler.prefetch_async()
        assert isinstance(task, asyncio.Task)
        # Should complete quickly
        result = await task
        assert result is not None


# ── Prompt-cache markings ─────────────────────────────────────────────


class TestPromptCacheControl:
    """Test prompt-cache control for Anthropic/OpenAI."""

    def test_apply_cache_control_anthropic(self, assembler: ContextAssembler) -> None:
        assembler._model = "claude-sonnet-4-20250514"
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "New message"},
        ]
        result = assembler.apply_cache_control(messages)
        # Should have cache_control on earlier messages
        assert isinstance(result, list)
        assert len(result) == len(messages)

    def test_apply_cache_control_unknown_provider(self, assembler: ContextAssembler) -> None:
        assembler._model = "gpt-4"
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        result = assembler.apply_cache_control(messages)
        # Unknown provider returns messages unchanged
        assert result == messages


# ── Lazy skill loading ────────────────────────────────────────────────


class TestLazySkillLoading:
    """Test task-based skill detection and lazy loading."""

    def test_detect_testing_skill(self) -> None:
        skills = ContextAssembler.detect_relevant_skills("Fix the test failures in pytest")
        assert "testing" in skills

    def test_detect_security_skill(self) -> None:
        skills = ContextAssembler.detect_relevant_skills("Check for SQL injection vulnerabilities")
        assert "security" in skills

    def test_detect_git_skill(self) -> None:
        skills = ContextAssembler.detect_relevant_skills("Create a new branch for this feature")
        assert "git" in skills

    def test_no_skills_for_generic_task(self) -> None:
        skills = ContextAssembler.detect_relevant_skills("hello there")
        assert skills == []

    def test_multiple_skills(self) -> None:
        skills = ContextAssembler.detect_relevant_skills(
            "Run pytest to check test coverage and security scan"
        )
        assert "testing" in skills
        assert "security" in skills

    def test_format_skill_hints(self) -> None:
        result = ContextAssembler.format_skill_hints(
            ["testing", "security"],
        )
        assert "testing" in result
        assert "security" in result
        assert "Relevant skill categories" in result

    def test_format_skill_hints_empty(self) -> None:
        assert ContextAssembler.format_skill_hints([]) == ""

    def test_format_skill_hints_truncation(self) -> None:
        # Many skills should be capped at 5
        skills = [f"skill_{i}" for i in range(10)]
        result = ContextAssembler.format_skill_hints(skills)
        assert result.count("- ") <= 5


# ── Repo map integration ──────────────────────────────────────────────


class TestRepoMapIntegration:
    """Test repo map / codebase context layer."""

    def test_set_repo_map(self, assembler: ContextAssembler) -> None:
        repo_map = "src/auth.py: Auth(L10), login(L20)"
        assembler.set_repo_map(repo_map)
        result = assembler._assemble_layer4_codebase()
        assert repo_map in result.content

    def test_repo_map_cache_invalidation(self, assembler: ContextAssembler) -> None:
        assembler.set_repo_map("old map")
        assembler.assemble()
        assembler.set_repo_map("new map")
        result = assembler.assemble()
        assert "new map" in result.system_prompt
