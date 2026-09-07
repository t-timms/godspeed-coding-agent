"""Tests for bounded context: Layer3 cache key, lazy repo map, LSP cache."""

from __future__ import annotations

from pathlib import Path


from godspeed.context.assembly import ContextAssembler
from godspeed.context.lsp_feedback import LSPFeedbackProvider, _paths_match


class TestContextAssemblerLayer3:
    def test_layer3_cache_key_is_query_independent(self, tmp_path: Path) -> None:
        assembler = ContextAssembler(cwd=tmp_path)
        r1 = assembler._assemble_layer3_memory(memory_store=None, recall_query="query A")
        r2 = assembler._assemble_layer3_memory(memory_store=None, recall_query="query B")
        # Both should hit the same cache entry (query-independent key)
        assert r1.cached is False
        assert r2.cached is True
        assert r2.cache_hit_key is not None

    def test_layer3_cache_invalidated_by_memory(self, tmp_path: Path) -> None:
        assembler = ContextAssembler(cwd=tmp_path)
        assembler._assemble_layer3_memory(memory_store=None)
        assembler.invalidate_memory_cache()
        r = assembler._assemble_layer3_memory(memory_store=None)
        assert r.cached is False


class TestContextAssemblerLayer4:
    def test_lazy_repo_map_loads_when_available(self, tmp_path: Path, monkeypatch) -> None:
        assembler = ContextAssembler(cwd=tmp_path)
        assert assembler._repo_map_loaded is False

        class FakeMapper:
            available = True

            def map_directory(self, directory: Path) -> str:
                return "fake repo map"

        monkeypatch.setattr("godspeed.context.repo_map.RepoMapper", lambda: FakeMapper())
        result = assembler._assemble_layer4_codebase()
        assert "fake repo map" in result.content
        assert assembler._repo_map_loaded is True

    def test_lazy_repo_map_degrades_when_unavailable(self, tmp_path: Path) -> None:
        assembler = ContextAssembler(cwd=tmp_path)
        result = assembler._assemble_layer4_codebase()
        assert result.content == ""


class TestLSPFeedbackBounded:
    def test_cache_is_bounded(self, tmp_path: Path) -> None:
        provider = LSPFeedbackProvider(project_dir=tmp_path, enabled=False)
        provider._max_cache_entries = 3
        for i in range(10):
            provider._set_cached(
                Path(f"/tmp/file_{i}.py"),
                f"hash{i}",
                _make_fd(f"/tmp/file_{i}.py"),
            )
        assert provider.cache_size <= 3

    def test_cache_lru_eviction(self, tmp_path: Path) -> None:
        provider = LSPFeedbackProvider(project_dir=tmp_path, enabled=False)
        provider._max_cache_entries = 2
        provider._set_cached(Path("/tmp/a.py"), "h1", _make_fd("/tmp/a.py"))
        provider._set_cached(Path("/tmp/b.py"), "h2", _make_fd("/tmp/b.py"))
        # Access a to promote it
        provider._get_cached(Path("/tmp/a.py"), "h1")
        provider._set_cached(Path("/tmp/c.py"), "h3", _make_fd("/tmp/c.py"))
        # b should be evicted (LRU), a should survive
        assert provider._get_cached(Path("/tmp/a.py"), "h1") is not None
        assert provider._get_cached(Path("/tmp/b.py"), "h2") is None

    def test_clear_cache(self, tmp_path: Path) -> None:
        provider = LSPFeedbackProvider(project_dir=tmp_path, enabled=False)
        provider._set_cached(Path("/tmp/a.py"), "h1", _make_fd("/tmp/a.py"))
        assert provider.clear_cache() == 1
        assert provider.cache_size == 0


class TestPathsMatch:
    def test_exact_match(self) -> None:
        assert _paths_match("/a/b/c.py", "/a/b/c.py")

    def test_separator_normalization(self) -> None:
        assert _paths_match("C:\\a\\b\\c.py", "C:/a/b/c.py")

    def test_different_files_no_match(self) -> None:
        assert not _paths_match("/a/b/c.py", "/a/b/d.py")

    def test_same_basename_different_dir_no_match(self) -> None:
        assert not _paths_match("/a/c.py", "/b/c.py")


def _make_fd(path: str):
    from godspeed.context.lsp_feedback import FileDiagnostics

    return FileDiagnostics(file_path=path, diagnostics=[], timestamp=0.0)


class TestContextAssemblerLayer1:
    def test_set_core_prompt_invalidates_layer1_cache(self, tmp_path: Path) -> None:
        assembler = ContextAssembler(cwd=tmp_path)
        assembler.set_core_prompt("first prompt")
        r1 = assembler._assemble_layer1_core()
        assert r1.cached is False
        assert r1.content == "first prompt"
        assert assembler._assemble_layer1_core().cached is True
        assembler.set_core_prompt("second prompt")
        r2 = assembler._assemble_layer1_core()
        assert r2.cached is False
        assert r2.content == "second prompt"
