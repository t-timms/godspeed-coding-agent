"""Tests for godspeed.tools.web_search."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.web_search import WebSearchTool, _parse_ddg_html


def _ctx() -> ToolContext:
    return ToolContext(cwd=Path.cwd(), session_id="test")


class TestWebSearchToolMetadata:
    def test_name(self):
        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_risk_level(self):
        tool = WebSearchTool()
        assert tool.risk_level.value == "read_only"

    def test_description_contains_keywords(self):
        tool = WebSearchTool()
        desc = tool.description.lower()
        assert "search" in desc
        assert "web" in desc

    def test_get_schema(self):
        tool = WebSearchTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "query" in schema["required"]


class TestWebSearchExecute:
    @pytest.mark.asyncio
    async def test_missing_query(self):
        tool = WebSearchTool()
        result = await tool.execute({}, _ctx())
        assert result.is_error is True
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": ""}, _ctx())
        assert result.is_error is True
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_query(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": "   "}, _ctx())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_search_success(self):
        tool = WebSearchTool()
        mock_results = [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
            {"title": "Result 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
        ]
        with patch("godspeed.tools.web_search._search_ddg", return_value=mock_results):
            result = await tool.execute({"query": "test query"}, _ctx())
            assert result.is_error is False
            assert "Result 1" in result.output
            assert "https://example.com/1" in result.output

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        tool = WebSearchTool()
        with patch("godspeed.tools.web_search._search_ddg", return_value=[]):
            result = await tool.execute({"query": "test query"}, _ctx())
            assert result.is_error is False
            assert "no results" in result.output.lower()

    @pytest.mark.asyncio
    async def test_search_with_max_results(self):
        tool = WebSearchTool()
        mock_results = [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet {i}"}
            for i in range(10)
        ]
        with patch("godspeed.tools.web_search._search_ddg", return_value=mock_results[:3]):
            result = await tool.execute({"query": "test", "max_results": 3}, _ctx())
            assert result.is_error is False
            assert "Result 1" in result.output

    @pytest.mark.asyncio
    async def test_search_max_results_capped(self):
        tool = WebSearchTool()
        # Even if user asks for 100, should cap at 15
        mock_results = []
        with patch(
            "godspeed.tools.web_search._search_ddg", return_value=mock_results
        ) as mock_search:
            await tool.execute({"query": "test", "max_results": 100}, _ctx())
            # Check that max_results was capped to 15
            call_args = mock_search.call_args
            assert call_args[0][1] <= 15

    @pytest.mark.asyncio
    async def test_search_exception(self):
        tool = WebSearchTool()
        with patch("godspeed.tools.web_search._search_ddg", side_effect=Exception("network error")):
            result = await tool.execute({"query": "test"}, _ctx())
            assert result.is_error is True
            assert (
                "search failed" in result.error.lower() or "network error" in result.error.lower()
            )


class TestParseDdgHtml:
    def test_parse_results(self):
        html = """
        <a class="result__a" href="https://example.com/1">Test Title 1</a>
        <a class="result__snippet">Snippet 1</a>
        <a class="result__a" href="https://example.com/2">Test Title 2</a>
        <a class="result__snippet">Snippet 2</a>
        """
        results = _parse_ddg_html(html, 5)
        assert len(results) == 2
        assert results[0]["title"] == "Test Title 1"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["snippet"] == "Snippet 1"

    def test_parse_with_ddg_redirect(self):
        html = """
        <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F1">Title</a>
        <a class="result__snippet">Snippet</a>
        """
        results = _parse_ddg_html(html, 5)
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/1"

    def test_parse_max_results(self):
        html = ""
        for i in range(10):
            html += f'<a class="result__a" href="https://example.com/{i}">Title {i}</a>\n'
            html += f'<a class="result__snippet">Snippet {i}</a>\n'
        results = _parse_ddg_html(html, 3)
        assert len(results) == 3

    def test_parse_no_results(self):
        html = "<html><body>No results here</body></html>"
        results = _parse_ddg_html(html, 5)
        assert len(results) == 0

    def test_parse_html_tags_stripped(self):
        html = """
        <a class="result__a" href="https://example.com/1"><b>Bold Title</b></a>
        <a class="result__snippet"><i>Italic snippet</i></a>
        """
        results = _parse_ddg_html(html, 5)
        assert len(results) == 1
        assert "<b>" not in results[0]["title"]
        assert "</b>" not in results[0]["title"]
        assert "Bold Title" in results[0]["title"]

    def test_parse_missing_snippet(self):
        html = """
        <a class="result__a" href="https://example.com/1">Title</a>
        """
        results = _parse_ddg_html(html, 5)
        assert len(results) == 1
        assert results[0]["snippet"] == ""


class TestWebCallCap:
    @pytest.mark.asyncio
    async def test_web_call_cap_enforced(self):
        from godspeed.tools.base import WEB_CALL_SESSION_CAP

        tool = WebSearchTool()
        ctx = _ctx()
        ctx.web_call_count = WEB_CALL_SESSION_CAP
        result = await tool.execute({"query": "x"}, ctx)
        assert result.is_error is True
        assert "cap" in (result.error or "")
