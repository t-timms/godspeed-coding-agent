"""Web search tool — search the web for documentation, errors, and solutions.

Uses DuckDuckGo HTML search (no API key required) for zero-config web search.
Falls back gracefully if the search fails.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from godspeed.tools.base import WEB_CALL_SESSION_CAP, RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT = 10
MAX_RESULTS = 8

# DuckDuckGo HTML search (no API key needed)
_DDG_URL = "https://html.duckduckgo.com/html/"


class WebSearchTool(Tool):
    """Search the web for documentation, error messages, and solutions.

    Uses DuckDuckGo (no API key required). Returns titles, URLs, and snippets
    for the top results. Use web_fetch to read full pages from the results.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for documentation, error messages, API references, "
            "or solutions. Returns titles, URLs, and snippets. No API key required. "
            "Use web_fetch to read full pages from results.\n\n"
            "Example: web_search(query='python asyncio timeout example')\n"
            "Example: web_search(query='FastAPI dependency injection', max_results=5)"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'python asyncio timeout example')",
                    "examples": ["python asyncio timeout example", "FastAPI middleware tutorial"],
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Maximum results to return (default: {MAX_RESULTS})",
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", MAX_RESULTS)

        if not isinstance(query, str) or not query.strip():
            return ToolResult.failure("query must be a non-empty string")

        context.web_call_count += 1
        if context.web_call_count > WEB_CALL_SESSION_CAP:
            return ToolResult.failure(f"Session web-call cap ({WEB_CALL_SESSION_CAP}) reached")

        max_results = min(max_results, 15)

        try:
            results = _search_ddg(query, max_results)
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return ToolResult.failure(f"Search failed: {exc}")

        if not results:
            return ToolResult.success(f"No results found for: {query}")

        # Format results
        lines = [f"Search results for: {query}\n"]
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result['title']}")
            lines.append(f"   {result['url']}")
            if result.get("snippet"):
                lines.append(f"   {result['snippet']}")
            lines.append("")

        return ToolResult.success("\n".join(lines))


def _search_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    """Search DuckDuckGo HTML endpoint and parse results."""
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        _DDG_URL,
        data=data,
        headers={
            "User-Agent": "Godspeed-Agent/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")

    return _parse_ddg_html(html, max_results)


def _parse_ddg_html(html_content: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML search results.

    DuckDuckGo HTML results have a consistent structure:
    - Result links in <a class="result__a" href="...">
    - Snippets in <a class="result__snippet" ...>
    """
    results: list[dict[str, str]] = []

    # Find result blocks
    result_pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    links = result_pattern.findall(html_content)
    snippets = snippet_pattern.findall(html_content)

    for i, (raw_url, raw_title) in enumerate(links[:max_results]):
        # Clean URL (DDG wraps URLs in a redirect)
        url = raw_url
        if "uddg=" in url:
            url_match = re.search(r"uddg=([^&]+)", url)
            if url_match:
                url = urllib.parse.unquote(url_match.group(1))

        # Clean title (strip HTML tags)
        title = re.sub(r"<[^>]+>", "", raw_title).strip()

        # Get snippet if available
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results
