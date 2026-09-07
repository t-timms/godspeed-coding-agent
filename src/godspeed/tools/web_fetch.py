"""Web fetch tool — retrieve web pages and extract readable text.

Uses urllib (stdlib) for zero-dependency HTTP, with a simple HTML tag stripper
for readable output. No external dependencies required.

Includes a 7-day disk cache (``~/.godspeed/cache/web/<sha1>.json``) so
repeated lookups of the same URL during a dev session — which happens
constantly with library docs and GitHub pages — don't re-hit the
network. Cache is bounded by ``_MAX_CACHE_BYTES`` total; older entries
are evicted when the budget is exceeded.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from godspeed.tools.base import WEB_CALL_SESSION_CAP, RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Max content size to download (1MB)
MAX_CONTENT_BYTES = 1_000_000
# Max text output to return
MAX_OUTPUT_CHARS = 10_000
# Request timeout
FETCH_TIMEOUT = 15

# ─── Disk cache settings ──────────────────────────────────────────────
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
_MAX_CACHE_BYTES = 50 * 1024 * 1024  # 50 MB total cap


def _cache_dir() -> Path:
    """Return the web-cache directory under ~/.godspeed/cache/web/.

    Created lazily on first use. Isolated under the user's Godspeed
    home so it's gitignored-by-default and easy to blow away.
    """
    d = Path.home() / ".godspeed" / "cache" / "web"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path_for(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _cache_dir() / f"{digest}.json"


def _cache_read(url: str) -> str | None:
    """Return cached text if fresh (within TTL); None otherwise.

    Silently returns None on any IO / JSON error — cache is advisory.
    """
    path = _cache_path_for(url)
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fetched_at = entry.get("fetched_at_ts", 0)
    if not isinstance(fetched_at, (int, float)):
        return None
    age = time.time() - fetched_at
    if age > CACHE_TTL_SECONDS:
        return None
    text = entry.get("text")
    if not isinstance(text, str):
        return None
    return text


def _cache_write(url: str, text: str) -> None:
    """Store a successful fetch. Best-effort; silent on errors.

    Runs an LRU-by-mtime eviction pass after writing so disk usage
    stays under _MAX_CACHE_BYTES.
    """
    path = _cache_path_for(url)
    entry = {"url": url, "fetched_at_ts": time.time(), "text": text}
    try:
        path.write_text(json.dumps(entry), encoding="utf-8")
    except OSError as exc:
        logger.debug("web_fetch cache write failed for %s: %s", url, exc)
        return
    with contextlib.suppress(OSError):
        _cache_evict_if_needed()


def _cache_evict_if_needed() -> None:
    """LRU eviction pass by file mtime until total size fits in _MAX_CACHE_BYTES."""
    cache_dir = _cache_dir()
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for p in cache_dir.glob("*.json"):
        try:
            stat = p.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, p))
        total += stat.st_size
    if total <= _MAX_CACHE_BYTES:
        return
    # Oldest first
    entries.sort(key=lambda e: e[0])
    for _mtime, size, p in entries:
        if total <= _MAX_CACHE_BYTES:
            break
        with contextlib.suppress(OSError):
            p.unlink()
        total -= size


# Tags whose content should be stripped entirely
_STRIP_TAGS = re.compile(
    r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# All HTML tags
_HTML_TAGS = re.compile(r"<[^>]+>")
# Collapse whitespace
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _html_to_text(raw_html: str) -> str:
    """Extract readable text from HTML — simple tag stripping."""
    text = _STRIP_TAGS.sub("", raw_html)
    text = _HTML_TAGS.sub("\n", text)
    text = html.unescape(text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _extract_content(raw_html: str, url: str) -> str:
    """Extract main article content using readability (fallback: tag stripping).

    Readability identifies the main content block of a page, removing
    navigation, sidebars, ads, and boilerplate. For pages without a clear
    article structure (APIs, raw data), falls back to simple HTML->text.
    """
    try:
        from readability import Document

        doc = Document(raw_html)
        title = doc.title() or ""
        content_html = doc.summary()

        # Extract text from the readability-cleaned HTML
        content = _html_to_text(content_html)

        if title and title.strip():
            content = f"Title: {title.strip()}\n\n{content}"

        # Readability sometimes returns very little for non-article pages.
        # If we got < 50 chars of useful content, fall back to full page.
        if len(content.strip()) < 50:
            logger.debug("Readability extracted too little from %s, falling back", url)
            return _html_to_text(raw_html)

        return content
    except Exception:
        logger.debug("Readability extraction failed for %s, falling back", url, exc_info=True)
        return _html_to_text(raw_html)


class WebFetchTool(Tool):
    """Fetch a web page and return readable text content.

    Strips HTML tags, scripts, and styles to return clean text.
    Useful for reading documentation, error messages, and API references.
    """

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL and return the page content as readable text. "
            "Strips HTML to plain text. Use for documentation, API refs, error lookups. "
            "Max 10K chars returned.\n\n"
            "Example: web_fetch(url='https://docs.python.org/3/library/asyncio.html')\n"
            "Example: web_fetch(url='https://fastapi.tiangolo.com/tutorial/dependencies/')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must start with http:// or https://)",
                },
                "no_cache": {
                    "type": "boolean",
                    "description": (
                        "Skip the 7-day disk cache and force a live fetch. "
                        "Default false — use only when the page is known to "
                        "change more often than weekly (e.g. live status pages)."
                    ),
                },
            },
            "required": ["url"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        url = arguments.get("url", "")

        if not isinstance(url, str) or not url:
            return ToolResult.failure("url must be a non-empty string")

        if not url.startswith(("http://", "https://")):
            return ToolResult.failure("url must start with http:// or https://")

        context.web_call_count += 1
        if context.web_call_count > WEB_CALL_SESSION_CAP:
            return ToolResult.failure(f"Session web-call cap ({WEB_CALL_SESSION_CAP}) reached")

        # Block local/private network access
        if _is_local_url(url):
            return ToolResult.failure("Cannot fetch local/private network URLs")

        # Cache hit? Return immediately without hitting the network.
        # Callers can bypass the cache by passing `no_cache=true`.
        if not arguments.get("no_cache", False):
            cached = _cache_read(url)
            if cached is not None:
                logger.debug("web_fetch cache hit: %s", url)
                return ToolResult.success(f"Content from {url} (cached):\n\n{cached}")

        try:
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={"User-Agent": "Godspeed-Agent/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:  # noqa: S310
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read(MAX_CONTENT_BYTES)

        except urllib.error.HTTPError as exc:
            return ToolResult.failure(f"HTTP {exc.code}: {exc.reason}")
        except urllib.error.URLError as exc:
            return ToolResult.failure(f"Failed to fetch URL: {exc.reason}")
        except TimeoutError:
            return ToolResult.failure(f"Request timed out after {FETCH_TIMEOUT}s")
        except Exception as exc:
            return ToolResult.failure(f"Fetch error: {exc}")

        # Decode
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", errors="replace")

        # Extract text from HTML — try readability first, fall back to tag stripping
        if "html" in content_type.lower():
            text = _extract_content(text, url)

        # Truncate
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n\n... (truncated, {len(text)} total chars)"

        if not text.strip():
            return ToolResult.success(f"Fetched {url} but no readable text content found.")

        # Persist successful fetch to disk cache (7-day TTL, 50 MB budget).
        _cache_write(url, text)

        return ToolResult.success(f"Content from {url}:\n\n{text}")


def _is_local_url(url: str) -> bool:
    """Check if a URL points to local/private network addresses."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or hostname.startswith(  # noqa: S104
        ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.")
    )
