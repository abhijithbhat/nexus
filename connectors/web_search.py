"""DuckDuckGo web search + page fetcher connector."""
import asyncio
from duckduckgo_search import DDGS
import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchConnector:
    """Search the web and fetch page contents using DuckDuckGo and httpx."""

    def __init__(self) -> None:
        self._ddgs = DDGS()

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Run a DuckDuckGo text search. Returns list of {title, url, snippet}."""
        try:
            results = await asyncio.to_thread(
                self._ddgs.text, query, max_results=max_results
            )
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]
            logger.info(f"[WebSearch] '{query}' → {len(formatted)} results")
            return formatted
        except Exception as exc:
            logger.error(f"[WebSearch] Search failed for '{query}': {exc}")
            return []

    async def fetch_page(self, url: str, max_chars: int = 5000) -> str:
        """Fetch a web page and return its text content (truncated)."""
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "NEXUS-Agent/2.0"})
                resp.raise_for_status()
                text = resp.text[:max_chars]
                logger.info(f"[WebSearch] Fetched {url} — {len(text)} chars")
                return text
        except Exception as exc:
            logger.error(f"[WebSearch] Fetch failed {url}: {exc}")
            return ""

    async def search_and_summarize(self, query: str) -> str:
        """Search + fetch top results into a combined summary string."""
        results = await self.search(query, max_results=3)
        if not results:
            return f"No search results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")
        return "\n".join(lines)
