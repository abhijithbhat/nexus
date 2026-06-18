import httpx
import asyncio
from html.parser import HTMLParser
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BodyTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.ignore_tags = {"script", "style", "head", "title", "meta", "link", "nav", "footer"}
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        self.tag_stack.append(tag.lower())

    def handle_endtag(self, tag):
        if self.tag_stack:
            self.tag_stack.pop()

    def handle_data(self, data):
        if any(ignored in self.tag_stack for ignored in self.ignore_tags):
            return
        self.text_parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.text_parts)


class WebSearchConnector:
    """
    Web search with multiple backends (tried in order):
    1. Brave Search API (primary) — free 2,000 queries/month, reliable
    2. DuckDuckGo (fallback) — free, no API key, but rate-limits on servers
    """

    BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self):
        self.brave_api_key = getattr(settings, "brave_search_api_key", "")

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Try Brave Search first, fall back to DuckDuckGo."""

        # Try Brave Search API (most reliable)
        if self.brave_api_key:
            results = await self._search_brave(query, max_results)
            if results:
                return results

        # Fallback: DuckDuckGo
        results = await self._search_ddg(query, max_results)
        if results:
            return results

        logger.warning(f"All search backends failed for query: '{query}'")
        return []

    async def _search_brave(self, query: str, max_results: int) -> list[dict]:
        """Brave Search API — free tier: 2,000 queries/month."""
        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.brave_api_key
            }
            params = {
                "q": query,
                "count": max_results,
                "text_decorations": False,
                "search_lang": "en"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.BRAVE_API_URL, headers=headers, params=params)

                if response.status_code != 200:
                    logger.warning(f"Brave Search API returned {response.status_code}: {response.text[:200]}")
                    return []

                data = response.json()
                web_results = data.get("web", {}).get("results", [])

                output = []
                for r in web_results[:max_results]:
                    output.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("description", "")
                    })

                logger.info(f"Brave Search returned {len(output)} results for: '{query}'")
                return output

        except Exception as e:
            logger.warning(f"Brave Search failed: {e}")
            return []

    async def _search_ddg(self, query: str, max_results: int) -> list[dict]:
        """DuckDuckGo search — free but often rate-limited on servers."""
        try:
            from duckduckgo_search import DDGS

            clean_query = query.replace("-", " ")
            clean_query = "".join([c if c.isalnum() or c.isspace() else " " for c in clean_query])
            clean_query = " ".join(clean_query.split())
            logger.info(f"DDG search: '{clean_query}'")

            def sync_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(clean_query, max_results=max_results))

            raw_results = await asyncio.to_thread(sync_search)

            output = []
            for r in raw_results:
                output.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            logger.info(f"DDG returned {len(output)} results")
            return output
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

    async def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                )
            }
            async with httpx.AsyncClient(timeout=10.0, headers=headers, follow_redirects=True) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page: {url}. Status: {response.status_code}")
                    return ""

                parser = BodyTextParser()
                parser.feed(response.text)
                raw_text = parser.get_text()

                # Deduplicate spaces
                cleaned = " ".join(raw_text.split())

                if len(cleaned) > max_chars:
                    cleaned = cleaned[:max_chars]

                logger.info(f"Fetched text from: {url} ({len(cleaned)} chars)")
                return cleaned
        except Exception as e:
            logger.warning(f"Exception while scraping URL {url}: {e}")
            return ""
