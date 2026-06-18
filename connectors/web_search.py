import httpx
import asyncio
import re
import urllib.parse
from html.parser import HTMLParser
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
    Web search using DuckDuckGo HTML endpoint (free, no API keys, no rate limits).
    Falls back to the duckduckgo-search library if HTML scraping fails.
    """

    DDG_HTML_URL = "https://html.duckduckgo.com/html/"

    def __init__(self):
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        }

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search using DDG HTML endpoint (most reliable), fall back to library."""

        # Primary: DuckDuckGo HTML endpoint (NOT the API — avoids rate limits)
        results = await self._search_ddg_html(query, max_results)
        if results:
            return results

        # Fallback: duckduckgo-search library (can be rate-limited)
        results = await self._search_ddg_library(query, max_results)
        if results:
            return results

        logger.warning(f"All search backends failed for query: '{query}'")
        return []

    async def _search_ddg_html(self, query: str, max_results: int) -> list[dict]:
        """Scrape DuckDuckGo HTML endpoint — free, reliable, no API rate limits."""
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers=self._headers,
                follow_redirects=True
            ) as client:
                response = await client.post(
                    self.DDG_HTML_URL,
                    data={"q": query, "b": ""},
                )

                if response.status_code != 200:
                    logger.warning(f"DDG HTML returned {response.status_code}")
                    return []

                html = response.text

                # Extract result links and titles
                # DDG HTML results have class "result__a" for the title link
                result_pattern = re.compile(
                    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    re.DOTALL
                )
                # Snippets have class "result__snippet"
                snippet_pattern = re.compile(
                    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                    re.DOTALL
                )

                raw_links = result_pattern.findall(html)
                raw_snippets = snippet_pattern.findall(html)

                output = []
                for i, (raw_url, raw_title) in enumerate(raw_links[:max_results]):
                    # Clean title (remove HTML tags)
                    title = re.sub(r'<[^>]+>', '', raw_title).strip()

                    # Extract actual URL from DDG redirect
                    if "uddg=" in raw_url:
                        parsed = urllib.parse.parse_qs(
                            urllib.parse.urlparse(raw_url).query
                        )
                        url = parsed.get("uddg", [""])[0]
                    else:
                        url = raw_url

                    # Get snippet if available
                    snippet = ""
                    if i < len(raw_snippets):
                        snippet = re.sub(r'<[^>]+>', '', raw_snippets[i]).strip()

                    if url and title:
                        output.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        })

                logger.info(f"DDG HTML returned {len(output)} results for: '{query}'")
                return output

        except Exception as e:
            logger.warning(f"DDG HTML scraping failed: {e}")
            return []

    async def _search_ddg_library(self, query: str, max_results: int) -> list[dict]:
        """DuckDuckGo library — free but often rate-limited on servers."""
        try:
            from duckduckgo_search import DDGS

            clean_query = query.replace("-", " ")
            clean_query = "".join([c if c.isalnum() or c.isspace() else " " for c in clean_query])
            clean_query = " ".join(clean_query.split())
            logger.info(f"DDG library search: '{clean_query}'")

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
            logger.info(f"DDG library returned {len(output)} results")
            return output
        except Exception as e:
            logger.warning(f"DDG library search failed: {e}")
            return []

    async def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers=self._headers,
                follow_redirects=True
            ) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page: {url}. Status: {response.status_code}")
                    return ""

                parser = BodyTextParser()
                parser.feed(response.text)
                raw_text = parser.get_text()

                cleaned = " ".join(raw_text.split())

                if len(cleaned) > max_chars:
                    cleaned = cleaned[:max_chars]

                logger.info(f"Fetched text from: {url} ({len(cleaned)} chars)")
                return cleaned
        except Exception as e:
            logger.warning(f"Exception while scraping URL {url}: {e}")
            return ""
