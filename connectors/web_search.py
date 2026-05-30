import httpx
import asyncio
from html.parser import HTMLParser
from duckduckgo_search import DDGS
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
    def __init__(self):
        pass

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            def sync_search():
                with DDGS() as ddgs:
                    # Retrieve web search listings
                    return list(ddgs.text(query, max_results=max_results))

            raw_results = await asyncio.to_thread(sync_search)
            
            output = []
            for r in raw_results:
                output.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            logger.info(f"DuckDuckGo search returned {len(output)} results for query: '{query}'")
            return output
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}")
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
                    
                logger.info(f"Fetched and extracted text from: {url} ({len(cleaned)} chars)")
                return cleaned
        except Exception as e:
            logger.warning(f"Exception while scraping URL {url}: {e}")
            return ""
