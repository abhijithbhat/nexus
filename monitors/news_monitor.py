"""Tech news monitor via RSS feeds."""
import asyncio
import feedparser
from utils.logger import get_logger

logger = get_logger(__name__)

NEWS_FEEDS = {
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "Hacker News Best": "https://hnrss.org/best",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
}


class NewsMonitor:
    """Monitors tech news RSS feeds for relevant updates."""

    async def fetch_news(self, max_per_feed: int = 5) -> list[dict]:
        """Fetch recent tech news from RSS feeds."""
        all_news = []
        for source, url in NEWS_FEEDS.items():
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                for entry in feed.entries[:max_per_feed]:
                    all_news.append({
                        "title": entry.get("title", "").strip(),
                        "summary": entry.get("summary", entry.get("description", ""))[:200],
                        "link": entry.get("link", ""),
                        "source": source,
                        "type": "tech_news",
                    })
                logger.debug(f"[News] Fetched {len(feed.entries)} from {source}")
            except Exception as exc:
                logger.warning(f"[News] Failed to fetch {source}: {exc}")

        logger.info(f"[News] Collected {len(all_news)} news items")
        return all_news
