import re
import asyncio
import feedparser
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)

class NewsMonitor:
    RSS_FEEDS = [
        "https://rss.arxiv.org/rss/cs.AI",
        "https://techcrunch.com/tag/artificial-intelligence/feed/",
        "https://www.wired.com/feed/tag/artificial-intelligence/latest/rss",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
    ]

    def __init__(self):
        pass

    def _titles_similar(self, title1: str, title2: str) -> bool:
        words1 = set(re.findall(r"\w+", title1.lower()))
        words2 = set(re.findall(r"\w+", title2.lower()))
        if not words1 or not words2:
            return False
        intersection = words1.intersection(words2)
        smaller_len = min(len(words1), len(words2))
        return len(intersection) / smaller_len > 0.6

    async def fetch_ai_news(self) -> list[dict]:
        logger.info("NewsMonitor starting AI news fetch...")
        cutoff = datetime.utcnow() - timedelta(hours=24)

        async def parse_feed(url: str) -> list[dict]:
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                source_name = feed.feed.get("title", url.split("/")[2] if len(url.split("/")) > 2 else "RSS Feed")
                
                entries = []
                for entry in feed.entries:
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_dt = datetime(*entry.published_parsed[:6])
                    else:
                        pub_dt = datetime.utcnow()
                        
                    if pub_dt >= cutoff:
                        title = entry.get("title", "")
                        summary = entry.get("summary", "") or entry.get("description", "")
                        cleaned_summary = re.sub(r"<[^>]+>", "", summary).strip()
                        cleaned_summary = " ".join(cleaned_summary.split())
                        
                        entries.append({
                            "title": title.strip(),
                            "summary": cleaned_summary[:200] + "..." if len(cleaned_summary) > 200 else cleaned_summary,
                            "url": entry.get("link", ""),
                            "published": pub_dt.isoformat(),
                            "source": source_name
                        })
                return entries
            except Exception as e:
                logger.error(f"Error parsing RSS feed at {url}: {e}")
                return []

        tasks = [parse_feed(url) for url in self.RSS_FEEDS]
        results = await asyncio.gather(*tasks)
        
        # Flatten results list
        all_entries = []
        for entry_list in results:
            all_entries.extend(entry_list)
            
        all_entries.sort(key=lambda x: x["published"], reverse=True)
        
        # Filter duplicates
        deduplicated = []
        for item in all_entries:
            is_dup = False
            for existing in deduplicated:
                if self._titles_similar(item["title"], existing["title"]):
                    is_dup = True
                    break
            if not is_dup:
                deduplicated.append(item)
                
        top_news = deduplicated[:10]
        logger.info(f"NewsMonitor finished. Gathered {len(top_news)} news entries.")
        return top_news
