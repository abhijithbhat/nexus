"""ArXiv monitor — fetches recent AI/ML papers matching user interests."""
import asyncio
import feedparser
from utils.logger import get_logger

logger = get_logger(__name__)

ARXIV_FEEDS = {
    "cs.AI": "http://export.arxiv.org/rss/cs.AI",
    "cs.LG": "http://export.arxiv.org/rss/cs.LG",
    "cs.CV": "http://export.arxiv.org/rss/cs.CV",
    "cs.CL": "http://export.arxiv.org/rss/cs.CL",
}

INTEREST_KEYWORDS = [
    "multi-agent", "langgraph", "autonomous agent", "vision transformer",
    "diffusion", "llm", "large language model", "reinforcement learning",
    "computer vision", "object detection", "transformer", "attention mechanism",
    "retrieval augmented", "RAG", "knowledge graph", "embedding",
]


class ArxivMonitor:
    """Fetches and scores recent ArXiv papers based on user interests."""

    async def fetch_papers(self, max_per_feed: int = 10) -> list[dict]:
        """Fetch recent papers from ArXiv RSS feeds."""
        all_papers = []
        for category, url in ARXIV_FEEDS.items():
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                for entry in feed.entries[:max_per_feed]:
                    title = entry.get("title", "").replace("\n", " ").strip()
                    summary = entry.get("summary", "").replace("\n", " ")[:300]
                    link = entry.get("link", "")
                    score = self._score_relevance(title, summary)
                    if score > 0:
                        all_papers.append({
                            "title": title,
                            "summary": summary,
                            "link": link,
                            "category": category,
                            "relevance_score": score,
                            "type": "arxiv_paper",
                        })
                logger.debug(f"[ArXiv] Fetched {len(feed.entries)} from {category}")
            except Exception as exc:
                logger.error(f"[ArXiv] Failed to fetch {category}: {exc}")

        # Sort by relevance and return top papers
        all_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"[ArXiv] Found {len(all_papers)} relevant papers")
        return all_papers[:15]

    @staticmethod
    def _score_relevance(title: str, summary: str) -> float:
        """Score a paper's relevance to user interests. 0.0-1.0."""
        text = (title + " " + summary).lower()
        hits = sum(1 for kw in INTEREST_KEYWORDS if kw.lower() in text)
        return min(hits / 3.0, 1.0)
