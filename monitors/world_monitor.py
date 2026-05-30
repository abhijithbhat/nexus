"""World monitor coordinator — runs all monitors and produces the morning brief."""
import asyncio
from monitors.arxiv_monitor import ArxivMonitor
from monitors.github_monitor import GithubMonitor
from monitors.news_monitor import NewsMonitor
from monitors.hackathon_monitor import HackathonMonitor
from memory.memory_manager import MemoryManager
from agents.communicator import CommunicatorAgent
from connectors.whatsapp import WhatsAppConnector
from utils.logger import get_logger

logger = get_logger(__name__)


class WorldMonitor:
    """Coordinates all monitoring feeds and sends daily briefs to user."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        communicator: CommunicatorAgent,
        whatsapp: WhatsAppConnector,
    ) -> None:
        self._memory = memory_manager
        self._communicator = communicator
        self._whatsapp = whatsapp
        self._arxiv = ArxivMonitor()
        self._github = GithubMonitor()
        self._news = NewsMonitor()
        self._hackathon = HackathonMonitor()

    async def run_full_scan(self) -> dict:
        """Run all monitors concurrently and store results in memory."""
        logger.info("[WorldMonitor] Starting full scan...")

        # Run all monitors concurrently
        arxiv_task = asyncio.create_task(self._arxiv.fetch_papers())
        github_task = asyncio.create_task(self._github.fetch_trending())
        news_task = asyncio.create_task(self._news.fetch_news())
        hackathon_task = asyncio.create_task(self._hackathon.fetch_hackathons())

        papers, repos, news, hackathons = await asyncio.gather(
            arxiv_task, github_task, news_task, hackathon_task,
            return_exceptions=True,
        )

        # Handle exceptions
        papers = papers if isinstance(papers, list) else []
        repos = repos if isinstance(repos, list) else []
        news = news if isinstance(news, list) else []
        hackathons = hackathons if isinstance(hackathons, list) else []

        # Store high-relevance items in memory
        for paper in papers[:5]:
            await self._memory.remember(
                f"ArXiv: {paper['title']} — {paper['summary']}",
                type="world_signal",
                source="arxiv",
                importance=paper.get("relevance_score", 0.5),
            )

        for repo in repos[:3]:
            await self._memory.remember(
                f"GitHub Trending: {repo['title']} — {repo['description']} ({repo['stars']}★)",
                type="world_signal",
                source="github",
                importance=0.6,
            )

        results = {
            "papers": papers,
            "repos": repos,
            "news": news,
            "hackathons": hackathons,
        }
        logger.info(
            f"[WorldMonitor] Scan complete — "
            f"{len(papers)} papers, {len(repos)} repos, "
            f"{len(news)} news, {len(hackathons)} hackathons"
        )
        return results

    async def send_morning_brief(self) -> None:
        """Run a full scan and send a summarized morning brief via WhatsApp."""
        logger.info("[WorldMonitor] Preparing morning brief...")
        results = await self.run_full_scan()

        # Combine top items for summarization
        items = []
        for paper in results["papers"][:3]:
            items.append({"type": "📄 Paper", "title": paper["title"], "text": paper["summary"]})
        for repo in results["repos"][:2]:
            items.append({"type": "🔥 Repo", "title": repo["title"], "text": repo["description"]})
        for news_item in results["news"][:3]:
            items.append({"type": "📰 News", "title": news_item["title"], "text": news_item.get("summary", "")})
        for hack in results["hackathons"][:2]:
            items.append({"type": "🏆 Hackathon", "title": hack["title"], "text": hack.get("summary", "")})

        brief = await self._communicator.summarize_for_brief(items)
        header = "🌅 *NEXUS Morning Brief*\n━━━━━━━━━━━━━━━━━━━━━\n\n"
        full_brief = header + brief

        self._whatsapp.send_to_user(full_brief)
        await self._memory.remember(full_brief, type="morning_brief", source="nexus", importance=0.7)
        logger.info("[WorldMonitor] Morning brief sent!")
