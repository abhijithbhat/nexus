import json
from datetime import datetime
import asyncio
from utils.config import settings
from utils.logger import get_logger
from utils.gemini_client import GeminiClient
from utils.seen_tracker import SeenTracker
from memory.memory_manager import MemoryManager
from connectors.whatsapp import WhatsAppConnector
from monitors.arxiv_monitor import ArxivMonitor
from monitors.github_monitor import GitHubMonitor
from monitors.news_monitor import NewsMonitor
from monitors.hackathon_monitor import HackathonMonitor

logger = get_logger(__name__)

class WorldMonitor:
    def __init__(self, memory_manager: MemoryManager, whatsapp: WhatsAppConnector):
        self.memory_manager = memory_manager
        self.whatsapp = whatsapp
        self.gemini_client = GeminiClient()
        self.seen_tracker = SeenTracker()
        
        self.arxiv = ArxivMonitor()
        self.github = GitHubMonitor()
        self.news = NewsMonitor()
        self.hackathons = HackathonMonitor()
        logger.info("WorldMonitor initialized with sub-monitors.")

    async def run_full_scan(self) -> dict:
        logger.info("WorldMonitor starting background scan...")
        
        interests = [item.strip() for item in settings.user_interests.split(",") if item.strip()]
        
        # Parallel fetch from sources
        arxiv_task = self.arxiv.fetch_new_papers(interests, days_back=1)
        github_task = self.github.fetch_trending(language="python", since="daily")
        news_task = self.news.fetch_ai_news()
        hackathons_task = self.hackathons.fetch_upcoming_hackathons()
        
        raw_papers, raw_repos, raw_news, raw_hackathons = await asyncio.gather(
            arxiv_task, github_task, news_task, hackathons_task
        )
        
        scored_papers = []
        scored_repos = []
        scored_news = []
        scored_hackathons = []
        
        # Scorer functions
        async def score_paper(p):
            paper_id = p.get("url", p.get("title", ""))
            if self.seen_tracker.has_seen(paper_id):
                return
            score = await self.arxiv.score_relevance(p, settings.user_interests)
            if score >= 0.6:
                p["relevance_score"] = score
                scored_papers.append(p)
                txt = f"New ArXiv Paper: {p['title']} by {p['authors']}. Abstract: {p['abstract']}"
                await self.memory_manager.remember(txt, "monitor_signal", "arxiv_monitor", importance=score)
            self.seen_tracker.mark_seen(paper_id, p.get("title", ""))

        async def score_repo(r):
            repo_id = r.get("url", r.get("name", ""))
            if self.seen_tracker.has_seen(repo_id):
                return
            score = await self.github.score_relevance(r, settings.user_interests)
            if score >= 0.6:
                r["relevance_score"] = score
                scored_repos.append(r)
                txt = f"Trending Repo: {r['name']} ({r['language']}). Stars: {r['stars']}. Description: {r['description']}"
                await self.memory_manager.remember(txt, "monitor_signal", "github_monitor", importance=score)
            self.seen_tracker.mark_seen(repo_id, r.get("name", ""))

        async def score_news(n):
            news_id = n.get("url", n.get("title", ""))
            if self.seen_tracker.has_seen(news_id):
                return
            sys_p = "You are a tech news relevance evaluator for a personal intelligence agent."
            usr_m = (
                f"Rate 0.0-1.0 how relevant this news item is to someone with these interests: {settings.user_interests}\n\n"
                f"News:\nTitle: {n['title']}\nSummary: {n['summary']}\n\n"
                f"Return JSON: {{\"score\": 0.XX}}"
            )
            try:
                res = await self.gemini_client.generate_json(sys_p, usr_m, temperature=0.1)
                score = float(res.get("score", 0.0))
            except Exception:
                score = 0.0
                
            if score >= 0.6:
                n["relevance_score"] = score
                scored_news.append(n)
                txt = f"AI Tech News: {n['title']} ({n['source']}). Summary: {n['summary']}"
                await self.memory_manager.remember(txt, "monitor_signal", "news_monitor", importance=score)
            self.seen_tracker.mark_seen(news_id, n.get("title", ""))

        async def score_hackathon(h):
            hack_id = h.get("url", h.get("name", ""))
            if self.seen_tracker.has_seen(hack_id):
                return
            sys_p = "You are a hackathon relevance evaluator for a personal intelligence agent."
            usr_m = (
                f"Rate 0.0-1.0 how relevant this hackathon is to someone with these interests and goals: {settings.user_interests}, goals: {settings.user_goals}\n\n"
                f"Hackathon:\nName: {h['name']}\nTheme: {h['theme']}\nPrizes: {h['prize_pool']}\n\n"
                f"Return JSON: {{\"score\": 0.XX}}"
            )
            try:
                res = await self.gemini_client.generate_json(sys_p, usr_m, temperature=0.1)
                score = float(res.get("score", 0.0))
            except Exception:
                score = 0.0
                
            if score >= 0.6:
                h["relevance_score"] = score
                scored_hackathons.append(h)
                txt = f"Hackathon Event: {h['name']} (Deadline: {h['registration_deadline']}). Theme: {h['theme']}. Prizes: {h['prize_pool']}"
                await self.memory_manager.remember(txt, "monitor_signal", "hackathon_monitor", importance=score)
            self.seen_tracker.mark_seen(hack_id, h.get("name", ""))

        # Build list of async scoring tasks
        tasks = []
        for p in raw_papers:
            tasks.append(score_paper(p))
        for r in raw_repos:
            tasks.append(score_repo(r))
        for n in raw_news:
            tasks.append(score_news(n))
        for h in raw_hackathons:
            tasks.append(score_hackathon(h))
            
        if tasks:
            await asyncio.gather(*tasks)
            
        # Sort all arrays descending by relevance
        scored_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        scored_repos.sort(key=lambda x: x["relevance_score"], reverse=True)
        scored_news.sort(key=lambda x: x["relevance_score"], reverse=True)
        scored_hackathons.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(
            f"Finished scanning. Saved signals - Papers: {len(scored_papers)}, Repos: {len(scored_repos)}, "
            f"News: {len(scored_news)}, Hackathons: {len(scored_hackathons)}"
        )
        
        return {
            "papers": scored_papers,
            "repos": scored_repos,
            "news": scored_news,
            "hackathons": scored_hackathons,
            "scan_timestamp": datetime.utcnow().isoformat()
        }

    async def _get_email_summary(self) -> str:
        """Fetch unread email subjects for morning brief integration."""
        try:
            from connectors.gmail import GmailConnector
            gmail = GmailConnector()
            if not gmail.is_available:
                return "Gmail not configured."
            emails = gmail.get_unread_emails(max_results=5)
            if not emails:
                return "No unread emails."
            summary = []
            for e in emails[:3]:
                from_name = e['from'].split('<')[0].strip() if '<' in e['from'] else e['from'][:30]
                summary.append(f"• {from_name}: {e['subject'][:50]}")
            return "\n".join(summary)
        except Exception:
            return "Could not fetch emails."

    async def generate_morning_brief(self, scan_results: dict) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch email summary for morning brief
        email_summary = await self._get_email_summary()
        
        system_prompt = (
            f"You are NEXUS preparing a morning intelligence brief for {settings.user_name}. "
            "This is delivered via WhatsApp, so it must be: friendly, scannable, under 1400 characters total. "
            "Use emojis for section headers. Be selective — only include items with genuine relevance. "
            "End with one personalized recommendation."
        )
        
        user_message = (
            f"Create the morning brief from this data:\n"
            f"PAPERS: {json.dumps(scan_results['papers'][:3])}\n"
            f"GITHUB: {json.dumps(scan_results['repos'][:3])}\n"
            f"NEWS: {json.dumps(scan_results['news'][:3])}\n"
            f"HACKATHONS: {json.dumps(scan_results['hackathons'][:2])}\n"
            f"EMAILS: {email_summary}\n\n"
            f"Required format:\n"
            f"☀️ Morning Brief — {date_str} — {settings.user_name}\n\n"
            f"📧 EMAILS\n"
            f"[Unread email summary, 1-3 lines]\n\n"
            f"📄 PAPERS\n"
            f"[2-3 most relevant, one line each]\n\n"
            f"🔥 GITHUB\n"
            f"[2-3 repos, one line each]\n\n"
            f"📰 NEWS\n"
            f"[2-3 headlines, one line each]\n\n"
            f"🏆 HACKATHONS\n"
            f"[1-2 upcoming, with deadline]\n\n"
            f"💡 TODAY'S PICK\n"
            f"[One recommendation from NEXUS — what should {settings.user_name} pay attention to today and why]"
        )
        
        try:
            brief = await self.gemini_client.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.7
            )
            return brief.strip()
        except Exception as e:
            logger.error(f"Error generating morning brief: {e}")
            return f"Good morning {settings.user_name}! I had trouble fetching today's customized tech scan results."

    async def send_morning_brief(self) -> str:
        logger.info("Preparing daily morning brief dispatch...")
        results = await self.run_full_scan()
        brief = await self.generate_morning_brief(results)
        self.whatsapp.send_message(settings.user_whatsapp_number, brief)
        logger.info("Daily morning brief sent.")
        return brief
