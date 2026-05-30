import xml.etree.ElementTree as ET
import httpx
import asyncio
from datetime import datetime, timedelta
import pytz
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)

class ArxivMonitor:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def fetch_new_papers(self, topics: list[str], days_back: int = 1) -> list[dict]:
        logger.info(f"ArxivMonitor starting fetch for topics: {topics}")
        cutoff = datetime.now(pytz.utc) - timedelta(days=days_back)
        
        async def fetch_topic(topic: str) -> list[dict]:
            # Arxiv query formatting
            query_topic = topic.strip().replace(" ", "+")
            url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=ti:{query_topic}+OR+abs:{query_topic}"
                f"&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
            )
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.get(url)
                    if res.status_code != 200:
                        logger.warning(f"Arxiv API failed for topic '{topic}': status={res.status_code}")
                        return []
                        
                    root = ET.fromstring(res.content)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    
                    papers = []
                    for entry in root.findall("atom:entry", ns):
                        title = entry.find("atom:title", ns)
                        title_val = title.text.strip().replace("\n", " ") if title is not None else ""
                        
                        summary = entry.find("atom:summary", ns)
                        summary_val = summary.text.strip().replace("\n", " ") if summary is not None else ""
                        
                        published = entry.find("atom:published", ns)
                        pub_val = published.text.strip() if published is not None else ""
                        
                        link_id = entry.find("atom:id", ns)
                        url_val = link_id.text.strip() if link_id is not None else ""
                        
                        authors = []
                        for author in entry.findall("atom:author", ns):
                            name = author.find("atom:name", ns)
                            if name is not None:
                                authors.append(name.text.strip())
                                
                        authors_val = ", ".join(authors[:3])
                        if len(authors) > 3:
                            authors_val += " et al."
                            
                        try:
                            pub_dt = datetime.strptime(pub_val, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
                        except Exception:
                            pub_dt = datetime.now(pytz.utc)
                            
                        if pub_dt >= cutoff:
                            papers.append({
                                "title": title_val,
                                "authors": authors_val,
                                "abstract": summary_val[:300] + "..." if len(summary_val) > 300 else summary_val,
                                "url": url_val,
                                "published_date": pub_dt.isoformat()
                            })
                    return papers
            except Exception as e:
                logger.error(f"Error fetching arxiv topic '{topic}': {e}")
                return []

        # Run concurrent topic scraper operations
        tasks = [fetch_topic(topic) for topic in topics]
        results = await asyncio.gather(*tasks)
        
        # Deduplicate and flatten
        papers_dict = {}
        for paper_list in results:
            for paper in paper_list:
                papers_dict[paper["url"]] = paper
                
        logger.info(f"ArxivMonitor retrieved {len(papers_dict)} deduplicated papers.")
        return list(papers_dict.values())

    async def score_relevance(self, paper: dict, user_interests: str) -> float:
        system_prompt = "You are a research relevance evaluator for a personal intelligence agent."
        user_message = (
            f"Rate 0.0-1.0 how relevant this paper is to someone with these interests: {user_interests}\n\n"
            f"Paper:\n"
            f"Title: {paper['title']}\n"
            f"Abstract: {paper['abstract'][:200]}\n\n"
            f"Return JSON format: {{\"score\": 0.XX}}"
        )
        
        try:
            plan = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.1)
            score = float(plan.get("score", 0.0))
            return score
        except Exception as e:
            logger.error(f"Error scoring paper relevance: {e}")
            return 0.0
