import asyncio
from connectors.web_search import WebSearchConnector
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)

class ResearcherAgent:
    def __init__(self):
        self.web_search = WebSearchConnector()
        self.gemini_client = GeminiClient()

    async def run(self, task: str, context: str) -> str:
        logger.info(f"ResearcherAgent starting task: '{task}'")
        
        # 1. Query web search
        results = await self.web_search.search(task, max_results=5)
        if not results:
            return "No web results found for this research topic."
            
        # 2. Scrape pages in parallel
        scrape_tasks = [self.web_search.fetch_page(res["url"]) for res in results]
        page_contents = await asyncio.gather(*scrape_tasks)
        
        # 3. Assemble sources content
        sources_list = []
        for i, raw_content in enumerate(page_contents):
            res = results[i]
            # Use fallback snippet if scraping didn't yield text
            content_to_use = raw_content.strip() if raw_content.strip() else res["snippet"]
            sources_list.append(
                f"SOURCE {i+1}: {res['title']}\n"
                f"URL: {res['url']}\n"
                f"CONTENT:\n{content_to_use}\n"
                f"================================="
            )
            
        assembled_content = "\n\n".join(sources_list)
        
        system_prompt = (
            "You are an expert research analyst. Synthesize the following web sources into a structured research report. "
            "Focus specifically on the task. Format your response exactly as:\n\n"
            "SUMMARY\n"
            "[2-3 sentence overview]\n\n"
            "KEY FINDINGS\n"
            "• [finding 1]\n"
            "• [finding 2]\n"
            "• [finding 3]\n\n"
            "IMPORTANT DETAILS\n"
            "[1-2 paragraphs of depth]\n\n"
            "RECOMMENDED ACTIONS\n"
            "• [what the user should do with this information]\n\n"
            "SOURCE QUALITY\n"
            "[brief note on source reliability]"
        )
        
        user_message = (
            f"Task: {task}\n\n"
            f"User Context:\n{context[:500]}\n\n"
            f"Sources:\n{assembled_content}"
        )
        
        try:
            report = await self.gemini_client.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.2
            )
            return report
        except Exception as e:
            logger.error(f"Error during ResearcherAgent synthesization: {e}")
            return f"Error compiling research report: {e}"
