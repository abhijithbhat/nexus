"""Researcher agent — web research, synthesis, and fact extraction."""
from utils.gemini_client import gemini_client
from connectors.web_search import WebSearchConnector
from utils.logger import get_logger

logger = get_logger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are the Researcher agent within NEXUS, a personal intelligence system.
Your job is to find information, synthesize it, and present clear, actionable summaries.

When given a research query:
1. Analyze what information is needed
2. Use the search results provided to formulate a comprehensive answer
3. Cite sources when possible
4. Highlight key takeaways and action items
5. Be concise but thorough

USER CONTEXT:
{user_context}
"""


class ResearcherAgent:
    """Performs web research and synthesizes findings."""

    def __init__(self, web_search: WebSearchConnector) -> None:
        self._search = web_search

    async def research(self, query: str, user_context: str = "") -> str:
        """Execute a research query: search the web, then synthesize with Gemini."""
        logger.info(f"[Researcher] Starting research: {query[:80]}")

        # Step 1: Search the web
        search_results = await self._search.search_and_summarize(query)

        # Step 2: Synthesize with Gemini
        prompt = (
            f"Research query: {query}\n\n"
            f"Search results:\n{search_results}\n\n"
            "Synthesize a clear, comprehensive answer. Include:\n"
            "- Key findings\n"
            "- Relevant details\n"
            "- Action items if applicable\n"
            "Keep it concise (under 800 words)."
        )

        system = RESEARCHER_SYSTEM_PROMPT.format(user_context=user_context or "Not available")
        result = await gemini_client.generate(system, prompt, temperature=0.5)
        logger.info(f"[Researcher] Research complete: {len(result)} chars")
        return result

    async def quick_answer(self, question: str, user_context: str = "") -> str:
        """Answer a question using Gemini's knowledge, without web search."""
        system = RESEARCHER_SYSTEM_PROMPT.format(user_context=user_context or "Not available")
        result = await gemini_client.generate(system, question, temperature=0.5)
        return result
