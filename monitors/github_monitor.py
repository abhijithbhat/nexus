"""GitHub trending repos monitor."""
import asyncio
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)

GITHUB_TRENDING_URL = "https://api.github.com/search/repositories"

SEARCH_QUERIES = [
    "multi-agent AI",
    "langgraph",
    "autonomous agent",
    "AI agent framework",
    "LLM tool use",
]


class GithubMonitor:
    """Monitors GitHub for trending repositories matching user interests."""

    async def fetch_trending(self, max_results: int = 10) -> list[dict]:
        """Fetch trending repos from GitHub matching interest queries."""
        all_repos = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for query in SEARCH_QUERIES:
                try:
                    resp = await client.get(
                        GITHUB_TRENDING_URL,
                        params={
                            "q": f"{query} language:python",
                            "sort": "stars",
                            "order": "desc",
                            "per_page": 5,
                        },
                        headers={"Accept": "application/vnd.github.v3+json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("items", [])[:3]:
                        all_repos.append({
                            "title": item.get("full_name", ""),
                            "description": (item.get("description") or "")[:200],
                            "link": item.get("html_url", ""),
                            "stars": item.get("stargazers_count", 0),
                            "language": item.get("language", ""),
                            "type": "github_repo",
                        })
                    await asyncio.sleep(1.0)  # Be polite to GitHub API
                except Exception as exc:
                    logger.warning(f"[GitHub] Failed for '{query}': {exc}")

        # Deduplicate by repo name and sort by stars
        seen = set()
        unique = []
        for r in all_repos:
            if r["title"] not in seen:
                seen.add(r["title"])
                unique.append(r)
        unique.sort(key=lambda x: x["stars"], reverse=True)
        logger.info(f"[GitHub] Found {len(unique)} trending repos")
        return unique[:max_results]
