"""Hackathon listings monitor — Devfolio and similar platforms."""
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)


class HackathonMonitor:
    """Monitors hackathon platforms for upcoming events."""

    async def fetch_hackathons(self) -> list[dict]:
        """Fetch upcoming hackathons. Uses Devfolio GraphQL API."""
        hackathons = []

        # Devfolio GraphQL API
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                query = {
                    "operationName": "SearchHackathons",
                    "variables": {"limit": 10, "offset": 0, "type": "default"},
                    "query": """
                        query SearchHackathons($limit: Int, $offset: Int, $type: String) {
                            search_hackathons(limit: $limit, offset: $offset, type: $type) {
                                hackathons {
                                    name
                                    tagline
                                    slug
                                    starts_at
                                    ends_at
                                    hackathon_setting {
                                        is_online
                                    }
                                }
                            }
                        }
                    """,
                }
                resp = await client.post(
                    "https://api.devfolio.co/api/search/hackathons",
                    json=query,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for h in data.get("data", {}).get("search_hackathons", {}).get("hackathons", []):
                        hackathons.append({
                            "title": h.get("name", ""),
                            "summary": h.get("tagline", "")[:200],
                            "link": f"https://devfolio.co/hackathons/{h.get('slug', '')}",
                            "starts_at": h.get("starts_at", ""),
                            "ends_at": h.get("ends_at", ""),
                            "online": h.get("hackathon_setting", {}).get("is_online", False),
                            "type": "hackathon",
                        })
                    logger.info(f"[Hackathon] Found {len(hackathons)} hackathons from Devfolio")
                else:
                    logger.warning(f"[Hackathon] Devfolio API returned {resp.status_code}")
        except Exception as exc:
            logger.warning(f"[Hackathon] Devfolio fetch failed: {exc}")

        return hackathons
