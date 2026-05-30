import httpx
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

class HackathonMonitor:
    def __init__(self):
        pass

    async def fetch_upcoming_hackathons(self) -> list[dict]:
        logger.info("HackathonMonitor starting upcoming hackathons fetch...")
        output = []
        
        # 1. Devfolio API fetch
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                )
            }
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                res = await client.get("https://devfolio.co/api/hackathons/?limit=20")
                if res.status_code == 200:
                    data = res.json()
                    
                    hackathons = []
                    if isinstance(data, list):
                        hackathons = data
                    elif isinstance(data, dict):
                        # Attempt to find array of events in standard nesting structures
                        hackathons = data.get("result", []) or data.get("data", []) or data.get("hackathons", [])
                        
                    for item in hackathons:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name", "")
                        theme = item.get("tagline", "")
                        deadline_str = item.get("registrations_close") or item.get("end_date", "")
                        start_str = item.get("starts_at") or item.get("start_date", "")
                        
                        slug = item.get("slug", "")
                        url = item.get("url") or f"https://{slug}.devfolio.co" if slug else "https://devfolio.co"
                        prize = item.get("prize_money") or item.get("prizes_description", "")
                        
                        # Filtering past deadlines
                        now = datetime.utcnow()
                        is_future = True
                        if deadline_str:
                            try:
                                dl = datetime.fromisoformat(deadline_str.replace("Z", "+00:00")).replace(tzinfo=None)
                                if dl < now:
                                    is_future = False
                            except Exception:
                                pass
                                
                        # India or Online whitelist check
                        is_india_or_online = True
                        location = item.get("location", "")
                        if location and "India" not in location and "Online" not in location:
                            is_india_or_online = False
                            
                        if is_future and is_india_or_online:
                            output.append({
                                "name": name,
                                "theme": theme,
                                "registration_deadline": deadline_str,
                                "start_date": start_str,
                                "prize_pool": prize,
                                "url": url
                            })
                            
                    if output:
                        logger.info(f"HackathonMonitor fetched {len(output)} hackathons from Devfolio API.")
                        return output
                        
        except Exception as e:
            logger.warning(f"Error calling Devfolio API: {e}. Reverting to fallback records.")

        # 2. Return fallback listings
        logger.info("Loading fallback hackathons registry.")
        fallback_hackathons = [
            {
                "name": "Smart India Hackathon 2026",
                "theme": "Hardware & software models solving government ministries problems",
                "registration_deadline": (datetime.utcnow().replace(month=10, day=15)).isoformat() + "Z",
                "start_date": (datetime.utcnow().replace(month=11, day=10)).isoformat() + "Z",
                "prize_pool": "Rs. 1,00,000+ per category",
                "url": "https://sih.gov.in"
            },
            {
                "name": "TCS CodeVita Season 13",
                "theme": "Competitive programming algorithms",
                "registration_deadline": (datetime.utcnow().replace(month=9, day=30)).isoformat() + "Z",
                "start_date": (datetime.utcnow().replace(month=10, day=20)).isoformat() + "Z",
                "prize_pool": "$20,000 pool",
                "url": "https://tcscodevita.com"
            }
        ]
        return fallback_hackathons
