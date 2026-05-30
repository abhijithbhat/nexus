import httpx
from html.parser import HTMLParser
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)

class GithubHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.repos = []
        self.current_repo = None
        self.capture_desc = False
        self.capture_lang = False
        self.capture_stars = False
        self.collect_name = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        tag_class = attrs_dict.get("class", "")
        
        if tag == "article" and "Box-row" in tag_class:
            self.current_repo = {"name": "", "description": "", "stars": "", "url": "", "language": ""}
            
        if self.current_repo:
            if tag == "h2" or tag == "h1":
                self.collect_name = True
            elif tag == "a" and self.collect_name:
                href = attrs_dict.get("href", "")
                self.current_repo["url"] = f"https://github.com{href}"
                self.current_repo["name"] = href.strip("/")
            elif tag == "p":
                self.capture_desc = True
            elif tag == "span" and attrs_dict.get("itemprop") == "programmingLanguage":
                self.capture_lang = True
            elif tag == "a" and "muted" in tag_class and "href" in attrs_dict and "stargazers" in attrs_dict["href"]:
                self.capture_stars = True

    def handle_endtag(self, tag):
        if tag == "article" and self.current_repo:
            # Cleanup whitespace
            self.current_repo["description"] = " ".join(self.current_repo["description"].split())
            self.repos.append(self.current_repo)
            self.current_repo = None
        elif tag == "h2" or tag == "h1":
            self.collect_name = False
        elif tag == "p":
            self.capture_desc = False
        elif tag == "span":
            self.capture_lang = False
        elif tag == "a":
            self.capture_stars = False

    def handle_data(self, data):
        if self.current_repo:
            if self.capture_desc:
                self.current_repo["description"] += data
            elif self.capture_lang:
                self.current_repo["language"] = data.strip()
            elif self.capture_stars:
                cleaned = data.strip().replace(",", "")
                if cleaned:
                    self.current_repo["stars"] = cleaned

class GitHubMonitor:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def fetch_trending(self, language: str = "python", since: str = "daily") -> list[dict]:
        logger.info(f"GitHubMonitor fetching trending for language: {language}, since: {since}")
        url = f"https://github.com/trending/{language}?since={since}"
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            )
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                res = await client.get(url)
                if res.status_code != 200:
                    logger.warning(f"GitHub trending returned status code {res.status_code}")
                    return []
                    
                parser = GithubHTMLParser()
                parser.feed(res.text)
                
                # Limit to 10 repos
                repos = parser.repos[:10]
                logger.info(f"GitHubMonitor successfully parsed {len(repos)} trending repositories.")
                return repos
        except Exception as e:
            logger.error(f"Error fetching GitHub trending: {e}")
            return []

    async def score_relevance(self, repo: dict, user_interests: str) -> float:
        system_prompt = "You are a repository relevance evaluator for a personal intelligence agent."
        user_message = (
            f"Rate 0.0-1.0 how relevant this GitHub repository is to someone with these interests: {user_interests}\n\n"
            f"Repository:\n"
            f"Name: {repo['name']}\n"
            f"Description: {repo['description']}\n"
            f"Language: {repo['language']}\n"
            f"Stars: {repo['stars']}\n\n"
            f"Return JSON format: {{\"score\": 0.XX}}"
        )
        
        try:
            plan = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.1)
            score = float(plan.get("score", 0.0))
            return score
        except Exception as e:
            logger.error(f"Error scoring repository relevance: {e}")
            return 0.0
