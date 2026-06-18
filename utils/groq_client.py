import json
import time
import asyncio
from collections import deque
from groq import AsyncGroq
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqUnavailableError(Exception):
    pass


class GroqClient:
    """
    Fast utility LLM client using Groq (Llama 3.3 70B).
    Used for: routing, formatting, entity extraction, importance scoring,
    goal tracking, task planning — all the "utility" calls.
    Gemini stays reserved for the core intelligence (conversation, research).
    
    Groq free tier: 30 RPM, 14,400 req/day, ~6,000 tokens/min.
    """
    
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2.0
    MAX_RPM = 25  # Stay safely under Groq's 30 RPM free tier limit
    MODEL = "llama-3.3-70b-versatile"
    
    _calls = deque()
    _rate_limit_lock = asyncio.Lock()
    
    def __init__(self):
        api_key = settings.groq_api_key
        if not api_key or "your_groq" in api_key.lower():
            logger.warning("GROQ_API_KEY is not configured. Groq calls will fail.")
            self._client = None
        else:
            self._client = AsyncGroq(api_key=api_key)
            logger.info(f"GroqClient initialized with model: {self.MODEL}")
    
    @property
    def is_available(self) -> bool:
        return self._client is not None
    
    @classmethod
    async def _check_rate_limit(cls):
        async with cls._rate_limit_lock:
            now = time.time()
            while cls._calls and now - cls._calls[0] > 60:
                cls._calls.popleft()
            
            if len(cls._calls) >= cls.MAX_RPM:
                sleep_time = 60 - (now - cls._calls[0]) + 0.1
                if sleep_time > 0:
                    logger.warning(f"Groq rate limit reached. Sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                now = time.time()
                while cls._calls and now - cls._calls[0] > 60:
                    cls._calls.popleft()
            cls._calls.append(time.time())
    
    async def generate(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
        """Generate a text response via Groq."""
        if not self._client:
            raise GroqUnavailableError("Groq client not configured")
        
        await self._check_rate_limit()
        
        attempt = 0
        while attempt < self.MAX_RETRIES:
            start_time = time.time()
            try:
                response = await self._client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=2048
                )
                latency = (time.time() - start_time) * 1000
                text = response.choices[0].message.content
                logger.info(f"Groq call successful: latency={latency:.0f}ms, tokens={response.usage.total_tokens}")
                return text
            except Exception as e:
                attempt += 1
                delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Groq call failed (attempt {attempt}/{self.MAX_RETRIES}): {e}. Retrying in {delay}s...")
                if attempt >= self.MAX_RETRIES:
                    raise GroqUnavailableError(f"Groq API unavailable after {self.MAX_RETRIES} retries: {e}")
                await asyncio.sleep(delay)
        
        raise GroqUnavailableError("Groq call failed")
    
    async def generate_json(self, system_prompt: str, user_message: str, temperature: float = 0.1) -> dict:
        """Generate a JSON response via Groq."""
        if not self._client:
            raise GroqUnavailableError("Groq client not configured")
        
        await self._check_rate_limit()
        
        full_system = system_prompt + "\nIMPORTANT: Respond ONLY with valid JSON. No markdown. No explanation. No code blocks."
        
        attempt = 0
        while attempt < self.MAX_RETRIES:
            start_time = time.time()
            try:
                response = await self._client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                latency = (time.time() - start_time) * 1000
                text = response.choices[0].message.content
                logger.info(f"Groq JSON call successful: latency={latency:.0f}ms")
                
                # Parse JSON
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    else:
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                
                return json.loads(cleaned)
            except json.JSONDecodeError as je:
                attempt += 1
                logger.warning(f"Groq JSON parse failed: {je}. Retrying...")
                if attempt >= self.MAX_RETRIES:
                    raise ValueError(f"Failed to get valid JSON from Groq: {je}")
                await asyncio.sleep(self.BASE_RETRY_DELAY)
            except Exception as e:
                attempt += 1
                delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Groq JSON call failed: {e}. Retrying in {delay}s...")
                if attempt >= self.MAX_RETRIES:
                    raise GroqUnavailableError(f"Groq API unavailable: {e}")
                await asyncio.sleep(delay)
        
        raise GroqUnavailableError("Groq JSON call failed")
