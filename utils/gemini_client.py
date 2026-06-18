import json
import time
import asyncio
from collections import deque
import google.generativeai as genai
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded singleton to avoid circular imports
_usage_tracker = None

def _get_usage_tracker():
    global _usage_tracker
    if _usage_tracker is None:
        from utils.usage_tracker import UsageTracker
        _usage_tracker = UsageTracker()
    return _usage_tracker

class GeminiUnavailableError(Exception):
    pass

class GeminiClient:
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2.0
    
    _calls = deque()
    _rate_limit_lock = asyncio.Lock()
    
    def __init__(self):
        self.MODEL_NAME = settings.gemini_model
        api_key = settings.gemini_api_key
        if not api_key or "your_gemini_api" in api_key:
            logger.warning("GEMINI_API_KEY is not configured or is a placeholder. Gemini calls will fail.")
        genai.configure(api_key=api_key)
        
    @classmethod
    async def _check_rate_limit(cls):
        async with cls._rate_limit_lock:
            now = time.time()
            # Remove calls older than 60 seconds
            while cls._calls and now - cls._calls[0] > 60:
                cls._calls.popleft()
            
            if len(cls._calls) >= 30:
                sleep_time = 60 - (now - cls._calls[0]) + 0.1
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached (>30 RPM). Sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                # Recalculate after sleeping
                now = time.time()
                while cls._calls and now - cls._calls[0] > 60:
                    cls._calls.popleft()
            cls._calls.append(time.time())

    async def generate(self, system_prompt: str, user_message: str, history: list[dict] = None, temperature: float = 0.7) -> str:
        await self._check_rate_limit()
        
        contents = []
        if history:
            for msg in history:
                role = msg.get("role")
                if role == "assistant":
                    role = "model"
                contents.append({
                    "role": role,
                    "parts": [msg.get("content", "")]
                })
        contents.append({
            "role": "user",
            "parts": [user_message]
        })
        
        model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            system_instruction=system_prompt
        )
        
        config = genai.types.GenerationConfig(
            temperature=temperature
        )
        
        attempt = 0
        while attempt < self.MAX_RETRIES:
            start_time = time.time()
            try:
                response = await model.generate_content_async(
                    contents=contents,
                    generation_config=config
                )
                latency = (time.time() - start_time) * 1000
                text = response.text
                
                logger.info(
                    f"Gemini Call successful: model={self.MODEL_NAME}, "
                    f"prompt_len={len(user_message)}, response_len={len(text)}, "
                    f"latency={latency:.2f}ms"
                )
                # Record usage
                try:
                    _get_usage_tracker().record_call(len(user_message), len(text), self.MODEL_NAME)
                except Exception:
                    pass
                return text
            except Exception as e:
                attempt += 1
                latency = (time.time() - start_time) * 1000
                delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"Gemini Call failed (attempt {attempt}/{self.MAX_RETRIES}): {e}. "
                    f"Latency={latency:.2f}ms. Retrying in {delay}s..."
                )
                if attempt >= self.MAX_RETRIES:
                    logger.error("All Gemini retries exhausted.")
                    raise GeminiUnavailableError(f"Gemini API unavailable: {e}")
                await asyncio.sleep(delay)
        
        raise GeminiUnavailableError("Gemini call failed")

    async def generate_json(self, system_prompt: str, user_message: str, temperature: float = 0.1) -> dict:
        json_system_prompt = system_prompt + "\nIMPORTANT: Respond ONLY with valid JSON. No markdown. No explanation. No code blocks."
        
        await self._check_rate_limit()
        model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            system_instruction=json_system_prompt
        )
        
        config = genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json"
        )
        
        async def make_call(message_body):
            attempt = 0
            while attempt < self.MAX_RETRIES:
                start_time = time.time()
                try:
                    response = await model.generate_content_async(
                        contents=[{"role": "user", "parts": [message_body]}],
                        generation_config=config
                    )
                    latency = (time.time() - start_time) * 1000
                    text = response.text
                    logger.info(
                        f"Gemini JSON Call successful: model={self.MODEL_NAME}, "
                        f"latency={latency:.2f}ms"
                    )
                    # Record usage
                    try:
                        _get_usage_tracker().record_call(len(message_body), len(text), self.MODEL_NAME)
                    except Exception:
                        pass
                    return text
                except Exception as e:
                    attempt += 1
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Gemini JSON call failed: {e}. Retrying...")
                    if attempt >= self.MAX_RETRIES:
                        raise GeminiUnavailableError(f"Gemini API unavailable: {e}")
                    await asyncio.sleep(delay)
            raise GeminiUnavailableError("Gemini call failed")

        response_text = await make_call(user_message)
        
        def clean_and_parse(text):
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

        try:
            return clean_and_parse(response_text)
        except json.JSONDecodeError as je:
            logger.warning(f"JSON parsing failed on response: {response_text}. Retrying with warning prompt...")
            retry_message = f"Your previous response was not valid JSON. Error: {je}. Try again. Original request: {user_message}"
            response_text = await make_call(retry_message)
            try:
                return clean_and_parse(response_text)
            except json.JSONDecodeError as je2:
                logger.error(f"JSON parsing failed again: {je2}. Response was: {response_text}")
                raise ValueError(f"Failed to generate valid JSON: {je2}")
