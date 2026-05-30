"""
Gemini 2.0 Flash async client.
Features: retry with backoff, per-minute rate limiting, JSON mode, singleton.
"""
import asyncio
import json
import time
from collections import deque

import google.generativeai as genai

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0       # seconds; doubles each attempt
MAX_CALLS_PER_MINUTE = 28    # stay safely under Gemini free-tier limits


class GeminiUnavailableError(Exception):
    """Raised when Gemini API fails after all retries."""


class GeminiClient:
    """Async Gemini wrapper with rate limiting, retry, and JSON mode."""

    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._call_times: deque[float] = deque()

    # ── Rate limiting ─────────────────────────────────────────────────

    def _enforce_rate_limit(self) -> float:
        """Return seconds to sleep if rate limit is reached, else 0."""
        now = time.time()
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()
        if len(self._call_times) >= MAX_CALLS_PER_MINUTE:
            sleep = 60.0 - (now - self._call_times[0])
            return max(sleep, 0.0)
        return 0.0

    # ── Core generation ───────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text response. Retries up to MAX_RETRIES times."""
        wait = self._enforce_rate_limit()
        if wait > 0:
            logger.warning(f"[Gemini] Rate limit — sleeping {wait:.1f}s")
            await asyncio.sleep(wait)

        for attempt in range(MAX_RETRIES):
            try:
                model = genai.GenerativeModel(
                    MODEL_NAME,
                    system_instruction=system_prompt,
                )
                gen_config = genai.types.GenerationConfig(temperature=temperature)
                t0 = time.time()

                # Run blocking call in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(
                    model.generate_content,
                    user_message,
                    generation_config=gen_config,
                )
                latency_ms = int((time.time() - t0) * 1000)
                self._call_times.append(time.time())

                text = response.text
                logger.info(f"[Gemini] {len(text)} chars in {latency_ms}ms (attempt {attempt+1})")
                return text

            except Exception as exc:
                logger.warning(f"[Gemini] Attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BASE_RETRY_DELAY * (2 ** attempt))
                else:
                    raise GeminiUnavailableError(
                        f"Gemini unavailable after {MAX_RETRIES} retries. Last error: {exc}"
                    ) from exc
        return ""  # Unreachable; satisfies type checker

    async def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
    ) -> dict:
        """Generate and parse a JSON response. Retries once on parse failure."""
        json_system = (
            system_prompt
            + "\n\nCRITICAL: Your entire response must be valid JSON only. "
            "No markdown. No code fences. No explanation. No preamble. Pure JSON."
        )
        msg = user_message
        for attempt in range(2):
            raw = await self.generate(json_system, msg, temperature=temperature)
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"[Gemini] JSON parse failed attempt {attempt + 1}. Raw: {raw[:200]}")
                msg = (
                    f"Your previous response was not valid JSON:\n{raw}\n\n"
                    "Respond with ONLY valid JSON. Absolutely nothing else."
                )
        logger.error("[Gemini] JSON parse failed after 2 attempts. Returning {}")
        return {}


# Module-level singleton — import and use this everywhere
gemini_client = GeminiClient()
