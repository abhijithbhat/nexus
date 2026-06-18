"""
LLM client factory — returns the best available LLM client.

Strategy:
- Groq (Llama 3.3 70B) is the PRIMARY client when available (30 RPM free, instant)
- Gemini is the FALLBACK when Groq is not configured

This ensures NEXUS works well on the Gemini free tier (5 RPM, 20/day)
by routing almost all calls through Groq instead.
"""

from utils.groq_client import GroqClient
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)

# Cached singleton clients
_groq = None
_gemini = None


def get_primary_client():
    """Returns Groq if available, else Gemini. Use for ALL LLM calls."""
    global _groq, _gemini
    
    if _groq is None:
        _groq = GroqClient()
    if _gemini is None:
        _gemini = GeminiClient()
    
    if _groq.is_available:
        return _groq
    return _gemini


def get_gemini_client():
    """Returns Gemini client directly (for cases where you specifically need it)."""
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini


def get_groq_client():
    """Returns Groq client (may not be available)."""
    global _groq
    if _groq is None:
        _groq = GroqClient()
    return _groq
