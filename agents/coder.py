"""Coder agent — code generation, explanation, and debugging."""
from utils.gemini_client import gemini_client
from utils.logger import get_logger

logger = get_logger(__name__)

CODER_SYSTEM_PROMPT = """You are the Coder agent within NEXUS, a personal intelligence system for {user_name}.
Your job is to help with programming tasks: writing code, debugging, explaining concepts, and reviewing code.

When helping with code:
1. Write clean, well-commented, production-quality code
2. Explain your approach briefly
3. Mention any edge cases or limitations
4. Suggest improvements when appropriate
5. Use Python as the default language unless specified otherwise

The user is a 2nd year AI/ML engineering student interested in:
{user_interests}

USER CONTEXT:
{user_context}
"""


class CoderAgent:
    """Generates code, explains programming concepts, and helps with debugging."""

    def __init__(self, user_name: str, user_interests: str) -> None:
        self._user_name = user_name
        self._user_interests = user_interests

    async def generate_code(self, request: str, user_context: str = "") -> str:
        """Generate code based on a user request."""
        logger.info(f"[Coder] Generating code: {request[:80]}")

        system = CODER_SYSTEM_PROMPT.format(
            user_name=self._user_name,
            user_interests=self._user_interests,
            user_context=user_context or "Not available",
        )
        result = await gemini_client.generate(system, request, temperature=0.3)
        logger.info(f"[Coder] Code generated: {len(result)} chars")
        return result

    async def debug_code(self, code: str, error: str, user_context: str = "") -> str:
        """Debug code given an error message."""
        prompt = (
            f"Debug this code:\n```\n{code}\n```\n\n"
            f"Error:\n{error}\n\n"
            "Explain the bug and provide the fixed code."
        )
        system = CODER_SYSTEM_PROMPT.format(
            user_name=self._user_name,
            user_interests=self._user_interests,
            user_context=user_context or "Not available",
        )
        return await gemini_client.generate(system, prompt, temperature=0.2)

    async def explain_concept(self, concept: str, user_context: str = "") -> str:
        """Explain a programming or CS concept."""
        prompt = (
            f"Explain this concept clearly for an AI/ML engineering student: {concept}\n"
            "Include: definition, key points, a simple example, and how it relates to AI/ML."
        )
        system = CODER_SYSTEM_PROMPT.format(
            user_name=self._user_name,
            user_interests=self._user_interests,
            user_context=user_context or "Not available",
        )
        return await gemini_client.generate(system, prompt, temperature=0.5)
