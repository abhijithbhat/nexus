"""Communicator agent — message drafting and email composition."""
from utils.gemini_client import gemini_client
from utils.logger import get_logger

logger = get_logger(__name__)

COMMUNICATOR_SYSTEM_PROMPT = """You are the Communicator agent within NEXUS.
Your job is to draft messages, emails, and communications on behalf of {user_name}.

Guidelines:
1. Match the tone to the context (formal for emails, casual for WhatsApp)
2. Be concise but thorough
3. Always be professional and helpful
4. Personalize based on what you know about the user

USER CONTEXT:
{user_context}
"""


class CommunicatorAgent:
    """Drafts messages, emails, and communications."""

    def __init__(self, user_name: str) -> None:
        self._user_name = user_name

    async def draft_message(self, request: str, user_context: str = "") -> str:
        """Draft a message based on user request."""
        system = COMMUNICATOR_SYSTEM_PROMPT.format(
            user_name=self._user_name,
            user_context=user_context or "Not available",
        )
        prompt = f"Draft a message based on this request: {request}"
        return await gemini_client.generate(system, prompt, temperature=0.6)

    async def draft_email(self, to: str, subject: str, context: str, user_context: str = "") -> str:
        """Draft a professional email."""
        system = COMMUNICATOR_SYSTEM_PROMPT.format(
            user_name=self._user_name,
            user_context=user_context or "Not available",
        )
        prompt = (
            f"Draft a professional email:\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Context: {context}\n\n"
            "Include: greeting, body, and sign-off."
        )
        return await gemini_client.generate(system, prompt, temperature=0.5)

    async def summarize_for_brief(self, items: list[dict]) -> str:
        """Summarize a list of items into a morning brief format."""
        if not items:
            return "No new items to report."

        items_text = "\n".join(
            f"- [{item.get('type', '?')}] {item.get('title', item.get('text', ''))}"
            for item in items[:15]
        )
        system = COMMUNICATOR_SYSTEM_PROMPT.format(
            user_name=self._user_name,
            user_context="Morning brief preparation",
        )
        prompt = (
            f"Create a concise morning brief for {self._user_name} from these items:\n"
            f"{items_text}\n\n"
            "Format: emoji-rich, scannable, with action items highlighted."
        )
        return await gemini_client.generate(system, prompt, temperature=0.6)
