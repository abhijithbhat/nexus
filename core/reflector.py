"""Nightly self-reflection — NEXUS evaluates its day and extracts insights."""
from utils.gemini_client import gemini_client
from utils.config import settings
from memory.memory_manager import MemoryManager
from memory.knowledge_graph import KnowledgeGraph
from connectors.whatsapp import WhatsAppConnector
from utils.logger import get_logger

logger = get_logger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are NEXUS performing a nightly self-reflection for {user_name}.

Review today's interactions and world signals to:
1. Identify patterns in user behavior and preferences
2. Note any important information that should be remembered long-term
3. Suggest proactive actions for tomorrow
4. Evaluate how well you served the user today
5. Identify knowledge gaps to fill

Return ONLY JSON:
{{
    "summary": "Brief summary of today's activities",
    "key_insights": ["insight1", "insight2"],
    "user_patterns": ["pattern1"],
    "proactive_suggestions": ["suggestion1"],
    "self_evaluation": {{
        "helpfulness": 0.X,
        "proactiveness": 0.X,
        "memory_quality": 0.X
    }},
    "tomorrow_priorities": ["priority1"]
}}
"""


class Reflector:
    """Performs nightly self-reflection and stores insights."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        knowledge_graph: KnowledgeGraph,
        whatsapp: WhatsAppConnector,
    ) -> None:
        self._memory = memory_manager
        self._kg = knowledge_graph
        self._whatsapp = whatsapp

    async def run_nightly_reflection(self) -> dict:
        """Execute the nightly reflection cycle."""
        logger.info("[Reflector] Starting nightly reflection...")

        # Gather today's data
        recent = self._memory._vs.get_recent(hours=24)
        conversations = [e for e in recent if e["metadata"].get("type") == "conversation"]
        signals = [e for e in recent if e["metadata"].get("type") == "world_signal"]

        data_summary = (
            f"Conversations today: {len(conversations)}\n"
            f"World signals received: {len(signals)}\n\n"
        )
        for conv in conversations[:10]:
            data_summary += f"- {conv['text'][:150]}\n"
        for sig in signals[:5]:
            data_summary += f"- [signal] {sig['text'][:150]}\n"

        # Generate reflection
        system = REFLECTION_SYSTEM_PROMPT.format(user_name=settings.user_name)
        reflection = await gemini_client.generate_json(system, data_summary, temperature=0.3)

        # Store the reflection
        summary = reflection.get("summary", "No reflection generated")
        self._kg.add_reflection(summary, reflection)
        await self._memory.remember(
            f"Nightly reflection: {summary}",
            type="reflection",
            source="nexus_self",
            importance=0.85,
        )

        # Send summary to user if configured
        if settings.send_reflection_summary:
            brief = (
                "🌙 *NEXUS Nightly Reflection*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 {summary}\n\n"
            )
            insights = reflection.get("key_insights", [])
            if insights:
                brief += "💡 *Key Insights:*\n"
                for insight in insights[:3]:
                    brief += f"  • {insight}\n"
            suggestions = reflection.get("proactive_suggestions", [])
            if suggestions:
                brief += "\n🎯 *Tomorrow's Suggestions:*\n"
                for s in suggestions[:3]:
                    brief += f"  • {s}\n"

            self._whatsapp.send_to_user(brief)

        logger.info("[Reflector] Nightly reflection complete")
        return reflection
