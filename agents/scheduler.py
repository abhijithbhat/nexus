"""Scheduler agent — extracts dates/events from conversations and stores them."""
from datetime import datetime

from utils.gemini_client import gemini_client
from memory.knowledge_graph import KnowledgeGraph
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEDULER_SYSTEM_PROMPT = """You are the Scheduler agent within NEXUS.
Your job is to extract events, deadlines, and reminders from user messages.

When analyzing a message:
1. Identify any dates, times, or deadlines mentioned
2. Determine event titles and descriptions
3. Classify urgency (low, medium, high, critical)

Return ONLY JSON:
{{
    "events": [
        {{
            "title": "Event name",
            "description": "Details",
            "datetime": "YYYY-MM-DDTHH:MM:SS",
            "urgency": "low|medium|high|critical"
        }}
    ],
    "has_events": true/false
}}

If no events/dates are found, return: {{"events": [], "has_events": false}}
Current date/time: {current_time}
"""


class SchedulerAgent:
    """Extracts events and deadlines from messages, stores them in the knowledge graph."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self._kg = knowledge_graph

    async def extract_and_store_events(self, message: str) -> list[dict]:
        """Extract events from a message and store them. Returns list of stored events."""
        now = datetime.utcnow().isoformat()
        system = SCHEDULER_SYSTEM_PROMPT.format(current_time=now)

        result = await gemini_client.generate_json(system, message, temperature=0.1)

        if not result.get("has_events"):
            return []

        stored = []
        for event in result.get("events", []):
            try:
                title = event.get("title", "Untitled Event")
                desc = event.get("description", "")
                dt_str = event.get("datetime", "")
                if dt_str:
                    scheduled_at = datetime.fromisoformat(dt_str)
                    event_id = self._kg.add_event(title, desc, scheduled_at)
                    stored.append({"id": event_id, "title": title, "scheduled_at": dt_str})
                    logger.info(f"[Scheduler] Stored event: '{title}' at {dt_str}")
            except Exception as exc:
                logger.warning(f"[Scheduler] Failed to store event: {exc}")

        return stored

    def get_upcoming_reminders(self, hours: int = 24) -> list[dict]:
        """Get upcoming events for reminder delivery."""
        events = self._kg.get_upcoming_events(hours_ahead=hours)
        return [
            {
                "id": e.id,
                "title": e.title,
                "description": e.description,
                "scheduled_at": str(e.scheduled_at),
                "status": e.status,
            }
            for e in events
        ]
