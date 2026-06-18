import json
from datetime import datetime
from utils.gemini_client import GeminiClient
from memory.knowledge_graph import KnowledgeGraph
from utils.logger import get_logger

logger = get_logger(__name__)


class GoalTracker:
    """
    Monitors conversations for milestone events related to user's goals.
    Detects: task completion, progress updates, new goals, goal changes.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, gemini_client: GeminiClient, user_goals: str):
        self.kg = knowledge_graph
        self.gemini_client = gemini_client
        self.user_goals = user_goals
    
    async def check_for_milestone(self, conversation: str) -> dict | None:
        """
        Analyze a conversation for milestone events.
        Returns milestone dict or None.
        """
        # Skip very short or trivial messages
        if len(conversation) < 30:
            return None
        
        system_prompt = (
            "You are a goal progress analyzer for a personal AI. "
            "Detect if the user's message indicates goal progress, completion, or a new achievement. "
            "Be conservative — only flag clear milestones, not vague mentions."
        )
        user_message = (
            f"User's current goals: {self.user_goals}\n\n"
            f"Recent conversation:\n{conversation}\n\n"
            f"Did the user mention completing, progressing toward, or achieving any goal?\n"
            f"Return JSON:\n"
            f"{{\n"
            f"  \"milestone_detected\": true|false,\n"
            f"  \"goal\": \"which goal\",\n"
            f"  \"achievement\": \"what was achieved\",\n"
            f"  \"celebration_message\": \"a short congratulatory message from NEXUS\"\n"
            f"}}"
        )
        
        try:
            result = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.2)
            if result.get("milestone_detected", False):
                self.kg.add_fact(
                    subject="NEXUS_GOALS",
                    predicate="milestone_achieved",
                    object=f"{result.get('goal', '')} — {result.get('achievement', '')} ({datetime.utcnow().date()})",
                    confidence=0.9,
                    source="goal_tracker"
                )
                return result
        except Exception as e:
            logger.error(f"Goal tracking error: {e}")
        return None
