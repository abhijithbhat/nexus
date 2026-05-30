"""High-level task planner — breaks complex requests into actionable steps."""
from utils.gemini_client import gemini_client
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Planner module within NEXUS.
You break complex user requests into actionable, ordered steps.

For each step, specify:
1. What needs to be done
2. Which agent should handle it (researcher, coder, scheduler, communicator)
3. Any dependencies on prior steps

Return ONLY JSON:
{{
    "goal": "High-level goal description",
    "steps": [
        {{
            "step": 1,
            "action": "What to do",
            "agent": "researcher|coder|scheduler|communicator",
            "depends_on": [],
            "priority": "high|medium|low"
        }}
    ],
    "estimated_complexity": "simple|moderate|complex"
}}
"""


class Planner:
    """Breaks complex requests into actionable multi-step plans."""

    async def create_plan(self, request: str, user_context: str = "") -> dict:
        """Create an execution plan for a complex request."""
        prompt = (
            f"User request: {request}\n"
            f"User context: {user_context}\n\n"
            "Create a step-by-step execution plan."
        )
        result = await gemini_client.generate_json(PLANNER_SYSTEM_PROMPT, prompt, temperature=0.2)
        logger.info(f"[Planner] Created plan: {result.get('goal', 'unknown')} "
                     f"({len(result.get('steps', []))} steps)")
        return result
