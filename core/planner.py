import json
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)


class TaskPlanner:
    """
    Determines if a task needs single or multi-step execution.
    For complex tasks, returns an ordered list of sub-tasks with agent assignments.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
    
    async def analyze(self, user_input: str, context: str) -> dict:
        """
        Returns:
        {
            "is_complex": bool,
            "steps": [
                {"step": 1, "agent": "researcher", "task": "...", "depends_on": []},
                {"step": 2, "agent": "coder", "task": "...", "depends_on": [1]},
            ]
        }
        """
        system_prompt = (
            "You are a task complexity analyzer for a personal AI agent. "
            "Determine if a user request requires multiple sequential steps or is single-step. "
            "Most requests are single-step. Only flag as complex if truly multi-step."
        )
        
        user_message = (
            f"User request: {user_input}\n"
            f"User context: {context[:300]}\n\n"
            f"Available agents: researcher, coder, scheduler, communicator, gmail, direct\n\n"
            f"Analyze if this is multi-step. A task is multi-step if it requires:\n"
            f"- Research THEN coding (two different actions)\n"
            f"- Writing THEN scheduling\n"
            f"- Research THEN drafting a communication\n"
            f"- Any task with 'and then', 'after that', 'also'\n\n"
            f"Return ONLY JSON:\n"
            f"{{\n"
            f"  \"is_complex\": true|false,\n"
            f"  \"steps\": [\n"
            f"    {{\"step\": 1, \"agent\": \"agent_name\", \"task\": \"specific task description\", \"depends_on\": []}},\n"
            f"    {{\"step\": 2, \"agent\": \"agent_name\", \"task\": \"specific task description\", \"depends_on\": [1]}}\n"
            f"  ]\n"
            f"}}"
        )
        
        try:
            result = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.1)
            if not result.get("is_complex", False):
                result["steps"] = []
            return result
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            return {"is_complex": False, "steps": []}
