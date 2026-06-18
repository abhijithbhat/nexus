from utils.llm_factory import get_primary_client
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class CommunicatorAgent:
    def __init__(self):
        self.llm = get_primary_client()

    async def run(self, task: str, context: str, output_type: str = "message") -> str:
        logger.info(f"CommunicatorAgent starting draft for task: '{task}'")
        
        system_prompt = (
            f"You are a communication expert writing on behalf of {settings.user_name}, "
            f"an AI/ML engineering student in India. Write in first person in their voice — "
            f"professional but not stiff, direct, warm.\n"
            f"Output type: {output_type} (message/email/report/document)\n"
            f"Do not include any meta-commentary. Return only the drafted content."
        )
        
        user_message = (
            f"Task: {task}\n"
            f"Context: {context[:400]}"
        )
        
        try:
            response = await self.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.8
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error in CommunicatorAgent: {e}")
            return f"Error generating communication draft: {e}"
