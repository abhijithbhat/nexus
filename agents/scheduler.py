from datetime import datetime
import pytz
from utils.gemini_client import GeminiClient
from utils.logger import get_logger
from memory.memory_manager import MemoryManager

logger = get_logger(__name__)

class SchedulerAgent:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.gemini_client = GeminiClient()

    async def run(self, task: str, context: str) -> str:
        logger.info(f"SchedulerAgent starting task: '{task}'")
        
        # Determine local timezone (IST) current time
        ist_tz = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist_tz)
        now_iso = now_ist.isoformat()
        
        system_prompt = "Extract scheduling information from natural language."
        user_message = (
            f"Extract from: '{task}'. Return JSON:\n"
            f"{{\n"
            f"  \"title\": \"event name\",\n"
            f"  \"description\": \"brief description\",\n"
            f"  \"scheduled_datetime\": \"ISO 8601 datetime\",\n"
            f"  \"reminder_minutes_before\": 30\n"
            f"}}\n"
            f"Note: Current datetime is {now_iso}. Use IST timezone (+05:30) for relative dates (e.g. tomorrow, next Saturday)."
        )
        
        try:
            details = await self.gemini_client.generate_json(system_prompt, user_message)
            
            title = details.get("title", "Event")
            description = details.get("description", "")
            dt_str = details.get("scheduled_datetime")
            reminder_min = details.get("reminder_minutes_before", 30)
            
            if not dt_str:
                raise ValueError("Could not extract scheduled_datetime from request.")
                
            # Parse datetime
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(dt_str)
            
            # Save standard UTC naive datetime to DB
            if dt.tzinfo is not None:
                dt_utc = dt.astimezone(pytz.utc).replace(tzinfo=None)
            else:
                dt_utc = dt
                
            # Insert into database
            event_id = self.memory_manager.knowledge_graph.add_event(
                title=title,
                description=description,
                scheduled_at=dt_utc
            )
            
            # Save semantic entry
            dt_ist = dt_utc.replace(tzinfo=pytz.utc).astimezone(ist_tz)
            dt_ist_str = dt_ist.strftime("%Y-%m-%d %I:%M %p")
            
            memory_txt = f"Event scheduled: '{title}' on {dt_ist_str} (IST). Description: {description}"
            await self.memory_manager.remember(
                text=memory_txt,
                type="event",
                source="scheduler",
                importance=0.8
            )
            
            return f"✅ Scheduled: {title}\n📅 {dt_ist_str}\n⏰ Reminder {reminder_min} min before"
            
        except Exception as e:
            logger.error(f"Error in SchedulerAgent: {e}")
            return f"Error parsing or scheduling event: {e}"
