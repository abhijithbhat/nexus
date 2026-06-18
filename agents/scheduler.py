from datetime import datetime, timedelta
import pytz
from utils.llm_factory import get_primary_client
from utils.logger import get_logger
from memory.memory_manager import MemoryManager

logger = get_logger(__name__)


class SchedulerAgent:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.llm = get_primary_client()

    async def run(self, task: str, context: str) -> str:
        logger.info(f"SchedulerAgent starting task: '{task}'")
        
        task_lower = task.lower()
        
        # Detect intent: cancel/delete vs create
        cancel_words = ["cancel", "delete", "remove", "clear", "drop"]
        if any(word in task_lower for word in cancel_words):
            return await self._handle_cancel(task, context)
        
        # Check/list intent
        list_words = ["list", "show", "what", "upcoming", "my events", "my schedule"]
        if any(word in task_lower for word in list_words):
            return await self._handle_list(task, context)
        
        # Default: create event
        return await self._handle_create(task, context)

    async def _handle_create(self, task: str, context: str) -> str:
        """Create a new scheduled event."""
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
            details = await self.llm.generate_json(system_prompt, user_message)
            
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
            
            # Optionally sync to Google Calendar
            gcal_status = ""
            try:
                from connectors.gcalendar import GoogleCalendarConnector
                gcal = GoogleCalendarConnector()
                if gcal.is_available:
                    end_dt = dt_utc + timedelta(hours=1)
                    gcal.create_event(
                        title=title,
                        start_dt=dt_utc,
                        end_dt=end_dt,
                        description=description
                    )
                    gcal_status = "\n📆 Also added to Google Calendar"
            except Exception as e:
                logger.warning(f"Google Calendar sync failed (non-critical): {e}")
            
            return f"✅ Scheduled: {title}\n📅 {dt_ist_str}\n⏰ Reminder {reminder_min} min before{gcal_status}"
            
        except Exception as e:
            logger.error(f"Error in SchedulerAgent create: {e}")
            return f"Error parsing or scheduling event: {e}"

    async def _handle_cancel(self, task: str, context: str) -> str:
        """Cancel an existing event by searching for it by title."""
        # Extract what event to cancel
        system_prompt = "Extract the event title the user wants to cancel. Return JSON only."
        user_message = (
            f"The user wants to cancel this: '{task}'\n\n"
            f"Return JSON: {{\"search_title\": \"the event name to search for\"}}"
        )
        
        try:
            details = await self.llm.generate_json(system_prompt, user_message)
            search_title = details.get("search_title", "")
            
            if not search_title:
                return "❌ Could not determine which event to cancel. Please specify the event name."
            
            # Search for matching events
            events = self.memory_manager.knowledge_graph.find_events_by_title(search_title)
            
            if not events:
                return f"❌ No active events found matching '{search_title}'."
            
            # Cancel all matching events
            ist_tz = pytz.timezone("Asia/Kolkata")
            cancelled = []
            for event in events:
                self.memory_manager.knowledge_graph.cancel_event(event.id)
                dt_ist = event.scheduled_at.replace(tzinfo=pytz.utc).astimezone(ist_tz)
                cancelled.append(f"• {event.title} — {dt_ist.strftime('%Y-%m-%d %I:%M %p')}")
            
            # Save to memory
            await self.memory_manager.remember(
                text=f"Cancelled events: {', '.join([e.title for e in events])}",
                type="event_cancellation",
                source="scheduler",
                importance=0.5
            )
            
            result = f"🗑️ Cancelled {len(cancelled)} event(s):\n" + "\n".join(cancelled)
            return result
            
        except Exception as e:
            logger.error(f"Error in SchedulerAgent cancel: {e}")
            return f"Error cancelling event: {e}"

    async def _handle_list(self, task: str, context: str) -> str:
        """List upcoming events."""
        try:
            events = self.memory_manager.knowledge_graph.get_upcoming_events(hours_ahead=168)  # 7 days
            
            if not events:
                return "📭 No upcoming events in the next 7 days."
            
            ist_tz = pytz.timezone("Asia/Kolkata")
            lines = [f"📅 **Upcoming Events** ({len(events)}):\n"]
            for event in events:
                dt_ist = event.scheduled_at.replace(tzinfo=pytz.utc).astimezone(ist_tz)
                lines.append(
                    f"• {event.title}\n"
                    f"  📅 {dt_ist.strftime('%a, %b %d at %I:%M %p')}\n"
                    f"  Status: {event.status}"
                )
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error listing events: {e}")
            return f"Error listing events: {e}"
