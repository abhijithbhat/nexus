import json
from datetime import datetime, timedelta
import pytz
from utils.config import settings
from utils.logger import get_logger
from utils.gemini_client import GeminiClient
from memory.memory_manager import MemoryManager

logger = get_logger(__name__)

class NexusReflector:
    def __init__(self, memory_manager: MemoryManager, gemini_client: GeminiClient, whatsapp):
        self.memory_manager = memory_manager
        self.gemini_client = gemini_client
        self.whatsapp = whatsapp

    async def run_nightly_reflection(self) -> None:
        logger.info("NexusReflector running nightly self-evaluation...")
        
        # 1. Compile today's activities
        recent_mems = self.memory_manager.vector_store.get_recent(hours=24)
        
        conversations = [m for m in recent_mems if m["metadata"].get("type") == "conversation"]
        research_tasks = [m for m in recent_mems if m["metadata"].get("type") == "research_result"]
        events_scheduled = [m for m in recent_mems if m["metadata"].get("type") == "event"]
        monitor_signals = [m for m in recent_mems if m["metadata"].get("type") == "monitor_signal"]
        
        # Fetch last 3 reflections
        last_reflections = self.memory_manager.knowledge_graph.get_recent_reflections(days=3)
        reflections_summary = []
        for r in last_reflections:
            date_str = r.created_at.strftime("%Y-%m-%d")
            reflections_summary.append(f"{date_str}: {r.content[:150]}...")
        last_3_reflections = "\n".join(reflections_summary) if reflections_summary else "None"
        
        today_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        activity_summary = (
            f"DATE: {today_date}\n"
            f"CONVERSATIONS: {len(conversations)} messages exchanged\n"
            f"RESEARCH TASKS: {len(research_tasks)} completed\n"
            f"EVENTS SCHEDULED: {len(events_scheduled)} registered\n"
            f"MONITOR SIGNALS PROCESSED: {len(monitor_signals)} items saved\n"
            f"RECENT REFLECTIONS (last 3):\n{last_3_reflections}"
        )
        
        system_prompt = (
            "You are NEXUS performing a nightly self-evaluation. Be analytically honest, growth-oriented, and specific. "
            "Vague reflections are useless. Reference actual events from the activity log."
        )
        
        user_message = (
            f"Evaluate today's performance. Activity log:\n"
            f"{activity_summary}\n\n"
            f"Answer all questions:\n"
            f"1. What did I do well today? (specific examples from the log)\n"
            f"2. Where did I fall short, make errors, or give incomplete responses?\n"
            f"3. What did the user seem to need that I did not provide?\n"
            f"4. What new things did I learn about the user today?\n"
            f"5. What should I do differently tomorrow?\n"
            f"6. Are there any proactive things I should do for the user tomorrow morning?\n\n"
            f"Return JSON format:\n"
            f"{{\n"
            f"  \"strengths\": [\"...\"],\n"
            f"  \"weaknesses\": [\"...\"],\n"
            f"  \"user_needs_unmet\": [\"...\"],\n"
            f"  \"new_user_facts\": [{{\"subject\": \"Abhijith\", \"predicate\": \"...\", \"object\": \"...\"}}],\n"
            f"  \"tomorrow_changes\": [\"...\"],\n"
            f"  \"proactive_tomorrow\": [\"...\"]\n"
            f"}}"
        )
        
        try:
            reflection_json = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.4)
            
            # Save reflection block to SQLite
            self.memory_manager.knowledge_graph.add_reflection(activity_summary, reflection_json)
            
            # Save new user preferences/facts to SQLite and ChromaDB
            new_facts = reflection_json.get("new_user_facts", [])
            for fact in new_facts:
                sub = fact.get("subject", settings.user_name)
                pred = fact.get("predicate", "")
                obj = fact.get("object", "")
                if pred and obj:
                    self.memory_manager.knowledge_graph.add_fact(sub, pred, obj, confidence=0.8, source="reflection")
                    await self.memory_manager.remember(f"{sub} {pred} {obj}", "fact", "reflection", importance=0.75)
            
            # Schedule proactive follow-up tasks (naive UTC datetime corresponding to 9:00 AM IST tomorrow)
            proactive_items = reflection_json.get("proactive_tomorrow", [])
            ist_tz = pytz.timezone("Asia/Kolkata")
            tomorrow_ist = datetime.now(ist_tz) + timedelta(days=1)
            tomorrow_9am_ist = tomorrow_ist.replace(hour=9, minute=0, second=0, microsecond=0)
            tomorrow_9am_utc = tomorrow_9am_ist.astimezone(pytz.utc).replace(tzinfo=None)
            
            for task in proactive_items:
                self.memory_manager.knowledge_graph.add_event(
                    title=f"Proactive Checkin: {task}",
                    description="Automatically generated proactive reminder from nightly reflection.",
                    scheduled_at=tomorrow_9am_utc
                )
                
            # Send WhatsApp reflection summary
            weaknesses = reflection_json.get("weaknesses", [])
            changes = reflection_json.get("tomorrow_changes", [])
            proactive = reflection_json.get("proactive_tomorrow", [])
            
            summary_msg = (
                f"🌙 NEXUS nightly check-in\n\n"
                f"Today: {len(conversations)} conversations, {len(monitor_signals)} signals scanned, {len(research_tasks)} tasks completed.\n\n"
                f"I noticed: {weaknesses[0] if weaknesses else 'Systems operating correctly.'}\n\n"
                f"Tomorrow I'll: {changes[0] if changes else 'Continue proactive support.'}\n\n"
                f"Proactively checking in about: {', '.join(proactive) if proactive else 'Upcoming schedule tasks.'}\n\n"
                f"Good night, {settings.user_name}."
            )
            
            self.whatsapp.send_message(settings.user_whatsapp_number, summary_msg)
            logger.info("Nightly self-reflection finished and logged.")
            
        except Exception as e:
            logger.error(f"Error during nightly reflection run: {e}")
