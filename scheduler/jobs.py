from datetime import datetime
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")

class JobScheduler:
    def __init__(self, world_monitor, reflector, memory_manager, whatsapp, knowledge_graph):
        self.scheduler = AsyncIOScheduler(timezone=IST)
        self.world_monitor = world_monitor
        self.reflector = reflector
        self.memory_manager = memory_manager
        self.whatsapp = whatsapp
        self.knowledge_graph = knowledge_graph
        self._register_jobs()

    def _register_jobs(self):
        # 1. Morning brief job
        brief_hour, brief_min = map(int, settings.morning_brief_time.split(":"))
        self.scheduler.add_job(
            self.world_monitor.send_morning_brief,
            CronTrigger(hour=brief_hour, minute=brief_min, timezone=IST),
            id="morning_brief",
            name="Morning Intelligence Brief",
            replace_existing=True
        )

        # 2. Tech signals scanning interval job
        self.scheduler.add_job(
            self.world_monitor.run_full_scan,
            "interval", 
            hours=settings.monitor_interval_hours,
            id="background_scan",
            name="Background World Scan",
            replace_existing=True
        )

        # 3. Nightly self-reflection job
        reflect_hour, reflect_min = map(int, settings.reflection_time.split(":"))
        self.scheduler.add_job(
            self.reflector.run_nightly_reflection,
            CronTrigger(hour=reflect_hour, minute=reflect_min, timezone=IST),
            id="nightly_reflection",
            name="Nightly Self-Reflection",
            replace_existing=True
        )

        # 4. Deadline checker interval job
        self.scheduler.add_job(
            self._check_deadlines,
            "interval",
            hours=settings.deadline_check_interval_hours,
            id="deadline_check",
            name="Deadline & Event Reminder",
            replace_existing=True
        )

        # 5. Memory consolidation job (Sunday 3 AM IST)
        self.scheduler.add_job(
            self.memory_manager.consolidate_memory,
            CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=IST),
            id="memory_consolidation",
            name="Weekly Memory Consolidation",
            replace_existing=True
        )
        
        logger.info("All background jobs registered in APScheduler.")

    async def _check_deadlines(self):
        logger.info("Running scheduled deadline check...")
        upcoming = self.knowledge_graph.get_upcoming_events(hours_ahead=24)
        for event in upcoming:
            if event.status == "pending":
                now_utc = datetime.utcnow()
                hours_left = (event.scheduled_at - now_utc).total_seconds() / 3600
                
                msg = (
                    f"⏰ NEXUS Reminder\n\n"
                    f"*{event.title}*\n"
                    f"In {int(hours_left)} hours\n\n"
                    f"{event.description or 'No additional details provided.'}"
                )
                self.whatsapp.send_message(settings.user_whatsapp_number, msg)
                self.knowledge_graph.update_event_status(event.id, "reminded")
                logger.info(f"Fired deadline reminder to user for event: '{event.title}'")

    def start(self):
        self.scheduler.start()
        logger.info("JobScheduler loop started.")
        
    def shutdown(self):
        self.scheduler.shutdown()
        logger.info("JobScheduler loop shut down gracefully.")
