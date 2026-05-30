"""APScheduler job definitions — all recurring background tasks."""
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

IST = pytz.timezone("Asia/Kolkata")


def setup_scheduler(app) -> AsyncIOScheduler:
    """Configure and return the APScheduler with all recurring jobs."""
    scheduler = AsyncIOScheduler(timezone=IST)

    # Parse morning brief time
    brief_hour, brief_minute = map(int, settings.morning_brief_time.split(":"))
    reflect_hour, reflect_minute = map(int, settings.reflection_time.split(":"))

    # ── Morning Brief (daily at configured time) ──────────────────────
    async def morning_brief_job():
        try:
            logger.info("[Scheduler] Running morning brief job...")
            await app.state.world_monitor.send_morning_brief()
        except Exception as exc:
            logger.error(f"[Scheduler] Morning brief failed: {exc}")

    scheduler.add_job(
        morning_brief_job,
        trigger=CronTrigger(hour=brief_hour, minute=brief_minute, timezone=IST),
        id="morning_brief",
        name="Morning Brief",
        replace_existing=True,
    )

    # ── Nightly Reflection (daily at configured time) ─────────────────
    async def reflection_job():
        try:
            logger.info("[Scheduler] Running nightly reflection...")
            await app.state.reflector.run_nightly_reflection()
        except Exception as exc:
            logger.error(f"[Scheduler] Reflection failed: {exc}")

    scheduler.add_job(
        reflection_job,
        trigger=CronTrigger(hour=reflect_hour, minute=reflect_minute, timezone=IST),
        id="nightly_reflection",
        name="Nightly Reflection",
        replace_existing=True,
    )

    # ── World Monitor Scan (every N hours) ────────────────────────────
    async def world_scan_job():
        try:
            logger.info("[Scheduler] Running world monitor scan...")
            await app.state.world_monitor.run_full_scan()
        except Exception as exc:
            logger.error(f"[Scheduler] World scan failed: {exc}")

    scheduler.add_job(
        world_scan_job,
        trigger=IntervalTrigger(hours=settings.monitor_interval_hours),
        id="world_scan",
        name="World Monitor Scan",
        replace_existing=True,
    )

    # ── Deadline Check (every N hours) ────────────────────────────────
    async def deadline_check_job():
        try:
            logger.info("[Scheduler] Checking upcoming deadlines...")
            scheduler_agent = app.state.scheduler_agent
            events = scheduler_agent.get_upcoming_reminders(hours=24)
            if events:
                wa = app.state.whatsapp
                reminder_text = "⏰ *NEXUS Deadline Reminder*\n\n"
                for event in events[:5]:
                    reminder_text += f"📌 {event['title']}\n   🕐 {event['scheduled_at']}\n\n"
                wa.send_to_user(reminder_text)
                logger.info(f"[Scheduler] Sent {len(events)} deadline reminders")
        except Exception as exc:
            logger.error(f"[Scheduler] Deadline check failed: {exc}")

    scheduler.add_job(
        deadline_check_job,
        trigger=IntervalTrigger(hours=settings.deadline_check_interval_hours),
        id="deadline_check",
        name="Deadline Check",
        replace_existing=True,
    )

    # ── Weekly Memory Consolidation (every Sunday at 3 AM IST) ────────
    async def memory_consolidation_job():
        try:
            logger.info("[Scheduler] Running weekly memory consolidation...")
            await app.state.memory_manager.consolidate_memory()
        except Exception as exc:
            logger.error(f"[Scheduler] Memory consolidation failed: {exc}")

    scheduler.add_job(
        memory_consolidation_job,
        trigger=CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=IST),
        id="memory_consolidation",
        name="Weekly Memory Consolidation",
        replace_existing=True,
    )

    logger.info("[Scheduler] All jobs configured:")
    logger.info(f"  • Morning Brief: {settings.morning_brief_time} IST")
    logger.info(f"  • Nightly Reflection: {settings.reflection_time} IST")
    logger.info(f"  • World Scan: every {settings.monitor_interval_hours}h")
    logger.info(f"  • Deadline Check: every {settings.deadline_check_interval_hours}h")
    logger.info(f"  • Memory Consolidation: Sundays 3:00 AM IST")

    return scheduler
