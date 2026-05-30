"""
NEXUS — Proactive Personal Intelligence Agent
Main FastAPI application with full lifecycle management.

Run: uvicorn main:app --reload --port 8000
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from utils.config import settings
from utils.logger import get_logger
from models.database import create_tables
from memory.vector_store import VectorStore
from memory.knowledge_graph import KnowledgeGraph
from memory.memory_manager import MemoryManager
from connectors.whatsapp import WhatsAppConnector
from connectors.web_search import WebSearchConnector
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.scheduler import SchedulerAgent
from agents.communicator import CommunicatorAgent
from core.orchestrator import Orchestrator
from core.reflector import Reflector
from monitors.world_monitor import WorldMonitor
from scheduler.jobs import setup_scheduler
from api.health import router as health_router
from api.admin import router as admin_router
from api.webhooks import router as webhooks_router

logger = get_logger(__name__)


async def _seed_user_profile(memory: MemoryManager) -> None:
    """Seed initial user profile facts into memory on first run."""
    if memory._vs.count() > 0:
        logger.info("[Startup] Memory already populated — skipping seed")
        return

    logger.info("[Startup] Seeding initial user profile...")
    seeds = [
        (f"{settings.user_name} is a 2nd year AI/ML Engineering student at AJIET Mangaluru, VTU",
         "fact", "profile_seed", 0.95),
        (f"{settings.user_name}'s interests: {settings.user_interests}",
         "fact", "profile_seed", 0.9),
        (f"{settings.user_name}'s goals: {settings.user_goals}",
         "fact", "profile_seed", 0.9),
        (f"{settings.user_name} is located in {settings.user_location}",
         "fact", "profile_seed", 0.8),
        ("NEXUS is an autonomous multi-agent personal intelligence system built by the user",
         "fact", "profile_seed", 0.85),
    ]
    for text, type_, source, importance in seeds:
        await memory.remember(text, type=type_, source=source, importance=importance)
    logger.info(f"[Startup] Seeded {len(seeds)} initial facts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup and shutdown."""
    logger.info("=" * 60)
    logger.info("  NEXUS — Proactive Personal Intelligence Agent v2.0")
    logger.info(f"  Environment: {settings.app_env}")
    logger.info(f"  User: {settings.user_name}")
    logger.info("=" * 60)

    # ── Phase 1: Initialize databases ─────────────────────────────────
    logger.info("[Startup] Phase 1: Initializing databases...")
    create_tables()
    vector_store = VectorStore()
    knowledge_graph = KnowledgeGraph()
    memory_manager = MemoryManager(vector_store, knowledge_graph)

    # ── Phase 2: Initialize connectors ────────────────────────────────
    logger.info("[Startup] Phase 2: Initializing connectors...")
    whatsapp = WhatsAppConnector()
    web_search = WebSearchConnector()

    # ── Phase 3: Initialize agents ────────────────────────────────────
    logger.info("[Startup] Phase 3: Initializing agents...")
    researcher = ResearcherAgent(web_search)
    coder = CoderAgent(settings.user_name, settings.user_interests)
    scheduler_agent = SchedulerAgent(knowledge_graph)
    communicator = CommunicatorAgent(settings.user_name)

    # ── Phase 4: Initialize core ──────────────────────────────────────
    logger.info("[Startup] Phase 4: Initializing core orchestrator...")
    orchestrator = Orchestrator(memory_manager, researcher, coder, scheduler_agent, communicator)
    reflector = Reflector(memory_manager, knowledge_graph, whatsapp)
    world_monitor = WorldMonitor(memory_manager, communicator, whatsapp)

    # ── Store on app state ────────────────────────────────────────────
    app.state.vector_store = vector_store
    app.state.knowledge_graph = knowledge_graph
    app.state.memory_manager = memory_manager
    app.state.whatsapp = whatsapp
    app.state.web_search = web_search
    app.state.researcher = researcher
    app.state.coder = coder
    app.state.scheduler_agent = scheduler_agent
    app.state.communicator = communicator
    app.state.orchestrator = orchestrator
    app.state.reflector = reflector
    app.state.world_monitor = world_monitor

    # ── Phase 5: Seed user profile ────────────────────────────────────
    logger.info("[Startup] Phase 5: Seeding user profile...")
    await _seed_user_profile(memory_manager)

    # ── Phase 6: Start scheduler ──────────────────────────────────────
    logger.info("[Startup] Phase 6: Starting scheduler...")
    scheduler = setup_scheduler(app)
    scheduler.start()
    app.state.scheduler = scheduler

    logger.info("🚀 NEXUS is fully operational!")

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("[Shutdown] Stopping scheduler...")
    scheduler.shutdown(wait=False)
    logger.info("[Shutdown] NEXUS shut down gracefully.")


# ── Create the FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="NEXUS — Proactive Personal Intelligence Agent",
    description="24/7 autonomous multi-agent personal intelligence system",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Register routers ─────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(webhooks_router)


@app.get("/")
async def root():
    return {
        "name": "NEXUS",
        "version": "2.0.0",
        "status": "operational",
        "description": "Proactive Personal Intelligence Agent for Abhijith",
    }
