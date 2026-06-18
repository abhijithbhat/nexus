import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from utils.config import settings
from utils.logger import get_logger, ROOT_DIR
from utils.gemini_client import GeminiClient
from utils.usage_tracker import UsageTracker
from memory.memory_manager import MemoryManager
from connectors.whatsapp import WhatsAppConnector
from core.orchestrator import NexusOrchestrator
from scheduler.jobs import JobScheduler
from api.health import router as health_router
from api.webhooks import router as webhooks_router
from api.admin import router as admin_router

logger = get_logger(__name__)


async def seed_first_run(memory_manager: MemoryManager):
    kg = memory_manager.knowledge_graph
    
    # Check if system is seeded
    facts = kg.search_facts("is_seeded")
    seeded = False
    for f in facts:
        if f.subject == "system" and f.predicate == "status" and f.object == "is_seeded":
            seeded = True
            break
            
    if not seeded:
        logger.info("First run detected. Seeding user profile facts into knowledge graph...")
        user_name = settings.user_name
        kg.add_fact(user_name, "lives_in", settings.user_location, confidence=1.0, source="system_seed")
        
        # Add interests
        interests = [i.strip() for i in settings.user_interests.split(",") if i.strip()]
        for interest in interests:
            kg.add_fact(user_name, "interested_in", interest, confidence=1.0, source="system_seed")
            kg.add_entity(interest, "topic", f"Interest of {user_name}")
            
        # Add goals
        goals = [g.strip() for g in settings.user_goals.split(",") if g.strip()]
        for goal in goals:
            kg.add_fact(user_name, "has_goal", goal, confidence=1.0, source="system_seed")
            
        # Add seeded flag
        kg.add_fact("system", "status", "is_seeded", confidence=1.0, source="system_seed")
        
        # Store a welcome conversation memory in ChromaDB
        welcome_text = f"System initialized. Profile seeded for user {user_name}."
        await memory_manager.remember(welcome_text, "fact", "system_seed", importance=0.9)
        logger.info("First-run seeding completed successfully.")
    else:
        logger.info("First-run seeding already complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("NEXUS is starting up...")
    
    # Ensure data directory exists
    os.makedirs(ROOT_DIR / "data", exist_ok=True)
    
    # Initialize components
    memory_manager = MemoryManager()
    gemini_client = GeminiClient()
    whatsapp = WhatsAppConnector()
    usage_tracker = UsageTracker()
    orchestrator = NexusOrchestrator(memory_manager, gemini_client)
    
    from monitors.world_monitor import WorldMonitor
    from core.reflector import NexusReflector
    from core.feedback import FeedbackProcessor
    
    world_monitor = WorldMonitor(memory_manager, whatsapp)
    reflector = NexusReflector(memory_manager, gemini_client, whatsapp)
    feedback_processor = FeedbackProcessor(memory_manager.knowledge_graph)
    
    # Store on app.state
    app.state.memory_manager = memory_manager
    app.state.gemini_client = gemini_client
    app.state.whatsapp = whatsapp
    app.state.usage_tracker = usage_tracker
    app.state.orchestrator = orchestrator
    app.state.world_monitor = world_monitor
    app.state.reflector = reflector
    app.state.feedback_processor = feedback_processor
    
    # Seed first run user profile
    await seed_first_run(memory_manager)
    
    # Start scheduler
    scheduler = JobScheduler(
        world_monitor=world_monitor,
        reflector=reflector,
        memory_manager=memory_manager,
        whatsapp=whatsapp,
        knowledge_graph=memory_manager.knowledge_graph
    )
    app.state.scheduler = scheduler
    scheduler.start()
    
    logger.info("NEXUS is online.")
    
    yield
    
    # Shutdown
    logger.info("NEXUS is shutting down...")
    scheduler.shutdown()
    logger.info("NEXUS is offline.")

app = FastAPI(title="NEXUS", version="2.0.0", lifespan=lifespan)

# Mount routers
app.include_router(health_router)
app.include_router(webhooks_router)
app.include_router(admin_router)
