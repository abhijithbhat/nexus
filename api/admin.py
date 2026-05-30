from fastapi import APIRouter, Request, Header, HTTPException, Depends
from typing import List, Dict, Any
from models.schemas import MemoryStatsResponse, EventSchema
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

async def verify_admin_key(x_admin_key: str = Header(None)):
    if not x_admin_key or x_admin_key != settings.admin_secret_key:
        logger.warning(f"Failed admin authentication attempt with key: {x_admin_key}")
        raise HTTPException(status_code=401, detail="Unauthorized admin key")

@router.post("/admin/trigger-brief", dependencies=[Depends(verify_admin_key)])
async def trigger_brief(request: Request):
    if hasattr(request.app.state, "world_monitor"):
        try:
            # Send morning brief synchronously for the admin user's webhook response feedback
            await request.app.state.world_monitor.send_morning_brief()
            return {"status": "success", "detail": "Morning brief triggered and sent."}
        except Exception as e:
            logger.error(f"Error manually triggering brief: {e}")
            return {"status": "error", "detail": str(e)}
    else:
        return {"status": "warning", "detail": "World monitor is not initialized yet in Phase 1."}

@router.post("/admin/trigger-reflection", dependencies=[Depends(verify_admin_key)])
async def trigger_reflection(request: Request):
    if hasattr(request.app.state, "reflector"):
        try:
            await request.app.state.reflector.run_nightly_reflection()
            return {"status": "success", "detail": "Nightly self-reflection triggered."}
        except Exception as e:
            logger.error(f"Error manually triggering reflection: {e}")
            return {"status": "error", "detail": str(e)}
    else:
        return {"status": "warning", "detail": "Reflector is not initialized yet in Phase 1."}

@router.post("/admin/trigger-scan", dependencies=[Depends(verify_admin_key)])
async def trigger_scan(request: Request):
    if hasattr(request.app.state, "world_monitor"):
        try:
            results = await request.app.state.world_monitor.run_full_scan()
            return {"status": "success", "detail": "World scan triggered.", "results": results}
        except Exception as e:
            logger.error(f"Error manually triggering scan: {e}")
            return {"status": "error", "detail": str(e)}
    else:
        return {"status": "warning", "detail": "World monitor is not initialized yet in Phase 1."}

@router.get("/admin/memory-stats", response_model=MemoryStatsResponse, dependencies=[Depends(verify_admin_key)])
async def memory_stats(request: Request):
    memory_manager = request.app.state.memory_manager
    vector_store = memory_manager.vector_store
    
    total = vector_store.count()
    
    # Compile stats
    types = ["conversation", "fact", "monitor_signal", "consolidated_summary"]
    counts = {}
    for t in types:
        # ChromaDB query by type
        counts[t] = len(vector_store.get_by_type(t, limit=1000))
        
    recent = vector_store.get_recent(hours=24)
    recent_mems = []
    for r in recent[:10]:
        recent_mems.append({
            "id": r["id"],
            "text": r["text"],
            "metadata": r["metadata"]
        })
        
    return {
        "total_count": total,
        "counts_by_type": counts,
        "recent_memories": recent_mems
    }

@router.get("/admin/upcoming-events", response_model=List[EventSchema], dependencies=[Depends(verify_admin_key)])
async def upcoming_events(request: Request):
    knowledge_graph = request.app.state.memory_manager.knowledge_graph
    # Return events for the next 7 days (168 hours)
    events = knowledge_graph.get_upcoming_events(hours_ahead=168)
    return events
