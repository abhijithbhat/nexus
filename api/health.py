from fastapi import APIRouter, Request
from datetime import datetime
from models.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    memory_manager = request.app.state.memory_manager
    count = memory_manager.vector_store.count()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "memory_entries": count,
        "version": "1.0.0"
    }
