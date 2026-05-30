"""GET /health — system liveness check."""
from datetime import datetime
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Returns NEXUS operational status."""
    count = 0
    try:
        count = request.app.state.vector_store.count()
    except Exception:
        pass
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "memory_entries": count,
        "version": "2.0.0",
    }
