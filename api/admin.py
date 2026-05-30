"""Admin endpoints — protected by X-Admin-Key header. Never expose publicly."""
from fastapi import APIRouter, HTTPException, Request
from utils.config import settings

router = APIRouter(prefix="/admin")


def _auth(request: Request) -> None:
    if request.headers.get("X-Admin-Key", "") != settings.admin_secret_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/trigger-brief")
async def trigger_brief(request: Request) -> dict:
    _auth(request)
    await request.app.state.world_monitor.send_morning_brief()
    return {"status": "brief sent"}


@router.post("/trigger-reflection")
async def trigger_reflection(request: Request) -> dict:
    _auth(request)
    await request.app.state.reflector.run_nightly_reflection()
    return {"status": "reflection complete"}


@router.post("/trigger-scan")
async def trigger_scan(request: Request) -> dict:
    _auth(request)
    results = await request.app.state.world_monitor.run_full_scan()
    return {"status": "scan complete", "counts": {k: len(v) for k, v in results.items() if isinstance(v, list)}}


@router.get("/memory-stats")
async def memory_stats(request: Request) -> dict:
    _auth(request)
    return {"total_entries": request.app.state.vector_store.count()}
