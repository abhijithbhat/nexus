from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    memory_entries: int
    version: str

class EventSchema(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    scheduled_at: datetime
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class MemoryStatsResponse(BaseModel):
    total_count: int
    counts_by_type: Dict[str, int]
    recent_memories: List[Dict[str, Any]]
