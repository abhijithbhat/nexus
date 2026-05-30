"""Pydantic request/response schemas for NEXUS API."""
from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    memory_entries: int
    version: str


class WebhookMessage(BaseModel):
    from_number: str
    body: str
    message_sid: str = ""
    media_url: str | None = None
    timestamp: str = ""


class AdminTriggerResponse(BaseModel):
    status: str


class MemoryStatsResponse(BaseModel):
    total_entries: int


class EventCreate(BaseModel):
    title: str
    description: str = ""
    scheduled_at: datetime


class OrchestratorState(BaseModel):
    """The state object that flows through the LangGraph orchestrator."""
    user_message: str = ""
    intent: str = ""
    memory_context: str = ""
    user_profile: str = ""
    agent_outputs: dict = {}
    final_response: str = ""
    needs_followup: bool = False
    error: str = ""
