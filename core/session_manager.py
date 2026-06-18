import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manages in-session conversation history per WhatsApp number.
    Keeps last N messages in memory for active sessions.
    Sessions expire after TIMEOUT_MINUTES of inactivity.
    """
    
    MAX_HISTORY = 10          # Messages to keep per session
    TIMEOUT_MINUTES = 30      # Session expires after 30 min silence
    
    def __init__(self):
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self._last_activity: dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def add_message(self, number: str, role: str, content: str):
        """Add a message to the session. role: 'user' or 'assistant'"""
        async with self._lock:
            self._sessions[number].append({
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._last_activity[number] = datetime.utcnow()
    
    async def get_history(self, number: str) -> list[dict]:
        """Get session history as list of {role, content} dicts."""
        async with self._lock:
            self._cleanup_expired()
            return list(self._sessions[number])
    
    async def get_formatted_history(self, number: str) -> str:
        """Get session history as formatted string for injection into prompts."""
        history = await self.get_history(number)
        if not history:
            return "No conversation history in this session."
        lines = []
        for msg in history[-6:]:  # Last 6 for brevity
            role_label = "User" if msg["role"] == "user" else "NEXUS"
            lines.append(f"{role_label}: {msg['content'][:200]}")
        return "\n".join(lines)
    
    def _cleanup_expired(self):
        now = datetime.utcnow()
        expired = [
            num for num, last in self._last_activity.items()
            if now - last > timedelta(minutes=self.TIMEOUT_MINUTES)
        ]
        for num in expired:
            if num in self._sessions:
                del self._sessions[num]
            del self._last_activity[num]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions.")
    
    async def clear_session(self, number: str):
        async with self._lock:
            if number in self._sessions:
                del self._sessions[number]
            if number in self._last_activity:
                del self._last_activity[number]
