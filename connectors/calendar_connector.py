"""Google Calendar connector — placeholder for Phase 5+."""
from utils.logger import get_logger

logger = get_logger(__name__)


class CalendarConnector:
    """Google Calendar create/read events. Will be implemented in Phase 5."""

    def __init__(self) -> None:
        logger.info("[Calendar] Connector initialized (Phase 5 — placeholder)")

    async def create_event(self, title: str, start: str, end: str, description: str = "") -> bool:
        """Create a calendar event. Not yet implemented."""
        logger.warning("[Calendar] create_event called but not yet implemented (Phase 5)")
        return False

    async def get_upcoming(self, days: int = 7) -> list[dict]:
        """Get upcoming calendar events. Not yet implemented."""
        logger.warning("[Calendar] get_upcoming called but not yet implemented (Phase 5)")
        return []
