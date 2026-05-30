"""Gmail connector — placeholder for Phase 5+. OAuth2 with Google API."""
from utils.logger import get_logger

logger = get_logger(__name__)


class GmailConnector:
    """Gmail read/send via Google API. Will be implemented in Phase 5."""

    def __init__(self) -> None:
        logger.info("[Gmail] Connector initialized (Phase 5 — placeholder)")

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email. Not yet implemented."""
        logger.warning("[Gmail] send_email called but not yet implemented (Phase 5)")
        return False

    async def read_inbox(self, max_results: int = 10) -> list[dict]:
        """Read recent inbox emails. Not yet implemented."""
        logger.warning("[Gmail] read_inbox called but not yet implemented (Phase 5)")
        return []
