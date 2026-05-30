"""Twilio WhatsApp connector — outbound sends, inbound webhook parsing, signature validation."""
import time

from twilio.rest import Client
from twilio.request_validator import RequestValidator

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_WA_CHARS = 1500


class WhatsAppConnector:
    """Send and receive WhatsApp messages via Twilio."""

    def __init__(self) -> None:
        self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        self._validator = RequestValidator(settings.twilio_auth_token)
        logger.info("[WhatsApp] Connector initialized")

    def send_message(self, to: str, body: str) -> bool:
        """Send a WhatsApp message. Auto-splits messages over 1500 chars."""
        chunks = self._split(body)
        success = True
        for i, chunk in enumerate(chunks):
            try:
                msg = self._client.messages.create(
                    from_=settings.twilio_whatsapp_from, to=to, body=chunk
                )
                logger.info(f"[WhatsApp] Sent chunk {i+1}/{len(chunks)} SID={msg.sid}")
                if i < len(chunks) - 1:
                    time.sleep(0.5)  # Prevent flooding
            except Exception as exc:
                logger.error(f"[WhatsApp] Send failed chunk {i+1}: {exc}")
                success = False
        return success

    def send_to_user(self, body: str) -> bool:
        """Send a WhatsApp message to the configured user number."""
        return self.send_message(settings.user_whatsapp_number, body)

    def parse_incoming_webhook(self, form_data: dict) -> dict:
        """Parse Twilio inbound POST body into a clean dict."""
        return {
            "from_number": form_data.get("From", ""),
            "body": form_data.get("Body", "").strip(),
            "message_sid": form_data.get("MessageSid", ""),
            "media_url": form_data.get("MediaUrl0"),
            "timestamp": form_data.get("DateSent", ""),
        }

    def validate_request(self, url: str, params: dict, signature: str) -> bool:
        """Verify request genuinely came from Twilio. Always call this first."""
        return self._validator.validate(url, params, signature)

    @staticmethod
    def _split(text: str, limit: int = MAX_WA_CHARS) -> list[str]:
        """Split text at newline boundaries into chunks under limit."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            idx = text.rfind("\n", 0, limit)
            if idx == -1:
                idx = limit
            chunks.append(text[:idx].strip())
            text = text[idx:].strip()
        return chunks
