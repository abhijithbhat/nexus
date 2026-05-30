import time
from datetime import datetime
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class WhatsAppConnector:
    def __init__(self):
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.from_number = settings.twilio_whatsapp_from
        
        self.is_mock = (
            not self.account_sid 
            or "your_twilio_account" in self.account_sid
            or not self.auth_token
            or "your_twilio_auth" in self.auth_token
        )
        
        if self.is_mock:
            logger.warning("Twilio credentials not configured. WhatsApp messages will be logged instead of sent.")
            self.client = None
            self.validator = None
        else:
            self.client = Client(self.account_sid, self.auth_token)
            self.validator = RequestValidator(self.auth_token)

    def send_message(self, to: str, body: str) -> bool:
        if self.is_mock:
            logger.info(f"[MOCK WHATSAPP SEND] To: {to} | From: {self.from_number} | Body:\n{body}")
            return True
            
        try:
            if len(body) > 1500:
                split_idx = body[:1500].rfind("\n")
                if split_idx == -1:
                    split_idx = 1500
                    
                part1 = body[:split_idx]
                part2 = body[split_idx:].strip()
                
                success1 = self._send_single_message(to, part1)
                time.sleep(0.5)
                success2 = self._send_single_message(to, part2)
                return success1 and success2
            else:
                return self._send_single_message(to, body)
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return False

    def _send_single_message(self, to: str, body: str) -> bool:
        try:
            message = self.client.messages.create(
                body=body,
                from_=self.from_number,
                to=to
            )
            logger.info(f"WhatsApp message sent successfully. SID: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send single WhatsApp message: {e}")
            return False

    def parse_incoming_webhook(self, form_data: dict) -> dict:
        return {
            "from_number": form_data.get("From", ""),
            "body": form_data.get("Body", ""),
            "message_sid": form_data.get("MessageSid", ""),
            "media_url": form_data.get("MediaUrl0"),
            "timestamp": datetime.utcnow().isoformat()
        }

    def validate_request(self, url: str, params: dict, signature: str) -> bool:
        if self.is_mock:
            logger.info("Signature validation skipped in mock mode.")
            return True
            
        if not signature:
            return False
        return self.validator.validate(url, params, signature)
