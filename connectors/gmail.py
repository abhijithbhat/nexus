from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GmailConnector:
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send"
    ]
    
    def __init__(self):
        self._service = None
        self._init_service()
    
    def _init_service(self):
        try:
            if (not settings.google_refresh_token 
                or "your_google" in settings.google_refresh_token
                or not settings.google_client_id
                or "your_google" in settings.google_client_id):
                logger.info("Gmail connector: Google credentials not configured. Running in dormant mode.")
                return
            
            creds = Credentials(
                token=None,
                refresh_token=settings.google_refresh_token,
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
                token_uri="https://oauth2.googleapis.com/token",
                scopes=self.SCOPES
            )
            self._service = build("gmail", "v1", credentials=creds)
            logger.info("Gmail connector initialized successfully.")
        except Exception as e:
            logger.error(f"Gmail connector init failed: {e}")
            self._service = None
    
    @property
    def is_available(self) -> bool:
        return self._service is not None
    
    def get_unread_emails(self, max_results: int = 10) -> list[dict]:
        """Fetch recent unread emails."""
        if not self.is_available:
            return []
        try:
            result = self._service.users().messages().list(
                userId="me",
                labelIds=["INBOX", "UNREAD"],
                maxResults=max_results
            ).execute()
            
            messages = result.get("messages", [])
            emails = []
            for msg in messages:
                detail = self._service.users().messages().get(
                    userId="me",
                    id=msg["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"]
                ).execute()
                
                headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
                snippet = detail.get("snippet", "")
                
                emails.append({
                    "id": msg["id"],
                    "subject": headers.get("Subject", "(no subject)"),
                    "from": headers.get("From", "unknown"),
                    "date": headers.get("Date", ""),
                    "snippet": snippet,
                    "thread_id": detail.get("threadId", "")
                })
            
            logger.info(f"Fetched {len(emails)} unread emails.")
            return emails
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return []
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email."""
        if not self.is_available:
            logger.warning("Gmail not available. Cannot send.")
            return False
        try:
            mime_msg = MIMEText(body)
            mime_msg["to"] = to
            mime_msg["subject"] = subject
            raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
            
            self._service.users().messages().send(
                userId="me",
                body={"raw": raw}
            ).execute()
            
            logger.info(f"Email sent to {to}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def get_email_body(self, message_id: str) -> str:
        """Get the full body of a specific email."""
        if not self.is_available:
            return ""
        try:
            detail = self._service.users().messages().get(
                userId="me",
                id=message_id,
                format="full"
            ).execute()
            
            payload = detail.get("payload", {})
            return self._extract_body(payload)
        except Exception as e:
            logger.error(f"Error getting email body: {e}")
            return ""
    
    def _extract_body(self, payload: dict) -> str:
        body = ""
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                        break
        elif payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        return body
