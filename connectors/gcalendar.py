from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pytz
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GoogleCalendarConnector:
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    
    def __init__(self):
        self._service = None
        self._init_service()
    
    def _init_service(self):
        try:
            if (not settings.google_refresh_token 
                or "your_google" in settings.google_refresh_token
                or not settings.google_client_id
                or "your_google" in settings.google_client_id):
                logger.info("Calendar connector: Google credentials not configured. Running in dormant mode.")
                return
            
            creds = Credentials(
                token=None,
                refresh_token=settings.google_refresh_token,
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
                token_uri="https://oauth2.googleapis.com/token",
                scopes=self.SCOPES
            )
            self._service = build("calendar", "v3", credentials=creds)
            logger.info("Google Calendar connector initialized.")
        except Exception as e:
            logger.error(f"Calendar init failed: {e}")
            self._service = None
    
    @property
    def is_available(self) -> bool:
        return self._service is not None
    
    def get_upcoming_events(self, days_ahead: int = 7) -> list[dict]:
        if not self.is_available:
            return []
        try:
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist)
            end = now + timedelta(days=days_ahead)
            
            result = self._service.events().list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=20
            ).execute()
            
            events = []
            for item in result.get("items", []):
                start = item.get("start", {})
                events.append({
                    "title": item.get("summary", "No title"),
                    "start": start.get("dateTime", start.get("date", "")),
                    "location": item.get("location", ""),
                    "description": item.get("description", ""),
                    "id": item["id"]
                })
            
            logger.info(f"Fetched {len(events)} calendar events.")
            return events
        except Exception as e:
            logger.error(f"Error fetching calendar events: {e}")
            return []
    
    def create_event(self, title: str, start_dt: datetime, end_dt: datetime,
                     description: str = "", location: str = "") -> bool:
        if not self.is_available:
            return False
        try:
            ist = pytz.timezone("Asia/Kolkata")
            if start_dt.tzinfo is None:
                start_dt = ist.localize(start_dt)
            if end_dt.tzinfo is None:
                end_dt = ist.localize(end_dt)
            
            event_body = {
                "summary": title,
                "description": description,
                "location": location,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Kolkata"},
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 30},
                        {"method": "email", "minutes": 60}
                    ]
                }
            }
            
            created = self._service.events().insert(calendarId="primary", body=event_body).execute()
            logger.info(f"Calendar event created: {created.get('htmlLink')}")
            return True
        except Exception as e:
            logger.error(f"Error creating calendar event: {e}")
            return False
