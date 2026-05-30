import os
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    gemini_api_key: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_whatsapp_from: str
    user_whatsapp_number: str
    google_client_id: str
    google_client_secret: str
    google_refresh_token: str
    sqlite_db_path: str
    chroma_db_path: str
    app_env: str
    log_level: str
    admin_secret_key: str
    user_name: str
    user_interests: str
    user_goals: str
    user_location: str
    morning_brief_time: str
    reflection_time: str
    monitor_interval_hours: int
    deadline_check_interval_hours: int

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def set_production_paths(self) -> "Settings":
        if self.app_env == "production":
            self.sqlite_db_path = "/data/nexus.db"
            self.chroma_db_path = "/data/chroma_store"
        else:
            # Resolve db paths relative to project root if they are relative
            if not os.path.isabs(self.sqlite_db_path):
                self.sqlite_db_path = str((ROOT_DIR / self.sqlite_db_path).resolve())
            if not os.path.isabs(self.chroma_db_path):
                self.chroma_db_path = str((ROOT_DIR / self.chroma_db_path).resolve())
        return self

settings = Settings()
