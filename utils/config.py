"""Settings singleton loaded from .env via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    gemini_api_key: str

    # Twilio
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_whatsapp_from: str
    user_whatsapp_number: str

    # Google (optional until Phase 5)
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    # Paths
    sqlite_db_path: str = "./data/nexus.db"
    chroma_db_path: str = "./data/chroma_store"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    admin_secret_key: str = "change_me"

    # User profile
    user_name: str = "Abhijith"
    user_interests: str = "AI/ML,hackathons,Python,Flutter"
    user_goals: str = "master AI/ML,build portfolio,crack interviews"
    user_location: str = "Mangaluru, Karnataka, India"

    # Scheduler (24h IST)
    morning_brief_time: str = "08:00"
    reflection_time: str = "23:30"
    monitor_interval_hours: int = 6
    deadline_check_interval_hours: int = 4
    send_reflection_summary: bool = True

    @model_validator(mode="after")
    def override_paths_in_production(self) -> "Settings":
        """On Railway (production), use the persistent /data volume."""
        if self.app_env == "production":
            self.sqlite_db_path = "/data/nexus.db"
            self.chroma_db_path = "/data/chroma_store"
        return self


settings = Settings()
