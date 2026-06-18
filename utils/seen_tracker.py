import json
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class SeenTracker:
    """Tracks already-seen monitor items to prevent duplicate alerts."""
    
    def __init__(self, path: str = "./data/seen_items.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()
    
    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
    
    def has_seen(self, item_id: str) -> bool:
        return item_id in self._data
    
    def mark_seen(self, item_id: str, title: str = ""):
        self._data[item_id] = {
            "seen_at": datetime.utcnow().isoformat(),
            "title": title
        }
        # Keep only last 1000 items to prevent unbounded growth
        if len(self._data) > 1000:
            oldest_keys = sorted(
                self._data, key=lambda k: self._data[k].get("seen_at", "")
            )[:100]
            for k in oldest_keys:
                del self._data[k]
        self._save()
