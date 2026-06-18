import json
from datetime import datetime, date
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class UsageTracker:
    """Track Gemini API calls and estimated costs."""
    
    # Gemini Flash pricing (approximate, update as needed)
    COST_PER_1K_INPUT_TOKENS = 0.000075   # $0.075 per 1M = $0.000075 per 1K
    COST_PER_1K_OUTPUT_TOKENS = 0.0003    # $0.30 per 1M = $0.0003 per 1K
    
    def __init__(self, path: str = "./data/usage.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()
    
    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
    
    def record_call(self, prompt_len: int, response_len: int, model: str = "gemini-flash"):
        """Record an API call with estimated token counts."""
        today = str(date.today())
        if today not in self._data:
            self._data[today] = {
                "calls": 0,
                "input_chars": 0,
                "output_chars": 0,
                "est_cost_usd": 0.0
            }
        
        # Rough char → token conversion (1 token ≈ 4 chars)
        input_tokens = prompt_len / 4
        output_tokens = response_len / 4
        cost = (
            (input_tokens / 1000 * self.COST_PER_1K_INPUT_TOKENS) +
            (output_tokens / 1000 * self.COST_PER_1K_OUTPUT_TOKENS)
        )
        
        self._data[today]["calls"] += 1
        self._data[today]["input_chars"] += prompt_len
        self._data[today]["output_chars"] += response_len
        self._data[today]["est_cost_usd"] = round(
            self._data[today]["est_cost_usd"] + cost, 6
        )
        self._save()
    
    def get_daily_summary(self) -> dict:
        today = str(date.today())
        return self._data.get(today, {
            "calls": 0, "input_chars": 0,
            "output_chars": 0, "est_cost_usd": 0.0
        })
    
    def get_monthly_cost(self) -> float:
        month = str(date.today())[:7]  # "2025-01"
        total = sum(
            v.get("est_cost_usd", 0.0)
            for k, v in self._data.items()
            if k.startswith(month)
        )
        return round(total, 4)
