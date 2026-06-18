import re

KEYWORD_HIGH = [
    "deadline", "exam", "project", "goal", "interview", "hackathon",
    "submission", "launch", "presentation", "contest", "due", "must",
    "important", "critical", "urgent", "remember", "don't forget",
    "placement", "internship", "offer", "accepted", "rejected"
]
KEYWORD_MEDIUM = [
    "likes", "prefers", "interested", "learning", "working on",
    "building", "studying", "practicing", "schedule", "reminder",
    "wants to", "plans to", "thinking about", "considering"
]
KEYWORD_LOW = [
    "hi", "hello", "thanks", "ok", "sure", "yes", "no", "bye",
    "good morning", "good night", "haha", "lol", "cool", "nice"
]


def fast_score(text: str, type: str, source: str) -> float | None:
    """
    Rule-based importance scorer. Returns float 0.0-1.0 or None if ambiguous.
    None means: call Gemini for accurate scoring.
    """
    text_lower = text.lower()
    
    # Type-based overrides (non-conversation types)
    if type in ["consolidated_summary", "fact", "event"]:
        return 0.85
    if type == "monitor_signal":
        return 0.70
    
    # Keyword matching (runs before short-message check so important short
    # messages like "deadline tomorrow" still get high scores)
    if any(kw in text_lower for kw in KEYWORD_HIGH):
        return 0.80
    if any(kw in text_lower for kw in KEYWORD_LOW):
        return 0.10
    if any(kw in text_lower for kw in KEYWORD_MEDIUM):
        return 0.55
    
    # Short conversation heuristic (only after keywords didn't match)
    if type == "conversation" and len(text) < 20:
        return 0.15  # Short messages with no keywords — trivial
    
    # Ambiguous — let Gemini decide
    return None
