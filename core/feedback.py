from datetime import datetime
from utils.logger import get_logger
from memory.knowledge_graph import KnowledgeGraph

logger = get_logger(__name__)


class FeedbackProcessor:
    """
    Processes user feedback signals embedded in WhatsApp messages.
    Recognizes: 👍/good/perfect → positive
                👎/wrong/bad/fix → negative
                "that was wrong", "actually" → correction
    """
    
    POSITIVE_SIGNALS = [
        "👍", "✅", "good", "perfect", "great", "correct", "exactly",
        "yes that's right", "nice", "awesome", "thanks that worked"
    ]
    NEGATIVE_SIGNALS = [
        "👎", "❌", "wrong", "bad", "incorrect", "that's not right",
        "fix that", "not what i meant", "no that's wrong"
    ]
    CORRECTION_SIGNALS = [
        "actually", "i meant", "no wait", "correction:",
        "the right answer is", "what i actually need"
    ]
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
    
    def detect_feedback(self, message: str) -> dict | None:
        msg_lower = message.lower().strip()
        
        # Only trigger on short messages that are clearly feedback
        # (not longer messages that happen to contain these words)
        if len(msg_lower) > 100:
            return None
        
        if any(sig in msg_lower for sig in self.POSITIVE_SIGNALS):
            return {"type": "positive", "message": message}
        if any(sig in msg_lower for sig in self.NEGATIVE_SIGNALS):
            return {"type": "negative", "message": message}
        if any(sig in msg_lower for sig in self.CORRECTION_SIGNALS):
            return {"type": "correction", "message": message}
        return None
    
    def record_feedback(self, feedback_type: str, context: str, response_summary: str):
        """Record feedback for pattern learning."""
        self.kg.add_fact(
            subject="NEXUS_QUALITY",
            predicate=f"received_{feedback_type}_feedback",
            object=f"{datetime.utcnow().date()}: {response_summary[:100]}",
            confidence=1.0,
            source="user_feedback"
        )
        logger.info(f"Feedback recorded: {feedback_type}")
