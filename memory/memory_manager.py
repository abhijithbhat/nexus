"""Unified memory interface — the single entry point for all NEXUS memory operations."""
import asyncio
from datetime import datetime

from memory.vector_store import VectorStore
from memory.knowledge_graph import KnowledgeGraph
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_IMPORTANCE_PROMPT = (
    "Rate the long-term importance of storing this for a personal AI assistant.\n"
    "Type: {type_}\nSource: {source}\nContent: {text}\n\n"
    "Scale: 0.0-0.3=trivial, 0.4-0.6=useful, 0.7-0.9=important, 0.9-1.0=critical\n"
    'Return ONLY JSON: {{"score": 0.XX, "reason": "one sentence"}}'
)

_EXTRACT_PROMPT = (
    "Extract entities and facts from this text.\n"
    "Text: {text}\n"
    'Return ONLY JSON: {{"entities": [{{"name": "...", "type": "person|project|tool|topic", '
    '"description": "..."}}], "facts": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}'
)


class MemoryManager:
    """Single interface for all NEXUS memory: vector search + knowledge graph."""

    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph) -> None:
        self._vs = vector_store
        self._kg = knowledge_graph
        self._gemini = None  # Lazy import to avoid circular dependency

    def _gc(self):
        """Lazy-loaded gemini_client singleton."""
        if self._gemini is None:
            from utils.gemini_client import gemini_client
            self._gemini = gemini_client
        return self._gemini

    async def _score_importance(self, text: str, type_: str, source: str) -> float:
        try:
            prompt = _IMPORTANCE_PROMPT.format(type_=type_, source=source, text=text[:400])
            result = await self._gc().generate_json("You score memory importance.", prompt, temperature=0.1)
            return float(result.get("score", 0.5))
        except Exception:
            return 0.5

    async def remember(self, text: str, type: str, source: str, importance: float | None = None) -> str:
        """Store a memory entry. Auto-scores importance if not provided."""
        if importance is None:
            importance = await self._score_importance(text, type, source)

        doc_id = self._vs.add(text=text, metadata={
            "type": type, "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "importance": importance,
        })

        # For high-importance entries: also extract structured knowledge
        if importance >= 0.7:
            try:
                extracted = await self._gc().generate_json(
                    "You extract named entities and factual triples.",
                    _EXTRACT_PROMPT.format(text=text[:600]),
                    temperature=0.1,
                )
                for ent in extracted.get("entities", [])[:4]:
                    self._kg.add_entity(ent.get("name", ""), ent.get("type", "topic"), ent.get("description", ""))
                for fact in extracted.get("facts", [])[:4]:
                    self._kg.add_fact(fact.get("subject", ""), fact.get("predicate", ""), fact.get("object", ""),
                                      confidence=importance, source=source)
            except Exception as exc:
                logger.debug(f"[Memory] Entity extraction skipped: {exc}")

        return doc_id

    async def recall(self, query: str, limit: int = 8) -> str:
        """Return formatted memory context relevant to a query."""
        results = self._vs.search(query, n_results=limit)
        facts = self._kg.search_facts(query)

        if not results and not facts:
            return "No relevant memory found."

        lines = ["[Relevant memory]"]
        for r in results:
            m = r["metadata"]
            ts = m.get("timestamp", "")[:10]
            lines.append(f"• [{m.get('type', '?')}] {r['text']} (via {m.get('source', '?')}, {ts})")
        for f in facts[:4]:
            lines.append(f"• [fact] {f.subject} {f.predicate} {f.object} (confidence {f.confidence:.1f})")
        return "\n".join(lines)

    async def get_full_user_context(self) -> str:
        """Assemble the full user context string injected into every system prompt."""
        profile = self._kg.get_user_profile()
        reflections = self._kg.get_recent_reflections(days=3)
        recent = self._vs.get_recent(hours=48)
        convs = [r for r in recent if r["metadata"].get("type") == "conversation"][:4]

        lines = [
            f"USER: {settings.user_name}",
            f"LOCATION: {settings.user_location}",
            f"INTERESTS: {settings.user_interests}",
            f"GOALS: {settings.user_goals}",
        ]
        if profile["facts"]:
            lines.append("\nKNOWN FACTS:")
            for s, p, o in profile["facts"][:8]:
                lines.append(f"  · {s} {p} {o}")
        if profile["upcoming_events"]:
            lines.append("\nUPCOMING EVENTS:")
            for title, dt in profile["upcoming_events"]:
                lines.append(f"  · {title} — {dt}")
        if reflections:
            lines.append(f"\nLAST REFLECTION INSIGHT: {reflections[0].content[:200]}...")
        if convs:
            lines.append("\nRECENT CONVERSATIONS:")
            for r in convs[:3]:
                lines.append(f"  · {r['text'][:100]}")
        return "\n".join(lines)

    async def consolidate_memory(self) -> None:
        """Weekly compression: summarize old conversations, remove low-importance entries."""
        logger.info("[Memory] Starting weekly consolidation...")
        old = self._vs.get_recent(hours=168)
        convs = [e for e in old if e["metadata"].get("type") == "conversation"]
        if len(convs) < 5:
            logger.info("[Memory] Insufficient conversations to consolidate — skipping")
            return
        combined = "\n".join(e["text"] for e in convs[:50])
        try:
            summary = await self._gc().generate_json(
                "You consolidate AI memory.",
                f"Summarize key facts, preferences, patterns from:\n{combined}\n"
                'Return JSON: {{"key_facts": [], "preferences": [], "patterns": [], "updated_goals": []}}',
                temperature=0.2,
            )
            await self.remember(str(summary), "consolidated_summary", "consolidation", importance=0.95)
            removed = 0
            for entry in convs:
                if entry["metadata"].get("importance", 1.0) < 0.4:
                    self._vs.delete(entry["id"])
                    removed += 1
            logger.info(f"[Memory] Consolidation done. {removed} low-importance entries removed.")
        except Exception as exc:
            logger.error(f"[Memory] Consolidation failed: {exc}")
