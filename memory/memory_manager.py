import json
from datetime import datetime, timedelta
from utils.gemini_client import GeminiClient
from utils.logger import get_logger
from memory.vector_store import VectorStore
from memory.knowledge_graph import KnowledgeGraph
from memory.importance_scorer import fast_score

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self):
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()
        self.gemini_client = GeminiClient()

    async def _score_importance(self, text: str, type: str, source: str) -> float:
        system_prompt = (
            "You are a memory relevance scorer for a personal AI assistant. "
            "Your job is to rate how important a piece of information is for long-term retention."
        )
        
        user_message = (
            f"Rate the long-term importance of storing this information.\n"
            f"Type: {type}\n"
            f"Source: {source}\n"
            f"Content: {text}\n\n"
            f"Scale:\n"
            f"0.0-0.3 = trivial or temporary (greetings, filler phrases, one-time requests)\n"
            f"0.4-0.6 = moderately useful (general knowledge, repeated topics)\n"
            f"0.7-0.9 = important (user preferences, project details, goals, deadlines)\n"
            f"0.9-1.0 = critical (key life goals, strong preferences, major decisions)\n\n"
            f"Return ONLY JSON: {{\"score\": 0.XX, \"reason\": \"one sentence\"}}"
        )
        
        try:
            result = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.1)
            score = float(result.get("score", 0.3))
            logger.info(f"Memory relevance score: {score:.2f} for type={type}, reason={result.get('reason')}")
            return score
        except Exception as e:
            logger.error(f"Error scoring importance: {e}. Defaulting to 0.3")
            return 0.3

    async def remember(self, text: str, type: str, source: str, importance: float = None) -> str:
        if importance is None:
            # Try fast rule-based scorer first (saves ~80% of Gemini calls)
            fast = fast_score(text, type, source)
            if fast is not None:
                importance = fast
                logger.info(f"Fast-scored importance: {importance:.2f} for type={type}")
            else:
                # Default to 0.4 for ambiguous cases instead of calling Gemini
                # This saves 1 API call per message on the free tier
                importance = 0.4
                logger.info(f"Default importance: {importance:.2f} for ambiguous type={type}")
            
        timestamp = datetime.utcnow().isoformat()
        metadata = {
            "type": type,
            "source": source,
            "timestamp": timestamp,
            "importance": importance
        }
        
        # Save to vector store
        doc_id = self.vector_store.add(text, metadata)
        
        # If importance > 0.7, extract entities and facts
        if importance >= 0.7:
            logger.info(f"Extracting entities & facts for high-importance memory: {doc_id}")
            extract_system = (
                "You are an information extraction assistant. Your job is to extract entities and facts from the text.\n"
                "Entity types: person, project, topic, place, tool\n"
                "Fact format: subject, predicate, object (e.g. User, uses, Python)\n"
                "Return ONLY JSON."
            )
            extract_user = (
                f"Extract entities and facts from:\n"
                f"'{text}'\n\n"
                f"JSON format:\n"
                f"{{\n"
                f"  \"entities\": [\n"
                f"    {{\"name\": \"entity name\", \"type\": \"type\", \"description\": \"brief description\"}}\n"
                f"  ],\n"
                f"  \"facts\": [\n"
                f"    {{\"subject\": \"subject\", \"predicate\": \"predicate\", \"object\": \"object\"}}\n"
                f"  ]\n"
                f"}}"
            )
            
            try:
                extracted = await self.gemini_client.generate_json(extract_system, extract_user, temperature=0.1)
                
                # Add to knowledge graph
                for entity in extracted.get("entities", []):
                    self.knowledge_graph.add_entity(
                        name=entity["name"],
                        type=entity["type"],
                        description=entity["description"]
                    )
                for fact in extracted.get("facts", []):
                    self.knowledge_graph.add_fact(
                        subject=fact["subject"],
                        predicate=fact["predicate"],
                        object=fact["object"],
                        confidence=0.9,
                        source=f"vector_store:{doc_id}"
                    )
            except Exception as e:
                logger.error(f"Failed to extract entities/facts: {e}")
                
        return doc_id

    async def recall(self, query: str, limit: int = 8) -> str:
        # Search vector store
        vec_results = self.vector_store.search(query, n_results=limit)
        
        # Search knowledge graph
        facts = self.knowledge_graph.search_facts(query)
        
        if not vec_results and not facts:
            return "No relevant memory found for this query."
            
        context_lines = ["[What I remember about this]"]
        
        # Format vector memories
        for res in vec_results:
            meta = res["metadata"]
            t_str = meta.get("timestamp_iso", "").split("T")[0] if "timestamp_iso" in meta else ""
            context_lines.append(f"• [{meta.get('type')}] {res['text']} ({t_str}, from: {meta.get('source')})")
            
        # Format facts
        for f in facts:
            context_lines.append(f"• [fact] {f.subject} → {f.predicate} → {f.object} (confidence: {f.confidence:.1f})")
            
        return "\n".join(context_lines)

    async def get_full_user_context(self) -> str:
        profile = self.knowledge_graph.get_user_profile()
        reflections = self.knowledge_graph.get_recent_reflections(days=3)
        
        # Recall recent conversations from last 48 hours
        recent_mems = self.vector_store.get_recent(hours=48)
        recent_chats = [m for m in recent_mems if m["metadata"].get("type") == "conversation"]
        # Limit to top 5 recent
        recent_chats = recent_chats[-5:]
        
        context_parts = []
        
        # User profile
        context_parts.append("[USER PROFILE FACT GRAPH]")
        if profile["facts"]:
            for f in profile["facts"]:
                context_parts.append(f"- {f['subject']} {f['predicate']} {f['object']}")
        else:
            context_parts.append("- No core profile facts recorded yet.")
            
        context_parts.append("\n[USER PROJECTS]")
        if profile["projects"]:
            for p in profile["projects"]:
                context_parts.append(f"- {p['name']} ({p['type']}): {p['description']}")
        else:
            context_parts.append("- No user projects noted.")
            
        context_parts.append("\n[UPCOMING EVENTS]")
        if profile["upcoming_events"]:
            for ev in profile["upcoming_events"]:
                context_parts.append(f"- Event: {ev['title']} at {ev['scheduled_at']} ({ev.get('description') or 'no desc'})")
        else:
            context_parts.append("- No upcoming events scheduled.")
            
        context_parts.append("\n[NIGHTLY REFLECTION INSIGHTS (Last 3 Days)]")
        if reflections:
            for r in reflections[:3]:
                date_str = r.created_at.strftime("%Y-%m-%d")
                try:
                    insights = json.loads(r.insights)
                    changes = ", ".join(insights.get("tomorrow_changes", []))
                    context_parts.append(f"- {date_str}: Focus changes -> {changes}")
                except Exception:
                    context_parts.append(f"- {date_str}: {r.content[:150]}...")
        else:
            context_parts.append("- No nightly reflections found.")
            
        context_parts.append("\n[RECENT CHAT HISTORY]")
        if recent_chats:
            for chat in recent_chats:
                context_parts.append(f"- {chat['text']}")
        else:
            context_parts.append("- No recent conversation history found.")
            
        return "\n".join(context_parts)

    async def consolidate_memory(self):
        logger.info("Starting memory consolidation job...")
        try:
            all_conversations = self.vector_store.get_by_type("conversation", limit=500)
            cutoff = datetime.utcnow() - timedelta(days=7)
            
            old_convs = [
                c for c in all_conversations
                if float(c["metadata"].get("timestamp", 0)) < cutoff.timestamp()
            ]
            
            if not old_convs:
                logger.info("No old conversations to consolidate.")
                return
            
            logger.info(f"Consolidating {len(old_convs)} old conversations in chunks...")
            
            # CHUNK into groups of 30 to avoid token overflow
            CHUNK_SIZE = 30
            chunk_summaries = []
            
            for i in range(0, len(old_convs), CHUNK_SIZE):
                chunk = old_convs[i:i + CHUNK_SIZE]
                chunk_blob = "\n".join([
                    f"({c['metadata'].get('timestamp_iso', '')}): {c['text']}"
                    for c in chunk
                ])
                
                system_prompt = (
                    "Compress these conversation logs into key facts and preferences. "
                    "Be highly concise. Return JSON only."
                )
                user_message = (
                    f"Conversations:\n{chunk_blob}\n\n"
                    f"Return JSON: {{\"key_facts\": [], \"preferences\": [], \"patterns\": []}}"
                )
                
                try:
                    chunk_result = await self.gemini_client.generate_json(
                        system_prompt, user_message, temperature=0.1
                    )
                    chunk_summaries.append(chunk_result)
                except Exception as e:
                    logger.error(f"Error consolidating chunk {i}: {e}")
                    continue
            
            if not chunk_summaries:
                return
            
            # Merge all chunk summaries
            all_facts = []
            all_prefs = []
            all_patterns = []
            for cs in chunk_summaries:
                all_facts.extend(cs.get("key_facts", []))
                all_prefs.extend(cs.get("preferences", []))
                all_patterns.extend(cs.get("patterns", []))
            
            summary_text = (
                f"Consolidated memory summary (older than 7 days):\n"
                f"Facts: {', '.join(str(f) for f in all_facts[:20])}\n"
                f"Preferences: {', '.join(str(p) for p in all_prefs[:20])}\n"
                f"Patterns: {', '.join(str(p) for p in all_patterns[:20])}"
            )
            
            await self.remember(summary_text, "consolidated_summary", "consolidation_job", importance=0.95)
            
            # Delete low-importance old conversations
            deleted_count = 0
            for conv in old_convs:
                if float(conv["metadata"].get("importance", 0.0)) < 0.5:
                    self.vector_store.delete(conv["id"])
                    deleted_count += 1
            
            logger.info(f"Consolidation complete. Deleted {deleted_count} low-importance memories.")
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
