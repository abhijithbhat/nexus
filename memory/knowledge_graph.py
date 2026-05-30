"""SQLAlchemy knowledge graph — structured facts, entities, events, reflections."""
import json
from datetime import datetime, timedelta

from models.database import SessionLocal, Entity, Relationship, Fact, Event, Reflection
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """Manages structured long-term memory as a graph of facts and events."""

    # ── Entities ──────────────────────────────────────────────────────

    def add_entity(self, name: str, type_: str, description: str = "") -> int:
        """Add entity. Returns existing id if name already exists."""
        with SessionLocal() as db:
            existing = db.query(Entity).filter(Entity.name == name).first()
            if existing:
                return existing.id
            e = Entity(name=name, type=type_, description=description)
            db.add(e)
            db.commit()
            db.refresh(e)
            return e.id

    def add_relationship(self, entity_a: str, relationship: str, entity_b: str) -> None:
        """Create a directed relationship between two entities."""
        a_id = self.add_entity(entity_a, "unknown")
        b_id = self.add_entity(entity_b, "unknown")
        with SessionLocal() as db:
            r = Relationship(entity_a_id=a_id, relationship=relationship, entity_b_id=b_id)
            db.add(r)
            db.commit()

    # ── Facts ─────────────────────────────────────────────────────────

    def add_fact(self, subject: str, predicate: str, object_: str,
                 confidence: float = 1.0, source: str = "") -> None:
        """Store a knowledge triple."""
        with SessionLocal() as db:
            f = Fact(subject=subject, predicate=predicate, object=object_,
                     confidence=confidence, source=source)
            db.add(f)
            db.commit()

    def search_facts(self, query_term: str) -> list[Fact]:
        """Search facts by subject or object containing the query term."""
        with SessionLocal() as db:
            return db.query(Fact).filter(
                Fact.subject.contains(query_term) | Fact.object.contains(query_term)
            ).limit(20).all()

    # ── Events ────────────────────────────────────────────────────────

    def add_event(self, title: str, description: str, scheduled_at: datetime) -> int:
        """Store a scheduled event or deadline."""
        with SessionLocal() as db:
            e = Event(title=title, description=description, scheduled_at=scheduled_at)
            db.add(e)
            db.commit()
            db.refresh(e)
            logger.info(f"[KG] Event stored: '{title}' at {scheduled_at}")
            return e.id

    def update_event_status(self, event_id: int, status: str) -> None:
        """Update an event's status (pending → reminded → done → cancelled)."""
        with SessionLocal() as db:
            e = db.query(Event).filter(Event.id == event_id).first()
            if e:
                e.status = status
                db.commit()

    def get_upcoming_events(self, hours_ahead: int = 24) -> list[Event]:
        """Return pending events within the next N hours."""
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        with SessionLocal() as db:
            return (
                db.query(Event)
                .filter(Event.scheduled_at >= now, Event.scheduled_at <= cutoff, Event.status == "pending")
                .order_by(Event.scheduled_at)
                .all()
            )

    # ── Reflections ───────────────────────────────────────────────────

    def add_reflection(self, content: str, insights: dict) -> None:
        """Store a nightly self-reflection entry."""
        with SessionLocal() as db:
            r = Reflection(content=content, insights=json.dumps(insights))
            db.add(r)
            db.commit()

    def get_recent_reflections(self, days: int = 7) -> list[Reflection]:
        """Return reflections from the past N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with SessionLocal() as db:
            return (
                db.query(Reflection)
                .filter(Reflection.created_at >= cutoff)
                .order_by(Reflection.created_at.desc())
                .all()
            )

    # ── Profile assembly ──────────────────────────────────────────────

    def get_user_profile(self) -> dict:
        """Assemble a structured snapshot of what NEXUS knows about the user."""
        facts = self.search_facts(settings.user_name)
        upcoming = self.get_upcoming_events(hours_ahead=72)
        return {
            "name": settings.user_name,
            "facts": [(f.subject, f.predicate, f.object) for f in facts[:15]],
            "upcoming_events": [(e.title, str(e.scheduled_at)) for e in upcoming[:5]],
        }
