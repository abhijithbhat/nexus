import json
from datetime import datetime, timedelta
from sqlalchemy import or_
from models.database import init_db, Entity, Relationship, Fact, Event, Reflection
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class KnowledgeGraph:
    def __init__(self):
        self.engine, self.Session = init_db(settings.sqlite_db_path)
        logger.info(f"SQLite KnowledgeGraph initialized at {settings.sqlite_db_path}")

    def add_entity(self, name: str, type: str, description: str) -> int:
        session = self.Session()
        try:
            entity = session.query(Entity).filter(Entity.name == name).first()
            if entity:
                entity.type = type
                entity.description = description
                entity.updated_at = datetime.utcnow()
                logger.info(f"Updated entity: {name} ({type})")
            else:
                entity = Entity(name=name, type=type, description=description)
                session.add(entity)
                logger.info(f"Added entity: {name} ({type})")
            session.commit()
            return entity.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding entity: {e}")
            raise e
        finally:
            session.close()

    def add_relationship(self, entity_a_name: str, relationship: str, entity_b_name: str):
        session = self.Session()
        try:
            ent_a = session.query(Entity).filter(Entity.name == entity_a_name).first()
            if not ent_a:
                ent_a = Entity(name=entity_a_name, type="unknown", description="")
                session.add(ent_a)
                session.flush()
                
            ent_b = session.query(Entity).filter(Entity.name == entity_b_name).first()
            if not ent_b:
                ent_b = Entity(name=entity_b_name, type="unknown", description="")
                session.add(ent_b)
                session.flush()
                
            rel = session.query(Relationship).filter(
                Relationship.entity_a_id == ent_a.id,
                Relationship.relationship == relationship,
                Relationship.entity_b_id == ent_b.id
            ).first()
            
            if not rel:
                rel = Relationship(entity_a_id=ent_a.id, relationship=relationship, entity_b_id=ent_b.id)
                session.add(rel)
                logger.info(f"Added relationship: {entity_a_name} -> {relationship} -> {entity_b_name}")
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding relationship: {e}")
            raise e
        finally:
            session.close()

    def add_fact(self, subject: str, predicate: str, object: str, confidence: float = 1.0, source: str = ""):
        session = self.Session()
        try:
            fact = session.query(Fact).filter(
                Fact.subject == subject,
                Fact.predicate == predicate,
                Fact.object == object
            ).first()
            
            if fact:
                fact.confidence = confidence
                fact.source = source
                logger.info(f"Updated fact: {subject} -> {predicate} -> {object}")
            else:
                fact = Fact(subject=subject, predicate=predicate, object=object, confidence=confidence, source=source)
                session.add(fact)
                logger.info(f"Added fact: {subject} -> {predicate} -> {object}")
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding fact: {e}")
            raise e
        finally:
            session.close()

    def add_event(self, title: str, description: str, scheduled_at: datetime) -> int:
        session = self.Session()
        try:
            event = Event(title=title, description=description, scheduled_at=scheduled_at, status="pending")
            session.add(event)
            session.commit()
            logger.info(f"Added event: {title} scheduled at {scheduled_at}")
            return event.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding event: {e}")
            raise e
        finally:
            session.close()

    def update_event_status(self, event_id: int, status: str):
        session = self.Session()
        try:
            event = session.query(Event).filter(Event.id == event_id).first()
            if event:
                event.status = status
                session.commit()
                logger.info(f"Updated event {event_id} status to {status}")
            else:
                logger.warning(f"Event {event_id} not found to update status")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating event status: {e}")
            raise e
        finally:
            session.close()

    def get_upcoming_events(self, hours_ahead: int = 24) -> list:
        session = self.Session()
        try:
            now = datetime.utcnow()
            future_limit = now + timedelta(hours=hours_ahead)
            events = session.query(Event).filter(
                Event.scheduled_at >= now,
                Event.scheduled_at <= future_limit,
                Event.status.in_(["pending", "reminded"])
            ).all()
            session.expunge_all()
            return events
        except Exception as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []
        finally:
            session.close()

    def add_reflection(self, content: str, insights: dict):
        session = self.Session()
        try:
            reflection = Reflection(content=content, insights=json.dumps(insights))
            session.add(reflection)
            session.commit()
            logger.info("Added nightly reflection database entry.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding reflection: {e}")
            raise e
        finally:
            session.close()

    def get_recent_reflections(self, days: int = 7) -> list:
        session = self.Session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            reflections = session.query(Reflection).filter(
                Reflection.created_at >= cutoff
            ).order_by(Reflection.created_at.desc()).all()
            session.expunge_all()
            return reflections
        except Exception as e:
            logger.error(f"Error getting recent reflections: {e}")
            return []
        finally:
            session.close()

    def get_user_profile(self) -> dict:
        session = self.Session()
        try:
            user_name = settings.user_name
            user_facts = session.query(Fact).filter(
                Fact.subject.like(f"%{user_name}%")
            ).all()
            
            projects = session.query(Entity).filter(
                Entity.type.in_(["project", "topic"])
            ).all()
            
            now = datetime.utcnow()
            upcoming_events = session.query(Event).filter(
                Event.scheduled_at >= now,
                Event.status == "pending"
            ).order_by(Event.scheduled_at.asc()).limit(10).all()
            
            facts_list = [{"subject": f.subject, "predicate": f.predicate, "object": f.object} for f in user_facts]
            projects_list = [{"name": p.name, "type": p.type, "description": p.description} for p in projects]
            events_list = [{"title": ev.title, "scheduled_at": ev.scheduled_at.isoformat(), "description": ev.description} for ev in upcoming_events]
            
            return {
                "facts": facts_list,
                "projects": projects_list,
                "upcoming_events": events_list
            }
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {"facts": [], "projects": [], "upcoming_events": []}
        finally:
            session.close()

    def search_facts(self, query_term: str) -> list:
        session = self.Session()
        try:
            facts = session.query(Fact).filter(
                or_(
                    Fact.subject.like(f"%{query_term}%"),
                    Fact.predicate.like(f"%{query_term}%"),
                    Fact.object.like(f"%{query_term}%")
                )
            ).limit(20).all()
            session.expunge_all()
            return facts
        except Exception as e:
            logger.error(f"Error searching facts: {e}")
            return []
        finally:
            session.close()
