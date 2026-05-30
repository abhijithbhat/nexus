"""SQLAlchemy ORM models for NEXUS structured knowledge storage."""
import os
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from utils.config import settings

# Ensure data directory exists before creating engine
os.makedirs(os.path.dirname(os.path.abspath(settings.sqlite_db_path)), exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.sqlite_db_path}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Entity(Base):
    """A named thing NEXUS knows about: person, project, tool, topic, place."""
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(256), nullable=False, index=True)
    type = Column(String(64))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Relationship(Base):
    """Directed relationship between two entities."""
    __tablename__ = "relationships"
    id = Column(Integer, primary_key=True)
    entity_a_id = Column(Integer, ForeignKey("entities.id"))
    relationship = Column(String(128))
    entity_b_id = Column(Integer, ForeignKey("entities.id"))
    created_at = Column(DateTime, default=datetime.utcnow)


class Fact(Base):
    """A discrete knowledge triple: subject → predicate → object."""
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True)
    subject = Column(String(256), nullable=False, index=True)
    predicate = Column(String(256), nullable=False)
    object = Column(String(512), nullable=False)
    confidence = Column(Float, default=1.0)
    source = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class Event(Base):
    """A scheduled event or deadline that NEXUS tracks and reminds about."""
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    title = Column(String(256), nullable=False)
    description = Column(Text, default="")
    scheduled_at = Column(DateTime)
    status = Column(String(32), default="pending")  # pending | reminded | done | cancelled
    created_at = Column(DateTime, default=datetime.utcnow)


class Reflection(Base):
    """Nightly self-evaluation records."""
    __tablename__ = "reflections"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    insights = Column(Text)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables() -> None:
    """Create all tables if they do not already exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
