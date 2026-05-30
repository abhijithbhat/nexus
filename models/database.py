from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    type = Column(String)  # person, project, topic, place, tool
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Relationship(Base):
    __tablename__ = "relationships"
    id = Column(Integer, primary_key=True)
    entity_a_id = Column(Integer, ForeignKey("entities.id"))
    relationship = Column(String)  # "works_on", "knows", "uses", "attends"
    entity_b_id = Column(Integer, ForeignKey("entities.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True)
    subject = Column(String, nullable=False, index=True)
    predicate = Column(String, nullable=False)
    object = Column(String, nullable=False)
    confidence = Column(Float, default=1.0)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    scheduled_at = Column(DateTime, nullable=False)
    status = Column(String, default="pending")  # pending, reminded, done, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

class Reflection(Base):
    __tablename__ = "reflections"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    insights = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db(db_path: str):
    import os
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
