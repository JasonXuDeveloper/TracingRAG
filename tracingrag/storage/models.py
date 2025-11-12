"""SQLAlchemy models for PostgreSQL database"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from tracingrag.storage.database import Base


class MemoryStateDB(Base):
    """SQLAlchemy model for memory states"""

    __tablename__ = "memory_states"

    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    topic = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Embedding (stored as array for compatibility)
    embedding = Column(ARRAY(Float), nullable=True)

    # Trace relationship
    parent_state_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_states.id"), nullable=True)
    parent = relationship("MemoryStateDB", remote_side=[id], backref="children")

    # Metadata
    metadata = Column(JSONB, nullable=False, default=dict)
    tags = Column(ARRAY(String), nullable=False, default=list)
    confidence = Column(Float, nullable=False, default=1.0)
    source = Column(String(255), nullable=True)
    created_by = Column(String(255), nullable=True)

    # Analytics and storage management
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime, nullable=False, default=datetime.utcnow)
    importance_score = Column(Float, nullable=False, default=0.5)
    storage_tier = Column(String(50), nullable=False, default="active")

    # Consolidation tracking
    consolidated_from = Column(ARRAY(PGUUID(as_uuid=True)), nullable=True)
    is_consolidated = Column(Boolean, nullable=False, default=False)
    consolidation_level = Column(Integer, nullable=False, default=0)

    # Diff-based storage
    diff_from_parent = Column(Text, nullable=True)
    is_delta = Column(Boolean, nullable=False, default=False)

    # Optional entity typing
    entity_type = Column(String(100), nullable=True, index=True)
    entity_schema = Column(JSONB, nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index("ix_memory_states_topic_version", "topic", "version"),
        Index("ix_memory_states_entity_type", "entity_type"),
        Index("ix_memory_states_storage_tier", "storage_tier"),
        Index("ix_memory_states_is_consolidated", "is_consolidated"),
        Index("ix_memory_states_tags", "tags", postgresql_using="gin"),
        Index("ix_memory_states_metadata", "metadata", postgresql_using="gin"),
    )


class TraceDB(Base):
    """SQLAlchemy model for traces"""

    __tablename__ = "traces"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    topic = Column(String(500), nullable=False, unique=True, index=True)
    state_ids = Column(ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSONB, nullable=False, default=dict)
    tags = Column(ARRAY(String), nullable=False, default=list)
    is_active = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_traces_is_active", "is_active"),
        Index("ix_traces_tags", "tags", postgresql_using="gin"),
    )


class TopicLatestStateDB(Base):
    """
    Materialized view for O(1) latest state lookup.
    Maps topic -> latest_state_id for instant access.
    """

    __tablename__ = "topic_latest_states"

    topic = Column(String(500), primary_key=True, index=True)
    latest_state_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_states.id"), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to actual state
    latest_state = relationship("MemoryStateDB", foreign_keys=[latest_state_id])
