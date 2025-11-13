"""SQLAlchemy models for PostgreSQL database"""

import json
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
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from tracingrag.storage.database import Base


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID objects"""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


# Type compatibility layer for SQLite
class JSONEncodedList(TypeDecorator):
    """Stores Python list as JSON string for SQLite compatibility"""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # Use custom encoder to handle UUIDs
        return json.dumps(value, cls=UUIDEncoder)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        data = json.loads(value)
        # Convert string UUIDs back to UUID objects if they look like UUIDs
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str) and len(item) == 36:
                    try:
                        result.append(UUID(item))
                    except (ValueError, AttributeError):
                        result.append(item)
                else:
                    result.append(item)
            return result
        return data


class JSONEncodedDict(TypeDecorator):
    """Stores Python dict as JSON string for SQLite compatibility"""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)


class MemoryStateDB(Base):
    """SQLAlchemy model for memory states"""

    __tablename__ = "memory_states"

    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    topic = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Embedding (using JSONEncodedList for SQLite compatibility)
    embedding = Column(JSONEncodedList, nullable=True)

    # Trace relationship
    parent_state_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_states.id"), nullable=True)
    parent = relationship("MemoryStateDB", remote_side=[id], backref="children")

    # Metadata (using JSONEncodedDict for SQLite compatibility)
    custom_metadata = Column("metadata", JSONEncodedDict, nullable=False, default=dict)
    tags = Column(JSONEncodedList, nullable=False, default=list)
    confidence = Column(Float, nullable=False, default=1.0)
    source = Column(String(255), nullable=True)
    created_by = Column(String(255), nullable=True)

    # Analytics and storage management
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime, nullable=False, default=datetime.utcnow)
    importance_score = Column(Float, nullable=False, default=0.5)
    storage_tier = Column(String(50), nullable=False, default="active")

    # Consolidation tracking (using JSONEncodedList for SQLite compatibility)
    consolidated_from = Column(JSONEncodedList, nullable=True)
    is_consolidated = Column(Boolean, nullable=False, default=False)
    consolidation_level = Column(Integer, nullable=False, default=0)

    # Diff-based storage
    diff_from_parent = Column(Text, nullable=True)
    is_delta = Column(Boolean, nullable=False, default=False)

    # Optional entity typing
    entity_type = Column(String(100), nullable=True, index=True)
    entity_schema = Column(JSONEncodedDict, nullable=True)

    # Indexes for common queries (excluding PostgreSQL-specific GIN indexes for SQLite compatibility)
    # Note: entity_type already has index=True on the column definition, so it's excluded here
    __table_args__ = (
        Index("ix_memory_states_topic_version", "topic", "version"),
        # Index for storage_tier removed as it's rarely queried alone
        # Index for is_consolidated removed as it's rarely queried alone
        # Note: GIN indexes for tags and metadata are PostgreSQL-specific and excluded for SQLite compatibility
    )


class TraceDB(Base):
    """SQLAlchemy model for traces"""

    __tablename__ = "traces"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    topic = Column(String(500), nullable=False, unique=True, index=True)
    state_ids = Column(JSONEncodedList, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    custom_metadata = Column("metadata", JSONEncodedDict, nullable=False, default=dict)
    tags = Column(JSONEncodedList, nullable=False, default=list)
    is_active = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_traces_is_active", "is_active"),
        # Note: GIN index for tags is PostgreSQL-specific and excluded for SQLite compatibility
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
