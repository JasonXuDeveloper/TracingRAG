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
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from tracingrag.storage.database import Base


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID objects"""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


# Type compatibility layer for cross-database support
class FloatArrayType(TypeDecorator):
    """Stores Python list of floats as JSON string for SQLite, native ARRAY for PostgreSQL"""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Use native ARRAY type for PostgreSQL, Text for others"""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_ARRAY(Float))
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL handles arrays natively
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, encode as JSON string
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL returns arrays directly
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, decode from JSON string
        return json.loads(value)


class StringArrayType(TypeDecorator):
    """Stores Python list of strings as JSON string for SQLite, native ARRAY for PostgreSQL"""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Use native ARRAY type for PostgreSQL, Text for others"""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_ARRAY(String))
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL handles arrays natively
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, encode as JSON string
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL returns arrays directly
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, decode from JSON string
        return json.loads(value)


class UUIDArrayType(TypeDecorator):
    """Stores Python list of UUIDs as JSON string for SQLite, native ARRAY for PostgreSQL"""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Use native ARRAY type for PostgreSQL, Text for others"""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_ARRAY(PGUUID(as_uuid=True)))
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL handles UUID arrays natively
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, encode as JSON string with UUID conversion
        return json.dumps(value, cls=UUIDEncoder)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL returns arrays directly
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, decode from JSON string and convert to UUIDs
        data = json.loads(value)
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
    """Stores Python dict as JSON string for SQLite, native JSONB for PostgreSQL"""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Use native JSONB type for PostgreSQL, Text for others"""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL handles JSONB natively
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, encode as JSON string
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # PostgreSQL returns dicts directly
        if dialect.name == "postgresql":
            return value
        # For SQLite/others, decode from JSON string
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

    # Embedding (float array - ARRAY(Float) for PostgreSQL, JSON for SQLite)
    embedding = Column(FloatArrayType, nullable=True)

    # Trace relationship
    parent_state_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_states.id"), nullable=True)
    parent = relationship("MemoryStateDB", remote_side=[id], backref="children")

    # Metadata (JSONB for PostgreSQL, JSON string for SQLite)
    custom_metadata = Column("metadata", JSONEncodedDict, nullable=False, default=dict)
    tags = Column(StringArrayType, nullable=False, default=list)
    confidence = Column(Float, nullable=False, default=1.0)
    source = Column(String(255), nullable=True)
    created_by = Column(String(255), nullable=True)

    # Analytics and storage management
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime, nullable=False, default=datetime.utcnow)
    importance_score = Column(Float, nullable=False, default=0.5)
    storage_tier = Column(String(50), nullable=False, default="active")

    # Consolidation tracking (UUID array - ARRAY(UUID) for PostgreSQL, JSON for SQLite)
    consolidated_from = Column(UUIDArrayType, nullable=True)
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
    state_ids = Column(UUIDArrayType, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    custom_metadata = Column("metadata", JSONEncodedDict, nullable=False, default=dict)
    tags = Column(StringArrayType, nullable=False, default=list)
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
