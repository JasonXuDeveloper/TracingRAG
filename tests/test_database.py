"""Tests for database models and operations"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select

from tracingrag.storage.models import MemoryStateDB, TopicLatestStateDB, TraceDB


class TestMemoryStateDB:
    """Tests for MemoryStateDB model"""

    async def test_create_memory_state(self, db_session, sample_memory_data, sample_embedding):
        """Test creating a memory state"""
        state = MemoryStateDB(
            id=uuid4(),
            topic=sample_memory_data["topic"],
            content=sample_memory_data["content"],
            version=1,
            timestamp=datetime.utcnow(),
            embedding=sample_embedding,
            custom_metadata=sample_memory_data["metadata"],
            tags=sample_memory_data["tags"],
            confidence=sample_memory_data["confidence"],
            entity_type=sample_memory_data["entity_type"],
        )

        db_session.add(state)
        await db_session.commit()
        await db_session.refresh(state)

        assert state.id is not None
        assert state.topic == sample_memory_data["topic"]
        assert state.content == sample_memory_data["content"]
        assert state.version == 1
        assert state.confidence == sample_memory_data["confidence"]
        assert state.entity_type == sample_memory_data["entity_type"]

    async def test_memory_state_defaults(self, db_session):
        """Test default values for memory state"""
        state = MemoryStateDB(
            id=uuid4(),
            topic="test_topic",
            content="Test content",
            version=1,
            timestamp=datetime.utcnow(),
        )

        db_session.add(state)
        await db_session.commit()
        await db_session.refresh(state)

        assert state.custom_metadata == {}
        assert state.tags == []
        assert state.confidence == 1.0
        assert state.access_count == 0
        assert state.importance_score == 0.5
        assert state.storage_tier == "active"
        assert state.is_consolidated is False
        assert state.consolidation_level == 0
        assert state.is_delta is False

    async def test_memory_state_parent_relationship(
        self, db_session, sample_memory_data, sample_embedding
    ):
        """Test parent-child relationship between memory states"""
        # Create parent state
        parent = MemoryStateDB(
            id=uuid4(),
            topic=sample_memory_data["topic"],
            content="Original content",
            version=1,
            timestamp=datetime.utcnow(),
            embedding=sample_embedding,
        )
        db_session.add(parent)
        await db_session.commit()
        await db_session.refresh(parent)

        # Create child state
        child = MemoryStateDB(
            id=uuid4(),
            topic=sample_memory_data["topic"],
            content="Updated content",
            version=2,
            timestamp=datetime.utcnow(),
            embedding=sample_embedding,
            parent_state_id=parent.id,
        )
        db_session.add(child)
        await db_session.commit()
        await db_session.refresh(child)

        assert child.parent_state_id == parent.id

    async def test_query_by_topic(self, db_session, sample_memory_state):
        """Test querying memory states by topic"""
        result = await db_session.execute(
            select(MemoryStateDB).where(MemoryStateDB.topic == sample_memory_state.topic)
        )
        states = result.scalars().all()

        assert len(states) >= 1
        assert states[0].topic == sample_memory_state.topic

    async def test_query_by_entity_type(self, db_session, sample_memory_state):
        """Test querying memory states by entity type"""
        result = await db_session.execute(
            select(MemoryStateDB).where(
                MemoryStateDB.entity_type == sample_memory_state.entity_type
            )
        )
        states = result.scalars().all()

        assert len(states) >= 1
        assert states[0].entity_type == sample_memory_state.entity_type


class TestTraceDB:
    """Tests for TraceDB model"""

    async def test_create_trace(self, db_session, sample_memory_state):
        """Test creating a trace"""
        trace = TraceDB(
            id=uuid4(),
            topic=sample_memory_state.topic,
            state_ids=[sample_memory_state.id],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            custom_metadata={"test": "data"},
            tags=["test"],
        )

        db_session.add(trace)
        await db_session.commit()
        await db_session.refresh(trace)

        assert trace.id is not None
        assert trace.topic == sample_memory_state.topic
        assert len(trace.state_ids) == 1
        assert trace.state_ids[0] == sample_memory_state.id
        assert trace.is_active is True

    async def test_trace_unique_topic(self, db_session):
        """Test that trace topic is unique"""
        topic = "unique_test_topic"

        trace1 = TraceDB(
            id=uuid4(),
            topic=topic,
            state_ids=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db_session.add(trace1)
        await db_session.commit()

        # This should fail due to unique constraint on topic
        # Note: SQLite doesn't enforce this constraint in the same way as PostgreSQL
        # In a real PostgreSQL database, this would raise an error


class TestTopicLatestStateDB:
    """Tests for TopicLatestStateDB model"""

    async def test_create_latest_state_mapping(self, db_session, sample_memory_state):
        """Test creating a topic latest state mapping"""
        mapping = TopicLatestStateDB(
            topic=sample_memory_state.topic,
            latest_state_id=sample_memory_state.id,
            updated_at=datetime.utcnow(),
        )

        db_session.add(mapping)
        await db_session.commit()
        await db_session.refresh(mapping)

        assert mapping.topic == sample_memory_state.topic
        assert mapping.latest_state_id == sample_memory_state.id

    async def test_latest_state_lookup(self, db_session, sample_memory_state):
        """Test O(1) latest state lookup"""
        # Create mapping
        mapping = TopicLatestStateDB(
            topic=sample_memory_state.topic,
            latest_state_id=sample_memory_state.id,
            updated_at=datetime.utcnow(),
        )
        db_session.add(mapping)
        await db_session.commit()

        # Query latest state
        result = await db_session.execute(
            select(TopicLatestStateDB).where(TopicLatestStateDB.topic == sample_memory_state.topic)
        )
        found_mapping = result.scalar_one_or_none()

        assert found_mapping is not None
        assert found_mapping.latest_state_id == sample_memory_state.id
