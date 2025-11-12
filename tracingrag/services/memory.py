"""Core memory service integrating PostgreSQL, Qdrant, and Neo4j"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tracingrag.services.embedding import generate_embedding, prepare_text_for_embedding
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB, TopicLatestStateDB, TraceDB
from tracingrag.storage.neo4j_client import (
    create_entity_relationship,
    create_evolution_edge,
    create_memory_node,
)
from tracingrag.storage.qdrant import upsert_embedding


class MemoryService:
    """Service for managing memory states across all storage layers"""

    async def create_memory_state(
        self,
        topic: str,
        content: str,
        parent_state_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        created_by: str | None = None,
        entity_type: str | None = None,
        entity_schema: dict[str, Any] | None = None,
        entities: list[tuple[str, str]] | None = None,  # [(entity_type, entity_name), ...]
    ) -> MemoryStateDB:
        """Create a new memory state

        Args:
            topic: Topic/key for the memory
            content: Main content of the memory
            parent_state_id: Optional parent state for evolution tracking
            metadata: Additional metadata
            tags: List of tags
            confidence: Confidence score (0-1)
            source: Source of the memory
            created_by: Creator identifier
            entity_type: Optional entity type
            entity_schema: Optional entity schema
            entities: List of related entities to link in graph

        Returns:
            Created MemoryStateDB instance
        """
        async with get_session() as session:
            # Determine version
            version = await self._get_next_version(session, topic)

            # Generate embedding
            text_for_embedding = await prepare_text_for_embedding(
                content,
                metadata={"topic": topic, "entity_type": entity_type, "tags": tags or []},
            )
            embedding = await generate_embedding(text_for_embedding)

            # Create database record
            state_id = uuid4()
            timestamp = datetime.utcnow()

            state = MemoryStateDB(
                id=state_id,
                topic=topic,
                content=content,
                version=version,
                timestamp=timestamp,
                embedding=embedding,
                parent_state_id=parent_state_id,
                metadata=metadata or {},
                tags=tags or [],
                confidence=confidence,
                source=source,
                created_by=created_by,
                entity_type=entity_type,
                entity_schema=entity_schema,
            )

            session.add(state)
            await session.flush()

            # Store embedding in Qdrant
            await upsert_embedding(
                state_id=state_id,
                embedding=embedding,
                payload={
                    "topic": topic,
                    "version": version,
                    "storage_tier": "active",
                    "entity_type": entity_type,
                    "is_consolidated": False,
                    "consolidation_level": 0,
                },
            )

            # Create graph node in Neo4j
            await create_memory_node(
                state_id=state_id,
                topic=topic,
                version=version,
                timestamp=timestamp.isoformat(),
                storage_tier="active",
                metadata=metadata or {},
            )

            # Create evolution edge if parent exists
            if parent_state_id:
                await create_evolution_edge(
                    parent_id=parent_state_id,
                    child_id=state_id,
                    relationship_type="EVOLVED_TO",
                    edge_properties={"timestamp": timestamp.isoformat()},
                )

            # Link entities in graph
            if entities:
                for entity_type_val, entity_name in entities:
                    await create_entity_relationship(
                        state_id=state_id,
                        entity_type=entity_type_val,
                        entity_name=entity_name,
                        relationship_type="MENTIONS",
                    )

            # Update or create trace
            await self._update_trace(session, topic, state_id)

            await session.commit()
            await session.refresh(state)

            return state

    async def get_memory_state(self, state_id: UUID) -> MemoryStateDB | None:
        """Get a memory state by ID

        Args:
            state_id: UUID of the memory state

        Returns:
            MemoryStateDB instance or None if not found
        """
        async with get_session() as session:
            result = await session.execute(
                select(MemoryStateDB).where(MemoryStateDB.id == state_id)
            )
            return result.scalar_one_or_none()

    async def get_latest_state(self, topic: str) -> MemoryStateDB | None:
        """Get the latest state for a topic (O(1) lookup)

        Args:
            topic: Topic to query

        Returns:
            Latest MemoryStateDB instance or None if not found
        """
        async with get_session() as session:
            # Use materialized view for O(1) lookup
            result = await session.execute(
                select(TopicLatestStateDB).where(TopicLatestStateDB.topic == topic)
            )
            latest_state_mapping = result.scalar_one_or_none()

            if latest_state_mapping is None:
                return None

            # Fetch the actual state
            state_result = await session.execute(
                select(MemoryStateDB).where(
                    MemoryStateDB.id == latest_state_mapping.latest_state_id
                )
            )
            return state_result.scalar_one_or_none()

    async def get_topic_history(
        self,
        topic: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryStateDB]:
        """Get version history for a topic

        Args:
            topic: Topic to query
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of MemoryStateDB instances ordered by version DESC
        """
        async with get_session() as session:
            result = await session.execute(
                select(MemoryStateDB)
                .where(MemoryStateDB.topic == topic)
                .order_by(MemoryStateDB.version.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def update_memory_state(
        self,
        state_id: UUID,
        access_count_increment: int = 0,
        importance_score: float | None = None,
        storage_tier: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryStateDB | None:
        """Update memory state metadata

        Args:
            state_id: UUID of the memory state
            access_count_increment: Amount to increment access count
            importance_score: New importance score
            storage_tier: New storage tier
            metadata: Metadata to merge with existing

        Returns:
            Updated MemoryStateDB instance or None if not found
        """
        async with get_session() as session:
            result = await session.execute(
                select(MemoryStateDB).where(MemoryStateDB.id == state_id)
            )
            state = result.scalar_one_or_none()

            if state is None:
                return None

            # Update fields
            if access_count_increment > 0:
                state.access_count += access_count_increment
                state.last_accessed = datetime.utcnow()

            if importance_score is not None:
                state.importance_score = importance_score

            if storage_tier is not None:
                state.storage_tier = storage_tier

            if metadata is not None:
                # Merge metadata
                current_metadata = state.metadata or {}
                current_metadata.update(metadata)
                state.metadata = current_metadata

            await session.commit()
            await session.refresh(state)

            return state

    async def _get_next_version(self, session: AsyncSession, topic: str) -> int:
        """Get the next version number for a topic

        Args:
            session: Database session
            topic: Topic to query

        Returns:
            Next version number
        """
        result = await session.execute(
            select(MemoryStateDB.version)
            .where(MemoryStateDB.topic == topic)
            .order_by(MemoryStateDB.version.desc())
            .limit(1)
        )
        latest_version = result.scalar_one_or_none()

        return (latest_version or 0) + 1

    async def _update_trace(self, session: AsyncSession, topic: str, state_id: UUID) -> None:
        """Update or create trace for a topic

        Args:
            session: Database session
            topic: Topic name
            state_id: UUID of the new state
        """
        result = await session.execute(select(TraceDB).where(TraceDB.topic == topic))
        trace = result.scalar_one_or_none()

        if trace is None:
            # Create new trace
            trace = TraceDB(
                id=uuid4(),
                topic=topic,
                state_ids=[state_id],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(trace)
        else:
            # Update existing trace
            state_ids = trace.state_ids or []
            state_ids.append(state_id)
            trace.state_ids = state_ids
            trace.updated_at = datetime.utcnow()

        await session.flush()
