"""Core memory service integrating PostgreSQL, Qdrant, and Neo4j"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tracingrag.config import settings
from tracingrag.services.embedding import generate_embedding, prepare_text_for_embedding
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB, TopicLatestStateDB, TraceDB
from tracingrag.storage.neo4j_client import (
    create_evolution_edge,
    create_memory_node,
    inherit_parent_relationships,
)
from tracingrag.storage.qdrant import upsert_embedding
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


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

            # Auto-infer parent_state_id if not provided and this is not the first version
            if parent_state_id is None and version > 1:
                # Get the latest state for this topic to use as parent
                result = await session.execute(
                    select(MemoryStateDB)
                    .where(MemoryStateDB.topic == topic)
                    .order_by(MemoryStateDB.version.desc())
                    .limit(1)
                )
                latest_state = result.scalar_one_or_none()
                if latest_state:
                    parent_state_id = latest_state.id
                    logger.info(
                        f"Auto-inferred parent_state_id for {topic} v{version}: {parent_state_id}"
                    )

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
                custom_metadata=metadata or {},
                tags=tags or [],
                confidence=confidence,
                source=source,
                created_by=created_by,
                entity_type=entity_type,
                entity_schema=entity_schema,
            )

            session.add(state)
            await session.flush()

            # Concurrently write to Qdrant and Neo4j (PERFORMANCE OPTIMIZATION!)
            import asyncio

            # First: Create node in parallel with embedding
            await asyncio.gather(
                # Store embedding in Qdrant
                upsert_embedding(
                    state_id=state_id,
                    embedding=embedding,
                    payload={
                        "topic": topic,
                        "version": version,
                        "storage_tier": "active",
                        "entity_type": entity_type,
                        "is_consolidated": False,
                        "consolidation_level": 0,
                        "is_latest": True,  # New states are always latest
                        "is_active": True,  # New states are active by default
                    },
                ),
                # Create graph node in Neo4j
                create_memory_node(
                    state_id=state_id,
                    topic=topic,
                    version=version,
                    timestamp=timestamp.isoformat(),
                    storage_tier="active",
                    metadata=metadata or {},
                ),
            )

            # Second: Create evolution edge AFTER node is created
            # IMPORTANT: Must happen after create_memory_node completes, otherwise MATCH will fail
            if parent_state_id:
                await create_evolution_edge(
                    parent_id=parent_state_id,
                    child_id=state_id,
                    relationship_type="EVOLVED_TO",
                    edge_properties={"timestamp": timestamp.isoformat()},
                )

            # Update parent state's is_latest/is_active to False + archive storage_tier
            if parent_state_id:
                try:
                    from tracingrag.storage.qdrant import get_qdrant_client

                    qdrant_client = get_qdrant_client()
                    qdrant_client.set_payload(
                        collection_name="memory_states",
                        payload={
                            "is_latest": False,
                            "is_active": False,
                            "storage_tier": "archived",  # Archive old versions
                        },
                        points=[str(parent_state_id)],
                    )
                    logger.debug(
                        f"Updated parent {parent_state_id}: is_active=False, storage_tier=archived"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update parent flags (non-critical): {e}")

            # Note: Relationship inheritance handled by RelationshipManager (after transaction)

            # Update or create trace
            await self._update_trace(session, topic, state_id)

            await session.commit()
            await session.refresh(state)

        # Cascading evolution - evolve related topics based on new memory (BEFORE relationship updates)
        # This must run BEFORE RelationshipManager so the newly evolved states are included in relationship analysis
        if settings.enable_cascading_evolution and not metadata.get("cascading_evolution"):
            try:
                from tracingrag.services.cascading_evolution import get_cascading_evolution_manager

                cascading_manager = get_cascading_evolution_manager()
                evolution_stats = await cascading_manager.trigger_related_evolutions(
                    new_state=state,
                    similarity_threshold=settings.cascading_evolution_similarity_threshold,
                )
                logger.info(
                    f"Cascading evolution completed: "
                    f"{len(evolution_stats['evolved_topics'])} topics evolved, "
                    f"{len(evolution_stats['skipped_topics'])} skipped"
                )
            except Exception as e:
                # Don't fail memory creation if cascading evolution fails
                logger.error(f"Cascading evolution failed (non-critical): {e}")

        # Intelligent relationship management (outside transaction) - always enabled
        try:
            from tracingrag.services.relationship_manager import get_relationship_manager

            relationship_manager = get_relationship_manager()

            # Evolution scenario: Update relationships based on parent
            if parent_state_id:
                stats = await relationship_manager.update_relationships_on_evolution(
                    new_state=state,
                    parent_state_id=parent_state_id,
                    similarity_threshold=settings.relationship_update_similarity_threshold,
                )
                logger.info(
                    f"Intelligent relationship update completed: "
                    f"{stats['kept']} kept, {stats['updated']} updated, "
                    f"{stats['created']} created, {stats['deleted']} deleted"
                )
            # First-time creation scenario: Create initial relationships
            else:
                stats = await relationship_manager.create_initial_relationships(
                    new_state=state,
                    similarity_threshold=settings.relationship_update_similarity_threshold,
                )
                logger.info(f"Initial relationships created: {stats['created']} relationships")
        except Exception as e:
            # Fallback to simple approach if intelligent update fails
            logger.error(f"Intelligent relationship management failed: {e}")

            if parent_state_id:
                # Fallback for evolution: simple inheritance
                logger.info("Falling back to simple inheritance")
                try:
                    inherited_count = await inherit_parent_relationships(
                        parent_id=parent_state_id,
                        child_id=state.id,
                    )
                    if inherited_count > 0:
                        logger.info(f"Inherited {inherited_count} relationships (fallback)")
                except Exception as fallback_e:
                    logger.error(f"Fallback inheritance also failed: {fallback_e}")
            else:
                # No fallback - RelationshipManager handles all relationships
                logger.info("No parent relationships (first version of this topic)")

        # Note: Auto-linking removed - RelationshipManager handles all relationships intelligently

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

    async def count_states(self, latest_only: bool = False, storage_tier: str | None = None) -> int:
        """Count total number of memory states in database

        Args:
            latest_only: If True, only count latest states per topic
            storage_tier: If provided, only count states with this storage tier (e.g., "active", "archived")

        Returns:
            Total count of memory states
        """
        async with get_session() as session:
            if latest_only:
                # Count distinct topics (one per topic)
                query = select(func.count(func.distinct(MemoryStateDB.topic)))
                if storage_tier:
                    query = query.where(MemoryStateDB.storage_tier == storage_tier)
                result = await session.execute(query)
            else:
                # Count all states
                query = select(func.count()).select_from(MemoryStateDB)
                if storage_tier:
                    query = query.where(MemoryStateDB.storage_tier == storage_tier)
                result = await session.execute(query)
            count = result.scalar_one()
            logger.info(
                f"count_states: {count} (latest_only={latest_only}, storage_tier={storage_tier})"
            )
            return count

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
                current_metadata = state.custom_metadata or {}
                current_metadata.update(metadata)
                state.custom_metadata = current_metadata

            await session.commit()
            await session.refresh(state)

            return state

    async def delete_memory_state(self, state_id: UUID) -> bool:
        """Delete a memory state from all storage layers

        Args:
            state_id: UUID of the memory state to delete

        Returns:
            True if deleted successfully, False if not found
        """
        from sqlalchemy import delete

        from tracingrag.storage.neo4j_client import delete_memory_node
        from tracingrag.storage.qdrant import delete_embedding

        async with get_session() as session:
            # Get the state first to check if it exists
            result = await session.execute(
                select(MemoryStateDB).where(MemoryStateDB.id == state_id)
            )
            state = result.scalar_one_or_none()

            if state is None:
                return False

            # Delete from topic_latest_states first (foreign key constraint)
            await session.execute(
                delete(TopicLatestStateDB).where(TopicLatestStateDB.latest_state_id == state_id)
            )
            logger.info(f"Deleted from topic_latest_states: {state_id}")

            # Delete from PostgreSQL
            await session.delete(state)

            # Delete from Qdrant
            try:
                await delete_embedding(state_id=state_id)
                logger.info(f"Deleted embedding from Qdrant: {state_id}")
            except Exception as e:
                logger.error(f"Failed to delete from Qdrant (non-critical): {e}")

            # Delete from Neo4j
            try:
                await delete_memory_node(state_id=state_id)
                logger.info(f"Deleted node from Neo4j: {state_id}")
            except Exception as e:
                logger.error(f"Failed to delete from Neo4j (non-critical): {e}")

            await session.commit()

            logger.info(f"Successfully deleted memory state: {state_id} (topic: {state.topic})")
            return True

    async def _get_next_version(self, session: AsyncSession, topic: str) -> int:
        """Get the next version number for a topic (with row-level lock to prevent concurrent conflicts)

        Args:
            session: Database session
            topic: Topic to query

        Returns:
            Next version number
        """
        # Use SELECT FOR UPDATE to lock the latest version row
        # This prevents concurrent transactions from getting the same version number
        result = await session.execute(
            select(MemoryStateDB.version)
            .where(MemoryStateDB.topic == topic)
            .order_by(MemoryStateDB.version.desc())
            .limit(1)
            .with_for_update()  # Row-level lock
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
