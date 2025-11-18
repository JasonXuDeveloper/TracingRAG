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
)
from tracingrag.storage.qdrant import upsert_embedding
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryService:
    """Service for managing memory states across all storage layers"""

    async def _create_state_only(
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
    ) -> MemoryStateDB:
        """Create memory state only, without processing relationships (for batch scenarios)

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

        Returns:
            Created MemoryStateDB instance (without relationships processed)
        """
        async with get_session() as session:
            # Determine version
            version = await self._get_next_version(session, topic)

            # Auto-infer parent_state_id if not provided and this is not the first version
            # CRITICAL: Use Qdrant (not PostgreSQL) to find latest parent because:
            # 1. Qdrant updates are immediate (not in transaction)
            # 2. Avoids race conditions from concurrent state creation
            # 3. Respects is_latest flag which is set before PostgreSQL commit
            if parent_state_id is None and version > 1:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                from tracingrag.storage.qdrant import get_qdrant_client

                qdrant_client = get_qdrant_client()

                # Query Qdrant for the latest state of this topic
                results = qdrant_client.scroll(
                    collection_name="memory_states",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="topic", match=MatchValue(value=topic)),
                            FieldCondition(key="is_latest", match=MatchValue(value=True)),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )

                if results[0]:
                    latest_point = results[0][0]
                    parent_state_id = UUID(latest_point.id)
                    logger.info(
                        f"Auto-inferred parent_state_id for {topic} v{version} from Qdrant: {parent_state_id}"
                    )
                else:
                    # Fallback to PostgreSQL if Qdrant query fails
                    logger.warning(
                        f"Qdrant found no latest state for {topic}, falling back to PostgreSQL"
                    )
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
                            f"Auto-inferred parent_state_id for {topic} v{version} from PostgreSQL: {parent_state_id}"
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

            # Concurrently write to Qdrant and Neo4j (node only, relationships processed later)
            import asyncio

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
                        "is_latest": True,
                        "is_active": True,
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

            # Create evolution edge if parent exists
            if parent_state_id:
                await create_evolution_edge(
                    parent_id=parent_state_id,
                    child_id=state_id,
                    relationship_type="EVOLVED_TO",
                    edge_properties={"timestamp": timestamp.isoformat()},
                )

                # CRITICAL: Update parent state's flags immediately (before commit)
                # This ensures subsequent operations see the correct state
                # Update both Qdrant AND Neo4j
                try:
                    from tracingrag.storage.neo4j_client import (
                        update_memory_node_storage_tier,
                    )
                    from tracingrag.storage.qdrant import get_qdrant_client

                    qdrant_client = get_qdrant_client()

                    # Update Qdrant payload
                    qdrant_client.set_payload(
                        collection_name="memory_states",
                        payload={
                            "is_latest": False,
                            "is_active": False,
                            "storage_tier": "archived",
                        },
                        points=[str(parent_state_id)],
                    )

                    # Update Neo4j node storage_tier
                    await update_memory_node_storage_tier(
                        state_id=parent_state_id,
                        storage_tier="archived",
                    )

                    logger.debug(
                        f"Updated parent {parent_state_id} in Qdrant and Neo4j: is_active=False, storage_tier=archived"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update parent flags (non-critical): {e}")

            # Update trace
            await self._update_trace(session, topic, state_id)

            await session.commit()
            await session.refresh(state)

        return state

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
        """Create a new memory state (optimized: batch relationship management + directional cleanup)

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
        import asyncio

        # Step 1: Create primary state (without processing relationships)
        state = await self._create_state_only(
            topic=topic,
            content=content,
            parent_state_id=parent_state_id,
            metadata=metadata,
            tags=tags,
            confidence=confidence,
            source=source,
            created_by=created_by,
            entity_type=entity_type,
            entity_schema=entity_schema,
        )

        # Step 3: Cascading evolution (creates cascade states, returns state list)
        # Note: Cascade states are created WITHOUT relationship processing
        # All relationship processing happens in batch later
        cascaded_states = []
        if settings.enable_cascading_evolution and not (metadata or {}).get("cascading_evolution"):
            try:
                from tracingrag.services.cascading_evolution import get_cascading_evolution_manager

                cascading_manager = get_cascading_evolution_manager()
                cascade_result = await cascading_manager.trigger_related_evolutions(
                    new_state=state,
                    similarity_threshold=settings.cascading_evolution_similarity_threshold,
                )

                # Validate result structure
                if not isinstance(cascade_result, dict):
                    logger.error(
                        f"Cascade returned invalid type: {type(cascade_result)}, expected dict"
                    )
                    cascade_result = {"created_states": [], "statistics": {}}

                cascaded_states = cascade_result.get("created_states", [])
                logger.info(f"Cascading evolution created {len(cascaded_states)} states")
            except Exception as e:
                logger.error(f"Cascading evolution failed (non-critical): {e}")
                import traceback

                logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Step 4: Collect all new states for batch processing
        # Note: Promote mechanism is handled separately via promotion.py
        all_new_states = [state] + cascaded_states
        logger.info(f"📦 Collected {len(all_new_states)} states for batch processing")

        # Step 6: Batch concurrent relationship analysis (shared candidate cache)
        try:
            from tracingrag.services.relationship_manager import get_relationship_manager

            relationship_manager = get_relationship_manager()

            # Generate batch ID for Redis caching
            batch_id = f"{state.id}:{datetime.utcnow().timestamp()}"

            # Batch process relationships for all new states
            # (threshold read from settings.relationship_update_similarity_threshold)
            stats = await relationship_manager.batch_update_relationships(
                new_states=all_new_states,
                batch_id=batch_id,
            )

            logger.info(
                f"✅ Batch relationship management: {stats.get('total_relationships', 0)} relationships created"
            )

        except Exception as e:
            logger.error(f"Batch relationship management failed: {e}")

            # Fallback: process sequentially (old logic)
            logger.warning("Falling back to sequential relationship processing")
            for new_state in all_new_states:
                try:
                    if new_state.parent_state_id:
                        stats = await relationship_manager.update_relationships_on_evolution(
                            new_state=new_state,
                            parent_state_id=new_state.parent_state_id,
                            similarity_threshold=settings.relationship_update_similarity_threshold,
                        )
                    else:
                        stats = await relationship_manager.create_initial_relationships(
                            new_state=new_state,
                            similarity_threshold=settings.relationship_update_similarity_threshold,
                        )
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback relationship processing failed for {new_state.topic}: {fallback_e}"
                    )

        # Step 7: Batch cleanup all parent states' outgoing relationships in Neo4j
        # Note: Qdrant flags already updated in _create_state_only for each parent
        # Collect all parent_state_ids from all new states
        parent_ids_to_cleanup = []
        for new_state in all_new_states:
            if new_state.parent_state_id:
                parent_ids_to_cleanup.append(new_state.parent_state_id)

        if parent_ids_to_cleanup:
            logger.info(f"🗑️ Cleaning up {len(parent_ids_to_cleanup)} parent states...")
            try:
                from tracingrag.storage.neo4j_client import (
                    cleanup_old_version_outgoing_relationships,
                )

                # Cleanup all parent states' outgoing relationships concurrently
                cleanup_results = await asyncio.gather(
                    *[
                        cleanup_old_version_outgoing_relationships(
                            state_id=pid,
                            keep_relationship_types=["EVOLVED_TO", "BELONGS_TO"],
                        )
                        for pid in parent_ids_to_cleanup
                    ],
                    return_exceptions=True,
                )

                # Log results
                total_deleted = sum(
                    r.get("deleted_count", 0) for r in cleanup_results if isinstance(r, dict)
                )
                logger.info(
                    f"✅ Cleaned {len(parent_ids_to_cleanup)} parent states: "
                    f"{total_deleted} outgoing relationships deleted"
                )

            except Exception as e:
                logger.warning(f"Failed to cleanup parent relationships (non-critical): {e}")

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

    async def cleanup_all_data(self) -> dict[str, Any]:
        """Cleanup all TracingRAG data from all storage layers

        WARNING: This will permanently delete ALL TracingRAG data!
        - PostgreSQL: MemoryStateDB and TraceDB tables
        - Qdrant: memory_states collection
        - Neo4j: MemoryState nodes and their relationships
        - Redis: TracingRAG cache keys (prefix: tracingrag:*)

        Returns:
            Statistics about deleted data
        """
        from sqlalchemy import delete

        from tracingrag.storage.models import TraceDB
        from tracingrag.storage.neo4j_client import get_neo4j_driver
        from tracingrag.storage.qdrant import get_qdrant_client

        stats = {
            "postgresql": {"memories": 0, "traces": 0, "topic_latest_states": 0},
            "qdrant": {"points": 0},
            "neo4j": {"nodes": 0, "relationships": 0},
            "redis": {"keys": 0},
        }

        logger.warning("🗑️ Starting cleanup of ALL TracingRAG data from all databases...")

        # Step 1: Count and delete from PostgreSQL
        try:
            async with get_session() as session:
                # Count memories
                memories_count_result = await session.execute(select(func.count(MemoryStateDB.id)))
                stats["postgresql"]["memories"] = memories_count_result.scalar() or 0

                # Count traces
                traces_count_result = await session.execute(select(func.count(TraceDB.id)))
                stats["postgresql"]["traces"] = traces_count_result.scalar() or 0

                # Count topic_latest_states
                topic_latest_count_result = await session.execute(
                    select(func.count(TopicLatestStateDB.topic))
                )
                stats["postgresql"]["topic_latest_states"] = topic_latest_count_result.scalar() or 0

                # Delete in correct order (respect foreign key constraints):
                # 1. First delete topic_latest_states (references memory_states)
                await session.execute(delete(TopicLatestStateDB))

                # 2. Then delete traces
                await session.execute(delete(TraceDB))

                # 3. Finally delete memory_states
                await session.execute(delete(MemoryStateDB))

                await session.commit()

            logger.info(
                f"   ✓ PostgreSQL: Deleted {stats['postgresql']['memories']} memories, "
                f"{stats['postgresql']['traces']} traces, "
                f"{stats['postgresql']['topic_latest_states']} topic latest states"
            )
        except Exception as e:
            logger.error(f"   ✗ PostgreSQL cleanup failed: {e}")
            raise

        # Step 2: Delete from Qdrant (only memory_states collection)
        try:
            qdrant_client = get_qdrant_client()

            # Get count before deletion
            try:
                collection_info = qdrant_client.get_collection("memory_states")
                stats["qdrant"]["points"] = collection_info.points_count
            except Exception:
                stats["qdrant"]["points"] = 0

            # Delete collection and recreate
            from tracingrag.services.embedding import get_embedding_dimension

            embedding_dim = get_embedding_dimension()

            qdrant_client.delete_collection("memory_states")
            logger.info(f"   ✓ Qdrant: Deleted {stats['qdrant']['points']} points")

            # Recreate empty collection
            from tracingrag.storage.qdrant import init_qdrant_collection

            await init_qdrant_collection(
                collection_name="memory_states",
                vector_size=embedding_dim,
            )
            logger.info(f"   ✓ Qdrant: Recreated empty collection (dimension: {embedding_dim})")

        except Exception as e:
            logger.error(f"   ✗ Qdrant cleanup failed: {e}")

        # Step 3: Delete from Neo4j (only MemoryState nodes)
        try:
            driver = get_neo4j_driver()
            async with driver.session(database=settings.neo4j_database) as session:
                # Count MemoryState nodes and their relationships
                count_result = await session.run(
                    """
                    MATCH (m:MemoryState)
                    OPTIONAL MATCH (m)-[r]-()
                    RETURN count(DISTINCT m) AS node_count, count(DISTINCT r) AS rel_count
                    """
                )
                record = await count_result.single()
                stats["neo4j"]["nodes"] = record["node_count"] or 0
                stats["neo4j"]["relationships"] = record["rel_count"] or 0

                # Delete only MemoryState nodes and their relationships
                await session.run("MATCH (m:MemoryState) DETACH DELETE m")

            logger.info(
                f"   ✓ Neo4j: Deleted {stats['neo4j']['nodes']} MemoryState nodes, "
                f"{stats['neo4j']['relationships']} relationships"
            )

        except Exception as e:
            logger.error(f"   ✗ Neo4j cleanup failed: {e}")

        # Step 4: Clear Redis cache (only TracingRAG keys with prefix)
        try:
            from tracingrag.storage.redis_client import get_redis_client

            # Delete keys with TracingRAG prefix (e.g., "tracingrag:*", "candidates:*", etc.)
            redis_client = await get_redis_client()
            deleted_keys = 0
            prefixes = ["tracingrag:", "candidates:", "batch:"]

            for prefix in prefixes:
                keys = await redis_client.keys(f"{prefix}*")
                if keys:
                    await redis_client.delete(*keys)
                    deleted_keys += len(keys)

            stats["redis"]["keys"] = deleted_keys

            logger.info(f"   ✓ Redis: Deleted {stats['redis']['keys']} TracingRAG cache keys")

        except Exception as e:
            logger.error(f"   ✗ Redis cleanup failed: {e}")

        logger.warning("✅ All TracingRAG data cleanup complete!")

        return stats
