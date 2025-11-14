"""Core memory service integrating PostgreSQL, Qdrant, and Neo4j"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tracingrag.config import settings
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

        # Auto-link related memories (outside transaction to avoid blocking)
        try:
            linked = await self.auto_link_related_memories(
                new_state=state,
                max_candidates=10,
                similarity_threshold=0.6,
            )
            if linked:
                print(f"[MemoryService] Auto-linked {len(linked)} related memories for {topic}")
        except Exception as e:
            # Don't fail memory creation if auto-linking fails
            print(f"[MemoryService] Auto-linking failed (non-critical): {e}")

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
        from tracingrag.storage.neo4j_client import delete_memory_node
        from tracingrag.storage.qdrant import delete_embedding
        from sqlalchemy import delete

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
            print(f"[MemoryService] Deleted from topic_latest_states: {state_id}")

            # Delete from PostgreSQL
            await session.delete(state)

            # Delete from Qdrant
            try:
                await delete_embedding(state_id=state_id)
                print(f"[MemoryService] Deleted embedding from Qdrant: {state_id}")
            except Exception as e:
                print(f"[MemoryService] Failed to delete from Qdrant (non-critical): {e}")

            # Delete from Neo4j
            try:
                await delete_memory_node(state_id=state_id)
                print(f"[MemoryService] Deleted node from Neo4j: {state_id}")
            except Exception as e:
                print(f"[MemoryService] Failed to delete from Neo4j (non-critical): {e}")

            await session.commit()

            print(f"[MemoryService] Successfully deleted memory state: {state_id} (topic: {state.topic})")
            return True

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

    async def auto_link_related_memories(
        self,
        new_state: MemoryStateDB,
        max_candidates: int = 10,
        similarity_threshold: float = 0.6,
    ) -> list[tuple[UUID, str, float]]:
        """Automatically find and link related memories using embedding + LLM

        Uses a two-stage approach:
        1. Embedding-based similarity search (fast, finds candidates)
        2. LLM-based relevance analysis (accurate, confirms relationships)

        Args:
            new_state: Newly created memory state
            max_candidates: Maximum candidates to consider for LLM analysis
            similarity_threshold: Minimum embedding similarity score

        Returns:
            List of (related_state_id, relationship_type, confidence) tuples
        """
        from tracingrag.services.retrieval import RetrievalService
        from tracingrag.services.llm import get_llm_client
        from tracingrag.core.models.rag import LLMRequest
        import json

        # Stage 1: Find similar memories using embeddings (fast)
        retrieval_service = RetrievalService()
        candidates = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=max_candidates,
            score_threshold=similarity_threshold,
            latest_only=True,  # Only link to latest versions
        )

        # Filter out self and same topic
        candidates = [c for c in candidates if c.state.id != new_state.id and c.state.topic != new_state.topic]

        if not candidates:
            print(f"[AutoLink] No similar candidates found for {new_state.topic}")
            return []

        print(f"[AutoLink] Found {len(candidates)} candidates for {new_state.topic}, analyzing with LLM...")

        # Stage 2: Use LLM to analyze which ones are truly related
        # Prepare context
        new_memory_context = f"Topic: {new_state.topic}\nContent: {new_state.content}"

        # Analyze in batches to avoid context limit (max 5 at a time)
        batch_size = 5
        linked_memories = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]

            # Build candidate descriptions
            candidate_descriptions = []
            for idx, result in enumerate(batch):
                candidate_descriptions.append(
                    f"{idx+1}. Topic: {result.state.topic}\n"
                    f"   Content: {result.state.content[:200]}...\n"
                    f"   Embedding Similarity: {result.score:.2f}"
                )

            candidates_text = "\n\n".join(candidate_descriptions)

            # LLM analysis prompt
            analysis_prompt = f"""Analyze if the new memory is semantically related to any of the candidate memories.

NEW MEMORY:
{new_memory_context}

CANDIDATE MEMORIES:
{candidates_text}

For each candidate that IS related to the new memory, determine:
1. Whether they are related (yes/no)
2. Relationship type: "RELATED_TO", "DEPENDS_ON", "CAUSED_BY", "SIMILAR_TO", "MENTIONS"
3. Confidence (0.0-1.0)

Respond with JSON array of related memories only (empty array if none):
[
  {{"candidate_id": 1, "relationship_type": "RELATED_TO", "confidence": 0.85, "reasoning": "brief explanation"}},
  ...
]

Only include candidates with confidence >= 0.7."""

            # Query LLM
            llm_client = get_llm_client()
            try:
                request = LLMRequest(
                    system_prompt="You are an expert at analyzing semantic relationships between pieces of information.",
                    user_message=analysis_prompt,
                    context="",
                    model=settings.auto_link_model,
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=1000,
                    metadata={"task": "auto_link_analysis"},
                )

                response = await llm_client.generate(request)

                # Parse JSON response
                try:
                    # Extract JSON from response (might have markdown)
                    content = response.content.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]

                    analysis = json.loads(content)

                    # Create relationships
                    for item in analysis:
                        candidate_idx = item["candidate_id"] - 1  # Convert to 0-indexed
                        if 0 <= candidate_idx < len(batch):
                            related_state = batch[candidate_idx].state
                            relationship_type = item["relationship_type"]
                            confidence = item["confidence"]

                            print(f"[AutoLink] Linking {new_state.topic} -> {related_state.topic} ({relationship_type}, conf={confidence:.2f})")

                            # Create relationship in Neo4j (MemoryState to MemoryState)
                            await create_evolution_edge(
                                parent_id=new_state.id,
                                child_id=related_state.id,
                                relationship_type=relationship_type,
                                edge_properties={
                                    "confidence": confidence,
                                    "reasoning": item.get("reasoning", ""),
                                    "auto_generated": True,
                                },
                            )

                            linked_memories.append((related_state.id, relationship_type, confidence))

                except json.JSONDecodeError as e:
                    print(f"[AutoLink] Failed to parse LLM response: {e}")
                    print(f"[AutoLink] Response was: {response.content[:200]}")

            except Exception as e:
                print(f"[AutoLink] Error during LLM analysis: {e}")

        print(f"[AutoLink] Created {len(linked_memories)} automatic links for {new_state.topic}")
        return linked_memories
