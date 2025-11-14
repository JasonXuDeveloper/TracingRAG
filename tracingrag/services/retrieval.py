"""Retrieval service implementing semantic, graph, and temporal retrieval strategies"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, select

from tracingrag.services.embedding import generate_embedding
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB, TopicLatestStateDB
from tracingrag.storage.neo4j_client import get_related_memories
from tracingrag.storage.qdrant import search_similar


class RetrievalResult:
    """Container for retrieval results with metadata"""

    def __init__(
        self,
        state: MemoryStateDB,
        score: float,
        retrieval_type: str,
        related_states: list[dict[str, Any]] | None = None,
        historical_context: list[MemoryStateDB] | None = None,
    ):
        self.state = state
        self.score = score
        self.retrieval_type = retrieval_type
        self.related_states = related_states or []
        self.historical_context = historical_context or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "state": {
                "id": str(self.state.id),
                "topic": self.state.topic,
                "content": self.state.content,
                "version": self.state.version,
                "timestamp": self.state.timestamp.isoformat(),
                "confidence": self.state.confidence,
                "entity_type": self.state.entity_type,
                "tags": self.state.tags,
            },
            "score": self.score,
            "retrieval_type": self.retrieval_type,
            "related_states": self.related_states,
            "historical_context": [
                {
                    "id": str(state.id),
                    "topic": state.topic,
                    "content": state.content,
                    "version": state.version,
                    "timestamp": state.timestamp.isoformat(),
                }
                for state in self.historical_context
            ],
        }


class RetrievalService:
    """Service for retrieving memory states using various strategies"""

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_conditions: dict[str, Any] | None = None,
        latest_only: bool = True,
    ) -> list[RetrievalResult]:
        """Semantic retrieval with latest state tracking

        Args:
            query: Natural language query
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Qdrant filter conditions
            latest_only: If True, only return latest state per topic

        Returns:
            List of retrieval results ranked by semantic similarity
        """
        # Generate query embedding
        query_embedding = await generate_embedding(query)

        # Vector search in Qdrant
        candidates = await search_similar(
            query_vector=query_embedding,
            limit=limit * 2 if latest_only else limit,  # Get more for filtering
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        # Retrieve full state objects from database
        async with get_session() as session:
            # Convert to UUID if string, otherwise use as-is
            state_ids = [
                UUID(candidate["id"]) if isinstance(candidate["id"], str) else candidate["id"]
                for candidate in candidates
            ]

            result = await session.execute(
                select(MemoryStateDB).where(MemoryStateDB.id.in_(state_ids))
            )
            states = {state.id: state for state in result.scalars().all()}

        # Filter to latest states if requested
        if latest_only:
            latest_states = await self._filter_to_latest_per_topic(candidates, states)
        else:
            latest_states = candidates

        # Create retrieval results
        results = []
        for candidate in latest_states[:limit]:
            state_id = (
                candidate["id"] if isinstance(candidate["id"], UUID) else UUID(candidate["id"])
            )
            if state_id in states:
                results.append(
                    RetrievalResult(
                        state=states[state_id],
                        score=candidate["score"],
                        retrieval_type="semantic",
                    )
                )

        return results

    async def graph_enhanced_retrieval(
        self,
        query: str,
        depth: int = 2,
        limit: int = 5,
        relationship_types: list[str] | None = None,
        include_historical: bool = True,
        historical_steps: int = 3,
        start_nodes: list[UUID] | None = None,
    ) -> list[RetrievalResult]:
        """Graph-enhanced retrieval with edge-based relevance

        Args:
            query: Natural language query
            depth: Maximum graph traversal depth
            limit: Maximum number of initial semantic matches
            relationship_types: Types of edges to traverse
            include_historical: Whether to include trace history
            historical_steps: Number of historical states to include
            start_nodes: Optional list of node UUIDs to start traversal from

        Returns:
            List of retrieval results with related states and historical context
        """
        # Get initial semantic matches or use provided start nodes
        if start_nodes:
            # If start nodes provided, use them instead of semantic search
            from tracingrag.storage.database import get_session
            from tracingrag.storage.models import MemoryStateDB
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_session() as session:
                result = await session.execute(
                    select(MemoryStateDB).where(MemoryStateDB.id.in_(start_nodes))
                )
                states = list(result.scalars().all())
                # Trigger loading of all attributes while session is active
                for state in states:
                    _ = state.content  # Access content to load it
                    _ = state.custom_metadata  # Load metadata
                    _ = state.embedding  # Load embedding

            # Create results after session is closed
            initial_matches = [
                RetrievalResult(state=state, score=1.0, retrieval_type="direct")
                for state in states
            ]
        else:
            # Get initial semantic matches
            initial_matches = await self.semantic_search(
                query=query,
                limit=limit,
                latest_only=True,
            )

        # Enhance each match with graph traversal
        enhanced_results = []
        for match in initial_matches:
            print(f"[RetrievalService] Enhancing match: {match.state.topic} (id={match.state.id})")
            # Get connected states via graph traversal
            related_states = await get_related_memories(
                state_id=match.state.id,
                relationship_types=relationship_types,
                max_depth=depth,
                limit=50,  # Get more related states
            )
            print(
                f"[RetrievalService] Found {len(related_states)} related states from Neo4j for {match.state.topic}"
            )

            # Get historical context from trace
            historical_context = []
            if include_historical:
                historical_context = await self._get_trace_context(
                    topic=match.state.topic,
                    current_version=match.state.version,
                    steps_back=historical_steps,
                )

            enhanced_results.append(
                RetrievalResult(
                    state=match.state,
                    score=match.score,
                    retrieval_type="graph_enhanced",
                    related_states=related_states,
                    historical_context=historical_context,
                )
            )
            print(
                f"[RetrievalService] Enhanced result created with {len(related_states)} related states"
            )

        return enhanced_results

    async def temporal_query(
        self,
        topic: str,
        timestamp: datetime,
    ) -> MemoryStateDB | None:
        """Time-travel query: get state as it existed at a specific time

        Args:
            topic: Topic to query
            timestamp: Point in time to query

        Returns:
            Memory state as it existed at that time, or None if it didn't exist
        """
        async with get_session() as session:
            # Query for the latest state before or at the timestamp
            result = await session.execute(
                select(MemoryStateDB)
                .where(
                    and_(
                        MemoryStateDB.topic == topic,
                        MemoryStateDB.timestamp <= timestamp,
                    )
                )
                .order_by(MemoryStateDB.timestamp.desc())
                .limit(1)
            )

            return result.scalar_one_or_none()

    async def temporal_range_query(
        self,
        topic: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> list[MemoryStateDB]:
        """Get all states for a topic within a time range

        Args:
            topic: Topic to query
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of states to return

        Returns:
            List of memory states within the time range
        """
        async with get_session() as session:
            result = await session.execute(
                select(MemoryStateDB)
                .where(
                    and_(
                        MemoryStateDB.topic == topic,
                        MemoryStateDB.timestamp >= start_time,
                        MemoryStateDB.timestamp <= end_time,
                    )
                )
                .order_by(MemoryStateDB.timestamp.desc())
                .limit(limit)
            )

            return list(result.scalars().all())

    async def hybrid_retrieval(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.5,
        graph_weight: float = 0.3,
        temporal_weight: float = 0.2,
        time_window: tuple[datetime, datetime] | None = None,
        graph_depth: int = 2,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval combining semantic, graph, and temporal strategies

        Args:
            query: Natural language query
            limit: Maximum number of results
            semantic_weight: Weight for semantic similarity (0-1)
            graph_weight: Weight for graph relevance (0-1)
            temporal_weight: Weight for temporal recency (0-1)
            time_window: Optional time range filter
            graph_depth: Graph traversal depth

        Returns:
            List of retrieval results with combined scoring
        """
        # Validate weights sum to 1.0
        total_weight = semantic_weight + graph_weight + temporal_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Build filter conditions for time window
        filter_conditions = None
        if time_window:
            start_time, end_time = time_window
            # Note: This is a simplified filter - actual implementation
            # would need to filter by timestamp range in Qdrant

        # Get semantic results
        semantic_results = await self.semantic_search(
            query=query,
            limit=limit * 2,  # Get more for re-ranking
            latest_only=True,
            filter_conditions=filter_conditions,
        )

        # Get graph-enhanced results
        graph_results = await self.graph_enhanced_retrieval(
            query=query,
            depth=graph_depth,
            limit=limit,
        )

        # Combine and re-rank results
        combined_scores: dict[UUID, float] = {}
        state_map: dict[UUID, RetrievalResult] = {}

        # Add semantic scores
        for result in semantic_results:
            combined_scores[result.state.id] = semantic_weight * result.score
            state_map[result.state.id] = result

        # Add graph scores (based on number of related states)
        for result in graph_results:
            state_id = result.state.id
            # Normalize graph score by number of related states
            graph_score = min(len(result.related_states) / 10.0, 1.0)

            if state_id in combined_scores:
                combined_scores[state_id] += graph_weight * graph_score
                # Update with graph-enhanced version
                state_map[state_id] = result
            else:
                combined_scores[state_id] = graph_weight * graph_score
                state_map[state_id] = result

        # Add temporal recency scores
        if time_window:
            _, end_time = time_window
        else:
            end_time = datetime.utcnow()

        for state_id, result in state_map.items():
            # Calculate recency score (exponential decay)
            time_diff = (end_time - result.state.timestamp).total_seconds()
            days_old = time_diff / 86400  # Convert to days
            recency_score = max(0.0, 1.0 - (days_old / 365))  # Decay over a year

            combined_scores[state_id] += temporal_weight * recency_score

        # Sort by combined score and return top results
        sorted_states = sorted(
            state_map.items(),
            key=lambda x: combined_scores[x[0]],
            reverse=True,
        )

        # Update scores and return
        results = []
        for state_id, result in sorted_states[:limit]:
            result.score = combined_scores[state_id]
            result.retrieval_type = "hybrid"
            results.append(result)

        return results

    async def get_latest_states(
        self,
        topics: list[str] | None = None,
        limit: int = 100,
    ) -> list[MemoryStateDB]:
        """Get latest states for topics using O(1) materialized view lookup

        Args:
            topics: Optional list of specific topics to retrieve
            limit: Maximum number of results

        Returns:
            List of latest memory states
        """
        async with get_session() as session:
            query = select(TopicLatestStateDB)

            if topics:
                query = query.where(TopicLatestStateDB.topic.in_(topics))

            query = query.limit(limit)

            result = await session.execute(query)
            mappings = result.scalars().all()

            # Fetch actual states
            state_ids = [mapping.latest_state_id for mapping in mappings]
            states_result = await session.execute(
                select(MemoryStateDB).where(MemoryStateDB.id.in_(state_ids))
            )

            return list(states_result.scalars().all())

    async def _filter_to_latest_per_topic(
        self,
        candidates: list[dict[str, Any]],
        states: dict[UUID, MemoryStateDB],
    ) -> list[dict[str, Any]]:
        """Filter candidates to only latest state per topic

        Args:
            candidates: List of candidate results from vector search
            states: Map of state IDs to MemoryStateDB objects

        Returns:
            Filtered list with only latest state per topic
        """
        async with get_session() as session:
            # Get latest state mappings for all topics
            topics = list({states[c["id"]].topic for c in candidates if c["id"] in states})

            result = await session.execute(
                select(TopicLatestStateDB).where(TopicLatestStateDB.topic.in_(topics))
            )
            latest_mappings = {m.topic: m.latest_state_id for m in result.scalars().all()}

        # Filter to only latest states
        filtered = []
        seen_topics = set()

        for candidate in candidates:
            state_id = (
                candidate["id"] if isinstance(candidate["id"], UUID) else UUID(candidate["id"])
            )
            if state_id not in states:
                continue

            state = states[state_id]
            topic = state.topic

            # Skip if we've already seen this topic or this isn't the latest
            if topic in seen_topics:
                continue

            latest_id = latest_mappings.get(topic)
            if latest_id == state_id:
                filtered.append(candidate)
                seen_topics.add(topic)

        return filtered

    async def _get_trace_context(
        self,
        topic: str,
        current_version: int,
        steps_back: int = 3,
    ) -> list[MemoryStateDB]:
        """Get historical context from trace

        Args:
            topic: Topic to query
            current_version: Current version number
            steps_back: Number of previous versions to retrieve

        Returns:
            List of historical memory states
        """
        async with get_session() as session:
            # Get previous versions
            result = await session.execute(
                select(MemoryStateDB)
                .where(
                    and_(
                        MemoryStateDB.topic == topic,
                        MemoryStateDB.version < current_version,
                    )
                )
                .order_by(MemoryStateDB.version.desc())
                .limit(steps_back)
            )

            return list(result.scalars().all())
