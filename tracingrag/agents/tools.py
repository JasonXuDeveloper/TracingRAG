"""Agent tools for LangGraph agents"""

import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from tracingrag.services.embedding import generate_embedding
from tracingrag.services.graph import GraphService
from tracingrag.services.memory import MemoryService
from tracingrag.services.retrieval import RetrievalService
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


class VectorSearchInput(BaseModel):
    """Input for vector search tool"""

    query: str = Field(..., description="Query to search for")
    limit: int = Field(default=10, description="Maximum number of results")
    score_threshold: float = Field(default=0.5, description="Minimum similarity score")


class GraphTraversalInput(BaseModel):
    """Input for graph traversal tool"""

    start_node_id: str = Field(..., description="Starting node UUID")
    depth: int = Field(default=2, description="Maximum traversal depth")
    limit: int = Field(default=20, description="Maximum nodes to return")


class TraceHistoryInput(BaseModel):
    """Input for trace history tool"""

    topic: str = Field(..., description="Topic to get history for")
    limit: int = Field(default=10, description="Maximum number of versions")


class CreateMemoryInput(BaseModel):
    """Input for creating a memory"""

    topic: str = Field(..., description="Memory topic")
    content: str = Field(..., description="Memory content")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateEdgeInput(BaseModel):
    """Input for creating an edge"""

    source_id: str = Field(..., description="Source node UUID")
    target_id: str = Field(..., description="Target node UUID")
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentTools:
    """Tools for LangGraph agents"""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        memory_service: MemoryService | None = None,
        graph_service: GraphService | None = None,
    ):
        """
        Initialize agent tools

        Args:
            retrieval_service: Service for retrieval operations
            memory_service: Service for memory operations
            graph_service: Service for graph operations
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.memory_service = memory_service or MemoryService()
        self.graph_service = graph_service or GraphService()

    async def vector_search(
        self, query: str, limit: int = 10, score_threshold: float = 0.5
    ) -> dict[str, Any]:
        """
        Search for similar memory states using vector similarity with graph enhancement

        Args:
            query: Query to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            Dictionary with search results (includes graph-related memories)
        """
        start_time = time.time()

        try:
            logger.info(f"vector_search called with: query='{query}', limit={limit}, score_threshold={score_threshold}")

            # Use graph-enhanced retrieval to include related memories
            results = await self.retrieval_service.graph_enhanced_retrieval(
                query=query,
                depth=2,
                limit=limit,
                include_historical=False,  # Don't need history for agent
            )
            logger.info(f"graph_enhanced_retrieval returned {len(results)} results")
            # Collect all unique states (initial matches + related states)
            all_states = []
            seen_ids = set()

            for result in results:
                # Add the initial match
                state_id_str = str(result.state.id)
                if state_id_str not in seen_ids:
                    all_states.append(
                        {
                            "id": state_id_str,
                            "topic": result.state.topic,
                            "content": result.state.content,
                            "score": result.score,
                            "version": result.state.version,
                            "retrieval_type": "semantic",
                        }
                    )
                    seen_ids.add(state_id_str)

                # Add related states from graph
                if result.related_states:
                    logger.info(f"Found {len(result.related_states)} related states for {result.state.topic}")
                    for related_info in result.related_states:
                        related_id_str = related_info.get("memory", {}).get("id")
                        if related_id_str and related_id_str not in seen_ids:
                            # Fetch the full state
                            try:
                                from uuid import UUID

                                related_id = UUID(related_id_str)
                                related_state = await self.memory_service.get_memory_state(
                                    related_id
                                )

                                if related_state:
                                    all_states.append(
                                        {
                                            "id": str(related_state.id),
                                            "topic": related_state.topic,
                                            "content": related_state.content,
                                            "score": result.score
                                            * 0.8,  # Slightly lower score for related
                                            "version": related_state.version,
                                            "retrieval_type": "graph_related",
                                            "related_via": related_info.get(
                                                "relationship_path", []
                                            ),
                                        }
                                    )
                                    seen_ids.add(related_id_str)
                                    logger.info(f"Added related state: {related_state.topic}")
                            except Exception as e:
                                logger.error(f"Failed to fetch related state: {e}")
            logger.info(f"vector_search returning {len(all_states)} total states")
            return {
                "success": True,
                "count": len(all_states),
                "results": all_states,
                "duration_ms": (time.time() - start_time) * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

    async def graph_traversal(
        self, start_node_id: str, depth: int = 2, limit: int = 20
    ) -> dict[str, Any]:
        """
        Traverse graph from a starting node

        Args:
            start_node_id: Starting node UUID
            depth: Maximum traversal depth
            limit: Maximum nodes to return

        Returns:
            Dictionary with traversal results
        """
        start_time = time.time()

        try:
            start_uuid = UUID(start_node_id)
            results = await self.retrieval_service.graph_enhanced_retrieval(
                query="",  # Not used for pure traversal
                depth=depth,
                limit=limit,
                start_nodes=[start_uuid],
            )

            return {
                "success": True,
                "count": len(results),
                "results": [
                    {
                        "id": str(result.state.id),
                        "topic": result.state.topic,
                        "content": result.state.content[:200] + "...",  # Truncate for brevity
                        "score": result.score,
                    }
                    for result in results
                ],
                "duration_ms": (time.time() - start_time) * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

    async def trace_history(self, topic: str, limit: int = 10) -> dict[str, Any]:
        """
        Get version history for a topic

        Args:
            topic: Topic to get history for
            limit: Maximum number of versions

        Returns:
            Dictionary with version history
        """
        start_time = time.time()

        try:
            versions = await self.memory_service.get_topic_history(topic=topic, limit=limit)

            return {
                "success": True,
                "count": len(versions),
                "versions": [
                    {
                        "id": str(v.id),
                        "version": v.version,
                        "content": v.content[:200] + "...",  # Truncate
                        "timestamp": v.timestamp.isoformat(),
                        "confidence": v.confidence,
                    }
                    for v in versions
                ],
                "duration_ms": (time.time() - start_time) * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

    async def create_memory(
        self,
        topic: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new memory state

        Args:
            topic: Memory topic
            content: Memory content
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Dictionary with created memory info
        """
        start_time = time.time()

        try:
            # Generate embedding
            embedding = await generate_embedding(content)

            # Create memory state
            state = await self.memory_service.create_memory(
                topic=topic,
                content=content,
                embedding=embedding,
                tags=tags or [],
                custom_metadata=metadata or {},
            )

            return {
                "success": True,
                "memory_id": str(state.id),
                "topic": state.topic,
                "version": state.version,
                "duration_ms": (time.time() - start_time) * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 0.5,
    ) -> dict[str, Any]:
        """
        Create an edge between two memory states

        Args:
            source_id: Source node UUID
            target_id: Target node UUID
            relationship_type: Type of relationship
            strength: Edge strength (0-1)

        Returns:
            Dictionary with edge info
        """
        start_time = time.time()

        try:
            source_uuid = UUID(source_id)
            target_uuid = UUID(target_id)

            edge = await self.graph_service.create_edge(
                source_state_id=source_uuid,
                target_state_id=target_uuid,
                relationship_type=relationship_type,
                strength=strength,
            )

            return {
                "success": True,
                "edge_id": str(edge.id),
                "relationship_type": edge.relationship_type,
                "duration_ms": (time.time() - start_time) * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for LangGraph

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "vector_search",
                "description": "Search for similar memory states using vector similarity",
                "parameters": VectorSearchInput.model_json_schema(),
                "function": self.vector_search,
            },
            {
                "name": "graph_traversal",
                "description": "Traverse the knowledge graph from a starting node",
                "parameters": GraphTraversalInput.model_json_schema(),
                "function": self.graph_traversal,
            },
            {
                "name": "trace_history",
                "description": "Get version history for a topic",
                "parameters": TraceHistoryInput.model_json_schema(),
                "function": self.trace_history,
            },
            {
                "name": "create_memory",
                "description": "Create a new memory state",
                "parameters": CreateMemoryInput.model_json_schema(),
                "function": self.create_memory,
            },
            {
                "name": "create_edge",
                "description": "Create an edge between two memory states",
                "parameters": CreateEdgeInput.model_json_schema(),
                "function": self.create_edge,
            },
        ]
