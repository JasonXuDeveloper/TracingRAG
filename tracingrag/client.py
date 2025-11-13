"""Python client for TracingRAG REST API"""

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

# ============================================================================
# Response Models
# ============================================================================


class MemoryState(BaseModel):
    """Memory state response model"""

    id: str
    topic: str
    content: str
    version: int
    timestamp: datetime
    parent_state_id: str | None = None
    tags: list[str] = []
    confidence: float = 1.0
    custom_metadata: dict[str, Any] = {}
    storage_tier: str = "active"
    access_count: int = 0
    is_consolidated: bool = False
    consolidation_level: int = 0


class QueryResult(BaseModel):
    """Query result model"""

    state: MemoryState
    score: float
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Query response model"""

    query: str
    answer: str | None = None
    retrieved_states: list[QueryResult] = []
    confidence: float = 0.0
    reasoning_steps: list[str] = []
    sources: list[str] = []


class PromotionResult(BaseModel):
    """Promotion result model"""

    new_state: MemoryState
    promoted_from_version: int
    conflicts_detected: list[dict] = []
    conflicts_resolved: list[dict] = []
    synthesis_sources: list[str] = []
    quality_checks: dict[str, Any] = {}


class PromotionCandidate(BaseModel):
    """Promotion candidate model"""

    topic: str
    current_version: int
    versions_since_consolidation: int
    time_since_update_hours: float
    access_count: int
    priority_score: float


# ============================================================================
# Client
# ============================================================================


class TracingRAGClient:
    """Python client for TracingRAG REST API

    Example:
        >>> client = TracingRAGClient("http://localhost:8000")
        >>> state = client.create_memory(
        ...     topic="project_alpha",
        ...     content="Initial design for API system"
        ... )
        >>> results = client.query("What is project alpha?")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize TracingRAG client

        Args:
            base_url: Base URL of TracingRAG API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Create HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close HTTP client"""
        self._client.close()

    # ========================================================================
    # System Endpoints
    # ========================================================================

    def health(self) -> dict:
        """Check API health

        Returns:
            Health status dict

        Example:
            >>> status = client.health()
            >>> print(status["status"])
            healthy
        """
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def metrics(self) -> dict:
        """Get system metrics

        Returns:
            System metrics dict

        Example:
            >>> metrics = client.metrics()
            >>> print(f"Total memories: {metrics['total_memories']}")
        """
        response = self._client.get("/metrics")
        response.raise_for_status()
        return response.json()

    # ========================================================================
    # Memory Endpoints
    # ========================================================================

    def create_memory(
        self,
        topic: str,
        content: str,
        parent_state_id: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        custom_metadata: dict | None = None,
    ) -> MemoryState:
        """Create a new memory state

        Args:
            topic: Topic name (e.g., "project_alpha")
            content: Memory content
            parent_state_id: Optional parent state ID for versioning
            tags: Optional tags
            confidence: Confidence score (0.0-1.0)
            custom_metadata: Optional metadata dict

        Returns:
            Created memory state

        Example:
            >>> state = client.create_memory(
            ...     topic="user_preferences",
            ...     content="User prefers dark mode",
            ...     tags=["ui", "preferences"],
            ...     confidence=0.95
            ... )
        """
        data = {
            "topic": topic,
            "content": content,
            "parent_state_id": parent_state_id,
            "tags": tags or [],
            "confidence": confidence,
            "custom_metadata": custom_metadata or {},
        }

        response = self._client.post("/api/v1/memories", json=data)
        response.raise_for_status()
        return MemoryState(**response.json())

    def get_memory(self, memory_id: str) -> MemoryState:
        """Get memory by ID

        Args:
            memory_id: Memory state ID

        Returns:
            Memory state

        Example:
            >>> state = client.get_memory("mem_123")
            >>> print(state.content)
        """
        response = self._client.get(f"/api/v1/memories/{memory_id}")
        response.raise_for_status()
        return MemoryState(**response.json())

    def list_memories(
        self,
        topic: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryState]:
        """List memory states with optional filtering

        Args:
            topic: Optional topic filter
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of memory states

        Example:
            >>> memories = client.list_memories(topic="project_alpha", limit=10)
            >>> for mem in memories:
            ...     print(f"v{mem.version}: {mem.content[:50]}")
        """
        params = {"limit": limit, "offset": offset}
        if topic:
            params["topic"] = topic

        response = self._client.get("/api/v1/memories", params=params)
        response.raise_for_status()
        return [MemoryState(**m) for m in response.json()]

    def get_trace(self, topic: str) -> list[MemoryState]:
        """Get version history for a topic

        Args:
            topic: Topic name

        Returns:
            List of memory states ordered by version

        Example:
            >>> trace = client.get_trace("project_alpha")
            >>> print(f"Evolution: {len(trace)} versions")
            >>> for state in trace:
            ...     print(f"v{state.version}: {state.timestamp}")
        """
        response = self._client.get(f"/api/v1/traces/{topic}")
        response.raise_for_status()
        return [MemoryState(**m) for m in response.json()]

    # ========================================================================
    # Query Endpoints
    # ========================================================================

    def query(
        self,
        query: str,
        limit: int = 10,
        use_agent: bool = False,
        include_history: bool = False,
        include_related: bool = True,
        depth: int = 2,
    ) -> QueryResponse:
        """Query the RAG system

        Args:
            query: Natural language query
            limit: Maximum number of results
            use_agent: Whether to use agent-based retrieval (slower but smarter)
            include_history: Include version history in context
            include_related: Include graph-related states
            depth: Graph traversal depth

        Returns:
            Query response with answer and sources

        Example:
            >>> # Simple query
            >>> result = client.query("What is project alpha?")
            >>> print(result.answer)

            >>> # Agent-based query with more context
            >>> result = client.query(
            ...     "What changed in project alpha?",
            ...     use_agent=True,
            ...     include_history=True
            ... )
            >>> print(f"Answer: {result.answer}")
            >>> print(f"Reasoning: {result.reasoning_steps}")
        """
        data = {
            "query": query,
            "limit": limit,
            "use_agent": use_agent,
            "include_history": include_history,
            "include_related": include_related,
            "depth": depth,
        }

        response = self._client.post("/api/v1/query", json=data)
        response.raise_for_status()
        return QueryResponse(**response.json())

    # ========================================================================
    # Promotion Endpoints
    # ========================================================================

    def promote_memory(
        self,
        topic: str,
        reason: str | None = None,
        mode: str = "automatic",
    ) -> PromotionResult:
        """Promote a memory to a new state

        This synthesizes historical information and related context
        into a new, consolidated memory state.

        Args:
            topic: Topic to promote
            reason: Optional reason for promotion
            mode: Promotion mode ("automatic", "manual", "scheduled")

        Returns:
            Promotion result with new state

        Example:
            >>> result = client.promote_memory(
            ...     topic="project_alpha",
            ...     reason="Major milestone reached"
            ... )
            >>> print(f"Promoted to v{result.new_state.version}")
            >>> print(f"Synthesized from {len(result.synthesis_sources)} sources")
        """
        data = {
            "topic": topic,
            "reason": reason,
            "mode": mode,
        }

        response = self._client.post("/api/v1/promote", json=data)
        response.raise_for_status()
        return PromotionResult(**response.json())

    def get_promotion_candidates(
        self,
        limit: int = 10,
        min_priority: float = 5.0,
    ) -> list[PromotionCandidate]:
        """Get topics that are candidates for promotion

        Args:
            limit: Maximum number of candidates
            min_priority: Minimum priority score

        Returns:
            List of promotion candidates ordered by priority

        Example:
            >>> candidates = client.get_promotion_candidates(limit=5)
            >>> for candidate in candidates:
            ...     print(f"{candidate.topic}: priority {candidate.priority_score:.1f}")
        """
        params = {"limit": limit, "min_priority": min_priority}

        response = self._client.get("/api/v1/promotion-candidates", params=params)
        response.raise_for_status()
        return [PromotionCandidate(**c) for c in response.json()]


# ============================================================================
# Async Client
# ============================================================================


class AsyncTracingRAGClient:
    """Async Python client for TracingRAG REST API

    Example:
        >>> async with AsyncTracingRAGClient() as client:
        ...     state = await client.create_memory(
        ...         topic="project_alpha",
        ...         content="Initial design"
        ...     )
        ...     results = await client.query("What is project alpha?")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize async TracingRAG client

        Args:
            base_url: Base URL of TracingRAG API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        """Close HTTP client"""
        await self._client.aclose()

    # Async versions of all methods
    async def health(self) -> dict:
        """Check API health"""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def metrics(self) -> dict:
        """Get system metrics"""
        response = await self._client.get("/metrics")
        response.raise_for_status()
        return response.json()

    async def create_memory(
        self,
        topic: str,
        content: str,
        parent_state_id: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        custom_metadata: dict | None = None,
    ) -> MemoryState:
        """Create a new memory state"""
        data = {
            "topic": topic,
            "content": content,
            "parent_state_id": parent_state_id,
            "tags": tags or [],
            "confidence": confidence,
            "custom_metadata": custom_metadata or {},
        }

        response = await self._client.post("/api/v1/memories", json=data)
        response.raise_for_status()
        return MemoryState(**response.json())

    async def get_memory(self, memory_id: str) -> MemoryState:
        """Get memory by ID"""
        response = await self._client.get(f"/api/v1/memories/{memory_id}")
        response.raise_for_status()
        return MemoryState(**response.json())

    async def list_memories(
        self,
        topic: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryState]:
        """List memory states"""
        params = {"limit": limit, "offset": offset}
        if topic:
            params["topic"] = topic

        response = await self._client.get("/api/v1/memories", params=params)
        response.raise_for_status()
        return [MemoryState(**m) for m in response.json()]

    async def get_trace(self, topic: str) -> list[MemoryState]:
        """Get version history for a topic"""
        response = await self._client.get(f"/api/v1/traces/{topic}")
        response.raise_for_status()
        return [MemoryState(**m) for m in response.json()]

    async def query(
        self,
        query: str,
        limit: int = 10,
        use_agent: bool = False,
        include_history: bool = False,
        include_related: bool = True,
        depth: int = 2,
    ) -> QueryResponse:
        """Query the RAG system"""
        data = {
            "query": query,
            "limit": limit,
            "use_agent": use_agent,
            "include_history": include_history,
            "include_related": include_related,
            "depth": depth,
        }

        response = await self._client.post("/api/v1/query", json=data)
        response.raise_for_status()
        return QueryResponse(**response.json())

    async def promote_memory(
        self,
        topic: str,
        reason: str | None = None,
        mode: str = "automatic",
    ) -> PromotionResult:
        """Promote a memory to a new state"""
        data = {
            "topic": topic,
            "reason": reason,
            "mode": mode,
        }

        response = await self._client.post("/api/v1/promote", json=data)
        response.raise_for_status()
        return PromotionResult(**response.json())

    async def get_promotion_candidates(
        self,
        limit: int = 10,
        min_priority: float = 5.0,
    ) -> list[PromotionCandidate]:
        """Get topics that are candidates for promotion"""
        params = {"limit": limit, "min_priority": min_priority}

        response = await self._client.get("/api/v1/promotion-candidates", params=params)
        response.raise_for_status()
        return [PromotionCandidate(**c) for c in response.json()]
