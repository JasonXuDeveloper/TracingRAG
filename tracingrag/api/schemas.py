"""API schemas for request and response models"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ============================================================================
# Memory API Schemas
# ============================================================================


class CreateMemoryRequest(BaseModel):
    """Request to create a new memory state"""

    topic: str = Field(..., description="Topic/key for the memory")
    content: str = Field(..., description="Main content of the memory")
    parent_state_id: UUID | None = Field(default=None, description="Parent state ID for evolution")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: list[str] = Field(default_factory=list, description="List of tags")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source: str | None = Field(default=None, description="Source of the memory")


class MemoryStateResponse(BaseModel):
    """Response containing a memory state"""

    id: UUID
    topic: str
    content: str
    version: int
    timestamp: datetime
    parent_state_id: UUID | None
    metadata: dict[str, Any]
    tags: list[str]
    confidence: float
    source: str | None


class MemoryListResponse(BaseModel):
    """Response containing a list of memory states"""

    memories: list[MemoryStateResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Query/RAG API Schemas
# ============================================================================


class QueryRequest(BaseModel):
    """Request to query the RAG system"""

    query: str = Field(..., description="User query")
    include_history: bool = Field(default=True, description="Include trace history")
    include_related: bool = Field(default=True, description="Include related states")
    depth: int = Field(default=2, ge=1, le=5, description="Graph traversal depth")
    limit: int = Field(default=10, ge=1, le=100, description="Max results to retrieve")
    use_agent: bool = Field(default=False, description="Use agent-based retrieval")


class QueryResponse(BaseModel):
    """Response from RAG query"""

    answer: str = Field(..., description="Generated answer")
    sources: list[MemoryStateResponse] = Field(default_factory=list, description="Source memories")
    confidence: float = Field(..., description="Confidence in answer")
    reasoning: str | None = Field(default=None, description="Reasoning for answer")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Promotion API Schemas
# ============================================================================


class PromoteMemoryRequest(BaseModel):
    """Request to promote a memory state"""

    topic: str = Field(..., description="Topic to promote")
    reason: str = Field(..., description="Reason for promotion")
    include_related: bool = Field(default=True, description="Include related states")
    max_sources: int = Field(default=10, ge=1, le=50, description="Max sources for synthesis")


class PromotionCandidateResponse(BaseModel):
    """Response containing a promotion candidate"""

    topic: str
    trigger: str
    priority: int
    reasoning: str
    current_version_count: int
    last_promoted: datetime | None
    confidence: float


class PromotionCandidatesResponse(BaseModel):
    """Response containing promotion candidates"""

    candidates: list[PromotionCandidateResponse]
    total: int


class PromoteMemoryResponse(BaseModel):
    """Response from memory promotion"""

    success: bool
    topic: str
    new_version: int | None
    previous_state_id: UUID | None
    new_state_id: UUID | None
    synthesized_from_count: int
    conflicts_detected_count: int
    conflicts_resolved_count: int
    edges_updated_count: int
    quality_checks_count: int
    reasoning: str
    confidence: float
    manual_review_needed: bool
    error_message: str | None


# ============================================================================
# Health & Metrics Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Overall status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: dict[str, str] = Field(default_factory=dict, description="Service health status")


class MetricsResponse(BaseModel):
    """System metrics response"""

    total_memories: int = Field(..., description="Total memory states")
    total_topics: int = Field(..., description="Total unique topics")
    total_promotions: int = Field(..., description="Total promotions performed")
    avg_versions_per_topic: float = Field(..., description="Average versions per topic")
    cache_hit_rate: float | None = Field(default=None, description="Cache hit rate if available")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
