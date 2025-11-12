"""Core data models for memory states, edges, and traces"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between memory states"""

    # Temporal relationships
    EVOLVES_TO = "evolves_to"  # State A evolved into state B
    SUPERSEDES = "supersedes"  # State B replaces state A

    # Logical relationships
    RELATES_TO = "relates_to"  # General relationship
    DEPENDS_ON = "depends_on"  # State A depends on state B
    CAUSES = "causes"  # State A causes state B
    CONTRADICTS = "contradicts"  # State A contradicts state B
    SUPPORTS = "supports"  # State A supports/validates state B

    # Reference relationships
    REFERENCES = "references"  # State A references state B
    DERIVED_FROM = "derived_from"  # State A is derived from state B
    PART_OF = "part_of"  # State A is part of state B
    CONTAINS = "contains"  # State A contains state B

    # Context relationships
    CONTEXT_FOR = "context_for"  # State A provides context for state B
    EXAMPLE_OF = "example_of"  # State A is an example of state B

    # Consolidation relationships
    SUMMARIZES = "summarizes"  # State A summarizes state B (for consolidation)


class StorageTier(str, Enum):
    """Storage tiers for memory lifecycle management"""

    WORKING = "working"  # Hot storage, frequently accessed
    ACTIVE = "active"  # Normal storage, regularly accessed
    ARCHIVED = "archived"  # Cold storage, rarely accessed


class EntityType(str, Enum):
    """Types of entities for domain-specific use cases (novels, games, etc.)"""

    # Generic
    GENERIC = "generic"  # Default, untyped memory

    # Narrative entities
    CHARACTER = "character"  # Character in story/game
    LOCATION = "location"  # Physical location
    PLOT_THREAD = "plot_thread"  # Story plot line
    EVENT = "event"  # Significant event
    RELATIONSHIP = "relationship"  # Relationship between entities

    # World building
    WORLD_RULE = "world_rule"  # Rules of the world (physics, magic, etc.)
    FACTION = "faction"  # Group, organization, nation
    ITEM = "item"  # Significant object
    CONCEPT = "concept"  # Abstract concept or idea

    # Hierarchical
    VOLUME = "volume"  # Volume in a series
    ARC = "arc"  # Story arc
    CHAPTER = "chapter"  # Chapter or section


class MemoryState(BaseModel):
    """A single state in a memory trace representing knowledge at a point in time"""

    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(..., description="What this memory is about")
    content: str = Field(..., description="The actual knowledge/information")
    version: int = Field(ge=1, description="Version number in the trace")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[list[float]] = Field(default=None, description="Vector representation")
    parent_state_id: Optional[UUID] = Field(
        default=None, description="Previous version in the trace"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for this state"
    )
    source: Optional[str] = Field(default=None, description="Where this information came from")
    created_by: Optional[str] = Field(default=None, description="Who/what created this state")

    # Human memory simulation fields
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Learned importance (0-1)"
    )
    memory_strength: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Current memory strength (decays over time)"
    )
    storage_tier: StorageTier = Field(default=StorageTier.ACTIVE)

    # Consolidation tracking
    consolidated_from: Optional[list[UUID]] = Field(
        default=None, description="State IDs this was consolidated from"
    )
    is_consolidated: bool = Field(default=False, description="Is this a consolidated summary?")
    consolidation_level: int = Field(
        default=0, description="0=raw, 1=daily, 2=weekly, 3=monthly"
    )

    # Diff-based storage for efficiency
    diff_from_parent: Optional[str] = Field(
        default=None, description="Diff from parent state (for storage efficiency)"
    )
    is_delta: bool = Field(default=False, description="Is this stored as diff?")

    # Domain-specific entity typing (for novels, games, etc.)
    entity_type: EntityType = Field(
        default=EntityType.GENERIC, description="Type of entity this memory represents"
    )
    entity_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="Type-specific structured data (e.g., character traits, location props)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "project_alpha_development",
                "content": "Started implementation of feature X using approach Y. Initial tests passing.",
                "version": 1,
                "tags": ["project", "development", "feature_x"],
                "confidence": 0.95,
                "source": "development_log",
            }
        }


class MemoryEdge(BaseModel):
    """An edge connecting two memory states in the graph"""

    id: UUID = Field(default_factory=uuid4)
    source_state_id: UUID = Field(..., description="Origin state")
    target_state_id: UUID = Field(..., description="Destination state")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    strength: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Strength/importance of the relationship"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    bidirectional: bool = Field(
        default=False, description="Whether the relationship works both ways"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = Field(
        default=None, description="Human-readable description of why this edge exists"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source_state_id": "550e8400-e29b-41d4-a716-446655440000",
                "target_state_id": "550e8400-e29b-41d4-a716-446655440001",
                "relationship_type": "relates_to",
                "strength": 0.8,
                "description": "Bug discovery prompted code review",
            }
        }


class Trace(BaseModel):
    """A complete trace representing the evolution of a topic over time"""

    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(..., description="The topic this trace tracks")
    state_ids: list[UUID] = Field(
        default_factory=list, description="Ordered list of state IDs (oldest to newest)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    is_active: bool = Field(default=True, description="Whether this trace is still being updated")

    @property
    def current_version(self) -> int:
        """Get the current version number (latest state)"""
        return len(self.state_ids)

    @property
    def latest_state_id(self) -> Optional[UUID]:
        """Get the ID of the latest state"""
        return self.state_ids[-1] if self.state_ids else None

    def add_state(self, state_id: UUID) -> None:
        """Add a new state to the trace"""
        self.state_ids.append(state_id)
        self.updated_at = datetime.utcnow()

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "project_alpha_development",
                "tags": ["project", "development"],
                "is_active": True,
            }
        }


class QueryContext(BaseModel):
    """Context for a query including retrieval parameters"""

    query: str = Field(..., description="The query text")
    include_history: bool = Field(default=False, description="Include trace history")
    include_related: bool = Field(default=True, description="Include graph-connected states")
    graph_depth: int = Field(default=2, ge=1, le=5, description="Graph traversal depth")
    time_window: Optional[tuple[datetime, datetime]] = Field(
        default=None, description="Filter by time range"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    min_relevance: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum relevance score"
    )
    tags: Optional[list[str]] = Field(default=None, description="Filter by tags")


class RetrievalResult(BaseModel):
    """Result from a retrieval operation"""

    state: MemoryState
    relevance_score: float = Field(ge=0.0, le=1.0)
    related_states: list["MemoryState"] = Field(default_factory=list)
    historical_context: list["MemoryState"] = Field(default_factory=list)
    trace_id: Optional[UUID] = None
    reasoning: Optional[str] = Field(
        default=None, description="Why this result was retrieved"
    )


class PromotionRequest(BaseModel):
    """Request to promote a memory to a new state"""

    topic: str = Field(..., description="Topic to promote")
    reason: str = Field(..., description="Why this promotion is needed")
    context: Optional[str] = Field(default=None, description="Additional context for synthesis")
    include_related: bool = Field(default=True, description="Include related states in synthesis")
    max_history: int = Field(default=10, description="Max historical states to consider")
    tags: list[str] = Field(default_factory=list, description="Tags for new state")


class PromotionResult(BaseModel):
    """Result of a memory promotion"""

    new_state: MemoryState
    previous_state_id: Optional[UUID]
    synthesized_from: list[UUID] = Field(
        default_factory=list, description="States that contributed to synthesis"
    )
    new_edges: list[MemoryEdge] = Field(default_factory=list, description="Edges created")
    reasoning: str = Field(..., description="Why/how the new state was created")
