"""Graph models for edges and relationships"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between memory states"""

    # Evolution relationships
    EVOLVED_TO = "evolved_to"
    EVOLVED_FROM = "evolved_from"
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"

    # Content relationships
    RELATES_TO = "relates_to"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    DEPENDS_ON = "depends_on"
    REQUIRED_BY = "required_by"

    # Causal relationships
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    ENABLED_BY = "enabled_by"
    PREVENTS = "prevents"
    PREVENTED_BY = "prevented_by"

    # Semantic relationships
    SUPPORTS = "supports"
    SUPPORTED_BY = "supported_by"
    CONTRADICTS = "contradicts"
    CONTRADICTED_BY = "contradicted_by"
    EXPLAINS = "explains"
    EXPLAINED_BY = "explained_by"

    # Structural relationships
    PART_OF = "part_of"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"

    # Domain-specific relationships (can be extended)
    MENTIONS = "mentions"
    MENTIONED_BY = "mentioned_by"
    IMPLEMENTS = "implements"
    IMPLEMENTED_BY = "implemented_by"


class Edge(BaseModel):
    """Model for a relationship edge between memory states"""

    id: str | None = None
    source_state_id: UUID = Field(description="Origin state")
    target_state_id: UUID = Field(description="Destination state")
    relationship_type: RelationshipType = Field(description="Type of relationship")

    # Edge properties
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance/importance score (0-1)",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable explanation of the relationship",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Temporal validity (critical for accuracy)
    valid_from: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this edge became true",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="When this edge stopped being true (None = still valid)",
    )
    superseded_by: str | None = Field(
        default=None,
        description="Edge ID that replaced this one",
    )

    # Metadata
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the relationship",
    )

    @property
    def is_active(self) -> bool:
        """Check if edge is currently active"""
        if self.valid_until is None:
            return True
        return datetime.utcnow() < self.valid_until

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j relationship properties"""
        # Handle relationship_type - it may be string or enum depending on use_enum_values
        rel_type = (
            self.relationship_type.value
            if hasattr(self.relationship_type, "value")
            else self.relationship_type
        )

        return {
            "relationship_type": rel_type,
            "strength": self.strength,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "superseded_by": self.superseded_by,
            "is_active": self.is_active,
            "metadata": self.custom_metadata,
        }

    class Config:
        use_enum_values = True


class EdgeStrengthFactors(BaseModel):
    """Factors for calculating edge strength"""

    semantic_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity between connected states",
    )
    temporal_proximity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How close in time the states were created",
    )
    co_occurrence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How often states are retrieved together",
    )
    explicit_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Manually assigned importance",
    )
    access_pattern: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Strength from access patterns",
    )


class GraphStats(BaseModel):
    """Statistics about the knowledge graph"""

    total_nodes: int = 0
    total_edges: int = 0
    active_edges: int = 0
    inactive_edges: int = 0
    avg_edges_per_node: float = 0.0
    relationship_type_distribution: dict[str, int] = Field(default_factory=dict)
    avg_edge_strength: float = 0.0
