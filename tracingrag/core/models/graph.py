"""Graph models for edges and relationships"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RelationshipType(str, Enum):
    """Types of relationships between memory states"""

    # Evolution relationships
    EVOLVED_TO = "EVOLVED_TO"
    EVOLVED_FROM = "EVOLVED_FROM"
    SUPERSEDES = "SUPERSEDES"
    SUPERSEDED_BY = "SUPERSEDED_BY"

    # Content relationships
    RELATES_TO = "RELATES_TO"
    REFERENCES = "REFERENCES"
    REFERENCED_BY = "REFERENCED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    REQUIRED_BY = "REQUIRED_BY"

    # Causal relationships
    CAUSES = "CAUSES"
    CAUSED_BY = "CAUSED_BY"
    ENABLES = "ENABLES"
    ENABLED_BY = "ENABLED_BY"
    PREVENTS = "PREVENTS"
    PREVENTED_BY = "PREVENTED_BY"

    # Semantic relationships
    SUPPORTS = "SUPPORTS"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTS = "CONTRADICTS"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    EXPLAINS = "EXPLAINS"
    EXPLAINED_BY = "EXPLAINED_BY"

    # Structural relationships
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    SIMILAR_TO = "SIMILAR_TO"
    DERIVED_FROM = "DERIVED_FROM"

    # Domain-specific relationships (can be extended)
    MENTIONS = "MENTIONS"
    MENTIONED_BY = "MENTIONED_BY"
    IMPLEMENTS = "IMPLEMENTS"
    IMPLEMENTED_BY = "IMPLEMENTED_BY"


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

    model_config = ConfigDict(use_enum_values=True)


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
