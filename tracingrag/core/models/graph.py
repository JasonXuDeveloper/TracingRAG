"""Graph models for edges and relationships"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RelationshipType(str, Enum):
    """Types of relationships between memory states

    Naming convention:
    - Version-bound relationships: Past tense (CAUSED_BY, CREATED_BY) - bind to specific version
    - Dynamic relationships: Present tense (RELATES_TO, AWARE_OF) - update to latest version
    """

    # Evolution relationships (version-bound - record historical transitions)
    EVOLVED_TO = "EVOLVED_TO"
    EVOLVED_FROM = "EVOLVED_FROM"
    SUPERSEDES = "SUPERSEDES"
    SUPERSEDED_BY = "SUPERSEDED_BY"

    # Content relationships (dynamic - update to latest)
    RELATES_TO = "RELATES_TO"
    REFERENCES = "REFERENCES"
    REFERENCED_BY = "REFERENCED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    REQUIRED_BY = "REQUIRED_BY"

    # Causal relationships (version-bound - historical causality)
    CAUSES = "CAUSES"
    CAUSED_BY = "CAUSED_BY"
    ENABLES = "ENABLES"
    ENABLED_BY = "ENABLED_BY"
    PREVENTS = "PREVENTS"
    PREVENTED_BY = "PREVENTED_BY"

    # Semantic relationships (dynamic - based on current content)
    SUPPORTS = "SUPPORTS"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTS = "CONTRADICTS"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    EXPLAINS = "EXPLAINS"
    EXPLAINED_BY = "EXPLAINED_BY"

    # Structural relationships (dynamic - hierarchical)
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    SIMILAR_TO = "SIMILAR_TO"
    DERIVED_FROM = "DERIVED_FROM"

    # Domain-specific relationships (can be extended)
    MENTIONS = "MENTIONS"
    MENTIONED_BY = "MENTIONED_BY"
    IMPLEMENTS = "IMPLEMENTS"
    IMPLEMENTED_BY = "IMPLEMENTED_BY"

    # Awareness and visibility relationships (asymmetric knowledge)
    AWARE_OF = "AWARE_OF"  # A knows about B (dynamic - should know latest)
    KNOWN_BY = "KNOWN_BY"  # A is known by B
    MONITORS = "MONITORS"  # A actively tracks B (dynamic - needs latest)
    MONITORED_BY = "MONITORED_BY"  # A is monitored by B
    HIDDEN_FROM = "HIDDEN_FROM"  # A is hidden from B (version-bound)
    UNAWARE_OF = "UNAWARE_OF"  # A doesn't know about B (version-bound)

    # Interaction relationships (version-bound - record specific events)
    INFLUENCED_BY = "INFLUENCED_BY"  # A was influenced by B at specific time
    INFLUENCED = "INFLUENCED"  # A influenced B
    CREATED_BY = "CREATED_BY"  # A was created by B (historical fact)
    CREATED = "CREATED"  # A created B

    @classmethod
    def is_version_bound(cls, rel_type: "RelationshipType") -> bool:
        """Check if relationship type should bind to specific version

        Version-bound relationships record historical facts and should NOT auto-update.
        Dynamic relationships reflect current state and SHOULD update to latest version.

        Returns:
            True if relationship should bind to specific version (historical fact)
            False if relationship should update to latest version (current state)
        """
        # Version-bound: Historical causality, events, and awareness states
        VERSION_BOUND_TYPES = {
            # Evolution (historical transitions)
            cls.EVOLVED_TO,
            cls.EVOLVED_FROM,
            cls.SUPERSEDES,
            cls.SUPERSEDED_BY,
            # Causal (past events)
            cls.CAUSES,
            cls.CAUSED_BY,
            cls.ENABLES,
            cls.ENABLED_BY,
            cls.PREVENTS,
            cls.PREVENTED_BY,
            # Interaction (specific events)
            cls.INFLUENCED_BY,
            cls.INFLUENCED,
            cls.CREATED_BY,
            cls.CREATED,
            # Awareness states (what was known at specific time)
            cls.HIDDEN_FROM,
            cls.UNAWARE_OF,
        }
        return rel_type in VERSION_BOUND_TYPES


class Edge(BaseModel):
    """Model for a relationship edge between memory states"""

    id: str | None = None
    source_state_id: UUID = Field(description="Origin state")
    target_state_id: UUID = Field(description="Destination state")
    relationship_type: RelationshipType = Field(description="Type of relationship")

    # Version binding (critical for historical accuracy)
    bound_to_version: bool = Field(
        default=False,
        description="If True, relationship is bound to specific target version and won't auto-update",
    )
    target_version_at_creation: int | None = Field(
        default=None,
        description="Target's version when this relationship was created (for historical tracking)",
    )

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

    # Importance tracking
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Computed importance score for prioritization (0-1)",
    )
    access_count: int = Field(
        default=0,
        description="Number of times this relationship was accessed in queries",
    )
    last_accessed: datetime | None = Field(
        default=None,
        description="Last time this relationship was accessed",
    )

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

    def compute_importance(self, target_confidence: float = 0.5) -> float:
        """
        Compute importance score for relationship prioritization

        Formula:
        importance = (
            strength * 0.4 +                    # Relationship strength
            access_frequency * 0.3 +            # How often accessed
            target_confidence * 0.2 +           # Quality of target state
            time_decay * 0.1                    # Recency factor
        )

        Args:
            target_confidence: Confidence score of target state (0-1)

        Returns:
            Importance score (0-1)
        """
        # Access frequency (normalized by sigmoid)
        # 0 accesses = 0.0, 10 accesses = 0.5, 100+ accesses = 1.0
        access_freq = min(1.0, self.access_count / 50.0) if self.access_count else 0.0

        # Time decay: recent access = higher importance
        # 0 days = 1.0, 30 days = 0.5, 180+ days = 0.1
        if self.last_accessed:
            days_since_access = (datetime.utcnow() - self.last_accessed).days
            time_decay = max(0.1, 1.0 - (days_since_access / 180.0))
        else:
            # Never accessed = use creation time
            days_since_creation = (datetime.utcnow() - self.created_at).days
            time_decay = max(0.1, 1.0 - (days_since_creation / 180.0))

        # Weighted combination
        importance = (
            self.strength * 0.4 + access_freq * 0.3 + target_confidence * 0.2 + time_decay * 0.1
        )

        return min(1.0, max(0.0, importance))

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
            # Version binding fields
            "bound_to_version": self.bound_to_version,
            "target_version_at_creation": self.target_version_at_creation,
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
