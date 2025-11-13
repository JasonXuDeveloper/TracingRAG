"""Models for memory promotion and synthesis"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PromotionMode(str, Enum):
    """Modes for promotion automation"""

    MANUAL = "manual"  # Only on explicit user request
    AUTOMATIC = "automatic"  # LLM decides and executes


class PromotionTrigger(str, Enum):
    """Triggers that can initiate memory promotion"""

    MANUAL = "manual"  # User-initiated
    AUTO_VERSION_COUNT = "auto_version_count"  # Too many versions
    AUTO_TIME_BASED = "auto_time_based"  # Time-based consolidation
    AUTO_COMPLEXITY = "auto_complexity"  # Growing complexity
    AUTO_CONFLICT = "auto_conflict"  # Conflicting information
    AUTO_RELATED_GROWTH = "auto_related_growth"  # Many related states


class ConflictType(str, Enum):
    """Types of conflicts between memory states"""

    CONTRADICTION = "contradiction"  # Direct contradiction
    INCONSISTENCY = "inconsistency"  # Inconsistent information
    AMBIGUITY = "ambiguity"  # Ambiguous or unclear
    TEMPORAL = "temporal"  # Time-based conflict


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts"""

    LATEST_WINS = "latest_wins"  # Use most recent
    HIGHEST_CONFIDENCE = "highest_confidence"  # Use highest confidence
    MERGE = "merge"  # Merge information
    MANUAL = "manual"  # Require human intervention
    LLM_DECIDE = "llm_decide"  # Let LLM decide


class QualityCheckType(str, Enum):
    """Types of quality checks"""

    HALLUCINATION = "hallucination"  # Check for hallucinations
    CITATION = "citation"  # Verify citations
    CONSISTENCY = "consistency"  # Check consistency
    COMPLETENESS = "completeness"  # Check completeness
    RELEVANCE = "relevance"  # Check relevance


class Conflict(BaseModel):
    """Represents a conflict between memory states"""

    state_ids: list[UUID] = Field(..., description="IDs of conflicting states")
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    description: str = Field(..., description="Description of the conflict")
    severity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Severity (0.0-1.0)"
    )
    resolution_strategy: ConflictResolutionStrategy = Field(
        default=ConflictResolutionStrategy.LLM_DECIDE,
        description="Recommended resolution strategy",
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional conflict details"
    )


class ConflictResolution(BaseModel):
    """Result of resolving a conflict"""

    conflict: Conflict = Field(..., description="Original conflict")
    strategy_used: ConflictResolutionStrategy = Field(
        ..., description="Strategy that was used"
    )
    resolution: str = Field(..., description="How the conflict was resolved")
    winning_state_id: UUID | None = Field(
        default=None, description="ID of winning state (if applicable)"
    )
    merged_content: str | None = Field(
        default=None, description="Merged content (if applicable)"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in resolution"
    )
    manual_review_needed: bool = Field(
        default=False, description="Whether manual review is needed"
    )


class QualityCheck(BaseModel):
    """Result of a quality check"""

    check_type: QualityCheckType = Field(..., description="Type of check performed")
    passed: bool = Field(..., description="Whether check passed")
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score")
    issues: list[str] = Field(default_factory=list, description="Issues found")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )


class EdgeUpdate(BaseModel):
    """Represents an edge update during promotion"""

    source_id: UUID = Field(..., description="Source state ID")
    target_id: UUID = Field(..., description="Target state ID")
    relationship_type: str = Field(..., description="Relationship type")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Edge strength")
    action: str = Field(
        ..., description="Action (create, update, delete, carry_forward)"
    )
    reasoning: str = Field(default="", description="Why this edge update")


class SynthesisSource(BaseModel):
    """Represents a source used in synthesis"""

    state_id: UUID = Field(..., description="Source state ID")
    topic: str = Field(..., description="State topic")
    content: str = Field(..., description="State content")
    version: int = Field(..., description="State version")
    timestamp: datetime = Field(..., description="State timestamp")
    confidence: float = Field(..., description="State confidence")
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Weight in synthesis"
    )
    reasoning: str = Field(default="", description="Why this source was included")


class PromotionCandidate(BaseModel):
    """Candidate for memory promotion"""

    topic: str = Field(..., description="Topic to promote")
    trigger: PromotionTrigger = Field(..., description="What triggered consideration")
    priority: int = Field(
        default=5, ge=1, le=10, description="Priority (1=low, 10=high)"
    )
    reasoning: str = Field(..., description="Why this should be promoted")
    current_version_count: int = Field(..., description="Current number of versions")
    last_promoted: datetime | None = Field(
        default=None, description="When last promoted"
    )
    estimated_complexity: int = Field(
        default=5, ge=1, le=10, description="Estimated complexity"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in need for promotion"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class PromotionResult(BaseModel):
    """Result of a memory promotion operation"""

    success: bool = Field(..., description="Whether promotion succeeded")
    previous_state_id: UUID | None = Field(
        default=None, description="ID of previous state"
    )
    new_state_id: UUID | None = Field(default=None, description="ID of new state")
    topic: str = Field(..., description="Topic that was promoted")
    new_version: int | None = Field(default=None, description="New version number")
    synthesized_from: list[SynthesisSource] = Field(
        default_factory=list, description="Sources used in synthesis"
    )
    conflicts_detected: list[Conflict] = Field(
        default_factory=list, description="Conflicts detected"
    )
    conflicts_resolved: list[ConflictResolution] = Field(
        default_factory=list, description="Conflicts resolved"
    )
    edges_updated: list[EdgeUpdate] = Field(
        default_factory=list, description="Edge updates performed"
    )
    quality_checks: list[QualityCheck] = Field(
        default_factory=list, description="Quality checks performed"
    )
    reasoning: str = Field(..., description="Explanation of synthesis")
    synthesized_content: str | None = Field(
        default=None, description="New synthesized content"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in promotion"
    )
    manual_review_needed: bool = Field(
        default=False, description="Whether manual review is needed"
    )
    error_message: str | None = Field(default=None, description="Error if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When promotion occurred"
    )


class PromotionRequest(BaseModel):
    """Request to promote a memory state"""

    topic: str = Field(..., description="Topic to promote")
    reason: str = Field(..., description="Reason for promotion")
    trigger: PromotionTrigger = Field(
        default=PromotionTrigger.MANUAL, description="What triggered promotion"
    )
    include_related: bool = Field(
        default=True, description="Include related states in synthesis"
    )
    max_sources: int = Field(
        default=10, description="Maximum sources to include in synthesis"
    )
    conflict_resolution_strategy: ConflictResolutionStrategy = Field(
        default=ConflictResolutionStrategy.LLM_DECIDE,
        description="How to resolve conflicts",
    )
    quality_checks_enabled: bool = Field(
        default=True, description="Enable quality checks"
    )
    auto_resolve_conflicts: bool = Field(
        default=True, description="Auto-resolve conflicts if possible"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class PromotionPolicy(BaseModel):
    """Policy configuration for memory promotion automation"""

    mode: PromotionMode = Field(
        default=PromotionMode.MANUAL, description="Automation mode"
    )

    # Detection thresholds
    version_count_threshold: int = Field(
        default=5, ge=2, description="Min versions before considering promotion"
    )
    time_threshold_days: int = Field(
        default=7, ge=1, description="Days since last promotion before considering"
    )
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Min confidence for auto-promotion"
    )

    # LLM-based evaluation
    use_llm_evaluation: bool = Field(
        default=True, description="Use LLM to evaluate promotion necessity"
    )
    evaluation_model: str = Field(
        default="deepseek/deepseek-chat-v3-0324:free",
        description="Model for promotion evaluation",
    )

    # Safety settings
    require_approval_for_conflicts: bool = Field(
        default=True, description="Require approval if conflicts detected"
    )
    notify_on_auto_promotion: bool = Field(
        default=True, description="Notify user after automatic promotion"
    )
    dry_run: bool = Field(
        default=False, description="Simulate promotions without executing"
    )

    # Trigger-specific settings
    enabled_triggers: list[PromotionTrigger] = Field(
        default_factory=lambda: [
            PromotionTrigger.AUTO_VERSION_COUNT,
            PromotionTrigger.AUTO_TIME_BASED,
        ],
        description="Which auto-promotion triggers are enabled",
    )

    # Resource limits
    max_concurrent_promotions: int = Field(
        default=3, ge=1, le=10, description="Max concurrent promotions in batch"
    )
    max_candidates_per_scan: int = Field(
        default=20, ge=1, description="Max candidates to evaluate per scan"
    )


class PromotionEvaluation(BaseModel):
    """Result of evaluating whether a topic should be promoted"""

    topic: str = Field(..., description="Topic evaluated")
    should_promote: bool = Field(..., description="Whether promotion is recommended")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in recommendation"
    )
    priority: int = Field(..., ge=1, le=10, description="Priority (1=low, 10=high)")
    trigger: PromotionTrigger = Field(..., description="What triggered evaluation")
    reasoning: str = Field(..., description="Explanation of decision")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Metrics used in evaluation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When evaluation occurred"
    )
