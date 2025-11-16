"""Shared type definitions for TracingRAG

This module contains all shared Pydantic models used across:
- API server (schemas)
- SDK client
- Agent system
- Internal services

Centralizing types here avoids duplication and ensures consistency.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ============================================================================
# Core Response Types
# ============================================================================


class Citation(BaseModel):
    """Citation referencing a source topic"""

    topic: str = Field(..., description="Exact topic name from the source")
    insight: str = Field(..., description="What this source contributes to the answer")


class MemoryStateResponse(BaseModel):
    """Memory state response (used by API and SDK)"""

    model_config = {"from_attributes": True}

    id: UUID
    topic: str
    content: str
    version: int
    timestamp: datetime
    parent_state_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source: str | None = None


# ============================================================================
# Agent Types
# ============================================================================


class MemorySuggestion(BaseModel):
    """Suggestion for memory management (used by agents)"""

    suggestion_type: str = Field(
        ..., description="Type: promotion, connection, consolidation, conflict"
    )
    target_states: list[UUID] = Field(default_factory=list, description="States involved")
    rationale: str = Field(..., description="Why this suggestion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    priority: int = Field(default=5, ge=1, le=10, description="Priority 1-10")
    metadata: dict[str, Any] = Field(default_factory=dict)
