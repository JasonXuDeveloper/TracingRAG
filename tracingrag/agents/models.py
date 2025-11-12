"""Data models for agent system"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    """Types of actions an agent can take"""

    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    TRACE_HISTORY = "trace_history"
    CREATE_MEMORY = "create_memory"
    CREATE_EDGE = "create_edge"
    ANALYZE_QUERY = "analyze_query"
    GENERATE_RESPONSE = "generate_response"
    VALIDATE_RESULTS = "validate_results"


class RetrievalPlan(BaseModel):
    """Plan for retrieving information"""

    query: str = Field(..., description="Original query")
    steps: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered steps to execute"
    )
    rationale: str = Field(default="", description="Explanation of the plan")
    estimated_complexity: int = Field(default=1, description="1-10 complexity score")


class AgentStep(BaseModel):
    """A single step in agent execution"""

    step_id: UUID = Field(default_factory=uuid4)
    action: AgentAction = Field(..., description="Action to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action"
    )
    result: Any = Field(default=None, description="Result of executing the step")
    error: str | None = Field(default=None, description="Error if step failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = Field(default=0.0, description="Execution time in ms")


class AgentState(BaseModel):
    """State for agent execution"""

    # Query information
    query: str = Field(..., description="User query")
    query_type: str = Field(default="general", description="Detected query type")
    query_embedding: list[float] | None = Field(
        default=None, description="Query vector embedding"
    )

    # Planning
    plan: RetrievalPlan | None = Field(default=None, description="Retrieval plan")
    current_step: int = Field(default=0, description="Current step index")

    # Execution
    steps: list[AgentStep] = Field(
        default_factory=list, description="Executed steps"
    )
    retrieved_states: list[Any] = Field(
        default_factory=list, description="Retrieved memory states"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Accumulated context"
    )

    # Results
    answer: str = Field(default="", description="Generated answer")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[UUID] = Field(default_factory=list, description="Source state IDs")

    # Metadata
    needs_replanning: bool = Field(
        default=False, description="Whether to replan the strategy"
    )
    iteration: int = Field(default=0, description="Number of replanning iterations")
    max_iterations: int = Field(default=3, description="Max replanning iterations")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySuggestion(BaseModel):
    """Suggestion for memory management"""

    suggestion_type: str = Field(
        ..., description="Type: promotion, connection, consolidation, conflict"
    )
    target_states: list[UUID] = Field(
        default_factory=list, description="States involved"
    )
    rationale: str = Field(..., description="Why this suggestion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    priority: int = Field(default=5, ge=1, le=10, description="Priority 1-10")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Result from agent execution"""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    reasoning_steps: list[AgentStep] = Field(
        default_factory=list, description="Steps taken by agent"
    )
    sources: list[UUID] = Field(default_factory=list, description="Source state IDs")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    plan_used: RetrievalPlan | None = Field(default=None)
    replanning_count: int = Field(default=0, description="Number of replan iterations")
    total_time_ms: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
