"""RAG-specific models for query processing and response generation"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .memory import MemoryState


class QueryType(str, Enum):
    """Types of user queries"""

    STATUS = "status"  # "What's the current status?"
    RECENT = "recent"  # "What happened last week?"
    OVERVIEW = "overview"  # "Summarize the project"
    WHY = "why"  # "Why did X happen?"
    HOW = "how"  # "How does X work?"
    WHAT = "what"  # "What is X?"
    WHEN = "when"  # "When did X happen?"
    COMPARISON = "comparison"  # "Compare X and Y"
    GENERAL = "general"  # Generic query


class ConsolidationLevel(int, Enum):
    """Consolidation levels for historical context"""

    RAW = 0  # Raw states, no consolidation
    DAILY = 1  # Daily summaries
    WEEKLY = 2  # Weekly summaries
    MONTHLY = 3  # Monthly summaries


class RAGContext(BaseModel):
    """Complete context for RAG pipeline"""

    query: str = Field(..., description="Original user query")
    query_type: QueryType = Field(
        default=QueryType.GENERAL, description="Detected query type"
    )
    query_embedding: list[float] | None = Field(
        default=None, description="Vector embedding of query"
    )

    # Retrieved states organized by category
    latest_states: list[MemoryState] = Field(
        default_factory=list,
        description="Latest states for relevant topics (always included)",
    )
    summaries: list[MemoryState] = Field(
        default_factory=list, description="Consolidated summaries"
    )
    detailed_states: list[MemoryState] = Field(
        default_factory=list, description="Detailed states from selective drill-down"
    )
    related_graph: list[dict[str, Any]] = Field(
        default_factory=list, description="Graph relationships between states"
    )

    # Context window management
    max_tokens: int = Field(default=100000, description="Max context tokens")
    tokens_used: int = Field(default=0, description="Tokens used so far")
    tokens_remaining: int = Field(default=100000, description="Tokens remaining")

    # Metadata
    consolidation_level: ConsolidationLevel = Field(
        default=ConsolidationLevel.RAW, description="Consolidation level used"
    )
    topics_considered: list[str] = Field(
        default_factory=list, description="Topics included in context"
    )
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional retrieval info"
    )


class LLMRequest(BaseModel):
    """Request to LLM for generation"""

    system_prompt: str = Field(..., description="System instructions")
    user_message: str = Field(..., description="User query")
    context: str = Field(..., description="Retrieved context formatted for LLM")
    model: str = Field(default="anthropic/claude-3.5-sonnet", description="Model ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    json_mode: bool = Field(
        default=False, description="Request structured JSON output (simple mode)"
    )
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for structured output (preferred over json_mode)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from LLM"""

    content: str = Field(..., description="Generated response")
    model: str = Field(..., description="Model used")
    tokens_used: int = Field(default=0, description="Tokens used in generation")
    finish_reason: str | None = Field(default=None, description="Why generation stopped")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Complete RAG pipeline response"""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: list[UUID] = Field(
        default_factory=list, description="Memory state IDs used as sources"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in answer"
    )

    # Context information
    context_used: dict[str, Any] = Field(
        default_factory=dict, description="Summary of context used"
    )
    query_type: QueryType = Field(
        default=QueryType.GENERAL, description="Detected query type"
    )

    # Performance metrics
    retrieval_time_ms: float = Field(default=0.0, description="Time for retrieval")
    generation_time_ms: float = Field(default=0.0, description="Time for generation")
    total_time_ms: float = Field(default=0.0, description="Total time")
    tokens_retrieved: int = Field(default=0, description="Tokens in retrieved context")
    tokens_generated: int = Field(default=0, description="Tokens in generated response")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBudget(BaseModel):
    """Budget allocation for context window"""

    max_tokens: int = Field(..., description="Total available tokens")
    reserved_buffer: int = Field(
        default=10000, description="Reserve for LLM reasoning"
    )
    latest_states_budget: int = Field(
        default=20000, description="Budget for latest states"
    )
    summaries_budget: int = Field(default=30000, description="Budget for summaries")
    details_budget: int = Field(
        default=0, description="Budget for detailed states (computed)"
    )

    def compute_details_budget(self) -> int:
        """Compute remaining budget for details"""
        allocated = (
            self.reserved_buffer + self.latest_states_budget + self.summaries_budget
        )
        remaining = self.max_tokens - allocated
        self.details_budget = max(0, remaining)
        return self.details_budget


class TokenEstimate(BaseModel):
    """Token count estimate for text"""

    text: str = Field(..., description="Text to estimate")
    estimated_tokens: int = Field(..., description="Estimated token count")
    method: str = Field(default="char_count", description="Estimation method used")

    @staticmethod
    def estimate(text: str, method: str = "char_count") -> "TokenEstimate":
        """Estimate tokens for given text"""
        if method == "char_count":
            # Rough estimate: ~4 chars per token
            tokens = len(text) // 4
        elif method == "word_count":
            # Rough estimate: ~1.3 tokens per word
            tokens = int(len(text.split()) * 1.3)
        else:
            # Default to char count
            tokens = len(text) // 4

        return TokenEstimate(text=text, estimated_tokens=tokens, method=method)
