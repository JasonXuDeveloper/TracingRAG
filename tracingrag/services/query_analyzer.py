"""LLM-based query analysis for intelligent query understanding"""

import json
from typing import Any

from pydantic import BaseModel, Field

from tracingrag.config import settings
from tracingrag.core.models.rag import ConsolidationLevel, QueryType
from tracingrag.services.llm import LLMClient, get_llm_client


class QueryAnalysisSchema(BaseModel):
    """Pydantic schema for query analysis structured output"""

    query_type: QueryType = Field(..., description="Type of query")
    consolidation_level: ConsolidationLevel = Field(
        ..., description="Consolidation level (0=RAW, 1=DAILY, 2=WEEKLY, 3=MONTHLY)"
    )
    needs_history: bool = Field(..., description="Whether historical context is needed")
    needs_graph: bool = Field(..., description="Whether graph relationships are needed")
    time_scope: str = Field(
        ...,
        description="Time scope of the query",
        pattern="^(current|recent|historical|all)$",
    )
    entities: list[str] = Field(default_factory=list, description="Extracted entities from query")
    reasoning: str = Field(..., description="Brief explanation of classification")


class QueryAnalyzer:
    """Analyzes queries using LLM to understand intent, type, and requirements"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        default_model: str | None = None,
    ):
        """
        Initialize query analyzer

        Args:
            llm_client: LLM client for query analysis (uses singleton if None)
            default_model: Default model for query classification (uses settings if None)
        """
        self._llm_client = llm_client  # Store provided client or None
        self.default_model = default_model or settings.query_analyzer_model

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client only when needed"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def analyze_query(self, query: str, use_llm: bool = True) -> dict[str, Any]:
        """
        Analyze query to determine type, consolidation level, and other metadata

        Args:
            query: User query in any language
            use_llm: Whether to use LLM analysis (fallback to rules if False)

        Returns:
            Dictionary with:
            - query_type: QueryType enum value
            - consolidation_level: ConsolidationLevel enum value
            - needs_history: bool
            - needs_graph: bool
            - time_scope: str (recent, current, historical, all)
            - entities: list[str] (extracted entities)
            - reasoning: str (why this classification)
        """
        if use_llm:
            return await self._analyze_with_llm(query)
        else:
            return self._analyze_with_rules(query)

    async def _analyze_with_llm(self, query: str) -> dict[str, Any]:
        """
        Use LLM to analyze query with structured output

        This works for ANY language, not just English
        """
        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(query)

        # Create LLM request for structured output
        from tracingrag.core.models.rag import LLMRequest

        request = LLMRequest(
            system_prompt=self._get_analysis_system_prompt(),
            user_message=analysis_prompt,
            context="",  # No context needed for analysis
            model=self.default_model,  # Use configurable model (cheap/free recommended)
            temperature=0.0,  # Deterministic for classification
            max_tokens=4000,  # Generous limit to prevent truncation while keeping costs reasonable
            json_schema=QueryAnalysisSchema.model_json_schema(),  # Pydantic schema (auto-formatted by LLM client)
            metadata={"task": "query_analysis", "schema_name": "query_analysis"},
        )

        # Get response
        response = await self.llm_client.generate(request)

        # Parse and validate structured JSON response using Pydantic
        try:
            # Parse JSON and validate with Pydantic (enums are auto-converted)
            analysis_data = json.loads(response.content)
            validated = QueryAnalysisSchema.model_validate(analysis_data)

            return {
                "query_type": validated.query_type,  # Already QueryType enum
                "consolidation_level": validated.consolidation_level,  # Already ConsolidationLevel enum
                "needs_history": validated.needs_history,
                "needs_graph": validated.needs_graph,
                "time_scope": validated.time_scope,
                "entities": validated.entities,
                "reasoning": validated.reasoning,
            }
        except (json.JSONDecodeError, KeyError, ValueError, Exception) as e:
            # Fallback to rule-based if LLM response is invalid
            from tracingrag.utils.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Query analysis LLM parsing failed: {e}. Falling back to rules.")
            return self._analyze_with_rules(query)

    def _build_analysis_prompt(self, query: str) -> str:
        """Build prompt for query analysis"""
        # Get available query types
        query_types = {
            "status": "Current state queries (What's happening now?)",
            "recent": "Recent history queries (What happened lately?)",
            "overview": "Summary/overview queries (Summarize everything)",
            "why": "Causal/explanation queries (Why did X happen?)",
            "how": "Process/method queries (How does X work?)",
            "what": "Definition/information queries (What is X?)",
            "when": "Temporal queries (When did X happen?)",
            "comparison": "Comparison queries (Compare X and Y)",
            "general": "General queries that don't fit other categories",
        }

        # Get consolidation levels
        consolidation_info = {
            0: "RAW - Detailed states, no consolidation (for precise queries)",
            1: "DAILY - Daily summaries (for recent queries)",
            2: "WEEKLY - Weekly summaries (default, balanced)",
            3: "MONTHLY - Monthly summaries (for broad overviews)",
        }

        prompt = f"""Analyze this user query and provide structured classification.

Query: "{query}"

Your task:
1. Classify the query type from the available types
2. Determine appropriate consolidation level
3. Identify if historical context is needed
4. Identify if graph relationships are needed
5. Determine time scope
6. Extract key entities mentioned

Available Query Types:
{json.dumps(query_types, indent=2)}

Consolidation Levels:
{json.dumps(consolidation_info, indent=2)}

Time Scopes:
- "current": Focus on current/latest state
- "recent": Recent history (last few weeks)
- "historical": Broader history
- "all": Complete history

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
    "query_type": "status|recent|overview|why|how|what|when|comparison|general",
    "consolidation_level": 0|1|2|3,
    "needs_history": true|false,
    "needs_graph": true|false,
    "time_scope": "current|recent|historical|all",
    "entities": ["entity1", "entity2"],
    "reasoning": "Brief explanation of classification"
}}

Remember:
- Work with ANY language, not just English
- Focus on query INTENT, not specific words
- Use "general" type if uncertain
- Default consolidation_level is 2 (weekly)
- Default needs_graph is true
"""
        return prompt

    def _get_analysis_system_prompt(self) -> str:
        """System prompt for query analysis"""
        return """You are a query analysis assistant that classifies user queries to optimize retrieval strategies.

Your goal is to understand the user's INTENT regardless of language, phrasing, or complexity.

Key principles:
1. Language-agnostic: Work with English, Chinese, Spanish, or any language
2. Intent-focused: Understand what the user wants, not just keywords
3. Structured output: Always respond with valid JSON
4. Conservative: When uncertain, use "general" type and default values
5. Helpful: Provide reasoning to explain your classification

You are part of a RAG (Retrieval-Augmented Generation) system with:
- Temporal tracking (states evolve over time)
- Graph relationships (connections between concepts)
- Hierarchical consolidation (summaries at different granularities)

Your classification helps optimize:
- What context to retrieve
- How much history to include
- Which consolidation level to use
- Whether to traverse graph relationships"""

    def _analyze_with_rules(self, query: str) -> dict[str, Any]:
        """
        Fallback rule-based analysis (English only)

        This is the original rule-based approach, kept as fallback
        """
        query_lower = query.lower()

        # Detect query type
        query_type = QueryType.GENERAL

        # Status queries
        if any(
            phrase in query_lower
            for phrase in [
                "current status",
                "what's happening",
                "latest",
                "current state",
                "what is the status",
                "what's the status",
            ]
        ):
            query_type = QueryType.STATUS

        # Recent queries
        elif any(
            phrase in query_lower
            for phrase in [
                "last week",
                "last month",
                "recently",
                "what happened",
                "recent",
            ]
        ):
            query_type = QueryType.RECENT

        # Overview queries
        elif any(
            phrase in query_lower
            for phrase in ["summarize", "overview", "summary", "entire", "complete"]
        ):
            query_type = QueryType.OVERVIEW

        # Why queries
        elif query_lower.startswith("why") or "why did" in query_lower or "why " in query_lower:
            query_type = QueryType.WHY

        # How queries
        elif query_lower.startswith("how") or "how does" in query_lower or "how to" in query_lower:
            query_type = QueryType.HOW

        # Comparison queries
        elif any(
            phrase in query_lower for phrase in ["compare", "difference", "versus", "vs ", " vs"]
        ):
            query_type = QueryType.COMPARISON

        # What queries
        elif query_lower.startswith("what") or "what is" in query_lower:
            query_type = QueryType.WHAT

        # When queries
        elif query_lower.startswith("when") or "when did" in query_lower:
            query_type = QueryType.WHEN

        # Determine consolidation level based on query type
        consolidation_level = self._determine_consolidation_level_rules(query_type)

        # Determine other properties
        needs_history = query_type in [
            QueryType.RECENT,
            QueryType.OVERVIEW,
            QueryType.WHY,
            QueryType.HOW,
        ]
        needs_graph = query_type not in [QueryType.STATUS, QueryType.WHAT]

        time_scope = "current"
        if query_type == QueryType.RECENT:
            time_scope = "recent"
        elif query_type == QueryType.OVERVIEW:
            time_scope = "all"
        elif query_type in [QueryType.WHY, QueryType.HOW]:
            time_scope = "historical"

        return {
            "query_type": query_type,
            "consolidation_level": consolidation_level,
            "needs_history": needs_history,
            "needs_graph": needs_graph,
            "time_scope": time_scope,
            "entities": [],  # Can't extract entities with rules
            "reasoning": "Rule-based classification (fallback mode)",
        }

    def _determine_consolidation_level_rules(self, query_type: QueryType) -> ConsolidationLevel:
        """Determine consolidation level using rules"""
        if query_type == QueryType.STATUS:
            return ConsolidationLevel.RAW

        elif query_type == QueryType.RECENT:
            return ConsolidationLevel.DAILY

        elif query_type == QueryType.OVERVIEW:
            return ConsolidationLevel.MONTHLY

        elif query_type in [QueryType.WHY, QueryType.HOW]:
            return ConsolidationLevel.RAW

        return ConsolidationLevel.WEEKLY  # Default


# Singleton instance
_query_analyzer: QueryAnalyzer | None = None


def get_query_analyzer(
    llm_client: LLMClient | None = None,
    default_model: str | None = None,
) -> QueryAnalyzer:
    """
    Get or create query analyzer singleton

    Args:
        llm_client: Optional LLM client
        default_model: Default model for query classification (uses settings if None)

    Returns:
        QueryAnalyzer instance
    """
    global _query_analyzer

    if _query_analyzer is None:
        # Use settings.query_analyzer_model if no model specified
        model = default_model or settings.query_analyzer_model
        _query_analyzer = QueryAnalyzer(llm_client=llm_client, default_model=model)

    return _query_analyzer
