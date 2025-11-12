"""LLM-based query analysis for intelligent query understanding"""

import json
from typing import Any

from tracingrag.core.models.rag import ConsolidationLevel, QueryType
from tracingrag.services.llm import LLMClient, get_llm_client


class QueryAnalyzer:
    """Analyzes queries using LLM to understand intent, type, and requirements"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        default_model: str = "deepseek/deepseek-chat-v3-0324:free",
    ):
        """
        Initialize query analyzer

        Args:
            llm_client: LLM client for query analysis (uses singleton if None)
            default_model: Default model for query classification (free/cheap model recommended)
        """
        self.llm_client = llm_client or get_llm_client()
        self.default_model = default_model

    async def analyze_query(
        self, query: str, use_llm: bool = True
    ) -> dict[str, Any]:
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

        # Define JSON schema for query analysis
        json_schema = {
            "name": "query_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "status",
                            "recent",
                            "overview",
                            "why",
                            "how",
                            "what",
                            "when",
                            "comparison",
                            "general",
                        ],
                        "description": "Type of query",
                    },
                    "consolidation_level": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Consolidation level (0=RAW, 1=DAILY, 2=WEEKLY, 3=MONTHLY)",
                    },
                    "needs_history": {
                        "type": "boolean",
                        "description": "Whether historical context is needed",
                    },
                    "needs_graph": {
                        "type": "boolean",
                        "description": "Whether graph relationships are needed",
                    },
                    "time_scope": {
                        "type": "string",
                        "enum": ["current", "recent", "historical", "all"],
                        "description": "Time scope of the query",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extracted entities from query",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of classification",
                    },
                },
                "required": [
                    "query_type",
                    "consolidation_level",
                    "needs_history",
                    "needs_graph",
                    "time_scope",
                    "entities",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt=self._get_analysis_system_prompt(),
            user_message=analysis_prompt,
            context="",  # No context needed for analysis
            model=self.default_model,  # Use configurable model (cheap/free recommended)
            temperature=0.0,  # Deterministic for classification
            max_tokens=500,  # Short response
            json_schema=json_schema,  # Use JSON schema for strict structured output
            metadata={"task": "query_analysis"},
        )

        # Get response
        response = await self.llm_client.generate(request)

        # Parse structured JSON response
        try:
            analysis = json.loads(response.content)

            # Convert string values to enums
            query_type = QueryType(analysis.get("query_type", "general"))
            consolidation_level = ConsolidationLevel(
                analysis.get("consolidation_level", 2)
            )

            return {
                "query_type": query_type,
                "consolidation_level": consolidation_level,
                "needs_history": analysis.get("needs_history", False),
                "needs_graph": analysis.get("needs_graph", True),
                "time_scope": analysis.get("time_scope", "current"),
                "entities": analysis.get("entities", []),
                "reasoning": analysis.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to rule-based if LLM response is invalid
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
        elif (
            query_lower.startswith("why")
            or "why did" in query_lower
            or "why " in query_lower
        ):
            query_type = QueryType.WHY

        # How queries
        elif (
            query_lower.startswith("how")
            or "how does" in query_lower
            or "how to" in query_lower
        ):
            query_type = QueryType.HOW

        # Comparison queries
        elif any(
            phrase in query_lower
            for phrase in ["compare", "difference", "versus", "vs ", " vs"]
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

    def _determine_consolidation_level_rules(
        self, query_type: QueryType
    ) -> ConsolidationLevel:
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
    default_model: str = "deepseek/deepseek-chat-v3-0324:free",
) -> QueryAnalyzer:
    """
    Get or create query analyzer singleton

    Args:
        llm_client: Optional LLM client
        default_model: Default model for query classification (cheap/free model recommended)

    Returns:
        QueryAnalyzer instance
    """
    global _query_analyzer

    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer(
            llm_client=llm_client, default_model=default_model
        )

    return _query_analyzer
