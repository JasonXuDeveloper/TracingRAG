"""Context building for RAG pipeline"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from tracingrag.core.models.memory import MemoryState
from tracingrag.core.models.rag import (
    ConsolidationLevel,
    ContextBudget,
    QueryType,
    RAGContext,
    TokenEstimate,
)
from tracingrag.services.query_analyzer import QueryAnalyzer, get_query_analyzer
from tracingrag.services.retrieval import RetrievalResult, RetrievalService


class ContextBuilder:
    """Builds context for RAG pipeline with intelligent budgeting"""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        use_llm_analysis: bool = True,
    ):
        """
        Initialize context builder

        Args:
            retrieval_service: Service for retrieving memory states
            query_analyzer: Query analyzer for intent detection
            use_llm_analysis: Whether to use LLM for query analysis (vs rules)
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.query_analyzer = query_analyzer or get_query_analyzer()
        self.use_llm_analysis = use_llm_analysis

    async def build_context(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        max_tokens: int = 100000,
        query_type: QueryType | None = None,
    ) -> RAGContext:
        """
        Build complete context for RAG pipeline

        Args:
            query: User query (any language)
            query_embedding: Pre-computed query embedding
            max_tokens: Maximum tokens for context
            query_type: Detected query type (auto-detect if None)

        Returns:
            RAG context with retrieved states and metadata
        """
        # Analyze query if type not provided
        if query_type is None:
            # Use LLM-based analysis for language-agnostic, intelligent detection
            analysis = await self.query_analyzer.analyze_query(
                query, use_llm=self.use_llm_analysis
            )
            query_type = analysis["query_type"]
            consolidation_level = analysis["consolidation_level"]
        else:
            # Use provided query type
            consolidation_level = self._determine_consolidation_level_rules(query_type)

        # Initialize context
        context = RAGContext(
            query=query,
            query_type=query_type,
            query_embedding=query_embedding,
            max_tokens=max_tokens,
            consolidation_level=consolidation_level,
        )

        # Create budget allocation
        budget = self._create_budget(max_tokens, query_type)

        # Phase 1: Get latest states (always included)
        latest_states = await self._get_latest_states(
            query=query,
            query_embedding=query_embedding,
            max_tokens=budget.latest_states_budget,
        )
        context.latest_states = latest_states

        # Update token counts
        latest_tokens = sum(
            TokenEstimate.estimate(state.content).estimated_tokens
            for state in latest_states
        )
        context.tokens_used += latest_tokens
        context.tokens_remaining = context.max_tokens - context.tokens_used

        # Phase 2: Get summaries (if needed)

        if context.consolidation_level != ConsolidationLevel.RAW:
            summaries = await self._get_summaries(
                latest_states=latest_states,
                consolidation_level=context.consolidation_level,
                max_tokens=budget.summaries_budget,
            )
            context.summaries = summaries

            summary_tokens = sum(
                TokenEstimate.estimate(state.content).estimated_tokens
                for state in summaries
            )
            context.tokens_used += summary_tokens
            context.tokens_remaining = context.max_tokens - context.tokens_used

        # Phase 3: Selective drill-down (if budget allows)
        if context.tokens_remaining > budget.reserved_buffer:
            detailed_states = await self._get_detailed_states(
                query=query,
                query_embedding=query_embedding,
                latest_states=latest_states,
                max_tokens=context.tokens_remaining - budget.reserved_buffer,
            )
            context.detailed_states = detailed_states

            detail_tokens = sum(
                TokenEstimate.estimate(state.content).estimated_tokens
                for state in detailed_states
            )
            context.tokens_used += detail_tokens
            context.tokens_remaining = context.max_tokens - context.tokens_used

        # Extract topics
        all_states = (
            context.latest_states + context.summaries + context.detailed_states
        )
        context.topics_considered = list(set(state.topic for state in all_states))

        return context

    def format_context_for_llm(self, context: RAGContext) -> str:
        """
        Format RAG context into LLM-readable text

        Args:
            context: RAG context to format

        Returns:
            Formatted context string
        """
        sections = []

        # Section 1: Latest States
        if context.latest_states:
            sections.append("# Current State\n")
            for state in context.latest_states:
                sections.append(self._format_state(state, include_metadata=True))

        # Section 2: Summaries
        if context.summaries:
            sections.append("\n# Historical Summaries\n")
            for state in context.summaries:
                sections.append(self._format_state(state, include_metadata=False))

        # Section 3: Detailed States
        if context.detailed_states:
            sections.append("\n# Detailed Context\n")
            for state in context.detailed_states:
                sections.append(self._format_state(state, include_metadata=True))

        # Section 4: Graph Relationships
        if context.related_graph:
            sections.append("\n# Related Connections\n")
            for rel in context.related_graph:
                sections.append(
                    f"- {rel.get('source_topic')} --[{rel.get('relationship')}]--> "
                    f"{rel.get('target_topic')} (strength: {rel.get('strength', 0):.2f})\n"
                )

        return "\n".join(sections)

    def _format_state(self, state: MemoryState, include_metadata: bool = True) -> str:
        """Format a single memory state"""
        lines = [f"## Topic: {state.topic}\n"]

        if include_metadata:
            lines.append(
                f"Version: {state.version} | "
                f"Timestamp: {state.timestamp.isoformat()} | "
                f"Confidence: {state.confidence}\n"
            )
            if state.tags:
                lines.append(f"Tags: {', '.join(state.tags)}\n")

        lines.append(f"\n{state.content}\n")

        return "".join(lines)

    def _determine_consolidation_level_rules(self, query_type: QueryType) -> ConsolidationLevel:
        """
        Determine appropriate consolidation level for query type

        Args:
            query_type: Type of query

        Returns:
            Consolidation level to use
        """
        if query_type == QueryType.STATUS:
            return ConsolidationLevel.RAW  # Just latest, no history

        elif query_type == QueryType.RECENT:
            return ConsolidationLevel.DAILY  # Daily summaries

        elif query_type == QueryType.OVERVIEW:
            return ConsolidationLevel.MONTHLY  # Monthly summaries

        elif query_type in [QueryType.WHY, QueryType.HOW]:
            return ConsolidationLevel.RAW  # Detailed, graph-filtered

        return ConsolidationLevel.WEEKLY  # Default: weekly summaries

    def _create_budget(
        self, max_tokens: int, query_type: QueryType
    ) -> ContextBudget:
        """
        Create token budget based on query type

        Args:
            max_tokens: Maximum available tokens
            query_type: Type of query

        Returns:
            Context budget allocation
        """
        # Adjust budget based on query type
        if query_type == QueryType.STATUS:
            # Status queries: prioritize latest states
            return ContextBudget(
                max_tokens=max_tokens,
                reserved_buffer=10000,
                latest_states_budget=int(max_tokens * 0.7),
                summaries_budget=0,
            )

        elif query_type == QueryType.OVERVIEW:
            # Overview queries: prioritize summaries
            return ContextBudget(
                max_tokens=max_tokens,
                reserved_buffer=10000,
                latest_states_budget=int(max_tokens * 0.2),
                summaries_budget=int(max_tokens * 0.5),
            )

        else:
            # Balanced approach for other queries
            return ContextBudget(
                max_tokens=max_tokens,
                reserved_buffer=10000,
                latest_states_budget=int(max_tokens * 0.3),
                summaries_budget=int(max_tokens * 0.3),
            )

    async def _get_latest_states(
        self,
        query: str,
        query_embedding: list[float] | None,
        max_tokens: int,
    ) -> list[MemoryState]:
        """Get latest states for query"""
        # Use retrieval service for semantic search
        results = await self.retrieval_service.semantic_search(
            query=query,
            limit=20,
            score_threshold=0.5,
            latest_only=True,
        )

        # Convert to memory states and apply token budget
        states = []
        tokens_used = 0

        for result in results:
            state_tokens = TokenEstimate.estimate(result.state.content).estimated_tokens
            if tokens_used + state_tokens > max_tokens:
                break

            states.append(result.state)
            tokens_used += state_tokens

        return states

    async def _get_summaries(
        self,
        latest_states: list[MemoryState],
        consolidation_level: ConsolidationLevel,
        max_tokens: int,
    ) -> list[MemoryState]:
        """Get consolidated summaries"""
        # For now, return empty list
        # In future, this would query for consolidated states
        # based on consolidation_level
        return []

    async def _get_detailed_states(
        self,
        query: str,
        query_embedding: list[float] | None,
        latest_states: list[MemoryState],
        max_tokens: int,
    ) -> list[MemoryState]:
        """Get detailed states via selective drill-down"""
        if not latest_states:
            return []

        # Use graph-enhanced retrieval for detailed context
        results = await self.retrieval_service.graph_enhanced_retrieval(
            query=query,
            depth=2,
            limit=10,
            include_historical=True,
            historical_steps=5,
        )

        # Convert to states and apply token budget
        states = []
        tokens_used = 0
        seen_ids = {state.id for state in latest_states}

        for result in results:
            # Skip if already included in latest states
            if result.state.id in seen_ids:
                continue

            state_tokens = TokenEstimate.estimate(result.state.content).estimated_tokens
            if tokens_used + state_tokens > max_tokens:
                break

            states.append(result.state)
            tokens_used += state_tokens
            seen_ids.add(result.state.id)

        return states
