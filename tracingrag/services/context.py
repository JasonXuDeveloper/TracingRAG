"""Context building for RAG pipeline"""


from tracingrag.core.models.memory import MemoryState
from tracingrag.core.models.rag import (
    ConsolidationLevel,
    ContextBudget,
    QueryType,
    RAGContext,
    TokenEstimate,
)
from tracingrag.services.memory import MemoryService
from tracingrag.services.query_analyzer import QueryAnalyzer, get_query_analyzer
from tracingrag.services.retrieval import RetrievalService


class ContextBuilder:
    """Builds context for RAG pipeline with intelligent budgeting"""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        memory_service: MemoryService | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        use_llm_analysis: bool = True,
    ):
        """
        Initialize context builder

        Args:
            retrieval_service: Service for retrieving memory states
            memory_service: Service for memory operations
            query_analyzer: Query analyzer for intent detection
            use_llm_analysis: Whether to use LLM for query analysis (vs rules)
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.memory_service = memory_service or MemoryService()
        self._query_analyzer = query_analyzer  # Store provided analyzer or None
        self.use_llm_analysis = use_llm_analysis

    @property
    def query_analyzer(self) -> QueryAnalyzer:
        """Lazy-load query analyzer only when needed"""
        if self._query_analyzer is None:
            self._query_analyzer = get_query_analyzer()
        return self._query_analyzer

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
            analysis = await self.query_analyzer.analyze_query(query, use_llm=self.use_llm_analysis)
            query_type = analysis["query_type"]
            consolidation_level = analysis["consolidation_level"]
        else:
            # Use provided query type
            consolidation_level = self._determine_consolidation_level_rules(query_type)

        logger.info(f"Building context for query_type={query_type.value}, max_tokens={max_tokens}")

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
        logger.info(f"Budget: latest={budget.latest_states_budget}, summaries={budget.summaries_budget}, buffer={budget.reserved_buffer}")

        # Phase 1: Get latest states (always included)
        latest_states = await self._get_latest_states(
            query=query,
            query_embedding=query_embedding,
            max_tokens=budget.latest_states_budget,
        )
        context.latest_states = latest_states

        # Update token counts
        latest_tokens = sum(
            TokenEstimate.estimate(state.content).estimated_tokens for state in latest_states
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
                TokenEstimate.estimate(state.content).estimated_tokens for state in summaries
            )
            context.tokens_used += summary_tokens
            context.tokens_remaining = context.max_tokens - context.tokens_used

        # Phase 3: Selective drill-down (if budget allows)
        logger.info(f"Phase 3 check: tokens_remaining={context.tokens_remaining}, buffer={budget.reserved_buffer}")
        if context.tokens_remaining > budget.reserved_buffer:
            logger.info("Entering Phase 3: selective drill-down with graph traversal")
            detailed_states = await self._get_detailed_states(
                query=query,
                query_embedding=query_embedding,
                latest_states=latest_states,
                max_tokens=context.tokens_remaining - budget.reserved_buffer,
            )
            context.detailed_states = detailed_states
            logger.info(f"Phase 3 completed: retrieved {len(detailed_states)} detailed states")

            detail_tokens = sum(
                TokenEstimate.estimate(state.content).estimated_tokens for state in detailed_states
            )
            context.tokens_used += detail_tokens
            context.tokens_remaining = context.max_tokens - context.tokens_used
        else:
            logger.info("Skipping Phase 3: insufficient token budget")
        # Phase 4: Historical context (for queries needing evolution tracking)
        if query_type in [QueryType.RECENT, QueryType.WHY, QueryType.HOW, QueryType.OVERVIEW]:
            logger.info(f"Phase 4: query type {query_type.value} needs historical context")
            if context.tokens_remaining > budget.reserved_buffer:
                logger.info("Fetching historical context...")
                history_states = await self._get_historical_context(
                    latest_states=latest_states,
                    max_tokens=min(context.tokens_remaining - budget.reserved_buffer, 2000),
                )
                # Add to detailed states
                context.detailed_states.extend(history_states)
                logger.info(f"Phase 4 completed: added {len(history_states)} historical states")

                history_tokens = sum(
                    TokenEstimate.estimate(state.content).estimated_tokens
                    for state in history_states
                )
                context.tokens_used += history_tokens
                context.tokens_remaining = context.max_tokens - context.tokens_used
            else:
                logger.info("Skipping Phase 4: insufficient token budget")
        else:
            logger.info(f"Skipping Phase 4: query type {query_type.value} doesn't need history")

        # Extract topics
        all_states = context.latest_states + context.summaries + context.detailed_states
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

    async def _get_historical_context(
        self,
        latest_states: list[MemoryState],
        max_tokens: int,
    ) -> list[MemoryState]:
        """Get historical context for topics in latest states

        Args:
            latest_states: Current latest states
            max_tokens: Maximum tokens to use

        Returns:
            List of historical memory states
        """
        if not latest_states:
            return []

        # Get unique topics from latest states
        topics = list(set(state.topic for state in latest_states))

        # Limit to top 3 most relevant topics to avoid token overflow
        topics = topics[:3]

        history_states = []
        tokens_used = 0
        seen_ids = {state.id for state in latest_states}

        for topic in topics:
            # Get version history for this topic (limit to 3-5 versions per topic)
            versions = await self.memory_service.get_topic_history(
                topic=topic,
                limit=5,
            )

            # Skip the latest version (already in latest_states)
            for version_state in versions[1:]:
                # Skip if already seen
                if version_state.id in seen_ids:
                    continue

                state_tokens = TokenEstimate.estimate(version_state.content).estimated_tokens
                if tokens_used + state_tokens > max_tokens:
                    # Budget exhausted
                    return history_states

                # Convert DB model to MemoryState
                history_state = MemoryState(
                    id=version_state.id,
                    topic=version_state.topic,
                    content=version_state.content,
                    version=version_state.version,
                    timestamp=version_state.timestamp,
                    confidence=version_state.confidence,
                    source=version_state.source,
                )

                history_states.append(history_state)
                tokens_used += state_tokens
                seen_ids.add(version_state.id)

        return history_states

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

    def _create_budget(self, max_tokens: int, query_type: QueryType) -> ContextBudget:
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
        logger.info(f"_get_latest_states: query='{query[:50]}...', max_tokens={max_tokens}")
        # Get total states count from MemoryService (unified method)
        total_states = await self.memory_service.count_states(latest_only=False)

        # Use dynamic limit based on actual database size (minimum 100)
        search_limit = max(total_states, 100)
        logger.info(f"_get_latest_states: database has {total_states} states, using limit={search_limit}")
        # Use retrieval service for semantic search
        # Lower threshold for broad queries (0.3 allows more relevant results)
        results = await self.retrieval_service.semantic_search(
            query=query,
            limit=search_limit,
            score_threshold=0.3,
            latest_only=True,
        )

        logger.info(f"_get_latest_states: semantic_search returned {len(results)} results")
        # Convert to memory states and apply token budget
        states = []
        tokens_used = 0

        for result in results:
            state_tokens = TokenEstimate.estimate(result.state.content).estimated_tokens
            if tokens_used + state_tokens > max_tokens:
                logger.info(f"_get_latest_states: token budget exceeded, stopping at {len(states)} states")
                break

            states.append(result.state)
            tokens_used += state_tokens

        logger.info(f"_get_latest_states: returning {len(states)} states, tokens_used={tokens_used}")
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
            logger.info("_get_detailed_states: no latest_states, returning empty")
            return []

        # Get total state count from database for dynamic limit
        total_states = await self.memory_service.count_states(latest_only=False)
        # Use all states, no artificial limit
        retrieval_limit = total_states if total_states > 0 else 1000

        logger.info(f"_get_detailed_states: calling graph_enhanced_retrieval with depth=2, limit={retrieval_limit} (dynamic based on DB count)")
        # Use graph-enhanced retrieval for detailed context
        results = await self.retrieval_service.graph_enhanced_retrieval(
            query=query,
            depth=2,
            limit=retrieval_limit,
            include_historical=True,
            historical_steps=5,
        )
        logger.info(f"_get_detailed_states: got {len(results)} results from graph_enhanced_retrieval")

        # Convert to states and apply token budget
        states = []
        tokens_used = 0
        seen_ids = {state.id for state in latest_states}

        for result in results:
            logger.info(f"Processing result: topic={result.state.topic}, id={result.state.id}, related_states={len(result.related_states) if result.related_states else 0}")

            # Skip if already included in latest states
            if result.state.id in seen_ids:
                logger.info(f"Skipping {result.state.topic} (already in latest_states)")
                # Still process related states even if main state is skipped
                if result.related_states:
                    logger.info(f"But still checking {len(result.related_states)} related states...")
                    # Jump to related states processing
                else:
                    continue
            else:
                state_tokens = TokenEstimate.estimate(result.state.content).estimated_tokens
                if tokens_used + state_tokens > max_tokens:
                    break

                states.append(result.state)
                tokens_used += state_tokens
                seen_ids.add(result.state.id)

            # Also include related states from graph traversal
            if result.related_states:
                logger.info(f"Found {len(result.related_states)} related states for {result.state.topic}")
                for related_info in result.related_states:
                    # Extract state_id from Neo4j node
                    related_id_str = related_info.get("memory", {}).get("id")
                    if not related_id_str:
                        continue

                    try:
                        from uuid import UUID

                        related_id = UUID(related_id_str)

                        # Skip if already seen
                        if related_id in seen_ids:
                            continue

                        # Fetch full state from database
                        related_state_db = await self.memory_service.get_memory_state(related_id)
                        if not related_state_db:
                            continue

                        # Check token budget
                        related_tokens = TokenEstimate.estimate(
                            related_state_db.content
                        ).estimated_tokens
                        if tokens_used + related_tokens > max_tokens:
                            # Budget exhausted
                            return states

                        # Convert to MemoryState
                        related_state = MemoryState(
                            id=related_state_db.id,
                            topic=related_state_db.topic,
                            content=related_state_db.content,
                            version=related_state_db.version,
                            timestamp=related_state_db.timestamp,
                            confidence=related_state_db.confidence,
                            source=related_state_db.source,
                        )

                        states.append(related_state)
                        tokens_used += related_tokens
                        seen_ids.add(related_id)
                        logger.info(f"Added related state: {related_state.topic} (depth={related_info.get('depth', '?')})")

                    except Exception as e:
                        # Skip invalid IDs
                        logger.error(f"Failed to fetch related state: {e}")
                        continue

        return states
