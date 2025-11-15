"""RAG service orchestrating the complete retrieval-augmented generation pipeline"""

import asyncio
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from tracingrag.config import settings
from tracingrag.core.models.rag import (
    LLMRequest,
    RAGContext,
    RAGResponse,
)
from tracingrag.services.context import ContextBuilder
from tracingrag.services.embedding import generate_embedding
from tracingrag.services.llm import LLMClient, get_llm_client
from tracingrag.services.retrieval import RetrievalService
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """Service orchestrating the complete RAG pipeline"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        context_builder: ContextBuilder | None = None,
        retrieval_service: RetrievalService | None = None,
        max_context_tokens: int = 100000,
        default_system_prompt: str | None = None,
    ):
        """
        Initialize RAG service

        Args:
            llm_client: LLM client for generation
            context_builder: Context builder for assembling retrieval results
            retrieval_service: Service for retrieving memory states
            max_context_tokens: Maximum tokens for context window
            default_system_prompt: Default system prompt for LLM
        """
        self._llm_client = llm_client  # Store provided client or None
        self.retrieval_service = retrieval_service or RetrievalService()
        self.context_builder = context_builder or ContextBuilder(self.retrieval_service)
        self.max_context_tokens = max_context_tokens

        self.default_system_prompt = default_system_prompt or self._get_default_system_prompt()

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client only when needed"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def query(
        self,
        query: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        metadata: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Execute complete RAG pipeline: retrieve, contextualize, generate

        Args:
            query: User query
            system_prompt: Custom system prompt (uses default if None)
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            metadata: Additional metadata

        Returns:
            RAG response with answer and metadata
        """
        start_time = time.time()

        # Phase 1: Retrieval - Get relevant context
        retrieval_start = time.time()

        # Generate query embedding for semantic search
        query_embedding = await generate_embedding(query)

        # Build context with intelligent budgeting
        rag_context = await self.context_builder.build_context(
            query=query,
            query_embedding=query_embedding,
            max_tokens=self.max_context_tokens,
        )

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # Phase 2: Format context for LLM
        formatted_context = self.context_builder.format_context_for_llm(rag_context)

        # Phase 3: Generation - Generate response with LLM
        generation_start = time.time()

        llm_request = LLMRequest(
            system_prompt=system_prompt or self.default_system_prompt,
            user_message=query,
            context=formatted_context,
            model=model or "anthropic/claude-3.5-sonnet",
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {},
        )

        llm_response = await self.llm_client.generate(llm_request)

        generation_time_ms = (time.time() - generation_start) * 1000

        # Phase 4: Package response
        total_time_ms = (time.time() - start_time) * 1000

        # Extract source IDs
        source_ids = self._extract_source_ids(rag_context)

        response = RAGResponse(
            query=query,
            answer=llm_response.content,
            sources=source_ids,
            confidence=self._calculate_confidence(rag_context, llm_response),
            context_used={
                "latest_states_count": len(rag_context.latest_states),
                "summaries_count": len(rag_context.summaries),
                "detailed_states_count": len(rag_context.detailed_states),
                "topics": rag_context.topics_considered,
                "consolidation_level": rag_context.consolidation_level.value,
            },
            query_type=rag_context.query_type,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            tokens_retrieved=rag_context.tokens_used,
            tokens_generated=llm_response.tokens_used,
            timestamp=datetime.utcnow(),
            metadata={
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "llm_metadata": llm_response.metadata,
            },
        )

        return response

    async def query_with_context(
        self,
        query: str,
        rag_context: RAGContext,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> RAGResponse:
        """
        Execute RAG pipeline with pre-built context

        Useful when you want to customize context building

        Args:
            query: User query
            rag_context: Pre-built RAG context
            system_prompt: Custom system prompt
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for generation

        Returns:
            RAG response with answer and metadata
        """
        start_time = time.time()

        # Format context
        formatted_context = self.context_builder.format_context_for_llm(rag_context)

        # Generate with LLM
        generation_start = time.time()

        llm_request = LLMRequest(
            system_prompt=system_prompt or self.default_system_prompt,
            user_message=query,
            context=formatted_context,
            model=model or "anthropic/claude-3.5-sonnet",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_response = await self.llm_client.generate(llm_request)

        generation_time_ms = (time.time() - generation_start) * 1000
        total_time_ms = (time.time() - start_time) * 1000

        # Package response
        source_ids = self._extract_source_ids(rag_context)

        response = RAGResponse(
            query=query,
            answer=llm_response.content,
            sources=source_ids,
            confidence=self._calculate_confidence(rag_context, llm_response),
            context_used={
                "latest_states_count": len(rag_context.latest_states),
                "summaries_count": len(rag_context.summaries),
                "detailed_states_count": len(rag_context.detailed_states),
                "topics": rag_context.topics_considered,
            },
            query_type=rag_context.query_type,
            retrieval_time_ms=0.0,  # Context was pre-built
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            tokens_retrieved=rag_context.tokens_used,
            tokens_generated=llm_response.tokens_used,
            timestamp=datetime.utcnow(),
            metadata={
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
            },
        )

        return response

    async def query_iterative(
        self,
        query: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 50000,
        metadata: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Execute RAG pipeline with iterative multi-round processing for large contexts

        Strategy:
        - Retrieve ALL relevant states (is_latest=True, full search)
        - Round 1: Query + first batch of references (fit in 20k tokens)
        - Round 2+: Query + previous answer summary + next batch of references
        - Each round accumulates answer with globally numbered citations
        - Final answer synthesizes all rounds

        Args:
            query: User query
            system_prompt: Custom system prompt (uses default if None)
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for ENTIRE generation (default 50k for multi-round)
            metadata: Additional metadata

        Returns:
            RAG response with synthesized answer and all sources
        """
        from tracingrag.core.models.memory import MemoryState

        start_time = time.time()
        MAX_TOKENS_PER_ROUND = 20000  # Target token count per round for references

        # Phase 1: Retrieval - Get ALL relevant context
        retrieval_start = time.time()

        # Full semantic search (no limit, get everything relevant)
        retrieval_results = await self.retrieval_service.semantic_search(
            query=query,
            limit=1000,  # Large limit to get comprehensive results
            score_threshold=0.3,  # Lower threshold for completeness
            latest_only=True,  # Only latest versions
        )

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # Extract all states
        all_states: list[MemoryState] = [result.state for result in retrieval_results]
        total_states = len(all_states)

        if total_states == 0:
            # No relevant context found, return basic response
            return RAGResponse(
                query=query,
                answer="I don't have sufficient context to answer this question.",
                sources=[],
                confidence=0.0,
                context_used={},
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0.0,
                total_time_ms=(time.time() - start_time) * 1000,
                tokens_retrieved=0,
                tokens_generated=0,
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
            )

        # Phase 2: Iterative multi-round generation
        generation_start = time.time()

        # Calculate how many states fit in each round
        state_idx = 0
        round_answers = []
        round_num = 1
        total_tokens_generated = 0

        while state_idx < total_states:
            # Determine states for this round
            available_tokens = MAX_TOKENS_PER_ROUND
            round_states = []

            while state_idx < total_states and available_tokens > 0:
                state = all_states[state_idx]
                state_tokens = self.context_builder.estimate_state_tokens(state)

                if state_tokens > available_tokens and round_states:
                    # This state won't fit, start next round
                    break

                round_states.append(state)
                available_tokens -= state_tokens
                state_idx += 1

            if not round_states:
                # Edge case: single state too large, take it anyway
                round_states = [all_states[state_idx]]
                state_idx += 1

            start_ref = state_idx - len(round_states) + 1  # 1-based
            end_ref = state_idx

            # Format batch with global reference numbers
            formatted_batch = self.context_builder.format_states_batch_for_llm(
                states=round_states,
                start_index=start_ref,
                total_count=total_states,
                include_metadata=True,
            )

            # Build prompt for this round
            if round_num == 1:
                # Round 1: Initial query
                round_prompt = f"""**QUERY**: {query}

**AVAILABLE REFERENCES** (Total: {total_states}):
{formatted_batch}

**INSTRUCTIONS**:
1. Answer the query using ONLY the provided references
2. Cite references using their numbers: [1], [2], etc.
3. This is Round {round_num} of multiple rounds
4. More references may be provided in subsequent rounds
5. Provide a partial answer based on refs {start_ref}-{end_ref}

Answer:"""
            else:
                # Round 2+: Continue with previous context (FULL CONTENT, NO TRUNCATION)
                previous_summary = "\n\n".join(
                    f"**Round {i+1} Answer**:\n{ans}" for i, ans in enumerate(round_answers)
                )

                round_prompt = f"""**QUERY**: {query}

**PREVIOUS ROUNDS** ({round_num-1} rounds completed):
{previous_summary}

**ADDITIONAL REFERENCES**:
{formatted_batch}

**INSTRUCTIONS**:
1. Continue answering the query, incorporating refs {start_ref}-{end_ref}
2. Cite using global numbers: [{start_ref}], [{start_ref+1}], ...
3. Synthesize with previous rounds
4. This is Round {round_num}

Updated Answer:"""

            # Call LLM for this round
            llm_request = LLMRequest(
                system_prompt=system_prompt or self.default_system_prompt,
                user_message=round_prompt,
                context="",
                model=model or settings.default_llm_model,
                temperature=temperature,
                max_tokens=min(8000, max_tokens - total_tokens_generated),
            )

            llm_response = await self.llm_client.generate(llm_request)
            round_answers.append(llm_response.content)
            total_tokens_generated += llm_response.tokens_used

            round_num += 1

        generation_time_ms = (time.time() - generation_start) * 1000

        # Phase 3: Final synthesis (if multiple rounds)
        if len(round_answers) > 1:
            # Build round-by-round answers (avoid backslash in f-string)
            newline = "\n"
            round_by_round = chr(10).join(
                f"Round {i+1}:{newline}{ans}{newline}" for i, ans in enumerate(round_answers)
            )

            synthesis_prompt = f"""**ORIGINAL QUERY**: {query}

**ROUND-BY-ROUND ANSWERS**:
{round_by_round}

**INSTRUCTIONS**:
Synthesize a single, coherent answer that:
1. Combines insights from all rounds
2. Maintains all citation numbers ([1], [2], etc.)
3. Removes redundancy
4. Flows naturally

Final synthesized answer:"""

            synthesis_request = LLMRequest(
                system_prompt=system_prompt or self.default_system_prompt,
                user_message=synthesis_prompt,
                context="",
                model=model or settings.default_llm_model,
                temperature=temperature,
                max_tokens=min(8000, max_tokens - total_tokens_generated),
            )

            synthesis_response = await self.llm_client.generate(synthesis_request)
            final_answer = synthesis_response.content
            total_tokens_generated += synthesis_response.tokens_used
        else:
            final_answer = round_answers[0]

        # Phase 4: Package response
        total_time_ms = (time.time() - start_time) * 1000

        source_ids = [state.id for state in all_states]

        response = RAGResponse(
            query=query,
            answer=final_answer,
            sources=source_ids,
            confidence=self._calculate_confidence_from_states(all_states),
            context_used={
                "total_states": total_states,
                "rounds": len(round_answers),
                "iterative_mode": True,
            },
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            tokens_retrieved=sum(self.context_builder.estimate_state_tokens(s) for s in all_states),
            tokens_generated=total_tokens_generated,
            timestamp=datetime.utcnow(),
            metadata={
                "model": model or settings.default_llm_model,
                "rounds": len(round_answers),
                **(metadata or {}),
            },
        )

        return response

    async def query_iterative_parallel(
        self,
        query: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 40000,
        metadata: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Execute RAG pipeline with parallel single-layer Map-Reduce processing

        Strategy (Optimized Single-Layer MapReduce):
        - Map Phase: Partition states → parallel LLM calls → k detailed analyses with global refs
        - Reduce Phase: Single comprehensive merge → final answer (NO multi-layer reduce)

        Key features:
        - Parallel map processing for efficiency
        - Single-layer reduce to prevent information loss from multi-tier summarization
        - Global reference numbering maintained throughout
        - Comprehensive final synthesis (6-15 paragraphs, max 80k tokens)
        - Map round adds citations, reduce preserves all existing ones
        - Full content retrieval (no truncation)

        Args:
            query: User query
            system_prompt: Custom system prompt
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens per map LLM call (default 40k, reduce uses 80k)
            metadata: Additional metadata

        Returns:
            RAG response with comprehensive answer covering all important points and all sources
        """
        from tracingrag.core.models.memory import MemoryState

        start_time = time.time()
        MAX_TOKENS_PER_PARTITION = 20000

        # Phase 1: Retrieval - Get ALL relevant context
        retrieval_start = time.time()

        # Get total active states for comprehensive search
        from tracingrag.services.memory import MemoryService

        memory_service = MemoryService()
        total_active_states = await memory_service.count_states(storage_tier="active")
        search_limit = max(total_active_states, 1000)

        retrieval_results = await self.retrieval_service.semantic_search(
            query=query,
            limit=search_limit,  # Use all active states as limit
            score_threshold=0.3,
            latest_only=True,
        )

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        all_states: list[MemoryState] = [result.state for result in retrieval_results]
        total_states = len(all_states)

        if total_states == 0:
            return RAGResponse(
                query=query,
                answer="I don't have sufficient context to answer this question.",
                sources=[],
                confidence=0.0,
                context_used={},
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0.0,
                total_time_ms=(time.time() - start_time) * 1000,
                tokens_retrieved=0,
                tokens_generated=0,
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
            )

        # Phase 2: Map phase - Partition states and process in parallel
        generation_start = time.time()

        # Partition states into chunks
        state_partitions = self._partition_states(all_states, MAX_TOKENS_PER_PARTITION)
        total_partitions = len(state_partitions)

        # Process all partitions in parallel
        map_tasks = []
        global_idx = 1

        for partition_idx, partition in enumerate(state_partitions):
            start_ref = global_idx
            end_ref = global_idx + len(partition) - 1

            # Use generous max_tokens for query analysis
            partition_max_tokens = 50000  # Generous limit for detailed analysis

            task = self._process_states_partition(
                query=query,
                partition=partition,
                partition_num=partition_idx + 1,
                total_partitions=total_partitions,
                start_ref=start_ref,
                end_ref=end_ref,
                total_states=total_states,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=partition_max_tokens,
            )
            map_tasks.append(task)
            global_idx = end_ref + 1

        # Execute map phase in parallel
        map_results = await asyncio.gather(*map_tasks)
        total_tokens_generated = sum(r["tokens"] for r in map_results)
        map_outputs = [r["content"] for r in map_results]

        # Phase 3: Single-layer Reduce - Merge ALL map outputs in one step
        # Strategy: Concatenate all map outputs and let LLM synthesize comprehensively
        # This avoids information loss from multi-layer reduce

        if len(map_outputs) == 1:
            # Only one partition, no need to merge
            final_answer = map_outputs[0]
        else:
            # Merge all map outputs in a single LLM call (no recursive reduce)
            logger.info(
                f"   Merging {len(map_outputs)} map outputs into final answer (single-layer reduce)"
            )

            merge_result = await self._merge_all_results(
                query=query,
                all_results=map_outputs,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=80000,  # Very generous limit for comprehensive final synthesis
            )
            total_tokens_generated += merge_result["tokens"]
            final_answer = merge_result["content"]
        generation_time_ms = (time.time() - generation_start) * 1000

        # Phase 4: Package response
        total_time_ms = (time.time() - start_time) * 1000
        source_ids = [state.id for state in all_states]

        response = RAGResponse(
            query=query,
            answer=final_answer,
            sources=source_ids,
            confidence=self._calculate_confidence_from_states(all_states),
            context_used={
                "total_states": total_states,
                "map_partitions": total_partitions,
                "reduce_strategy": "single_layer",  # Changed from multi-layer to single-layer
                "parallel_mode": True,
            },
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            tokens_retrieved=sum(self.context_builder.estimate_state_tokens(s) for s in all_states),
            tokens_generated=total_tokens_generated,
            timestamp=datetime.utcnow(),
            metadata={
                "model": model or settings.default_llm_model,
                "map_partitions": total_partitions,
                "reduce_strategy": "single_layer",
                **(metadata or {}),
            },
        )

        return response

    def _partition_states(
        self, states: list[Any], max_tokens_per_partition: int
    ) -> list[list[Any]]:
        """
        Partition states into chunks, each ≤ max_tokens

        Args:
            states: List of memory states
            max_tokens_per_partition: Maximum tokens per partition

        Returns:
            List of state partitions
        """
        partitions = []
        current_partition = []
        current_tokens = 0

        for state in states:
            state_tokens = self.context_builder.estimate_state_tokens(state)

            if current_tokens + state_tokens > max_tokens_per_partition and current_partition:
                # Current partition is full, start new one
                partitions.append(current_partition)
                current_partition = [state]
                current_tokens = state_tokens
            else:
                current_partition.append(state)
                current_tokens += state_tokens

        if current_partition:
            partitions.append(current_partition)

        return partitions

    async def _process_states_partition(
        self,
        query: str,
        partition: list[Any],
        partition_num: int,
        total_partitions: int,
        start_ref: int,
        end_ref: int,
        total_states: int,
        system_prompt: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        Process a partition of states (Map phase)

        Args:
            query: User query
            partition: States in this partition
            partition_num: Current partition number (1-indexed)
            total_partitions: Total number of partitions
            start_ref: Starting global reference number
            end_ref: Ending global reference number
            total_states: Total number of states across all partitions
            system_prompt: System prompt
            model: LLM model
            temperature: Temperature
            max_tokens: Max tokens for generation

        Returns:
            Dict with "content" and "tokens" keys
        """
        # Format batch with global reference numbers
        formatted_batch = self.context_builder.format_states_batch_for_llm(
            states=partition,
            start_index=start_ref,
            total_count=total_states,
            include_metadata=True,
        )

        # Build prompt for Map phase
        prompt = f"""**QUERY**: {query}

**YOUR TASK**:
Analyze partition {partition_num}/{total_partitions} (references [{start_ref}]-[{end_ref}] of {total_states} total).

**REFERENCES**:
{formatted_batch}

**INSTRUCTIONS**:
1. Answer using ONLY the provided references
2. Cite using GLOBAL numbers: [{start_ref}], [{start_ref+1}], etc.
3. Provide CONCISE but COMPLETE analysis (2-4 paragraphs)
4. Focus on KEY insights (will be merged later)
5. For each relevant reference, explain HOW it relates to the query

Answer (2-4 paragraphs):"""

        # Call LLM
        llm_request = LLMRequest(
            system_prompt=system_prompt or self.default_system_prompt,
            user_message=prompt,
            context="",
            model=model or settings.default_llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_response = await self.llm_client.generate(llm_request)

        return {
            "content": llm_response.content,
            "tokens": llm_response.tokens_used,
        }

    async def _merge_all_results(
        self,
        query: str,
        all_results: list[str],
        system_prompt: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        Merge ALL map results in a single comprehensive synthesis (Single-layer Reduce)

        This method replaces recursive multi-layer reduce to avoid information loss.
        All partition analyses are merged in one LLM call with generous token limit.

        Args:
            query: Original user query
            all_results: ALL map outputs to merge
            system_prompt: System prompt
            model: LLM model
            temperature: Temperature
            max_tokens: Max tokens for generation (very generous for comprehensive output)

        Returns:
            Dict with "content" and "tokens" keys
        """
        # Combine all results
        combined_results = "\n\n---\n\n".join(
            f"**Partition {i+1} Analysis**:\n{result}" for i, result in enumerate(all_results)
        )

        # Build comprehensive merge prompt
        prompt = f"""**ORIGINAL QUERY**: {query}

**YOUR TASK**:
Synthesize ALL {len(all_results)} partition analyses into ONE comprehensive final answer.

**ALL PARTITION ANALYSES**:
{combined_results}

**CRITICAL REQUIREMENTS FOR COMPREHENSIVE SYNTHESIS**:

⚠️ **INFORMATION COMPLETENESS (MANDATORY)**:
   - Include ALL key points, facts, and insights from every partition
   - DO NOT omit or skip any important information from any partition
   - Each partition may contain unique details - preserve them all
   - If partitions cover different aspects, include all aspects comprehensively

⚠️ **NO INFORMATION LOSS (MANDATORY)**:
   - This is a SINGLE-LAYER merge - there will be NO further reduce rounds
   - You must capture everything important in this final answer
   - Be thorough and detailed - do not over-summarize
   - Length: As many paragraphs as needed to cover all important points (typically 6-15 paragraphs)

⚠️ **CITATION PRESERVATION**:
   - PRESERVE ALL existing citations exactly as they appear (e.g., [1], [2], [45])
   - DO NOT add new citations - only use existing ones from the analyses
   - Maintain citation accuracy when combining information

⚠️ **ORGANIZATION**:
   - Structure logically with clear sections if multiple aspects are covered
   - Remove redundancy while keeping all unique information
   - Ensure natural flow and coherent narrative
   - Maintain factual accuracy - no hallucinations or inferences

Comprehensive final answer (6-15 paragraphs, covering ALL important points):"""

        # Call LLM with very generous max_tokens
        llm_request = LLMRequest(
            system_prompt=system_prompt or self.default_system_prompt,
            user_message=prompt,
            context="",
            model=model or settings.default_llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_response = await self.llm_client.generate(llm_request)

        return {
            "content": llm_response.content,
            "tokens": llm_response.tokens_used,
        }

    def _calculate_confidence_from_states(self, states: list[Any]) -> float:
        """Calculate confidence from list of states"""
        if not states:
            return 0.5

        avg_confidence = sum(s.confidence for s in states) / len(states)
        count_factor = min(1.0, len(states) / 10)
        return avg_confidence * 0.7 + count_factor * 0.3

    def _extract_source_ids(self, rag_context: RAGContext) -> list[UUID]:
        """Extract unique source IDs from context"""
        source_ids = set()

        for state in rag_context.latest_states:
            source_ids.add(state.id)

        for state in rag_context.summaries:
            source_ids.add(state.id)

        for state in rag_context.detailed_states:
            source_ids.add(state.id)

        return list(source_ids)

    def _calculate_confidence(self, rag_context: RAGContext, llm_response: Any) -> float:
        """
        Calculate confidence score for the response

        Factors:
        - Number of sources retrieved
        - Quality of sources (confidence scores)
        - Token usage vs available budget
        """
        # Base confidence from retrieved states
        all_states = rag_context.latest_states + rag_context.summaries + rag_context.detailed_states

        if not all_states:
            return 0.5  # Default if no states

        # Average confidence of source states
        avg_state_confidence = sum(state.confidence for state in all_states) / len(all_states)

        # Coverage factor: how much of available budget was used
        coverage_factor = min(1.0, rag_context.tokens_used / (rag_context.max_tokens * 0.5))

        # Source count factor: more sources generally better
        source_count_factor = min(1.0, len(all_states) / 10)

        # Combined confidence (weighted average)
        confidence = (
            avg_state_confidence * 0.5  # 50% from state quality
            + coverage_factor * 0.3  # 30% from coverage
            + source_count_factor * 0.2  # 20% from source count
        )

        return min(1.0, max(0.0, confidence))

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """You are a knowledgeable assistant with access to a temporal knowledge graph.

Your knowledge base contains:
- **Current State**: Latest information on various topics
- **Historical Context**: How information evolved over time
- **Relationships**: Connections between different concepts and topics

Guidelines for responses:
1. Base your answers on the provided context
2. If information evolved over time, explain the current state and mention key changes
3. Cite specific topics/states when relevant
4. If context is insufficient, acknowledge limitations rather than speculating
5. For "why" questions, explain causal relationships from the graph
6. For temporal questions, trace the evolution through versions

Remember: The context represents verified knowledge with temporal tracking and relationships.
Use this to provide accurate, well-reasoned answers."""


# Convenience function for quick queries
async def query_rag(
    query: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> RAGResponse:
    """
    Convenience function for quick RAG queries

    Args:
        query: User query
        model: LLM model to use
        temperature: Sampling temperature
        max_tokens: Max tokens for generation

    Returns:
        RAG response
    """
    rag_service = RAGService()
    return await rag_service.query(
        query=query,
        model=model or settings.default_llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
