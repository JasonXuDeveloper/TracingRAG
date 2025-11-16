"""Iterative query agent for multi-round intelligent retrieval

This agent implements a conversation-style iterative approach:
1. Initial retrieval via embedding search
2. LLM evaluates if information is sufficient
3. If not, request specific topic histories
4. Continue iterating until sufficient or max rounds reached
"""

import json
import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import LLMClient, get_llm_client
from tracingrag.services.memory import MemoryService
from tracingrag.services.retrieval import RetrievalService
from tracingrag.storage.models import MemoryStateDB
from tracingrag.types import Citation
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Pydantic Schemas for LLM Structured Outputs
# ============================================================================


class IterativeAnalysisResponse(BaseModel):
    """Single round analysis response"""

    analysis: str = Field(..., description="Current analysis result based on available information")
    sufficient: bool = Field(..., description="Whether current information is sufficient to answer")
    needed_topics: list[str] = Field(
        default_factory=list,
        description="List of topic names that need historical states (empty if sufficient)",
    )
    reasoning: str = Field(
        ..., max_length=200, description="Why these topics are needed or why sufficient"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in current analysis (0-1)"
    )


class FinalAnswerResponse(BaseModel):
    """Structured final answer from LLM"""

    answer: str = Field(..., description="Comprehensive answer to the user's query")
    key_findings: list[str] = Field(
        ...,
        description="3-5 key findings or insights (bullet points). Use empty list [] if none.",
        max_length=5,
    )
    citations: list[Citation] = Field(
        ...,
        description="Specific citations referencing source topics. Use empty list [] if none.",
    )
    uncertainties: list[str] = Field(
        ...,
        description="Any uncertainties or gaps in available information. Use empty list [] if none.",
    )


class IterativeQueryResult(BaseModel):
    """Result from iterative query processing"""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Final answer")
    key_findings: list[str] = Field(default_factory=list, description="Key findings from analysis")
    citations: list[Citation] = Field(
        default_factory=list, description="Citations referencing source topics"
    )
    uncertainties: list[str] = Field(
        default_factory=list, description="Uncertainties or information gaps"
    )
    rounds: list[dict[str, Any]] = Field(
        default_factory=list, description="Information from each round"
    )
    total_rounds: int = Field(..., description="Total rounds executed")
    source_ids: list[UUID] = Field(default_factory=list, description="Source state IDs used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence score")
    total_time_ms: float = Field(..., description="Total execution time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IterativeQueryAgent:
    """Agent for multi-round iterative query processing"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        retrieval_service: RetrievalService | None = None,
        memory_service: MemoryService | None = None,
        analysis_model: str | None = None,
        answer_model: str | None = None,
    ):
        """
        Initialize iterative query agent

        Args:
            llm_client: LLM client for generation
            retrieval_service: Retrieval service for semantic search
            memory_service: Memory service for topic history
            analysis_model: Model for iterative analysis (cheap model recommended)
            answer_model: Model for final answer generation
        """
        self.llm_client = llm_client or get_llm_client()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.memory_service = memory_service or MemoryService()
        self.analysis_model = analysis_model or settings.planner_model
        self.answer_model = answer_model or settings.default_llm_model

    async def query(
        self,
        query: str,
        max_rounds: int = 5,
        max_tokens_per_round: int = 20000,
    ) -> IterativeQueryResult:
        """
        Process query using iterative multi-round approach

        Strategy:
        1. Round 1: Embedding search → LLM evaluates sufficiency
        2. Round 2+: Add requested topic histories → Re-evaluate
        3. Continue until sufficient or max_rounds reached
        4. Generate final answer from accumulated context

        Args:
            query: User query
            max_rounds: Maximum number of rounds (default 5)
            max_tokens_per_round: Max tokens per round (default 20k)

        Returns:
            IterativeQueryResult with answer and metadata
        """
        start_time = time.time()

        # Track state across rounds
        rounds_info = []
        accumulated_states: dict[UUID, MemoryStateDB] = {}  # Dedup by ID
        needed_topics_queue: set[str] = set()  # Topics to fetch in next round
        final_analysis = ""
        final_confidence = 0.0

        logger.info(f"Starting iterative query: '{query}' (max_rounds={max_rounds})")

        for round_num in range(1, max_rounds + 1):
            round_start = time.time()

            logger.info(f"\n{'='*70}")
            logger.info(f"Round {round_num}/{max_rounds}")
            logger.info(f"{'='*70}")

            # Step 1: Retrieve states for this round
            if round_num == 1:
                # Round 1: Initial embedding search
                logger.info("Round 1: Initial embedding search for query")
                new_states = await self._initial_retrieval(
                    query=query,
                    max_tokens=max_tokens_per_round,
                )
            else:
                # Round 2+: Fetch topic histories requested by previous round
                logger.info(
                    f"Round {round_num}: Fetching {len(needed_topics_queue)} topic histories"
                )
                new_states = await self._fetch_topic_histories(
                    topics=list(needed_topics_queue),
                    max_tokens=max_tokens_per_round,
                    already_seen_ids=set(accumulated_states.keys()),
                )
                needed_topics_queue.clear()  # Clear queue after fetching

            # Step 2: Add new states to accumulated context (dedup by ID)
            states_added = 0
            for state in new_states:
                if state.id not in accumulated_states:
                    accumulated_states[state.id] = state
                    states_added += 1

            logger.info(
                f"   Retrieved {len(new_states)} states, added {states_added} new (total: {len(accumulated_states)})"
            )

            # Step 3: Build context from accumulated states (within token budget)
            context_states, context_text = self._build_context_within_budget(
                states=list(accumulated_states.values()),
                query=query,
                max_tokens=max_tokens_per_round,
            )

            logger.info(
                f"   Context: {len(context_states)} states within {max_tokens_per_round} token budget"
            )

            # Step 4: LLM analyzes current context and decides if sufficient
            # Extract available topic names from context to constrain LLM
            available_topics = list({state.topic for state in list(accumulated_states.values())})

            analysis_response = await self._llm_analyze_sufficiency(
                query=query,
                context_text=context_text,
                available_topics=available_topics,
                previous_analysis=(
                    rounds_info[-1]["analysis"] if rounds_info else None
                ),  # Include previous round's analysis
                round_num=round_num,
            )

            if analysis_response is None:
                # LLM analysis failed, fall back to simple response
                logger.warning(f"Round {round_num}: LLM analysis failed, stopping early")
                final_analysis = "Failed to analyze sufficiency due to LLM error"
                final_confidence = 0.3
                break

            # Step 5: Record round info
            round_duration = (time.time() - round_start) * 1000
            round_info = {
                "round": round_num,
                "states_retrieved": len(new_states),
                "states_added": states_added,
                "total_accumulated": len(accumulated_states),
                "context_states": len(context_states),
                "analysis": analysis_response.analysis,
                "sufficient": analysis_response.sufficient,
                "needed_topics": analysis_response.needed_topics,
                "reasoning": analysis_response.reasoning,
                "confidence": analysis_response.confidence,
                "duration_ms": round_duration,
            }
            rounds_info.append(round_info)

            logger.info(f"   Analysis: {analysis_response.analysis[:100]}...")
            logger.info(f"   Sufficient: {analysis_response.sufficient}")
            logger.info(f"   Confidence: {analysis_response.confidence:.2f}")

            # Step 6: Check termination conditions
            if analysis_response.sufficient:
                logger.info(f"✓ Information sufficient after {round_num} rounds")
                final_analysis = analysis_response.analysis
                final_confidence = analysis_response.confidence
                break
            elif round_num == max_rounds:
                logger.info(
                    f"⚠ Reached max rounds ({max_rounds}), stopping with current information"
                )
                final_analysis = analysis_response.analysis
                final_confidence = analysis_response.confidence
                break
            else:
                # Check if LLM requested any topics
                if not analysis_response.needed_topics:
                    logger.info(
                        f"✓ No more topics needed after {round_num} rounds (LLM returned empty needed_topics)"
                    )
                    final_analysis = analysis_response.analysis
                    final_confidence = analysis_response.confidence
                    break

                # Check if we retrieved anything new this round
                if round_num > 1 and len(new_states) == 0:
                    logger.info(f"✓ No new states retrieved in round {round_num}, stopping early")
                    final_analysis = analysis_response.analysis
                    final_confidence = analysis_response.confidence
                    break

                # Queue topics for next round
                needed_topics_queue.update(analysis_response.needed_topics)
                logger.info(
                    f"   → Next round will fetch {len(needed_topics_queue)} topics: {analysis_response.needed_topics[:3]}..."
                )

        # Step 7: Generate final answer using accumulated context
        logger.info(f"\n{'='*70}")
        logger.info("Generating final answer...")
        logger.info(f"{'='*70}")

        answer_response = await self._generate_final_answer(
            query=query,
            accumulated_states=list(accumulated_states.values()),
            final_analysis=final_analysis,
            max_tokens_per_round=max_tokens_per_round,
        )

        # Step 8: Build result
        total_time = (time.time() - start_time) * 1000

        result = IterativeQueryResult(
            query=query,
            answer=answer_response.answer if answer_response else "Failed to generate answer",
            key_findings=answer_response.key_findings if answer_response else [],
            citations=answer_response.citations if answer_response else [],
            uncertainties=answer_response.uncertainties if answer_response else [],
            rounds=rounds_info,
            total_rounds=len(rounds_info),
            source_ids=list(accumulated_states.keys()),
            confidence=final_confidence,
            total_time_ms=total_time,
            metadata={
                "max_rounds": max_rounds,
                "max_tokens_per_round": max_tokens_per_round,
                "total_states_used": len(accumulated_states),
            },
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"Iterative query complete: {len(rounds_info)} rounds, {total_time:.0f}ms")
        logger.info(f"{'='*70}\n")

        return result

    async def _initial_retrieval(
        self,
        query: str,
        max_tokens: int,
    ) -> list[MemoryStateDB]:
        """Initial retrieval via embedding search

        Args:
            query: User query
            max_tokens: Maximum tokens for this round

        Returns:
            List of retrieved memory states
        """
        # Get total state count for dynamic limit
        total_states = await self.memory_service.count_states(latest_only=False)
        search_limit = max(total_states, 100)

        # Semantic search with broad threshold
        results = await self.retrieval_service.semantic_search(
            query=query,
            limit=search_limit,
            score_threshold=0.3,  # Broad threshold
            latest_only=True,  # Only latest versions initially
        )

        # Convert to MemoryStateDB and apply token budget
        states = []
        tokens_used = 0

        for result in results:
            state_tokens = self._estimate_tokens(result.state.content)
            if tokens_used + state_tokens > max_tokens:
                break

            states.append(result.state)
            tokens_used += state_tokens

        return states

    async def _fetch_topic_histories(
        self,
        topics: list[str],
        max_tokens: int,
        already_seen_ids: set[UUID],
    ) -> list[MemoryStateDB]:
        """Fetch version histories for requested topics

        Args:
            topics: List of topic names
            max_tokens: Maximum tokens for this round
            already_seen_ids: IDs already in accumulated context (skip these)

        Returns:
            List of historical memory states
        """
        all_states = []
        tokens_used = 0

        # Deduplicate topics
        unique_topics = list(set(topics))

        for topic in unique_topics:
            # Fetch version history for this topic
            versions = await self.memory_service.get_topic_history(
                topic=topic,
                limit=10,  # Get up to 10 versions per topic
            )

            # Add versions in reverse chronological order (newest first)
            for version_state in versions:
                # Skip if already seen
                if version_state.id in already_seen_ids:
                    continue

                state_tokens = self._estimate_tokens(version_state.content)
                if tokens_used + state_tokens > max_tokens:
                    # Budget exhausted, return what we have
                    logger.info(
                        f"   Token budget exhausted, fetched {len(all_states)} states from {len(unique_topics)} topics"
                    )
                    return all_states

                all_states.append(version_state)
                tokens_used += state_tokens

        return all_states

    def _build_context_within_budget(
        self,
        states: list[MemoryStateDB],
        query: str,
        max_tokens: int,
    ) -> tuple[list[MemoryStateDB], str]:
        """Build formatted context within token budget

        Args:
            states: All accumulated states
            query: User query (for context)
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (states_used, formatted_context_text)
        """
        # Reserve tokens for query and instructions
        reserved = 2000
        available_for_states = max_tokens - reserved

        # Sort states by relevance (TODO: could use embedding similarity)
        # For now, use latest versions first
        sorted_states = sorted(states, key=lambda s: s.timestamp, reverse=True)

        # Build context incrementally
        context_lines = []
        states_used = []
        tokens_used = 0

        for state in sorted_states:
            state_text = f"## Topic: {state.topic} (v{state.version})\n{state.content}\n"
            state_tokens = self._estimate_tokens(state_text)

            if tokens_used + state_tokens > available_for_states:
                break

            context_lines.append(state_text)
            states_used.append(state)
            tokens_used += state_tokens

        formatted_context = "\n".join(context_lines)
        return states_used, formatted_context

    async def _llm_analyze_sufficiency(
        self,
        query: str,
        context_text: str,
        available_topics: list[str],
        previous_analysis: str | None,
        round_num: int,
    ) -> IterativeAnalysisResponse | None:
        """Use LLM to analyze if current context is sufficient

        Args:
            query: User query
            context_text: Formatted context text
            available_topics: List of topic names available in current context
            previous_analysis: Analysis from previous round (if any)
            round_num: Current round number

        Returns:
            IterativeAnalysisResponse or None if LLM call fails
        """
        # Build prompt
        previous_context = ""
        if previous_analysis:
            previous_context = f"""
**PREVIOUS ROUND ANALYSIS**:
{previous_analysis}

"""

        # Format available topics
        topics_list = "\n".join(f"  - {topic}" for topic in available_topics)

        prompt = f"""{previous_context}**USER QUERY**:
{query}

**CURRENT AVAILABLE INFORMATION** (Round {round_num}):
{context_text}

**AVAILABLE TOPICS** (these are the ONLY topics in the system):
{topics_list}

**TASK**:
Analyze whether the current information is sufficient to answer the user's query.

**INSTRUCTIONS**:
1. **analysis**: Provide your current analysis based on available information (1-3 paragraphs)
2. **sufficient**: Set to `true` if you can provide a complete answer, `false` if you need more information
3. **needed_topics**: If insufficient, list EXACT topic names from the "AVAILABLE TOPICS" list above that would help (empty list if sufficient)
4. **reasoning**: Explain why sufficient or what additional information is needed (max 200 chars)
5. **confidence**: Your confidence in the current analysis (0.0-1.0)

**CRITICAL RULES**:
- ONLY request topics from the "AVAILABLE TOPICS" list above
- Do NOT invent or guess topic names
- If no relevant topics exist in the list, set sufficient=true and work with what you have
- If you already have all versions of relevant topics, set sufficient=true

Respond with ONLY valid JSON following the schema, NO additional text.
"""

        # Calculate max_tokens
        input_tokens = self._estimate_tokens(prompt)
        max_tokens = max(2000, min(8000, 30000 - input_tokens))

        request = LLMRequest(
            system_prompt="You are an expert information analyst. Evaluate whether available information is sufficient to answer queries accurately. Be realistic about what information is needed.",
            user_message=prompt,
            context="",
            model=self.analysis_model,
            temperature=0.2,  # Lower temperature for consistent evaluation
            max_tokens=max_tokens,
            json_schema=IterativeAnalysisResponse.model_json_schema(),
            metadata={"task": "iterative_analysis", "round": round_num},
        )

        try:
            response = await self.llm_client.generate(request)

            # Parse and validate
            result = json.loads(response.content)
            validated = IterativeAnalysisResponse.model_validate(result)
            return validated

        except Exception as e:
            logger.error(f"Failed to parse iterative analysis response: {e}")
            return None

    async def _generate_final_answer(
        self,
        query: str,
        accumulated_states: list[MemoryStateDB],
        final_analysis: str,
        max_tokens_per_round: int,
    ) -> FinalAnswerResponse | None:
        """Generate structured final answer from accumulated context

        Args:
            query: User query
            accumulated_states: All accumulated states from all rounds
            final_analysis: Final analysis from last round
            max_tokens_per_round: Token budget

        Returns:
            FinalAnswerResponse or None if generation fails
        """
        # Build context (within budget)
        _, context_text = self._build_context_within_budget(
            states=accumulated_states,
            query=query,
            max_tokens=max_tokens_per_round,
        )

        # Extract topic names for citations
        available_topics = list({state.topic for state in accumulated_states})

        # Build prompt
        prompt = f"""**ANALYSIS FROM PREVIOUS ROUNDS**:
{final_analysis}

**AVAILABLE INFORMATION**:
{context_text}

**AVAILABLE TOPICS** (for citations):
{chr(10).join(f"  - {topic}" for topic in available_topics)}

**USER QUERY**:
{query}

**TASK**:
Generate a comprehensive, structured answer to the user's query.

**RESPONSE FORMAT**:
1. **answer**: A comprehensive answer synthesizing all available information (2-4 paragraphs)
2. **key_findings**: 3-5 key findings or insights as bullet points
3. **citations**: List of citations in format [{{"topic": "exact_topic_name", "insight": "what this source contributes"}}]
   - ONLY use topics from the "AVAILABLE TOPICS" list above
   - Include specific insights from each source
4. **uncertainties**: List any information gaps or uncertainties (empty list if none)

**GUIDELINES**:
- Base your answer entirely on the retrieved information
- Synthesize information from multiple sources
- Be specific about what each source contributes
- Acknowledge limitations honestly
- If information evolved over time, note the changes
- Use clear, professional language

Respond with ONLY valid JSON following the schema, NO additional text.
"""

        # Calculate max_tokens
        input_tokens = self._estimate_tokens(prompt)
        max_tokens = max(4000, min(16000, 50000 - input_tokens))

        request = LLMRequest(
            system_prompt="You are a knowledgeable assistant with access to a temporal knowledge graph. Provide clear, structured, accurate answers based on retrieved context.",
            user_message=prompt,
            context="",
            model=self.answer_model,
            temperature=0.7,
            max_tokens=max_tokens,
            json_schema=FinalAnswerResponse.model_json_schema(),
            metadata={"task": "final_answer_generation"},
        )

        try:
            response = await self.llm_client.generate(request)
            result = json.loads(response.content)
            validated = FinalAnswerResponse.model_validate(result)
            return validated
        except Exception as e:
            logger.error(f"Failed to parse final answer response: {e}")
            return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 3 characters (mixed English/Chinese)
        return len(text) // 3


# Global instance
_iterative_query_agent = None


def get_iterative_query_agent() -> IterativeQueryAgent:
    """Get or create the global iterative query agent instance"""
    global _iterative_query_agent
    if _iterative_query_agent is None:
        _iterative_query_agent = IterativeQueryAgent()
    return _iterative_query_agent
