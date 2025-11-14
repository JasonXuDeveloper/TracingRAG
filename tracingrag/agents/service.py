"""Agent service for orchestrating intelligent query processing"""

import time

from tracingrag.agents.memory_manager import MemoryManagerAgent
from tracingrag.agents.models import AgentResult, AgentState
from tracingrag.agents.query_planner import QueryPlannerAgent
from tracingrag.agents.tools import AgentTools
from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.embedding import generate_embedding
from tracingrag.services.llm import LLMClient, get_llm_client


class AgentService:
    """Service for agent-based query processing"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tools: AgentTools | None = None,
        query_planner: QueryPlannerAgent | None = None,
        memory_manager: MemoryManagerAgent | None = None,
        answer_model: str | None = None,
        max_answer_tokens: int = 16000,
    ):
        """
        Initialize agent service

        Args:
            llm_client: LLM client for generation
            tools: Agent tools
            query_planner: Query planning agent
            memory_manager: Memory management agent
            answer_model: Model for final answer generation
            max_answer_tokens: Maximum tokens for answer generation (default 16000)
        """
        self._llm_client = llm_client  # Store provided client or None
        self._query_planner = query_planner
        self._memory_manager = memory_manager
        self.tools = tools or AgentTools()
        self.answer_model = answer_model or settings.default_llm_model
        self.max_answer_tokens = max_answer_tokens

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client only when needed"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def query_planner(self) -> QueryPlannerAgent:
        """Lazy-load query planner only when needed"""
        if self._query_planner is None:
            self._query_planner = QueryPlannerAgent(
                llm_client=self.llm_client, tools=self.tools
            )
        return self._query_planner

    @property
    def memory_manager(self) -> MemoryManagerAgent:
        """Lazy-load memory manager only when needed"""
        if self._memory_manager is None:
            self._memory_manager = MemoryManagerAgent(llm_client=self.llm_client)
        return self._memory_manager

    async def query_with_agent(self, query: str, max_iterations: int = 3) -> AgentResult:
        """
        Process query using agents with planning and execution

        Args:
            query: User query
            max_iterations: Maximum replanning iterations

        Returns:
            Agent result with answer and reasoning
        """
        start_time = time.time()

        # Initialize agent state
        state = AgentState(
            query=query,
            max_iterations=max_iterations,
        )

        # Generate query embedding
        state.query_embedding = await generate_embedding(query)

        # Main agent loop
        for iteration in range(max_iterations):
            state.iteration = iteration

            # Step 1: Analyze query and create plan
            analysis, plan = await self.query_planner.analyze_and_plan(query)
            state.query_type = analysis["query_type"].value
            state.plan = plan

            # Step 2: Execute plan
            executed_steps, results = await self.query_planner.execute_plan(plan)
            state.steps.extend(executed_steps)
            state.retrieved_states.extend(results.get("retrieved_states", []))
            state.context.update(results.get("context", {}))

            # Step 3: Check if replanning needed
            should_replan, reason = await self.query_planner.should_replan(
                query, results, iteration
            )

            if not should_replan:
                break

            state.needs_replanning = True

        # Step 4: Generate final answer
        answer, confidence = await self._generate_answer(state)
        state.answer = answer
        state.confidence = confidence

        # Extract source IDs
        source_ids = self._extract_source_ids(state)
        state.sources = source_ids

        # Build result
        total_time_ms = (time.time() - start_time) * 1000

        result = AgentResult(
            query=query,
            answer=answer,
            reasoning_steps=state.steps,
            sources=source_ids,
            confidence=confidence,
            plan_used=state.plan,
            replanning_count=state.iteration,
            total_time_ms=total_time_ms,
            metadata={
                "query_type": state.query_type,
                "states_retrieved": len(state.retrieved_states),
                "steps_executed": len(state.steps),
            },
        )

        return result

    async def _generate_answer(self, state: AgentState) -> tuple[str, float]:
        """
        Generate final answer from retrieved context

        Args:
            state: Agent state with retrieved information

        Returns:
            Tuple of (answer, confidence)
        """
        # Format retrieved states as context
        context_lines = ["# Retrieved Information\n"]

        for i, item in enumerate(state.retrieved_states[:20], 1):  # Limit to top 20
            topic = item.get("topic", "Unknown")
            content = item.get("content", "")
            score = item.get("score", 0.0)

            context_lines.append(f"## Source {i}: {topic} (relevance: {score:.2f})")
            context_lines.append(f"{content}\n")

        formatted_context = "\n".join(context_lines)

        # Build prompt
        system_prompt = """You are a knowledgeable assistant with access to a temporal knowledge graph.

Your knowledge base contains information retrieved through intelligent planning and multi-step reasoning.

Guidelines:
1. Base your answer on the retrieved information
2. Synthesize information from multiple sources when relevant
3. Acknowledge if information is insufficient
4. Cite sources by their number when making claims
5. For temporal queries, note how information evolved
6. Be concise but thorough

Always provide clear, accurate answers grounded in the context."""

        # Dynamically calculate max_tokens based on input size
        # Rough estimation: 4 chars per token
        system_tokens = len(system_prompt) // 4
        query_tokens = len(state.query) // 4
        context_tokens = len(formatted_context) // 4

        # Total input tokens
        input_tokens = system_tokens + query_tokens + context_tokens

        # Calculate output tokens based on conservative context window estimate
        # Use 50K as safe upper bound - most models support at least this
        model_context_window = 50000

        # Reserve space for input and ensure reasonable output
        available_for_output = model_context_window - input_tokens - 1000  # 1K buffer

        # Use configured max or available space, whichever is smaller
        # But ensure at least 4K tokens for answer
        max_tokens = max(4000, min(self.max_answer_tokens, available_for_output))

        request = LLMRequest(
            system_prompt=system_prompt,
            user_message=state.query,
            context=formatted_context,
            model=self.answer_model,
            temperature=0.7,
            max_tokens=max_tokens,
            metadata={"task": "agent_answer_generation", "input_tokens": input_tokens},
        )

        response = await self.llm_client.generate(request)

        # Calculate confidence based on retrieval quality
        confidence = self._calculate_confidence(state)

        return response.content, confidence

    def _calculate_confidence(self, state: AgentState) -> float:
        """
        Calculate confidence in the answer

        Args:
            state: Agent state

        Returns:
            Confidence score 0-1
        """
        # Base confidence on number of sources
        source_count = len(state.retrieved_states)
        source_factor = min(1.0, source_count / 10)  # Max at 10 sources

        # Factor in retrieval scores
        scores = [item.get("score", 0.0) for item in state.retrieved_states if "score" in item]
        avg_score = sum(scores) / len(scores) if scores else 0.5

        # Factor in planning success (fewer retries = higher confidence)
        planning_factor = 1.0 - (state.iteration * 0.15)  # Penalty for replanning

        # Combined confidence
        confidence = (
            source_factor * 0.4  # 40% from source count
            + avg_score * 0.4  # 40% from retrieval quality
            + planning_factor * 0.2  # 20% from planning success
        )

        return min(1.0, max(0.0, confidence))

    def _extract_source_ids(self, state: AgentState) -> list:
        """
        Extract unique source IDs from state

        Args:
            state: Agent state

        Returns:
            List of source UUIDs
        """
        from uuid import UUID

        ids = set()
        for item in state.retrieved_states:
            if "id" in item:
                try:
                    ids.add(UUID(item["id"]))
                except (ValueError, TypeError):
                    pass

        return list(ids)


# Convenience function
async def query_with_agent(query: str, max_iterations: int = 3) -> AgentResult:
    """
    Convenience function for agent-based query processing

    Args:
        query: User query
        max_iterations: Maximum replanning iterations

    Returns:
        Agent result
    """
    service = AgentService()
    return await service.query_with_agent(query, max_iterations)
