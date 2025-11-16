"""Agent service for orchestrating intelligent query processing"""

import asyncio

from tracingrag.agents.iterative_query_agent import IterativeQueryAgent, IterativeQueryResult
from tracingrag.agents.memory_manager import MemoryManagerAgent
from tracingrag.config import settings
from tracingrag.services.llm import LLMClient, get_llm_client


class AgentService:
    """Service for agent-based query processing"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory_manager: MemoryManagerAgent | None = None,
        iterative_agent: IterativeQueryAgent | None = None,
    ):
        """
        Initialize agent service

        Args:
            llm_client: LLM client for generation
            memory_manager: Memory management agent
            iterative_agent: Iterative query agent
        """
        self._llm_client = llm_client  # Store provided client or None
        self._memory_manager = memory_manager
        self._iterative_agent = iterative_agent

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client only when needed"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def memory_manager(self) -> MemoryManagerAgent:
        """Lazy-load memory manager only when needed"""
        if self._memory_manager is None:
            self._memory_manager = MemoryManagerAgent(llm_client=self.llm_client)
        return self._memory_manager

    @property
    def iterative_agent(self) -> IterativeQueryAgent:
        """Lazy-load iterative query agent only when needed"""
        if self._iterative_agent is None:
            self._iterative_agent = IterativeQueryAgent(llm_client=self.llm_client)
        return self._iterative_agent

    async def query(
        self,
        query: str,
        max_rounds: int = 5,
        max_tokens_per_round: int = 20000,
        timeout: int | None = None,
    ) -> IterativeQueryResult:
        """
        Process query using iterative multi-round agent

        Uses embedding search + LLM-guided iteration with topic history.

        Args:
            query: User query
            max_rounds: Maximum number of iterative rounds (default 5)
            max_tokens_per_round: Max tokens per round (default 20k)
            timeout: Timeout in seconds (uses settings.agent_timeout_seconds if None)

        Returns:
            IterativeQueryResult with answer and round-by-round metadata

        Raises:
            asyncio.TimeoutError: If query exceeds timeout
        """
        timeout = timeout or settings.agent_timeout_seconds

        try:
            return await asyncio.wait_for(
                self.iterative_agent.query(
                    query=query,
                    max_rounds=max_rounds,
                    max_tokens_per_round=max_tokens_per_round,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            # Return partial result on timeout
            return IterativeQueryResult(
                query=query,
                answer=f"Query timed out after {timeout} seconds. Please try with a simpler query or increase the timeout.",
                rounds=[],
                total_rounds=0,
                source_ids=[],
                confidence=0.0,
                total_time_ms=timeout * 1000,
                metadata={"error": "timeout", "timeout_seconds": timeout},
            )
