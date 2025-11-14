"""RAG service orchestrating the complete retrieval-augmented generation pipeline"""

import time
from datetime import datetime
from typing import Any
from uuid import UUID

from tracingrag.core.models.rag import (
    LLMRequest,
    RAGContext,
    RAGResponse,
)
from tracingrag.services.context import ContextBuilder
from tracingrag.services.embedding import generate_embedding
from tracingrag.services.llm import LLMClient, get_llm_client
from tracingrag.services.retrieval import RetrievalService


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
    model: str = "anthropic/claude-3.5-sonnet",
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
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
