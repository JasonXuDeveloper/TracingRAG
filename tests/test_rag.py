"""Tests for RAG pipeline components"""

from datetime import datetime
from uuid import uuid4

import pytest

from tracingrag.core.models.memory import MemoryState
from tracingrag.core.models.rag import (
    ConsolidationLevel,
    ContextBudget,
    LLMRequest,
    LLMResponse,
    QueryType,
    RAGContext,
    RAGResponse,
    TokenEstimate,
)
from tracingrag.services.context import ContextBuilder


class TestRAGModels:
    """Tests for RAG data models"""

    def test_query_type_enum(self):
        """Test QueryType enum"""
        assert hasattr(QueryType, "STATUS")
        assert hasattr(QueryType, "WHY")
        assert hasattr(QueryType, "OVERVIEW")
        assert QueryType.STATUS.value == "status"
        assert QueryType.WHY.value == "why"

    def test_consolidation_level_enum(self):
        """Test ConsolidationLevel enum"""
        assert ConsolidationLevel.RAW == 0
        assert ConsolidationLevel.DAILY == 1
        assert ConsolidationLevel.WEEKLY == 2
        assert ConsolidationLevel.MONTHLY == 3

    def test_rag_context_creation(self):
        """Test RAGContext model creation"""
        context = RAGContext(
            query="What is the status?",
            query_type=QueryType.STATUS,
            max_tokens=100000,
        )

        assert context.query == "What is the status?"
        assert context.query_type == QueryType.STATUS
        assert context.max_tokens == 100000
        assert context.tokens_used == 0
        assert context.latest_states == []
        assert context.summaries == []
        assert context.detailed_states == []

    def test_llm_request_creation(self):
        """Test LLMRequest model creation"""
        request = LLMRequest(
            system_prompt="You are a helpful assistant",
            user_message="What is TracingRAG?",
            context="TracingRAG is a temporal graph-based RAG system",
            model="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=4096,
        )

        assert request.system_prompt == "You are a helpful assistant"
        assert request.user_message == "What is TracingRAG?"
        assert request.model == "anthropic/claude-3.5-sonnet"
        assert request.temperature == 0.7
        assert request.max_tokens == 4096

    def test_llm_response_creation(self):
        """Test LLMResponse model creation"""
        response = LLMResponse(
            content="TracingRAG is a system...",
            model="anthropic/claude-3.5-sonnet",
            tokens_used=150,
            finish_reason="stop",
        )

        assert response.content == "TracingRAG is a system..."
        assert response.model == "anthropic/claude-3.5-sonnet"
        assert response.tokens_used == 150
        assert response.finish_reason == "stop"

    def test_rag_response_creation(self):
        """Test RAGResponse model creation"""
        response = RAGResponse(
            query="What is the status?",
            answer="The current status is...",
            sources=[uuid4(), uuid4()],
            confidence=0.95,
            query_type=QueryType.STATUS,
            retrieval_time_ms=123.45,
            generation_time_ms=456.78,
            total_time_ms=580.23,
            tokens_retrieved=5000,
            tokens_generated=200,
        )

        assert response.query == "What is the status?"
        assert response.answer == "The current status is..."
        assert len(response.sources) == 2
        assert response.confidence == 0.95
        assert response.query_type == QueryType.STATUS
        assert response.retrieval_time_ms == 123.45
        assert response.generation_time_ms == 456.78
        assert response.total_time_ms == 580.23

    def test_context_budget_creation(self):
        """Test ContextBudget model creation"""
        budget = ContextBudget(
            max_tokens=100000,
            reserved_buffer=10000,
            latest_states_budget=30000,
            summaries_budget=30000,
        )

        assert budget.max_tokens == 100000
        assert budget.reserved_buffer == 10000
        assert budget.latest_states_budget == 30000
        assert budget.summaries_budget == 30000

        # Test details budget computation
        details_budget = budget.compute_details_budget()
        assert details_budget == 30000  # 100000 - 10000 - 30000 - 30000
        assert budget.details_budget == 30000

    def test_context_budget_overflow(self):
        """Test ContextBudget handles budget overflow"""
        budget = ContextBudget(
            max_tokens=50000,
            reserved_buffer=20000,
            latest_states_budget=30000,
            summaries_budget=30000,
        )

        details_budget = budget.compute_details_budget()
        assert details_budget == 0  # Can't go negative
        assert budget.details_budget == 0

    def test_token_estimate_char_count(self):
        """Test TokenEstimate with char_count method"""
        text = "This is a test string with multiple words"
        estimate = TokenEstimate.estimate(text, method="char_count")

        assert estimate.text == text
        assert estimate.estimated_tokens == len(text) // 4
        assert estimate.method == "char_count"

    def test_token_estimate_word_count(self):
        """Test TokenEstimate with word_count method"""
        text = "This is a test"  # 4 words
        estimate = TokenEstimate.estimate(text, method="word_count")

        assert estimate.text == text
        assert estimate.estimated_tokens == int(4 * 1.3)
        assert estimate.method == "word_count"


class TestContextBuilder:
    """Tests for ContextBuilder"""

    @pytest.fixture
    def context_builder(self):
        """Create context builder instance"""
        return ContextBuilder()

    def test_context_builder_instantiation(self, context_builder):
        """Test that context builder can be instantiated"""
        assert context_builder is not None
        assert isinstance(context_builder, ContextBuilder)

    def test_detect_query_type_status(self, context_builder):
        """Test query type detection for status queries"""
        query_type = context_builder._detect_query_type("What is the current status?")
        assert query_type == QueryType.STATUS

        query_type = context_builder._detect_query_type("Show me the latest updates")
        assert query_type == QueryType.STATUS

    def test_detect_query_type_why(self, context_builder):
        """Test query type detection for why queries"""
        query_type = context_builder._detect_query_type("Why did this happen?")
        assert query_type == QueryType.WHY

        query_type = context_builder._detect_query_type("Tell me why X failed")
        assert query_type == QueryType.WHY

    def test_detect_query_type_how(self, context_builder):
        """Test query type detection for how queries"""
        query_type = context_builder._detect_query_type("How does this work?")
        assert query_type == QueryType.HOW

        query_type = context_builder._detect_query_type("Show me how to do X")
        assert query_type == QueryType.HOW

    def test_detect_query_type_overview(self, context_builder):
        """Test query type detection for overview queries"""
        query_type = context_builder._detect_query_type("Summarize the project")
        assert query_type == QueryType.OVERVIEW

        query_type = context_builder._detect_query_type("Give me an overview")
        assert query_type == QueryType.OVERVIEW

    def test_detect_query_type_recent(self, context_builder):
        """Test query type detection for recent queries"""
        query_type = context_builder._detect_query_type("What happened last week?")
        assert query_type == QueryType.RECENT

        query_type = context_builder._detect_query_type("Show recent changes")
        assert query_type == QueryType.RECENT

    def test_detect_query_type_comparison(self, context_builder):
        """Test query type detection for comparison queries"""
        query_type = context_builder._detect_query_type("Compare X and Y")
        assert query_type == QueryType.COMPARISON

        query_type = context_builder._detect_query_type("What's the difference between A vs B")
        assert query_type == QueryType.COMPARISON

    def test_determine_consolidation_level_status(self, context_builder):
        """Test consolidation level for status queries"""
        level = context_builder._determine_consolidation_level(QueryType.STATUS)
        assert level == ConsolidationLevel.RAW

    def test_determine_consolidation_level_recent(self, context_builder):
        """Test consolidation level for recent queries"""
        level = context_builder._determine_consolidation_level(QueryType.RECENT)
        assert level == ConsolidationLevel.DAILY

    def test_determine_consolidation_level_overview(self, context_builder):
        """Test consolidation level for overview queries"""
        level = context_builder._determine_consolidation_level(QueryType.OVERVIEW)
        assert level == ConsolidationLevel.MONTHLY

    def test_determine_consolidation_level_why(self, context_builder):
        """Test consolidation level for why queries (detailed)"""
        level = context_builder._determine_consolidation_level(QueryType.WHY)
        assert level == ConsolidationLevel.RAW

    def test_create_budget_status_query(self, context_builder):
        """Test budget creation for status queries"""
        budget = context_builder._create_budget(100000, QueryType.STATUS)

        assert budget.max_tokens == 100000
        assert budget.reserved_buffer == 10000
        # Status queries prioritize latest states
        assert budget.latest_states_budget == int(100000 * 0.7)
        assert budget.summaries_budget == 0

    def test_create_budget_overview_query(self, context_builder):
        """Test budget creation for overview queries"""
        budget = context_builder._create_budget(100000, QueryType.OVERVIEW)

        assert budget.max_tokens == 100000
        # Overview queries prioritize summaries
        assert budget.latest_states_budget == int(100000 * 0.2)
        assert budget.summaries_budget == int(100000 * 0.5)

    def test_create_budget_general_query(self, context_builder):
        """Test budget creation for general queries"""
        budget = context_builder._create_budget(100000, QueryType.GENERAL)

        assert budget.max_tokens == 100000
        # Balanced approach
        assert budget.latest_states_budget == int(100000 * 0.3)
        assert budget.summaries_budget == int(100000 * 0.3)

    def test_format_state(self, context_builder):
        """Test memory state formatting"""
        state = MemoryState(
            topic="test_topic",
            content="This is test content",
            version=1,
            timestamp=datetime.utcnow(),
            tags=["test", "example"],
            confidence=0.95,
        )

        formatted = context_builder._format_state(state, include_metadata=True)

        assert "test_topic" in formatted
        assert "This is test content" in formatted
        assert "Version: 1" in formatted
        assert "test, example" in formatted
        assert "0.95" in formatted

    def test_format_state_without_metadata(self, context_builder):
        """Test memory state formatting without metadata"""
        state = MemoryState(
            topic="test_topic",
            content="This is test content",
            version=1,
        )

        formatted = context_builder._format_state(state, include_metadata=False)

        assert "test_topic" in formatted
        assert "This is test content" in formatted
        assert "Version" not in formatted

    def test_format_context_for_llm(self, context_builder):
        """Test formatting RAG context for LLM"""
        state1 = MemoryState(
            topic="topic1",
            content="Content 1",
            version=1,
        )
        state2 = MemoryState(
            topic="topic2",
            content="Content 2",
            version=1,
        )

        context = RAGContext(
            query="Test query",
            query_type=QueryType.GENERAL,
            latest_states=[state1],
            detailed_states=[state2],
        )

        formatted = context_builder.format_context_for_llm(context)

        assert "# Current State" in formatted
        assert "topic1" in formatted
        assert "Content 1" in formatted
        assert "# Detailed Context" in formatted
        assert "topic2" in formatted
        assert "Content 2" in formatted


class TestRAGIntegration:
    """Integration tests for RAG components"""

    def test_rag_models_import(self):
        """Test that RAG models can be imported"""
        from tracingrag.core.models import (
            ConsolidationLevel,
            ContextBudget,
            LLMRequest,
            LLMResponse,
            QueryType,
            RAGContext,
            RAGResponse,
            TokenEstimate,
        )

        assert QueryType is not None
        assert RAGContext is not None
        assert LLMRequest is not None
        assert LLMResponse is not None
        assert RAGResponse is not None

    def test_rag_services_import(self):
        """Test that RAG services can be imported"""
        from tracingrag.services import ContextBuilder, LLMClient, RAGService, query_rag

        assert ContextBuilder is not None
        assert LLMClient is not None
        assert RAGService is not None
        assert query_rag is not None

    def test_end_to_end_imports(self):
        """Test complete import chain"""
        # Models
        from tracingrag.core.models.rag import QueryType, RAGContext

        # Services
        from tracingrag.services.context import ContextBuilder
        from tracingrag.services.rag import RAGService

        # Create instances
        builder = ContextBuilder()
        assert builder is not None

        # Verify query type detection
        query_type = builder._detect_query_type("What is the status?")
        assert query_type == QueryType.STATUS
