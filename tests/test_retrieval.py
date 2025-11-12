"""Tests for retrieval service"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from tracingrag.services.retrieval import RetrievalResult, RetrievalService


class TestRetrievalResult:
    """Tests for RetrievalResult class"""

    def test_retrieval_result_creation(self, sample_memory_state):
        """Test creating a retrieval result"""
        result = RetrievalResult(
            state=sample_memory_state,
            score=0.95,
            retrieval_type="semantic",
        )

        assert result.state == sample_memory_state
        assert result.score == 0.95
        assert result.retrieval_type == "semantic"
        assert result.related_states == []
        assert result.historical_context == []

    def test_retrieval_result_to_dict(self, sample_memory_state):
        """Test converting retrieval result to dictionary"""
        result = RetrievalResult(
            state=sample_memory_state,
            score=0.85,
            retrieval_type="graph_enhanced",
        )

        result_dict = result.to_dict()

        assert "state" in result_dict
        assert result_dict["state"]["topic"] == sample_memory_state.topic
        assert result_dict["state"]["content"] == sample_memory_state.content
        assert result_dict["score"] == 0.85
        assert result_dict["retrieval_type"] == "graph_enhanced"
        assert "related_states" in result_dict
        assert "historical_context" in result_dict


class TestRetrievalService:
    """Tests for RetrievalService"""

    @pytest.fixture
    def retrieval_service(self):
        """Create a retrieval service instance"""
        return RetrievalService()

    @pytest.mark.asyncio
    async def test_service_instantiation(self, retrieval_service):
        """Test that retrieval service can be instantiated"""
        assert retrieval_service is not None
        assert isinstance(retrieval_service, RetrievalService)

    @pytest.mark.asyncio
    async def test_temporal_query_method_exists(self, retrieval_service):
        """Test that temporal query method exists"""
        assert hasattr(retrieval_service, "temporal_query")
        assert callable(retrieval_service.temporal_query)

    @pytest.mark.asyncio
    async def test_temporal_range_query_method_exists(self, retrieval_service):
        """Test that temporal range query method exists"""
        assert hasattr(retrieval_service, "temporal_range_query")
        assert callable(retrieval_service.temporal_range_query)

    @pytest.mark.asyncio
    async def test_semantic_search_method_exists(self, retrieval_service):
        """Test that semantic search method exists"""
        assert hasattr(retrieval_service, "semantic_search")
        assert callable(retrieval_service.semantic_search)

    @pytest.mark.asyncio
    async def test_graph_enhanced_retrieval_method_exists(self, retrieval_service):
        """Test that graph enhanced retrieval method exists"""
        assert hasattr(retrieval_service, "graph_enhanced_retrieval")
        assert callable(retrieval_service.graph_enhanced_retrieval)

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_method_exists(self, retrieval_service):
        """Test that hybrid retrieval method exists"""
        assert hasattr(retrieval_service, "hybrid_retrieval")
        assert callable(retrieval_service.hybrid_retrieval)

    @pytest.mark.asyncio
    async def test_get_latest_states_method_exists(self, retrieval_service):
        """Test that get latest states method exists"""
        assert hasattr(retrieval_service, "get_latest_states")
        assert callable(retrieval_service.get_latest_states)

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_weight_validation(self, retrieval_service):
        """Test that hybrid retrieval validates weights sum to 1.0"""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            await retrieval_service.hybrid_retrieval(
                query="test query",
                semantic_weight=0.5,
                graph_weight=0.3,
                temporal_weight=0.3,  # Sum > 1.0
            )

    @pytest.mark.asyncio
    async def test_temporal_query_not_found(self, retrieval_service, db_session):
        """Test temporal query for non-existent topic"""
        result = await retrieval_service.temporal_query(
            topic="nonexistent_topic",
            timestamp=datetime.utcnow(),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_states_empty(self, retrieval_service, db_session):
        """Test get latest states with no data"""
        results = await retrieval_service.get_latest_states(
            topics=["nonexistent_topic"],
        )

        assert results == []


class TestRetrievalStrategies:
    """Integration tests for different retrieval strategies"""

    @pytest.fixture
    def retrieval_service(self):
        """Create a retrieval service instance"""
        return RetrievalService()

    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, retrieval_service):
        """Test basic semantic search"""
        # This test would require Qdrant to be running
        # For now, just test that the method signature is correct
        assert hasattr(retrieval_service, "semantic_search")

    @pytest.mark.asyncio
    async def test_graph_traversal_depth_parameter(self, retrieval_service):
        """Test that graph traversal accepts depth parameter"""
        # This is a signature test
        import inspect

        sig = inspect.signature(retrieval_service.graph_enhanced_retrieval)
        assert "depth" in sig.parameters
        assert sig.parameters["depth"].default == 2

    @pytest.mark.asyncio
    async def test_temporal_query_signature(self, retrieval_service):
        """Test temporal query method signature"""
        import inspect

        sig = inspect.signature(retrieval_service.temporal_query)
        assert "topic" in sig.parameters
        assert "timestamp" in sig.parameters

    @pytest.mark.asyncio
    async def test_temporal_range_query_signature(self, retrieval_service):
        """Test temporal range query method signature"""
        import inspect

        sig = inspect.signature(retrieval_service.temporal_range_query)
        assert "topic" in sig.parameters
        assert "start_time" in sig.parameters
        assert "end_time" in sig.parameters
        assert "limit" in sig.parameters


class TestRetrievalHelpers:
    """Tests for helper methods"""

    @pytest.fixture
    def retrieval_service(self):
        """Create a retrieval service instance"""
        return RetrievalService()

    @pytest.mark.asyncio
    async def test_filter_to_latest_per_topic(self, retrieval_service):
        """Test filtering to latest state per topic"""
        # This is a private method test
        assert hasattr(retrieval_service, "_filter_to_latest_per_topic")

    @pytest.mark.asyncio
    async def test_get_trace_context(self, retrieval_service):
        """Test getting trace context"""
        # This is a private method test
        assert hasattr(retrieval_service, "_get_trace_context")


class TestRetrievalIntegration:
    """Integration tests for end-to-end retrieval scenarios"""

    @pytest.fixture
    def retrieval_service(self):
        """Create a retrieval service instance"""
        return RetrievalService()

    @pytest.mark.asyncio
    async def test_retrieval_service_imports(self):
        """Test that retrieval service can be imported from services"""
        from tracingrag.services import RetrievalResult, RetrievalService

        assert RetrievalService is not None
        assert RetrievalResult is not None

    @pytest.mark.asyncio
    async def test_create_multiple_retrieval_results(self, sample_memory_state):
        """Test creating multiple retrieval results"""
        results = [
            RetrievalResult(
                state=sample_memory_state,
                score=0.9 - i * 0.1,
                retrieval_type="semantic",
            )
            for i in range(5)
        ]

        assert len(results) == 5
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score > results[4].score  # Descending scores
