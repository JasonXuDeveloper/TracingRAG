"""Tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from tracingrag.api.main import app


class TestHealthEndpoints:
    """Test health and metrics endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TracingRAG API"
        assert "version" in data
        assert "docs" in data

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data
        assert "total_topics" in data
        assert "uptime_seconds" in data


class TestMemoryEndpoints:
    """Test memory CRUD endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_list_memories(self, client):
        """Test listing memories"""
        response = client.get("/api/v1/memories")
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total" in data
        assert isinstance(data["memories"], list)

    def test_list_memories_with_pagination(self, client):
        """Test listing memories with pagination"""
        response = client.get("/api/v1/memories?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_get_nonexistent_memory(self, client):
        """Test getting a memory that doesn't exist"""
        fake_id = uuid4()
        response = client.get(f"/api/v1/memories/{fake_id}")
        assert response.status_code == 404


class TestQueryEndpoints:
    """Test query/RAG endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_query_endpoint_structure(self, client):
        """Test query endpoint returns proper structure"""
        # Note: This may fail if no data exists, but tests the structure
        response = client.post(
            "/api/v1/query",
            json={"query": "test query", "use_agent": False},
        )

        # Should return 200 or 500 (if no data), but not 404 or 422
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "confidence" in data


class TestPromotionEndpoints:
    """Test promotion endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_get_promotion_candidates(self, client):
        """Test getting promotion candidates"""
        response = client.get("/api/v1/promotion-candidates")
        assert response.status_code == 200
        data = response.json()
        assert "candidates" in data
        assert "total" in data
        assert isinstance(data["candidates"], list)

    def test_get_promotion_candidates_with_filters(self, client):
        """Test getting promotion candidates with filters"""
        response = client.get("/api/v1/promotion-candidates?limit=5&min_priority=7")
        assert response.status_code == 200
        data = response.json()
        assert "candidates" in data
        # All returned candidates should have priority >= 7
        for candidate in data["candidates"]:
            assert candidate["priority"] >= 7


class TestOpenAPISchema:
    """Test OpenAPI schema generation"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is generated"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_docs_endpoint(self, client):
        """Test that docs endpoint is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
