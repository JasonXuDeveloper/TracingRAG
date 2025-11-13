"""Tests for caching service"""

import pytest
from datetime import timedelta

from tracingrag.services.cache import CacheService


class TestCacheService:
    """Test cache service functionality"""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service instance"""
        service = CacheService()
        yield service
        # Cleanup
        try:
            await service.clear_all()
            await service.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_cache_service_creation(self, cache_service):
        """Test that cache service can be instantiated"""
        assert cache_service is not None
        assert isinstance(cache_service, CacheService)

    @pytest.mark.asyncio
    async def test_basic_get_set(self, cache_service):
        """Test basic get/set operations"""
        # Set a value
        await cache_service.set("test_key", "test_value")

        # Get the value
        value = await cache_service.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache_service):
        """Test getting a key that doesn't exist"""
        value = await cache_service.get("nonexistent_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache_service):
        """Test setting a value with TTL"""
        # Set with 1 second TTL
        await cache_service.set("ttl_key", "ttl_value", ttl=1)

        # Should exist immediately
        value = await cache_service.get("ttl_key")
        assert value == "ttl_value"

        # Check TTL
        ttl = await cache_service.ttl("ttl_key")
        assert 0 < ttl <= 1

    @pytest.mark.asyncio
    async def test_set_with_timedelta_ttl(self, cache_service):
        """Test setting a value with timedelta TTL"""
        await cache_service.set("td_key", "td_value", ttl=timedelta(hours=1))

        # Should exist
        value = await cache_service.get("td_key")
        assert value == "td_value"

        # TTL should be around 3600 seconds
        ttl = await cache_service.ttl("td_key")
        assert 3500 < ttl <= 3600

    @pytest.mark.asyncio
    async def test_delete_key(self, cache_service):
        """Test deleting a key"""
        await cache_service.set("delete_me", "value")
        assert await cache_service.exists("delete_me")

        deleted = await cache_service.delete("delete_me")
        assert deleted == 1
        assert not await cache_service.exists("delete_me")

    @pytest.mark.asyncio
    async def test_delete_pattern(self, cache_service):
        """Test deleting keys by pattern"""
        # Set multiple keys
        await cache_service.set("prefix:key1", "value1")
        await cache_service.set("prefix:key2", "value2")
        await cache_service.set("other:key3", "value3")

        # Delete by pattern
        deleted = await cache_service.delete_pattern("prefix:*")
        assert deleted == 2

        # Verify
        assert not await cache_service.exists("prefix:key1")
        assert not await cache_service.exists("prefix:key2")
        assert await cache_service.exists("other:key3")

    @pytest.mark.asyncio
    async def test_dict_serialization(self, cache_service):
        """Test serialization of dictionaries"""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        await cache_service.set("dict_key", data)

        retrieved = await cache_service.get("dict_key")
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_list_serialization(self, cache_service):
        """Test serialization of lists"""
        data = [1, 2, 3, "four", {"five": 5}]
        await cache_service.set("list_key", data)

        retrieved = await cache_service.get("list_key")
        assert retrieved == data


class TestEmbeddingCache:
    """Test embedding-specific caching"""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service instance"""
        service = CacheService()
        yield service
        try:
            await service.clear_all()
            await service.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_embedding_cache(self, cache_service):
        """Test embedding caching"""
        text = "Hello, world!"
        model = "test-model"
        embedding = [0.1, 0.2, 0.3]

        # Set embedding
        await cache_service.set_embedding(text, model, embedding)

        # Get embedding
        cached = await cache_service.get_embedding(text, model)
        assert cached == embedding

    @pytest.mark.asyncio
    async def test_embedding_cache_different_models(self, cache_service):
        """Test that embeddings are cached per model"""
        text = "Hello, world!"
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        # Cache for two different models
        await cache_service.set_embedding(text, "model1", embedding1)
        await cache_service.set_embedding(text, "model2", embedding2)

        # Retrieve both
        cached1 = await cache_service.get_embedding(text, "model1")
        cached2 = await cache_service.get_embedding(text, "model2")

        assert cached1 == embedding1
        assert cached2 == embedding2

    @pytest.mark.asyncio
    async def test_invalidate_embeddings(self, cache_service):
        """Test invalidating embedding cache"""
        # Cache multiple embeddings
        await cache_service.set_embedding("text1", "model1", [0.1])
        await cache_service.set_embedding("text2", "model1", [0.2])
        await cache_service.set_embedding("text3", "model2", [0.3])

        # Invalidate model1 only
        deleted = await cache_service.invalidate_embeddings("model1")
        assert deleted == 2

        # Verify
        assert await cache_service.get_embedding("text1", "model1") is None
        assert await cache_service.get_embedding("text2", "model1") is None
        assert await cache_service.get_embedding("text3", "model2") == [0.3]


class TestQueryCache:
    """Test query result caching"""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service instance"""
        service = CacheService()
        yield service
        try:
            await cache_service.clear_all()
            await service.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_query_cache(self, cache_service):
        """Test query result caching"""
        query = "What is the status?"
        params = {"limit": 10, "include_history": True}
        result = {"answer": "All good", "confidence": 0.9}

        # Cache result
        await cache_service.set_query_result(query, params, result)

        # Retrieve result
        cached = await cache_service.get_query_result(query, params)
        assert cached == result

    @pytest.mark.asyncio
    async def test_query_cache_different_params(self, cache_service):
        """Test that queries with different params are cached separately"""
        query = "What is the status?"
        result1 = {"answer": "Result 1"}
        result2 = {"answer": "Result 2"}

        # Cache with different params
        await cache_service.set_query_result(query, {"limit": 10}, result1)
        await cache_service.set_query_result(query, {"limit": 20}, result2)

        # Retrieve both
        cached1 = await cache_service.get_query_result(query, {"limit": 10})
        cached2 = await cache_service.get_query_result(query, {"limit": 20})

        assert cached1 == result1
        assert cached2 == result2

    @pytest.mark.asyncio
    async def test_invalidate_queries(self, cache_service):
        """Test invalidating all query caches"""
        await cache_service.set_query_result("q1", {}, {"a": 1})
        await cache_service.set_query_result("q2", {}, {"a": 2})

        deleted = await cache_service.invalidate_queries()
        assert deleted >= 2

        assert await cache_service.get_query_result("q1", {}) is None
        assert await cache_service.get_query_result("q2", {}) is None


class TestWorkingMemoryCache:
    """Test working memory caching"""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service instance"""
        service = CacheService()
        yield service
        try:
            await cache_service.clear_all()
            await service.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_working_memory(self, cache_service):
        """Test working memory caching"""
        context_id = "session_123"
        context = {"active_topic": "project_alpha", "recent_queries": ["q1", "q2"]}

        # Set working memory
        await cache_service.set_working_memory(context_id, context)

        # Get working memory
        cached = await cache_service.get_working_memory(context_id)
        assert cached == context

    @pytest.mark.asyncio
    async def test_invalidate_working_memory(self, cache_service):
        """Test invalidating working memory"""
        await cache_service.set_working_memory("ctx1", {"data": 1})
        await cache_service.set_working_memory("ctx2", {"data": 2})

        # Invalidate specific context
        deleted = await cache_service.invalidate_working_memory("ctx1")
        assert deleted == 1

        assert await cache_service.get_working_memory("ctx1") is None
        assert await cache_service.get_working_memory("ctx2") == {"data": 2}


class TestLatestStateCache:
    """Test latest state caching"""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service instance"""
        service = CacheService()
        yield service
        try:
            await cache_service.clear_all()
            await service.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_latest_state_cache(self, cache_service):
        """Test latest state caching"""
        topic = "project_alpha"
        state = {
            "id": "123",
            "content": "Latest version",
            "version": 5,
        }

        # Cache latest state
        await cache_service.set_latest_state(topic, state)

        # Retrieve latest state
        cached = await cache_service.get_latest_state(topic)
        assert cached == state

    @pytest.mark.asyncio
    async def test_invalidate_latest_state(self, cache_service):
        """Test invalidating latest state cache"""
        await cache_service.set_latest_state("topic1", {"v": 1})
        await cache_service.set_latest_state("topic2", {"v": 2})

        # Invalidate specific topic
        deleted = await cache_service.invalidate_latest_state("topic1")
        assert deleted == 1

        assert await cache_service.get_latest_state("topic1") is None
        assert await cache_service.get_latest_state("topic2") == {"v": 2}


class TestCacheStats:
    """Test cache statistics"""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service instance"""
        service = CacheService()
        yield service
        try:
            await cache_service.clear_all()
            await service.close()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_get_stats(self, cache_service):
        """Test getting cache statistics"""
        # Add some data
        await cache_service.set_embedding("text", "model", [0.1])
        await cache_service.set_query_result("query", {}, {"answer": "yes"})
        await cache_service.set_working_memory("ctx", {"data": 1})
        await cache_service.set_latest_state("topic", {"v": 1})

        # Get stats
        stats = await cache_service.get_stats()

        assert "redis_version" in stats
        assert "total_keys" in stats
        assert "embedding_keys" in stats
        assert "query_keys" in stats
        assert "working_memory_keys" in stats
        assert "latest_state_keys" in stats

        # Should have at least 4 keys
        assert stats["total_keys"] >= 4
