"""Caching service with Redis backend"""

import hashlib
import json
import os
from datetime import timedelta
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis


class CacheService:
    """Redis-based caching service for embeddings, query results, and working memory"""

    def __init__(self, redis_url: str | None = None):
        """Initialize cache service

        Args:
            redis_url: Redis connection URL (default: from REDIS_URL env var)
        """
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self._client: Redis | None = None

    async def _get_client(self) -> Redis:
        """Get or create Redis client"""
        if self._client is None:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ========================================================================
    # Generic Cache Operations
    # ========================================================================

    async def get(self, key: str) -> Any | None:
        """Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        client = await self._get_client()
        value = await client.get(key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | timedelta | None = None,
    ) -> bool:
        """Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (seconds or timedelta)

        Returns:
            True if successful
        """
        client = await self._get_client()

        # Serialize value
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value)
        elif isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value)

        # Convert timedelta to seconds
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())

        if ttl:
            return await client.setex(key, ttl, serialized)
        else:
            return await client.set(key, serialized)

    async def delete(self, key: str) -> int:
        """Delete key from cache

        Args:
            key: Cache key

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()
        return await client.delete(key)

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern

        Args:
            pattern: Redis pattern (e.g., "embedding:*")

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()
        keys = await client.keys(pattern)
        if keys:
            return await client.delete(*keys)
        return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        client = await self._get_client()
        return await client.exists(key) > 0

    async def ttl(self, key: str) -> int:
        """Get time to live for key

        Args:
            key: Cache key

        Returns:
            TTL in seconds (-2 if key doesn't exist, -1 if no expiry)
        """
        client = await self._get_client()
        return await client.ttl(key)

    # ========================================================================
    # Embedding Cache
    # ========================================================================

    def _embedding_key(self, text: str, model: str) -> str:
        """Generate cache key for embedding

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            Cache key
        """
        # Hash text to avoid key length issues
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{model}:{text_hash}"

    async def get_embedding(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            Embedding vector or None if not cached
        """
        key = self._embedding_key(text, model)
        return await self.get(key)

    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: list[float],
        ttl: int | timedelta = timedelta(days=7),
    ) -> bool:
        """Cache embedding

        Args:
            text: Text that was embedded
            model: Embedding model name
            embedding: Embedding vector
            ttl: Time to live (default: 7 days)

        Returns:
            True if successful
        """
        key = self._embedding_key(text, model)
        return await self.set(key, embedding, ttl)

    async def invalidate_embeddings(self, model: str | None = None) -> int:
        """Invalidate embedding cache

        Args:
            model: Specific model to invalidate, or None for all

        Returns:
            Number of keys deleted
        """
        if model:
            pattern = f"embedding:{model}:*"
        else:
            pattern = "embedding:*"
        return await self.delete_pattern(pattern)

    # ========================================================================
    # Query Result Cache
    # ========================================================================

    def _query_key(self, query: str, params: dict[str, Any]) -> str:
        """Generate cache key for query result

        Args:
            query: Query text
            params: Query parameters

        Returns:
            Cache key
        """
        # Create deterministic key from query and params
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{query}:{params_str}"
        query_hash = hashlib.sha256(combined.encode()).hexdigest()
        return f"query:{query_hash}"

    async def get_query_result(
        self, query: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Get cached query result

        Args:
            query: Query text
            params: Query parameters

        Returns:
            Cached result or None if not found
        """
        key = self._query_key(query, params)
        return await self.get(key)

    async def set_query_result(
        self,
        query: str,
        params: dict[str, Any],
        result: dict[str, Any],
        ttl: int | timedelta = timedelta(hours=1),
    ) -> bool:
        """Cache query result

        Args:
            query: Query text
            params: Query parameters
            result: Query result
            ttl: Time to live (default: 1 hour)

        Returns:
            True if successful
        """
        key = self._query_key(query, params)
        return await self.set(key, result, ttl)

    async def invalidate_queries(self) -> int:
        """Invalidate all query result caches

        Returns:
            Number of keys deleted
        """
        return await self.delete_pattern("query:*")

    # ========================================================================
    # Working Memory Cache
    # ========================================================================

    def _working_memory_key(self, context_id: str) -> str:
        """Generate cache key for working memory

        Args:
            context_id: Context identifier

        Returns:
            Cache key
        """
        return f"working_memory:{context_id}"

    async def get_working_memory(self, context_id: str) -> dict[str, Any] | None:
        """Get working memory context

        Args:
            context_id: Context identifier

        Returns:
            Working memory context or None
        """
        key = self._working_memory_key(context_id)
        return await self.get(key)

    async def set_working_memory(
        self,
        context_id: str,
        context: dict[str, Any],
        ttl: int | timedelta = timedelta(minutes=30),
    ) -> bool:
        """Set working memory context

        Args:
            context_id: Context identifier
            context: Working memory context
            ttl: Time to live (default: 30 minutes)

        Returns:
            True if successful
        """
        key = self._working_memory_key(context_id)
        return await self.set(key, context, ttl)

    async def invalidate_working_memory(self, context_id: str | None = None) -> int:
        """Invalidate working memory

        Args:
            context_id: Specific context to invalidate, or None for all

        Returns:
            Number of keys deleted
        """
        if context_id:
            key = self._working_memory_key(context_id)
            return await self.delete(key)
        else:
            return await self.delete_pattern("working_memory:*")

    # ========================================================================
    # Memory State Cache (Latest States)
    # ========================================================================

    def _latest_state_key(self, topic: str) -> str:
        """Generate cache key for latest state

        Args:
            topic: Memory topic

        Returns:
            Cache key
        """
        return f"latest_state:{topic}"

    async def get_latest_state(self, topic: str) -> dict[str, Any] | None:
        """Get cached latest state

        Args:
            topic: Memory topic

        Returns:
            Latest state or None
        """
        key = self._latest_state_key(topic)
        return await self.get(key)

    async def set_latest_state(
        self,
        topic: str,
        state: dict[str, Any],
        ttl: int | timedelta = timedelta(hours=24),
    ) -> bool:
        """Cache latest state

        Args:
            topic: Memory topic
            state: Latest state
            ttl: Time to live (default: 24 hours)

        Returns:
            True if successful
        """
        key = self._latest_state_key(topic)
        return await self.set(key, state, ttl)

    async def invalidate_latest_state(self, topic: str | None = None) -> int:
        """Invalidate latest state cache

        Args:
            topic: Specific topic to invalidate, or None for all

        Returns:
            Number of keys deleted
        """
        if topic:
            key = self._latest_state_key(topic)
            return await self.delete(key)
        else:
            return await self.delete_pattern("latest_state:*")

    # ========================================================================
    # Cache Statistics
    # ========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics

        Returns:
            Cache statistics
        """
        client = await self._get_client()

        # Get Redis info
        info = await client.info()

        # Count keys by prefix
        embedding_keys = len(await client.keys("embedding:*"))
        query_keys = len(await client.keys("query:*"))
        working_memory_keys = len(await client.keys("working_memory:*"))
        latest_state_keys = len(await client.keys("latest_state:*"))

        return {
            "redis_version": info.get("redis_version"),
            "used_memory_mb": info.get("used_memory") / (1024 * 1024)
            if info.get("used_memory")
            else 0,
            "connected_clients": info.get("connected_clients", 0),
            "total_keys": await client.dbsize(),
            "embedding_keys": embedding_keys,
            "query_keys": query_keys,
            "working_memory_keys": working_memory_keys,
            "latest_state_keys": latest_state_keys,
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0)
            / max(
                info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                1,
            ),
        }

    async def clear_all(self) -> bool:
        """Clear entire cache (use with caution!)

        Returns:
            True if successful
        """
        client = await self._get_client()
        return await client.flushdb()

    # ========================================================================
    # Cache Warming
    # ========================================================================

    async def warm_embeddings(
        self, texts: list[str], model: str, batch_size: int = 100
    ) -> int:
        """Pre-warm embedding cache

        Args:
            texts: Texts to pre-compute embeddings for
            model: Embedding model name
            batch_size: Batch size for processing

        Returns:
            Number of embeddings cached
        """
        from tracingrag.services.embedding import EmbeddingService

        embedding_service = EmbeddingService()
        count = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await embedding_service.embed_batch(batch)

            for text, embedding in zip(batch, embeddings):
                await self.set_embedding(text, model, embedding)
                count += 1

        return count

    async def warm_latest_states(self, topics: list[str]) -> int:
        """Pre-warm latest state cache

        Args:
            topics: Topics to pre-cache latest states for

        Returns:
            Number of states cached
        """
        from tracingrag.services.memory import MemoryService

        memory_service = MemoryService()
        count = 0

        for topic in topics:
            # Get latest state from database
            state = await memory_service.get_latest_state(topic)
            if state:
                # Cache it
                await self.set_latest_state(
                    topic,
                    {
                        "id": str(state.id),
                        "topic": state.topic,
                        "content": state.content,
                        "version": state.version,
                        "timestamp": state.timestamp.isoformat(),
                        "confidence": state.confidence,
                    },
                )
                count += 1

        return count


# Global cache instance
_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """Get global cache service instance

    Returns:
        Cache service instance
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
