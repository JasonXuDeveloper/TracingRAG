"""Redis client for caching"""

import json
from typing import Any

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore

from tracingrag.config import settings
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

# Global Redis connection pool
_redis_pool: "redis.ConnectionPool | None" = None if REDIS_AVAILABLE else None


def get_redis_pool() -> redis.ConnectionPool:
    """Get or create Redis connection pool (singleton)"""
    global _redis_pool

    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,  # Auto-decode bytes to str
        )

    return _redis_pool


async def get_redis_client() -> redis.Redis:
    """Get Redis client from pool"""
    pool = get_redis_pool()
    return redis.Redis(connection_pool=pool)


async def close_redis_pool() -> None:
    """Close Redis connection pool (call on shutdown)"""
    global _redis_pool

    if _redis_pool is not None:
        await _redis_pool.disconnect()
        _redis_pool = None


class CacheService:
    """Service for caching formatted states and retrieval results"""

    def __init__(self):
        self.default_ttl = 900  # 15 minutes

    async def get_formatted_state(self, state_id: str) -> str | None:
        """
        Get cached formatted state

        Args:
            state_id: Memory state ID

        Returns:
            Formatted state string or None if not cached
        """
        if not REDIS_AVAILABLE:
            return None

        try:
            client = await get_redis_client()
            cache_key = f"state:formatted:{state_id}"
            cached = await client.get(cache_key)
            return cached
        except Exception as e:
            # Fail gracefully - caching is optional
            logger.debug(f"Cache get failed: {e}")
            return None

    async def set_formatted_state(
        self, state_id: str, formatted_content: str, ttl: int | None = None
    ) -> None:
        """
        Cache formatted state

        Args:
            state_id: Memory state ID
            formatted_content: Formatted state content
            ttl: Time to live in seconds (default: 15 minutes)
        """
        if not REDIS_AVAILABLE:
            return

        try:
            client = await get_redis_client()
            cache_key = f"state:formatted:{state_id}"
            ttl = ttl or self.default_ttl
            await client.setex(cache_key, ttl, formatted_content)
        except Exception as e:
            # Fail gracefully - caching is optional
            logger.debug(f"Cache set failed: {e}")
            pass

    async def get_retrieval_results(
        self, query_hash: str, limit: int, threshold: float
    ) -> list[dict[str, Any]] | None:
        """
        Get cached retrieval results

        Args:
            query_hash: Hash of query text
            limit: Result limit
            threshold: Score threshold

        Returns:
            List of retrieval results or None if not cached
        """
        if not REDIS_AVAILABLE:
            return None

        try:
            client = await get_redis_client()
            cache_key = f"retrieval:{query_hash}:limit{limit}:thresh{threshold:.2f}"
            cached = await client.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            return None

    async def set_retrieval_results(
        self,
        query_hash: str,
        limit: int,
        threshold: float,
        results: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> None:
        """
        Cache retrieval results

        Args:
            query_hash: Hash of query text
            limit: Result limit
            threshold: Score threshold
            results: Retrieval results to cache
            ttl: Time to live in seconds (default: 15 minutes)
        """
        if not REDIS_AVAILABLE:
            return

        try:
            client = await get_redis_client()
            cache_key = f"retrieval:{query_hash}:limit{limit}:thresh{threshold:.2f}"
            ttl = ttl or self.default_ttl
            # Convert to JSON
            serialized = json.dumps(results)
            await client.setex(cache_key, ttl, serialized)
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
            pass

    async def invalidate_state(self, state_id: str) -> None:
        """
        Invalidate all caches related to a state

        Args:
            state_id: Memory state ID to invalidate
        """
        if not REDIS_AVAILABLE:
            return

        try:
            client = await get_redis_client()
            # Delete formatted state cache
            await client.delete(f"state:formatted:{state_id}")

            # NOTE: We don't invalidate retrieval caches here because:
            # 1. They expire automatically in 15 minutes
            # 2. Deleting specific retrieval caches is complex (need to match patterns)
            # 3. Stale retrieval results for 15 min is acceptable trade-off
        except Exception as e:
            logger.debug(f"Cache invalidate failed: {e}")
            pass


# Global cache service instance
cache_service = CacheService()
