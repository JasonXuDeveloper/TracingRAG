"""Embedding service using sentence-transformers for vector generation"""

import asyncio
from functools import lru_cache
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from tracingrag.config import settings

# Global model cache
_embedding_model: SentenceTransformer | None = None
_device: str | None = None


def get_device() -> str:
    """Determine the best device to use for embeddings"""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"  # Apple Silicon
        else:
            _device = "cpu"
    return _device


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model (cached singleton)

    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model
    if _embedding_model is None:
        device = get_device()
        _embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=device,
        )
    return _embedding_model


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for a single text

    Args:
        text: Input text to embed

    Returns:
        Embedding vector as list of floats
    """
    model = get_embedding_model()

    # Run encoding in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None,
        lambda: model.encode(text, convert_to_numpy=True),
    )

    return embedding.tolist()


async def generate_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for multiple texts in batches

    Args:
        texts: List of input texts to embed
        batch_size: Number of texts to process at once

    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()

    # Run batch encoding in thread pool
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        ),
    )

    return [emb.tolist() for emb in embeddings]


async def compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Compute cosine similarity between two embeddings

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    model = get_embedding_model()

    # Convert to tensors
    tensor1 = torch.tensor(embedding1)
    tensor2 = torch.tensor(embedding2)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        tensor1.unsqueeze(0),
        tensor2.unsqueeze(0),
    )

    return float(similarity.item())


async def compute_similarities_batch(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
) -> list[float]:
    """Compute cosine similarities between a query and multiple candidates

    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate embedding vectors

    Returns:
        List of similarity scores
    """
    # Convert to tensors
    query_tensor = torch.tensor(query_embedding).unsqueeze(0)
    candidate_tensor = torch.tensor(candidate_embeddings)

    # Compute similarities
    similarities = torch.nn.functional.cosine_similarity(
        query_tensor,
        candidate_tensor,
    )

    return similarities.tolist()


def get_embedding_dimension() -> int:
    """Get the dimensionality of the embedding model

    Returns:
        Embedding dimension
    """
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()


async def prepare_text_for_embedding(
    content: str,
    metadata: dict[str, Any] | None = None,
    max_length: int = 8000,
) -> str:
    """Prepare text for embedding by combining content and metadata

    Args:
        content: Main content text
        metadata: Optional metadata to include in embedding
        max_length: Maximum character length

    Returns:
        Prepared text for embedding
    """
    # Start with content
    text = content

    # Add relevant metadata if provided
    if metadata:
        # Add topic if available
        if "topic" in metadata:
            text = f"Topic: {metadata['topic']}\n\n{text}"

        # Add entity type if available
        if "entity_type" in metadata:
            text = f"Type: {metadata['entity_type']}\n{text}"

        # Add tags if available
        if "tags" in metadata and metadata["tags"]:
            tags_str = ", ".join(metadata["tags"])
            text = f"{text}\n\nTags: {tags_str}"

    # Truncate if necessary
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


class EmbeddingCache:
    """Simple in-memory cache for embeddings"""

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size

    async def get(self, text: str) -> list[float] | None:
        """Get embedding from cache"""
        return self._cache.get(text)

    async def set(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache with LRU eviction"""
        if len(self._cache) >= self._max_size:
            # Remove oldest item (simple FIFO, not true LRU)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[text] = embedding

    async def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()


# Global cache instance
_embedding_cache = EmbeddingCache(max_size=1000)


async def generate_embedding_cached(text: str, use_cache: bool = True) -> list[float]:
    """Generate embedding with optional caching

    Args:
        text: Input text to embed
        use_cache: Whether to use the cache

    Returns:
        Embedding vector
    """
    if use_cache:
        # Check cache first
        cached = await _embedding_cache.get(text)
        if cached is not None:
            return cached

    # Generate embedding
    embedding = await generate_embedding(text)

    if use_cache:
        # Store in cache
        await _embedding_cache.set(text, embedding)

    return embedding


class EmbeddingService:
    """Embedding service with Redis caching support"""

    def __init__(self, use_redis_cache: bool = True):
        """Initialize embedding service

        Args:
            use_redis_cache: Whether to use Redis cache (falls back to in-memory if False)
        """
        self.use_redis_cache = use_redis_cache
        self._cache_service = None

    async def _get_cache(self):
        """Get cache service (lazy initialization)"""
        if self.use_redis_cache and self._cache_service is None:
            from tracingrag.services.cache import get_cache_service

            self._cache_service = get_cache_service()
        return self._cache_service

    async def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """Generate embedding with caching

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        if not use_cache:
            return await generate_embedding(text)

        # Try Redis cache first
        if self.use_redis_cache:
            cache = await self._get_cache()
            if cache:
                try:
                    cached = await cache.get_embedding(
                        text, settings.embedding_model
                    )
                    if cached is not None:
                        return cached
                except Exception:
                    # Fall back to no cache on Redis errors
                    pass

        # Fall back to in-memory cache
        cached = await _embedding_cache.get(text)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = await generate_embedding(text)

        # Store in caches
        if use_cache:
            # Store in in-memory cache
            await _embedding_cache.set(text, embedding)

            # Store in Redis cache
            if self.use_redis_cache:
                cache = await self._get_cache()
                if cache:
                    try:
                        await cache.set_embedding(
                            text, settings.embedding_model, embedding
                        )
                    except Exception:
                        # Silently fail on cache write errors
                        pass

        return embedding

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32, use_cache: bool = True
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with caching

        Args:
            texts: Texts to embed
            batch_size: Batch size for processing
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors
        """
        if not use_cache:
            return await generate_embeddings_batch(texts, batch_size)

        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache for each text
        cache = await self._get_cache() if self.use_redis_cache else None

        for i, text in enumerate(texts):
            cached = None

            # Try Redis cache first
            if cache:
                try:
                    cached = await cache.get_embedding(
                        text, settings.embedding_model
                    )
                except Exception:
                    pass

            # Fall back to in-memory cache
            if cached is None:
                cached = await _embedding_cache.get(text)

            if cached is not None:
                results.append(cached)
            else:
                results.append([])  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            uncached_embeddings = await generate_embeddings_batch(
                uncached_texts, batch_size
            )

            # Store in caches and fill results
            for idx, text, embedding in zip(
                uncached_indices, uncached_texts, uncached_embeddings
            ):
                results[idx] = embedding

                # Store in in-memory cache
                await _embedding_cache.set(text, embedding)

                # Store in Redis cache
                if cache:
                    try:
                        await cache.set_embedding(
                            text, settings.embedding_model, embedding
                        )
                    except Exception:
                        pass

        return results

    async def clear_cache(self, redis_only: bool = False):
        """Clear embedding cache

        Args:
            redis_only: If True, only clear Redis cache, not in-memory
        """
        if not redis_only:
            await _embedding_cache.clear()

        if self.use_redis_cache:
            cache = await self._get_cache()
            if cache:
                await cache.invalidate_embeddings(settings.embedding_model)
