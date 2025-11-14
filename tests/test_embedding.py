"""Tests for embedding service"""

import pytest

from tracingrag.services.embedding import (
    compute_similarities_batch,
    compute_similarity,
    generate_embedding,
    generate_embedding_cached,
    generate_embeddings_batch,
    get_embedding_dimension,
    prepare_text_for_embedding,
)


class TestEmbeddingGeneration:
    """Tests for embedding generation"""

    @pytest.mark.asyncio
    async def test_generate_embedding(self):
        """Test generating a single embedding"""
        text = "This is a test sentence for embedding."
        embedding = await generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # Default sentence-transformers dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self):
        """Test generating multiple embeddings in batch"""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]
        embeddings = await generate_embeddings_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        assert all(isinstance(x, float) for emb in embeddings for x in emb)

    @pytest.mark.asyncio
    async def test_embedding_dimension(self):
        """Test getting embedding dimension"""
        dimension = get_embedding_dimension()
        assert dimension == 768

    @pytest.mark.asyncio
    async def test_embedding_deterministic(self):
        """Test that same text produces same embedding"""
        text = "Deterministic test sentence."
        embedding1 = await generate_embedding(text)
        embedding2 = await generate_embedding(text)

        # Embeddings should be very similar (allowing for floating point precision)
        for v1, v2 in zip(embedding1, embedding2, strict=False):
            assert abs(v1 - v2) < 1e-6


class TestSimilarityComputation:
    """Tests for similarity computation"""

    @pytest.mark.asyncio
    async def test_compute_similarity_identical(self):
        """Test similarity of identical embeddings"""
        text = "Test sentence for similarity."
        embedding = await generate_embedding(text)

        similarity = await compute_similarity(embedding, embedding)
        assert 0.99 < similarity < 1.01  # Should be very close to 1.0, allow floating point precision

    @pytest.mark.asyncio
    async def test_compute_similarity_different(self):
        """Test similarity of different embeddings"""
        text1 = "The cat sat on the mat."
        text2 = "Quantum physics is fascinating."

        embedding1 = await generate_embedding(text1)
        embedding2 = await generate_embedding(text2)

        similarity = await compute_similarity(embedding1, embedding2)
        assert 0.0 <= similarity < 0.5  # Should be relatively low

    @pytest.mark.asyncio
    async def test_compute_similarity_similar(self):
        """Test similarity of semantically similar texts"""
        text1 = "The cat is sleeping on the couch."
        text2 = "A cat is resting on the sofa."

        embedding1 = await generate_embedding(text1)
        embedding2 = await generate_embedding(text2)

        similarity = await compute_similarity(embedding1, embedding2)
        assert 0.5 < similarity < 1.0  # Should be moderately high

    @pytest.mark.asyncio
    async def test_compute_similarities_batch(self):
        """Test batch similarity computation"""
        query_text = "The cat sat on the mat."
        candidate_texts = [
            "The cat is on the mat.",
            "Dogs are playing in the park.",
            "Quantum mechanics is complex.",
        ]

        query_embedding = await generate_embedding(query_text)
        candidate_embeddings = await generate_embeddings_batch(candidate_texts)

        similarities = await compute_similarities_batch(query_embedding, candidate_embeddings)

        assert len(similarities) == 3
        assert all(0.0 <= sim <= 1.0 for sim in similarities)
        # First candidate should be most similar
        assert similarities[0] > similarities[1]
        assert similarities[0] > similarities[2]


class TestTextPreparation:
    """Tests for text preparation"""

    @pytest.mark.asyncio
    async def test_prepare_text_basic(self):
        """Test basic text preparation"""
        content = "This is test content."
        prepared = await prepare_text_for_embedding(content)

        assert prepared == content

    @pytest.mark.asyncio
    async def test_prepare_text_with_topic(self):
        """Test text preparation with topic metadata"""
        content = "This is test content."
        metadata = {"topic": "test_topic"}

        prepared = await prepare_text_for_embedding(content, metadata)

        assert "test_topic" in prepared
        assert content in prepared

    @pytest.mark.asyncio
    async def test_prepare_text_with_entity_type(self):
        """Test text preparation with entity type"""
        content = "This is test content."
        metadata = {"entity_type": "Character"}

        prepared = await prepare_text_for_embedding(content, metadata)

        assert "Character" in prepared
        assert content in prepared

    @pytest.mark.asyncio
    async def test_prepare_text_with_tags(self):
        """Test text preparation with tags"""
        content = "This is test content."
        metadata = {"tags": ["tag1", "tag2", "tag3"]}

        prepared = await prepare_text_for_embedding(content, metadata)

        assert "tag1" in prepared
        assert "tag2" in prepared
        assert "tag3" in prepared
        assert content in prepared

    @pytest.mark.asyncio
    async def test_prepare_text_truncation(self):
        """Test text truncation for long content"""
        content = "x" * 10000
        prepared = await prepare_text_for_embedding(content, max_length=8000)

        assert len(prepared) <= 8003  # 8000 + "..."


class TestEmbeddingCache:
    """Tests for embedding caching"""

    @pytest.mark.asyncio
    async def test_cached_embedding_hit(self):
        """Test cache hit for repeated text"""
        text = "Cached test sentence."

        # First call - cache miss
        embedding1 = await generate_embedding_cached(text, use_cache=True)

        # Second call - cache hit (should be faster and identical)
        embedding2 = await generate_embedding_cached(text, use_cache=True)

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_cached_embedding_disabled(self):
        """Test with caching disabled"""
        text = "Non-cached test sentence."

        embedding1 = await generate_embedding_cached(text, use_cache=False)
        embedding2 = await generate_embedding_cached(text, use_cache=False)

        # Should be very similar but not necessarily identical references
        assert len(embedding1) == len(embedding2)
