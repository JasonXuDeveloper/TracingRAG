"""Qdrant vector database client for embedding storage and similarity search"""

from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams

from tracingrag.config import settings

# Global Qdrant client instance
_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if hasattr(settings, "qdrant_api_key") else None,
            timeout=30,
        )
    return _qdrant_client


async def init_qdrant_collection(
    collection_name: str = "memory_states",
    vector_size: int = 768,  # Default for sentence-transformers all-mpnet-base-v2
    distance: Distance = Distance.COSINE,
) -> None:
    """Initialize Qdrant collection with proper configuration

    Args:
        collection_name: Name of the collection to create
        vector_size: Dimension of the embedding vectors
        distance: Distance metric to use (COSINE, EUCLID, or DOT)
    """
    client = get_qdrant_client()

    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)

    if not collection_exists:
        # Create collection with optimized settings
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
                on_disk=False,  # Keep vectors in RAM for faster access
            ),
            # Optimized for fast search
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,  # Build index after 10k vectors
            ),
            # Use HNSW index for fast approximate nearest neighbor search
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Number of edges per node
                ef_construct=100,  # Construction time parameter
                full_scan_threshold=10000,  # Use full scan for small collections
            ),
        )

    # Create payload indexes (idempotent - will not fail if already exists)
    # This ensures indexes are created for both new and existing collections
    _ensure_payload_indexes(client, collection_name)


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Ensure all required payload indexes exist (idempotent)

    Args:
        client: Qdrant client
        collection_name: Collection name
    """
    indexes_to_create = [
        ("topic", models.PayloadSchemaType.KEYWORD),
        ("storage_tier", models.PayloadSchemaType.KEYWORD),
        ("entity_type", models.PayloadSchemaType.KEYWORD),
        ("is_consolidated", models.PayloadSchemaType.BOOL),
        ("consolidation_level", models.PayloadSchemaType.INTEGER),
        ("is_latest", models.PayloadSchemaType.BOOL),
        ("is_active", models.PayloadSchemaType.BOOL),
    ]

    for field_name, field_schema in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            # Index already exists, ignore
            pass


async def upsert_embedding(
    state_id: UUID,
    embedding: list[float],
    payload: dict[str, Any],
    collection_name: str = "memory_states",
) -> None:
    """Insert or update an embedding in Qdrant

    Args:
        state_id: UUID of the memory state
        embedding: Vector embedding
        payload: Additional metadata to store with the vector
        collection_name: Name of the collection
    """
    client = get_qdrant_client()

    # Auto-create collection and ensure indexes exist
    try:
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)

        if not collection_exists:
            vector_size = len(embedding)
            await init_qdrant_collection(
                collection_name=collection_name,
                vector_size=vector_size,
            )
        else:
            # Collection exists, ensure indexes are up to date
            _ensure_payload_indexes(client, collection_name)
    except Exception:
        # If collection check fails, try to create it
        vector_size = len(embedding)
        await init_qdrant_collection(
            collection_name=collection_name,
            vector_size=vector_size,
        )

    point = PointStruct(
        id=str(state_id),
        vector=embedding,
        payload=payload,
    )

    client.upsert(
        collection_name=collection_name,
        points=[point],
    )


async def search_similar(
    query_vector: list[float],
    limit: int = 10,
    score_threshold: float | None = None,
    filter_conditions: dict[str, Any] | None = None,
    latest_only: bool = False,
    collection_name: str = "memory_states",
) -> list[dict[str, Any]]:
    """Search for similar vectors in Qdrant

    Args:
        query_vector: Query embedding vector
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0-1 for cosine)
        filter_conditions: Qdrant filter conditions for metadata filtering
        latest_only: If True, only return latest version per topic (filters by is_latest=True)
        collection_name: Name of the collection to search

    Returns:
        List of search results with id, score, and payload
    """
    client = get_qdrant_client()

    # Build filter from conditions
    query_filter = None
    if filter_conditions or latest_only:
        must_conditions = []

        # Add latest_only filter
        if latest_only:
            must_conditions.append(
                models.FieldCondition(
                    key="is_latest",
                    match=models.MatchValue(value=True),
                )
            )

        # Add custom filter conditions
        if filter_conditions:
            for field, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value),
                    )
                )

        query_filter = models.Filter(must=must_conditions)

    # Perform search
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
    )

    # Convert to dict format
    return [
        {
            "id": UUID(result.id) if isinstance(result.id, str) else result.id,
            "score": result.score,
            "payload": result.payload,
        }
        for result in results
    ]


async def delete_embedding(state_id: UUID, collection_name: str = "memory_states") -> None:
    """Delete an embedding from Qdrant

    Args:
        state_id: UUID of the memory state to delete
        collection_name: Name of the collection
    """
    client = get_qdrant_client()

    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(
            points=[str(state_id)],
        ),
    )


async def batch_upsert_embeddings(
    embeddings: list[tuple[UUID, list[float], dict[str, Any]]],
    collection_name: str = "memory_states",
) -> None:
    """Batch insert/update embeddings in Qdrant for better performance

    Args:
        embeddings: List of (state_id, embedding, payload) tuples
        collection_name: Name of the collection
    """
    client = get_qdrant_client()

    points = [
        PointStruct(
            id=str(state_id),
            vector=embedding,
            payload=payload,
        )
        for state_id, embedding, payload in embeddings
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
    )


async def close_qdrant() -> None:
    """Close Qdrant client connection"""
    global _qdrant_client
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None
