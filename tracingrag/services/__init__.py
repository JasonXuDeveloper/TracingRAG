"""TracingRAG services module"""

from tracingrag.services.embedding import (
    compute_similarities_batch,
    compute_similarity,
    generate_embedding,
    generate_embedding_cached,
    generate_embeddings_batch,
    get_embedding_dimension,
    get_embedding_model,
    prepare_text_for_embedding,
)

__all__ = [
    "generate_embedding",
    "generate_embedding_cached",
    "generate_embeddings_batch",
    "compute_similarity",
    "compute_similarities_batch",
    "get_embedding_dimension",
    "get_embedding_model",
    "prepare_text_for_embedding",
]
