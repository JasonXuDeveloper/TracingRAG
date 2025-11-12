"""TracingRAG services module"""

from tracingrag.services.context import ContextBuilder
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
from tracingrag.services.graph import GraphService
from tracingrag.services.llm import LLMClient, close_llm_client, get_llm_client
from tracingrag.services.memory import MemoryService
from tracingrag.services.promotion import PromotionService, get_promotion_service
from tracingrag.services.query_analyzer import QueryAnalyzer, get_query_analyzer
from tracingrag.services.rag import RAGService, query_rag
from tracingrag.services.retrieval import RetrievalResult, RetrievalService

__all__ = [
    # Embedding services
    "generate_embedding",
    "generate_embedding_cached",
    "generate_embeddings_batch",
    "compute_similarity",
    "compute_similarities_batch",
    "get_embedding_dimension",
    "get_embedding_model",
    "prepare_text_for_embedding",
    # Core services
    "MemoryService",
    "RetrievalService",
    "RetrievalResult",
    "GraphService",
    "PromotionService",
    "get_promotion_service",
    # LLM services
    "LLMClient",
    "get_llm_client",
    "close_llm_client",
    # RAG services
    "ContextBuilder",
    "RAGService",
    "query_rag",
    "QueryAnalyzer",
    "get_query_analyzer",
]
