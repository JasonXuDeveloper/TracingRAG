"""Prometheus metrics service for monitoring"""

import time
from collections.abc import Callable
from functools import wraps

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ============================================================================
# System Metrics
# ============================================================================

system_info = Info("tracingrag_system", "TracingRAG system information")
system_info.info(
    {
        "version": "0.2.0",
        "component": "tracingrag",
    }
)


# ============================================================================
# API Metrics
# ============================================================================

# Request metrics
api_requests_total = Counter(
    "tracingrag_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

api_request_duration_seconds = Histogram(
    "tracingrag_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

api_active_requests = Gauge(
    "tracingrag_api_active_requests",
    "Number of active API requests",
    ["method", "endpoint"],
)


# ============================================================================
# Memory Metrics
# ============================================================================

memory_states_total = Counter(
    "tracingrag_memory_states_total",
    "Total number of memory states created",
    ["source"],
)

memory_states_current = Gauge(
    "tracingrag_memory_states_current",
    "Current number of memory states",
)

memory_topics_current = Gauge(
    "tracingrag_memory_topics_current",
    "Current number of unique topics",
)

memory_versions_avg = Gauge(
    "tracingrag_memory_versions_avg",
    "Average number of versions per topic",
)


# ============================================================================
# Query Metrics
# ============================================================================

query_requests_total = Counter(
    "tracingrag_query_requests_total",
    "Total number of query requests",
    ["use_agent"],
)

query_duration_seconds = Histogram(
    "tracingrag_query_duration_seconds",
    "Query duration in seconds",
    ["use_agent", "phase"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

query_states_retrieved = Histogram(
    "tracingrag_query_states_retrieved",
    "Number of states retrieved per query",
    buckets=(1, 5, 10, 20, 50, 100, 200),
)

query_confidence_score = Histogram(
    "tracingrag_query_confidence_score",
    "Query confidence scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
)


# ============================================================================
# Embedding Metrics
# ============================================================================

embedding_requests_total = Counter(
    "tracingrag_embedding_requests_total",
    "Total number of embedding requests",
    ["model"],
)

embedding_cache_hits = Counter(
    "tracingrag_embedding_cache_hits",
    "Number of embedding cache hits",
    ["cache_type"],
)

embedding_cache_misses = Counter(
    "tracingrag_embedding_cache_misses",
    "Number of embedding cache misses",
)

embedding_duration_seconds = Histogram(
    "tracingrag_embedding_duration_seconds",
    "Embedding generation duration in seconds",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)


# ============================================================================
# LLM Metrics
# ============================================================================

llm_requests_total = Counter(
    "tracingrag_llm_requests_total",
    "Total number of LLM requests",
    ["model", "operation"],
)

llm_tokens_total = Counter(
    "tracingrag_llm_tokens_total",
    "Total number of LLM tokens consumed",
    ["model", "token_type"],
)

llm_duration_seconds = Histogram(
    "tracingrag_llm_duration_seconds",
    "LLM request duration in seconds",
    ["model", "operation"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

llm_errors_total = Counter(
    "tracingrag_llm_errors_total",
    "Total number of LLM errors",
    ["model", "error_type"],
)


# ============================================================================
# Promotion Metrics
# ============================================================================

promotion_requests_total = Counter(
    "tracingrag_promotion_requests_total",
    "Total number of promotion requests",
    ["trigger"],
)

promotion_success_total = Counter(
    "tracingrag_promotion_success_total",
    "Total number of successful promotions",
)

promotion_failures_total = Counter(
    "tracingrag_promotion_failures_total",
    "Total number of failed promotions",
)

promotion_duration_seconds = Histogram(
    "tracingrag_promotion_duration_seconds",
    "Promotion duration in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

promotion_states_synthesized = Histogram(
    "tracingrag_promotion_states_synthesized",
    "Number of states synthesized per promotion",
    buckets=(1, 5, 10, 20, 50, 100),
)

promotion_conflicts_detected = Histogram(
    "tracingrag_promotion_conflicts_detected",
    "Number of conflicts detected per promotion",
    buckets=(0, 1, 2, 5, 10, 20),
)


# ============================================================================
# Cache Metrics
# ============================================================================

cache_operations_total = Counter(
    "tracingrag_cache_operations_total",
    "Total number of cache operations",
    ["operation", "cache_type"],
)

cache_hit_ratio = Gauge(
    "tracingrag_cache_hit_ratio",
    "Cache hit ratio",
    ["cache_type"],
)

cache_size_bytes = Gauge(
    "tracingrag_cache_size_bytes",
    "Cache size in bytes",
)

cache_keys_total = Gauge(
    "tracingrag_cache_keys_total",
    "Total number of cache keys",
    ["cache_type"],
)


# ============================================================================
# Consolidation Metrics
# ============================================================================

consolidation_runs_total = Counter(
    "tracingrag_consolidation_runs_total",
    "Total number of consolidation runs",
    ["level"],
)

consolidation_topics_processed = Counter(
    "tracingrag_consolidation_topics_processed",
    "Total number of topics consolidated",
    ["level"],
)

consolidation_duration_seconds = Histogram(
    "tracingrag_consolidation_duration_seconds",
    "Consolidation duration in seconds",
    ["level"],
    buckets=(1.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0),
)


# ============================================================================
# Database Metrics
# ============================================================================

db_connections_active = Gauge(
    "tracingrag_db_connections_active",
    "Number of active database connections",
    ["database"],
)

db_query_duration_seconds = Histogram(
    "tracingrag_db_query_duration_seconds",
    "Database query duration in seconds",
    ["database", "operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

db_errors_total = Counter(
    "tracingrag_db_errors_total",
    "Total number of database errors",
    ["database", "error_type"],
)


# ============================================================================
# Graph Metrics
# ============================================================================

graph_nodes_total = Gauge(
    "tracingrag_graph_nodes_total",
    "Total number of graph nodes",
)

graph_edges_total = Gauge(
    "tracingrag_graph_edges_total",
    "Total number of graph edges",
)

graph_traversal_duration_seconds = Histogram(
    "tracingrag_graph_traversal_duration_seconds",
    "Graph traversal duration in seconds",
    ["depth"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)


# ============================================================================
# Decorator for timing operations
# ============================================================================


def track_duration(histogram: Histogram, **labels):
    """Decorator to track operation duration

    Args:
        histogram: Histogram metric to track duration
        **labels: Labels for the histogram
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                histogram.labels(**labels).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                histogram.labels(**labels).observe(duration)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# Utility functions
# ============================================================================


def get_metrics() -> bytes:
    """Get Prometheus metrics in text format

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest()


def get_content_type() -> str:
    """Get Prometheus metrics content type

    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST


class MetricsCollector:
    """Helper class for collecting metrics"""

    @staticmethod
    def record_api_request(method: str, endpoint: str, status: int, duration: float):
        """Record API request metrics"""
        api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    @staticmethod
    def record_query(
        use_agent: bool,
        duration: float,
        states_retrieved: int,
        confidence: float,
    ):
        """Record query metrics"""
        agent_str = "true" if use_agent else "false"
        query_requests_total.labels(use_agent=agent_str).inc()
        query_duration_seconds.labels(use_agent=agent_str, phase="total").observe(duration)
        query_states_retrieved.observe(states_retrieved)
        query_confidence_score.observe(confidence)

    @staticmethod
    def record_embedding(model: str, duration: float, cache_hit: bool, cache_type: str = "redis"):
        """Record embedding metrics"""
        embedding_requests_total.labels(model=model).inc()
        if cache_hit:
            embedding_cache_hits.labels(cache_type=cache_type).inc()
        else:
            embedding_cache_misses.inc()
            embedding_duration_seconds.labels(model=model).observe(duration)

    @staticmethod
    def record_llm_request(
        model: str,
        operation: str,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        """Record LLM request metrics"""
        llm_requests_total.labels(model=model, operation=operation).inc()
        llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
        llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)
        llm_duration_seconds.labels(model=model, operation=operation).observe(duration)

    @staticmethod
    def record_llm_error(model: str, error_type: str):
        """Record LLM error"""
        llm_errors_total.labels(model=model, error_type=error_type).inc()

    @staticmethod
    def record_promotion(
        trigger: str,
        success: bool,
        duration: float,
        states_synthesized: int = 0,
        conflicts_detected: int = 0,
    ):
        """Record promotion metrics"""
        promotion_requests_total.labels(trigger=trigger).inc()
        if success:
            promotion_success_total.inc()
            promotion_states_synthesized.observe(states_synthesized)
            promotion_conflicts_detected.observe(conflicts_detected)
        else:
            promotion_failures_total.inc()
        promotion_duration_seconds.observe(duration)

    @staticmethod
    def update_cache_stats(cache_type: str, hit_ratio: float, size_bytes: int, keys_count: int):
        """Update cache statistics"""
        cache_hit_ratio.labels(cache_type=cache_type).set(hit_ratio)
        cache_size_bytes.set(size_bytes)
        cache_keys_total.labels(cache_type=cache_type).set(keys_count)

    @staticmethod
    def record_consolidation(level: str, duration: float, topics_processed: int):
        """Record consolidation metrics"""
        consolidation_runs_total.labels(level=level).inc()
        consolidation_topics_processed.labels(level=level).inc(topics_processed)
        consolidation_duration_seconds.labels(level=level).observe(duration)

    @staticmethod
    def update_memory_stats(total_states: int, total_topics: int, avg_versions: float):
        """Update memory statistics"""
        memory_states_current.set(total_states)
        memory_topics_current.set(total_topics)
        memory_versions_avg.set(avg_versions)

    @staticmethod
    def record_memory_state_created(source: str = "api"):
        """Record memory state creation"""
        memory_states_total.labels(source=source).inc()

    @staticmethod
    def update_graph_stats(nodes: int, edges: int):
        """Update graph statistics"""
        graph_nodes_total.set(nodes)
        graph_edges_total.set(edges)

    @staticmethod
    def record_graph_traversal(depth: int, duration: float):
        """Record graph traversal"""
        graph_traversal_duration_seconds.labels(depth=str(depth)).observe(duration)

    @staticmethod
    def update_db_connections(database: str, active: int):
        """Update database connection count"""
        db_connections_active.labels(database=database).set(active)

    @staticmethod
    def record_db_query(database: str, operation: str, duration: float):
        """Record database query"""
        db_query_duration_seconds.labels(database=database, operation=operation).observe(duration)

    @staticmethod
    def record_db_error(database: str, error_type: str):
        """Record database error"""
        db_errors_total.labels(database=database, error_type=error_type).inc()
