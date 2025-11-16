"""Configuration management using Pydantic settings"""

from pathlib import Path as _Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Don't try to parse JSON for env vars - use validators instead
        env_parse_none_str="null",
    )

    # Application
    app_name: str = "TracingRAG"
    app_version: str = "0.2.0"
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = True

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # LLM Provider (OpenRouter)
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_llm_model: str = "openai/gpt-4o-mini"
    fallback_llm_model: str = "openai/gpt-4o-mini"
    analysis_model: str = "openai/gpt-4o-mini"  # For conflict detection, quality checks
    evaluation_model: str = "openai/gpt-4o-mini"  # For promotion evaluation
    query_analyzer_model: str = "openai/gpt-4o-mini"  # For query analysis
    planner_model: str = "openai/gpt-4o-mini"  # For agent query planning
    manager_model: str = "openai/gpt-4o-mini"  # For agent memory management

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # Qdrant (Vector Database)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "tracingrag_memories"
    qdrant_grpc_port: int = 6334
    qdrant_use_grpc: bool = False

    # Neo4j (Graph Database)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "tracingrag123"
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50

    # PostgreSQL (Document Store)
    database_url: str = "postgresql+asyncpg://tracingrag:tracingrag123@localhost:5432/tracingrag"
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_echo: bool = False

    # Redis (Caching)
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10
    cache_ttl: int = 3600
    cache_enabled: bool = True

    # Retrieval Configuration
    default_retrieval_limit: int = 10
    max_retrieval_limit: int = 100
    default_graph_depth: int = 2
    max_graph_depth: int = 5
    enable_hybrid_search: bool = True

    # Memory Promotion
    promotion_confidence_threshold: float = 0.7
    auto_promotion_enabled: bool = False
    max_trace_history_context: int = 10

    # Relationship Management (always enabled via RelationshipManager)
    relationship_update_similarity_threshold: float = (
        0.45  # Min similarity for candidates (higher = more selective)
    )
    relationship_update_llm_batch_size: int = 30  # Batch size for LLM processing

    # Cascading Evolution (single-level evolution of related topics when new memory is created)
    # Note: Only evolves directly related topics (no multi-level cascade)
    # Relationship propagation is handled by RelationshipManager
    enable_cascading_evolution: bool = True
    cascading_evolution_similarity_threshold: float = (
        0.45  # Min similarity for evolution candidates (higher = more selective)
    )
    cascading_evolution_batch_size: int = (
        5  # Process this many topics per LLM call (smaller = less token usage, more parallel calls)
    )

    # Agent Configuration
    agent_max_iterations: int = 10
    agent_timeout_seconds: int = 60
    enable_agent_memory: bool = True
    agent_thinking_budget: int = 1000

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9100
    enable_tracing: bool = False
    jaeger_endpoint: str = "http://localhost:14268/api/traces"

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100

    # LLM Retry Configuration
    llm_max_retries: int = 5  # Maximum number of retry attempts for LLM calls
    llm_retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    llm_retry_max_delay: float = 60.0  # Maximum delay in seconds between retries
    fallback_llm_max_retries: int = 3  # Maximum retries for fallback model after primary fails

    # Neo4j Snapshot Backup
    neo4j_snapshot_enabled: bool = Field(
        default=True, description="Enable Neo4j graph snapshots before mutations"
    )
    neo4j_snapshot_path: str = Field(
        default="./data/neo4j_snapshots", description="Directory to store Neo4j graph snapshots"
    )
    neo4j_snapshot_max_count: int = Field(
        default=100, description="Maximum number of snapshots to keep (oldest will be deleted)"
    )
    neo4j_snapshot_compression: bool = Field(
        default=True, description="Compress snapshots with gzip"
    )

    # Security
    secret_key: str = "your-secret-key-here-change-in-production"
    allowed_origins: str | list[str] = Field(default="http://localhost:3000,http://localhost:8000")
    allowed_hosts: str | list[str] = Field(default="*")

    @field_validator("allowed_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: Any) -> list[str]:
        """Parse comma-separated string into list"""
        if isinstance(v, str):
            if not v.strip():  # Empty string
                return []
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return v
        return []

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() in ("development", "dev", "local")

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() in ("production", "prod")


# Global settings instance
settings = Settings()

# HACK: Force model settings from .env file to override environment variables
# This is needed because Claude Code parent process sets environment variables
# that override .env file values
_env_file = _Path(".env")
if _env_file.exists():
    _model_overrides = {}
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _key, _value = _line.split("=", 1)
                _key = _key.strip()
                _value = _value.strip()
                # Override model settings
                if _key in [
                    "DEFAULT_LLM_MODEL",
                    "FALLBACK_LLM_MODEL",
                    "ANALYSIS_MODEL",
                    "EVALUATION_MODEL",
                    "QUERY_ANALYZER_MODEL",
                    "PLANNER_MODEL",
                    "MANAGER_MODEL",
                ]:
                    setattr(settings, _key.lower(), _value)
