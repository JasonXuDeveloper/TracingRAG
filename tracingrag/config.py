"""Configuration management using Pydantic settings"""


from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "TracingRAG"
    app_version: str = "0.1.0"
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
    default_llm_model: str = "anthropic/claude-3.5-sonnet"
    fallback_llm_model: str = "openai/gpt-4-turbo"

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

    # Security
    secret_key: str = "your-secret-key-here-change-in-production"
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    allowed_hosts: list[str] = ["*"]

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
