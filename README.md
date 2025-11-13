# TracingRAG

An enhanced Retrieval-Augmented Generation (RAG) system that combines temporal tracing, graph relationships, and agentic retrieval to provide intelligent, context-aware knowledge management.

## Key Features

### Core Capabilities
- **Temporal Tracing**: Track the evolution of knowledge over time with full history
- **Graph Relationships**: Connect related concepts and states
- **Agentic Retrieval**: Intelligent, multi-step retrieval strategies
- **Memory Promotion**: Synthesize new knowledge states from historical data
- **Time-Travel Queries**: Query knowledge as it existed at any point in time

### Human-Like Memory (What Makes TracingRAG Special)
- **Edge-Based Relevance**: Contextual importance via edge weights, NOT filtering
  - All states always accessible - nothing forgotten
  - Edge strength represents contextual relevance (0.0-1.0)
  - Low-strength edges: connections exist but ranked lower
  - High-strength edges: prioritized during graph traversal
  - Graph structure ensures even "distant" memories are discoverable
- **Importance Learning**: System learns what's important from access patterns
- **Working Memory**: Context-aware hot cache pre-loads related memories (<10ms queries)
- **Hierarchical Consolidation**: Auto-summarizes at daily/weekly/monthly levels (like human sleep)
- **Latest State Tracking**: Instant O(1) lookup for "what's the current status?" queries
- **Graph-Based Relevance**: Even old/low-strength memories found via edges to latest states
- **Storage Tiers**: Hot/warm/cold storage mimics human memory (working/active/archived)

### Scale & Performance
- **Instant Latest**: <10ms for current state queries
- **Working Set**: <10ms for active context (in-memory)
- **Full Search**: <100ms with caching and optimization
- **Massive Scale**: Handles millions of states with partitioning and sharding
- **Space Efficient**: 95% storage savings with diff-based versioning

## Architecture

TracingRAG uses a multi-layer architecture:
- **Storage Layer**: Qdrant (vectors), Neo4j (graphs), PostgreSQL (documents)
- **Core Services**: Memory management, graph operations, embeddings
- **Agentic Layer**: Query planning, memory promotion, retrieval orchestration
- **API Layer**: REST/GraphQL endpoints

See [DESIGN.md](DESIGN.md) for detailed architecture and design decisions.

## Tech Stack

- **Python 3.11+** with FastAPI
- **Qdrant** for vector storage
- **Neo4j** for graph database
- **PostgreSQL + TimescaleDB** for document storage
- **LangGraph** for agentic workflows
- **OpenRouter** for LLM access

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TracingRAG
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Start infrastructure services:
```bash
docker-compose up -d
```

6. Run database migrations:
```bash
poetry run alembic upgrade head
```

7. Start the API server:
```bash
poetry run uvicorn tracingrag.api.main:app --reload
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## REST API

TracingRAG provides a comprehensive REST API for all operations:

### Available Endpoints

**System**:
- `GET /` - API information
- `GET /health` - Health check
- `GET /metrics` - System metrics

**Memory Management**:
- `POST /api/v1/memories` - Create memory state
- `GET /api/v1/memories/{id}` - Get memory by ID
- `GET /api/v1/memories` - List memories (with pagination and filtering)
- `GET /api/v1/traces/{topic}` - Get version history for a topic

**Query/RAG**:
- `POST /api/v1/query` - Query the RAG system (supports both standard and agent-based retrieval)

**Promotion**:
- `POST /api/v1/promote` - Promote a memory state
- `GET /api/v1/promotion-candidates` - Get topics that are candidates for promotion

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer
- **ReDoc**: `http://localhost:8000/redoc` - API reference documentation
- **OpenAPI JSON**: `http://localhost:8000/openapi.json` - Machine-readable API spec

### Quick API Example

```bash
# Create a memory
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "project_alpha",
    "content": "Initial design for API authentication",
    "tags": ["design", "security"],
    "confidence": 0.95
  }'

# Query the RAG system
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the status of project alpha?",
    "use_agent": false
  }'

# Get promotion candidates
curl "http://localhost:8000/api/v1/promotion-candidates?limit=10&min_priority=7"
```

For complete API documentation, see [docs/API_GUIDE.md](docs/API_GUIDE.md).

## Project Structure

```
TracingRAG/
├── tracingrag/                 # Main package
│   ├── __init__.py
│   ├── core/                   # Core domain models and logic
│   │   ├── models/            # Data models
│   │   ├── services/          # Business logic
│   │   └── interfaces/        # Abstract interfaces
│   ├── storage/               # Storage layer
│   │   ├── vector/            # Qdrant integration
│   │   ├── graph/             # Neo4j integration
│   │   └── document/          # PostgreSQL integration
│   ├── agents/                # Agentic layer
│   │   ├── query_agent.py
│   │   ├── memory_agent.py
│   │   └── planning_agent.py
│   ├── api/                   # API layer
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas/
│   └── utils/                 # Utilities
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── docs/                      # Additional documentation
├── docker-compose.yml         # Local development infrastructure
├── Dockerfile                 # Application container
├── pyproject.toml            # Poetry configuration
├── .env.example              # Environment variables template
├── DESIGN.md                 # Detailed design document
└── README.md                 # This file
```

## Usage Examples

### Creating a Memory State

```python
from tracingrag.client import TracingRAGClient

client = TracingRAGClient("http://localhost:8000")

# Create initial memory state
state = client.create_memory(
    topic="project_alpha",
    content="Starting development of feature X with approach Y",
    tags=["project", "development"]
)
```

### Querying with Context

```python
# Query for relevant memories
results = client.query(
    query="What's the status of project alpha?",
    include_history=True,  # Include trace context
    include_related=True,  # Include graph connections
    depth=2  # Graph traversal depth
)

for result in results:
    print(f"Topic: {result.topic}")
    print(f"Content: {result.content}")
    print(f"Version: {result.version}")
    print(f"Related: {[r.topic for r in result.related_states]}")
```

### Promoting Memory

```python
# Promote memory to new state with synthesis
new_state = client.promote_memory(
    topic="project_alpha",
    reason="Bug discovered and fixed, feature complete"
)

# The system will:
# 1. Analyze trace history
# 2. Find related states (e.g., bug reports)
# 3. Synthesize new state with LLM
# 4. Create appropriate graph edges
```

### Time-Travel Query

```python
from datetime import datetime, timedelta

# What did we know about this topic 2 weeks ago?
past_state = client.query_at_time(
    topic="project_alpha",
    timestamp=datetime.now() - timedelta(weeks=2)
)
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Linting and Formatting

```bash
poetry run ruff check .
poetry run ruff format .
```

### Type Checking

```bash
poetry run mypy tracingrag
```

## Configuration

Key environment variables (see `.env.example`):

- `OPENROUTER_API_KEY`: OpenRouter API key for LLM access
- `QDRANT_URL`: Qdrant server URL
- `QDRANT_API_KEY`: Qdrant API key (if using cloud)
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `DATABASE_URL`: PostgreSQL connection URL
- `EMBEDDING_MODEL`: Model to use for embeddings (default: all-mpnet-base-v2)

## Roadmap

- [x] **Phase 1: Foundation** - Core data models, storage interfaces, and basic infrastructure
- [x] **Phase 2: Retrieval Services** - Semantic search, graph-enhanced retrieval, temporal queries, hybrid retrieval
- [x] **Phase 3: Graph Layer** - Edge management, relationship types, temporal validity, graph traversal
- [x] **Phase 4: Basic RAG** - Query processing, context building, LLM integration, response generation
- [x] **Phase 5: Agentic Layer** - Intelligent agents for query planning and memory management
- [x] **Phase 6: Memory Promotion** - Synthesis capabilities and knowledge consolidation
- [x] **Phase 7: Advanced Features** - Redis caching, hierarchical consolidation, performance optimization
- [x] **Phase 8: Production Ready** - Security, monitoring, CI/CD, Kubernetes deployment

**🎉 TracingRAG is now production-ready!**

## Production Deployment

TracingRAG is fully production-ready with:
- **Security**: JWT authentication, API key support, rate limiting, input validation
- **Monitoring**: Prometheus metrics (50+ metrics), structured logging, health checks
- **CI/CD**: Automated testing, linting, Docker builds, security scanning
- **Kubernetes**: Complete K8s manifests with autoscaling (HPA), ingress, TLS support
- **Performance**: Multi-stage Docker builds, caching layers, optimized resource allocation

See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for complete deployment instructions.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by:
- [Zep's Graphiti](https://www.getzep.com/) - Temporal knowledge graphs
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Graph-based RAG
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agentic workflows
