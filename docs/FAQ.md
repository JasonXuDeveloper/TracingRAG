
# Frequently Asked Questions (FAQ)

## General Questions

### What is TracingRAG?

TracingRAG is an enhanced Retrieval-Augmented Generation (RAG) system that combines temporal tracing, graph relationships, and agentic retrieval. Unlike traditional RAG systems that treat each document as isolated, TracingRAG maintains:
- **Temporal history**: Complete evolution of knowledge over time
- **Graph relationships**: Connections between related concepts
- **Agentic intelligence**: LLM-powered query planning and synthesis

### When should I use TracingRAG instead of traditional RAG?

Use TracingRAG when you need:
- **Version history**: Track how information changes over time
- **Relationship tracking**: Understand how concepts connect
- **Context switching**: Quickly resume work on different topics
- **Knowledge evolution**: Synthesize new insights from historical data
- **Long-term memory**: Remember context across sessions (days, weeks, months)

**Examples**: Software project documentation, NPC memory in games, novel writing, research notes, customer relationship tracking.

### How is TracingRAG different from vector databases like Pinecone or Weaviate?

Vector databases store and search embeddings. TracingRAG is a complete system built ON TOP of vector databases that adds:
- **Temporal tracing**: Version history with parent-child relationships
- **Graph layer**: Neo4j for relationship tracking
- **Relational storage**: PostgreSQL for structured data
- **Agentic retrieval**: LLM-based query planning
- **Memory synthesis**: Automatic consolidation and promotion

Think of it as: Vector DB + Graph DB + Temporal DB + LLM Intelligence = TracingRAG

## Setup & Installation

### Do I need all the services (PostgreSQL, Neo4j, Qdrant, Redis)?

**Required:**
- ✅ PostgreSQL - Document storage and temporal queries
- ✅ Neo4j - Graph relationships
- ✅ Qdrant - Vector search
- ✅ OpenRouter API key - LLM access for query analysis, synthesis, agents

**Optional:**
- ⭐ Redis - Caching layer (highly recommended for production)
  - Without Redis: In-memory cache only (1000 items max)
  - With Redis: Full caching for embeddings, queries, working memory

### Can I run TracingRAG without Docker?

Yes, but you'll need to install and configure services manually:
1. PostgreSQL 14+ with TimescaleDB extension
2. Neo4j 5.x Community or Enterprise
3. Qdrant 1.7+
4. (Optional) Redis 7+

We strongly recommend using Docker Compose for simplicity.

### Why is my setup failing?

Run the verification script:
```bash
poetry run python scripts/verify_setup.py
```

Common issues:
1. **Neo4j not ready**: Wait 30-60 seconds after `docker-compose up`
2. **Port conflicts**: Change ports in `docker-compose.yml` if 5432, 6333, 6379, 7687 are taken
3. **Missing API key**: Set `OPENROUTER_API_KEY` in `.env`
4. **Wrong Python version**: Requires Python 3.11+

### How much memory/CPU do I need?

**Minimum (Development):**
- 8GB RAM
- 4 CPU cores
- 10GB disk space

**Recommended (Production):**
- 16GB+ RAM
- 8+ CPU cores
- 50GB+ disk space (depends on data size)

Services resource usage:
- PostgreSQL: ~500MB RAM
- Neo4j: ~1-2GB RAM
- Qdrant: ~500MB RAM
- Redis: ~100MB RAM
- TracingRAG API: ~1-2GB RAM (for embedding model)

## Usage & Features

### How do I create a memory?

**Option 1: Python Client**
```python
from tracingrag.client import TracingRAGClient

client = TracingRAGClient("http://localhost:8000")
state = client.create_memory(
    topic="project_alpha",
    content="Initial design completed",
    tags=["project", "milestone"]
)
```

**Option 2: REST API**
```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "project_alpha",
    "content": "Initial design completed",
    "tags": ["project", "milestone"]
  }'
```

**Option 3: Swagger UI**
Navigate to `http://localhost:8000/docs`

### What's the difference between a "memory" and a "state"?

- **Memory** = A topic's complete history (e.g., "project_alpha")
- **State** = A specific version in that history (e.g., "project_alpha v3")

Each time you update a memory, you create a new state. TracingRAG tracks the entire evolution.

### When should I use "promote" vs "create"?

- **Create**: Add a new version manually
  ```python
  client.create_memory(topic="...", content="...", parent_state_id="...")
  ```

- **Promote**: Let the LLM synthesize a new state from history
  ```python
  client.promote_memory(topic="...", reason="Major milestone reached")
  ```

Use promotion when you want the system to intelligently combine multiple sources of information.

### How does caching work?

**Without Redis** (optional):
- In-memory cache: 1000 embeddings max
- Lost on API restart
- Good for development

**With Redis** (recommended):
- Persistent caching for:
  - Embeddings (7-day TTL)
  - Query results (1-hour TTL)
  - Working memory (30-min TTL)
  - Latest states (24-hour TTL)
- Survives API restarts
- Shared across multiple API instances

To enable Redis:
1. Ensure Redis is running: `docker-compose ps`
2. Set in `.env`: `CACHE_ENABLED=true` (default)
3. Configure: `REDIS_URL=redis://localhost:6379/0`

### What does "agent-based query" mean?

**Standard query** (fast):
- Direct semantic search
- Simple retrieval
- ~100-200ms response time

**Agent-based query** (intelligent):
- LLM analyzes query intent
- Plans multi-step retrieval strategy
- Executes plan with replanning if needed
- Synthesizes answer from results
- ~2-5s response time

Use agent queries for complex questions that require reasoning.

### Can I use my own LLM instead of OpenRouter?

Yes! TracingRAG uses OpenAI-compatible APIs. You can use:
- **OpenRouter** (default) - Access to 100+ models
- **OpenAI** - Direct OpenAI API
- **Local models** - Any OpenAI-compatible endpoint (LM Studio, Ollama with OpenAI plugin, vLLM)

Configure in `.env`:
```env
# For OpenAI
OPENROUTER_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=sk-...
DEFAULT_LLM_MODEL=gpt-4-turbo

# For local LM Studio
OPENROUTER_BASE_URL=http://localhost:1234/v1
OPENROUTER_API_KEY=lm-studio
DEFAULT_LLM_MODEL=local-model
```

## Performance & Scaling

### How fast are queries?

**Latest state lookup**: <10ms (PostgreSQL materialized view)
**Cached queries**: <10ms (Redis)
**Semantic search**: 50-200ms (Qdrant)
**Agent queries**: 2-5s (includes LLM calls)

### How many memories can TracingRAG handle?

Tested with:
- **100K+ states** - Good performance
- **1M+ states** - Requires database tuning
- **10M+ states** - Requires sharding (not yet automated)

### How do I scale TracingRAG for production?

1. **Horizontal scaling**: Run multiple API instances behind load balancer
2. **Database optimization**:
   - Increase PostgreSQL connection pool
   - Add read replicas
   - Enable TimescaleDB partitioning for large datasets
3. **Caching**: Use Redis for shared cache
4. **Resource allocation**: See Kubernetes HPA in `k8s/hpa.yaml`

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details.

### Can I disable certain features to save resources?

Yes! Configure in `.env`:

```env
# Disable Redis caching (use in-memory only)
REDIS_URL=  # Leave empty

# Disable automatic promotion
AUTO_PROMOTION_ENABLED=false

# Use smaller embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # 384 dim vs 768
```

## Development

### How do I contribute?

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

### How do I add a new storage backend?

1. Create interface in `tracingrag/core/models/`
2. Implement in `tracingrag/storage/`
3. Add to dependency injection in services
4. Write tests
5. Update documentation

### Can I extend the graph relationship types?

Yes! Edit `tracingrag/core/models/graph.py`:

```python
class RelationshipType(str, Enum):
    # Add your custom types
    CUSTOM_RELATION = "custom_relation"
```

Relationships are stored in Neo4j and fully queryable.

## Troubleshooting

### "Connection refused" errors

Check services are running:
```bash
docker-compose ps  # All should be "Up"
docker-compose logs [service-name]  # Check for errors
```

### Neo4j initialization fails

Wait longer - Neo4j can take 30-60 seconds to fully start:
```bash
docker-compose logs -f neo4j
# Wait for: "Remote interface available at http://localhost:7474/"
```

### Out of memory errors

Reduce resource usage:
1. Use smaller embedding model (see above)
2. Reduce database connection pools:
   ```env
   DATABASE_POOL_SIZE=10  # Default: 20
   NEO4J_MAX_CONNECTION_POOL_SIZE=20  # Default: 50
   ```
3. Limit concurrent requests (use reverse proxy with rate limiting)

### Queries are slow

1. **Check cache**: Ensure Redis is running and connected
2. **Warm cache**: Pre-load frequent queries
3. **Optimize indexes**: Neo4j and PostgreSQL indexes should auto-create
4. **Use latest-only queries**: Set `include_history=False` for faster queries
5. **Reduce graph depth**: Set `depth=1` instead of default `depth=2`

### API responses are in wrong language

TracingRAG uses LLM-based query analysis which is language-agnostic. However:
- Query responses depend on the LLM model's capabilities
- Memory content is stored as-is in your language
- The system does NOT translate - it preserves original language

## Security

### How do I secure TracingRAG?

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#security) for:
- JWT authentication
- API key setup
- Rate limiting
- Input validation
- TLS/SSL configuration

### Is my data encrypted?

**In transit**: Configure TLS for all services
**At rest**: Depends on your database configuration
- PostgreSQL: Enable encryption at rest
- Neo4j: Enterprise edition supports encryption
- Qdrant: Configure TLS and auth

### Can I use TracingRAG in production?

Yes! TracingRAG includes production-ready features:
- ✅ Authentication (JWT + API keys)
- ✅ Rate limiting
- ✅ Security headers
- ✅ Input validation
- ✅ Prometheus metrics
- ✅ Structured logging
- ✅ Health checks
- ✅ Kubernetes manifests
- ✅ CI/CD pipeline

See Phase 8 implementation for details.

## Cost & Pricing

### How much does it cost to run TracingRAG?

**Infrastructure** (self-hosted):
- Development: $0 (local Docker)
- Production: $50-500/month depending on scale (cloud VMs, databases)

**LLM API costs**:
- Query analysis: ~$0.001-0.01 per query (depends on model)
- Memory promotion: ~$0.01-0.10 per promotion (depends on complexity)
- Agent queries: ~$0.01-0.05 per query

**Cost optimization**:
- Use cheaper models for query analysis (e.g., DeepSeek R1 free tier)
- Cache aggressively (Redis)
- Batch promotions
- Use agent queries only when needed

### Can I use free LLM APIs?

Yes! Configure DeepSeek R1 (free tier) or use local models:

```env
# DeepSeek R1 (free on OpenRouter)
DEFAULT_LLM_MODEL=deepseek/deepseek-r1-distill-llama-70b

# Local Ollama (free, but slower)
OPENROUTER_BASE_URL=http://localhost:11434/v1
DEFAULT_LLM_MODEL=llama3.1:70b
```

## Getting Help

### Where can I get help?

1. **Documentation**: Check [docs/](.) folder
2. **Examples**: Run example scripts in `examples/`
3. **GitHub Issues**: [Report bugs or request features](https://github.com/JasonXuDeveloper/TracingRAG/issues)
4. **Verification script**: `poetry run python scripts/verify_setup.py`

### How do I report a bug?

1. Check [GitHub Issues](https://github.com/JasonXuDeveloper/TracingRAG/issues) for existing reports
2. Run `poetry run python scripts/verify_setup.py` and include output
3. Provide:
   - TracingRAG version
   - Python version
   - OS/platform
   - Error messages and logs
   - Steps to reproduce

### Feature requests?

Open a [GitHub Issue](https://github.com/JasonXuDeveloper/TracingRAG/issues) with:
- Use case description
- Proposed solution
- Why existing features don't work

We're especially interested in:
- New use cases
- Storage backend integrations
- Performance optimizations
- Production deployment experiences
