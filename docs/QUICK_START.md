# Quick Start Guide

This guide will help you get TracingRAG up and running in under 10 minutes.

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- 4GB+ RAM available for services

## Step 1: Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd TracingRAG

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and configure
nano .env  # or use your preferred editor
```

**Required configuration:**
```env
OPENROUTER_API_KEY=sk-or-v1-...
```

**Embedding configuration (choose one):**

*Option 1: Local embeddings (default, free)*
```env
# English only (default, no changes needed)
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# OR for multilingual support (50+ languages)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

*Option 2: OpenAI embeddings (best multilingual, API costs)*
```env
OPENAI_API_KEY=sk-...  # Your OpenAI API key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # 100+ languages
```

**Note:** If `OPENAI_API_KEY` is set, TracingRAG will use OpenAI embeddings. Otherwise, it uses the local model specified in `EMBEDDING_MODEL`.

## Step 3: Start Infrastructure Services

```bash
# Start all services (Qdrant, Neo4j, PostgreSQL, Redis)
docker-compose up -d

# Wait for services to initialize (30-60 seconds)
# You can watch the logs:
docker-compose logs -f
```

**Required services:**
- ✅ `tracingrag-postgres` on port 5432 - Document storage
- ✅ `tracingrag-neo4j` on ports 7474, 7687 - Graph database
- ✅ `tracingrag-qdrant` on ports 6333, 6334 - Vector search

**Optional (but recommended) services:**
- ⭐ `tracingrag-redis` on port 6379 - Caching layer
  - **Without Redis**: Only in-memory caching (1000 items max)
  - **With Redis**: Full caching for embeddings, queries, working memory

Verify services are running:
```bash
docker-compose ps
```

## Step 4: Initialize Databases

```bash
# Run PostgreSQL migrations
poetry run alembic upgrade head

# Initialize Neo4j schema (constraints and indexes)
poetry run python scripts/init_neo4j.py

# Initialize Qdrant collection
poetry run python scripts/init_qdrant.py
```

## Step 5: Verify Setup

**Important:** Run the verification script to ensure everything is configured correctly:

```bash
poetry run python scripts/verify_setup.py
```

This will check:
- ✅ PostgreSQL connection and migrations
- ✅ Neo4j connection and schema
- ✅ Qdrant collection setup
- ✅ OpenRouter API key configuration
- ✅ Embedding model loading
- ⭐ Redis caching (optional)

**Expected output:**
```
✅ All required services are working!
✅ Optional Redis caching is enabled  # or warning if Redis unavailable
🚀 You're ready to start TracingRAG!
```

## Step 6: Start the API

```bash
# Development mode (with auto-reload)
poetry run uvicorn tracingrag.api.main:app --reload

# Production mode
poetry run uvicorn tracingrag.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## Step 7: Test the Installation

### Option A: Using the Web UI

Navigate to `http://localhost:8000/docs` and use the Swagger UI to test endpoints.

### Option B: Using the Python Client

The Python client provides a convenient interface to the REST API:

```python
from tracingrag.client import TracingRAGClient

# Initialize client
client = TracingRAGClient("http://localhost:8000")

# Create your first memory
state = client.create_memory(
    topic="test_memory",
    content="This is my first TracingRAG memory!",
    tags=["test", "first"]
)

print(f"Created memory: {state.id}")

# Query it back
results = client.query("first TracingRAG memory")
print(f"Found {len(results)} results")
print(f"Content: {results[0].state.content}")
```

### Option C: Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Create a memory
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "test_memory",
    "content": "This is my first TracingRAG memory!",
    "tags": ["test", "first"]
  }'

# Query memories
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "first TracingRAG memory",
    "limit": 10
  }'
```

## Next Steps

### Try the Examples

```bash
# Example 1: Project memory
poetry run python examples/01_project_memory.py

# Example 2: NPC simulation
poetry run python examples/02_npc_simulation.py

# Example 3: Novel writing
poetry run python examples/03_novel_writing.py
```

### Explore the UI

- **Neo4j Browser**: `http://localhost:7474`
  - Username: `neo4j`
  - Password: `tracingrag123`
  - Explore your memory graph visually!

- **Qdrant Dashboard**: `http://localhost:6333/dashboard`
  - View vector collections and points

- **Grafana**: `http://localhost:3000`
  - Username: `admin`
  - Password: `tracingrag123`
  - Monitor system performance

### Read the Documentation

- [Design Document](../DESIGN.md) - Architecture and design decisions
- [Use Cases](USE_CASES.md) - Detailed use case examples
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Development Guide](DEVELOPMENT.md) - Contributing and development

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Restart specific service
docker-compose restart neo4j

# Reset everything
docker-compose down -v
docker-compose up -d
```

### Can't connect to Neo4j

```bash
# Wait for Neo4j to fully start (can take 30-60 seconds)
docker-compose logs -f neo4j

# Look for: "Remote interface available at http://localhost:7474/"
```

### Import errors in Python

```bash
# Ensure you're in the right directory
cd /path/to/TracingRAG

# Reinstall dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Port conflicts

If ports 6333, 7474, 7687, 5432, or 6379 are already in use, edit `docker-compose.yml`:

```yaml
ports:
  - "16333:6333"  # Change first number to any available port
```

Then restart services:
```bash
docker-compose down
docker-compose up -d
```

And update `.env`:
```env
QDRANT_URL=http://localhost:16333
```

## Performance Tips

### For Development

- Use smaller embedding model: `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- Reduce cache TTL: `CACHE_TTL=300`
- Enable debug logging: `LOG_LEVEL=DEBUG`

### For Production

- Use larger embedding model: `EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2`
- Increase connection pools:
  ```env
  DATABASE_POOL_SIZE=50
  NEO4J_MAX_CONNECTION_POOL_SIZE=100
  ```
- Enable caching: `CACHE_ENABLED=true`
- Set proper logging: `LOG_LEVEL=INFO`

## What's Next?

Now that TracingRAG is running, you can:

1. **Build a project knowledge base**: Track your codebase evolution
2. **Create NPC memories**: Simulate characters with persistent memory
3. **Organize your novel**: Track characters, plots, and world-building
4. **Integrate with your app**: Use the Python client or REST API

Check out the [examples/](../examples/) directory for inspiration!

## Getting Help

- Read the [FAQ](FAQ.md)
- Check [GitHub Issues](https://github.com/yourusername/TracingRAG/issues)
- Join the [Discord community](https://discord.gg/tracingrag)

Happy tracing! 🚀
