# Development Guide

This guide is for developers who want to contribute to TracingRAG or extend it for their own use cases.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Testing](#testing)
- [Adding Features](#adding-features)
- [Contributing](#contributing)

## Development Setup

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose
- Git

### Initial Setup

```bash
# Clone repository
git clone https://github.com/JasonXuDeveloper/TracingRAG.git
cd TracingRAG

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies (including dev dependencies)
poetry install

# Set up pre-commit hooks (optional but recommended)
poetry run pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Start infrastructure
docker-compose up -d

# Initialize databases
poetry run alembic upgrade head
poetry run python scripts/init_neo4j.py
poetry run python scripts/init_qdrant.py

# Verify setup
poetry run python scripts/verify_setup.py
```

### Running in Development Mode

```bash
# Start API with auto-reload
poetry run uvicorn tracingrag.api.main:app --reload --log-level debug

# Or use the convenience script (if created)
./scripts/dev-server.sh
```

## Project Structure

```
TracingRAG/
├── tracingrag/                     # Main package
│   ├── __init__.py
│   ├── client.py                   # Python client library
│   ├── core/                       # Core domain models
│   │   └── models/                 # Data models (Pydantic)
│   ├── storage/                    # Storage layer
│   │   ├── database.py             # PostgreSQL/SQLAlchemy
│   │   ├── qdrant.py               # Qdrant vector client
│   │   └── neo4j_client.py         # Neo4j graph client
│   ├── services/                   # Business logic
│   │   ├── memory.py               # Memory state management
│   │   ├── retrieval.py            # Retrieval strategies
│   │   ├── rag.py                  # RAG pipeline
│   │   ├── graph.py                # Graph operations
│   │   ├── promotion.py            # Memory promotion
│   │   ├── consolidation.py        # Hierarchical consolidation
│   │   ├── embedding.py            # Embedding generation
│   │   ├── cache.py                # Redis caching
│   │   ├── context.py              # Context building
│   │   ├── llm.py                  # LLM integration
│   │   ├── query_analyzer.py       # Query analysis
│   │   ├── metrics.py              # Prometheus metrics
│   │   ├── logging.py              # Structured logging
│   │   └── security.py             # Security utilities
│   ├── agents/                     # Agentic layer
│   │   ├── service.py              # Agent orchestration
│   │   ├── query_planner.py        # Query planning
│   │   ├── memory_manager.py       # Memory management
│   │   ├── tools.py                # Agent tools
│   │   └── models.py               # Agent data models
│   └── api/                        # API layer
│       ├── main.py                 # FastAPI application
│       ├── schemas.py              # API request/response models
│       └── security.py             # API security middleware
├── tests/                          # Test suite
│   ├── test_memory.py
│   ├── test_retrieval.py
│   ├── test_rag.py
│   ├── test_promotion.py
│   └── test_api.py
├── scripts/                        # Utility scripts
│   ├── init_neo4j.py
│   ├── init_qdrant.py
│   └── verify_setup.py
├── examples/                       # Example usage
│   ├── 01_project_memory.py
│   ├── 02_npc_simulation.py
│   └── 03_novel_writing.py
├── docs/                           # Documentation
├── k8s/                            # Kubernetes manifests
├── .github/workflows/              # CI/CD
│   └── ci.yml
├── alembic/                        # Database migrations
├── docker-compose.yml
├── Dockerfile.prod
├── pyproject.toml
└── README.md
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for sorting
- **Type hints**: Required for public functions
- **Docstrings**: Google style

### Linting and Formatting

```bash
# Format code with Ruff
poetry run ruff format .

# Lint code
poetry run ruff check .

# Fix auto-fixable issues
poetry run ruff check --fix .

# Type checking (optional)
poetry run mypy tracingrag
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format/lint before commits:

```bash
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=tracingrag --cov-report=html

# Run specific test file
poetry run pytest tests/test_memory.py

# Run specific test
poetry run pytest tests/test_memory.py::test_create_memory

# Run with verbose output
poetry run pytest -v

# Run and stop at first failure
poetry run pytest -x
```

### Writing Tests

```python
import pytest
from tracingrag.services.memory import MemoryService

@pytest.mark.asyncio
async def test_create_memory():
    """Test memory creation"""
    # Arrange
    service = MemoryService()

    # Act
    state = await service.create_memory_state(
        topic="test_topic",
        content="Test content",
    )

    # Assert
    assert state.id is not None
    assert state.topic == "test_topic"
    assert state.version == 1
```

### Test Structure

- Use `pytest` fixtures for setup/teardown
- Mark async tests with `@pytest.mark.asyncio`
- Use meaningful test names: `test_<action>_<expected_result>`
- Follow Arrange-Act-Assert pattern
- Mock external services (LLM calls, etc.)

## Adding Features

### Adding a New Service

1. **Create the service file**: `tracingrag/services/my_service.py`

```python
"""My new service"""

from tracingrag.core.models.memory import MemoryState

class MyService:
    """My service description"""

    def __init__(self, dependency: SomeDependency):
        self.dependency = dependency

    async def do_something(self, param: str) -> MemoryState:
        """Do something useful

        Args:
            param: Parameter description

        Returns:
            MemoryState object
        """
        # Implementation
        pass

# Singleton pattern
_instance = None

def get_my_service() -> MyService:
    """Get MyService singleton"""
    global _instance
    if _instance is None:
        _instance = MyService(dependency=get_dependency())
    return _instance
```

2. **Add tests**: `tests/test_my_service.py`

3. **Update API** (if needed): Add endpoints in `tracingrag/api/main.py`

4. **Document**: Add usage examples to relevant docs

### Adding a New Storage Backend

1. **Define interface** (if needed): `tracingrag/core/models/`

2. **Implement client**: `tracingrag/storage/my_backend.py`

```python
class MyBackendClient:
    """Client for MyBackend"""

    def __init__(self, url: str):
        self.url = url
        self._client = None

    async def connect(self):
        """Connect to backend"""
        pass

    async def close(self):
        """Close connection"""
        pass

    async def my_operation(self, data):
        """Perform operation"""
        pass
```

3. **Add configuration**: Update `.env.example` and settings

4. **Write integration tests**

5. **Update initialization scripts**: Create `scripts/init_my_backend.py`

### Adding a New API Endpoint

1. **Define request/response schemas**: `tracingrag/api/schemas.py`

```python
class MyRequest(BaseModel):
    """My request model"""
    param1: str
    param2: int = 10

class MyResponse(BaseModel):
    """My response model"""
    result: str
    metadata: dict = {}
```

2. **Add endpoint**: `tracingrag/api/main.py`

```python
@app.post("/api/v1/my-endpoint", response_model=MyResponse, tags=["MyFeature"])
async def my_endpoint(request: MyRequest):
    """My endpoint description

    This endpoint does something useful.
    """
    try:
        service = get_my_service()
        result = await service.do_something(request.param1)

        return MyResponse(
            result=str(result),
            metadata={"processed": True}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

3. **Add tests**: `tests/test_api.py`

4. **Update API docs**: The endpoint will appear automatically in Swagger UI

## Contributing

### Git Workflow

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TracingRAG.git
   cd TracingRAG
   git remote add upstream https://github.com/JasonXuDeveloper/TracingRAG.git
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

4. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "feat: add my new feature

   - Detailed description of changes
   - Why this change is needed
   - Any breaking changes"
   ```

5. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Create a Pull Request** on GitHub

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(api): add batch memory creation endpoint

fix(promotion): resolve conflict detection edge case

docs(readme): update installation instructions

refactor(retrieval): optimize graph traversal algorithm
```

### Pull Request Guidelines

**Before submitting**:
- ✅ All tests pass: `poetry run pytest`
- ✅ Code is formatted: `poetry run ruff format .`
- ✅ No linting errors: `poetry run ruff check .`
- ✅ Documentation updated (if applicable)
- ✅ Examples added (for new features)

**PR description should include**:
- What changes were made
- Why these changes are needed
- How to test the changes
- Screenshots (for UI changes)
- Breaking changes (if any)

### Code Review Process

1. Maintainers will review your PR
2. Address feedback by pushing new commits
3. Once approved, maintainers will merge

## Debugging

### Enable Debug Logging

```env
# .env
LOG_LEVEL=DEBUG
```

### Debug with VSCode

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "tracingrag.api.main:app",
                "--reload",
                "--log-level",
                "debug"
            ],
            "jinja": true,
            "justMyCode": false
        }
    ]
}
```

### Common Debugging Commands

```bash
# Check service logs
docker-compose logs -f [service-name]

# Enter PostgreSQL
docker-compose exec postgres psql -U tracingrag

# Enter Neo4j Cypher shell
docker-compose exec neo4j cypher-shell -u neo4j -p tracingrag123

# Enter Redis CLI
docker-compose exec redis redis-cli

# Check Qdrant collections
curl http://localhost:6333/collections
```

## Performance Profiling

### Profile API Endpoints

```python
import cProfile
import pstats

async def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here
    result = await my_function()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```bash
# Install memory profiler
poetry add --dev memory-profiler

# Profile script
poetry run python -m memory_profiler my_script.py
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions will build and publish Docker image
6. Create GitHub release with changelog

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/JasonXuDeveloper/TracingRAG/issues)
- **Discussions**: [Ask questions](https://github.com/JasonXuDeveloper/TracingRAG/discussions)
- **Documentation**: Check [docs/](.) folder

## License

TracingRAG is MIT licensed. By contributing, you agree to license your contributions under the same license.
