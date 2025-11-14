"""Pytest configuration and fixtures for TracingRAG tests"""

import os
from collections.abc import AsyncGenerator

# Set test environment variables FIRST, before any imports
os.environ["OPENROUTER_API_KEY"] = "test_key_for_testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
# Force disable OpenAI embeddings for tests - use local model instead
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = ""  # Set to empty string to ensure it's disabled

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tracingrag.storage.database import Base, get_engine
from tracingrag.storage.models import MemoryStateDB


@pytest.fixture(scope="session", autouse=True)
async def initialize_global_db():
    """Initialize the global database for all tests"""
    # This ensures the global engine is initialized for tests that use get_session()
    global_engine = get_engine()
    async with global_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await global_engine.dispose()


@pytest.fixture(autouse=True)
async def cleanup_global_db():
    """Clean up global database after each test to prevent state pollution"""
    yield  # Let the test run first

    # Clean up after the test
    global_engine = get_engine()
    async with global_engine.begin() as conn:
        # Delete all data from tables to prevent test pollution
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture(scope="session")
async def test_db_engine():
    """Create a test database engine"""
    # Use in-memory SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing with automatic rollback"""
    connection = await test_db_engine.connect()
    transaction = await connection.begin()

    async_session_factory = async_sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        # Always rollback the transaction to isolate tests
        await transaction.rollback()
        await connection.close()


@pytest.fixture
def sample_memory_data():
    """Sample data for memory state testing"""
    return {
        "topic": "test_character",
        "content": "John is a brave knight who protects the kingdom.",
        "metadata": {"domain": "fantasy", "chapter": 1},
        "tags": ["character", "protagonist"],
        "confidence": 0.95,
        "entity_type": "Character",
    }


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing"""
    # 768-dimensional vector (matching sentence-transformers default)
    return [0.1] * 768


@pytest.fixture
async def sample_memory_state(db_session, sample_memory_data, sample_embedding):
    """Create a sample memory state in the database"""
    from datetime import datetime
    from uuid import uuid4

    state = MemoryStateDB(
        id=uuid4(),
        topic=sample_memory_data["topic"],
        content=sample_memory_data["content"],
        version=1,
        timestamp=datetime.utcnow(),
        embedding=sample_embedding,
        custom_metadata=sample_memory_data["metadata"],
        tags=sample_memory_data["tags"],
        confidence=sample_memory_data["confidence"],
        entity_type=sample_memory_data["entity_type"],
    )

    db_session.add(state)
    await db_session.commit()
    await db_session.refresh(state)

    return state
