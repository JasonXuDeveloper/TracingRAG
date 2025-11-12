"""Pytest configuration and fixtures for TracingRAG tests"""

import asyncio
import os
from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set test environment variables before importing config
os.environ["OPENROUTER_API_KEY"] = "test_key_for_testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from tracingrag.storage.database import Base
from tracingrag.storage.models import MemoryStateDB, TopicLatestStateDB, TraceDB


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


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
    """Create a database session for testing"""
    async_session_factory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        await session.rollback()


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
