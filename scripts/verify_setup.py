#!/usr/bin/env python3
"""Verify that all TracingRAG services are properly configured and accessible"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def check_postgres():
    """Check PostgreSQL/TimescaleDB connection"""
    print("📦 Checking PostgreSQL...")
    try:
        from tracingrag.storage.database import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute("SELECT 1")
            await result.fetchone()
        print("   ✅ PostgreSQL: Connected")
        return True
    except Exception as e:
        print(f"   ❌ PostgreSQL: {e}")
        return False


async def check_neo4j():
    """Check Neo4j connection"""
    print("🕸️  Checking Neo4j...")
    try:
        from tracingrag.storage.neo4j_client import Neo4jClient

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "tracingrag123")

        client = Neo4jClient(uri=uri, username=username, password=password)
        stats = await client.get_graph_stats()
        await client.close()
        print(f"   ✅ Neo4j: Connected ({stats.node_count} nodes, {stats.edge_count} edges)")
        return True
    except Exception as e:
        print(f"   ❌ Neo4j: {e}")
        return False


async def check_qdrant():
    """Check Qdrant connection"""
    print("🔍 Checking Qdrant...")
    try:
        from tracingrag.storage.qdrant import QdrantClient

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = os.getenv("QDRANT_COLLECTION_NAME", "tracingrag_memories")

        client = QdrantClient(url=url, collection_name=collection)
        info = await client.get_collection_info()
        await client.close()
        print(f"   ✅ Qdrant: Connected ({info.get('points_count', 0)} points)")
        return True
    except Exception as e:
        print(f"   ❌ Qdrant: {e}")
        return False


async def check_redis():
    """Check Redis connection"""
    print("💾 Checking Redis...")
    try:
        from tracingrag.services.cache import get_cache_service

        cache = get_cache_service()
        # Try a simple operation
        await cache.set("_test_key", "test", ttl=10)
        value = await cache.get("_test_key")
        await cache.delete("_test_key")
        await cache.close()

        if value == "test":
            print("   ✅ Redis: Connected (caching enabled)")
            return True
        else:
            print("   ⚠️  Redis: Connected but caching not working properly")
            return False
    except Exception as e:
        print(f"   ⚠️  Redis: {e}")
        print("   ℹ️  Redis is OPTIONAL - in-memory caching will be used as fallback")
        return None  # None means optional service unavailable


async def check_llm():
    """Check LLM API configuration"""
    print("🤖 Checking LLM configuration...")
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key == "your_openrouter_api_key_here":
        print("   ❌ OPENROUTER_API_KEY not configured in .env")
        return False

    print("   ✅ OPENROUTER_API_KEY: Configured")
    print("   ℹ️  Note: API key validity not tested (requires actual API call)")
    return True


async def check_embedding_model():
    """Check embedding model"""
    print("🧠 Checking embedding model...")
    try:
        from tracingrag.services.embedding import generate_embedding

        # Try to generate a test embedding
        embedding = await generate_embedding("test")
        dimension = len(embedding)
        print(f"   ✅ Embedding model: Loaded ({dimension} dimensions)")
        return True
    except Exception as e:
        print(f"   ❌ Embedding model: {e}")
        return False


async def main():
    """Run all checks"""
    print("=" * 60)
    print("TracingRAG Setup Verification")
    print("=" * 60)
    print()

    # Check environment file
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("⚠️  Warning: .env file not found")
        print("   Copy .env.example to .env and configure it")
        print()

    # Run all checks
    results = {}
    results["PostgreSQL"] = await check_postgres()
    results["Neo4j"] = await check_neo4j()
    results["Qdrant"] = await check_qdrant()
    results["Redis"] = await check_redis()
    results["LLM"] = await check_llm()
    results["Embedding"] = await check_embedding_model()

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    required_services = ["PostgreSQL", "Neo4j", "Qdrant", "LLM", "Embedding"]
    optional_services = ["Redis"]

    all_required_ok = all(results.get(s, False) for s in required_services)
    redis_ok = results.get("Redis")

    if all_required_ok:
        print("✅ All required services are working!")
        if redis_ok:
            print("✅ Optional Redis caching is enabled")
        elif redis_ok is None:
            print("⚠️  Redis not available - using in-memory cache (limited)")
        print()
        print("🚀 You're ready to start TracingRAG!")
        print("   Run: poetry run uvicorn tracingrag.api.main:app --reload")
        return 0
    else:
        print("❌ Some required services are not working")
        print()
        print("Required services:")
        for service in required_services:
            status = "✅" if results.get(service) else "❌"
            print(f"   {status} {service}")

        print()
        print("Please fix the failed services and try again.")
        print()
        print("Troubleshooting:")
        print("   1. Start services: docker-compose up -d")
        print("   2. Wait for services to initialize (30-60 seconds)")
        print("   3. Check logs: docker-compose logs")
        print("   4. Initialize databases:")
        print("      - poetry run python scripts/init_neo4j.py")
        print("      - poetry run python scripts/init_qdrant.py")
        print("      - poetry run alembic upgrade head")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification cancelled by user")
        sys.exit(1)
