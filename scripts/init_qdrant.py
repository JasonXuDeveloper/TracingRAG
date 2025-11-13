#!/usr/bin/env python3
"""Initialize Qdrant vector database collections"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.storage.qdrant import QdrantClient as TracingRAGQdrantClient


async def main():
    """Initialize Qdrant collections"""
    print("🔧 Initializing Qdrant collections...")

    # Get connection details from environment
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "tracingrag_memories")
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    print(f"   Connecting to: {url}")
    print(f"   Collection: {collection_name}")
    print(f"   Embedding dimension: {embedding_dimension}")

    try:
        # Initialize client
        client = TracingRAGQdrantClient(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
        )

        # Initialize collection (creates if doesn't exist)
        await client.initialize_collection()

        print("✅ Qdrant collection initialized successfully!")
        print("\nConfiguration:")
        print("   - Vector size: 768 dimensions (all-mpnet-base-v2)")
        print("   - Distance metric: Cosine similarity")
        print("   - HNSW index: m=16, ef_construct=100")
        print("\nIndexes created for:")
        print("   - topic (keyword)")
        print("   - storage_tier (keyword)")
        print("   - entity_type (keyword)")
        print("   - is_consolidated (keyword)")
        print("   - consolidation_level (integer)")

        # Verify collection exists
        info = await client.get_collection_info()
        print(f"\nCollection status:")
        print(f"   - Points count: {info.get('points_count', 0)}")
        print(f"   - Status: {info.get('status', 'unknown')}")

        await client.close()

    except Exception as e:
        print(f"❌ Error initializing Qdrant: {e}")
        print("\nTroubleshooting:")
        print("   1. Ensure Qdrant is running: docker-compose ps")
        print("   2. Check connection URL in .env file")
        print("   3. Verify Qdrant is accessible: curl http://localhost:6333/collections")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
