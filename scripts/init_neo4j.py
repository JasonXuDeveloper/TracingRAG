#!/usr/bin/env python3
"""Initialize Neo4j database schema"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.storage.neo4j_client import Neo4jClient


async def main():
    """Initialize Neo4j schema"""
    print("🔧 Initializing Neo4j schema...")

    # Get connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "tracingrag123")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    print(f"   Connecting to: {uri}")
    print(f"   Database: {database}")

    try:
        # Initialize client
        client = Neo4jClient(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )

        # Initialize schema (creates constraints and indexes)
        await client.initialize_schema()

        print("✅ Neo4j schema initialized successfully!")
        print("\nCreated:")
        print("   - Constraints for unique IDs and topics")
        print("   - Indexes for timestamps and entity types")
        print("   - Indexes for edge properties")

        # Verify by getting stats
        stats = await client.get_graph_stats()
        print(f"\nCurrent stats:")
        print(f"   - Nodes: {stats.node_count}")
        print(f"   - Edges: {stats.edge_count}")

        await client.close()

    except Exception as e:
        print(f"❌ Error initializing Neo4j: {e}")
        print("\nTroubleshooting:")
        print("   1. Ensure Neo4j is running: docker-compose ps")
        print("   2. Check connection details in .env file")
        print("   3. Wait for Neo4j to fully start (can take 30-60 seconds)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
