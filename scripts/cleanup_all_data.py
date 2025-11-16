"""Cleanup script to delete all TracingRAG data (optimized to use bulk API)"""

import asyncio
import sys

from tracingrag.client import AsyncTracingRAGClient


async def main():
    """Delete all TracingRAG data from all storage layers"""
    client = AsyncTracingRAGClient("http://localhost:8000")

    # Check if --force flag is provided
    force_delete = "--force" in sys.argv or "-f" in sys.argv

    print("=" * 70)
    print("⚠️  WARNING: TracingRAG Data Cleanup Utility")
    print("=" * 70)
    print()
    print("This will permanently DELETE ALL TracingRAG data from:")
    print("  • PostgreSQL (MemoryStateDB, TraceDB tables)")
    print("  • Qdrant (memory_states collection)")
    print("  • Neo4j (MemoryState nodes and relationships)")
    print("  • Redis (TracingRAG cache keys)")
    print()

    if not force_delete:
        confirm = input("Type 'DELETE_ALL_DATA' to confirm: ")
        if confirm != "DELETE_ALL_DATA":
            print("❌ Cleanup cancelled.")
            return
    else:
        print("🚀 Force mode: Proceeding with cleanup...")

    # Call optimized bulk cleanup API
    print("\n🗑️  Deleting all data...")
    try:
        result = await client.cleanup_all()

        if result.get("success"):
            stats = result.get("statistics", {})

            print("\n" + "=" * 70)
            print("✅ Cleanup Complete!")
            print("=" * 70)

            # PostgreSQL stats
            pg_stats = stats.get("postgresql", {})
            print("PostgreSQL:")
            print(f"  ✓ Deleted {pg_stats.get('memories', 0)} memories")
            print(f"  ✓ Deleted {pg_stats.get('traces', 0)} traces")
            print(f"  ✓ Deleted {pg_stats.get('topic_latest_states', 0)} topic latest states")

            # Qdrant stats
            qdrant_stats = stats.get("qdrant", {})
            print("Qdrant:")
            print(f"  ✓ Deleted {qdrant_stats.get('points', 0)} vector points")

            # Neo4j stats
            neo4j_stats = stats.get("neo4j", {})
            print("Neo4j:")
            print(f"  ✓ Deleted {neo4j_stats.get('nodes', 0)} nodes")
            print(f"  ✓ Deleted {neo4j_stats.get('relationships', 0)} relationships")

            # Redis stats
            redis_stats = stats.get("redis", {})
            print("Redis:")
            print(f"  ✓ Deleted {redis_stats.get('keys', 0)} cache keys")

            print("=" * 70)
        else:
            print(f"❌ Cleanup failed: {result.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
