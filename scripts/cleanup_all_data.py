"""Cleanup script to list and delete all memory states"""

import asyncio
import sys

from tracingrag.client import AsyncTracingRAGClient


async def main():
    """List and delete all memory states"""
    client = AsyncTracingRAGClient("http://localhost:8000")

    # Check if --force flag is provided
    force_delete = "--force" in sys.argv or "-f" in sys.argv

    print("=" * 70)
    print("📋 Listing all memory states...")
    print("=" * 70)

    # Get all memories
    memories = await client.list_memories(limit=1000)
    print(f"\nFound {len(memories)} memory states:\n")

    # Group by topic
    topics = {}
    for memory in memories:
        if memory.topic not in topics:
            topics[memory.topic] = []
        topics[memory.topic].append(memory)

    # Display grouped by topic
    for topic, states in sorted(topics.items()):
        print(f"📌 {topic}: {len(states)} version(s)")
        for state in sorted(states, key=lambda s: s.version):
            print(
                f"   v{state.version} - {state.id} - {state.timestamp.strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"      {state.content[:80]}...")

    if not memories:
        print("✅ Database is already empty!")
        return

    # Confirm deletion
    print("\n" + "=" * 70)
    print("⚠️  WARNING: About to delete ALL memory states!")
    print("=" * 70)

    if not force_delete:
        response = input(f"\nDelete all {len(memories)} memories? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Deletion cancelled.")
            return
    else:
        print(f"\n🚀 Force mode: Deleting all {len(memories)} memories...")

    # Delete all memories
    print("\n🗑️  Deleting all memories...")
    deleted_count = 0
    failed_count = 0

    for memory in memories:
        try:
            await client.delete_memory(memory.id)
            deleted_count += 1
            print(f"   ✓ Deleted {memory.topic} v{memory.version} ({memory.id})")
        except Exception as e:
            failed_count += 1
            print(f"   ✗ Failed to delete {memory.topic} v{memory.version}: {e}")

    print("\n" + "=" * 70)
    print("✅ Cleanup complete!")
    print(f"   Deleted: {deleted_count}")
    print(f"   Failed:  {failed_count}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
