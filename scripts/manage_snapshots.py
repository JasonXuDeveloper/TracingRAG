"""CLI tool for managing Neo4j snapshots"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import tracingrag modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.storage.neo4j_snapshot import get_snapshot_manager
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


async def list_snapshots():
    """List all available snapshots"""
    manager = get_snapshot_manager()
    snapshots = manager.list_snapshots()

    if not snapshots:
        print("No snapshots found.")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(snapshots)} snapshots:")
    print(f"{'='*80}\n")

    for i, snap in enumerate(snapshots, 1):
        print(f"{i}. {snap['filename']}")
        print(f"   Created: {snap['created_at']}")
        print(f"   Operation: {snap['operation']}")
        print(f"   State ID: {snap['state_id']}")
        print(f"   Size: {snap['size_kb']:.2f} KB")
        print()


async def restore_snapshot(snapshot_path: str):
    """Restore from a snapshot"""
    manager = get_snapshot_manager()

    print("\n⚠️  WARNING ⚠️")
    print("This will REPLACE the current Neo4j graph with:")
    print(f"  {snapshot_path}")
    print()

    confirm = input("Type 'YES' to confirm: ")

    if confirm != "YES":
        print("Restore cancelled.")
        return

    print("\nRestoring...")
    success = await manager.restore_snapshot(snapshot_path)

    if success:
        print("✅ Restore completed successfully!")
    else:
        print("❌ Restore failed!")
        sys.exit(1)


async def create_manual_snapshot(operation: str = "manual"):
    """Create a manual snapshot"""
    from uuid import uuid4

    manager = get_snapshot_manager()

    snapshot_path = await manager.create_snapshot(
        state_id=uuid4(),
        operation=operation,
        metadata={"manual": True},
    )

    if snapshot_path:
        print(f"✅ Snapshot created: {snapshot_path}")
    else:
        print("❌ Failed to create snapshot")


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_snapshots.py list")
        print("  python manage_snapshots.py restore <snapshot_path>")
        print("  python manage_snapshots.py create [operation_name]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        asyncio.run(list_snapshots())

    elif command == "restore":
        if len(sys.argv) < 3:
            print("Error: Please provide snapshot path")
            sys.exit(1)
        snapshot_path = sys.argv[2]
        asyncio.run(restore_snapshot(snapshot_path))

    elif command == "create":
        operation = sys.argv[2] if len(sys.argv) > 2 else "manual"
        asyncio.run(create_manual_snapshot(operation))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
