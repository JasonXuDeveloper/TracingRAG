#!/usr/bin/env python3
"""Neo4j Native Restore Script

This script restores a Neo4j graph database from a backup file.

Usage:
    # List all available backups
    python scripts/restore_neo4j.py --list

    # Restore from latest backup
    python scripts/restore_neo4j.py --latest

    # Restore from specific backup
    python scripts/restore_neo4j.py <backup_file>

WARNING: This will CLEAR the current Neo4j graph!

Backup files should be in: ./data/neo4j_backups/
"""

import argparse
import asyncio
import gzip
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.config import settings
from tracingrag.storage.neo4j_client import get_neo4j_driver
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

# Backup configuration
BACKUP_DIR = Path("./data/neo4j_backups")


def list_backups():
    """List all available backups"""
    if not BACKUP_DIR.exists():
        logger.info("No backups found (backup directory doesn't exist)")
        return []

    backup_files = sorted(
        [f for f in BACKUP_DIR.iterdir() if f.is_file() and f.name.startswith("neo4j_backup_")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not backup_files:
        logger.info("No backups found")
        return []

    logger.info(f"\n{'='*80}")
    logger.info(f"Available Backups ({len(backup_files)} total)")
    logger.info(f"{'='*80}\n")

    for i, filepath in enumerate(backup_files, 1):
        try:
            # Load metadata
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                backup_data = json.load(f)
                metadata = backup_data["metadata"]

            import os

            file_size = os.path.getsize(filepath)
            logger.info(f"{i}. {filepath.name}")
            logger.info(f"   Created: {metadata['created_at']}")
            logger.info(
                f"   Size: {file_size / 1024:.2f} KB | "
                f"Nodes: {metadata['node_count']} | "
                f"Relationships: {metadata['relationship_count']}"
            )
            logger.info(f"   Path: {filepath}\n")

        except Exception as e:
            logger.warning(f"{i}. {filepath.name} (Failed to read metadata: {e})\n")

    logger.info(f"{'='*80}\n")

    if backup_files:
        logger.info(f"Latest backup: {backup_files[0].name}")
        logger.info(f"Path: {backup_files[0]}\n")

    return backup_files


async def restore_backup(backup_path: str | Path) -> bool:
    """Restore Neo4j graph from a backup file

    WARNING: This will CLEAR the current graph!

    Args:
        backup_path: Path to the backup file

    Returns:
        True if successful, False otherwise
    """
    backup_path = Path(backup_path)

    if not backup_path.exists():
        logger.error(f"❌ Backup file not found: {backup_path}")
        return False

    try:
        logger.warning(f"\n{'='*80}")
        logger.warning("⚠️  WARNING: This will CLEAR the current Neo4j graph!")
        logger.warning(f"{'='*80}\n")
        logger.info(f"Backup to restore: {backup_path}\n")

        # Load backup
        logger.info("Loading backup file...")
        with gzip.open(backup_path, "rt", encoding="utf-8") as f:
            backup_data = json.load(f)

        metadata = backup_data["metadata"]
        graph_data = backup_data["data"]

        logger.info("Backup metadata:")
        logger.info(f"  Created: {metadata['created_at']}")
        logger.info(f"  Nodes: {metadata['node_count']}")
        logger.info(f"  Relationships: {metadata['relationship_count']}")
        logger.info(f"  Database: {metadata['neo4j_database']}\n")

        driver = get_neo4j_driver()

        async with driver.session(database=settings.neo4j_database) as session:
            # Clear existing graph
            logger.info("Clearing existing graph...")
            await session.run("MATCH (n:MemoryState) DETACH DELETE n")
            await session.run("MATCH (t:Topic) DELETE t")
            logger.info("✓ Existing graph cleared\n")

            # Restore nodes
            logger.info(f"Restoring {len(graph_data['nodes'])} nodes...")
            for i, node in enumerate(graph_data["nodes"]):
                await session.run(
                    """
                    MERGE (t:Topic {name: $topic})
                    CREATE (m:MemoryState {
                        id: $id,
                        topic: $topic,
                        version: $version,
                        timestamp: datetime($timestamp),
                        storage_tier: $storage_tier,
                        metadata: $metadata
                    })
                    CREATE (m)-[:BELONGS_TO]->(t)
                    """,
                    **node,
                )

                if (i + 1) % 100 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(graph_data['nodes'])} nodes")

            logger.info(f"✓ Restored {len(graph_data['nodes'])} nodes\n")

            # Restore relationships
            logger.info(f"Restoring {len(graph_data['relationships'])} relationships...")
            for i, rel in enumerate(graph_data["relationships"]):
                # Skip BELONGS_TO as they're created with nodes
                if rel["type"] == "BELONGS_TO":
                    continue

                # Convert properties dict to JSON string for Neo4j
                props_json = json.dumps(rel["properties"])

                await session.run(
                    f"""
                    MATCH (source:MemoryState {{id: $source_id}})
                    MATCH (target:MemoryState {{id: $target_id}})
                    CREATE (source)-[:{rel['type']} {{properties: $properties}}]->(target)
                    """,
                    source_id=rel["source_id"],
                    target_id=rel["target_id"],
                    properties=props_json,
                )

                if (i + 1) % 100 == 0:
                    logger.info(
                        f"  Progress: {i + 1}/{len(graph_data['relationships'])} relationships"
                    )

            logger.info(f"✓ Restored {len(graph_data['relationships'])} relationships\n")

        logger.info(f"{'='*80}")
        logger.info(f"✅ Backup successfully restored from: {backup_path}")
        logger.info(f"{'='*80}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to restore backup: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Neo4j Native Restore Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all backups
  python scripts/restore_neo4j.py --list

  # Restore from latest backup (with confirmation)
  python scripts/restore_neo4j.py --latest

  # Restore from specific backup
  python scripts/restore_neo4j.py data/neo4j_backups/neo4j_backup_20241119_000000.json.gz

WARNING: Restore will CLEAR the current Neo4j graph!

Backup files are stored in: ./data/neo4j_backups/
        """,
    )

    parser.add_argument("backup_file", nargs="?", help="Path to backup file to restore")

    parser.add_argument("--list", action="store_true", help="List all available backups")

    parser.add_argument("--latest", action="store_true", help="Restore from the latest backup")

    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt (dangerous!)"
    )

    args = parser.parse_args()

    # List backups
    if args.list:
        list_backups()
        return

    # Determine backup file
    backup_path = None

    if args.latest:
        backups = list_backups()
        if not backups:
            logger.error("❌ No backups found!")
            sys.exit(1)
        backup_path = backups[0]
    elif args.backup_file:
        backup_path = Path(args.backup_file)
    else:
        parser.print_help()
        return

    # Confirm restore (unless --yes flag)
    if not args.yes:
        logger.warning("\n⚠️  WARNING: This will CLEAR the current Neo4j graph!\n")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("Restore cancelled.")
            return

    # Restore backup
    success = await restore_backup(backup_path)

    if success:
        logger.info("\n✅ Restore completed successfully!")
    else:
        logger.error("\n❌ Restore failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
