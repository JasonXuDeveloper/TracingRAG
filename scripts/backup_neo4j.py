#!/usr/bin/env python3
"""Neo4j Native Backup Script

This script creates a backup of the Neo4j graph database using Cypher export.
Backups are stored as compressed Cypher script files that can be restored later.

Usage:
    python scripts/backup_neo4j.py [--max-backups N]

Features:
    - Online backup (no need to stop Neo4j)
    - Automatic compression with gzip
    - Rotation of old backups (keeps latest N backups)
    - Timestamped backup files
    - Atomic backup (all-or-nothing)

Backup files are stored in: ./data/neo4j_backups/
"""

import argparse
import asyncio
import gzip
import json
import os

# Add project root to path
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.config import settings
from tracingrag.storage.neo4j_client import get_neo4j_driver
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

# Backup configuration
BACKUP_DIR = Path("./data/neo4j_backups")
DEFAULT_MAX_BACKUPS = 30  # Keep last 30 backups


def serialize_neo4j_value(obj):
    """Convert Neo4j types to JSON-serializable format"""
    if obj is None:
        return None
    elif hasattr(obj, "isoformat"):  # DateTime
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_neo4j_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_neo4j_value(item) for item in obj]
    else:
        return obj


async def export_graph() -> dict:
    """Export entire Neo4j graph as structured data

    Returns:
        dict with nodes, relationships, and statistics
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        # Export all MemoryState nodes
        logger.info("Exporting nodes...")
        nodes = []
        node_result = await session.run(
            """
            MATCH (m:MemoryState)
            RETURN m.id AS id,
                   m.topic AS topic,
                   m.version AS version,
                   m.timestamp AS timestamp,
                   m.storage_tier AS storage_tier,
                   m.metadata AS metadata
            ORDER BY m.timestamp
            """
        )

        async for record in node_result:
            nodes.append(
                serialize_neo4j_value(
                    {
                        "id": record["id"],
                        "topic": record["topic"],
                        "version": record["version"],
                        "timestamp": record["timestamp"],
                        "storage_tier": record["storage_tier"],
                        "metadata": record["metadata"],
                    }
                )
            )

        # Export all relationships
        logger.info("Exporting relationships...")
        relationships = []
        rel_result = await session.run(
            """
            MATCH (source:MemoryState)-[r]->(target:MemoryState)
            RETURN source.id AS source_id,
                   target.id AS target_id,
                   type(r) AS rel_type,
                   properties(r) AS properties
            ORDER BY source.timestamp
            """
        )

        async for record in rel_result:
            relationships.append(
                serialize_neo4j_value(
                    {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "type": record["rel_type"],
                        "properties": dict(record["properties"]) if record["properties"] else {},
                    }
                )
            )

        return {
            "nodes": nodes,
            "relationships": relationships,
            "statistics": {
                "node_count": len(nodes),
                "relationship_count": len(relationships),
            },
        }


async def create_backup(max_backups: int = DEFAULT_MAX_BACKUPS) -> str:
    """Create a backup of the Neo4j graph

    Args:
        max_backups: Maximum number of backups to keep

    Returns:
        Path to the created backup file
    """
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Generate backup filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"neo4j_backup_{timestamp}.json.gz"
    backup_path = BACKUP_DIR / backup_filename

    logger.info(f"📦 Creating Neo4j backup: {backup_filename}")

    try:
        # Export graph data
        graph_data = await export_graph()

        # Create backup file with metadata
        backup_data = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "neo4j_uri": settings.neo4j_uri,
                "neo4j_database": settings.neo4j_database,
                "node_count": graph_data["statistics"]["node_count"],
                "relationship_count": graph_data["statistics"]["relationship_count"],
            },
            "data": graph_data,
        }

        # Write compressed backup
        with gzip.open(backup_path, "wt", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2)

        file_size = os.path.getsize(backup_path)
        logger.info(
            f"✅ Backup created: {backup_filename} "
            f"({file_size / 1024:.2f} KB, "
            f"{graph_data['statistics']['node_count']} nodes, "
            f"{graph_data['statistics']['relationship_count']} relationships)"
        )

        # Cleanup old backups
        cleanup_old_backups(max_backups)

        return str(backup_path)

    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        # Remove incomplete backup
        if backup_path.exists():
            backup_path.unlink()
        raise


def cleanup_old_backups(max_backups: int):
    """Remove old backups, keeping only the latest N

    Args:
        max_backups: Maximum number of backups to keep
    """
    try:
        # List all backup files
        backup_files = sorted(
            [f for f in BACKUP_DIR.iterdir() if f.is_file() and f.name.startswith("neo4j_backup_")],
            key=lambda f: f.stat().st_mtime,
        )

        # Delete oldest if exceeding limit
        deleted_count = 0
        while len(backup_files) > max_backups:
            oldest = backup_files.pop(0)
            oldest.unlink()
            deleted_count += 1
            logger.debug(f"🗑️ Deleted old backup: {oldest.name}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backups (keeping latest {max_backups})")

    except Exception as e:
        logger.error(f"Failed to cleanup old backups: {e}")


def list_backups():
    """List all available backups"""
    if not BACKUP_DIR.exists():
        logger.info("No backups found (backup directory doesn't exist)")
        return

    backup_files = sorted(
        [f for f in BACKUP_DIR.iterdir() if f.is_file() and f.name.startswith("neo4j_backup_")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not backup_files:
        logger.info("No backups found")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Found {len(backup_files)} backup(s):\n")

    for i, filepath in enumerate(backup_files, 1):
        try:
            # Load metadata
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                backup_data = json.load(f)
                metadata = backup_data["metadata"]

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


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Neo4j Native Backup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a backup (keep last 30 backups)
  python scripts/backup_neo4j.py

  # Create a backup (keep last 10 backups)
  python scripts/backup_neo4j.py --max-backups 10

  # List all backups
  python scripts/backup_neo4j.py --list

Backup files are stored in: ./data/neo4j_backups/

To restore a backup, use:
  python scripts/restore_neo4j.py <backup_file>
        """,
    )

    parser.add_argument(
        "--max-backups",
        type=int,
        default=DEFAULT_MAX_BACKUPS,
        help=f"Maximum number of backups to keep (default: {DEFAULT_MAX_BACKUPS})",
    )

    parser.add_argument("--list", action="store_true", help="List all available backups")

    args = parser.parse_args()

    if args.list:
        list_backups()
    else:
        try:
            backup_path = await create_backup(max_backups=args.max_backups)
            logger.info(f"\n✅ Backup successfully created: {backup_path}")
            logger.info(
                f"\nTo restore this backup, run:\n  python scripts/restore_neo4j.py {backup_path}"
            )
        except Exception as e:
            logger.error(f"\n❌ Backup failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
