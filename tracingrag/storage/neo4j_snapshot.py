"""Neo4j graph snapshot manager for backup and recovery"""

import gzip
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from tracingrag.config import settings
from tracingrag.storage.neo4j_client import get_neo4j_driver
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


def serialize_neo4j_value(obj: Any) -> Any:
    """Recursively convert Neo4j types to JSON-serializable Python types

    Args:
        obj: Object to serialize (can be dict, list, DateTime, etc.)

    Returns:
        JSON-serializable version of the object
    """
    # Import here to avoid circular dependency
    try:
        from neo4j.time import DateTime
    except ImportError:
        DateTime = None  # type: ignore

    if DateTime and isinstance(obj, DateTime):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_neo4j_value(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        return [serialize_neo4j_value(item) for item in obj]
    elif isinstance(obj, UUID):
        return str(obj)
    else:
        return obj


class Neo4jSnapshotManager:
    """Manages Neo4j graph snapshots for backup and recovery"""

    def __init__(self):
        self.snapshot_dir = Path(settings.neo4j_snapshot_path)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = settings.neo4j_snapshot_max_count
        self.compression = settings.neo4j_snapshot_compression

    def _get_snapshot_filename(self, state_id: UUID, operation: str) -> str:
        """Generate snapshot filename

        Args:
            state_id: UUID of the state that triggered the snapshot
            operation: Operation type (create/cascade/promote/update)

        Returns:
            Filename in format: {timestamp}_{operation}_{state_id}.json[.gz]
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{operation}_{str(state_id)[:8]}.json"

        if self.compression:
            filename += ".gz"

        return filename

    async def create_snapshot(
        self,
        state_id: UUID,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Create a snapshot of the current Neo4j graph state

        Args:
            state_id: UUID of the state that triggered this snapshot
            operation: Operation type (create/cascade/promote/update)
            metadata: Additional metadata to include in snapshot

        Returns:
            Path to the created snapshot file, or None if snapshots disabled
        """
        if not settings.neo4j_snapshot_enabled:
            return None

        try:
            filename = self._get_snapshot_filename(state_id, operation)
            filepath = self.snapshot_dir / filename

            logger.info(f"📸 Creating snapshot: {filename}")

            # Capture graph data
            content = await self._export_graph()

            # Add metadata
            snapshot_data = {
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "state_id": str(state_id),
                    "operation": operation,
                    **(metadata or {}),
                },
                "content": content,
            }

            # Write to file
            if self.compression:
                with gzip.open(filepath, "wt", encoding="utf-8") as f:
                    json.dump(snapshot_data, f, indent=2)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(snapshot_data, f, indent=2)

            file_size = os.path.getsize(filepath)
            logger.info(f"✅ Snapshot created: {filename} ({file_size / 1024:.2f} KB)")

            # Cleanup old snapshots
            await self._cleanup_old_snapshots()

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None

    async def _export_graph(self) -> dict[str, Any]:
        """Export graph as JSON structure"""
        driver = get_neo4j_driver()

        async with driver.session(database=settings.neo4j_database) as session:
            # Export nodes
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

            # Export relationships
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
                            "properties": (
                                dict(record["properties"]) if record["properties"] else {}
                            ),
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

    async def _cleanup_old_snapshots(self) -> int:
        """Delete oldest snapshots if exceeding max count

        Returns:
            Number of snapshots deleted
        """
        try:
            # List all snapshot files
            snapshot_files = sorted(
                [f for f in self.snapshot_dir.iterdir() if f.is_file()],
                key=lambda f: f.stat().st_mtime,
            )

            # Delete oldest if exceeding limit
            deleted_count = 0
            while len(snapshot_files) > self.max_snapshots:
                oldest = snapshot_files.pop(0)
                oldest.unlink()
                deleted_count += 1
                logger.debug(f"🗑️ Deleted old snapshot: {oldest.name}")

            if deleted_count > 0:
                logger.info(
                    f"Cleaned up {deleted_count} old snapshots (keeping {self.max_snapshots})"
                )

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old snapshots: {e}")
            return 0

    async def restore_snapshot(self, snapshot_path: str) -> bool:
        """Restore graph from a snapshot file

        WARNING: This will REPLACE the current graph!

        Args:
            snapshot_path: Path to the snapshot file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning(f"⚠️ Restoring snapshot: {snapshot_path}")
            logger.warning("This will CLEAR the current graph!")

            # Load snapshot
            if snapshot_path.endswith(".gz"):
                with gzip.open(snapshot_path, "rt", encoding="utf-8") as f:
                    snapshot_data = json.load(f)
            else:
                with open(snapshot_path, encoding="utf-8") as f:
                    snapshot_data = json.load(f)

            metadata = snapshot_data["metadata"]
            content = snapshot_data["content"]

            logger.info(f"Snapshot metadata: {metadata}")

            driver = get_neo4j_driver()

            async with driver.session(database=settings.neo4j_database) as session:
                # Clear existing graph
                await session.run("MATCH (n:MemoryState) DETACH DELETE n")
                logger.info("Cleared existing graph")

                # Create nodes
                for node in content["nodes"]:
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

                logger.info(f"Restored {len(content['nodes'])} nodes")

                # Create relationships
                for rel in content["relationships"]:
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

                logger.info(f"Restored {len(content['relationships'])} relationships")

            logger.info(f"✅ Snapshot restored successfully from: {snapshot_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all available snapshots with metadata

        Returns:
            List of snapshot info dicts sorted by creation time (newest first)
        """
        snapshots = []

        for filepath in self.snapshot_dir.iterdir():
            if not filepath.is_file():
                continue

            try:
                # Load metadata
                if filepath.name.endswith(".gz"):
                    with gzip.open(filepath, "rt", encoding="utf-8") as f:
                        snapshot_data = json.load(f)
                else:
                    with open(filepath, encoding="utf-8") as f:
                        snapshot_data = json.load(f)

                metadata = snapshot_data["metadata"]
                file_size = os.path.getsize(filepath)

                snapshots.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "created_at": metadata["created_at"],
                        "state_id": metadata["state_id"],
                        "operation": metadata["operation"],
                        "size_kb": file_size / 1024,
                        "compressed": filepath.name.endswith(".gz"),
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to read snapshot {filepath.name}: {e}")

        # Sort by creation time (newest first)
        snapshots.sort(key=lambda s: s["created_at"], reverse=True)

        return snapshots


# Global instance
_snapshot_manager = None


def get_snapshot_manager() -> Neo4jSnapshotManager:
    """Get or create the global snapshot manager instance"""
    global _snapshot_manager
    if _snapshot_manager is None:
        _snapshot_manager = Neo4jSnapshotManager()
    return _snapshot_manager
