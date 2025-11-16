"""Neo4j graph database client for knowledge graph storage and relationship tracking"""

import json
from typing import Any
from uuid import UUID

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

from tracingrag.config import settings
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

# Global Neo4j driver instance
_neo4j_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    """Get or create Neo4j driver instance"""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        )
    return _neo4j_driver


async def verify_neo4j_connection() -> bool:
    """Verify Neo4j connection is working

    Returns:
        True if connection is successful, False otherwise
    """
    driver = get_neo4j_driver()
    try:
        await driver.verify_connectivity()
        return True
    except ServiceUnavailable:
        return False


async def init_neo4j_schema() -> None:
    """Initialize Neo4j schema with constraints and indexes"""
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        # Create uniqueness constraints
        await session.run(
            """
            CREATE CONSTRAINT memory_state_id IF NOT EXISTS
            FOR (m:MemoryState) REQUIRE m.id IS UNIQUE
            """
        )

        await session.run(
            """
            CREATE CONSTRAINT topic_name IF NOT EXISTS
            FOR (t:Topic) REQUIRE t.name IS UNIQUE
            """
        )

        # Create indexes for common queries
        await session.run(
            """
            CREATE INDEX memory_state_timestamp IF NOT EXISTS
            FOR (m:MemoryState) ON (m.timestamp)
            """
        )

        await session.run(
            """
            CREATE INDEX memory_state_version IF NOT EXISTS
            FOR (m:MemoryState) ON (m.version)
            """
        )

        await session.run(
            """
            CREATE INDEX memory_state_storage_tier IF NOT EXISTS
            FOR (m:MemoryState) ON (m.storage_tier)
            """
        )


async def create_memory_node(
    state_id: UUID,
    topic: str,
    version: int,
    timestamp: str,
    storage_tier: str = "active",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Create a MemoryState node in Neo4j

    Args:
        state_id: UUID of the memory state
        topic: Topic/key for the memory
        version: Version number
        timestamp: ISO timestamp string
        storage_tier: Storage tier (active/archived/cold)
        metadata: Additional metadata (will be stored as JSON string)
    """
    driver = get_neo4j_driver()

    # Convert metadata dict to JSON string for Neo4j storage
    metadata_json = json.dumps(metadata) if metadata else "{}"

    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """
            MERGE (t:Topic {name: $topic})
            CREATE (m:MemoryState {
                id: $state_id,
                topic: $topic,
                version: $version,
                timestamp: datetime($timestamp),
                storage_tier: $storage_tier,
                metadata: $metadata
            })
            CREATE (m)-[:BELONGS_TO]->(t)
            """,
            state_id=str(state_id),
            topic=topic,
            version=version,
            timestamp=timestamp,
            storage_tier=storage_tier,
            metadata=metadata_json,
        )


async def create_evolution_edge(
    parent_id: UUID,
    child_id: UUID,
    relationship_type: str = "EVOLVED_TO",
    edge_properties: dict[str, Any] | None = None,
) -> None:
    """Create an evolution relationship between two memory states

    Args:
        parent_id: UUID of the parent state
        child_id: UUID of the child state
        relationship_type: Type of relationship (EVOLVED_TO, SUPERSEDED_BY, etc.)
        edge_properties: Additional properties for the relationship (will be stored as JSON string)

    Raises:
        ValueError: If relationship_type is not in RelationshipType enum
    """
    from tracingrag.core.models.graph import RelationshipType

    # Validate relationship type (must be in enum)
    try:
        RelationshipType(relationship_type)
    except ValueError:
        raise ValueError(
            f"Invalid relationship type '{relationship_type}'. "
            f"Must be one of: {[rt.value for rt in RelationshipType]}"
        )

    driver = get_neo4j_driver()

    # Prepare edge properties - ensure they're a dict
    props = edge_properties if edge_properties else {}

    async with driver.session(database=settings.neo4j_database) as session:
        query = f"""
        MATCH (parent:MemoryState {{id: $parent_id}})
        MATCH (child:MemoryState {{id: $child_id}})
        MERGE (parent)-[r:{relationship_type}]->(child)
        ON CREATE SET r += $properties
        RETURN r
        """

        await session.run(
            query,
            parent_id=str(parent_id),
            child_id=str(child_id),
            properties=props,
        )


async def get_parent_relationships(
    parent_id: UUID,
) -> list[dict[str, Any]]:
    """Get all relationships from a parent state that should be considered for inheritance

    Args:
        parent_id: UUID of the parent state

    Returns:
        List of relationship details: {target_id, target_topic, target_version, rel_type, properties}
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = """
        MATCH (parent:MemoryState {id: $parent_id})-[r]->(target:MemoryState)
        WHERE type(r) IN ['RELATES_TO', 'DEPENDS_ON', 'CAUSED_BY', 'SIMILAR_TO']
        RETURN
            target.id as target_id,
            target.topic as target_topic,
            target.version as target_version,
            type(r) as rel_type,
            properties(r) as properties
        """

        result = await session.run(query, parent_id=str(parent_id))
        relationships = []

        async for record in result:
            # Skip relationships with invalid target_id
            target_id = record["target_id"]
            if not target_id or not str(target_id).strip():
                logger.warning(f"Skipping relationship with invalid target_id: {target_id}")
                continue

            relationships.append(
                {
                    "target_id": target_id,
                    "target_topic": record["target_topic"],
                    "target_version": record["target_version"],
                    "rel_type": record["rel_type"],
                    "properties": record["properties"],
                }
            )

        return relationships


async def get_latest_version_of_topic(topic: str) -> dict[str, Any] | None:
    """Get the latest version of a specific topic

    Args:
        topic: Topic name

    Returns:
        Dict with id, topic, version, timestamp or None if not found
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = """
        MATCH (m:MemoryState {topic: $topic})
        RETURN m.id as id, m.topic as topic, m.version as version, m.timestamp as timestamp
        ORDER BY m.version DESC
        LIMIT 1
        """

        result = await session.run(query, topic=topic)
        record = await result.single()

        if record:
            return {
                "id": record["id"],
                "topic": record["topic"],
                "version": record["version"],
                "timestamp": record["timestamp"],
            }
        return None


async def inherit_parent_relationships(
    parent_id: UUID,
    child_id: UUID,
    exclude_types: list[str] | None = None,
) -> int:
    """Inherit relationships from parent state to child state (simple copy)

    NOTE: This is a simple inheritance mechanism. For intelligent relationship updates,
    use update_relationships_on_evolution() instead.

    Args:
        parent_id: UUID of the parent state
        child_id: UUID of the child state
        exclude_types: List of relationship types to exclude

    Returns:
        Number of relationships inherited
    """
    driver = get_neo4j_driver()
    exclude_types = exclude_types or ["EVOLVED_TO", "SUPERSEDED_BY", "BELONGS_TO"]

    async with driver.session(database=settings.neo4j_database) as session:
        # Copy RELATES_TO relationships
        related_query = """
        MATCH (parent:MemoryState {id: $parent_id})-[r:RELATES_TO]->(target)
        MATCH (child:MemoryState {id: $child_id})
        WITH child, target, properties(r) as rel_props
        MERGE (child)-[new_r:RELATES_TO]->(target)
        SET new_r = rel_props
        RETURN count(new_r) as count
        """

        total_inherited = 0

        # Copy RELATES_TO relationships
        result = await session.run(
            related_query,
            parent_id=str(parent_id),
            child_id=str(child_id),
        )
        record = await result.single()
        if record:
            total_inherited += record["count"]

        return total_inherited


async def create_memory_relationship(
    source_id: UUID,
    target_id: UUID,
    relationship_type: str,
    properties: dict[str, Any] | None = None,
    replace_existing: bool = False,
    bound_to_version: bool | None = None,
) -> None:
    """Create a relationship between two memory states

    Args:
        source_id: UUID of the source memory state
        target_id: UUID of the target memory state
        relationship_type: Type of relationship (RELATES_TO, DEPENDS_ON, etc.)
        properties: Additional properties for the relationship
        replace_existing: If True, replace existing relationship of same type between these nodes
        bound_to_version: If True, bind to specific target version (won't auto-update).
                         If None, auto-determine based on relationship type.
    """
    from tracingrag.core.models.graph import RelationshipType

    driver = get_neo4j_driver()

    # Validate relationship type
    try:
        rel_type_enum = RelationshipType(relationship_type)
    except ValueError:
        # Invalid relationship type - log warning and skip creation
        logger.warning(
            f"Invalid relationship type '{relationship_type}' - must be one of {[e.value for e in RelationshipType]}. "
            f"Skipping relationship creation between {source_id} and {target_id}."
        )
        return

    # Auto-determine version binding based on relationship type
    if bound_to_version is None:
        bound_to_version = RelationshipType.is_version_bound(rel_type_enum)

    async with driver.session(database=settings.neo4j_database) as session:
        # Get target version for version binding
        target_version = None
        if bound_to_version:
            version_query = """
            MATCH (target:MemoryState {id: $target_id})
            RETURN target.version as version
            """
            result = await session.run(version_query, target_id=str(target_id))
            record = await result.single()
            if record:
                target_version = record["version"]

        # Merge properties with version binding info
        props = properties.copy() if properties else {}
        props["bound_to_version"] = bound_to_version
        if target_version is not None:
            props["target_version_at_creation"] = target_version

        # Initialize importance tracking fields
        if "importance" not in props:
            props["importance"] = props.get("strength", 0.5)  # Default to strength
        if "access_count" not in props:
            props["access_count"] = 0
        if "last_accessed" not in props:
            props["last_accessed"] = None

        properties_json = json.dumps(props)

        if replace_existing:
            # Delete existing relationship of same type first
            delete_query = f"""
            MATCH (source:MemoryState {{id: $source_id}})-[r:{relationship_type}]->(target:MemoryState {{id: $target_id}})
            DELETE r
            """
            await session.run(
                delete_query,
                source_id=str(source_id),
                target_id=str(target_id),
            )

        # Use MERGE to avoid creating duplicate relationships
        create_query = f"""
        MATCH (source:MemoryState {{id: $source_id}})
        MATCH (target:MemoryState {{id: $target_id}})
        MERGE (source)-[r:{relationship_type}]->(target)
        ON CREATE SET r.properties = $properties
        ON MATCH SET r.properties = $properties
        RETURN r
        """

        await session.run(
            create_query,
            source_id=str(source_id),
            target_id=str(target_id),
            properties=properties_json,
        )


async def delete_memory_relationship(
    source_id: UUID,
    target_id: UUID,
    relationship_type: str | None = None,
) -> int:
    """Delete relationship(s) between two memory states

    Args:
        source_id: UUID of the source memory state
        target_id: UUID of the target memory state
        relationship_type: Optional specific relationship type to delete (if None, deletes all)

    Returns:
        Number of relationships deleted
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        if relationship_type:
            query = f"""
            MATCH (source:MemoryState {{id: $source_id}})-[r:{relationship_type}]->(target:MemoryState {{id: $target_id}})
            DELETE r
            RETURN count(r) as deleted_count
            """
        else:
            query = """
            MATCH (source:MemoryState {id: $source_id})-[r]->(target:MemoryState {id: $target_id})
            WHERE type(r) IN ['RELATES_TO', 'DEPENDS_ON', 'CAUSED_BY', 'SIMILAR_TO']
            DELETE r
            RETURN count(r) as deleted_count
            """

        result = await session.run(
            query,
            source_id=str(source_id),
            target_id=str(target_id),
        )
        record = await result.single()
        return record["deleted_count"] if record else 0


async def update_relationship_access(
    source_id: UUID,
    target_id: UUID,
    relationship_type: str,
) -> None:
    """Update access tracking for a relationship (called when relationship is used in query)

    Args:
        source_id: UUID of the source memory state
        target_id: UUID of the target memory state
        relationship_type: Type of relationship
    """
    from datetime import datetime

    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = f"""
        MATCH (source:MemoryState {{id: $source_id}})-[r:{relationship_type}]->(target:MemoryState {{id: $target_id}})
        SET r.properties = apoc.convert.fromJsonMap(
            apoc.convert.toJson(
                apoc.map.setKey(
                    apoc.map.setKey(
                        apoc.convert.fromJsonMap(r.properties),
                        'access_count',
                        coalesce(apoc.convert.fromJsonMap(r.properties).access_count, 0) + 1
                    ),
                    'last_accessed',
                    $timestamp
                )
            )
        )
        RETURN r
        """

        await session.run(
            query,
            source_id=str(source_id),
            target_id=str(target_id),
            timestamp=datetime.utcnow().isoformat(),
        )


async def get_topic_history(
    topic: str,
    limit: int = 100,
    storage_tier: str | None = None,
) -> list[dict[str, Any]]:
    """Get evolution history for a topic

    Args:
        topic: Topic name to query
        limit: Maximum number of states to return
        storage_tier: Filter by storage tier (optional)

    Returns:
        List of memory states ordered by version
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = """
        MATCH (m:MemoryState {topic: $topic})
        WHERE $storage_tier IS NULL OR m.storage_tier = $storage_tier
        RETURN m
        ORDER BY m.version DESC
        LIMIT $limit
        """

        result = await session.run(
            query,
            topic=topic,
            storage_tier=storage_tier,
            limit=limit,
        )

        records = await result.data()
        return [record["m"] for record in records]


async def get_related_memories(
    state_id: UUID,
    relationship_types: list[str] | None = None,
    max_depth: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get memories related to a given state through graph traversal

    Args:
        state_id: UUID of the starting memory state
        relationship_types: List of relationship types to traverse (None = all)
        max_depth: Maximum depth of graph traversal
        limit: Maximum number of results

    Returns:
        List of related memory states with relationship information
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        # Build relationship filter
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"

        query = f"""
        MATCH path = (start:MemoryState {{id: $state_id}})-[r{rel_filter}*1..{max_depth}]-(related:MemoryState)
        WHERE start <> related
        RETURN DISTINCT related,
               [rel IN relationships(path) | type(rel)] as relationship_path,
               length(path) as depth
        ORDER BY depth ASC
        LIMIT $limit
        """

        result = await session.run(
            query,
            state_id=str(state_id),
            limit=limit,
        )

        records = await result.data()
        return [
            {
                "memory": record["related"],
                "relationship_path": record["relationship_path"],
                "depth": record["depth"],
            }
            for record in records
        ]


async def update_memory_node_storage_tier(
    state_id: UUID,
    storage_tier: str,
) -> None:
    """Update storage tier for a memory state node in Neo4j

    Args:
        state_id: UUID of the memory state
        storage_tier: New storage tier (active/archived/cold)
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})
            SET m.storage_tier = $storage_tier
            """,
            state_id=str(state_id),
            storage_tier=storage_tier,
        )


async def delete_memory_node(state_id: UUID) -> None:
    """Delete a memory state node and clean up orphan Topic nodes

    This function:
    1. Deletes the MemoryState node and all its relationships
    2. Finds orphan Topic nodes (no remaining MemoryState connections)
    3. Deletes all orphan Topic nodes

    Args:
        state_id: UUID of the memory state to delete
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        # Step 1: Get Topic info BEFORE deletion
        result = await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})
            OPTIONAL MATCH (m)-[:BELONGS_TO]->(t:Topic)
            RETURN t.name as topic_name
            """,
            state_id=str(state_id),
        )
        record = await result.single()

        # Step 2: Now delete the MemoryState
        await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})
            DETACH DELETE m
            """,
            state_id=str(state_id),
        )

        if record and record["topic_name"]:
            topic_name = record["topic_name"]

            # Check if Topic is now orphan (no remaining MemoryStates)
            orphan_check = await session.run(
                """
                MATCH (t:Topic {name: $topic_name})
                WHERE NOT EXISTS((t)<-[:BELONGS_TO]-(:MemoryState))
                RETURN t.name as orphan_topic
                """,
                topic_name=topic_name,
            )
            orphan_topic = await orphan_check.single()

            if orphan_topic and orphan_topic["orphan_topic"]:
                # Delete orphan Topic
                await session.run(
                    """
                    MATCH (t:Topic {name: $topic_name})
                    DELETE t
                    """,
                    topic_name=topic_name,
                )
                logger.info(f"Deleted orphan Topic node: {topic_name}")


async def cleanup_old_version_outgoing_relationships(
    state_id: UUID,
    keep_relationship_types: list[str] | None = None,
) -> dict[str, Any]:
    """Cleanup outgoing relationships from a specific state (directional deletion)

    Only deletes relationships FROM this state TO other states (outgoing edges),
    preserving relationships FROM other states TO this state (incoming edges).
    This allows other states to reference old versions while preventing old
    versions from maintaining outdated relationships.

    Args:
        state_id: State ID to clean up
        keep_relationship_types: Relationship types to preserve, defaults to ['EVOLVED_TO', 'BELONGS_TO']

    Returns:
        {
            "deleted_count": int,  # Number of relationships deleted
            "kept_count": int,     # Number of relationships kept
            "deleted_types": dict[str, int],  # Count of deletions by type
        }
    """
    driver = get_neo4j_driver()

    # Default: preserve EVOLVED_TO (historical trace) and BELONGS_TO (topic membership)
    if keep_relationship_types is None:
        keep_relationship_types = ["EVOLVED_TO", "BELONGS_TO"]

    async with driver.session(database=settings.neo4j_database) as session:
        # Step 1: Count relationships to be deleted
        stats_result = await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})-[r]->()
            WHERE NOT type(r) IN $keep_types
            RETURN type(r) AS rel_type, count(r) AS count
            """,
            state_id=str(state_id),
            keep_types=keep_relationship_types,
        )

        deleted_types = {}
        async for record in stats_result:
            deleted_types[record["rel_type"]] = record["count"]

        # Step 2: Delete outgoing relationships (only FROM this node)
        delete_result = await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})-[r]->()
            WHERE NOT type(r) IN $keep_types
            DELETE r
            RETURN count(r) AS deleted_count
            """,
            state_id=str(state_id),
            keep_types=keep_relationship_types,
        )

        record = await delete_result.single()
        deleted_count = record["deleted_count"] if record else 0

        # Step 3: Count remaining relationships (including outgoing + all incoming)
        kept_result = await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})-[r]-()
            RETURN count(r) AS kept_count
            """,
            state_id=str(state_id),
        )

        record = await kept_result.single()
        kept_count = record["kept_count"] if record else 0

        result = {
            "deleted_count": deleted_count,
            "kept_count": kept_count,
            "deleted_types": deleted_types,
        }

        logger.info(
            f"🗑️ Cleaned state {state_id}: "
            f"deleted {deleted_count} outgoing relationships, "
            f"kept {kept_count} relationships (including incoming)"
        )
        logger.debug(f"   Deleted types: {deleted_types}")

        return result


async def batch_create_relationships(relationships: list[dict[str, Any]]) -> None:
    """Batch create relationships using UNWIND optimization

    Args:
        relationships: [
            {
                "source_id": UUID,
                "target_id": UUID,
                "rel_type": str,
                "properties": dict,
            },
            ...
        ]
    """
    if not relationships:
        return

    driver = get_neo4j_driver()

    # Convert to Neo4j-compatible format
    rel_data = [
        {
            "source_id": str(r["source_id"]),
            "target_id": str(r["target_id"]),
            "rel_type": r["rel_type"],
            "properties": json.dumps(r.get("properties", {})),
        }
        for r in relationships
    ]

    async with driver.session(database=settings.neo4j_database) as session:
        # Use UNWIND for batch creation - process each relationship type separately
        # Group by type first
        rels_by_type = {}
        for rel in rel_data:
            rel_type = rel["rel_type"]
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)

        # Batch create for each type
        total_created = 0
        for rel_type, rels in rels_by_type.items():
            result = await session.run(
                f"""
                UNWIND $relationships AS rel
                MATCH (source:MemoryState {{id: rel.source_id}})
                MATCH (target:MemoryState {{id: rel.target_id}})
                CREATE (source)-[r:{rel_type}]->(target)
                SET r.properties = rel.properties,
                    r.created_at = datetime()
                RETURN count(r) AS created_count
                """,
                relationships=rels,
            )

            record = await result.single()
            count = record["created_count"] if record else 0
            total_created += count

        logger.info(f"✅ Batch created {total_created} relationships ({len(rels_by_type)} types)")


async def close_neo4j() -> None:
    """Close Neo4j driver connection"""
    global _neo4j_driver
    if _neo4j_driver is not None:
        await _neo4j_driver.close()
        _neo4j_driver = None
