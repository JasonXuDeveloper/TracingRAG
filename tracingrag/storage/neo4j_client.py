"""Neo4j graph database client for knowledge graph storage and relationship tracking"""

import json
from typing import Any
from uuid import UUID

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

from tracingrag.config import settings

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
    """
    driver = get_neo4j_driver()

    # Convert edge properties dict to JSON string for Neo4j storage
    properties_json = json.dumps(edge_properties) if edge_properties else "{}"

    async with driver.session(database=settings.neo4j_database) as session:
        query = f"""
        MATCH (parent:MemoryState {{id: $parent_id}})
        MATCH (child:MemoryState {{id: $child_id}})
        CREATE (parent)-[r:{relationship_type} {{properties: $properties}}]->(child)
        RETURN r
        """

        await session.run(
            query,
            parent_id=str(parent_id),
            child_id=str(child_id),
            properties=properties_json,
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
        WHERE type(r) IN ['RELATED_TO', 'DEPENDS_ON', 'CAUSED_BY', 'SIMILAR_TO']
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
            relationships.append({
                "target_id": record["target_id"],
                "target_topic": record["target_topic"],
                "target_version": record["target_version"],
                "rel_type": record["rel_type"],
                "properties": record["properties"],
            })

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
        # Copy RELATED_TO relationships
        related_query = """
        MATCH (parent:MemoryState {id: $parent_id})-[r:RELATED_TO]->(target)
        MATCH (child:MemoryState {id: $child_id})
        WITH child, target, properties(r) as rel_props
        MERGE (child)-[new_r:RELATED_TO]->(target)
        SET new_r = rel_props
        RETURN count(new_r) as count
        """

        total_inherited = 0

        # Copy RELATED_TO relationships
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
) -> None:
    """Create a relationship between two memory states

    Args:
        source_id: UUID of the source memory state
        target_id: UUID of the target memory state
        relationship_type: Type of relationship (RELATED_TO, DEPENDS_ON, etc.)
        properties: Additional properties for the relationship
        replace_existing: If True, replace existing relationship of same type between these nodes
    """
    driver = get_neo4j_driver()
    properties_json = json.dumps(properties) if properties else "{}"

    async with driver.session(database=settings.neo4j_database) as session:
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

        # Create new relationship
        create_query = f"""
        MATCH (source:MemoryState {{id: $source_id}})
        MATCH (target:MemoryState {{id: $target_id}})
        CREATE (source)-[r:{relationship_type} {{properties: $properties}}]->(target)
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
            WHERE type(r) IN ['RELATED_TO', 'DEPENDS_ON', 'CAUSED_BY', 'SIMILAR_TO']
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
                OPTIONAL MATCH (t)<-[:BELONGS_TO]-(m:MemoryState)
                WITH t, count(m) as state_count
                WHERE state_count = 0
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
                print(f"[Neo4j] Deleted orphan Topic node: {topic_name}")


async def close_neo4j() -> None:
    """Close Neo4j driver connection"""
    global _neo4j_driver
    if _neo4j_driver is not None:
        await _neo4j_driver.close()
        _neo4j_driver = None
