"""Neo4j graph database client for knowledge graph storage and relationship tracking"""

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

        await session.run(
            """
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity) REQUIRE (e.type, e.name) IS UNIQUE
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

        await session.run(
            """
            CREATE INDEX entity_type IF NOT EXISTS
            FOR (e:Entity) ON (e.type)
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
        metadata: Additional metadata
    """
    driver = get_neo4j_driver()

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
            metadata=metadata or {},
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
        edge_properties: Additional properties for the relationship
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = f"""
        MATCH (parent:MemoryState {{id: $parent_id}})
        MATCH (child:MemoryState {{id: $child_id}})
        CREATE (parent)-[r:{relationship_type} $properties]->(child)
        RETURN r
        """

        await session.run(
            query,
            parent_id=str(parent_id),
            child_id=str(child_id),
            properties=edge_properties or {},
        )


async def create_entity_node(
    entity_type: str,
    entity_name: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Create or update an Entity node

    Args:
        entity_type: Type of entity (e.g., Character, Location, Concept)
        entity_name: Name of the entity
        properties: Additional properties for the entity
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """
            MERGE (e:Entity {type: $entity_type, name: $entity_name})
            SET e += $properties
            """,
            entity_type=entity_type,
            entity_name=entity_name,
            properties=properties or {},
        )


async def create_entity_relationship(
    state_id: UUID,
    entity_type: str,
    entity_name: str,
    relationship_type: str = "MENTIONS",
    properties: dict[str, Any] | None = None,
) -> None:
    """Create a relationship between a memory state and an entity

    Args:
        state_id: UUID of the memory state
        entity_type: Type of entity
        entity_name: Name of entity
        relationship_type: Type of relationship (MENTIONS, DEFINES, MODIFIES, etc.)
        properties: Additional properties for the relationship
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = f"""
        MATCH (m:MemoryState {{id: $state_id}})
        MERGE (e:Entity {{type: $entity_type, name: $entity_name}})
        CREATE (m)-[r:{relationship_type} $properties]->(e)
        RETURN r
        """

        await session.run(
            query,
            state_id=str(state_id),
            entity_type=entity_type,
            entity_name=entity_name,
            properties=properties or {},
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


async def get_entity_mentions(
    entity_type: str,
    entity_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get all memory states that mention a specific entity

    Args:
        entity_type: Type of entity
        entity_name: Name of entity
        limit: Maximum number of results

    Returns:
        List of memory states that mention the entity
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        query = """
        MATCH (m:MemoryState)-[r]->(e:Entity {type: $entity_type, name: $entity_name})
        RETURN m, type(r) as relationship_type, r.properties as relationship_properties
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """

        result = await session.run(
            query,
            entity_type=entity_type,
            entity_name=entity_name,
            limit=limit,
        )

        records = await result.data()
        return [
            {
                "memory": record["m"],
                "relationship_type": record["relationship_type"],
                "relationship_properties": record["relationship_properties"],
            }
            for record in records
        ]


async def delete_memory_node(state_id: UUID) -> None:
    """Delete a memory state node and its relationships

    Args:
        state_id: UUID of the memory state to delete
    """
    driver = get_neo4j_driver()

    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """
            MATCH (m:MemoryState {id: $state_id})
            DETACH DELETE m
            """,
            state_id=str(state_id),
        )


async def close_neo4j() -> None:
    """Close Neo4j driver connection"""
    global _neo4j_driver
    if _neo4j_driver is not None:
        await _neo4j_driver.close()
        _neo4j_driver = None
