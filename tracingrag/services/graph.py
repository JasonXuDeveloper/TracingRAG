"""Graph service for edge management and relationship tracking"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from tracingrag.core.models.graph import Edge, EdgeStrengthFactors, RelationshipType
from tracingrag.services.embedding import compute_similarity, generate_embedding
from tracingrag.storage.neo4j_client import (
    create_evolution_edge as neo4j_create_edge,
    get_neo4j_driver,
)


class GraphService:
    """Service for managing graph relationships and edges"""

    async def create_edge(
        self,
        source_state_id: UUID,
        target_state_id: UUID,
        relationship_type: RelationshipType | str,
        strength: float | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_calculate_strength: bool = True,
    ) -> Edge:
        """Create a new edge between memory states

        Args:
            source_state_id: Origin state
            target_state_id: Destination state
            relationship_type: Type of relationship
            strength: Manual strength override (0-1)
            description: Human-readable explanation
            metadata: Additional context
            auto_calculate_strength: Calculate strength automatically if not provided

        Returns:
            Created edge
        """
        # Convert string to enum if needed
        if isinstance(relationship_type, str):
            relationship_type = RelationshipType(relationship_type)

        # Calculate strength if not provided
        if strength is None and auto_calculate_strength:
            strength = await self._calculate_edge_strength(
                source_state_id=source_state_id,
                target_state_id=target_state_id,
                relationship_type=relationship_type,
            )
        elif strength is None:
            strength = 0.5  # Default

        # Create edge model
        edge = Edge(
            source_state_id=source_state_id,
            target_state_id=target_state_id,
            relationship_type=relationship_type,
            strength=strength,
            description=description,
            custom_metadata=metadata or {},
        )

        # Create in Neo4j
        await neo4j_create_edge(
            parent_id=source_state_id,
            child_id=target_state_id,
            relationship_type=relationship_type.value,
            edge_properties=edge.to_neo4j_properties(),
        )

        return edge

    async def update_edge_strength(
        self,
        source_state_id: UUID,
        target_state_id: UUID,
        relationship_type: RelationshipType | str,
        new_strength: float,
    ) -> None:
        """Update the strength of an existing edge

        Args:
            source_state_id: Origin state
            target_state_id: Destination state
            relationship_type: Type of relationship
            new_strength: New strength value (0-1)
        """
        if isinstance(relationship_type, str):
            relationship_type = RelationshipType(relationship_type)

        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run(
                f"""
                MATCH (source:MemoryState {{id: $source_id}})
                      -[r:{relationship_type.value}]->
                      (target:MemoryState {{id: $target_id}})
                SET r.strength = $new_strength
                RETURN r
                """,
                source_id=str(source_state_id),
                target_id=str(target_state_id),
                new_strength=new_strength,
            )

    async def mark_edge_obsolete(
        self,
        source_state_id: UUID,
        target_state_id: UUID,
        relationship_type: RelationshipType | str,
        superseded_by: str | None = None,
    ) -> None:
        """Mark an edge as obsolete (no longer valid)

        Args:
            source_state_id: Origin state
            target_state_id: Destination state
            relationship_type: Type of relationship
            superseded_by: Optional edge ID that replaces this one
        """
        if isinstance(relationship_type, str):
            relationship_type = RelationshipType(relationship_type)

        driver = get_neo4j_driver()
        async with driver.session() as session:
            await session.run(
                f"""
                MATCH (source:MemoryState {{id: $source_id}})
                      -[r:{relationship_type.value}]->
                      (target:MemoryState {{id: $target_id}})
                SET r.valid_until = $valid_until,
                    r.is_active = false,
                    r.superseded_by = $superseded_by
                RETURN r
                """,
                source_id=str(source_state_id),
                target_id=str(target_state_id),
                valid_until=datetime.utcnow().isoformat(),
                superseded_by=superseded_by,
            )

    async def find_related_states(
        self,
        state_id: UUID,
        relationship_types: list[RelationshipType | str] | None = None,
        min_strength: float = 0.0,
        only_active: bool = True,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Find states related to a given state

        Args:
            state_id: State to start from
            relationship_types: Filter by specific relationship types
            min_strength: Minimum edge strength
            only_active: Only return active edges
            max_results: Maximum number of results

        Returns:
            List of related states with edge information
        """
        driver = get_neo4j_driver()

        # Build relationship filter
        rel_filter = ""
        if relationship_types:
            rel_types = [
                rt.value if isinstance(rt, RelationshipType) else rt
                for rt in relationship_types
            ]
            rel_filter = f":{':'.join(rel_types)}"

        # Build query
        query = f"""
        MATCH (start:MemoryState {{id: $state_id}})-[r{rel_filter}]-(related:MemoryState)
        WHERE r.strength >= $min_strength
        {" AND r.is_active = true" if only_active else ""}
        RETURN related, r, type(r) as relationship_type
        ORDER BY r.strength DESC
        LIMIT $max_results
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                state_id=str(state_id),
                min_strength=min_strength,
                max_results=max_results,
            )

            records = await result.data()
            return [
                {
                    "state": record["related"],
                    "edge_strength": record["r"]["strength"],
                    "relationship_type": record["relationship_type"],
                    "is_active": record["r"].get("is_active", True),
                    "edge_properties": dict(record["r"]),
                }
                for record in records
            ]

    async def get_inverse_relationship(
        self,
        relationship_type: RelationshipType,
    ) -> RelationshipType:
        """Get the inverse of a relationship type

        Args:
            relationship_type: Original relationship type

        Returns:
            Inverse relationship type
        """
        inversions = {
            RelationshipType.EVOLVED_TO: RelationshipType.EVOLVED_FROM,
            RelationshipType.EVOLVED_FROM: RelationshipType.EVOLVED_TO,
            RelationshipType.SUPERSEDES: RelationshipType.SUPERSEDED_BY,
            RelationshipType.SUPERSEDED_BY: RelationshipType.SUPERSEDES,
            RelationshipType.REFERENCES: RelationshipType.REFERENCED_BY,
            RelationshipType.REFERENCED_BY: RelationshipType.REFERENCES,
            RelationshipType.DEPENDS_ON: RelationshipType.REQUIRED_BY,
            RelationshipType.REQUIRED_BY: RelationshipType.DEPENDS_ON,
            RelationshipType.CAUSES: RelationshipType.CAUSED_BY,
            RelationshipType.CAUSED_BY: RelationshipType.CAUSES,
            RelationshipType.ENABLES: RelationshipType.ENABLED_BY,
            RelationshipType.ENABLED_BY: RelationshipType.ENABLES,
            RelationshipType.PREVENTS: RelationshipType.PREVENTED_BY,
            RelationshipType.PREVENTED_BY: RelationshipType.PREVENTS,
            RelationshipType.SUPPORTS: RelationshipType.SUPPORTED_BY,
            RelationshipType.SUPPORTED_BY: RelationshipType.SUPPORTS,
            RelationshipType.CONTRADICTS: RelationshipType.CONTRADICTED_BY,
            RelationshipType.CONTRADICTED_BY: RelationshipType.CONTRADICTS,
            RelationshipType.EXPLAINS: RelationshipType.EXPLAINED_BY,
            RelationshipType.EXPLAINED_BY: RelationshipType.EXPLAINS,
            RelationshipType.PART_OF: RelationshipType.CONTAINS,
            RelationshipType.CONTAINS: RelationshipType.PART_OF,
            RelationshipType.MENTIONS: RelationshipType.MENTIONED_BY,
            RelationshipType.MENTIONED_BY: RelationshipType.MENTIONS,
            RelationshipType.IMPLEMENTS: RelationshipType.IMPLEMENTED_BY,
            RelationshipType.IMPLEMENTED_BY: RelationshipType.IMPLEMENTS,
        }

        # Self-inverse relationships
        if relationship_type in [
            RelationshipType.RELATES_TO,
            RelationshipType.SIMILAR_TO,
        ]:
            return relationship_type

        return inversions.get(relationship_type, RelationshipType.RELATES_TO)

    async def create_bidirectional_edge(
        self,
        state_id_a: UUID,
        state_id_b: UUID,
        relationship_type: RelationshipType,
        strength: float | None = None,
        description: str | None = None,
    ) -> tuple[Edge, Edge]:
        """Create bidirectional edges between two states

        Args:
            state_id_a: First state
            state_id_b: Second state
            relationship_type: Type of relationship
            strength: Edge strength
            description: Description

        Returns:
            Tuple of (forward edge, inverse edge)
        """
        # Create forward edge
        forward_edge = await self.create_edge(
            source_state_id=state_id_a,
            target_state_id=state_id_b,
            relationship_type=relationship_type,
            strength=strength,
            description=description,
        )

        # Create inverse edge
        inverse_type = await self.get_inverse_relationship(relationship_type)
        inverse_edge = await self.create_edge(
            source_state_id=state_id_b,
            target_state_id=state_id_a,
            relationship_type=inverse_type,
            strength=strength,
            description=description,
        )

        return forward_edge, inverse_edge

    async def _calculate_edge_strength(
        self,
        source_state_id: UUID,
        target_state_id: UUID,
        relationship_type: RelationshipType,
        factors: EdgeStrengthFactors | None = None,
    ) -> float:
        """Calculate edge strength based on multiple factors

        Args:
            source_state_id: Origin state
            target_state_id: Destination state
            relationship_type: Type of relationship
            factors: Optional pre-computed factors

        Returns:
            Calculated strength (0-1)
        """
        if factors is None:
            factors = EdgeStrengthFactors()

        # Weight configuration for different factors
        weights = {
            "semantic": 0.4,
            "temporal": 0.2,
            "explicit": 0.3,
            "co_occurrence": 0.1,
        }

        strength = 0.0

        # Semantic similarity (if available)
        if factors.semantic_similarity is not None:
            strength += weights["semantic"] * factors.semantic_similarity

        # Temporal proximity (if available)
        if factors.temporal_proximity is not None:
            strength += weights["temporal"] * factors.temporal_proximity

        # Explicit weight (if set)
        if factors.explicit_weight is not None:
            strength += weights["explicit"] * factors.explicit_weight
        else:
            # Default based on relationship type
            strength += weights["explicit"] * self._get_default_strength(relationship_type)

        # Co-occurrence (if available)
        if factors.co_occurrence is not None:
            strength += weights["co_occurrence"] * factors.co_occurrence

        # Normalize to 0-1 range
        return min(max(strength, 0.0), 1.0)

    def _get_default_strength(self, relationship_type: RelationshipType) -> float:
        """Get default strength for a relationship type

        Args:
            relationship_type: Type of relationship

        Returns:
            Default strength value
        """
        # Strong relationships
        if relationship_type in [
            RelationshipType.EVOLVED_TO,
            RelationshipType.SUPERSEDES,
            RelationshipType.CAUSES,
            RelationshipType.CONTRADICTS,
        ]:
            return 0.8

        # Medium relationships
        if relationship_type in [
            RelationshipType.DEPENDS_ON,
            RelationshipType.REFERENCES,
            RelationshipType.SUPPORTS,
            RelationshipType.EXPLAINS,
        ]:
            return 0.6

        # Weak relationships
        if relationship_type in [
            RelationshipType.RELATES_TO,
            RelationshipType.SIMILAR_TO,
            RelationshipType.MENTIONS,
        ]:
            return 0.4

        # Default
        return 0.5

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph

        Returns:
            Dictionary with graph statistics
        """
        driver = get_neo4j_driver()

        async with driver.session() as session:
            # Count nodes
            node_result = await session.run(
                "MATCH (n:MemoryState) RETURN count(n) as count"
            )
            node_count = (await node_result.single())["count"]

            # Count edges (all relationships)
            edge_result = await session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            )
            edge_count = (await edge_result.single())["count"]

            # Count active edges
            active_result = await session.run(
                "MATCH ()-[r]->() WHERE r.is_active = true RETURN count(r) as count"
            )
            active_count = (await active_result.single())["count"]

            # Relationship type distribution
            type_result = await session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
                """
            )
            type_distribution = {
                record["rel_type"]: record["count"]
                for record in await type_result.data()
            }

            return {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "active_edges": active_count,
                "inactive_edges": edge_count - active_count,
                "avg_edges_per_node": edge_count / node_count if node_count > 0 else 0,
                "relationship_type_distribution": type_distribution,
            }
