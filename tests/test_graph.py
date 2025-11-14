"""Tests for graph models and service"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from tracingrag.core.models.graph import Edge, EdgeStrengthFactors, RelationshipType
from tracingrag.services.graph import GraphService


class TestRelationshipType:
    """Tests for RelationshipType enum"""

    def test_relationship_types_exist(self):
        """Test that all relationship types are defined"""
        assert hasattr(RelationshipType, "EVOLVED_TO")
        assert hasattr(RelationshipType, "RELATES_TO")
        assert hasattr(RelationshipType, "CAUSES")
        assert hasattr(RelationshipType, "SUPPORTS")
        assert hasattr(RelationshipType, "CONTRADICTS")
        assert hasattr(RelationshipType, "MENTIONS")

    def test_relationship_type_values(self):
        """Test relationship type string values"""
        assert RelationshipType.EVOLVED_TO.value == "EVOLVED_TO"
        assert RelationshipType.RELATES_TO.value == "RELATES_TO"
        assert RelationshipType.CAUSES.value == "CAUSES"

    def test_relationship_type_from_string(self):
        """Test creating relationship type from string"""
        rel_type = RelationshipType("RELATES_TO")
        assert rel_type == RelationshipType.RELATES_TO


class TestEdge:
    """Tests for Edge model"""

    def test_edge_creation(self):
        """Test creating an edge"""
        source_id = uuid4()
        target_id = uuid4()

        edge = Edge(
            source_state_id=source_id,
            target_state_id=target_id,
            relationship_type=RelationshipType.RELATES_TO,
            strength=0.8,
        )

        assert edge.source_state_id == source_id
        assert edge.target_state_id == target_id
        assert edge.relationship_type == RelationshipType.RELATES_TO
        assert edge.strength == 0.8
        assert edge.is_active is True  # No valid_until set

    def test_edge_defaults(self):
        """Test edge default values"""
        edge = Edge(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
        )

        assert edge.strength == 0.5  # Default
        assert edge.description is None
        assert edge.valid_until is None
        assert edge.superseded_by is None
        assert edge.custom_metadata == {}
        assert edge.is_active is True

    def test_edge_is_active_with_future_expiry(self):
        """Test edge is active with future expiry"""
        edge = Edge(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
            valid_until=datetime.utcnow() + timedelta(days=1),
        )

        assert edge.is_active is True

    def test_edge_is_inactive_with_past_expiry(self):
        """Test edge is inactive with past expiry"""
        edge = Edge(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
            valid_until=datetime.utcnow() - timedelta(days=1),
        )

        assert edge.is_active is False

    def test_edge_strength_validation(self):
        """Test edge strength is validated to 0-1 range"""
        # Valid strengths
        edge1 = Edge(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
            strength=0.0,
        )
        assert edge1.strength == 0.0

        edge2 = Edge(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
            strength=1.0,
        )
        assert edge2.strength == 1.0

        # Invalid strength should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            Edge(
                source_state_id=uuid4(),
                target_state_id=uuid4(),
                relationship_type=RelationshipType.RELATES_TO,
                strength=1.5,
            )

    def test_edge_to_neo4j_properties(self):
        """Test converting edge to Neo4j properties"""
        edge = Edge(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.CAUSES,
            strength=0.9,
            description="Test edge",
            custom_metadata={"key": "value"},
        )

        props = edge.to_neo4j_properties()

        assert "relationship_type" in props
        assert props["relationship_type"] == "causes"
        assert props["strength"] == 0.9
        assert props["description"] == "Test edge"
        assert props["is_active"] is True
        assert props["metadata"] == {"key": "value"}


class TestEdgeStrengthFactors:
    """Tests for EdgeStrengthFactors model"""

    def test_edge_strength_factors_creation(self):
        """Test creating edge strength factors"""
        factors = EdgeStrengthFactors(
            semantic_similarity=0.8,
            temporal_proximity=0.6,
            explicit_weight=0.9,
        )

        assert factors.semantic_similarity == 0.8
        assert factors.temporal_proximity == 0.6
        assert factors.explicit_weight == 0.9
        assert factors.co_occurrence is None
        assert factors.access_pattern is None

    def test_edge_strength_factors_validation(self):
        """Test edge strength factors are validated to 0-1 range"""
        # Valid factors
        factors = EdgeStrengthFactors(
            semantic_similarity=0.0,
            temporal_proximity=1.0,
        )
        assert factors.semantic_similarity == 0.0
        assert factors.temporal_proximity == 1.0

        # Invalid factor should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            EdgeStrengthFactors(semantic_similarity=1.5)


class TestGraphService:
    """Tests for GraphService"""

    @pytest.fixture
    def graph_service(self):
        """Create a graph service instance"""
        return GraphService()

    @pytest.mark.asyncio
    async def test_service_instantiation(self, graph_service):
        """Test that graph service can be instantiated"""
        assert graph_service is not None
        assert isinstance(graph_service, GraphService)

    @pytest.mark.asyncio
    async def test_create_edge_method_exists(self, graph_service):
        """Test that create_edge method exists"""
        assert hasattr(graph_service, "create_edge")
        assert callable(graph_service.create_edge)

    @pytest.mark.asyncio
    async def test_update_edge_strength_method_exists(self, graph_service):
        """Test that update_edge_strength method exists"""
        assert hasattr(graph_service, "update_edge_strength")
        assert callable(graph_service.update_edge_strength)

    @pytest.mark.asyncio
    async def test_mark_edge_obsolete_method_exists(self, graph_service):
        """Test that mark_edge_obsolete method exists"""
        assert hasattr(graph_service, "mark_edge_obsolete")
        assert callable(graph_service.mark_edge_obsolete)

    @pytest.mark.asyncio
    async def test_find_related_states_method_exists(self, graph_service):
        """Test that find_related_states method exists"""
        assert hasattr(graph_service, "find_related_states")
        assert callable(graph_service.find_related_states)

    @pytest.mark.asyncio
    async def test_get_inverse_relationship(self, graph_service):
        """Test getting inverse relationships"""
        # Forward and inverse pairs
        assert (
            await graph_service.get_inverse_relationship(RelationshipType.CAUSES)
            == RelationshipType.CAUSED_BY
        )
        assert (
            await graph_service.get_inverse_relationship(RelationshipType.CAUSED_BY)
            == RelationshipType.CAUSES
        )
        assert (
            await graph_service.get_inverse_relationship(RelationshipType.SUPPORTS)
            == RelationshipType.SUPPORTED_BY
        )

        # Self-inverse relationships
        assert (
            await graph_service.get_inverse_relationship(RelationshipType.RELATES_TO)
            == RelationshipType.RELATES_TO
        )
        assert (
            await graph_service.get_inverse_relationship(RelationshipType.SIMILAR_TO)
            == RelationshipType.SIMILAR_TO
        )

    @pytest.mark.asyncio
    async def test_get_default_strength(self, graph_service):
        """Test default strength calculation for relationship types"""
        # Strong relationships should have high default strength
        strong_strength = graph_service._get_default_strength(RelationshipType.EVOLVED_TO)
        assert strong_strength == 0.8

        # Medium relationships
        medium_strength = graph_service._get_default_strength(RelationshipType.DEPENDS_ON)
        assert medium_strength == 0.6

        # Weak relationships
        weak_strength = graph_service._get_default_strength(RelationshipType.RELATES_TO)
        assert weak_strength == 0.4

    @pytest.mark.asyncio
    async def test_calculate_edge_strength_with_factors(self, graph_service):
        """Test edge strength calculation with provided factors"""
        factors = EdgeStrengthFactors(
            semantic_similarity=0.9,
            temporal_proximity=0.7,
            explicit_weight=0.8,
        )

        strength = await graph_service._calculate_edge_strength(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
            factors=factors,
        )

        # Strength should be between 0 and 1
        assert 0.0 <= strength <= 1.0
        # Should be reasonably high given the factors
        assert strength > 0.5

    @pytest.mark.asyncio
    async def test_calculate_edge_strength_default(self, graph_service):
        """Test edge strength calculation with default factors"""
        strength = await graph_service._calculate_edge_strength(
            source_state_id=uuid4(),
            target_state_id=uuid4(),
            relationship_type=RelationshipType.CAUSES,
        )

        # Strength should be between 0 and 1
        assert 0.0 <= strength <= 1.0


class TestGraphIntegration:
    """Integration tests for graph functionality"""

    @pytest.mark.asyncio
    async def test_graph_service_imports(self):
        """Test that graph service can be imported from services"""
        from tracingrag.services import GraphService

        assert GraphService is not None

    @pytest.mark.asyncio
    async def test_graph_models_imports(self):
        """Test that graph models can be imported from core.models"""
        from tracingrag.core.models import Edge, EdgeStrengthFactors, RelationshipType

        assert Edge is not None
        assert EdgeStrengthFactors is not None
        assert RelationshipType is not None

    @pytest.mark.asyncio
    async def test_create_multiple_edges(self):
        """Test creating multiple edge instances"""
        edges = [
            Edge(
                source_state_id=uuid4(),
                target_state_id=uuid4(),
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.5 + i * 0.1,
            )
            for i in range(5)
        ]

        assert len(edges) == 5
        assert all(isinstance(e, Edge) for e in edges)
        # Strengths should be increasing
        assert edges[0].strength < edges[4].strength
