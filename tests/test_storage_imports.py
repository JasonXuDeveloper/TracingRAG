"""Tests for storage module imports and basic functionality"""



class TestStorageImports:
    """Test that all storage modules can be imported"""

    def test_import_database(self):
        """Test importing database module"""
        from tracingrag.storage import database

        assert hasattr(database, "get_engine")
        assert hasattr(database, "get_session")
        assert hasattr(database, "init_db")
        assert hasattr(database, "close_db")

    def test_import_models(self):
        """Test importing models module"""
        from tracingrag.storage.models import MemoryStateDB, TopicLatestStateDB, TraceDB

        assert MemoryStateDB is not None
        assert TraceDB is not None
        assert TopicLatestStateDB is not None

    def test_import_qdrant(self):
        """Test importing Qdrant module"""
        from tracingrag.storage import qdrant

        assert hasattr(qdrant, "get_qdrant_client")
        assert hasattr(qdrant, "init_qdrant_collection")
        assert hasattr(qdrant, "upsert_embedding")
        assert hasattr(qdrant, "search_similar")
        assert hasattr(qdrant, "delete_embedding")
        assert hasattr(qdrant, "batch_upsert_embeddings")

    def test_import_neo4j_client(self):
        """Test importing Neo4j client module"""
        from tracingrag.storage import neo4j_client

        assert hasattr(neo4j_client, "get_neo4j_driver")
        assert hasattr(neo4j_client, "verify_neo4j_connection")
        assert hasattr(neo4j_client, "init_neo4j_schema")
        assert hasattr(neo4j_client, "create_memory_node")
        assert hasattr(neo4j_client, "create_evolution_edge")
        assert hasattr(neo4j_client, "create_entity_node")
        assert hasattr(neo4j_client, "create_entity_relationship")
        assert hasattr(neo4j_client, "get_topic_history")
        assert hasattr(neo4j_client, "get_related_memories")


class TestServiceImports:
    """Test that all service modules can be imported"""

    def test_import_embedding_service(self):
        """Test importing embedding service"""
        from tracingrag.services import embedding

        assert hasattr(embedding, "generate_embedding")
        assert hasattr(embedding, "generate_embeddings_batch")
        assert hasattr(embedding, "compute_similarity")
        assert hasattr(embedding, "get_embedding_dimension")

    def test_import_memory_service(self):
        """Test importing memory service"""
        from tracingrag.services.memory import MemoryService

        assert MemoryService is not None
        service = MemoryService()
        assert hasattr(service, "create_memory_state")
        assert hasattr(service, "get_memory_state")
        assert hasattr(service, "get_latest_state")
        assert hasattr(service, "get_topic_history")
        assert hasattr(service, "update_memory_state")


class TestModels:
    """Test model instantiation"""

    def test_memory_state_model(self):
        """Test MemoryStateDB model instantiation"""
        from datetime import datetime
        from uuid import uuid4

        from tracingrag.storage.models import MemoryStateDB

        state = MemoryStateDB(
            id=uuid4(),
            topic="test",
            content="test content",
            version=1,
            timestamp=datetime.utcnow(),
        )

        assert state.topic == "test"
        assert state.content == "test content"
        assert state.version == 1
        # Note: SQLAlchemy defaults are applied at DB insert time, not instantiation
        # These fields will be None until inserted into database
        assert state.custom_metadata is None or state.custom_metadata == {}
        assert state.tags is None or state.tags == []
        assert state.confidence is None or state.confidence == 1.0

    def test_trace_model(self):
        """Test TraceDB model instantiation"""
        from datetime import datetime
        from uuid import uuid4

        from tracingrag.storage.models import TraceDB

        trace = TraceDB(
            id=uuid4(),
            topic="test_trace",
            state_ids=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert trace.topic == "test_trace"
        assert trace.state_ids == []
        # Note: SQLAlchemy defaults are applied at DB insert time, not instantiation
        assert trace.is_active is None or trace.is_active is True

    def test_topic_latest_state_model(self):
        """Test TopicLatestStateDB model instantiation"""
        from datetime import datetime
        from uuid import uuid4

        from tracingrag.storage.models import TopicLatestStateDB

        mapping = TopicLatestStateDB(
            topic="test_topic",
            latest_state_id=uuid4(),
            updated_at=datetime.utcnow(),
        )

        assert mapping.topic == "test_topic"
        assert mapping.latest_state_id is not None
