"""FastAPI application for TracingRAG"""

import time
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import String, cast, func, literal, select
from sqlalchemy.dialects.postgresql import ARRAY
from starlette.middleware.base import BaseHTTPMiddleware

from tracingrag.agents.service import AgentService
from tracingrag.api.schemas import (
    CreateMemoryRequest,
    HealthResponse,
    MemoryListResponse,
    MemoryStateResponse,
    MetricsResponse,
    PromoteMemoryRequest,
    PromoteMemoryResponse,
    PromotionCandidateResponse,
    PromotionCandidatesResponse,
    QueryRequest,
    QueryResponse,
)
from tracingrag.core.models.promotion import PromotionRequest as PromotionServiceRequest
from tracingrag.services.memory import MemoryService
from tracingrag.services.metrics import MetricsCollector, get_content_type, get_metrics
from tracingrag.services.promotion import PromotionService
from tracingrag.services.rag import RAGService
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

# Global services
memory_service: MemoryService | None = None
rag_service: RAGService | None = None
agent_service: AgentService | None = None
promotion_service: PromotionService | None = None
app_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global memory_service, rag_service, agent_service, promotion_service

    # Startup
    logger.info("Initializing TracingRAG services...")
    # Initialize database
    from tracingrag.storage.database import init_db

    await init_db()
    logger.info("✓ Database initialized")
    # Initialize Qdrant collection
    try:
        from tracingrag.services.embedding import get_embedding_dimension
        from tracingrag.storage.qdrant import init_qdrant_collection

        embedding_dim = get_embedding_dimension()  # Synchronous function, no await
        await init_qdrant_collection(
            collection_name="memory_states",
            vector_size=embedding_dim,
        )
        logger.info(f"✓ Qdrant collection initialized (dimension: {embedding_dim})")
    except Exception as e:
        logger.error(f"⚠ Qdrant initialization failed (will retry on first use): {e}")
    # Initialize Neo4j schema (indexes and constraints)
    try:
        from tracingrag.storage.neo4j_client import init_neo4j_schema

        await init_neo4j_schema()
        logger.info("✓ Neo4j schema initialized")
    except Exception as e:
        logger.error(f"⚠ Neo4j initialization failed (optional): {e}")
    # Initialize services (with lazy loading, no LLM client needed)
    from tracingrag.config import settings
    from tracingrag.core.models.promotion import PromotionMode, PromotionPolicy

    memory_service = MemoryService()
    rag_service = RAGService()
    agent_service = AgentService()

    # Initialize promotion service with auto-promotion policy if enabled
    promotion_policy = PromotionPolicy(
        mode=PromotionMode.AUTOMATIC if settings.auto_promotion_enabled else PromotionMode.MANUAL,
        confidence_threshold=settings.promotion_confidence_threshold,
    )
    promotion_service = PromotionService(policy=promotion_policy)

    # Print model configuration
    logger.warning("\n" + "=" * 70)
    logger.warning("🤖 LLM Model Configuration:")
    logger.warning("=" * 70)
    logger.warning(f"  Main Query Model:      {settings.default_llm_model}")
    logger.warning(f"  Fallback Model:        {settings.fallback_llm_model}")
    logger.warning(f"  Analysis Model:        {settings.analysis_model}")
    logger.warning(f"  Evaluation Model:      {settings.evaluation_model}")
    logger.warning(f"  Query Analyzer Model:  {settings.query_analyzer_model}")
    logger.warning(f"  Planner Model:         {settings.planner_model}")
    logger.warning(f"  Manager Model:         {settings.manager_model}")
    logger.warning("=" * 70)
    if settings.auto_promotion_enabled:
        logger.warning("✓ TracingRAG services initialized (Auto-promotion: ENABLED)")
    else:
        logger.warning("✓ TracingRAG services initialized (Auto-promotion: Manual)")
    logger.warning("")
    yield

    # Shutdown
    logger.info("Shutting down TracingRAG services...")
    from tracingrag.storage.database import close_db
    from tracingrag.storage.neo4j_client import close_neo4j
    from tracingrag.storage.qdrant import close_qdrant

    await close_db()
    await close_qdrant()
    await close_neo4j()
    logger.info("✓ Connections closed")


# Create FastAPI app
app = FastAPI(
    title="TracingRAG API",
    description="Enhanced RAG system with temporal tracing, graph relationships, and agentic retrieval",
    version="0.2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware
# ============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track API request metrics"""

    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()

        # Track active requests
        from tracingrag.services.metrics import api_active_requests

        api_active_requests.labels(method=request.method, endpoint=request.url.path).inc()

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Record metrics
            MetricsCollector.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration,
            )

            return response
        finally:
            # Decrease active requests
            api_active_requests.labels(method=request.method, endpoint=request.url.path).dec()


# Add metrics middleware
app.add_middleware(MetricsMiddleware)


# ============================================================================
# Helper Functions
# ============================================================================


def serialize_for_json(obj):
    """Recursively convert UUIDs and other non-serializable objects to JSON-safe types"""
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_for_json(item) for item in obj)
    else:
        return obj


def memory_db_to_response(memory: MemoryStateDB) -> MemoryStateResponse:
    """Convert database model to API response model"""
    return MemoryStateResponse(
        id=memory.id,
        topic=memory.topic,
        content=memory.content,
        version=memory.version,
        timestamp=memory.timestamp,
        parent_state_id=memory.parent_state_id,
        metadata=serialize_for_json(memory.custom_metadata or {}),
        tags=memory.tags,
        confidence=memory.confidence,
        source=memory.source,
    )


# ============================================================================
# Health & Metrics Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health"""
    services_status = {
        "memory_service": "healthy" if memory_service else "unavailable",
        "rag_service": "healthy" if rag_service else "unavailable",
        "agent_service": "healthy" if agent_service else "unavailable",
        "promotion_service": "healthy" if promotion_service else "unavailable",
    }

    overall_status = (
        "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        version="0.2.0",
        services=services_status,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["System"])
async def get_application_metrics():
    """Get application metrics in JSON format"""
    async with get_session() as session:
        # Count total memories
        total_memories_result = await session.execute(select(func.count(MemoryStateDB.id)))
        total_memories = total_memories_result.scalar() or 0

        # Count unique topics
        total_topics_result = await session.execute(
            select(func.count(func.distinct(MemoryStateDB.topic)))
        )
        total_topics = total_topics_result.scalar() or 0

        # Calculate average versions per topic
        avg_versions = total_memories / total_topics if total_topics > 0 else 0

        # Count promotions (states with "promoted" tag)
        # Check database dialect to use appropriate query
        dialect_name = session.bind.dialect.name
        if dialect_name == "postgresql":
            # Use overlap operator for PostgreSQL ARRAY type: tags && ARRAY['promoted']
            promotions_result = await session.execute(
                select(func.count(MemoryStateDB.id)).where(
                    MemoryStateDB.tags.op("&&")(cast(literal(["promoted"]), ARRAY(String)))
                )
            )
        else:
            # For SQLite, tags are stored as JSON text, so use LIKE for simple search
            promotions_result = await session.execute(
                select(func.count(MemoryStateDB.id)).where(
                    MemoryStateDB.tags.cast(String).like("%promoted%")
                )
            )
        total_promotions = promotions_result.scalar() or 0

    uptime = time.time() - app_start_time

    return MetricsResponse(
        total_memories=total_memories,
        total_topics=total_topics,
        total_promotions=total_promotions,
        avg_versions_per_topic=avg_versions,
        uptime_seconds=uptime,
    )


# ============================================================================
# Memory Endpoints
# ============================================================================


@app.post(
    "/api/v1/memories",
    response_model=MemoryStateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Memory"],
)
async def create_memory(request: CreateMemoryRequest):
    """Create a new memory state"""
    try:
        memory = await memory_service.create_memory_state(
            topic=request.topic,
            content=request.content,
            parent_state_id=request.parent_state_id,
            metadata=request.metadata,
            tags=request.tags,
            confidence=request.confidence,
            source=request.source,
        )

        # Check if auto-promotion should be triggered
        await promotion_service.evaluate_after_insertion(
            topic=request.topic,
            new_state_id=memory.id,
        )

        return memory_db_to_response(memory)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory: {str(e)}",
        )


@app.get("/api/v1/memories/{memory_id}", response_model=MemoryStateResponse, tags=["Memory"])
async def get_memory(memory_id: UUID):
    """Get a memory state by ID"""
    try:
        memory = await memory_service.get_memory_state(memory_id)

        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory with ID {memory_id} not found",
            )

        return memory_db_to_response(memory)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}",
        )


@app.get("/api/v1/memories", response_model=MemoryListResponse, tags=["Memory"])
async def list_memories(
    topic: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List memory states with optional topic filter"""
    try:
        async with get_session() as session:
            query = select(MemoryStateDB)

            if topic:
                query = query.where(MemoryStateDB.topic == topic)

            query = query.order_by(MemoryStateDB.timestamp.desc()).limit(limit).offset(offset)

            result = await session.execute(query)
            memories = result.scalars().all()

            # Count total
            count_query = select(func.count(MemoryStateDB.id))
            if topic:
                count_query = count_query.where(MemoryStateDB.topic == topic)

            total_result = await session.execute(count_query)
            total = total_result.scalar() or 0

        return MemoryListResponse(
            memories=[memory_db_to_response(m) for m in memories],
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list memories: {str(e)}",
        )


@app.get("/api/v1/traces/{topic}", response_model=MemoryListResponse, tags=["Memory"])
async def get_trace_history(topic: str, limit: int = 100):
    """Get version history for a topic"""
    try:
        memories = await memory_service.get_topic_history(topic=topic, limit=limit)

        return MemoryListResponse(
            memories=[memory_db_to_response(m) for m in memories],
            total=len(memories),
            limit=limit,
            offset=0,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trace history: {str(e)}",
        )


@app.delete("/api/v1/memories/{memory_id}", tags=["Memory"])
async def delete_memory(memory_id: str):
    """Delete a memory state from all storage layers (PostgreSQL, Qdrant, Neo4j)"""
    try:
        from uuid import UUID

        # Parse UUID
        try:
            state_id = UUID(memory_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid UUID format: {memory_id}",
            )

        # Delete from all storage layers
        deleted = await memory_service.delete_memory_state(state_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory state not found: {memory_id}",
            )

        return {
            "success": True,
            "message": f"Memory state deleted successfully: {memory_id}",
            "id": memory_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}",
        )


@app.delete("/api/v1/memories", tags=["Memory"])
async def cleanup_all_memories(confirm: str | None = None):
    """Delete ALL TracingRAG data from all storage layers

    WARNING: This will permanently delete ALL data!
    - PostgreSQL: MemoryStateDB and TraceDB tables
    - Qdrant: memory_states collection
    - Neo4j: MemoryState nodes and relationships
    - Redis: TracingRAG cache keys

    Args:
        confirm: Must be "DELETE_ALL_DATA" to proceed

    Returns:
        Statistics about deleted data
    """
    # Safety check: require explicit confirmation
    if confirm != "DELETE_ALL_DATA":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Must provide confirmation: confirm="DELETE_ALL_DATA"',
        )

    try:
        stats = await memory_service.cleanup_all_data()

        return {
            "success": True,
            "message": "All TracingRAG data deleted successfully",
            "statistics": stats,
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup data: {str(e)}",
        )


# ============================================================================
# Query/RAG Endpoints
# ============================================================================


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """Query the RAG system

    Uses iterative agent mode by default (max_rounds=5).
    Set max_rounds=0 to use simple RAG mode without agent.
    """
    try:
        if request.max_rounds > 0:
            # Use iterative agent (multi-round with topic history)
            result = await agent_service.query(
                query=request.query,
                max_rounds=request.max_rounds,
                max_tokens_per_round=request.max_tokens_per_round,
            )

            # Fetch memory states from source IDs
            sources = []
            for source_id in result.source_ids:
                memory_state = await memory_service.get_memory_state(source_id)
                if memory_state:
                    sources.append(memory_db_to_response(memory_state))

            # Generate user-friendly reasoning summary
            reasoning = None
            if result.rounds:
                total_rounds = len(result.rounds)
                total_states = result.metadata.get("total_states_used", len(result.source_ids))
                final_confidence = result.confidence

                if total_rounds == 1:
                    reasoning = (
                        f"Retrieved {total_states} relevant documents in 1 round "
                        f"(confidence: {final_confidence:.0%})"
                    )
                else:
                    # Multi-round: show progression
                    topics_explored = []
                    for r in result.rounds:
                        if r.get("needed_topics"):
                            topics_explored.extend(r["needed_topics"])

                    if topics_explored:
                        unique_topics = list(set(topics_explored))
                        reasoning = (
                            f"Analyzed across {total_rounds} rounds, exploring topics: "
                            f"{', '.join(unique_topics[:3])}{'...' if len(unique_topics) > 3 else ''} "
                            f"({total_states} documents, confidence: {final_confidence:.0%})"
                        )
                    else:
                        reasoning = (
                            f"Analyzed across {total_rounds} rounds "
                            f"({total_states} documents, confidence: {final_confidence:.0%})"
                        )

            return QueryResponse(
                answer=result.answer,
                sources=sources,
                confidence=result.confidence,
                key_findings=result.key_findings,
                citations=result.citations,
                uncertainties=result.uncertainties,
                reasoning=reasoning,
                metadata=serialize_for_json(result.metadata),
            )
        else:
            # Simple RAG mode (no agent, direct retrieval)
            result = await rag_service.query_iterative_parallel(query=request.query)

            # Fetch memory states from source IDs
            sources = []
            for source_id in result.sources:
                memory_state = await memory_service.get_memory_state(source_id)
                if memory_state:
                    sources.append(memory_db_to_response(memory_state))

            return QueryResponse(
                answer=result.answer,
                sources=sources,
                confidence=result.confidence,
                metadata=serialize_for_json(result.metadata),
            )

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        logger.error(f"Query error traceback:\n{error_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


# ============================================================================
# Promotion Endpoints
# ============================================================================


@app.post("/api/v1/promote", response_model=PromoteMemoryResponse, tags=["Promotion"])
async def promote_memory(request: PromoteMemoryRequest):
    """Promote a memory state"""
    try:
        service_request = PromotionServiceRequest(
            topic=request.topic,
            reason=request.reason,
            include_related=request.include_related,
            max_sources=request.max_sources,
        )

        result = await promotion_service.promote_memory(service_request)

        # Convert SynthesisSource objects to dicts for JSON serialization
        synthesis_sources = [
            {
                "state_id": str(source.state_id),
                "topic": source.topic,
                "content": source.content[:200] + "..."
                if len(source.content) > 200
                else source.content,
                "version": source.version,
                "timestamp": source.timestamp.isoformat(),
                "confidence": source.confidence,
                "weight": source.weight,
                "reasoning": source.reasoning,
            }
            for source in result.synthesized_from
        ]

        return PromoteMemoryResponse(
            success=result.success,
            topic=result.topic,
            new_version=result.new_version,
            previous_state_id=result.previous_state_id,
            new_state_id=result.new_state_id,
            synthesized_from_count=len(result.synthesized_from),
            synthesis_sources=synthesis_sources,
            conflicts_detected_count=len(result.conflicts_detected),
            conflicts_resolved_count=len(result.conflicts_resolved),
            edges_updated_count=len(result.edges_updated),
            quality_checks_count=len(result.quality_checks),
            reasoning=result.reasoning,
            confidence=result.confidence,
            manual_review_needed=result.manual_review_needed,
            error_message=result.error_message,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Promotion failed: {str(e)}",
        )


@app.get(
    "/api/v1/promotion-candidates",
    response_model=PromotionCandidatesResponse,
    tags=["Promotion"],
)
async def get_promotion_candidates(limit: int = 10, min_priority: int = 5):
    """Get topics that are candidates for promotion"""
    try:
        candidates = await promotion_service.find_promotion_candidates(
            limit=limit, min_priority=min_priority
        )

        return PromotionCandidatesResponse(
            candidates=[
                PromotionCandidateResponse(
                    topic=c.topic,
                    trigger=c.trigger.value,
                    priority=c.priority,
                    reasoning=c.reasoning,
                    current_version_count=c.current_version_count,
                    last_promoted=c.last_promoted,
                    confidence=c.confidence,
                )
                for c in candidates
            ],
            total=len(candidates),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get promotion candidates: {str(e)}",
        )


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=get_metrics(), media_type=get_content_type())


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "TracingRAG API",
        "version": "0.2.0",
        "description": "Enhanced RAG system with temporal tracing, graph relationships, and agentic retrieval",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/api/v1/metrics",
        "prometheus_metrics": "/metrics",
    }
