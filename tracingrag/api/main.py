"""FastAPI application for TracingRAG"""

import time
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
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
    print("Initializing TracingRAG services...")
    memory_service = MemoryService()
    rag_service = RAGService()
    agent_service = AgentService()
    promotion_service = PromotionService()
    print("TracingRAG services initialized successfully")

    yield

    # Shutdown
    print("Shutting down TracingRAG services...")
    # Close any connections if needed


# Create FastAPI app
app = FastAPI(
    title="TracingRAG API",
    description="Enhanced RAG system with temporal tracing, graph relationships, and agentic retrieval",
    version="0.1.0",
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

        api_active_requests.labels(
            method=request.method, endpoint=request.url.path
        ).inc()

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
            api_active_requests.labels(
                method=request.method, endpoint=request.url.path
            ).dec()


# Add metrics middleware
app.add_middleware(MetricsMiddleware)


# ============================================================================
# Helper Functions
# ============================================================================


def memory_db_to_response(memory: MemoryStateDB) -> MemoryStateResponse:
    """Convert database model to API response model"""
    return MemoryStateResponse(
        id=memory.id,
        topic=memory.topic,
        content=memory.content,
        version=memory.version,
        timestamp=memory.timestamp,
        parent_state_id=memory.parent_state_id,
        metadata=memory.custom_metadata,
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

    overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        services=services_status,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """Get system metrics"""
    async with get_session() as session:
        # Count total memories
        total_memories_result = await session.execute(
            select(func.count(MemoryStateDB.id))
        )
        total_memories = total_memories_result.scalar() or 0

        # Count unique topics
        total_topics_result = await session.execute(
            select(func.count(func.distinct(MemoryStateDB.topic)))
        )
        total_topics = total_topics_result.scalar() or 0

        # Calculate average versions per topic
        avg_versions = total_memories / total_topics if total_topics > 0 else 0

        # Count promotions (states with "promoted" tag)
        promotions_result = await session.execute(
            select(func.count(MemoryStateDB.id)).where(
                MemoryStateDB.tags.contains(["promoted"])
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


# ============================================================================
# Query/RAG Endpoints
# ============================================================================


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    try:
        if request.use_agent:
            # Use agent-based retrieval
            result = await agent_service.query_with_agent(query=request.query)

            return QueryResponse(
                answer=result.answer,
                sources=[memory_db_to_response(s) for s in result.states],
                confidence=result.confidence,
                reasoning=result.reasoning,
                metadata=result.metadata,
            )
        else:
            # Use standard RAG
            result = await rag_service.query(
                query=request.query,
                limit=request.limit,
            )

            return QueryResponse(
                answer=result.answer,
                sources=[memory_db_to_response(s) for s in result.states],
                confidence=result.confidence,
                metadata=result.metadata,
            )

    except Exception as e:
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

        return PromoteMemoryResponse(
            success=result.success,
            topic=result.topic,
            new_version=result.new_version,
            previous_state_id=result.previous_state_id,
            new_state_id=result.new_state_id,
            synthesized_from_count=len(result.synthesized_from),
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
        "version": "0.1.0",
        "description": "Enhanced RAG system with temporal tracing, graph relationships, and agentic retrieval",
        "docs": "/docs",
        "health": "/health",
        "prometheus_metrics": "/metrics",
    }
