# TracingRAG REST API Guide

This guide explains how to use TracingRAG's RESTful API to interact with the system.

## Overview

TracingRAG provides a comprehensive REST API built with FastAPI. All endpoints follow RESTful conventions and return JSON responses.

**Base URL**: `http://localhost:8000` (default development server)

**API Documentation**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Starting the API Server

```bash
# Using Poetry
poetry run uvicorn tracingrag.api.main:app --reload

# Or with custom host/port
poetry run uvicorn tracingrag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Authentication

TracingRAG supports multiple authentication methods for production deployments:

- **JWT (JSON Web Tokens)**: For user-based authentication
- **API Keys**: For service-to-service authentication
- **Rate Limiting**: Built-in protection against abuse

For local development, authentication is disabled by default. To enable authentication in production, configure the following in your `.env`:

```env
SECRET_KEY=your-secret-key-here-change-in-production
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

See `tracingrag/api/security.py` for authentication implementation details.

## API Endpoints

### System Endpoints

#### Get API Information
```http
GET /
```

**Response**:
```json
{
  "name": "TracingRAG API",
  "version": "0.2.0",
  "description": "Enhanced RAG system with temporal tracing, graph relationships, and agentic retrieval",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

#### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "services": {
    "memory_service": "healthy",
    "rag_service": "healthy",
    "agent_service": "healthy",
    "promotion_service": "healthy"
  }
}
```

#### Get System Metrics
```http
GET /metrics
```

**Response**:
```json
{
  "total_memories": 150,
  "total_topics": 42,
  "total_promotions": 8,
  "avg_versions_per_topic": 3.57,
  "uptime_seconds": 3600.5
}
```

---

### Memory Endpoints

#### Create Memory State
```http
POST /api/v1/memories
```

**Request Body**:
```json
{
  "topic": "project_alpha",
  "content": "Initial design for API authentication system",
  "parent_state_id": null,
  "metadata": {
    "author": "Alice",
    "priority": "high"
  },
  "tags": ["design", "security"],
  "confidence": 0.95,
  "source": "design_meeting"
}
```

**Response** (201 Created):
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "topic": "project_alpha",
  "content": "Initial design for API authentication system",
  "version": 1,
  "timestamp": "2025-01-15T10:30:00Z",
  "parent_state_id": null,
  "metadata": {
    "author": "Alice",
    "priority": "high"
  },
  "tags": ["design", "security"],
  "confidence": 0.95,
  "source": "design_meeting"
}
```

#### Get Memory by ID
```http
GET /api/v1/memories/{memory_id}
```

**Response** (200 OK):
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "topic": "project_alpha",
  "content": "Initial design for API authentication system",
  "version": 1,
  "timestamp": "2025-01-15T10:30:00Z",
  "parent_state_id": null,
  "metadata": {
    "author": "Alice",
    "priority": "high"
  },
  "tags": ["design", "security"],
  "confidence": 0.95,
  "source": "design_meeting"
}
```

**Error Response** (404 Not Found):
```json
{
  "detail": "Memory with ID 123e4567-e89b-12d3-a456-426614174000 not found"
}
```

#### List Memories
```http
GET /api/v1/memories?topic={topic}&limit={limit}&offset={offset}
```

**Query Parameters**:
- `topic` (optional): Filter by topic
- `limit` (optional, default=50): Number of results to return
- `offset` (optional, default=0): Number of results to skip

**Example**:
```http
GET /api/v1/memories?topic=project_alpha&limit=10&offset=0
```

**Response** (200 OK):
```json
{
  "memories": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "topic": "project_alpha",
      "content": "Initial design for API authentication system",
      "version": 1,
      "timestamp": "2025-01-15T10:30:00Z",
      "parent_state_id": null,
      "metadata": {},
      "tags": ["design", "security"],
      "confidence": 0.95,
      "source": "design_meeting"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

#### Get Trace History
```http
GET /api/v1/traces/{topic}?limit={limit}
```

**Query Parameters**:
- `limit` (optional, default=100): Maximum number of versions to return

**Example**:
```http
GET /api/v1/traces/project_alpha?limit=50
```

**Response** (200 OK):
```json
{
  "memories": [
    {
      "id": "789...",
      "topic": "project_alpha",
      "content": "Latest version after security review",
      "version": 3,
      "timestamp": "2025-01-17T14:00:00Z",
      "parent_state_id": "456...",
      "metadata": {},
      "tags": ["design", "security", "reviewed"],
      "confidence": 0.98,
      "source": "security_review"
    },
    {
      "id": "456...",
      "topic": "project_alpha",
      "content": "Updated design based on feedback",
      "version": 2,
      "timestamp": "2025-01-16T09:15:00Z",
      "parent_state_id": "123...",
      "metadata": {},
      "tags": ["design", "security"],
      "confidence": 0.95,
      "source": "design_meeting"
    },
    {
      "id": "123...",
      "topic": "project_alpha",
      "content": "Initial design for API authentication system",
      "version": 1,
      "timestamp": "2025-01-15T10:30:00Z",
      "parent_state_id": null,
      "metadata": {},
      "tags": ["design", "security"],
      "confidence": 0.95,
      "source": "design_meeting"
    }
  ],
  "total": 3,
  "limit": 50,
  "offset": 0
}
```

---

### Query/RAG Endpoints

#### Query the RAG System
```http
POST /api/v1/query
```

**Request Body**:
```json
{
  "query": "What's the status of project alpha?",
  "include_history": true,
  "include_related": true,
  "depth": 2,
  "limit": 10,
  "use_agent": false
}
```

**Parameters**:
- `query` (required): The user's question
- `include_history` (optional, default=true): Include historical versions
- `include_related` (optional, default=true): Include related states via graph
- `depth` (optional, default=2, range=1-5): Graph traversal depth
- `limit` (optional, default=10, range=1-100): Max results to retrieve
- `use_agent` (optional, default=false): Use agent-based retrieval

**Response** (200 OK):
```json
{
  "answer": "Project Alpha is currently in security review phase. The initial design was created on Jan 15th, updated based on feedback on Jan 16th, and reviewed by the security team on Jan 17th. The authentication system uses JWT tokens with OAuth2 integration.",
  "sources": [
    {
      "id": "789...",
      "topic": "project_alpha",
      "content": "Latest version after security review",
      "version": 3,
      "timestamp": "2025-01-17T14:00:00Z",
      "parent_state_id": "456...",
      "metadata": {},
      "tags": ["design", "security", "reviewed"],
      "confidence": 0.98,
      "source": "security_review"
    }
  ],
  "confidence": 0.92,
  "reasoning": "Retrieved latest state of project_alpha, found historical context showing evolution from initial design through feedback and security review",
  "metadata": {
    "retrieval_time_ms": 45,
    "generation_time_ms": 850,
    "total_time_ms": 895,
    "num_states_retrieved": 3,
    "context_tokens": 512,
    "completion_tokens": 67
  }
}
```

**Example with Agent-Based Retrieval**:
```json
{
  "query": "Compare the initial design of project alpha with its current implementation",
  "use_agent": true
}
```

**Response** (200 OK):
```json
{
  "answer": "The project evolved significantly from the initial design...",
  "sources": [...],
  "confidence": 0.88,
  "reasoning": "Agent executed multi-step retrieval: 1) Retrieved trace history, 2) Identified initial vs current versions, 3) Compared design approaches, 4) Traversed related bug reports",
  "metadata": {
    "agent_steps": 4,
    "retrieval_time_ms": 120,
    "generation_time_ms": 950,
    "total_time_ms": 1070
  }
}
```

---

### Promotion Endpoints

#### Promote Memory
```http
POST /api/v1/promote
```

**Request Body**:
```json
{
  "topic": "project_alpha",
  "reason": "Consolidating 5 versions after feature completion",
  "include_related": true,
  "max_sources": 10
}
```

**Parameters**:
- `topic` (required): Topic to promote
- `reason` (required): Reason for promotion
- `include_related` (optional, default=true): Include related states in synthesis
- `max_sources` (optional, default=10, range=1-50): Max sources to synthesize from

**Response** (200 OK):
```json
{
  "success": true,
  "topic": "project_alpha",
  "new_version": 4,
  "previous_state_id": "789...",
  "new_state_id": "abc...",
  "synthesized_from_count": 3,
  "conflicts_detected_count": 1,
  "conflicts_resolved_count": 1,
  "edges_updated_count": 5,
  "quality_checks_count": 3,
  "reasoning": "Successfully consolidated 3 versions of project_alpha. Resolved 1 conflict between v2 and v3 regarding authentication method. Carried forward 5 high-strength edges to related bug reports and design documents.",
  "confidence": 0.92,
  "manual_review_needed": false,
  "error_message": null
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "detail": "Promotion failed: No states found for topic 'nonexistent_topic'"
}
```

#### Get Promotion Candidates
```http
GET /api/v1/promotion-candidates?limit={limit}&min_priority={min_priority}
```

**Query Parameters**:
- `limit` (optional, default=10): Number of candidates to return
- `min_priority` (optional, default=5): Minimum priority score (1-10)

**Example**:
```http
GET /api/v1/promotion-candidates?limit=5&min_priority=7
```

**Response** (200 OK):
```json
{
  "candidates": [
    {
      "topic": "bug_authentication",
      "trigger": "auto_version_count",
      "priority": 8,
      "reasoning": "12 versions accumulated over 3 days with iterative bug fixes. High activity suggests consolidation would improve clarity.",
      "current_version_count": 12,
      "last_promoted": "2025-01-10T08:00:00Z",
      "confidence": 0.85
    },
    {
      "topic": "feature_dashboard",
      "trigger": "auto_related_growth",
      "priority": 7,
      "reasoning": "15 related states detected with strong connections. Consolidation would create comprehensive overview.",
      "current_version_count": 5,
      "last_promoted": null,
      "confidence": 0.78
    }
  ],
  "total": 2
}
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Request succeeded
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

**Error Response Format**:
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Validation Error Format** (422):
```json
{
  "detail": [
    {
      "loc": ["body", "topic"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## CORS Configuration

The API allows cross-origin requests from all origins by default (for development). Configure this in production:

```python
# tracingrag/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Usage Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Create a memory
response = requests.post(
    f"{BASE_URL}/api/v1/memories",
    json={
        "topic": "project_alpha",
        "content": "Initial design for API authentication",
        "tags": ["design", "security"],
        "confidence": 0.95
    }
)
memory = response.json()
print(f"Created memory: {memory['id']}")

# Query the RAG system
response = requests.post(
    f"{BASE_URL}/api/v1/query",
    json={
        "query": "What's the status of project alpha?",
        "use_agent": False
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")

# Get promotion candidates
response = requests.get(
    f"{BASE_URL}/api/v1/promotion-candidates",
    params={"limit": 10, "min_priority": 7}
)
candidates = response.json()
print(f"Found {candidates['total']} candidates for promotion")
```

### JavaScript/TypeScript

```typescript
const BASE_URL = "http://localhost:8000";

// Create a memory
const createMemory = async () => {
  const response = await fetch(`${BASE_URL}/api/v1/memories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic: "project_alpha",
      content: "Initial design for API authentication",
      tags: ["design", "security"],
      confidence: 0.95
    })
  });
  const memory = await response.json();
  console.log(`Created memory: ${memory.id}`);
};

// Query the RAG system
const queryRAG = async () => {
  const response = await fetch(`${BASE_URL}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "What's the status of project alpha?",
      use_agent: false
    })
  });
  const result = await response.json();
  console.log(`Answer: ${result.answer}`);
  console.log(`Confidence: ${result.confidence}`);
};

// Get promotion candidates
const getCandidates = async () => {
  const response = await fetch(
    `${BASE_URL}/api/v1/promotion-candidates?limit=10&min_priority=7`
  );
  const candidates = await response.json();
  console.log(`Found ${candidates.total} candidates for promotion`);
};
```

### cURL

```bash
# Create a memory
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "project_alpha",
    "content": "Initial design for API authentication",
    "tags": ["design", "security"],
    "confidence": 0.95
  }'

# Query the RAG system
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the status of project alpha?",
    "use_agent": false
  }'

# Get promotion candidates
curl -X GET "http://localhost:8000/api/v1/promotion-candidates?limit=10&min_priority=7"

# Check health
curl -X GET http://localhost:8000/health
```

---

## Rate Limiting

Rate limiting will be added in Phase 8 (Production Ready). Current development server has no rate limits.

---

## Performance Tips

1. **Pagination**: Use `limit` and `offset` parameters for large result sets
2. **Filtering**: Filter by topic when possible to reduce result size
3. **Agent Mode**: Use `use_agent=true` only for complex queries requiring multi-step reasoning
4. **Depth Control**: Use lower `depth` values (1-2) for faster graph traversal
5. **Source Limit**: Reduce `max_sources` for promotion to speed up synthesis

---

## WebSocket Support

WebSocket support for real-time updates will be added in a future phase.

---

## Next Steps

- See [AUTOMATIC_PROMOTION_GUIDE.md](./AUTOMATIC_PROMOTION_GUIDE.md) for advanced promotion features
- See [QUICK_START.md](./QUICK_START.md) for Python SDK usage
- See [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md) for upcoming features
- Visit `/docs` endpoint for interactive API documentation
