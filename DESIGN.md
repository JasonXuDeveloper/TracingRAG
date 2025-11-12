# TracingRAG: Temporal Graph-Based RAG System

## Overview

TracingRAG is an enhanced Retrieval-Augmented Generation system that combines:
- **Temporal Tracing**: Snapshots that reveal the evolution of knowledge over time
- **Graph Relationships**: Connections between states and concepts
- **Agentic Retrieval**: Dynamic, intelligent retrieval strategies
- **Memory Promotion**: Synthesis of historical information into new, refined states

## Core Concepts

### 1. Memory States (Traces)
Each piece of knowledge exists as a series of **states** over time:
```
Project Memory (v1) → Project Memory (v2) → Project Memory (v3)
                            ↓ connects to
                        Bug Memory (v1) → Bug Memory (v2)
```

**Properties of a Memory State:**
- `id`: Unique identifier
- `topic`: What this memory is about
- `content`: The actual information/knowledge
- `version`: Version number in the trace
- `timestamp`: When this state was created
- `embedding`: Vector representation for semantic search
- `metadata`: Additional context (source, confidence, etc.)

**Analytics & Storage Management:**
- `access_count`: How many times accessed (analytics only, not used for ranking)
- `last_accessed`: When last retrieved (analytics only)
- `importance_score`: Learned importance 0-1 (can inform edge weights)
- `storage_tier`: working/active/archived (storage optimization)

**Consolidation:**
- `consolidated_from`: States this summarizes
- `is_consolidated`: Whether this is a summary
- `consolidation_level`: 0=raw, 1=daily, 2=weekly, 3=monthly

**Storage Efficiency:**
- `diff_from_parent`: Stores only changes (95% space savings)
- `is_delta`: Stored as diff vs full content

### 2. Traces
A **trace** is the complete history of a topic's evolution:
- Ordered sequence of states
- Maintains causality and temporal relationships
- Non-lossy (all historical data preserved)
- Supports time-travel queries

### 3. Graph Connections
**Edges** represent relationships between states:
- **Temporal edges**: Within a trace (v1 → v2 → v3)
- **Cross-topic edges**: Between different traces (Project → Bug)
- **Causal edges**: Cause-and-effect relationships
- **Reference edges**: Citations or dependencies

**Edge Properties:**
- `source_state_id`: Origin state
- `target_state_id`: Destination state
- `relationship_type`: Type of connection
- `strength`: Relevance/importance score
- `timestamp`: When connection was created
- `metadata`: Why this connection exists

**Edge Lifecycle & Temporal Validity (Critical for Accuracy):**
- `valid_from`: When this edge became true (default: creation time)
- `valid_until`: When this edge stopped being true (None = still valid)
- `superseded_by`: Edge ID that replaced this one (if applicable)
- `is_active`: Whether edge is currently valid (computed from valid_until)

**Example: Edge Becomes Obsolete**
```
Time T1: "Approach X rejected because causes Bug Y"
→ Edge: approach_X --[rejected_because]--> bug_Y
→ valid_from: T1, valid_until: None, is_active: True

Time T2: Codebase refactored, Bug Y fixed
→ Update edge: valid_until = T2, is_active = False
→ Create new state: approach_X v2 "Now viable after refactoring"
→ New edge: approach_X_v2 --[viable_after_fix]--> refactoring_event
→ valid_from: T2, valid_until: None, is_active: True

Query at T3: "Why not approach X?"
→ System finds approach_X latest state (v2)
→ Traverses ACTIVE edges only
→ Returns: "Approach X now viable (refactored T2), was rejected due to Bug Y (fixed)"
→ Old edge still in graph (for history) but marked inactive
```

### 4. Memory Promotion
Creating a new state by synthesizing information from traces and graphs:

```
1. Analyze current latest state
2. Traverse graph to find related states
3. Walk backwards through traces for context
4. Use LLM to synthesize new state
5. Create new state with updated information
6. Establish new edges to related concepts
```

### 5. Memory Lifecycle & Consolidation (Human-Like Benefits, Not Downsides)

**Memory Strength Dynamics (For Ranking, Not Filtering):**
Unlike human memory, we implement the BENEFITS (prioritization) without DOWNSIDES (forgetting):
- **Strength Score**: Used for retrieval ranking/prioritization (0.0-1.0)
- **Decay**: Score decreases over time → lower ranking, NOT invisibility
- **Reinforcement**: Each access increases score → higher ranking
- **Importance**: Important memories maintain high scores
- **Never Lost**: All memories remain searchable and accessible
  - Low-strength memories: Still in database, still in graph, still retrievable
  - Just ranked lower in search results
  - Always accessible via: graph traversal, explicit query, or trace history

**Key Principle: Strength = Priority, Not Existence**
- Traditional human memory: forgot = gone
- TracingRAG: "forgot" = deprioritized, but still accessible
- Graph connections ensure nothing is truly lost
- Latest state edges maintain relevance chains

**Graph-Based Relevance Discovery:**
Even if a memory has low strength, it remains discoverable via graph:
- **Latest State Edges**: Follow edges from latest state to find all connected context
- **Trace Relationships**: Walk backwards through traces to find historical context
- **Cross-Topic Connections**: Related topics maintain edges regardless of strength
- **Example**: Bug from 6 months ago (low strength) → still connected to current project state → found when querying project

**Storage Tiers:**
- **Working Memory**: Hot cache (Redis), recently/frequently accessed, <1000 items
- **Active Memory**: Normal storage, regularly accessed, full search capability
- **Archived Memory**: Cold storage (S3), rarely accessed, retrieved on demand
  - Note: Archive location, NOT deleted - still in graph and searchable

**Hierarchical Consolidation (Like Sleep):**
Automatic summarization at multiple time scales:
- **Daily**: Consolidate each day's changes into summary
- **Weekly**: Roll up daily summaries into weekly
- **Monthly**: Roll up weekly summaries into monthly
- **Benefit**: Query "what happened last month?" → instant summary, can drill down if needed

### 6. Latest State Tracking (O(1) Lookup)

**Critical for "What's the current status?" queries:**
- Materialized view: `topic → latest_state_id` mapping
- O(1) lookup via Redis cache + PostgreSQL index
- Auto-updated on new state creation (database trigger)
- Enables instant "give me the latest" without searching all versions

**Performance:**
- Without: Search through all versions, filter, sort → 100-500ms
- With latest tracking: Direct lookup → <10ms

### 7. Working Memory System

**Context-Aware Hot Cache:**
When working on a topic, pre-load related memories:
- Latest states of main topic
- Recent states (last 10 versions)
- High-importance states
- Related topics from graph

**Benefits:**
- Queries against working set: <10ms (in-memory)
- No database hits for active context
- Mimics human "having something in mind"
- Automatically expands as you explore connections

## System Architecture

### Layer 1: Storage Layer
```
┌─────────────────────────────────────────────────────┐
│  Vector Database (Qdrant)                           │
│  - Stores embeddings of memory states               │
│  - Enables semantic similarity search               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Graph Database (Neo4j)                             │
│  - Stores memory states as nodes                    │
│  - Stores relationships as edges                    │
│  - Temporal indexing for time-based queries         │
│  - Supports graph traversal algorithms              │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Document Store (PostgreSQL with TimescaleDB)       │
│  - Stores full memory state content                 │
│  - Maintains audit trail                            │
│  - Time-series optimized queries                    │
└─────────────────────────────────────────────────────┘
```

### Layer 2: Core Services

#### Memory Manager
- CRUD operations for memory states
- Trace management
- Version control

#### Graph Manager
- Edge creation and management
- Graph traversal algorithms
- Community detection (for related topics)

#### Embedding Service
- Text embedding generation
- Embedding updates
- Multiple embedding model support

#### Retrieval Agent
- Semantic search via vectors
- Graph-based retrieval
- Temporal queries
- Hybrid retrieval strategies

#### Promotion Service
- Analyzes traces and graphs
- Synthesizes new states
- Manages state transitions

### Layer 3: Agentic Layer

#### Query Agent
```python
class QueryAgent:
    def process_query(self, query: str):
        1. Analyze query intent
        2. Decide retrieval strategy:
           - Latest state only?
           - Need historical context?
           - Need related topics?
        3. Execute retrieval
        4. Re-rank results
        5. Return context for generation
```

#### Memory Agent
```python
class MemoryAgent:
    def promote_memory(self, topic: str):
        1. Find latest state
        2. Traverse graph for related states
        3. Walk trace backwards for context
        4. Identify what changed
        5. Synthesize new state
        6. Create appropriate edges
```

#### Planning Agent
```python
class PlanningAgent:
    def plan_retrieval(self, context: dict):
        1. Assess information needs
        2. Generate multi-step retrieval plan
        3. Execute plan with other agents
        4. Validate retrieved information
```

### Layer 4: API & Interface

```
REST API / GraphQL API
- Query endpoint
- Memory CRUD endpoints
- Promotion endpoint
- Graph exploration endpoint
- Time-travel queries
```

## Tech Stack Recommendations

### Core Framework
- **Python 3.11+** - Main language
- **FastAPI** - API framework (async, high performance)
- **Pydantic v2** - Data validation and serialization

### Vector Database
**Option 1: Qdrant (Recommended)**
- Open source, MIT license
- Excellent performance
- Built-in filtering and hybrid search
- Easy to deploy (Docker)
- Python client is mature

**Option 2: Milvus**
- More enterprise features
- Slightly more complex setup

### Graph Database
**Neo4j (Recommended)**
- Industry standard for graph databases
- Cypher query language is powerful
- Excellent temporal support
- Good Python drivers (neo4j-driver)
- Community edition is free

**Alternative: NebulaGraph**
- Open source, more scalable
- More complex to set up

### Document Store
**PostgreSQL with TimescaleDB extension**
- Reliable, mature
- Time-series optimization perfect for traces
- Can co-locate with main app data

### Embedding Models
**Sentence Transformers (Local)**
- `all-MiniLM-L6-v2` - Fast, good for development
- `all-mpnet-base-v2` - Better quality
- `e5-large-v2` - State-of-the-art open source

**OpenAI (API)**
- `text-embedding-3-large` - Highest quality
- `text-embedding-3-small` - Cost effective

### LLM Integration
**OpenRouter API (Recommended)**
- Access to multiple models
- Single API interface
- Cost effective
- Fallback options

**Models to support:**
- GPT-4 Turbo - High quality reasoning
- Claude 3.5 Sonnet - Excellent synthesis
- Llama 3.1 70B - Open source option
- Mixtral 8x7B - Fast, cost effective

### Agent Framework
**LangGraph (Recommended)**
- Built on LangChain
- Designed for agentic workflows
- State management built-in
- Graph-based agent orchestration

**Alternative: AutoGen**
- Microsoft's multi-agent framework
- More complex but powerful

### Observability
- **Langfuse** - LLM observability and tracing
- **Prometheus + Grafana** - System metrics
- **Jaeger** - Distributed tracing

### Development Tools
- **Poetry** - Dependency management
- **Ruff** - Linting and formatting
- **pytest** - Testing
- **Docker & Docker Compose** - Containerization

## Data Models

### Memory State Schema
```python
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class MemoryState(BaseModel):
    id: str  # UUID
    topic: str  # What this memory is about
    content: str  # The actual knowledge
    version: int  # Version in trace (1, 2, 3...)
    timestamp: datetime
    embedding: List[float]  # Vector representation
    parent_state_id: Optional[str]  # Previous version
    metadata: Dict[str, Any]
    tags: List[str]
    confidence: float  # 0.0 to 1.0
```

### Edge Schema
```python
class MemoryEdge(BaseModel):
    id: str
    source_state_id: str
    target_state_id: str
    relationship_type: str  # "evolves_to", "relates_to", "causes", etc.
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    bidirectional: bool
    metadata: Dict[str, Any]
```

### Trace Schema
```python
class Trace(BaseModel):
    id: str
    topic: str
    states: List[str]  # Ordered list of state IDs
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
```

## Key Algorithms

### 1. Semantic Retrieval with Latest State Tracking
```python
def search_memories(query: str, time_window: Optional[tuple] = None):
    # Generate query embedding
    query_emb = embed(query)

    # Vector search in Qdrant - retrieves all relevant states
    candidates = vector_db.search(
        vector=query_emb,
        limit=100,
        filter={
            "timestamp": time_window if time_window else None
        }
    )

    # Get latest states from each trace (O(1) lookup via materialized view)
    latest_states = filter_to_latest_per_trace(candidates)

    # Sort by semantic similarity
    latest_states.sort(key=lambda s: s.semantic_score, reverse=True)

    return latest_states
```

**Key Insight:**
- Nothing is filtered out - all semantically relevant states retrieved
- Latest state lookup is O(1) (< 10ms via materialized view)
- Ranking is based on semantic similarity
- Edge weights (strength) come into play during graph traversal (see next algorithm)

### 2. Graph-Enhanced Retrieval (Edge-Based Relevance)
```python
def graph_enhanced_retrieval(query: str, depth: int = 2):
    # Get initial semantic matches
    initial_matches = search_memories(query)

    # For each match, traverse graph
    enhanced_results = []
    for match in initial_matches:
        # Get connected states within depth
        # IMPORTANT: Follows ACTIVE edges regardless of weight
        # Edge.strength represents contextual relevance, not existence
        # All connections preserved - strength used for ranking, not filtering
        subgraph = graph_db.traverse(
            start_node=match.id,
            max_depth=depth,
            relationship_types=["relates_to", "causes", "references"],
            only_active=True,  # Only follows edges where valid_until is None or future
            include_edge_weights=True  # Include edge.strength for ranking
        )

        # Walk backwards in trace for context
        historical_context = get_trace_context(match, steps_back=3)

        # Rank related states by edge strength (contextual relevance)
        ranked_related = sorted(
            subgraph,
            key=lambda s: s.edge_strength,  # Edge weight from traversal
            reverse=True
        )

        enhanced_results.append({
            "state": match,
            "related_states": ranked_related,
            "historical_context": historical_context
        })

    return enhanced_results
```

**Key Insight:**
- Edge.strength represents contextual relevance between states
- All edges traversed (nothing filtered by strength)
- Strength used to rank related results
- Example: project_alpha → bug_report edge might have strength=0.9 (highly relevant)
  while project_alpha → old_discussion edge might have strength=0.3 (tangentially related)
- Both connections preserved and traversed, but ranked by contextual importance

### 3. Memory Promotion Algorithm
```python
async def promote_memory(topic: str, reason: str):
    # 1. Find current latest state
    current_state = get_latest_state(topic)

    # 2. Get trace history
    trace = get_trace(topic)

    # 3. Find related topics via graph
    related_states = graph_db.get_neighbors(
        current_state.id,
        relationship_types=["relates_to", "causes"]
    )

    # 4. Build context for LLM
    context = {
        "current_state": current_state.content,
        "history": [state.content for state in trace.states[-5:]],
        "related": [state.content for state in related_states],
        "reason": reason
    }

    # 5. Use LLM to synthesize new state
    new_content = await llm.generate(
        prompt=PROMOTION_PROMPT.format(**context)
    )

    # 6. Create new state
    new_state = create_memory_state(
        topic=topic,
        content=new_content,
        version=current_state.version + 1,
        parent_state_id=current_state.id
    )

    # 7. Create edges
    create_edge(current_state.id, new_state.id, "evolves_to")
    for related in related_states:
        if should_maintain_connection(related, new_state):
            create_edge(new_state.id, related.id, "relates_to")

    return new_state
```

### 4. Temporal Query
```python
def query_at_time(topic: str, timestamp: datetime):
    """Get the state of knowledge at a specific point in time"""
    trace = get_trace(topic)

    # Binary search through versions
    for state in reversed(trace.states):
        if state.timestamp <= timestamp:
            return state

    return None  # Didn't exist at that time
```

### 5. Handling Obsolete Edges (Temporal Accuracy)

```python
async def handle_fact_change(
    old_state_id: UUID,
    change_description: str
) -> PromotionResult:
    """
    Handle when a fact changes, making old edges obsolete.

    Example: "Bug Y fixed, Approach X now viable"
    """

    # 1. Create new state documenting the change
    new_state = await create_memory_state(
        topic=get_topic_from_state(old_state_id),
        content=change_description,
        parent_state_id=old_state_id
    )

    # 2. Find edges that are now obsolete
    affected_edges = await find_edges_to_update(old_state_id, change_description)

    # 3. Mark old edges as inactive
    for edge in affected_edges:
        await update_edge(
            edge.id,
            valid_until=datetime.utcnow(),
            superseded_by=None  # Could link to new edge if creating one
        )

    # 4. Create new edges reflecting current truth
    new_edges = []
    for old_edge in affected_edges:
        if should_create_inverse_edge(old_edge):
            # Example: "rejected_because" → "viable_after_fix"
            new_edge = await create_edge(
                source_state_id=new_state.id,
                target_state_id=get_related_state(old_edge),
                relationship_type=inverse_relationship(old_edge.relationship_type),
                description=f"Now viable after: {change_description}",
                valid_from=datetime.utcnow()
            )
            new_edges.append(new_edge)

            # Link old edge to new edge
            await update_edge(old_edge.id, superseded_by=new_edge.id)

    return PromotionResult(
        new_state=new_state,
        previous_state_id=old_state_id,
        new_edges=new_edges,
        obsolete_edges=[e.id for e in affected_edges]
    )

# Helper: Determine if edge should be inverted
def inverse_relationship(rel_type: RelationshipType) -> RelationshipType:
    """Get inverse relationship when fact changes"""
    inversions = {
        RelationshipType.CONTRADICTS: RelationshipType.SUPPORTS,
        "rejected_because": "viable_after_fix",
        "blocked_by": "enabled_by",
        # ... more inversions
    }
    return inversions.get(rel_type, RelationshipType.RELATES_TO)
```

**Example: Complete Scenario**

```python
# SCENARIO: Approach X rejected due to Bug Y, then Bug Y fixed

# === Time T1: Initial Decision ===
approach_x_v1 = create_memory(
    topic="design/approach_x",
    content="Considered approach X but rejected due to Bug Y"
)

bug_y = create_memory(
    topic="bugs/bug_y",
    content="Bug Y causes data corruption in approach X"
)

# Create rejection edge
rejection_edge = create_edge(
    source_state_id=approach_x_v1.id,
    target_state_id=bug_y.id,
    relationship_type="rejected_because",
    description="Approach X rejected because it triggers Bug Y",
    valid_from=T1,
    valid_until=None  # Currently valid
)

# === Time T2: Bug Fixed via Refactoring ===
refactoring = create_memory(
    topic="refactoring/2024_q1",
    content="Refactored data layer, Bug Y no longer occurs"
)

bug_y_v2 = promote_memory(
    topic="bugs/bug_y",
    reason="Fixed by refactoring",
    content="Bug Y fixed: data layer refactored to prevent corruption"
)

# Mark old rejection edge as obsolete
update_edge(
    rejection_edge.id,
    valid_until=T2,  # No longer valid
    superseded_by=None  # Will create new edge
)

# Create new state for approach X
approach_x_v2 = promote_memory(
    topic="design/approach_x",
    reason="Bug Y fixed, approach now viable",
    content="Approach X now viable after Bug Y fix (refactoring T2)"
)

# Create new "viable" edge
viable_edge = create_edge(
    source_state_id=approach_x_v2.id,
    target_state_id=refactoring.id,
    relationship_type="viable_after_fix",
    description="Approach X now viable due to refactoring",
    valid_from=T2,
    valid_until=None  # Currently valid
)

# Link old to new
update_edge(rejection_edge.id, superseded_by=viable_edge.id)

# === Time T3: Developer Queries ===
query("Why don't we use approach X?")

# System retrieves:
# 1. Latest state: approach_x_v2 (created T2)
# 2. Traverses ACTIVE edges only
# 3. Finds: viable_edge (T2, active) → refactoring
# 4. Does NOT follow: rejection_edge (T1-T2, inactive)

# Returns:
{
    "current_answer": "Approach X is now viable (as of T2)",
    "reasoning": "Enabled by refactoring that fixed Bug Y",
    "historical_context": "Was previously rejected (T1) due to Bug Y, but that's no longer true",
    "edges": [
        {
            "type": "viable_after_fix",
            "target": "refactoring",
            "active": True
        }
    ],
    "obsolete_edges": [
        {
            "type": "rejected_because",
            "target": "bug_y",
            "active": False,
            "valid_until": T2,
            "superseded_by": viable_edge.id
        }
    ]
}

# Historical query: "Why was approach X rejected at T1?"
query_at_time("design/approach_x", T1)
# Returns: approach_x_v1 with rejection_edge (which WAS active at T1)
# Correctly returns historical reason

# Time-aware query: "When did approach X become viable?"
find_edge_change("design/approach_x", "rejected_because", "viable_after_fix")
# Returns: T2 (when rejection_edge became inactive, viable_edge created)
```

**Key Benefits:**
- **Temporal Accuracy**: Old reasons don't pollute current queries
- **Historical Preservation**: Can still query "why was it rejected in 2023?"
- **Explicit Transitions**: Clear audit trail of when/why facts changed
- **No Data Loss**: Old edges kept but marked inactive
- **Supersession Tracking**: Know which edge replaced which

**Query Behavior:**
- Default queries: Only active edges (current truth)
- Historical queries: Edges active at that time
- Full history: All edges with validity periods

### 6. Context Window Management (Preventing Information Loss)

**The Challenge:**
With large knowledge bases (millions of states), it's impossible to fit all relevant information into an LLM's context window. The critical question: How do we ensure no information is lost while preventing hallucination?

**The Solution: Hierarchical Retrieval with Intelligent Context Budgeting**

```python
async def retrieve_within_budget(
    query: str,
    max_tokens: int = 100000  # Example: Claude 3.5 Sonnet context
) -> dict:
    """
    Intelligent retrieval that fits within context window.

    Strategy:
    1. Latest states (always included) - O(1) lookup
    2. Consolidated summaries (appropriate level)
    3. Selective drill-down (graph-guided filtering)
    4. Preserve buffer for LLM reasoning
    """

    # Phase 1: Latest states (the "index") - ALWAYS included
    # These provide current ground truth
    latest_states = await get_latest_states_for_query(query)
    context = {
        "latest": latest_states,  # ~20K tokens
        "summaries": [],
        "details": []
    }

    tokens_used = estimate_tokens(latest_states)
    buffer = 10000  # Reserve for LLM reasoning
    remaining = max_tokens - tokens_used - buffer

    # Phase 2: Determine appropriate consolidation level
    # Automatically adjust granularity based on query type
    level = determine_consolidation_level(query)
    # - "What's current status?" → level 0 (no history)
    # - "What happened last week?" → level 1 (daily summaries)
    # - "Summarize entire project" → level 3 (monthly summaries)
    # - "Why did X happen?" → level 0 (detailed, graph-filtered)

    # Phase 3: Get relevant summaries
    # Use graph edges from latest states to find relevant topics
    relevant_topics = extract_topics_via_graph(latest_states)

    summaries = await get_relevant_summaries(
        topics=relevant_topics,
        consolidation_level=level,
        max_tokens=min(30000, remaining * 0.4)
    )
    context["summaries"] = summaries
    remaining -= estimate_tokens(summaries)

    # Phase 4: Selective drill-down on most relevant
    # Graph-guided filtering ensures we get RIGHT information, not ALL
    ranked_topics = await semantic_rank(
        query=query,
        candidates=relevant_topics,
        query_embedding=await embed(query)
    )

    for topic in ranked_topics:
        if remaining < 5000:
            break  # Preserve buffer

        # Get recent states for this topic (not entire trace!)
        recent_states = await get_trace_recent(
            topic=topic,
            limit=10,  # Configurable
            time_window=(now - timedelta(days=30), now)
        )

        detail_tokens = estimate_tokens(recent_states)
        if detail_tokens <= remaining:
            context["details"].extend(recent_states)
            remaining -= detail_tokens

    return context


def determine_consolidation_level(query: str) -> int:
    """Auto-select granularity based on query intent"""

    # Status query: Just latest, no history
    if is_status_query(query):  # "What's current status?"
        return 0

    # Recent query: Daily summaries
    elif is_recent_query(query):  # "What happened last week?"
        return 1

    # Overview query: Monthly summaries
    elif is_overview_query(query):  # "Summarize the project"
        return 3

    # Why/cause query: Detailed states, graph-filtered
    elif is_why_query(query):  # "Why did character X do Y?"
        return 0  # But with graph filtering!

    return 2  # Default: weekly summaries


async def get_relevant_subgraph(
    query_embedding: list[float],
    max_nodes: int = 100
) -> list[MemoryState]:
    """
    Graph-guided filtering: Use latest states + edges to identify
    relevant historical states WITHOUT retrieving entire database.
    """

    # Step 1: Vector search finds semantically relevant latest states
    relevant_latest = await vector_search(
        embedding=query_embedding,
        filter={"is_latest": True},  # Only current states
        limit=10
    )

    # Step 2: Graph traversal with relevance-based pruning
    subgraph = []
    for state in relevant_latest:
        # Follow edges with strength > threshold
        # This is where edge.strength provides contextual filtering
        connected = await graph.traverse(
            start=state.id,
            direction="incoming",  # Historical states
            max_depth=3,
            min_edge_strength=0.5,  # Prune weak connections
            max_nodes_per_hop=5,    # Limit breadth
            only_active=True        # Current truth only
        )
        subgraph.extend(connected)

    # Return filtered subgraph (not entire database!)
    return subgraph[:max_nodes]
```

**Concrete Example: Novel Writing**

```python
# Query: "How would Sarah react if she saw the king right now?"

# === Phase 1: Latest States (Always Included) ===
latest_context = {
    "character/sarah": "Sarah is a skilled warrior, recovering from trauma...",
    "relationship/sarah_to_king": "Deep fear and hatred of the king",
}
# Tokens: ~2K

# === Phase 2: Graph Filtering ===
# System finds relevant personality traits via edges:
# sarah --[has_trait]--> trait/sarah_fear_of_king
# sarah --[has_trait]--> trait/sarah_desire_for_revenge

traits = {
    "trait/sarah_fear_of_king": {
        "latest": "High intensity fear",
        "caused_by": "event/parents_murdered"  # Edge preserves causality!
    },
    "trait/sarah_desire_for_revenge": {
        "latest": "Medium intensity",
        "caused_by": "event/kingdom_burned"
    }
}
# Tokens: ~3K (NOT 500K for all 1000 chapters!)

# === Phase 3: Recent Interactions ===
# Graph finds: sarah --[interacted_with]--> king (last 5 events only)
recent_interactions = [
    "Chapter 1003: Sarah fled when king appeared",
    "Chapter 998: Sarah watched king from shadows",
    # ... (last 5 interactions, not all 1000 chapters)
]
# Tokens: ~5K

# === Total: ~10K tokens instead of 500K ===
# But contains ALL critical information:
# - Latest character state (current truth)
# - Current relationship state
# - Key personality traits WITH causality (edges!)
# - Recent relevant history (not ancient irrelevant events)

# LLM can now accurately reason:
# "Sarah would experience intense fear (caused by parents' murder)
#  mixed with desire for revenge (caused by kingdom burning).
#  Given her recent pattern of avoidance (last 5 interactions),
#  she would likely retreat but prepare for future confrontation..."
```

**Why This Prevents Hallucination:**

1. **Latest states always included** → LLM knows current ground truth
   - Can't hallucinate "Sarah likes the king" when latest state says "fear and hatred"

2. **Graph edges preserve causality** → LLM sees WHY, not just WHAT
   - Can't hallucinate reasons: fear → murder event edge is explicit

3. **Summaries provide temporal context** → LLM knows WHEN without every detail
   - "Daily summary: 50 interactions with guards, 2 mentions of king"

4. **Semantic ranking ensures relevance** → Only related history included
   - Query about Sarah-King doesn't retrieve Sarah-merchant interactions

5. **Explicit drill-down capability** → LLM can request more detail
   - Multi-pass retrieval: "Need more detail on the murder event"

**Information Loss vs. Context Efficiency:**

**Key Insight: Not all information is equally relevant to every query!**

Novel example - Query: "How would Sarah react to the king?"

**DON'T need** (filtered out by graph):
- Every conversation Sarah had with other characters
- Detailed descriptions of every location
- Complete text of every chapter
- Sarah's interactions with merchants, healers, etc.

**DO need** (included via graph edges):
- Sarah's latest state
- Sarah-King relationship (latest)
- Relevant personality traits (fear, revenge)
- Causality edges (fear → murder, revenge → burned kingdom)
- Recent Sarah-King interactions (last 5, not all 50)

Codebase example - Query: "How does authentication work?"

**DON'T need:**
- Every commit message in 5-year history
- Every refactoring detail
- Test file contents
- Documentation changes

**DO need:**
- Current auth implementation (latest state)
- Related security middleware (graph-connected)
- Recent auth changes (last 30 days)
- Design decision summaries (weekly consolidation)

**The system uses intelligent filtering, not random sampling.**
Graph structure + semantic ranking + consolidation levels ensure the RIGHT information is included, even if not ALL information fits.

**Multi-Pass Retrieval (Agentic Pattern):**

```python
# Pass 1: LLM sees overview
context_v1 = {
    "latest_states": [...],
    "summary": "Jan-Mar 2024: Feature X developed, bug Y fixed..."
}

response_v1 = await llm.generate(query, context_v1)

# If LLM needs more detail:
if response_v1.needs_drill_down:
    # Pass 2: Drill down on specific topic
    context_v2 = await retrieve_within_budget(
        query=f"Details about {response_v1.drill_down_target}",
        max_tokens=40000
    )

    response_v2 = await llm.generate(
        f"{query} (drill-down: {response_v1.drill_down_target})",
        context_v2
    )
```

**Storage Tiers Enable This:**

- **Hot (working)**: Latest states + active edges → <10ms lookup
- **Warm (active)**: Recent states (last 30 days) → <100ms with caching
- **Cold (archived)**: Historical states → Available for drill-down

Consolidated summaries live in warm storage, originals in cold storage.
Drill-down pattern: summary (warm) → daily (cold) → originals (cold)

**Result:**
- Queries stay within context window
- No information truly lost (all states preserved)
- Graph ensures relevant information found
- LLM never hallucinates because latest states = ground truth
- Multi-pass retrieval allows deeper exploration when needed

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure
- [ ] Configure databases (Neo4j, Qdrant, PostgreSQL)
- [ ] Implement data models
- [ ] Basic CRUD for memory states
- [ ] Embedding service integration
- [ ] Simple vector search

### Phase 2: Tracing System (Weeks 3-4)
- [ ] Implement trace management
- [ ] Version control for memory states
- [ ] Temporal queries
- [ ] Parent-child relationships
- [ ] Time-travel functionality

### Phase 3: Graph Layer (Weeks 5-6)
- [ ] Graph database integration
- [ ] Edge creation and management
- [ ] Graph traversal algorithms
- [ ] Relationship type system
- [ ] Visualization endpoints

### Phase 4: Basic RAG (Weeks 7-8)
- [ ] Query processing
- [ ] Hybrid retrieval (vector + graph)
- [ ] LLM integration via OpenRouter
- [ ] Response generation
- [ ] Context management

### Phase 5: Agentic Layer (Weeks 9-11)
- [ ] LangGraph integration
- [ ] Query planning agent
- [ ] Memory management agent
- [ ] Retrieval strategy selection
- [ ] Multi-step reasoning

### Phase 6: Memory Promotion (Weeks 12-13)
- [ ] Promotion algorithm
- [ ] Synthesis prompts
- [ ] Edge update logic
- [ ] Conflict resolution
- [ ] Validation

### Phase 7: Advanced Features (Weeks 14-16)
- [ ] Community detection for topic clustering
- [ ] Automatic edge discovery
- [ ] Memory consolidation
- [ ] Quality scoring
- [ ] Performance optimization

### Phase 8: Production Ready (Weeks 17-18)
- [ ] API documentation
- [ ] Monitoring and observability
- [ ] Error handling and retry logic
- [ ] Rate limiting
- [ ] Security hardening
- [ ] Deployment configuration

## Example Use Cases

### Use Case 1: Bug Tracking in Development
```
1. Initial project state: "Working on feature X"
2. Bug discovered: Create bug trace, link to project
3. Bug investigation: Add states to bug trace
4. Bug fixed: Update bug trace
5. Promote project state: "Feature X complete, fixed bug Y"
   - New state references bug trace
   - Mentions why approach Z was avoided
   - Graph edge from project v3 → bug v5
```

### Use Case 2: Research Evolution
```
1. Initial understanding: "Topic A is understood as X"
2. New paper discovered: Creates related trace
3. Contradiction found: New state in Topic A trace
4. Resolution: Promote Topic A with synthesis
   - Incorporates new paper insights
   - Explains evolution of understanding
   - Maintains history for transparency
```

### Use Case 3: Decision Making
```
Query: "Why did we choose technology X over Y?"

Agent:
1. Finds latest decision state
2. Traverses to related evaluation states
3. Walks trace backwards to see alternatives considered
4. Provides comprehensive answer with full context
```

## Advantages Over Traditional RAG

### Core Capabilities
1. **Temporal Awareness**: Can answer "what did we know at time T?"
2. **Evolution Tracking**: Understand how knowledge changed and why
3. **Contextual Relationships**: Not just semantic similarity
4. **Non-Lossy**: Never lose historical context
5. **Causality**: Understand cause-and-effect relationships
6. **Synthesis**: Can create refined knowledge states
7. **Agentic**: Intelligent retrieval strategies, not just keyword matching

### Human-Like Memory (Critical Differentiator)
8. **Memory Strength**: Mimics Ebbinghaus forgetting curve - memories decay but strengthen with access
9. **Recency Bias**: Recent memories stronger, like human recall
10. **Importance Learning**: System learns what's important from usage patterns
11. **Working Memory**: Context-aware hot cache like human working memory (<10ms queries)
12. **Consolidation**: Hierarchical summaries like human sleep consolidation
13. **Latest State Tracking**: Instant O(1) lookup for "what's current?" queries
14. **Reconsolidation**: Memories update when recalled (like human memory reconsolidation)

### Scale & Performance
15. **Instant Latest**: <10ms for current state (vs 100-500ms in traditional RAG)
16. **Working Set**: <10ms for active context (vs full search every time)
17. **Massive Scale**: Millions/billions of states with partitioning and tiering
18. **Space Efficient**: 95% storage savings with diff-based versioning
19. **Smart Caching**: Dynamic TTL based on content recency and access patterns

## Challenges and Solutions

### Challenge 1: Graph Explosion
**Problem**: Too many edges, graph becomes unwieldy
**Solution**:
- Edge strength scoring
- Prune weak edges periodically
- Community detection for clustering
- Hierarchical graph structure

### Challenge 2: Conflicting Information
**Problem**: Different states have contradictory information
**Solution**:
- Confidence scoring
- Source tracking
- Explicit conflict resolution in promotion
- User validation when confidence is low

### Challenge 3: Performance
**Problem**: Graph traversal + vector search can be slow
**Solution**:
- Caching layer (Redis)
- Index optimization
- Limit traversal depth
- Parallel queries
- Approximate nearest neighbor for vectors

### Challenge 4: Embedding Drift
**Problem**: Embedding model changes affect similarity
**Solution**:
- Version embeddings
- Re-embed entire dataset when model changes
- Store model version with each embedding

## Next Steps

1. Review and approve this design
2. Set up development environment
3. Start Phase 1 implementation
4. Create initial examples and tests
5. Iterate based on findings
