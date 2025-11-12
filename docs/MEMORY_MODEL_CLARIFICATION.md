# Memory Strength Model Clarification

## The Better Model: Edge-Based Strength

### Current Implementation (To Be Reconsidered)
```python
class MemoryState:
    memory_strength: float  # State has inherent strength
    access_count: int
    last_accessed: datetime
```

### Better Model: Edge-Based Strength

**Key Insight**: Memory strength is **contextual** - it's about relationships, not absolute properties.

```python
class MemoryState:
    # NO memory_strength field
    # NO access_count (tracking can be separate if needed for analytics)
    # State is just content + metadata

class MemoryEdge:
    strength: float  # THIS is memory strength!
    # Represents: "How strongly does A relate to B?"
```

## Why Edge-Based is Better

### 1. Nothing is Forgotten
- States don't have decay
- All memories are always accessible
- No "weak memory" that might be filtered out
- **Strength is about relevance, not existence**

### 2. Contextual Relevance
```python
# Same memory, different contexts

# From current project state:
current_project --[strength=0.9]--> old_design_decision
# (Highly relevant to current state)

# From unrelated feature:
other_feature --[strength=0.2]--> old_design_decision
# (Less relevant, but still connected)
```

The decision isn't "weak" or "strong" inherently - its relevance depends on context!

### 3. Natural Graph Traversal
```python
def graph_retrieval(start_state: UUID):
    # Traverse graph from starting point
    # Edge strength naturally weights results
    neighbors = get_neighbors(start_state)

    for neighbor, edge in neighbors:
        # Use edge strength for ranking
        neighbor.relevance = edge.strength * semantic_similarity
```

### 4. No Artificial Decay
- Edge strengths can be learned/updated based on usage
- But no automatic decay that loses information
- System learns what's important, doesn't forget

## Revised Retrieval Model

### Current Query Flow (With State-Based Strength)
```python
# 1. Vector search
candidates = vector_search(query)

# 2. Apply memory_strength decay
for state in candidates:
    strength = calculate_decay(state.last_accessed, state.access_count)
    state.score = semantic_score * strength  # Might demote old info

# Problem: Old but relevant info gets demoted
```

### Better Query Flow (With Edge-Based Strength)
```python
# 1. Find contextually relevant starting points
start_states = vector_search(query)  # Get semantic matches

# 2. Expand via graph with edge weights
for start in start_states:
    # Traverse edges weighted by strength
    connected = graph_traverse(
        start,
        depth=2,
        weight_edges_by="strength"  # Use edge strength!
    )

    # Score = semantic_similarity × edge_path_strength
    for state in connected:
        path_strength = product(edge.strength for edge in path_to(state))
        state.score = state.semantic_score * path_strength

# Result: Old info found if strongly connected to relevant current states!
```

## How This Solves the "Approach X" Scenario

```python
# Scenario: Old decision now obsolete

# Time T1: Decision made
approach_x_v1 = create_state("Approach X rejected")
bug_y = create_state("Bug Y exists")

# Strong rejection edge
rejection_edge = create_edge(
    approach_x_v1 --> bug_y,
    strength=0.9,
    relationship="rejected_because"
)

# Time T2: Bug fixed
approach_x_v2 = create_state("Approach X now viable")
refactoring = create_state("Refactored, Bug Y fixed")

# Mark old edge inactive (temporal validity)
update_edge(rejection_edge, valid_until=T2)

# New viable edge
viable_edge = create_edge(
    approach_x_v2 --> refactoring,
    strength=0.9,
    relationship="viable_after_fix"
)

# Time T3: Query
query("Why not approach X?")

# System:
# 1. Finds approach_x_v2 (latest state, via latest_state_tracking)
# 2. Traverses ACTIVE edges only
# 3. Finds viable_edge (strength 0.9) → refactoring
# 4. Does NOT traverse rejection_edge (inactive)

# Result: Correct current answer!
```

## Implications for Memory Strength Fields

### Keep for Analytics (Optional)
```python
class MemoryState:
    # Optional: For analytics/monitoring only
    access_log: List[datetime] = []  # When accessed (for stats)
    view_count: int = 0  # How many times viewed (for stats)

    # NOT used for retrieval ranking!
```

### Use Edge Strength for Retrieval
```python
class MemoryEdge:
    strength: float  # PRIMARY relevance score
    valid_from: datetime
    valid_until: Optional[datetime]
    is_active: bool

    # Edge strength can be:
    # - Set manually (explicit connection strength)
    # - Learned from co-occurrence
    # - Updated based on usage patterns
    # - But NOT automatically decayed
```

## Implementation: Edge Strength Learning (Optional)

```python
class EdgeStrengthLearner:
    """Learn edge strengths from usage patterns"""

    async def update_from_query(
        self,
        query_state: UUID,
        retrieved_states: List[UUID]
    ):
        """When state A leads to retrieving state B, strengthen edge"""

        for retrieved in retrieved_states:
            # Find or create edge
            edge = get_or_create_edge(query_state, retrieved)

            # Increase strength slightly (reinforcement)
            new_strength = min(1.0, edge.strength + 0.05)
            update_edge(edge.id, strength=new_strength)

    # Edge strengths learned from actual usage
    # Not artificially decayed
```

## Summary: Fundamental Model Change

### OLD Model (State-Based)
- Memory strength is property of state
- Decays over time (Ebbinghaus curve)
- Used for ranking in retrieval
- Problem: Demotes old but relevant info

### NEW Model (Edge-Based)
- Memory strength is property of edges
- Represents contextual relevance
- No decay (nothing forgotten)
- Old info found via graph connections

### Benefits
1. **Nothing forgotten**: All states remain accessible
2. **Contextual**: Relevance depends on starting point
3. **Graph-native**: Natural use of graph structure
4. **Accurate**: Old info found when related to current state
5. **Learnable**: Edge strengths can improve with usage

### Implementation Note
Keep `access_log` and `view_count` on states for analytics, but don't use for retrieval ranking. All ranking comes from:
1. Semantic similarity (vector search)
2. Edge path strength (graph traversal)
3. Latest state tracking (temporal accuracy)

This is the correct model for "never forget, always find relevant".
