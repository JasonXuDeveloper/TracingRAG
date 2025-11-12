# Domain-Specific Usage Patterns (Examples, Not Requirements)

## Overview

**IMPORTANT**: TracingRAG's core system is **generic** and **domain-agnostic**. The features below are **usage patterns** and **examples** showing how to apply the generic system to specific domains (novels, NPCs, projects, etc.).

**Core System (Generic & Broad)**:
- Temporal traces (track ANY entity evolution)
- Graph relationships (connect ANY concepts)
- Memory strength (prioritize ANY information)
- Consolidation (summarize ANY time-series data)
- Latest state tracking (instant lookup of ANY topic)
- Temporal edge validity (accurate reasoning for ANY facts)

**Domain Patterns (Examples of Usage)**:
- Character tracking = trace + entity_schema (you define the schema)
- Relationship evolution = trace for relationship entity (generic traces)
- Character knowledge = filtered queries + metadata (generic filtering)
- Hierarchical structure = topic naming convention (generic topics)
- NPC decision engine = retrieval + LLM prompting (generic retrieval)

**You can use TracingRAG for ANY domain** that needs:
- Temporal tracking (how things evolved)
- Graph connections (how things relate)
- Accurate memory (what's currently true vs what was true)

The examples below show novel/NPC/project patterns, but the same generic primitives work for: research papers, business decisions, medical records, legal cases, scientific experiments, etc.

---

## 1. Structured Entity Types (User-Defined)

### Concept
TracingRAG doesn't prescribe entity types. You define whatever types your domain needs using the `entity_type` string field and structure them with `entity_schema`.

### Core Data Model (Generic)

```python
class MemoryState(BaseModel):
    # ... core fields ...

    # User-defined entity typing (domain-agnostic)
    entity_type: Optional[str] = None  # Any string: "character", "bug", "api_endpoint", etc.
    entity_schema: Optional[dict] = None  # Your custom structured data
```

**Examples of entity_type values across domains:**
- Novel writing: "character", "location", "plot_thread", "world_rule"
- Software dev: "bug", "feature", "api_endpoint", "database_schema"
- Research: "paper", "experiment", "hypothesis", "dataset"
- Medical: "patient", "diagnosis", "treatment", "symptom"
- Business: "customer", "transaction", "strategy", "competitor"

### Character Entity Schema
```python
{
    "entity_type": "character",
    "entity_schema": {
        "name": "Sarah",
        "personality_traits": ["brave", "impulsive", "loyal"],
        "age": 25,
        "occupation": "wizard",
        "emotional_state": "determined",
        "current_goal": "find ancient artifact",
        "knowledge_level": {
            "magic": "expert",
            "politics": "novice",
            "history": "intermediate"
        },
        "physical_description": "Tall, red hair, green eyes",
        "character_arc": "naive_to_wise"
    }
}
```

### Location Entity Schema
```python
{
    "entity_type": "location",
    "entity_schema": {
        "name": "Tavern of the Lost",
        "type": "building",
        "region": "Capital City",
        "description": "Dark, smoky tavern in lower district",
        "notable_features": ["hidden back room", "suspicious bartender"],
        "atmosphere": "tense",
        "first_appeared": "chapter_12"
    }
}
```

### World Rule Entity Schema
```python
{
    "entity_type": "world_rule",
    "entity_schema": {
        "category": "magic_system",
        "rule": "Magic exhausts user's life force",
        "exceptions": ["ley line locations", "full moon nights"],
        "established_in": "chapter_3",
        "last_demonstrated": "chapter_87"
    }
}
```

**Benefits:**
- Structured queries: "Get all characters in region X"
- Consistency checking: "Does this violate magic rules?"
- Type-specific operations: "Show character arc"

---

## 2. Relationship Tracking (First-Class, Not Just Edges)

### Problem
Current design has edges, but relationships need richer tracking for novels/NPCs.

### Solution: Relationship Entities

```python
class RelationshipEntity(BaseModel):
    """A relationship is itself a traced entity"""
    id: UUID
    relationship_type: str  # "friendship", "romantic", "rivalry", etc.
    participant_1: UUID  # Character ID
    participant_2: UUID  # Character ID

    # Temporal tracking
    trace_id: UUID  # This relationship has its own trace!
    current_state: str  # "close_friends", "enemies", "strangers"
    intensity: float  # 0.0-1.0

    # History
    key_events: List[UUID]  # Events that changed this relationship
    started_chapter: Optional[int]

    # Context
    public_vs_private: dict = {
        "public_perception": "enemies",
        "private_reality": "secret_allies"
    }
```

### Example: Character Relationship Evolution

```
Chapter 10: Sarah meets Marcus
→ Create relationship trace: sarah_marcus_relationship
→ State v1: "strangers" (intensity 0.0)

Chapter 25: Marcus saves Sarah's life
→ Promote relationship: sarah_marcus_relationship
→ State v2: "grateful_allies" (intensity 0.6)
→ Create edge: event_marcus_saves_sarah → relationship v2

Chapter 50: Betrayal revealed
→ Promote relationship: sarah_marcus_relationship
→ State v3: "bitter_enemies" (intensity 0.9)
→ Create edge: event_betrayal → relationship v3

Chapter 100: Reconciliation
→ Promote relationship: sarah_marcus_relationship
→ State v4: "forgiven_partners" (intensity 0.7)
```

**Query Examples:**
```python
# What's Sarah and Marcus's current relationship?
get_latest_state("sarah_marcus_relationship")
→ "forgiven_partners" (intensity 0.7)

# How did their relationship evolve?
get_trace("sarah_marcus_relationship")
→ strangers → allies → enemies → partners

# What event caused them to become enemies?
query_relationship_change("sarah_marcus_relationship", v2, v3)
→ event_betrayal (chapter 50)

# Show me all of Sarah's relationships
get_relationships(character="sarah")
→ sarah_marcus_relationship, sarah_father_relationship, etc.
```

---

## 3. Character Knowledge Base (What Does X Know?)

### Problem
In novels/games, characters have different knowledge. Sarah knows the truth, Marcus doesn't.

### Solution: Per-Character Knowledge Tracking

```python
class CharacterKnowledge(BaseModel):
    """Track what each character knows"""
    character_id: UUID
    knowledge_item_id: UUID  # ID of memory state
    learned_at: datetime  # When did they learn this?
    learned_from: Optional[UUID]  # Who/what told them?
    confidence: float  # How sure are they? (0.0-1.0)
    is_true: bool  # Is this actually true?

class KnowledgeGraph:
    """Separate graph per character"""

    def what_does_character_know(
        self,
        character_id: UUID,
        topic: str
    ) -> List[MemoryState]:
        """Get what this specific character knows about topic"""
        # Filter memories by character_knowledge
        return [
            state for state in query_memories(topic)
            if CharacterKnowledge.exists(character_id, state.id)
        ]

    def knowledge_gap(
        self,
        character_a: UUID,
        character_b: UUID
    ) -> dict:
        """What does A know that B doesn't?"""
        a_knowledge = get_all_knowledge(character_a)
        b_knowledge = get_all_knowledge(character_b)

        return {
            "a_knows_b_doesnt": a_knowledge - b_knowledge,
            "b_knows_a_doesnt": b_knowledge - a_knowledge,
            "shared": a_knowledge & b_knowledge
        }
```

### Example: Novel Writing

```
Chapter 100: Sarah discovers parents were murdered

# Create memory state
truth = create_memory(
    topic="sarah_parents_death",
    content="Parents were murdered by the king, not accident",
    entity_type="event"
)

# Sarah knows this
create_character_knowledge(
    character_id=sarah_id,
    knowledge_item_id=truth.id,
    learned_at=chapter_100_timestamp,
    confidence=1.0,
    is_true=True
)

# Marcus still believes the lie
official_story = get_memory("sarah_parents_death_official")
create_character_knowledge(
    character_id=marcus_id,
    knowledge_item_id=official_story.id,
    learned_at=chapter_1_timestamp,
    confidence=0.8,
    is_true=False  # It's actually false
)

# Later in Chapter 150: Writing a scene
query("What does Sarah know that Marcus doesn't about her parents?")
→ Returns: Sarah knows truth (murder), Marcus believes lie (accident)
→ Prevents continuity errors: Marcus won't reference murder he doesn't know about
```

### NPC Application

```python
# NPC Bob's knowledge base
bob_knowledge = {
    "player_alice": [
        "saved my shop",  # learned chapter 50, confidence 1.0
        "is trustworthy",  # learned chapter 50, confidence 1.0
        "likes spicy food"  # learned chapter 52, confidence 0.8
    ],
    "player_charlie": [
        "regular customer",  # learned chapter 30
        "always pays on time"  # learned chapter 35
    ],
    "world_events": [
        "dragon attacked",  # learned chapter 20
        "dragons fear silver"  # learned chapter 60
    ]
}

# When Alice interacts with Bob
bob_context = bob_knowledge["player_alice"]
# Bob remembers saving + knows Alice is trustworthy
# Affects dialogue and decisions
```

---

## 4. Hierarchical Structure for Content

### Problem
Novels have volumes → arcs → chapters. Need to query at any level.

### Solution: Hierarchical Topic Naming + Aggregation

```python
class HierarchyManager:
    """Manage hierarchical content structure"""

    # Naming convention
    PATTERNS = {
        "volume": "novel_name/volume_{n}",
        "arc": "novel_name/volume_{n}/arc_{m}",
        "chapter": "novel_name/volume_{n}/arc_{m}/chapter_{k}"
    }

    def query_volume_summary(self, volume: int):
        """Get summary of entire volume"""
        # Check if consolidated summary exists
        summary = get_consolidated_memory(f"novel/volume_{volume}")
        if summary:
            return summary

        # Otherwise aggregate from arcs
        arcs = query_pattern(f"novel/volume_{volume}/arc_*")
        return synthesize_summary(arcs)

    def get_all_in_hierarchy(self, level: str):
        """Get all items at hierarchical level"""
        # Get all chapters in arc 2 of volume 1
        chapters = query_pattern("novel/volume_1/arc_2/chapter_*")
        return chapters
```

### Query Examples

```python
# Volume-level
"What happened in Volume 3?"
→ Returns consolidated summary of entire volume
→ Can drill down to arc/chapter level if needed

# Arc-level
"Summarize the betrayal arc"
→ Returns summary of volume_2/arc_3
→ Includes key events across multiple chapters

# Chapter-level
"What happened in Chapter 145?"
→ Returns specific chapter state

# Cross-volume
"Trace Sarah's character arc across all volumes"
→ Finds all Sarah states across entire hierarchy
→ Returns evolution: V1 (naive) → V2 (learning) → V3 (master)
```

---

## 5. Consistency Checking & Validation

### Problem
Easy to introduce contradictions in long novels or complex game worlds.

### Solution: Automated Consistency Checker

```python
class ConsistencyChecker:
    """Validate new content against established facts"""

    async def check_new_content(
        self,
        content: str,
        chapter: int
    ) -> List[ConsistencyIssue]:
        """Check for potential issues"""
        issues = []

        # Extract entities mentioned
        entities = extract_entities(content)

        for entity in entities:
            # Check character knowledge
            if entity.type == "character":
                issues.extend(
                    await self.check_character_knowledge(entity, chapter)
                )

            # Check location consistency
            if entity.type == "location":
                issues.extend(
                    await self.check_location_rules(entity, chapter)
                )

            # Check world rules
            if mentions_magic(content):
                issues.extend(
                    await self.check_world_rules(content)
                )

        return issues

    async def check_character_knowledge(self, character, chapter):
        """Ensure character knows what they reference"""
        issues = []

        # What does character know at this point?
        knowledge = get_character_knowledge_at_time(
            character.id,
            chapter
        )

        # What are they referencing?
        references = extract_references(content)

        for ref in references:
            if ref not in knowledge:
                issues.append(
                    ConsistencyIssue(
                        type="knowledge_gap",
                        message=f"{character.name} references {ref} but "
                                f"shouldn't know about it until chapter {ref.revealed_in}",
                        severity="error"
                    )
                )

        return issues

    async def check_world_rules(self, content):
        """Ensure world rules aren't violated"""
        issues = []

        # Get established world rules
        rules = get_all_states(entity_type="world_rule")

        for rule in rules:
            if violates_rule(content, rule):
                issues.append(
                    ConsistencyIssue(
                        type="rule_violation",
                        message=f"Content violates established rule: {rule.content}",
                        severity="warning",
                        suggestion=f"Rule established in {rule.metadata['chapter']}"
                    )
                )

        return issues
```

### Example Usage

```python
# Writing Chapter 200
new_chapter = """
Sarah used magic to teleport across the kingdom instantly.
"""

issues = await consistency_checker.check_new_content(new_chapter, 200)
→ Returns: [
    ConsistencyIssue(
        type="rule_violation",
        message="Teleportation violates magic system rules",
        severity="error",
        suggestion="Magic system (Ch 3): 'Magic requires physical touch to target'"
    )
]

# Or:
new_chapter = """
Marcus mentioned Sarah's parents' murder.
"""

issues = await consistency_checker.check_new_content(new_chapter, 120)
→ Returns: [
    ConsistencyIssue(
        type="knowledge_gap",
        message="Marcus references murder but doesn't know until Chapter 180",
        severity="error",
        suggestion="Marcus still believes official story (accident)"
    )
]
```

---

## 6. Character Arc Tracking

### Problem
Need to track character development over time, not just current state.

### Solution: Explicit Arc Tracking

```python
class CharacterArc(BaseModel):
    """Track a character's development arc"""
    character_id: UUID
    arc_name: str  # "naive_to_wise", "villain_redemption", etc.

    # Stages
    stages: List[ArcStage] = []
    current_stage: int

    # Key moments
    inciting_incident: Optional[UUID]  # Event that started arc
    turning_points: List[UUID]  # Events that changed trajectory
    climax: Optional[UUID]
    resolution: Optional[UUID]

class ArcStage(BaseModel):
    stage_number: int
    name: str  # "denial", "acceptance", "mastery", etc.
    chapter_range: tuple[int, int]
    personality_state: UUID  # Link to personality trace state
    description: str

# Example: Sarah's arc
sarah_arc = CharacterArc(
    character_id=sarah_id,
    arc_name="naive_farm_girl_to_master_wizard",
    stages=[
        ArcStage(
            stage_number=1,
            name="denial",
            chapter_range=(1, 50),
            description="Afraid of magic, denies heritage"
        ),
        ArcStage(
            stage_number=2,
            name="discovery",
            chapter_range=(51, 150),
            description="Learns about powers, conflicted"
        ),
        ArcStage(
            stage_number=3,
            name="acceptance",
            chapter_range=(151, 250),
            description="Embraces magic, trains seriously"
        ),
        ArcStage(
            stage_number=4,
            name="mastery",
            chapter_range=(251, 350),
            description="Master wizard, mentoring others"
        )
    ],
    current_stage=4,
    turning_points=[
        event_first_magic_ch45,
        event_mentor_death_ch120,
        event_final_battle_ch240
    ]
)
```

### Query Examples

```python
# Where is Sarah in her arc at Chapter 180?
get_arc_stage(sarah_arc, chapter=180)
→ Stage 3: "acceptance" - Embraces magic, trains seriously

# What's Sarah's character trajectory?
visualize_arc(sarah_arc)
→ [denial] → [discovery] → [acceptance] → [mastery]

# What events shaped Sarah?
get_turning_points(sarah_arc)
→ First magic (Ch 45), Mentor death (Ch 120), Final battle (Ch 240)

# Should Sarah act this way in Chapter 200?
expected_behavior = get_expected_personality(sarah_arc, chapter=200)
→ "Confident with magic, still learning, not yet master"
validate_scene(new_scene, expected_behavior)
```

---

## 7. World State Management (Temporal World)

### Problem
World changes over time. Need to know "what is true at chapter X?"

### Solution: Temporal World State

```python
class WorldState(BaseModel):
    """State of the world at a specific time"""
    timestamp: datetime
    chapter: int

    # World properties at this time
    ruling_power: str
    active_conflicts: List[str]
    known_locations: List[UUID]
    active_factions: Dict[str, str]  # faction -> status
    world_rules: List[UUID]  # Active rules
    technology_level: str
    magic_availability: str

class TemporalWorldManager:
    """Manage world state over time"""

    def get_world_at_chapter(self, chapter: int) -> WorldState:
        """Get world state at specific chapter"""
        # Find latest world state before this chapter
        return query_world_state_before(chapter)

    def what_changed(
        self,
        chapter_start: int,
        chapter_end: int
    ) -> List[WorldChange]:
        """What changed in the world between chapters?"""
        start_state = self.get_world_at_chapter(chapter_start)
        end_state = self.get_world_at_chapter(chapter_end)

        return diff_world_states(start_state, end_state)
```

### Example

```python
# Chapter 50: Before the war
world_ch50 = get_world_at_chapter(50)
→ {
    "ruling_power": "Old Kingdom",
    "active_conflicts": [],
    "magic_availability": "rare"
}

# Chapter 200: After the war
world_ch200 = get_world_at_chapter(200)
→ {
    "ruling_power": "New Republic",
    "active_conflicts": ["border skirmishes"],
    "magic_availability": "common"
}

# What changed?
changes = what_changed(50, 200)
→ [
    "Old Kingdom fell, New Republic rose",
    "Magic became common after ley lines activated",
    "New conflict: border skirmishes with North"
]

# Consistency check
new_chapter = "In Chapter 210, the Old Kingdom still rules"
issues = check_world_consistency(new_chapter, 210)
→ Error: "Old Kingdom fell in Chapter 150"
```

---

## 8. NPC-Specific Features

### For Game NPCs: Emotional State & Goals

```python
class NPCState(BaseModel):
    """Current NPC state for game simulation"""
    npc_id: UUID

    # Emotional state
    emotional_state: str  # "happy", "angry", "fearful", etc.
    emotional_intensity: float  # 0.0-1.0
    mood_history: List[tuple[datetime, str]]  # Track mood changes

    # Goals & Motivations
    current_goal: str  # "sell goods", "find information", etc.
    goal_progress: float  # 0.0-1.0
    motivations: List[str]  # "greed", "revenge", "protection"

    # Personality
    personality_traits: Dict[str, float] = {
        "agreeableness": 0.7,
        "openness": 0.5,
        "extraversion": 0.8,
        "neuroticism": 0.3,
        "conscientiousness": 0.6
    }

    # Memory & Knowledge
    memory_trace_id: UUID  # This NPC's memory trace
    knowledge_graph_id: UUID  # What this NPC knows
    relationship_ids: List[UUID]  # This NPC's relationships

class NPCDecisionEngine:
    """Make decisions based on NPC state"""

    async def decide_action(
        self,
        npc: NPCState,
        context: dict
    ) -> Action:
        """Decide what NPC should do"""

        # Get NPC's memories relevant to context
        relevant_memories = query_npc_memories(
            npc.memory_trace_id,
            context
        )

        # Get NPC's relationships with entities in context
        relationships = get_relevant_relationships(
            npc.relationship_ids,
            context["entities"]
        )

        # Use LLM to decide action based on:
        # - Personality traits
        # - Current emotional state
        # - Current goal
        # - Relevant memories
        # - Relationships with entities present

        action = await llm.generate(
            prompt=f"""
            You are {npc.name} with personality: {npc.personality_traits}
            Current emotion: {npc.emotional_state} ({npc.emotional_intensity})
            Current goal: {npc.current_goal}

            Relevant memories:
            {format_memories(relevant_memories)}

            Relationships:
            {format_relationships(relationships)}

            Context: {context}

            What do you do?
            """
        )

        # Update NPC state based on action
        await self.update_npc_state(npc, action, context)

        return action
```

### Example: NPC Bob's Decision

```python
# Player Alice enters Bob's shop
context = {
    "location": "Bob's shop",
    "entities": ["player_alice"],
    "player_action": "enters and greets"
}

bob_state = get_npc_state("merchant_bob")
→ {
    "emotional_state": "grateful",
    "emotional_intensity": 0.8,
    "current_goal": "repay kindness to Alice",
    "personality_traits": {"agreeableness": 0.9, ...}
}

# Get Bob's memories of Alice
bob_memories = query_npc_memories(bob_state.memory_trace_id, "Alice")
→ [
    "Alice saved my shop from bandits",
    "Alice never asks for discounts",
    "Alice likes healing potions"
]

# Get Bob's relationship with Alice
relationship = get_relationship("bob", "alice")
→ "deeply_grateful" (intensity 0.9, since Chapter 50)

# Decide action
action = await npc_decision_engine.decide_action(bob_state, context)
→ "Bob greets Alice warmly, offers 30% discount, suggests new healing potion"

# Different player
context["entities"] = ["player_dave"]
bob_memories_dave = query_npc_memories(bob_state.memory_trace_id, "Dave")
→ [
    "Dave tried to steal from me once",
    "Dave was caught and apologized",
    "Dave hasn't caused trouble since"
]

relationship_dave = get_relationship("bob", "dave")
→ "wary_forgiveness" (intensity 0.4)

action_dave = await npc_decision_engine.decide_action(bob_state, context)
→ "Bob acknowledges Dave cautiously, watches him carefully, no discount"
```

---

## 9. Query Templates & Shortcuts

### Problem
Common questions should be easy to ask.

### Solution: Pre-built Query Templates

```python
class QueryTemplates:
    """Common queries for novels/games"""

    # Character queries
    def character_summary(self, character: str, chapter: int):
        """Who is this character at this point?"""
        return query_template(
            "character_summary",
            character=character,
            as_of_chapter=chapter
        )

    def character_relationships(self, character: str):
        """All relationships for a character"""
        return get_all_relationships(character_id=character)

    def character_arc(self, character: str):
        """Character's development arc"""
        return get_character_arc(character)

    def what_does_character_know(self, character: str, topic: str):
        """Character-specific knowledge"""
        return query_character_knowledge(character, topic)

    # Plot queries
    def plot_thread_status(self, thread: str):
        """Status of a plot thread"""
        return get_latest_state(f"plot/{thread}")

    def unresolved_plots(self):
        """All unresolved plot threads"""
        return query_unresolved(entity_type="plot_thread")

    def volume_summary(self, volume: int):
        """Summary of entire volume"""
        return get_consolidated_memory(f"novel/volume_{volume}")

    # World queries
    def world_at_chapter(self, chapter: int):
        """World state at specific chapter"""
        return get_world_at_chapter(chapter)

    def location_description(self, location: str, chapter: int):
        """What does location look like at this time?"""
        return query_location_state(location, chapter)

    # Consistency checks
    def can_character_do_this(
        self,
        character: str,
        action: str,
        chapter: int
    ):
        """Is this action consistent with character?"""
        return validate_character_action(character, action, chapter)

    def does_this_violate_rules(self, content: str):
        """Check for world rule violations"""
        return check_world_rules(content)

    # NPC queries
    def npc_opinion_of_player(self, npc: str, player: str):
        """What does NPC think of player?"""
        relationship = get_relationship(npc, player)
        memories = query_npc_memories(npc, player)
        return summarize_opinion(relationship, memories)

    def npc_decision(self, npc: str, situation: str):
        """What would NPC do in this situation?"""
        return npc_decision_engine.decide_action(npc, situation)
```

---

## 10. Implementation Priority for Novel/NPC Use Cases

### Phase 1: Essential (Implement First)
1. **Entity Type System** - Character, Location, Plot, World Rule types
2. **Character Knowledge Tracking** - What does X know?
3. **Latest State Tracking** - Already designed, critical for "current status"
4. **Hierarchical Structure** - Volume/Arc/Chapter organization

### Phase 2: Important (Next Priority)
5. **Relationship Entities** - Track relationship evolution
6. **Character Arc Tracking** - Explicit development tracking
7. **Consistency Checking** - Validate new content
8. **Query Templates** - Easy common queries

### Phase 3: Advanced (Nice to Have)
9. **World State Management** - Temporal world tracking
10. **NPC Decision Engine** - For game simulations
11. **Emotional State Tracking** - For NPCs
12. **Auto-linking** - Automatically detect entity mentions

---

## Complete Example: Novel Writing Workflow

```python
# Phase 1: Setup
novel = create_project("epic_fantasy_saga")

# Define main character
sarah = create_entity(
    type="character",
    name="Sarah",
    schema={
        "personality": ["brave", "impulsive"],
        "age": 18,
        "role": "protagonist"
    }
)

# Define character arc
sarah_arc = create_character_arc(
    character=sarah,
    arc_name="naive_to_master",
    stages=["denial", "discovery", "acceptance", "mastery"]
)

# Phase 2: Writing Chapter 1
chapter_1 = create_memory(
    topic="novel/volume_1/arc_1/chapter_1",
    content="Sarah discovers she has magic powers...",
    entity_type="event"
)

# Track character knowledge
mark_character_knowledge(
    character=sarah,
    knowledge="sarah_has_magic",
    learned_in=chapter_1
)

# Phase 3: Writing Chapter 100
# Check what Sarah knows
sarah_knowledge = what_does_character_know(sarah, "her_heritage")
→ Returns: Sarah learned about heritage in Chapter 50

# Check world state
world = get_world_at_chapter(100)
→ Returns: War started Chapter 75, kingdom fallen

# Write new content
new_scene = """
Sarah used her mastery of fire magic to defeat the enemy.
"""

# Consistency check
issues = check_consistency(new_scene, chapter=100)
→ Warning: "Sarah's magic is ice, not fire (established Ch 3)"

# Fix and proceed
corrected_scene = """
Sarah used her mastery of ice magic to freeze the enemy.
"""

# Save chapter
chapter_100 = create_memory(
    topic="novel/volume_2/arc_3/chapter_100",
    content=corrected_scene
)

# Update character arc
update_arc_progress(sarah_arc, chapter=100)
→ Sarah now in "acceptance" stage

# Phase 4: Querying for context
# Writing Chapter 200, need to recall what happened
volume_2_summary = query_volume_summary(2)
→ Returns consolidated summary of entire Volume 2

sarah_development = query_character_arc(sarah, chapter=200)
→ Returns: Sarah in "mastery" stage, confident, mentoring others

active_plots = query_unresolved_plots()
→ Returns: "Dark cult" plot (started Ch 80, unresolved)

# Phase 5: Consistency across entire novel
# Final review before publishing
issues = check_entire_novel_consistency()
→ Returns list of potential issues
→ "Character X disappeared after Chapter 50 with no explanation"
→ "Plot thread Y mentioned but never resolved"
```

---

## Benefits Summary

These domain-specific features provide:

### For Novel Writing:
✅ **Perfect Consistency**: Never forget character knowledge, world rules
✅ **Character Arc Tracking**: Ensure logical character development
✅ **Hierarchical Organization**: Volume/Arc/Chapter structure
✅ **Automated Checking**: Catch contradictions before they're published
✅ **Relationship Evolution**: Track how relationships change over time
✅ **World State**: Know what's true at any point in timeline

### For NPC Simulation:
✅ **Individual Memory**: Each NPC has own memory and knowledge
✅ **Personality-Driven**: Decisions based on traits + memories
✅ **Relationship Aware**: NPCs treat different players differently
✅ **Emotional State**: NPCs have moods that affect behavior
✅ **Goal-Oriented**: NPCs have motivations that drive actions
✅ **Evolving Thoughts**: NPC opinions change based on interactions

## Key Insight: Generic System, Specific Applications

**TracingRAG remains a general-purpose, domain-agnostic system**. The patterns above show specific applications, but you can apply the same generic primitives to ANY domain:

**Research**: Papers as entities, citations as edges, field evolution as traces
**Business**: Decisions as states, stakeholders as entities, strategies as traces
**Medical**: Patients as entities, treatments as traces, diagnoses as states
**Legal**: Cases as entities, precedents as edges, rulings as states
**Science**: Experiments as states, hypotheses as traces, findings as edges

The core system provides:
1. **Generic temporal traces** → You define what to track
2. **Generic graph relationships** → You define what connects
3. **Generic memory strength** → You define what's important
4. **Generic consolidation** → You define time scales
5. **Generic entity_schema** → You define structure

**The examples are illustrative, not prescriptive**. TracingRAG is broad and capable of solving any problem that involves:
- Memory evolution over time
- Complex interconnections
- Temporal accuracy
- Graph-based reasoning

These novel/NPC/project patterns simply demonstrate the power of the generic primitives.
