#!/usr/bin/env python3
"""
Example 2: NPC Memory & Character Simulation

This example demonstrates how to use TracingRAG for game NPCs and
AI characters with persistent memory, including:
- Tracking interactions with players
- Evolving relationships based on actions
- Remembering context across sessions
- Querying NPC knowledge and feelings

Use Case: Video games, interactive fiction, chatbots with memory
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.client import AsyncTracingRAGClient


async def simulate_npc_interactions():
    """Simulate NPC interactions with a player"""
    print("=" * 70)
    print("Example 2: NPC Memory & Character Simulation")
    print("=" * 70)
    print()

    # Initialize client
    print("📡 Connecting to TracingRAG API...")
    async with AsyncTracingRAGClient("http://localhost:8000") as client:
        health = await client.health()
        if health["status"] != "healthy":
            print("❌ API is not healthy. Please start the API server.")
            return
        print("✅ Connected to TracingRAG API\n")

        # ====================================================================
        # Setup: Define NPCs
        # ====================================================================
        print("🎭 Setting up NPCs...")
        print()

        # NPC 1: Merchant Elena
        _elena_intro = await client.create_memory(
            topic="npc_elena_merchant",
            content=(
                "Elena is a kind merchant who runs the general store in Rivertown. "
                "She's been trading for 20 years and knows everyone in town. "
                "She values honesty and fair dealing. Her husband was a blacksmith "
                "who passed away 5 years ago."
            ),
            tags=["npc", "merchant", "elena"],
            confidence=1.0,
            custom_metadata={"npc_type": "merchant", "location": "rivertown_store"},
        )
        print("   ✅ Created NPC: Elena (Merchant)")

        # NPC 2: Guard Captain Marcus
        _marcus_intro = await client.create_memory(
            topic="npc_marcus_guard",
            content=(
                "Marcus is the stern but fair captain of the town guard. "
                "He's suspicious of strangers and takes his duty seriously. "
                "He's been protecting Rivertown for 10 years and has seen many threats."
            ),
            tags=["npc", "guard", "marcus"],
            confidence=1.0,
            custom_metadata={"npc_type": "guard_captain", "location": "guard_barracks"},
        )
        print("   ✅ Created NPC: Marcus (Guard Captain)\n")

        # ====================================================================
        # Day 1: First Interactions
        # ====================================================================
        print("📅 Day 1: Player arrives in Rivertown...")
        print()

        # Player meets Elena
        elena_day1 = await client.create_memory(
            topic="elena_player_relationship",
            content=(
                "Player (adventurer named 'Aria') visited Elena's store for the first time. "
                "Elena was welcoming. Aria bought some basic supplies (rope, rations) "
                "and asked about the town. Elena shared information about local landmarks "
                "and warned about bandits on the eastern road. Aria was polite and paid fairly."
            ),
            tags=["interaction", "elena", "day1", "positive"],
            confidence=0.9,
            custom_metadata={"day": 1, "player_action": "friendly_purchase"},
        )
        print("   💬 Elena meets player (positive interaction)")

        # Player meets Marcus
        marcus_day1 = await client.create_memory(
            topic="marcus_player_relationship",
            content=(
                "Player 'Aria' approached Marcus at the guard post. Marcus was suspicious "
                "of the new adventurer and questioned their purpose in town. Aria explained "
                "they're just passing through and looking for work. Marcus gave a stern warning "
                "about following town laws. Aria agreed respectfully."
            ),
            tags=["interaction", "marcus", "day1", "neutral"],
            confidence=0.85,
            custom_metadata={"day": 1, "player_action": "respectful_compliance"},
        )
        print("   💬 Marcus meets player (suspicious but neutral)\n")

        # ====================================================================
        # Day 3: Player helps town
        # ====================================================================
        print("📅 Day 3: Player helps defend town from bandits...")
        print()

        marcus_day3 = await client.create_memory(
            topic="marcus_player_relationship",
            content=(
                "Bandits attacked Rivertown on Day 3. Player 'Aria' fought bravely alongside "
                "the town guard. Marcus was impressed by Aria's combat skills and willingness "
                "to help without being asked. After the battle, Marcus thanked Aria personally "
                "and said they've proven themselves trustworthy. Marcus now sees Aria as an ally."
            ),
            parent_state_id=marcus_day1.id,
            tags=["interaction", "marcus", "day3", "positive", "combat"],
            confidence=0.95,
            custom_metadata={"day": 3, "player_action": "heroic_defense"},
        )
        print("   ⚔️  Player helps defend town - Marcus is impressed")

        elena_day3 = await client.create_memory(
            topic="elena_player_relationship",
            content=(
                "Elena saw Aria fighting the bandits from her store window. After the battle, "
                "Aria came to check if Elena was safe. Elena was touched by the concern and "
                "offered Aria a free healing potion as thanks. She mentioned that Aria reminds "
                "her of her late husband - brave and kind-hearted."
            ),
            parent_state_id=elena_day1.id,
            tags=["interaction", "elena", "day3", "positive", "emotional"],
            confidence=0.92,
            custom_metadata={"day": 3, "player_action": "protective_concern"},
        )
        print("   💝 Elena grows fond of player\n")

        # ====================================================================
        # Day 7: Quest Decision
        # ====================================================================
        print("📅 Day 7: Player faces a moral choice...")
        print()

        _quest_memory = await client.create_memory(
            topic="player_moral_choice",
            content=(
                "Player discovered that a local noble has been embezzling town funds meant "
                "for the guard. Aria reported this to Marcus despite the noble's threats. "
                "Marcus investigated and arrested the noble. Elena heard about Aria's integrity "
                "and became even more trusting."
            ),
            tags=["quest", "moral_choice", "integrity"],
            confidence=1.0,
            custom_metadata={"day": 7, "choice": "honest_reporting"},
        )
        print("   ⚖️  Player chooses integrity over personal gain")

        # Update relationships
        _marcus_day7 = await client.create_memory(
            topic="marcus_player_relationship",
            content=(
                "Marcus deeply respects Aria now. Aria risked angering a powerful noble to "
                "do the right thing. Marcus considers Aria a true friend and ally. "
                "He's offered Aria a position as honorary guard if they ever want to settle down."
            ),
            parent_state_id=marcus_day3.id,
            tags=["interaction", "marcus", "day7", "very_positive", "friendship"],
            confidence=0.98,
            custom_metadata={"day": 7, "relationship_level": "friend"},
        )
        print("   🤝 Marcus now considers player a friend")

        _elena_day7 = await client.create_memory(
            topic="elena_player_relationship",
            content=(
                "Elena is proud of Aria's choice. She confided that her husband used to stand "
                "up to corruption too. Elena now treats Aria like family, always giving "
                "discounts and sharing local gossip. She worries about Aria's safety when "
                "they go on dangerous adventures."
            ),
            parent_state_id=elena_day3.id,
            tags=["interaction", "elena", "day7", "very_positive", "motherly"],
            confidence=0.96,
            custom_metadata={"day": 7, "relationship_level": "family_like"},
        )
        print("   👪 Elena treats player like family\n")

        # ====================================================================
        # Query NPC Memories
        # ====================================================================
        print("🔍 Querying NPC memories...")
        print()

        # Query Marcus's feelings
        print("   Q: 'How does Marcus feel about the player?'")
        marcus_query = await client.query(
            "How does Marcus the guard captain feel about player Aria?",
            include_history=True,
        )
        if marcus_query.answer:
            print(f"   A: {marcus_query.answer[:200]}...\n")

        # Query Elena's feelings
        print("   Q: 'What is Elena's relationship with the player?'")
        elena_query = await client.query(
            "What is Elena the merchant's relationship with player Aria?",
            include_history=True,
        )
        if elena_query.answer:
            print(f"   A: {elena_query.answer[:200]}...\n")

        # ====================================================================
        # View Relationship Evolution
        # ====================================================================
        print("📈 Viewing relationship evolution...")
        print()

        marcus_trace = await client.get_trace("marcus_player_relationship")
        print(f"   Marcus's opinion evolved over {len(marcus_trace)} interactions:")
        for i, state in enumerate(marcus_trace, 1):
            day = state.custom_metadata.get("day", "?")
            sentiment = "suspicious" if i == 1 else "impressed" if i == 2 else "friend"
            print(f"      Day {day}: {sentiment}")
        print()

        elena_trace = await client.get_trace("elena_player_relationship")
        print(f"   Elena's feelings grew over {len(elena_trace)} interactions:")
        for i, state in enumerate(elena_trace, 1):
            day = state.custom_metadata.get("day", "?")
            sentiment = "friendly" if i == 1 else "fond" if i == 2 else "motherly"
            print(f"      Day {day}: {sentiment}")
        print()

        # ====================================================================
        # Simulate New Session (30 days later)
        # ====================================================================
        print("⏰ [30 days pass - player returns to town]")
        print()

        print("   Q: 'Does Marcus remember me?'")
        memory_query = await client.query(
            "Does Marcus remember the adventurer Aria who helped defend the town?",
            include_history=True,
            use_agent=True,
        )
        if memory_query.answer:
            print(f"   A: {memory_query.answer[:250]}...\n")

    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("   - NPCs remember all interactions with persistent context")
    print("   - Relationships evolve naturally based on player actions")
    print("   - Memory spans multiple sessions (days, weeks, months)")
    print("   - Queries understand emotional context and history")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(simulate_npc_interactions())
    except KeyboardInterrupt:
        print("\n\n⚠️  Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("   1. TracingRAG API is running")
        print("   2. All services are healthy: poetry run python scripts/verify_setup.py")
        sys.exit(1)
