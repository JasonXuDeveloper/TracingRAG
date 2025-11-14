#!/usr/bin/env python3
"""
Example 3: Novel Writing & World-Building

This example demonstrates how to use TracingRAG for creative writing,
including:
- Tracking character development across chapters
- Maintaining world-building consistency
- Managing plot threads and their evolution
- Querying story elements for continuity

Use Case: Authors, screenwriters, worldbuilders, D&D dungeon masters
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.client import AsyncTracingRAGClient


async def novel_writing_example():
    """Demonstrate novel writing with TracingRAG"""
    print("=" * 70)
    print("Example 3: Novel Writing & World-Building")
    print("=" * 70)
    print()
    print("📖 Writing 'The Last Starship' - A Sci-Fi Novel")
    print()

    async with AsyncTracingRAGClient("http://localhost:8000") as client:
        health = await client.health()
        if health["status"] != "healthy":
            print("❌ API is not healthy. Please start the API server.")
            return
        print("✅ Connected to TracingRAG API\n")

        # ====================================================================
        # World-Building: Setting
        # ====================================================================
        print("🌍 Phase 1: World-Building...")
        print()

        await client.create_memory(
            topic="setting_starship_nova",
            content=(
                "The Starship Nova is humanity's last colony ship, carrying 10,000 "
                "survivors after Earth's destruction. The ship has been traveling for "
                "200 years toward planet Kepler-442b. The ship is divided into sectors: "
                "Command Deck, Residential Rings (A-E), Hydroponics Bay, Engineering Core, "
                "and Medical Ward."
            ),
            tags=["worldbuilding", "setting", "spaceship"],
            confidence=1.0,
        )
        print("   ✅ Created setting: Starship Nova")

        factions_v1 = await client.create_memory(
            topic="political_factions",
            content=(
                "Three main factions on the ship: "
                "1) The Council - elected leaders who maintain order "
                "2) The Engineers - workers who keep ship running, demand more representation "
                "3) The Purists - religious group who believe leaving Earth was a sin"
            ),
            tags=["worldbuilding", "politics", "factions"],
            confidence=0.9,
        )
        print("   ✅ Created political landscape")
        print()

        # ====================================================================
        # Characters: Initial Concepts
        # ====================================================================
        print("👥 Phase 2: Character Creation...")
        print()

        # Protagonist
        char_sarah_v1 = await client.create_memory(
            topic="character_sarah_chen",
            content=(
                "Sarah Chen, age 28, lead engineer in the Engineering Core. "
                "Born on the ship - part of the second generation. Brilliant but "
                "rebellious. Discovered a critical system failure that the Council "
                "is hiding. She's torn between loyalty to her engineering team and "
                "duty to all passengers."
            ),
            tags=["character", "protagonist", "engineer"],
            confidence=1.0,
            custom_metadata={"role": "protagonist", "age": 28, "sector": "engineering"},
        )
        print("   ✅ Created protagonist: Sarah Chen (Engineer)")

        # Antagonist
        await client.create_memory(
            topic="character_marcus_hall",
            content=(
                "Marcus Hall, age 45, Council Chairman. Former military officer. "
                "He believes in order above all else and is willing to suppress "
                "information to prevent panic. Has a hidden agenda: he knows the "
                "ship won't make it to Kepler-442b and is planning something drastic."
            ),
            tags=["character", "antagonist", "council"],
            confidence=1.0,
            custom_metadata={"role": "antagonist", "age": 45, "sector": "command"},
        )
        print("   ✅ Created antagonist: Marcus Hall (Council Chair)")

        # Supporting Character
        await client.create_memory(
            topic="character_david_wright",
            content=(
                "David Wright, age 30, medical officer and Sarah's childhood friend. "
                "Caught in the middle of the conflict. He wants to help both Sarah "
                "and the Council find a peaceful solution."
            ),
            tags=["character", "supporting", "medical"],
            confidence=1.0,
            custom_metadata={"role": "supporting", "age": 30, "sector": "medical"},
        )
        print("   ✅ Created supporting: David Wright (Medical Officer)")
        print()

        # ====================================================================
        # Plot Development: Chapter 1-5
        # ====================================================================
        print("📝 Phase 3: Writing early chapters...")
        print()

        await client.create_memory(
            topic="plot_discovery_arc",
            content=(
                "Chapters 1-5: Sarah discovers failing life support systems during "
                "routine maintenance. She reports to her supervisor but is told to stay "
                "quiet. David notices Sarah's stress and she confides in him. "
                "Marcus learns of Sarah's discovery and summons her to Command."
            ),
            tags=["plot", "chapters_1_5", "discovery"],
            confidence=0.95,
        )
        print("   ✅ Plotted chapters 1-5: Discovery arc")

        # ====================================================================
        # Character Development: Sarah's Evolution
        # ====================================================================
        print("🎭 Phase 4: Character development...")
        print()

        # Sarah learns a shocking truth and her character evolves
        await client.create_memory(
            topic="character_sarah_chen",
            content=(
                "Sarah Chen (Chapter 8): After confronting Marcus, Sarah learned the "
                "terrible truth - Kepler-442b was destroyed by a supernova 50 years ago "
                "but the Council hid this information. The ship has been traveling "
                "toward nothing. This revelation hardens Sarah. She's no longer just "
                "a brilliant engineer - she becomes a leader of a rebellion. Her goal "
                "is to take control of the ship and find a new destination."
            ),
            parent_state_id=char_sarah_v1.id,
            tags=["character", "protagonist", "engineer", "evolved"],
            confidence=0.92,
            custom_metadata={
                "role": "protagonist",
                "age": 28,
                "sector": "engineering",
                "chapter": 8,
                "arc": "rebellion_leader",
            },
        )
        print("   ✅ Sarah's character evolved (Chapter 8): Now a rebellion leader")

        # ====================================================================
        # Plot Twist: Factions Shift
        # ====================================================================
        print("🔀 Phase 5: Plot twist...")
        print()

        await client.create_memory(
            topic="political_factions",
            content=(
                "Factions after Chapter 10: "
                "1) The Council - split into loyalists (with Marcus) and reformists (who "
                "want to reveal the truth) "
                "2) The Engineers - united behind Sarah, demanding democratic reform "
                "3) The Purists - surprisingly ally with Sarah, believing truth is sacred "
                "4) NEW: The Navigators - neutral faction of scientists who have "
                "discovered a potential new planet, closer than Kepler-442b"
            ),
            parent_state_id=factions_v1.id,
            tags=["worldbuilding", "politics", "factions", "updated"],
            confidence=0.88,
        )
        print("   ✅ Political landscape evolved with new faction")

        # ====================================================================
        # Continuity Checks: Query Story Elements
        # ====================================================================
        print("\n🔍 Phase 6: Checking story continuity...")
        print()

        # Check Sarah's character arc
        print("   Q: 'How has Sarah changed since the beginning?'")
        sarah_arc = await client.query(
            "How has Sarah Chen's character developed from the beginning of the story?",
            include_history=True,
        )
        if sarah_arc.answer:
            print(f"   A: {sarah_arc.answer[:250]}...\n")

        # Check political situation
        print("   Q: 'What are the current political factions?'")
        factions_query = await client.query(
            "What are the political factions on the starship now?",
        )
        if factions_query.answer:
            print(f"   A: {factions_query.answer[:250]}...\n")

        # Check relationships
        print("   Q: 'What is the relationship between Sarah and David?'")
        relationship_query = await client.query(
            "What is the relationship between Sarah Chen and David Wright?",
            include_related=True,
        )
        if relationship_query.answer:
            print(f"   A: {relationship_query.answer[:250]}...\n")

        # ====================================================================
        # Planning Future Chapters
        # ====================================================================
        print("📅 Phase 7: Planning future chapters...")
        print()

        await client.create_memory(
            topic="plot_resolution_plan",
            content=(
                "Planned Chapters 15-20 (Resolution): "
                "Sarah and Marcus must work together when an external threat appears - "
                "an alien probe that's tracking the ship. The Navigators' new planet "
                "is real but occupied by unknown beings. Final choice: risk contact "
                "with aliens or continue drifting in space. Sarah proposes a democratic "
                "vote - first in ship's history. Themes: trust, democracy, hope."
            ),
            tags=["plot", "chapters_15_20", "resolution", "plan"],
            confidence=0.75,
            custom_metadata={"status": "planned", "needs_revision": False},
        )
        print("   ✅ Planned ending: Democratic vote on first contact")

        # ====================================================================
        # View Character Evolution
        # ====================================================================
        print("\n📈 Phase 8: Viewing character evolution...")
        print()

        sarah_trace = await client.get_trace("character_sarah_chen")
        print(f"   Sarah Chen evolved through {len(sarah_trace)} versions:")
        for state in sarah_trace:
            chapter = state.custom_metadata.get("chapter", "initial")
            arc = state.custom_metadata.get("arc", "intro")
            print(f"      {chapter}: {arc}")
        print()

        # ====================================================================
        # Check for Plot Holes
        # ====================================================================
        print("🔎 Phase 9: Checking for plot holes...")
        print()

        print("   Q: 'How did the Council hide the Kepler-442b supernova for 50 years?'")
        plot_hole = await client.query(
            "How did the Council hide the Kepler-442b supernova information for 50 years? "
            "Who else knew about it?",
            use_agent=True,
            include_related=True,
        )
        if plot_hole.answer:
            print(f"   A: {plot_hole.answer[:200]}...")
            if "not found" in plot_hole.answer.lower() or "unclear" in plot_hole.answer.lower():
                print("   ⚠️  Potential plot hole detected! Need to add this detail.")
        print()

        # ====================================================================
        # Promotion: Create Chapter Summary
        # ====================================================================
        print("📚 Phase 10: Promoting to chapter summary...")
        print()

        summary_result = await client.promote_memory(
            topic="plot_discovery_arc",
            reason="Completing discovery arc - synthesizing chapters 1-5",
        )
        print("   ✅ Created synthesized summary:")
        print(f"   {summary_result.new_state.content[:200]}...")
        if summary_result.synthesis_sources:
            print(f"   📝 Synthesized from {len(summary_result.synthesis_sources)} sources")
        print()

    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("   - Track character evolution across chapters")
    print("   - Maintain world-building consistency")
    print("   - Query for continuity and plot holes")
    print("   - Synthesize chapter summaries automatically")
    print("   - Perfect for complex narratives with many moving parts")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(novel_writing_example())
    except KeyboardInterrupt:
        print("\n\n⚠️  Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("   1. TracingRAG API is running")
        print("   2. All services are healthy: poetry run python scripts/verify_setup.py")
        sys.exit(1)
