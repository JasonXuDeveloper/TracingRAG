"""
Example 06: Version-Bound Relationships and Asymmetric Knowledge

This example demonstrates TracingRAG's advanced relationship management:
1. Version-bound relationships (historical facts that don't auto-update)
2. Asymmetric knowledge (A knows X but B doesn't)
3. Multiple relationships to different versions of the same entity
4. Complex relationship evolution with betrayal and secrets

Scenario: Corporate Power Struggle
- Sarah (CEO) trusts Marcus (CTO)
- Marcus secretly plans to leave with Alex (competitor CEO)
- Elena (engineer) is Marcus's friend, gets recruited
- Lisa (new hire) discovers the plot but is silenced
- David (HR) investigates and reports to Sarah
- Sarah learns about Marcus but not Elena
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.client import AsyncTracingRAGClient


async def main():
    """Demonstrate version-bound relationships and asymmetric knowledge"""

    # Initialize client
    client = AsyncTracingRAGClient(base_url="http://localhost:8000", timeout=3000.0)

    print("=" * 80)
    print("TracingRAG Example 06: Version-Bound Relationships & Asymmetric Knowledge")
    print("=" * 80)
    print()

    # ============================================================================
    # Phase 1: Initial State - Everyone at v1, normal relationships
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Initial State - Company Hierarchy")
    print("=" * 80)

    await client.create_memory(
        topic="Sarah Chen (CEO)",
        content="""Sarah Chen is the CEO of TechFlow Inc., a fast-growing AI startup.
        She founded the company 5 years ago and has built a strong team.
        Sarah trusts her CTO Marcus Rivera to handle all technical decisions.
        She values loyalty and transparency, and believes in her team's integrity.
        Sarah recently hired David Park as HR Director to improve company culture.""",
        tags=["person", "ceo", "founder"],
    )
    print("✓ Created Sarah v1: CEO who trusts her team")

    await client.create_memory(
        topic="Marcus Rivera (CTO)",
        content="""Marcus Rivera is the CTO of TechFlow Inc., hired 3 years ago by Sarah.
        He's technically brilliant and has led the engineering team to great success.
        Marcus is friends with Elena Kim, a senior engineer on his team.
        He feels undervalued and believes he deserves more equity and recognition.
        Marcus respects Sarah but is starting to feel frustrated with the slow growth.""",
        tags=["person", "cto", "engineering"],
    )
    print("✓ Created Marcus v1: Frustrated but loyal CTO")

    await client.create_memory(
        topic="Elena Kim (Senior Engineer)",
        content="""Elena Kim is a senior engineer at TechFlow Inc., been there for 2 years.
        She's Marcus's close friend and confidante, they often have lunch together.
        Elena is talented and hardworking, but feels the company doesn't pay competitively.
        She's loyal to Marcus and admires his technical vision.
        Elena doesn't know much about the executive dynamics between Sarah and Marcus.""",
        tags=["person", "engineer", "senior"],
    )
    print("✓ Created Elena v1: Senior engineer, Marcus's friend")

    await client.create_memory(
        topic="David Park (HR Director)",
        content="""David Park is the newly hired HR Director at TechFlow Inc.
        Sarah hired him to improve company culture and employee retention.
        David is perceptive and notices tension in the engineering team.
        He reports directly to Sarah and takes his duty of protecting the company seriously.
        David has started monitoring employee satisfaction and potential flight risks.""",
        tags=["person", "hr", "director"],
    )
    print("✓ Created David v1: Vigilant HR Director")

    await client.create_memory(
        topic="Lisa Wong (Junior Engineer)",
        content="""Lisa Wong just joined TechFlow Inc. two months ago as a junior engineer.
        She's enthusiastic and eager to learn, fresh out of college.
        Lisa reports to Elena and is learning the codebase.
        She's unaware of any company politics and focused on proving herself.
        Lisa admires the technical team and wants to grow her career here.""",
        tags=["person", "engineer", "junior"],
    )
    print("✓ Created Lisa v1: Naive new hire")

    await client.create_memory(
        topic="Alex Zhang (Competitor CEO)",
        content="""Alex Zhang is the CEO of NeuralWave, a direct competitor to TechFlow.
        His company has more funding and is aggressively recruiting talent.
        Alex has been watching TechFlow's technology and knows Marcus's work.
        He's been looking for an opportunity to poach TechFlow's CTO.
        Alex is ruthless in business and sees this as strategic acquisition of talent.""",
        tags=["person", "competitor", "ceo"],
    )
    print("✓ Created Alex v1: Aggressive competitor")

    await asyncio.sleep(2)

    # ============================================================================
    # Phase 2: Marcus's Betrayal - Secret contact with Alex
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Marcus's Betrayal (Secret from Sarah)")
    print("=" * 80)

    await client.create_memory(
        topic="Marcus Rivera (CTO)",
        content="""Marcus Rivera has been secretly meeting with Alex Zhang from NeuralWave.
        Alex offered Marcus a significantly better compensation package and equity stake.
        Marcus is planning to leave TechFlow and join NeuralWave as their Chief Architect.
        He's been sharing high-level technical insights with Alex during negotiations.
        Marcus hasn't told Sarah anything - he plans to resign with 2 weeks notice.
        He confided in Elena, his close friend, about the opportunity.
        Marcus feels guilty but believes this is best for his career growth.""",
        tags=["person", "cto", "betrayal", "leaving"],
    )
    print("✓ Evolved Marcus to v2: Secretly planning to leave")
    print("  → Sarah DOES NOT KNOW about this betrayal")
    print("  → Elena KNOWS about Marcus's plans")

    await asyncio.sleep(2)

    # ============================================================================
    # Phase 3: Elena Joins the Plot
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Elena Recruited (Lisa Discovers)")
    print("=" * 80)

    await client.create_memory(
        topic="Elena Kim (Senior Engineer)",
        content="""Elena Kim, after learning about Marcus's plan, was also contacted by Alex.
        Marcus recommended Elena to Alex, knowing she was also dissatisfied with TechFlow.
        Elena agreed to join NeuralWave and will leave with Marcus.
        She's been more careful about hiding this, not telling anyone except Marcus.
        Elena is excited about the opportunity but feels bad about leaving Sarah.
        Lisa Wong accidentally overheard Elena on a phone call with Alex.
        Elena discovered Lisa eavesdropping and warned her to stay silent.
        Elena told Lisa that if she speaks up, it would hurt her career prospects.""",
        tags=["person", "engineer", "betrayal", "leaving"],
    )
    print("✓ Evolved Elena to v2: Also planning to leave")
    print("  → Sarah DOES NOT KNOW about Elena's involvement")
    print("  → Marcus KNOWS Elena is joining him")
    print("  → Lisa KNOWS the secret but was threatened")

    await asyncio.sleep(2)

    await client.create_memory(
        topic="Lisa Wong (Junior Engineer)",
        content="""Lisa Wong accidentally discovered Elena's plan to leave with Marcus.
        She overheard Elena on a confidential phone call with Alex from NeuralWave.
        Elena caught Lisa eavesdropping and intimidated her into silence.
        Lisa is scared of damaging her career by speaking up.
        She's conflicted - should she tell David or Sarah? But Elena threatened her.
        Lisa is no longer naive about company politics and feels trapped.
        She's worried about what will happen when Marcus and Elena leave.""",
        tags=["person", "engineer", "knows_secret", "conflicted"],
    )
    print("✓ Evolved Lisa to v2: Knows the secret, intimidated")
    print("  → Lisa is HIDDEN_FROM Sarah (not reporting)")
    print("  → Lisa INFLUENCED_BY Elena's threats")

    await asyncio.sleep(2)

    # ============================================================================
    # Phase 4: David Investigates
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: David's Investigation")
    print("=" * 80)

    await client.create_memory(
        topic="David Park (HR Director)",
        content="""David Park noticed unusual behavior from Marcus and started investigating.
        He reviewed Marcus's calendar and found several unexplained meetings outside office.
        David ran a background check and discovered Marcus met with Alex Zhang multiple times.
        He also noticed Elena's behavior changed, becoming more secretive.
        David compiled a report with evidence of Marcus's meetings with competitors.
        He doesn't have concrete proof of Elena's involvement, only suspicions.
        David presented his findings to Sarah and recommended immediate action.
        He's monitoring the situation closely and preparing for potential resignations.""",
        tags=["person", "hr", "investigator", "discovered"],
    )
    print("✓ Evolved David to v2: Discovered Marcus's betrayal")
    print("  → David MONITORS Marcus and Elena")
    print("  → David AWARE_OF Marcus's betrayal")
    print("  → David suspects but doesn't know about Elena")

    await asyncio.sleep(2)

    # ============================================================================
    # Phase 5: Sarah Learns the Truth (Partial)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Sarah's Awakening (Partial Knowledge)")
    print("=" * 80)

    await client.create_memory(
        topic="Sarah Chen (CEO)",
        content="""Sarah Chen was shocked when David presented evidence of Marcus's betrayal.
        David showed her proof that Marcus has been meeting with Alex Zhang from NeuralWave.
        Sarah feels deeply hurt - she trusted Marcus completely and gave him autonomy.
        She now knows Marcus has been dishonest and planning to leave for a competitor.
        Sarah is preparing for Marcus's departure and planning damage control.
        She's started looking for a replacement CTO immediately.
        Sarah still trusts Elena and is counting on her to lead the team after Marcus leaves.
        Sarah plans to confide in Elena about the situation and ask her to step up.
        Sarah is unaware that Elena is also planning to leave with Marcus.""",
        tags=["person", "ceo", "betrayed", "partially_aware"],
    )
    print("✓ Evolved Sarah to v2: Knows about Marcus, not Elena")
    print("  → Sarah AWARE_OF Marcus v2's betrayal")
    print("  → Sarah still RELATED_TO Elena v1 (trusts her)")
    print("  → Sarah UNAWARE_OF Elena v2's involvement")
    print("  → This demonstrates ASYMMETRIC KNOWLEDGE!")

    await asyncio.sleep(2)

    # ============================================================================
    # Phase 6: Query and Analyze Relationships
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: Relationship Analysis")
    print("=" * 80)

    print("\n--- Sarah's Perspective (v2) ---")
    print("Sarah should have:")
    print("  • AWARE_OF Marcus v2 (knows about betrayal)")
    print("  • RELATED_TO Elena v1 (still trusts old Elena)")
    print("  • UNAWARE_OF Elena v2 (doesn't know she's also leaving)")
    print("  • MONITORS David v2 (relies on his investigation)")

    # Query: What does Sarah know about the situation?
    response = await client.query(
        query="What does Sarah know about Marcus and Elena's situation?",
        limit=5,
    )
    print("\n📊 Query: Sarah's knowledge about the situation")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print("\nSources:")
    for i, source in enumerate(response.sources, 1):
        print(f"  {i}. {source.topic} v{source.version}")
        print(f"     {source.content[:150]}...")

    print("\n--- Marcus's Perspective (v2) ---")
    print("Marcus should have:")
    print("  • INFLUENCED_BY Alex v1 (job offer)")
    print("  • RELATED_TO Elena v2 (both leaving together)")
    print("  • Historical: CAUSED_BY Sarah v1 (frustration led to decision)")

    response = await client.query(
        query="What is Marcus's current situation and plans?",
        limit=5,
    )
    print("\n📊 Query: Marcus's current plans")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print("\nSources:")
    for i, source in enumerate(response.sources, 1):
        print(f"  {i}. {source.topic} v{source.version}")
        print(f"     {source.content[:150]}...")

    print("\n--- Lisa's Dilemma (v2) ---")
    print("Lisa should have:")
    print("  • HIDDEN_FROM Sarah (not reporting what she knows)")
    print("  • INFLUENCED_BY Elena v2 (threatened into silence)")
    print("  • AWARE_OF Elena v2's plan")
    print("  • UNAWARE_OF David v2's investigation")

    response = await client.query(
        query="What does Lisa know and why hasn't she reported it?",
        limit=5,
    )
    print("\n📊 Query: Lisa's knowledge and dilemma")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print("\nSources:")
    for i, source in enumerate(response.sources, 1):
        print(f"  {i}. {source.topic} v{source.version}")
        print(f"     {source.content[:150]}...")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Complex Relationship Network")
    print("=" * 80)

    print("\n🔑 Key Relationship Concepts Demonstrated:")
    print()
    print("1. VERSION-BOUND RELATIONSHIPS (historical facts):")
    print("   • Marcus v2 INFLUENCED_BY Alex v1 (job offer at specific time)")
    print("   • Lisa v2 INFLUENCED_BY Elena v2 (threat at specific moment)")
    print("   • Sarah v2 CAUSED_BY Marcus v1's actions (frustration)")
    print()
    print("2. ASYMMETRIC KNOWLEDGE (A knows X, B doesn't):")
    print("   • Sarah v2 AWARE_OF Marcus v2's betrayal")
    print("   • Sarah v2 UNAWARE_OF Elena v2's involvement")
    print("   • Sarah v2 still RELATED_TO Elena v1 (trusts old version)")
    print("   → Sarah has TWO relationships to Elena: trusts v1, unaware of v2!")
    print()
    print("3. HIDDEN INFORMATION:")
    print("   • Lisa v2 HIDDEN_FROM Sarah v2 (knows but won't tell)")
    print("   • Marcus v2 was HIDDEN_FROM Sarah v1 (secret meetings)")
    print()
    print("4. DYNAMIC MONITORING:")
    print("   • David v2 MONITORS Marcus v2 and Elena v2")
    print("   • Sarah v2 DEPENDS_ON David v2 for intelligence")
    print()
    print("5. MULTIPLE VERSIONS OF SAME ENTITY:")
    print("   • Sarah relates to both Elena v1 (trusted friend) and Elena v2 (unknown betrayer)")
    print("   • This models realistic asymmetric knowledge in social networks!")
    print()

    print("\n" + "=" * 80)
    print("This example demonstrates TracingRAG's ability to model:")
    print("  ✓ Complex social dynamics with secrets and betrayal")
    print("  ✓ Asymmetric information (what each person knows)")
    print("  ✓ Historical events that don't change (version-bound)")
    print("  ✓ Dynamic relationships that update to latest state")
    print("  ✓ Multiple simultaneous relationships to different versions")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
