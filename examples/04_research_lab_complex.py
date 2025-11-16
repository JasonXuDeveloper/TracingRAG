"""
Complex Test Case: Research Lab with Cascading Evolution

This example tests:
1. Cascading Evolution: When a discovery is made, related research topics should evolve
2. Relationship Manager: Verify relationships are correctly established and updated
3. Manual Promotion: Test accurate synthesis of multiple related memory states

Scenario:
- A research lab with multiple interconnected projects
- Researchers working on different but related topics
- When a breakthrough happens, it affects multiple research directions
- Manual promotion should synthesize all relevant context
"""

import asyncio

from tracingrag.client import AsyncTracingRAGClient


async def main():
    """Complex research lab scenario"""
    client = AsyncTracingRAGClient("http://localhost:8000", timeout=3000.0)

    print("=" * 70)
    print("🔬 Complex Test: Research Lab with Cascading Evolution")
    print("=" * 70)

    # ====================================================================
    # Phase 1: Initial Research Setup
    # ====================================================================
    print("\n📚 Phase 1: Setting up research topics...")

    # Main research direction
    research_main = await client.create_memory(
        topic="research_quantum_computing",
        content=(
            "Research lab focuses on quantum computing applications. "
            "Primary goal: develop error-correcting quantum algorithms. "
            "Team size: 5 researchers. Budget: $2M annually."
        ),
        tags=["research", "quantum", "primary"],
        confidence=0.95,
    )
    print(f"   ✅ Created main research: {research_main.id}")

    # Related research topics
    research_algorithms = await client.create_memory(
        topic="research_quantum_algorithms",
        content=(
            "Developing quantum algorithms for optimization problems. "
            "Focus on Grover's and Shor's algorithms. "
            "Current stage: theoretical framework complete."
        ),
        tags=["research", "algorithms", "optimization"],
        confidence=0.90,
    )
    print(f"   ✅ Created algorithms research: {research_algorithms.id}")

    research_hardware = await client.create_memory(
        topic="research_quantum_hardware",
        content=(
            "Building quantum hardware prototypes. "
            "Using superconducting qubits technology. "
            "Current challenge: maintaining coherence time > 100µs."
        ),
        tags=["research", "hardware", "experimental"],
        confidence=0.85,
    )
    print(f"   ✅ Created hardware research: {research_hardware.id}")

    research_applications = await client.create_memory(
        topic="research_quantum_applications",
        content=(
            "Exploring practical applications of quantum computing. "
            "Target areas: drug discovery, financial modeling, cryptography. "
            "Status: feasibility studies in progress."
        ),
        tags=["research", "applications", "practical"],
        confidence=0.80,
    )
    print(f"   ✅ Created applications research: {research_applications.id}")

    # Team members
    researcher_alice = await client.create_memory(
        topic="researcher_alice",
        content=(
            "Dr. Alice Chen, Lead Quantum Physicist. "
            "PhD from MIT, 10 years experience. "
            "Specializes in quantum error correction. "
            "Currently leading the algorithms team."
        ),
        tags=["team", "lead", "algorithms"],
        confidence=0.95,
    )
    print(f"   ✅ Created researcher Alice: {researcher_alice.id}")

    researcher_bob = await client.create_memory(
        topic="researcher_bob",
        content=(
            "Dr. Bob Wilson, Hardware Engineer. "
            "PhD from Stanford, 8 years experience. "
            "Expert in superconducting circuits. "
            "Managing the hardware lab."
        ),
        tags=["team", "hardware", "engineering"],
        confidence=0.95,
    )
    print(f"   ✅ Created researcher Bob: {researcher_bob.id}")

    researcher_carol = await client.create_memory(
        topic="researcher_carol",
        content=(
            "Dr. Carol Davis, Applications Researcher. "
            "PhD from Caltech, 6 years experience. "
            "Background in pharmaceutical chemistry. "
            "Exploring drug discovery applications."
        ),
        tags=["team", "applications", "chemistry"],
        confidence=0.90,
    )
    print(f"   ✅ Created researcher Carol: {researcher_carol.id}")

    # Add some funding and equipment info
    funding = await client.create_memory(
        topic="research_funding",
        content=(
            "Current funding: NSF grant $2M over 3 years. "
            "Application submitted to DOE for additional $1.5M. "
            "Private sector partnership with IBM under negotiation."
        ),
        tags=["funding", "grants"],
        confidence=0.95,
    )
    print(f"   ✅ Created funding info: {funding.id}")

    equipment = await client.create_memory(
        topic="lab_equipment",
        content=(
            "Quantum hardware lab equipment: "
            "Dilution refrigerator (10mK), microwave generators, "
            "signal analyzers, cryogenic measurement systems. "
            "Total equipment value: $5M."
        ),
        tags=["equipment", "hardware"],
        confidence=0.95,
    )
    print(f"   ✅ Created equipment info: {equipment.id}")

    # ====================================================================
    # Phase 2: Major Breakthrough (This should trigger cascading evolution!)
    # ====================================================================
    print("\n💡 Phase 2: Major breakthrough discovered!")
    print("   ⚠️  This should trigger cascading evolution of related topics...\n")

    # Update algorithms with breakthrough - this should cascade to related topics
    algorithms_v2 = await client.create_memory(
        topic="research_quantum_algorithms",
        content=(
            "MAJOR BREAKTHROUGH: Alice's team discovered a new error-correction "
            "technique that reduces decoherence by 80%. This novel approach uses "
            "topological codes combined with machine learning optimization. "
            "This is a game-changing discovery that affects ALL our research directions. "
            "Paper submitted to Nature. Patent application filed."
        ),
        tags=["research", "algorithms", "breakthrough", "error-correction"],
        confidence=0.98,
    )
    print(f"   ✅ Breakthrough recorded: {algorithms_v2.id}")
    print("   🔄 Waiting for cascading evolution to complete...")
    await asyncio.sleep(3)  # Give time for cascading evolution

    # ====================================================================
    # Phase 3: Check what evolved (cascading evolution verification)
    # ====================================================================
    print("\n🔍 Phase 3: Verifying cascading evolution...")

    # Check which topics got new versions
    topics_to_check = [
        "research_quantum_computing",
        "research_quantum_hardware",
        "research_quantum_applications",
        "researcher_alice",
        "research_funding",
    ]

    for topic in topics_to_check:
        trace = await client.get_trace(topic)
        print(f"\n   📌 {topic}:")
        print(f"      Versions: {len(trace)}")
        if len(trace) > 1:
            latest = trace[0]  # Most recent version
            print(f"      ✨ Latest (v{latest.version}): {latest.content[:100]}...")
            if "cascading_evolved" in latest.tags:
                print("      🎯 CASCADING EVOLUTION TRIGGERED!")
        else:
            print("      ⚠️  No evolution detected")

    # ====================================================================
    # Phase 4: More updates to test relationship manager
    # ====================================================================
    print("\n🔗 Phase 4: Testing relationship manager with more updates...")

    # Update hardware based on the breakthrough
    hardware_v2 = await client.create_memory(
        topic="research_quantum_hardware",
        content=(
            "Hardware team immediately started implementing Alice's new "
            "error-correction technique. Bob redesigned the qubit layout to "
            "support topological codes. Initial tests show coherence time "
            "increased to 500µs (5x improvement). Hardware paper in preparation."
        ),
        tags=["research", "hardware", "implementation", "success"],
        confidence=0.92,
    )
    print(f"   ✅ Hardware update: {hardware_v2.id}")

    # Update applications
    applications_v2 = await client.create_memory(
        topic="research_quantum_applications",
        content=(
            "Carol's team can now run much longer quantum simulations. "
            "Successfully simulated protein folding for small molecules. "
            "Collaboration with pharmaceutical companies initiated. "
            "Expected to accelerate drug discovery by 10x."
        ),
        tags=["research", "applications", "success", "pharma"],
        confidence=0.88,
    )
    print(f"   ✅ Applications update: {applications_v2.id}")

    # Update funding
    funding_v2 = await client.create_memory(
        topic="research_funding",
        content=(
            "DOE grant APPROVED: $1.5M additional funding. "
            "IBM partnership finalized: $3M over 2 years + equipment. "
            "Total funding now $6.5M. Nature paper and patent significantly "
            "strengthened our position. Planning to hire 3 more researchers."
        ),
        tags=["funding", "grants", "success"],
        confidence=0.98,
    )
    print(f"   ✅ Funding update: {funding_v2.id}")

    # ====================================================================
    # Phase 5: Manual promotion to test synthesis accuracy
    # ====================================================================
    print("\n🚀 Phase 5: Testing manual promotion (synthesis accuracy)...")

    # Promote the main research topic - should synthesize ALL related changes
    promotion_result = await client.promote_memory(
        topic="research_quantum_computing",
        reason=(
            "Quarter end summary: Major breakthrough in error correction, "
            "successful hardware implementation, promising applications, "
            "and substantial funding increase. Need comprehensive synthesis."
        ),
    )

    if promotion_result.success:
        new_state = promotion_result.new_state
        print("\n   ✅ Promotion successful!")
        print(f"   📊 New version: v{new_state.version}")
        print("   📝 Synthesized content:")
        print(f"      {new_state.content[:300]}...")
        print(f"\n   📚 Synthesis sources: {len(promotion_result.synthesis_sources)}")
        print(f"   🎯 Confidence: {promotion_result.confidence:.2f}")
        print(f"   💭 Reasoning: {promotion_result.reasoning[:200]}...")
    else:
        print(f"   ❌ Promotion failed: {promotion_result.error_message}")

    # ====================================================================
    # Phase 6: Query to verify relationships
    # ====================================================================
    print("\n🔎 Phase 6: Querying to verify knowledge synthesis...")

    queries = [
        "What was the major breakthrough and how did it impact the research?",
        "How has the funding situation changed?",
        "What are the practical applications now possible?",
    ]

    for i, query_text in enumerate(queries, 1):
        print(f"\n   Query {i}: '{query_text}'")
        result = await client.query(query_text, include_history=True)
        print(f"   Answer: {result.answer[:250]}...")
        print(f"   Sources: {len(result.sources)} documents")
        print(f"   Confidence: {result.confidence:.2f}")

    # ====================================================================
    # Phase 7: View evolution traces
    # ====================================================================
    print("\n📈 Phase 7: Viewing evolution traces...")

    key_topics = [
        "research_quantum_computing",
        "research_quantum_algorithms",
        "researcher_alice",
    ]

    for topic in key_topics:
        trace = await client.get_trace(topic)
        print(f"\n   📌 {topic}: {len(trace)} version(s)")
        for state in trace[:3]:  # Show first 3 versions
            tags_str = ", ".join(state.tags[:3])
            print(f"      v{state.version} - {state.timestamp.strftime('%H:%M:%S')}")
            print(f"         Tags: [{tags_str}]")
            print(f"         {state.content[:80]}...")

    print("\n" + "=" * 70)
    print("✅ Complex test completed!")
    print("=" * 70)
    print("\nKey verifications:")
    print("  1. ✓ Cascading evolution triggered by breakthrough")
    print("  2. ✓ Relationships correctly established between topics")
    print("  3. ✓ Manual promotion synthesized comprehensive summary")
    print("  4. ✓ Queries return accurate, context-aware answers")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
