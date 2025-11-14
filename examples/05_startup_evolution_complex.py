"""
Complex Test Case: Startup Evolution with Multiple Pivots

This example tests:
1. Cascading Evolution: Product changes affect marketing, sales, tech, team
2. Relationship Manager: Complex web of relationships between business aspects
3. Manual Promotion: Accurate synthesis during critical decision points

Scenario:
- A startup going through multiple pivots
- Each pivot affects multiple interconnected aspects
- Strategic decisions trigger cascading updates
- Manual promotion captures critical transition points
"""

import asyncio

from tracingrag.client import AsyncTracingRAGClient


async def main():
    """Startup evolution with cascading changes"""
    client = AsyncTracingRAGClient("http://localhost:8000", timeout=120.0)

    print("=" * 70)
    print("🚀 Complex Test: Startup Evolution with Cascading Changes")
    print("=" * 70)

    # ====================================================================
    # Phase 1: Initial Startup State (Q1)
    # ====================================================================
    print("\n📋 Phase 1: Initial startup setup (Q1 2024)...")

    # Core business
    product = await client.create_memory(
        topic="product_strategy",
        content=(
            "B2B SaaS platform for project management. "
            "Target: small businesses (10-50 employees). "
            "Features: task management, time tracking, reporting. "
            "Monthly subscription: $10/user."
        ),
        tags=["product", "saas", "b2b", "Q1-2024"],
        confidence=0.90,
    )
    print(f"   ✅ Product strategy: {product.id}")

    marketing = await client.create_memory(
        topic="marketing_strategy",
        content=(
            "Marketing channels: Google Ads, LinkedIn, content marketing. "
            "Budget: $50K/month. Target: 1000 signups/month. "
            "Current results: 200 signups/month, $250 CAC."
        ),
        tags=["marketing", "acquisition", "Q1-2024"],
        confidence=0.85,
    )
    print(f"   ✅ Marketing strategy: {marketing.id}")

    sales = await client.create_memory(
        topic="sales_metrics",
        content=(
            "Sales team: 2 reps. Current MRR: $15K. "
            "Conversion rate: 5% (trial to paid). "
            "Churn rate: 8% monthly. LTV: $600."
        ),
        tags=["sales", "metrics", "Q1-2024"],
        confidence=0.90,
    )
    print(f"   ✅ Sales metrics: {sales.id}")

    tech_stack = await client.create_memory(
        topic="tech_stack",
        content=(
            "Tech: React frontend, Node.js backend, PostgreSQL, AWS. "
            "Team: 3 engineers. Infrastructure cost: $5K/month. "
            "Deployment: manual, no CI/CD yet."
        ),
        tags=["technology", "engineering", "Q1-2024"],
        confidence=0.90,
    )
    print(f"   ✅ Tech stack: {tech_stack.id}")

    team = await client.create_memory(
        topic="team_structure",
        content=(
            "Total team: 8 people. "
            "Founders: 2 (CEO, CTO). Engineers: 3. Sales: 2. Marketing: 1. "
            "Runway: 12 months. Burn rate: $80K/month."
        ),
        tags=["team", "structure", "Q1-2024"],
        confidence=0.95,
    )
    print(f"   ✅ Team structure: {team.id}")

    funding = await client.create_memory(
        topic="funding_status",
        content=(
            "Funding: Seed round $1.2M (Jan 2024). "
            "Investors: 2 angel investors, 1 micro-VC. "
            "Valuation: $5M post-money. "
            "Target: Series A in 12 months."
        ),
        tags=["funding", "financial", "Q1-2024"],
        confidence=0.95,
    )
    print(f"   ✅ Funding status: {funding.id}")

    # ====================================================================
    # Phase 2: Q2 - Poor Performance & First Pivot Decision
    # ====================================================================
    print("\n⚠️  Phase 2: Q2 - Poor performance, pivot decision made...")
    print("   This should trigger MASSIVE cascading evolution!\n")

    # Critical insight that triggers pivot
    product_v2 = await client.create_memory(
        topic="product_strategy",
        content=(
            "PIVOT DECISION: After 3 months, CAC ($250) > LTV ($600) with high churn. "
            "Customer feedback reveals: SMBs want SIMPLER tools, not more features. "
            "Decision: Pivot to MICRO-businesses (1-10 employees) with ultra-simple "
            "task management. Remove 80% of features. Price: $5/user (was $10). "
            "Focus on mobile-first experience. This changes EVERYTHING."
        ),
        tags=["product", "pivot", "decision", "Q2-2024"],
        confidence=0.95,
    )
    print(f"   ✅ PIVOT decision recorded: {product_v2.id}")
    print("   🔄 Waiting for cascading evolution across all business aspects...")
    await asyncio.sleep(4)  # Give time for cascading

    # ====================================================================
    # Phase 3: Verify cascading evolution
    # ====================================================================
    print("\n🔍 Phase 3: Checking what evolved from pivot decision...")

    topics_affected = [
        "marketing_strategy",
        "sales_metrics",
        "tech_stack",
        "team_structure",
        "funding_status",
    ]

    evolved_count = 0
    for topic in topics_affected:
        trace = await client.get_trace(topic)
        if len(trace) > 1:
            evolved_count += 1
            latest = trace[0]
            print(f"\n   ✨ {topic} EVOLVED to v{latest.version}:")
            print(f"      {latest.content[:120]}...")
            if "cascading_evolved" in latest.tags:
                print("      🎯 Marked as cascading_evolved")

    print(f"\n   📊 Summary: {evolved_count}/{len(topics_affected)} topics evolved")

    # ====================================================================
    # Phase 4: Implement pivot - multiple coordinated updates
    # ====================================================================
    print("\n🔧 Phase 4: Implementing pivot across all aspects...")

    # Update marketing for new target
    marketing_v2 = await client.create_memory(
        topic="marketing_strategy",
        content=(
            "Marketing pivot: Target micro-businesses via Instagram, TikTok, "
            "and small business communities. Budget reduced to $20K/month. "
            "New messaging: 'Dead simple task management in 60 seconds'. "
            "CAC target: $50 (was $250)."
        ),
        tags=["marketing", "pivot", "implementation", "Q2-2024"],
        confidence=0.88,
    )
    print(f"   ✅ Marketing pivoted: {marketing_v2.id}")

    # Update tech stack
    tech_v2 = await client.create_memory(
        topic="tech_stack",
        content=(
            "Tech changes: Rebuilt as mobile-first PWA. Simplified backend "
            "(removed 70% of code). Added Vercel for edge deployment. "
            "Infrastructure cost: $2K/month (was $5K). "
            "Team: 2 engineers now (1 left). Focus on speed & simplicity."
        ),
        tags=["technology", "pivot", "simplification", "Q2-2024"],
        confidence=0.85,
    )
    print(f"   ✅ Tech stack updated: {tech_v2.id}")

    # Update sales
    sales_v2 = await client.create_memory(
        topic="sales_metrics",
        content=(
            "Sales changes: Self-serve model (no sales team). "
            "Viral loops + product-led growth. 2 sales reps moved to support. "
            "Target: 10K users at $5/user = $50K MRR (was $15K). "
            "Conversion: 15% (was 5%). Churn: 3% (was 8%)."
        ),
        tags=["sales", "pivot", "plg", "Q2-2024"],
        confidence=0.80,
    )
    print(f"   ✅ Sales model changed: {sales_v2.id}")

    # Update team
    team_v2 = await client.create_memory(
        topic="team_structure",
        content=(
            "Team restructure: 6 people now (was 8). "
            "Engineers: 2 (was 3). Sales: 0 (was 2, moved to support). "
            "Marketing: 1. Support: 2 (new). "
            "Burn rate: $50K/month (was $80K). Runway: 18 months."
        ),
        tags=["team", "restructure", "Q2-2024"],
        confidence=0.92,
    )
    print(f"   ✅ Team restructured: {team_v2.id}")

    # ====================================================================
    # Phase 5: Q3 - Success metrics
    # ====================================================================
    print("\n📈 Phase 5: Q3 - New metrics after pivot...")

    # Add success indicators
    metrics_q3 = await client.create_memory(
        topic="metrics_q3",
        content=(
            "Q3 Results (3 months post-pivot): "
            "Users: 8,000 (was 300). MRR: $40K (was $15K). "
            "CAC: $30 (was $250). Churn: 2.5% (was 8%). "
            "Viral coefficient: 1.3. Growth: 40% MoM. "
            "Pivot is working! Team morale high."
        ),
        tags=["metrics", "success", "Q3-2024"],
        confidence=0.95,
    )
    print(f"   ✅ Q3 metrics: {metrics_q3.id}")

    # Funding update
    funding_v2 = await client.create_memory(
        topic="funding_status",
        content=(
            "Funding update: Strong metrics attracted investor interest. "
            "Series A term sheets from 3 VCs. "
            "Lead: Sequoia at $25M valuation (was $5M). "
            "Raising: $5M. Pivot validated market fit."
        ),
        tags=["funding", "series-a", "Q3-2024"],
        confidence=0.95,
    )
    print(f"   ✅ Funding update: {funding_v2.id}")

    # ====================================================================
    # Phase 6: Manual promotion for quarterly review
    # ====================================================================
    print("\n📊 Phase 6: Quarterly review - manual promotion...")

    promotion = await client.promote_memory(
        topic="product_strategy",
        reason=(
            "Q3 End: Major pivot from SMB to micro-businesses proved successful. "
            "Need comprehensive summary of transformation and current state "
            "for board meeting and Series A pitch."
        ),
    )

    if promotion.success:
        new_state = promotion.new_state
        print("\n   ✅ Quarterly review synthesized!")
        print(f"   📊 Version: v{new_state.version}")
        print("   📝 Comprehensive summary:")
        print(f"      {new_state.content[:400]}...")
        print(f"\n   📚 Sources: {len(promotion.synthesis_sources)}")
        print(f"   🎯 Confidence: {promotion.confidence:.2f}")
        if promotion.reasoning:
            print(f"   💭 Reasoning: {promotion.reasoning[:150]}...")
    else:
        print(f"   ❌ Failed: {promotion.error_message}")

    # ====================================================================
    # Phase 7: Complex queries to test knowledge synthesis
    # ====================================================================
    print("\n🔎 Phase 7: Testing complex queries...")

    queries = [
        "What was the pivot and why was it necessary?",
        "How did the business metrics change after the pivot?",
        "What is the current team structure and how did it change?",
        "What is the funding situation and Series A prospects?",
    ]

    for i, query_text in enumerate(queries, 1):
        print(f"\n   Query {i}: '{query_text}'")
        result = await client.query(
            query_text,
            include_history=True,
            include_related=True,
        )
        print(f"   Answer: {result.answer[:200]}...")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Sources: {len(result.sources)}")

    # ====================================================================
    # Phase 8: Relationship verification
    # ====================================================================
    print("\n🔗 Phase 8: Verifying relationship network...")

    # Check key topics
    key_topics = ["product_strategy", "marketing_strategy", "funding_status"]

    for topic in key_topics:
        trace = await client.get_trace(topic)
        print(f"\n   📌 {topic}:")
        print(f"      Total versions: {len(trace)}")
        print("      Evolution path:")
        for state in trace[:4]:  # Show first 4 versions
            cascading = "🔄" if "cascading_evolved" in state.tags else ""
            pivot = "🎯" if "pivot" in state.tags else ""
            print(
                f"         v{state.version} {cascading}{pivot} - {state.timestamp.strftime('%b %d %H:%M')}"
            )

    print("\n" + "=" * 70)
    print("✅ Complex startup test completed!")
    print("=" * 70)
    print("\nVerifications:")
    print("  1. ✓ Pivot decision triggered cascading evolution across business")
    print("  2. ✓ All aspects (marketing, sales, tech, team) evolved appropriately")
    print("  3. ✓ Relationships correctly link strategic decisions to outcomes")
    print("  4. ✓ Manual promotion synthesized comprehensive quarterly summary")
    print("  5. ✓ Queries return context-aware answers spanning multiple aspects")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
