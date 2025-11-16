#!/usr/bin/env python3
"""
Example 1: Project Memory Tracking

This example demonstrates how to use TracingRAG to track the evolution
of a software project, including:
- Creating memory states for project milestones
- Tracking design decisions and their reasoning
- Querying project history and evolution
- Using memory promotion to synthesize project summaries

Use Case: Knowledge management for software development teams
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracingrag.client import AsyncTracingRAGClient


async def main():
    """Run project memory tracking example"""
    print("=" * 70)
    print("Example 1: Project Memory Tracking")
    print("=" * 70)
    print()

    # Initialize client
    print("📡 Connecting to TracingRAG API...")
    async with AsyncTracingRAGClient("http://localhost:8000", timeout=3000.0) as client:
        # Check health
        health = await client.health()
        if health["status"] != "healthy":
            print("❌ API is not healthy. Please start the API server.")
            return
        print("✅ Connected to TracingRAG API\n")

        # ====================================================================
        # Phase 1: Initial Project Design
        # ====================================================================
        print("📝 Phase 1: Creating initial project design...")

        project_v1 = await client.create_memory(
            topic="project_microservices_migration",
            content=(
                "Decided to migrate from monolithic architecture to microservices. "
                "Plan to split into 5 services: User Service, Product Service, "
                "Order Service, Payment Service, and Notification Service. "
                "Using REST APIs for inter-service communication."
            ),
            tags=["architecture", "design", "microservices"],
            confidence=0.9,
        )
        print(f"   ✅ Created project v1: {project_v1.id}")

        # Create related technical decision
        tech_decision_v1 = await client.create_memory(
            topic="tech_stack_decision",
            content=(
                "Technology stack selected: Python (FastAPI) for services, "
                "PostgreSQL for databases, Redis for caching, Docker for containers, "
                "Kubernetes for orchestration."
            ),
            tags=["technology", "decision"],
            confidence=0.95,
        )
        print(f"   ✅ Created tech stack v1: {tech_decision_v1.id}\n")

        # ====================================================================
        # Phase 2: Discovery of Issues
        # ====================================================================
        print("⚠️  Phase 2: Discovering challenges...")

        challenge_v1 = await client.create_memory(
            topic="microservices_complexity_issue",
            content=(
                "Team is struggling with microservices complexity. "
                "Inter-service communication overhead is high. "
                "Debugging distributed systems is difficult. "
                "Team size (3 developers) may be too small to manage 5 services effectively."
            ),
            tags=["challenge", "complexity"],
            confidence=0.85,
        )
        print(f"   ✅ Documented challenge: {challenge_v1.id}\n")

        # ====================================================================
        # Phase 3: Pivot Decision
        # ====================================================================
        print("🔄 Phase 3: Making a pivot decision...")

        # Promote the project memory with new information
        project_promotion = await client.promote_memory(
            topic="project_microservices_migration",
            reason=(
                "Pivoting strategy based on team feedback and complexity challenges. "
                "Moving to modular monolith approach instead."
            ),
        )

        project_v2 = project_promotion.new_state
        print(f"   ✅ Promoted project to v{project_v2.version}")
        print(f"   Content: {project_v2.content[:100]}...")
        if project_promotion.synthesis_sources:
            print(f"   📚 Synthesized from {len(project_promotion.synthesis_sources)} sources\n")

        # Update tech stack
        tech_decision_v2 = await client.create_memory(
            topic="tech_stack_decision",
            content=(
                "Updated technology stack: Python (FastAPI) with modular structure, "
                "PostgreSQL with schema separation per module, Redis for caching, "
                "Docker for deployment, simplified orchestration with Docker Compose."
            ),
            parent_state_id=tech_decision_v1.id,
            tags=["technology", "decision", "updated"],
            confidence=0.95,
        )
        print(f"   ✅ Updated tech stack to v{tech_decision_v2.version}: {tech_decision_v2.id}\n")

        # ====================================================================
        # Phase 4: Query Project Knowledge
        # ====================================================================
        print("🔍 Phase 4: Querying project knowledge...")

        # Query 1: Simple query with agent mode for better retrieval
        print("\n   Query 1: 'What is our current architecture approach?'")
        result1 = await client.query(
            "What is our current architecture approach? Describe the technology stack and design decisions.",
            include_related=True,
        )
        if result1.answer:
            print(f"   Answer: {result1.answer[:250]}...")
            print(f"   Sources: {len(result1.sources)} documents")
            if result1.reasoning:
                print(f"   Reasoning: {result1.reasoning[:150]}...")

        # Query 2: Historical query (uses iterative agent by default)
        print("\n   Query 2: 'Why did we change from microservices to modular monolith?'")
        result2 = await client.query(
            "Why did we change from microservices to modular monolith? What were the challenges?",
            include_history=True,
            max_rounds=5,  # Agent mode is default (max_rounds > 0)
        )
        if result2.answer:
            print(f"   Answer: {result2.answer[:250]}...")
            if result2.reasoning:
                print(f"   Reasoning: {result2.reasoning[:150]}...")

        # Query 3: Technology-specific query
        print("\n   Query 3: 'What database and caching solutions are we using?'")
        result3 = await client.query(
            "What database and caching solutions are we using in our current tech stack?",
            include_related=True,
        )
        if result3.answer:
            print(f"   Answer: {result3.answer[:250]}...")
            print(f"   Sources: {len(result3.sources)} documents")

        # ====================================================================
        # Phase 5: View Project Evolution
        # ====================================================================
        print("\n📈 Phase 5: Viewing project evolution...")

        trace = await client.get_trace("project_microservices_migration")
        print(f"\n   Project has {len(trace)} versions:")
        for state in trace:
            print(f"      v{state.version} ({state.timestamp.strftime('%Y-%m-%d %H:%M')})")
            print(f"         {state.content[:80]}...")
            print()

        # ====================================================================
        # Phase 6: Check Promotion Candidates
        # ====================================================================
        print("🎯 Phase 6: Checking promotion candidates...")

        candidates = await client.get_promotion_candidates(limit=5)
        if candidates:
            print(f"\n   Found {len(candidates)} topics that could be promoted:")
            for candidate in candidates:
                print(f"      - {candidate.topic}")
                print(f"        Priority: {candidate.priority_score:.1f}")
                print(
                    f"        Versions since consolidation: {candidate.versions_since_consolidation}"
                )
                print()
        else:
            print("   No promotion candidates found yet (not enough data)")

    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("   - TracingRAG tracks complete project evolution")
    print("   - Each decision and change is preserved with context")
    print("   - Iterative agent mode is enabled by default (max_rounds=5)")
    print("   - Agent combines semantic search, graph relationships, and history")
    print("   - More specific queries with keywords yield better results")
    print("   - Memory promotion synthesizes information intelligently")
    print()
    print("Query Best Practices:")
    print("   ✓ Default mode uses intelligent iterative agent (max_rounds=5)")
    print("   ✓ Increase max_rounds for complex queries requiring deep analysis")
    print("   ✓ Set max_rounds=0 for fast simple queries (no agent)")
    print("   ✓ Use include_history=True for temporal questions")
    print("   ✓ Use include_related=True to leverage graph relationships")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("   1. TracingRAG API is running: uvicorn tracingrag.api.main:app --reload")
        print("   2. All services are healthy: poetry run python scripts/verify_setup.py")
        sys.exit(1)
