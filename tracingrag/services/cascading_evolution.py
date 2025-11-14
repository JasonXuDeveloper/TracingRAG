"""Cascading Evolution Manager for intelligent topic evolution

This module handles cascading updates when a new memory is created:
1. Find semantically related topics using embedding search
2. Deduplicate to keep only latest version of each topic
3. Use LLM to decide which topics should evolve based on new memory
4. Evolve those topics (create new versions)
5. Then RelationshipManager can update edges with the evolved states
"""

import json
from typing import Any
from uuid import UUID

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.models import MemoryStateDB


class CascadingEvolutionManager:
    """Manages cascading evolution of related topics when new memories are created"""

    def __init__(self):
        self.llm_client = get_llm_client()

    async def trigger_related_evolutions(
        self,
        new_state: MemoryStateDB,
        similarity_threshold: float = 0.4,
        max_evolutions: int = 10,
    ) -> dict[str, Any]:
        """Trigger evolution of related topics based on new memory

        Strategy:
        1. Embedding search for semantically related states
        2. Deduplicate to latest version per topic
        3. LLM decides which topics should evolve
        4. Evolve those topics with synthesized content
        5. Return statistics

        Args:
            new_state: The newly created memory state
            similarity_threshold: Minimum similarity for candidates (higher = more selective)
            max_evolutions: Maximum number of topics to evolve

        Returns:
            Dict with evolution statistics: {evolved_topics, skipped_topics, total_candidates}
        """
        from tracingrag.services.retrieval import RetrievalService

        stats = {
            "evolved_topics": [],  # Topics that were evolved
            "skipped_topics": [],  # Topics that were skipped (not relevant enough)
            "total_candidates": 0,
        }

        print(
            f"[CascadingEvolution] Analyzing impact of new memory: {new_state.topic} v{new_state.version}"
        )

        # Step 1: Get total memory states count for full database search
        from tracingrag.storage.database import get_session

        async with get_session() as db_session:
            from sqlalchemy import func, select

            from tracingrag.storage.models import MemoryStateDB

            result = await db_session.execute(select(func.count(MemoryStateDB.id)))
            total_states = result.scalar() or 0

        search_limit = max(total_states, 1000)  # Use actual count, minimum 1000
        print(f"   Database has {total_states} states, using limit={search_limit} for full search")

        # Step 2: Find semantically related states via embedding search (full database scan)
        retrieval_service = RetrievalService()
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=search_limit,  # Dynamic limit based on actual database size
            score_threshold=similarity_threshold,
            latest_only=False,  # Get all versions for deduplication
        )

        # Filter out self and same topic
        candidate_states_raw = [
            s
            for s in similar_states
            if s.state.id != new_state.id and s.state.topic != new_state.topic
        ]

        print(f"   Found {len(candidate_states_raw)} semantically related states")

        if not candidate_states_raw:
            print("   No related states found, skipping cascading evolution")
            return stats

        # Step 2: Deduplicate - keep only latest version of each topic
        topic_to_latest = {}
        for result in candidate_states_raw:
            topic = result.state.topic
            if topic not in topic_to_latest:
                topic_to_latest[topic] = result
            else:
                # Keep the one with higher version
                if result.state.version > topic_to_latest[topic].state.version:
                    topic_to_latest[topic] = result

        candidate_topics = list(topic_to_latest.values())
        stats["total_candidates"] = len(candidate_topics)

        print(f"   After deduplication: {len(candidate_topics)} unique topics to analyze")

        # Step 3: Use LLM to decide which topics should evolve
        evolution_decisions = await self._llm_evolution_analysis(
            new_state=new_state,
            candidate_topics=candidate_topics[:max_evolutions],  # Limit to prevent overload
        )

        # Step 4: Evolve the selected topics
        from tracingrag.services.memory import MemoryService

        memory_service = MemoryService()

        for decision in evolution_decisions:
            if not decision["should_evolve"]:
                stats["skipped_topics"].append(
                    {
                        "topic": decision["topic"],
                        "reason": decision["reasoning"],
                    }
                )
                continue

            try:
                # Evolve this topic by creating a new version
                evolved_state = await memory_service.create_memory_state(
                    topic=decision["topic"],
                    content=decision["evolved_content"],
                    parent_state_id=decision["current_state_id"],
                    metadata={
                        "cascading_evolution": True,
                        "triggered_by_topic": new_state.topic,
                        "triggered_by_state_id": str(new_state.id),
                        "evolution_reason": decision["reasoning"],
                    },
                    tags=[*decision.get("current_tags", []), "cascading_evolved"],
                    confidence=decision.get("confidence", 0.8),
                )

                stats["evolved_topics"].append(
                    {
                        "topic": decision["topic"],
                        "old_version": decision["current_version"],
                        "new_version": evolved_state.version,
                        "new_state_id": str(evolved_state.id),
                        "reasoning": decision["reasoning"],
                    }
                )

                print(
                    f"   ✓ Evolved: {decision['topic']} v{decision['current_version']} → v{evolved_state.version}"
                )

            except Exception as e:
                print(f"   ✗ Failed to evolve {decision['topic']}: {e}")
                stats["skipped_topics"].append(
                    {
                        "topic": decision["topic"],
                        "reason": f"Evolution failed: {str(e)}",
                    }
                )

        print(
            f"[CascadingEvolution] Summary: {len(stats['evolved_topics'])} evolved, "
            f"{len(stats['skipped_topics'])} skipped"
        )

        return stats

    async def _llm_evolution_analysis(
        self,
        new_state: MemoryStateDB,
        candidate_topics: list[Any],
    ) -> list[dict[str, Any]]:
        """Use LLM to analyze which topics should evolve based on new memory

        Args:
            new_state: The new memory state
            candidate_topics: List of QueryResult objects (latest version per topic)

        Returns:
            List of evolution decisions with evolved content
        """
        # Build prompt
        new_memory_desc = (
            f"**NEW MEMORY ADDED**\n"
            f"Topic: {new_state.topic}\n"
            f"Content: {new_state.content}\n\n"
        )

        candidate_desc = []
        for idx, result in enumerate(candidate_topics):
            candidate_desc.append(
                f"T{idx + 1}. {result.state.topic} v{result.state.version} (similarity: {result.score:.2f})\n"
                f"    Current content: {result.state.content[:300]}"
            )

        candidates_text = "\n\n".join(candidate_desc)

        prompt = f"""{new_memory_desc}

**RELATED TOPICS** (semantically related, need evolution analysis):
{candidates_text}

**TASK**: For EACH related topic (T1, T2, ...), decide if it should evolve based on the new memory.

**DECISION CRITERIA**:
1. **Should Evolve** if:
   - The new memory provides significant new information relevant to this topic
   - The topic's current content would benefit from incorporating this new information
   - The relationship is strong enough to warrant an update (not just tangentially related)

2. **Should NOT Evolve** if:
   - The new memory is only loosely related or tangentially connected
   - The topic's current content already covers the new information
   - The update would be redundant or insignificant

**FOR TOPICS THAT SHOULD EVOLVE**:
- Provide evolved_content: synthesized content that integrates the new memory with existing topic content
- Keep it concise but comprehensive (2-4 sentences typically)
- Maintain the topic's core focus while incorporating relevant new information

Respond with JSON array (include ALL topics, mark should_evolve true/false):
[
  {{
    "index": 1,
    "topic": "topic_name",
    "should_evolve": true|false,
    "reasoning": "brief explanation",
    "evolved_content": "synthesized content (only if should_evolve=true)",
    "confidence": 0.0-1.0
  }},
  ...
]
"""

        schema = {
            "name": "evolution_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "topic": {"type": "string"},
                                "should_evolve": {"type": "boolean"},
                                "reasoning": {"type": "string"},
                                "evolved_content": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": [
                                "index",
                                "topic",
                                "should_evolve",
                                "reasoning",
                                "confidence",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are an expert knowledge graph curator that decides when related topics should evolve based on new information. Be selective - only evolve topics that truly benefit from the new information.",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.3,
            max_tokens=4000,
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)
            result = json.loads(response.content, strict=False)

            decisions = []
            for decision in result.get("decisions", []):
                idx = decision["index"] - 1
                if idx < 0 or idx >= len(candidate_topics):
                    continue

                candidate = candidate_topics[idx]

                decision_dict = {
                    "topic": decision["topic"],
                    "should_evolve": decision["should_evolve"],
                    "reasoning": decision["reasoning"],
                    "confidence": decision["confidence"],
                    "current_state_id": candidate.state.id,
                    "current_version": candidate.state.version,
                    "current_tags": candidate.state.tags,
                }

                if decision["should_evolve"] and "evolved_content" in decision:
                    decision_dict["evolved_content"] = decision["evolved_content"]
                else:
                    decision_dict["evolved_content"] = None

                decisions.append(decision_dict)

            return decisions

        except Exception as e:
            print(f"[CascadingEvolution] LLM analysis failed: {e}")
            # Fallback: don't evolve anything
            return [
                {
                    "topic": result.state.topic,
                    "should_evolve": False,
                    "reasoning": f"LLM analysis failed: {str(e)}",
                    "confidence": 0.0,
                    "current_state_id": result.state.id,
                    "current_version": result.state.version,
                    "current_tags": result.state.tags,
                    "evolved_content": None,
                }
                for result in candidate_topics
            ]


# Global instance
_cascading_evolution_manager = None


def get_cascading_evolution_manager() -> CascadingEvolutionManager:
    """Get or create the global cascading evolution manager instance"""
    global _cascading_evolution_manager
    if _cascading_evolution_manager is None:
        _cascading_evolution_manager = CascadingEvolutionManager()
    return _cascading_evolution_manager
