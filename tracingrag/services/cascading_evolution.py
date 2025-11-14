"""Cascading Evolution Manager for intelligent topic evolution

This module handles cascading updates when a new memory is created:
1. Find semantically related topics using embedding search
2. Deduplicate to keep only latest version of each topic
3. Use LLM to decide which topics should evolve based on new memory
4. Evolve those topics (create new versions)
5. Then RelationshipManager can update edges with the evolved states
"""

import json
from typing import TYPE_CHECKING, Any

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.models import MemoryStateDB
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


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
        """Trigger single-level cascading evolution with concurrent processing

        Strategy:
        1. Find semantically related states (latest versions only)
        2. LLM decides which topics should evolve
        3. Concurrently evolve selected topics
        4. Relationship propagation handled by RelationshipManager

        Args:
            new_state: The newly created memory state
            similarity_threshold: Minimum similarity for candidates (higher = more selective)
            max_evolutions: Maximum number of topics to evolve

        Returns:
            Dict with evolution statistics
        """
        import asyncio

        from tracingrag.services.memory import MemoryService
        from tracingrag.services.retrieval import RetrievalService

        # Statistics
        stats = {
            "evolved_topics": [],
            "skipped_topics": [],
            "total_candidates": 0,
        }

        memory_service = MemoryService()
        retrieval_service = RetrievalService()

        logger.info(
            f"🌊 Starting single-level cascading evolution from: {new_state.topic} v{new_state.version}"
        )

        # Step 1: Find semantically related states (only search once!)
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=max_evolutions * 3,  # Get more candidates for LLM to filter
            score_threshold=similarity_threshold,
            latest_only=True,  # Only get latest versions (no manual deduplication needed)
        )

        # Filter out self
        candidate_topics = [
            result
            for result in similar_states
            if result.state.id != new_state.id and result.state.topic != new_state.topic
        ]

        stats["total_candidates"] = len(candidate_topics)

        logger.info(f"   Found {len(similar_states)} similar states")
        logger.info(f"   After filtering: {len(candidate_topics)} unique topics")

        if not candidate_topics:
            logger.info(f"   No candidates for {new_state.topic}")
            return stats

        # Step 2: LLM decides which topics should evolve
        evolution_decisions = await self._llm_evolution_analysis(
            new_state=new_state,
            candidate_topics=candidate_topics[:max_evolutions],
        )

        # Step 3: Concurrently evolve selected topics
        async def evolve_topic(decision: dict[str, Any]) -> dict[str, Any] | None:
            """Helper to evolve a single topic"""
            topic = decision["topic"]

            if not decision["should_evolve"]:
                logger.info(f"   ⊘ Skipping {topic}: {decision['reasoning']}")
                return {
                    "skipped": True,
                    "topic": topic,
                    "reason": decision["reasoning"],
                }

            try:
                # Evolve this topic
                evolved_state = await memory_service.create_memory_state(
                    topic=topic,
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

                evolution_info = {
                    "skipped": False,
                    "topic": topic,
                    "old_version": decision["current_version"],
                    "new_version": evolved_state.version,
                    "new_state_id": str(evolved_state.id),
                    "reasoning": decision["reasoning"],
                }

                logger.info(
                    f"   ✓ Evolved: {topic} v{decision['current_version']} → v{evolved_state.version}"
                )
                return evolution_info

            except Exception as e:
                logger.error(f"   ✗ Failed to evolve {topic}: {e}")
                return {
                    "skipped": True,
                    "topic": topic,
                    "reason": f"Evolution failed: {str(e)}",
                }

        # Execute all evolutions concurrently
        logger.info(f"   Executing {len(evolution_decisions)} evolutions concurrently...")
        results = await asyncio.gather(
            *[evolve_topic(decision) for decision in evolution_decisions]
        )

        # Collect results
        for result in results:
            if result:
                if result.get("skipped"):
                    stats["skipped_topics"].append(
                        {
                            "topic": result["topic"],
                            "reason": result["reason"],
                        }
                    )
                else:
                    stats["evolved_topics"].append(
                        {
                            "topic": result["topic"],
                            "old_version": result["old_version"],
                            "new_version": result["new_version"],
                            "new_state_id": result["new_state_id"],
                            "reasoning": result["reasoning"],
                        }
                    )

        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("🎉 Single-level Cascading Evolution Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"   Total Candidates: {stats['total_candidates']}")
        logger.info(f"   Total Evolved: {len(stats['evolved_topics'])}")
        logger.info(f"   Total Skipped: {len(stats['skipped_topics'])}")
        logger.info(f"{'='*70}\n")

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

**CRITICAL**: "reasoning" MUST be under 50 characters. Examples:
- GOOD: "new info relevant"
- GOOD: "already covered"
- BAD: "The new memory provides significant information that would benefit this topic"

Respond with JSON array (include ALL topics, mark should_evolve true/false):
[
  {{
    "index": 1,
    "topic": "topic_name",
    "should_evolve": true|false,
    "reasoning": "1-5 words max",
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
                                "reasoning": {"type": "string", "maxLength": 80},
                                "evolved_content": {
                                    "anyOf": [{"type": "string"}, {"type": "null"}]
                                },
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": [
                                "index",
                                "topic",
                                "should_evolve",
                                "reasoning",
                                "evolved_content",
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
            max_tokens=8000,  # Increased for better completion
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)

            # Check for empty response
            if not response.content or not response.content.strip():
                logger.error("Empty LLM response for cascading evolution")
                return []

            # Try to parse JSON with error handling
            try:
                result = json.loads(response.content, strict=False)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed: {json_err}")
                logger.error(f"Response content (first 500 chars): {response.content[:500]}")

                # Try to fix common issues
                fixed_content = response.content

                # 1. Remove leading text before JSON
                if "{" in fixed_content or "[" in fixed_content:
                    json_start = min(
                        fixed_content.find("{") if "{" in fixed_content else len(fixed_content),
                        fixed_content.find("[") if "[" in fixed_content else len(fixed_content),
                    )
                    fixed_content = fixed_content[json_start:]

                # 2. Remove trailing text after JSON
                if "}" in fixed_content or "]" in fixed_content:
                    json_end = (
                        max(
                            fixed_content.rfind("}") if "}" in fixed_content else -1,
                            fixed_content.rfind("]") if "]" in fixed_content else -1,
                        )
                        + 1
                    )
                    fixed_content = fixed_content[:json_end]

                # 3. Fix trailing commas
                fixed_content = fixed_content.replace(",]", "]").replace(",}", "}")

                # 4. Close incomplete JSON
                open_braces = fixed_content.count("{")
                close_braces = fixed_content.count("}")
                open_brackets = fixed_content.count("[")
                close_brackets = fixed_content.count("]")

                if open_braces > close_braces or open_brackets > close_brackets:
                    if fixed_content.count('"') % 2 != 0:
                        fixed_content += '"'
                    while open_brackets > close_brackets:
                        fixed_content += "]"
                        close_brackets += 1
                    while open_braces > close_braces:
                        fixed_content += "}"
                        close_braces += 1

                try:
                    result = json.loads(fixed_content, strict=False)
                    logger.info("Successfully parsed JSON after cleaning")
                except json.JSONDecodeError:
                    logger.error("Failed to fix JSON, returning empty list")
                    return []

            # Handle both {"decisions": [...]} and direct [...] formats
            # Some LLM providers return array directly despite json_schema specification
            if isinstance(result, list):
                logger.info("Note: LLM returned array format (expected object)")
                decisions_list = result
            elif isinstance(result, dict) and "decisions" in result:
                decisions_list = result["decisions"]
            else:
                logger.info(f"Unexpected response format: {type(result)}")
                logger.info(f"   Response: {response.content[:500]}")
                return []

            decisions = []
            for decision in decisions_list:
                # Use .get() for all fields since LLM may return incomplete objects
                idx = decision.get("index", 0) - 1
                if idx < 0 or idx >= len(candidate_topics):
                    logger.warning(f"Warning: Invalid index {idx + 1}, skipping")
                    continue

                candidate = candidate_topics[idx]

                # Validate required fields
                if not decision.get("topic") or decision.get("should_evolve") is None:
                    logger.warning("Warning: Missing required fields in decision, skipping")
                    logger.info(f"   Decision: {decision}")
                    continue

                decision_dict = {
                    "topic": decision.get("topic", ""),
                    "should_evolve": decision.get("should_evolve", False),
                    "reasoning": decision.get("reasoning", "No reasoning provided"),
                    "confidence": decision.get("confidence", 0.5),  # Default confidence
                    "current_state_id": candidate.state.id,
                    "current_version": candidate.state.version,
                    "current_tags": candidate.state.tags,
                    "evolved_content": decision.get("evolved_content", None),
                }

                decisions.append(decision_dict)

            return decisions

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
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
