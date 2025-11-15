"""Cascading Evolution Manager for intelligent topic evolution

This module handles cascading updates when a new memory is created:
1. Find semantically related topics using embedding search
2. Deduplicate to keep only latest version of each topic
3. Use LLM to decide which topics should evolve based on new memory
4. Evolve those topics (create new versions)
5. Then RelationshipManager can update edges with the evolved states
"""

from typing import TYPE_CHECKING, Any

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.models import MemoryStateDB
from tracingrag.utils.json_utils import parse_llm_json
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

        # Step 1: Get total count of active memory states for limit
        total_active_states = await memory_service.count_states(storage_tier="active")
        logger.info(f"   Total active states in database: {total_active_states}")

        # Step 2: Find semantically related states (search all active states)
        # Get ALL candidates above similarity threshold - let LLM decide which to evolve
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=total_active_states,  # Use all active states as limit
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

        # Step 2: LLM decides which topics should evolve (parallel batch processing)
        # Split into batches to avoid token limits
        batch_size = settings.cascading_evolution_batch_size

        # Create batches (process ALL candidates, not just first max_evolutions)
        batches = [
            candidate_topics[i : i + batch_size]
            for i in range(0, len(candidate_topics), batch_size)
        ]

        if len(batches) > 1:
            logger.info(
                f"   📦 Processing {len(candidate_topics)} candidates in {len(batches)} batches (parallel)"
            )
        else:
            logger.info(f"   📦 Processing {len(candidate_topics)} candidates in 1 batch")

        # Process all batches in parallel
        batch_results = await asyncio.gather(
            *[
                self._llm_evolution_analysis(new_state=new_state, candidate_topics=batch)
                for batch in batches
            ],
            return_exceptions=True,  # Don't fail all batches if one fails
        )

        # Collect results, filtering out exceptions
        all_decisions = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"   ✗ Batch {i+1}/{len(batches)} failed: {result}")
            else:
                logger.info(f"   ✓ Batch {i+1}/{len(batches)}: {len(result)} decisions")
                all_decisions.extend(result)

        evolution_decisions = all_decisions

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
                # Temporarily reduce RelationshipManager log level during cascade
                import logging

                from tracingrag.services.relationship_manager import logger as rel_logger

                old_level = rel_logger.level
                rel_logger.setLevel(logging.WARNING)  # Only show warnings/errors

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
                finally:
                    # Restore original log level
                    rel_logger.setLevel(old_level)

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
                f"    Current content: {result.state.content}"
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

**FOR TOPICS THAT SHOULD EVOLVE - CRITICAL REQUIREMENTS**:
⚠️ **INFORMATION PRESERVATION (MANDATORY)**:
   - You MUST preserve ALL existing factual information from the current content
   - Do NOT omit, summarize, or drop any existing facts, details, attributes, or relationships
   - Every piece of information in the current content must appear in evolved_content
   - ONLY add the new relevant information from the new memory

⚠️ **NO HALLUCINATION (MANDATORY)**:
   - Do NOT add information that is not explicitly stated in either the current content or new memory
   - Do NOT make inferences, assumptions, or logical deductions beyond what is stated
   - Do NOT add temporal markers (like "for years", "recently") unless explicitly stated
   - If uncertain about a detail, preserve the original wording exactly

⚠️ **CONTENT STRUCTURE**:
   - Provide evolved_content that integrates new information WITH all existing information
   - Length should be as long as needed to preserve all information (not just 2-4 sentences)
   - Maintain the topic's core focus while incorporating relevant new information
   - Structure: [ALL existing facts] + [new relevant facts from new memory]

**CRITICAL**: "reasoning" MUST be under 50 characters. Examples:
- GOOD: "new info relevant"
- GOOD: "already covered"
- BAD: "The new memory provides significant information that would benefit this topic"

**CRITICAL - JSON COMPLETENESS**:
⚠️ You MUST output a complete, valid JSON array
⚠️ Every object MUST have ALL required fields
⚠️ Every string MUST be properly closed with quotes
⚠️ The entire JSON array MUST be complete - do NOT truncate output
⚠️ If approaching token limit, prioritize completing the JSON structure over adding more details

Respond with ONLY the JSON array below, NO additional text or explanations:

[
  {{"index": 1, "topic": "Topic Name", "should_evolve": true, "reasoning": "brief reason", "evolved_content": "full content here", "confidence": 0.9}},
  {{"index": 2, "topic": "Another Topic", "should_evolve": false, "reasoning": "not relevant", "evolved_content": null, "confidence": 0.3}},
  ...
]
"""

        schema = {
            "name": "evolution_analysis",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "topic": {"type": "string"},
                        "should_evolve": {"type": "boolean"},
                        "reasoning": {"type": "string", "maxLength": 80},
                        "evolved_content": {"anyOf": [{"type": "string"}, {"type": "null"}]},
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
            },
        }

        # Adjust max_tokens based on batch size
        # Since we now preserve all existing content + add new info, evolved_content can be longer
        # Estimate: ~1500 tokens per topic (full content + evolved_content), add generous buffer
        # For Qwen2.5 32B: context is 32K, output can be up to 8K safely
        estimated_tokens = len(candidate_topics) * 1500 + 3000
        max_tokens = min(
            estimated_tokens, 24000
        )  # Increased to 24K for Qwen (supports up to 32K context)

        request = LLMRequest(
            system_prompt="""You are an expert knowledge graph curator that decides when related topics should evolve based on new information.

CRITICAL RULES:
1. Be selective - only evolve topics that truly benefit from the new information
2. NEVER drop or omit existing factual information when evolving
3. NEVER add information not explicitly stated in the source content
4. Preserve complete information integrity - every existing fact must remain""",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.3,
            max_tokens=max_tokens,
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)

            # Check for empty response
            if not response.content or not response.content.strip():
                logger.error("Empty LLM response for cascading evolution")
                return []

            # Parse JSON with automatic cleaning and error recovery
            # Schema expects array directly
            result = parse_llm_json(response.content, strict=False, fix_incomplete=True)
            if result is None:
                logger.error("Failed to parse cascading evolution JSON")
                return []

            # Handle both array (correct) and object with "decisions" key (LLM mistake)
            if isinstance(result, list):
                decisions_list = result
            elif isinstance(result, dict) and "decisions" in result:
                logger.debug("LLM returned object instead of array, extracting 'decisions' field")
                decisions_list = result["decisions"]
            else:
                logger.error(
                    f"Expected array but got {type(result)}, response: {response.content[:500]}"
                )
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
                if decision.get("should_evolve") is None:
                    logger.warning("Warning: Missing required fields in decision, skipping")
                    logger.info(f"   Decision: {decision}")
                    continue

                # IMPORTANT: Always use the candidate's actual topic name, not LLM's returned topic
                # LLM might include version numbers (e.g., "Lisa Wong v2" instead of "Lisa Wong")
                decision_dict = {
                    "topic": candidate.state.topic,  # Use actual topic from candidate
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
