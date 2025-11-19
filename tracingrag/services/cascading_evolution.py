"""Cascading Evolution Manager for intelligent topic evolution

This module handles cascading updates when a new memory is created:
1. Find semantically related topics using embedding search
2. Deduplicate to keep only latest version of each topic
3. Use LLM to decide which topics should evolve based on new memory
4. Evolve those topics (create new versions)
5. Then RelationshipManager can update edges with the evolved states
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.models import MemoryStateDB
from tracingrag.utils.json_utils import parse_llm_json
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


class EvolutionDecision(BaseModel):
    """Pydantic schema for evolution decision structured output"""

    index: int = Field(..., description="1-based index of the candidate topic")
    topic: str = Field(..., description="Topic name")
    should_evolve: bool = Field(..., description="Whether this topic should evolve")
    reasoning: str = Field(..., max_length=80, description="Brief reason for decision")
    evolved_content: str = Field(
        "",
        description="New evolved content if should_evolve is True (empty string if not evolving)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class EvolutionAnalysisResponse(BaseModel):
    """Schema for evolution analysis LLM response (wraps array for OpenAI compatibility)"""

    decisions: list[EvolutionDecision] = Field(
        default_factory=list, description="List of evolution decisions"
    )


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
        3. Concurrently create cascade states (NO relationship processing)
        4. Return created states for batch relationship processing by caller

        Args:
            new_state: The newly created memory state
            similarity_threshold: Minimum similarity for candidates (higher = more selective)

        Returns:
            Dict with:
            - "created_states": List of MemoryStateDB objects created by cascade
            - "statistics": Evolution statistics
        """
        import asyncio

        from tracingrag.services.memory import MemoryService
        from tracingrag.services.retrieval import RetrievalService

        # Result structure
        result = {
            "created_states": [],  # List of MemoryStateDB objects
            "statistics": {
                "evolved_topics": [],
                "skipped_topics": [],
                "total_candidates": 0,
            },
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
            search_result
            for search_result in similar_states
            if search_result.state.id != new_state.id
            and search_result.state.topic != new_state.topic
        ]

        result["statistics"]["total_candidates"] = len(candidate_topics)

        logger.info(f"   Found {len(similar_states)} similar states")
        logger.info(f"   After filtering: {len(candidate_topics)} unique topics")

        if not candidate_topics:
            logger.info(f"   No candidates for {new_state.topic}")
            return result

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
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"   ✗ Batch {i+1}/{len(batches)} failed: {batch_result}")
            else:
                logger.info(f"   ✓ Batch {i+1}/{len(batches)}: {len(batch_result)} decisions")
                all_decisions.extend(batch_result)

        evolution_decisions = all_decisions

        # Deduplicate decisions by topic (keep first decision for each topic)
        seen_topics = set()
        unique_decisions = []
        for decision in evolution_decisions:
            topic = decision.get("topic")
            if topic and topic not in seen_topics:
                seen_topics.add(topic)
                unique_decisions.append(decision)
            elif topic in seen_topics:
                logger.warning(f"   ⚠️ Duplicate decision for {topic}, keeping first decision only")

        evolution_decisions = unique_decisions
        logger.info(f"   After deduplication: {len(evolution_decisions)} unique topics to evolve")

        # Step 3: Concurrently create cascade states (NO relationship processing)
        async def evolve_topic(
            decision: dict[str, Any],
        ) -> tuple[MemoryStateDB | None, dict[str, Any]]:
            """Helper to create a cascade state

            Returns:
                Tuple of (created_state_or_None, evolution_info)
            """
            topic = decision["topic"]

            if not decision["should_evolve"]:
                logger.info(f"   ⊘ Skipping {topic}: {decision['reasoning']}")
                return None, {
                    "skipped": True,
                    "topic": topic,
                    "reason": decision["reasoning"],
                }

            try:
                # Create cascade state using _create_state_only (NO cascade/promote/relationship processing)
                evolved_state = await memory_service._create_state_only(
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
                    f"   ✓ Created cascade state: {topic} v{decision['current_version']} → v{evolved_state.version}"
                )
                return evolved_state, evolution_info

            except Exception as e:
                logger.error(f"   ✗ Failed to create cascade state for {topic}: {e}")
                return None, {
                    "skipped": True,
                    "topic": topic,
                    "reason": f"Creation failed: {str(e)}",
                }

        # Execute cascade state creations in batches to prevent connection pool exhaustion
        # Database pool: 20 + 10 overflow = 30 max connections
        # Use batch processing with conservative batch size to leave headroom for other operations
        batch_size = 8  # Process 8 cascade states per batch (conservative)

        # Split evolution decisions into batches
        batches = [
            evolution_decisions[i : i + batch_size]
            for i in range(0, len(evolution_decisions), batch_size)
        ]

        num_batches = len(batches)
        logger.info(
            f"   Creating {len(evolution_decisions)} cascade states in {num_batches} "
            f"batch(es) of up to {batch_size} each..."
        )

        # Process batches sequentially
        results = []
        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"   📦 Processing batch {batch_idx + 1}/{num_batches} " f"({len(batch)} states)..."
            )

            # Process all items in this batch concurrently
            batch_results = await asyncio.gather(
                *[evolve_topic(decision) for decision in batch],
                return_exceptions=True,  # Don't fail entire batch if one fails
            )

            # Filter out exceptions and log them
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(
                        f"   ✗ Failed to create cascade state in batch {batch_idx + 1}: {batch_result}"
                    )
                else:
                    results.append(batch_result)

            logger.info(
                f"   ✓ Batch {batch_idx + 1}/{num_batches} complete "
                f"({len([r for r in batch_results if not isinstance(r, Exception)])}/{len(batch)} succeeded)"
            )

        # Collect created states and statistics
        for state, info in results:
            # Validate info is dict
            if not isinstance(info, dict):
                logger.error(f"Invalid info type: {type(info)}, expected dict. Skipping.")
                continue

            if state:
                # Add to created_states list
                result["created_states"].append(state)
                # Add to evolved_topics statistics
                result["statistics"]["evolved_topics"].append(
                    {
                        "topic": info.get("topic", "unknown"),
                        "old_version": info.get("old_version", 0),
                        "new_version": info.get("new_version", 0),
                        "new_state_id": info.get("new_state_id", ""),
                        "reasoning": info.get("reasoning", ""),
                    }
                )
            elif info.get("skipped"):
                # Add to skipped_topics statistics
                result["statistics"]["skipped_topics"].append(
                    {
                        "topic": info.get("topic", "unknown"),
                        "reason": info.get("reason", ""),
                    }
                )

        # Final summary
        stats = result["statistics"]
        logger.info(f"\n{'='*70}")
        logger.info("🎉 Cascade States Creation Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"   Total Candidates: {stats['total_candidates']}")
        logger.info(f"   Total Created: {len(result['created_states'])}")
        logger.info(f"   Total Skipped: {len(stats['skipped_topics'])}")
        logger.info("   (Relationship processing deferred to caller)")
        logger.info(f"{'='*70}\n")

        return result

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
        for idx, candidate in enumerate(candidate_topics):
            candidate_desc.append(
                f"T{idx + 1}. {candidate.state.topic} v{candidate.state.version} (similarity: {candidate.score:.2f})\n"
                f"    Current content: {candidate.state.content}"
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
  {{"index": 2, "topic": "Another Topic", "should_evolve": false, "reasoning": "not relevant", "evolved_content": "", "confidence": 0.3}},
  ...
]
"""

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
            json_schema=EvolutionAnalysisResponse.model_json_schema(),  # Pydantic schema (auto-formatted)
            metadata={"schema_name": "evolution_analysis"},
        )

        try:
            response = await self.llm_client.generate(request)

            # Check for empty response
            if not response.content or not response.content.strip():
                logger.error("Empty LLM response for cascading evolution")
                return []

            # Parse and validate with Pydantic
            try:
                validated = EvolutionAnalysisResponse.model_validate_json(response.content)
                decisions_list = validated.decisions
            except Exception as e:
                logger.error(f"Failed to validate cascading evolution response with Pydantic: {e}")
                # Fallback to manual parsing
                result = parse_llm_json(response.content, strict=False, fix_incomplete=True)
                if result is None:
                    logger.error("Failed to parse cascading evolution JSON")
                    return []

                # Handle both array (legacy) and object with "decisions" key (current)
                if isinstance(result, list):
                    decisions_list = result
                elif isinstance(result, dict) and "decisions" in result:
                    decisions_list = result["decisions"]
                else:
                    logger.error(
                        f"Expected array or object with 'decisions' but got {type(result)}, response: {response.content[:500]}"
                    )
                    return []

            decisions = []
            seen_indices = set()  # Track used indices to detect LLM duplicates

            for decision in decisions_list:
                # Handle both Pydantic objects and dicts
                if isinstance(decision, EvolutionDecision):
                    # Pydantic object (preferred)
                    idx = decision.index - 1
                    should_evolve_val = decision.should_evolve
                    reasoning_val = decision.reasoning
                    evolved_content_val = decision.evolved_content
                    confidence_val = decision.confidence
                else:
                    # Fallback for dict (manual parsing)
                    idx = decision.get("index", 0) - 1
                    should_evolve_val = decision.get("should_evolve", False)
                    reasoning_val = decision.get("reasoning", "No reasoning provided")
                    evolved_content_val = decision.get("evolved_content", "")
                    confidence_val = decision.get("confidence", 0.5)

                # Check for duplicate index (LLM error)
                if idx in seen_indices:
                    logger.warning(f"Warning: Duplicate index {idx + 1} in LLM response, skipping")
                    continue

                if idx < 0 or idx >= len(candidate_topics):
                    logger.warning(f"Warning: Invalid index {idx + 1}, skipping")
                    continue

                seen_indices.add(idx)
                candidate = candidate_topics[idx]

                # Validate required fields
                if should_evolve_val is None:
                    logger.warning("Warning: Missing required fields in decision, skipping")
                    continue

                # IMPORTANT: Always use the candidate's actual topic name, not LLM's returned topic
                # LLM might include version numbers (e.g., "Lisa Wong v2" instead of "Lisa Wong")
                decision_dict = {
                    "topic": candidate.state.topic,  # Use actual topic from candidate
                    "should_evolve": should_evolve_val,
                    "reasoning": reasoning_val,
                    "confidence": confidence_val,
                    "current_state_id": candidate.state.id,
                    "current_version": candidate.state.version,
                    "current_tags": candidate.state.tags,
                    "evolved_content": evolved_content_val,
                }

                decisions.append(decision_dict)

            return decisions

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Fallback: don't evolve anything
            return [
                {
                    "topic": candidate.state.topic,
                    "should_evolve": False,
                    "reasoning": f"LLM analysis failed: {str(e)}",
                    "confidence": 0.0,
                    "current_state_id": candidate.state.id,
                    "current_version": candidate.state.version,
                    "current_tags": candidate.state.tags,
                    "evolved_content": "",
                }
                for candidate in candidate_topics
            ]


# Global instance
_cascading_evolution_manager = None


def get_cascading_evolution_manager() -> CascadingEvolutionManager:
    """Get or create the global cascading evolution manager instance"""
    global _cascading_evolution_manager
    if _cascading_evolution_manager is None:
        _cascading_evolution_manager = CascadingEvolutionManager()
    return _cascading_evolution_manager
