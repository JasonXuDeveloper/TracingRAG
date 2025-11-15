"""Intelligent relationship management for memory state evolution

This module handles smart relationship updates when memory states evolve,
ensuring relationships stay current and relevant across large-scale knowledge graphs.
"""

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB
from tracingrag.storage.neo4j_client import (
    create_memory_relationship,
    get_parent_relationships,
)
from tracingrag.utils.json_utils import parse_llm_json
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


class RelationshipManager:
    """Manages intelligent relationship updates during memory evolution"""

    def __init__(self):
        self.llm_client = get_llm_client()

    async def _call_llm_with_retry(
        self,
        request: LLMRequest,
        context_label: str = "LLM call",
    ) -> dict[str, Any] | None:
        """Call LLM with automatic retry on JSON parsing failures

        Args:
            request: LLM request to execute
            context_label: Label for logging (e.g., "round 1", "initial analysis")

        Returns:
            Parsed JSON result, or None if all retries fail
        """
        max_retries = settings.llm_max_retries
        base_delay = settings.llm_retry_base_delay
        original_prompt = request.user_message

        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(request)

                # Check for empty response
                if not response.content or not response.content.strip():
                    logger.warning(
                        f"Empty LLM response for {context_label}, attempt {attempt + 1}/{max_retries}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (2**attempt))
                        continue
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed with empty responses for {context_label}"
                        )
                        return None

                # Parse JSON with automatic cleaning and error recovery
                result = parse_llm_json(
                    response.content, strict=False, fix_incomplete=True, fix_malformed=True
                )
                if result is None:
                    logger.warning(
                        f"Failed to parse JSON for {context_label}, attempt {attempt + 1}/{max_retries}"
                    )
                    if attempt < max_retries - 1:
                        # Add error feedback to prompt for retry
                        logger.info("Retrying with enhanced prompt...")
                        request.user_message = (
                            original_prompt
                            + "\n\n**CRITICAL**: Previous response had JSON parsing errors. "
                            "Please ensure your response is VALID JSON with:\n"
                            "1. ALL property names in double quotes\n"
                            '2. Colons after property names: "name": value\n'
                            "3. Commas between array elements\n"
                            "4. Properly closed brackets and braces"
                        )
                        await asyncio.sleep(base_delay * (2**attempt))
                        continue
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed to produce valid JSON for {context_label}"
                        )
                        return None
                else:
                    # Success!
                    if attempt > 0:
                        logger.info(
                            f"✓ JSON parsing succeeded on attempt {attempt + 1} for {context_label}"
                        )
                    return result

            except Exception as e:
                logger.error(
                    f"LLM generation error for {context_label}, attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue
                else:
                    return None

        return None

    async def update_relationships_on_evolution(
        self,
        new_state: MemoryStateDB,
        parent_state_id: UUID,
        similarity_threshold: float = 0.4,
    ) -> dict[str, Any]:
        """Intelligently update relationships when a state evolves

        Strategy (optimized for accuracy and scale):
        1. Full vector DB search for all semantically related states (no limit)
        2. Deduplicate: keep only latest version of each topic
        3. Get parent's existing relationships
        4. LLM analyzes ALL candidates + existing relationships (batched if needed)
        5. Merge: (parent relationships to keep) + (new relationships) - (removed relationships)

        Args:
            new_state: The newly created memory state
            parent_state_id: ID of the parent state it evolved from
            similarity_threshold: Minimum embedding similarity score (lower = more comprehensive)

        Returns:
            Dict with update statistics: {updated, created, deleted, kept}
        """
        from tracingrag.services.retrieval import RetrievalService

        stats = {
            "updated": 0,  # Existing relationships updated to newer target versions
            "created": 0,  # New relationships created
            "deleted": 0,  # Old relationships removed
            "kept": 0,  # Existing relationships kept as-is
        }

        # Step 1: Get parent's current relationships
        parent_relationships = await get_parent_relationships(parent_state_id)

        logger.info(f"Analyzing relationships for {new_state.topic} v{new_state.version}")
        logger.info(f"   Parent has {len(parent_relationships)} existing relationships")

        # Step 2: Query EXISTING target states directly from DB (INCLUDING inactive/archived!)
        # Critical: These are states the parent was related to - we need full content for LLM analysis
        # regardless of is_active or storage_tier status
        existing_target_states = {}
        if parent_relationships:
            # Filter and convert target_ids to UUID, skipping any invalid ones
            existing_target_ids = []
            for rel in parent_relationships:
                try:
                    target_id = rel["target_id"]
                    if target_id and str(target_id).strip():
                        existing_target_ids.append(UUID(target_id))
                    else:
                        logger.warning(
                            f"Skipping relationship with empty target_id in parent {parent_state_id}"
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Skipping relationship with invalid target_id '{rel.get('target_id')}': {e}"
                    )
                    continue

            if not existing_target_ids:
                logger.info("   No valid existing target IDs found")
                existing_target_states = {}
            else:
                async with get_session() as session:
                    result = await session.execute(
                        select(MemoryStateDB).where(MemoryStateDB.id.in_(existing_target_ids))
                    )
                    existing_target_states = {state.id: state for state in result.scalars().all()}

            logger.info(
                f"   Loaded {len(existing_target_states)} existing target states (may include inactive/archived)"
            )

        # Step 3: Get total memory states count for comprehensive search
        from tracingrag.services.memory import MemoryService

        memory_service = MemoryService()
        total_states = await memory_service.count_states(latest_only=False)
        search_limit = max(total_states, 1000)

        # Step 4: Semantic search for NEW active candidates only
        # These are potential NEW relationships - only consider active states
        retrieval_service = RetrievalService()
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=search_limit,
            score_threshold=similarity_threshold,
            latest_only=True,
            filter_conditions={"is_active": True},  # NEW relationships only to active states
        )

        # Filter out self and same topic
        candidate_states = [
            s
            for s in similar_states
            if s.state.id != new_state.id and s.state.topic != new_state.topic
        ]

        logger.info(f"   Found {len(candidate_states)} active candidates for NEW relationships")

        # Step 5: Check if any existing target has newer active version
        # Build map of topic -> latest active state from candidates
        candidate_topic_map = {result.state.topic: result for result in candidate_states}

        # Enrich parent relationships with content and version info
        enriched_parent_rels = []
        for rel in parent_relationships:
            try:
                target_id = UUID(rel["target_id"])
                target_state = existing_target_states.get(target_id)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"   Invalid target_id in relationship: {e}, skipping")
                continue

            if not target_state:
                logger.warning(f"   Missing target state {rel.get('target_id')}, skipping")
                continue

            # Add target content for LLM analysis
            rel["target_content"] = target_state.content
            rel["target_is_active"] = target_state.is_active
            rel["target_storage_tier"] = target_state.storage_tier

            # Check if this topic has newer active version
            if rel["target_topic"] in candidate_topic_map:
                candidate = candidate_topic_map[rel["target_topic"]]
                if candidate.state.version > rel["target_version"]:
                    rel["has_newer_version"] = True
                    rel["latest_version"] = candidate.state.version
                    rel["latest_id"] = str(candidate.state.id)
                    rel["latest_content"] = candidate.state.content[:300]
                else:
                    rel["has_newer_version"] = False
            else:
                rel["has_newer_version"] = False

            enriched_parent_rels.append(rel)

        # Step 6: Use LLM to analyze all relationships (iterative multi-round)
        relationship_decisions = await self._llm_comprehensive_analysis_batched(
            new_state=new_state,
            existing_relationships=enriched_parent_rels,
            candidate_states=candidate_states,
        )

        # Step 5: Apply decisions to graph
        for decision in relationship_decisions:
            action = decision["action"]

            if action == "keep_existing":
                # Keep existing relationship as-is
                rel = decision["relationship"]
                try:
                    target_id = UUID(rel["target_id"])
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"   Invalid target_id in keep_existing: {e}, skipping")
                    continue

                await create_memory_relationship(
                    source_id=new_state.id,
                    target_id=target_id,
                    relationship_type=rel["rel_type"],
                    properties=rel.get("properties", {}),
                )
                stats["kept"] += 1

            elif action == "update_to_newer":
                # Update existing relationship to newer version of target
                rel = decision["relationship"]
                # Only update if we have the new target ID
                if "new_target_id" in decision:
                    try:
                        target_id = UUID(decision["new_target_id"])
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"   Invalid new_target_id in update_to_newer: {e}, skipping"
                        )
                        continue

                    await create_memory_relationship(
                        source_id=new_state.id,
                        target_id=target_id,
                        relationship_type=rel["rel_type"],
                        properties=rel.get("properties", {}),
                    )
                    stats["updated"] += 1
                    logger.info(
                        f"   ✓ Updated: {rel['target_topic']} v{rel['target_version']} → v{decision['new_version']}"
                    )
                else:
                    # Fallback: keep existing if no newer version available
                    try:
                        target_id = UUID(rel["target_id"])
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(
                            f"   Invalid target_id in update_to_newer fallback: {e}, skipping"
                        )
                        continue

                    await create_memory_relationship(
                        source_id=new_state.id,
                        target_id=target_id,
                        relationship_type=rel["rel_type"],
                        properties=rel.get("properties", {}),
                    )
                    stats["kept"] += 1
                    logger.warning(
                        f"   ⚠ Kept existing (no newer version): {rel['target_topic']} v{rel['target_version']}"
                    )

            elif action == "remove_existing":
                # Don't create this old relationship
                stats["deleted"] += 1
                rel = decision["relationship"]
                logger.info(f"   ✗ Removed: {rel['target_topic']} (no longer relevant)")

            elif action == "create_new":
                # Create new relationship to a candidate
                try:
                    target_id = UUID(decision["target_id"])
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"   Invalid target_id in create_new: {e}, skipping")
                    continue

                await create_memory_relationship(
                    source_id=new_state.id,
                    target_id=target_id,
                    relationship_type=decision["relationship_type"],
                    properties={"confidence": decision.get("confidence", 0.8)},
                )
                stats["created"] += 1
                logger.info(
                    f"   + Created: {decision['relationship_type']} → {decision['target_topic']}"
                )

        logger.info(
            f"Summary: {stats['kept']} kept, {stats['updated']} updated, "
            f"{stats['created']} created, {stats['deleted']} deleted"
        )

        return stats

    async def _llm_comprehensive_analysis_batched(
        self,
        new_state: MemoryStateDB,
        existing_relationships: list[dict[str, Any]],
        candidate_states: list[Any],
    ) -> list[dict[str, Any]]:
        """Iterative multi-round analysis to process unlimited number of states

        Strategy (with layered processing for large existing relationships):
        - Compute importance for all existing relationships
        - If existing_tokens > 60% of MAX: split into high/low priority layers
        - Round 1a: High-priority existing + top candidates
        - Round 1b: Low-priority existing + mid candidates (if needed)
        - Round 2+: Remaining candidates
        - Each round accumulates decisions
        - Deduplication ensures no state is analyzed twice

        Args:
            new_state: The new memory state
            existing_relationships: Enriched parent relationships (MUST all be processed)
            candidate_states: Candidate states SORTED by score (high to low)

        Returns:
            Combined list of all relationship decisions from all rounds
        """
        MAX_TOKENS_PER_ROUND = 20000  # Target token count per LLM call
        AVG_CHARS_PER_TOKEN = 3  # Rough estimate
        EXISTING_TOKEN_THRESHOLD = MAX_TOKENS_PER_ROUND * 0.6  # 12k tokens

        all_decisions = []
        processed_state_ids = set()  # Deduplication
        previous_decisions_summary = None
        candidate_idx = 0

        # STEP 1: Compute importance for existing relationships
        for rel in existing_relationships:
            # Compute importance score
            target_confidence = rel.get("target_confidence", 0.5)
            strength = rel.get("properties", {}).get("strength", 0.5)
            access_count = rel.get("properties", {}).get("access_count", 0)
            last_accessed = rel.get("properties", {}).get("last_accessed")

            # Simple importance calculation (mirroring Edge.compute_importance)
            access_freq = min(1.0, access_count / 50.0) if access_count else 0.0

            # Time decay
            if last_accessed:
                from datetime import datetime

                try:
                    last_access_dt = datetime.fromisoformat(last_accessed)
                    days_since = (datetime.utcnow() - last_access_dt).days
                    time_decay = max(0.1, 1.0 - (days_since / 180.0))
                except (ValueError, TypeError):
                    time_decay = 0.5
            else:
                time_decay = 0.5  # Neutral for new relationships

            importance = (
                strength * 0.4 + access_freq * 0.3 + target_confidence * 0.2 + time_decay * 0.1
            )
            rel["_importance"] = min(1.0, max(0.0, importance))

        # STEP 2: Sort existing relationships by importance (high to low)
        sorted_existing = sorted(
            existing_relationships, key=lambda r: r.get("_importance", 0.5), reverse=True
        )

        if sorted_existing:
            logger.info(
                f"   Starting iterative analysis: {len(existing_relationships)} existing "
                f"(importance: {sorted_existing[0]['_importance']:.2f} to {sorted_existing[-1]['_importance']:.2f}), "
                f"{len(candidate_states)} candidates"
            )
        else:
            logger.info(
                f"   Starting iterative analysis: no existing relationships, "
                f"{len(candidate_states)} candidates"
            )

        # STEP 3: Calculate existing tokens and decide layering strategy
        def estimate_rel_tokens(rel: dict) -> int:
            chars = (
                len(rel.get("target_content", "")[:500])
                + len(rel.get("latest_content", "")[:300])
                + 200  # Overhead
            )
            return chars // AVG_CHARS_PER_TOKEN

        total_existing_tokens = sum(estimate_rel_tokens(r) for r in sorted_existing)

        # STEP 4: Layered processing if existing is too large
        if total_existing_tokens > EXISTING_TOKEN_THRESHOLD:
            # Split existing into two layers
            mid_point = len(sorted_existing) // 2
            high_priority_existing = sorted_existing[:mid_point]
            low_priority_existing = sorted_existing[mid_point:]

            high_priority_tokens = sum(estimate_rel_tokens(r) for r in high_priority_existing)
            low_priority_tokens = sum(estimate_rel_tokens(r) for r in low_priority_existing)

            logger.info(
                f"   Layered processing: High-priority {len(high_priority_existing)} rels "
                f"({high_priority_tokens} tokens), Low-priority {len(low_priority_existing)} rels "
                f"({low_priority_tokens} tokens)"
            )

            # Process layers
            existing_layers = [
                ("high-priority", high_priority_existing, high_priority_tokens),
                ("low-priority", low_priority_existing, low_priority_tokens),
            ]
        else:
            # Single layer (all existing in one round)
            existing_layers = [("all", sorted_existing, total_existing_tokens)]
            logger.info(
                f"   Single-layer processing: {len(sorted_existing)} existing rels "
                f"({total_existing_tokens} tokens)"
            )

        # STEP 5: Process each layer + remaining candidates
        round_num = 1
        for layer_name, layer_existing, layer_tokens in existing_layers:
            # Calculate how many candidates fit with this layer
            reserved_tokens = 5000  # New state content + instructions
            available_for_candidates = MAX_TOKENS_PER_ROUND - layer_tokens - reserved_tokens

            avg_candidate_chars = 300 + 200  # Content + overhead
            candidates_this_layer = max(
                0, int(available_for_candidates * AVG_CHARS_PER_TOKEN / avg_candidate_chars)
            )
            candidates_this_layer = min(
                candidates_this_layer, len(candidate_states) - candidate_idx
            )

            # Get candidate slice for this layer
            layer_candidates = candidate_states[
                candidate_idx : candidate_idx + candidates_this_layer
            ]
            candidate_idx += candidates_this_layer

            logger.info(
                f"   Round {round_num} ({layer_name}): {len(layer_existing)} existing + "
                f"{len(layer_candidates)} candidates (remaining: {len(candidate_states) - candidate_idx})"
            )

            # Perform LLM analysis for this layer
            if layer_existing or layer_candidates:
                round_decisions = await self._llm_single_round_analysis(
                    new_state=new_state,
                    existing_relationships=layer_existing,
                    candidate_states=layer_candidates,
                    previous_decisions_summary=previous_decisions_summary,
                    round_num=round_num,
                )

                # Accumulate decisions
                all_decisions.extend(round_decisions)

                # Mark processed states
                for candidate in layer_candidates:
                    processed_state_ids.add(candidate.state.id)

                # Generate summary for next round
                if candidate_idx < len(candidate_states) or round_num < len(existing_layers):
                    previous_decisions_summary = self._generate_decisions_summary(
                        round_decisions, round_num
                    )

                round_num += 1

        # STEP 6: Process remaining candidates (after all existing processed)
        while candidate_idx < len(candidate_states):
            # Calculate available tokens for candidates only (no existing)
            available_tokens = MAX_TOKENS_PER_ROUND - 8000  # Reserve for summary + overhead
            avg_candidate_chars = 300 + 200
            candidates_this_round = int(
                available_tokens * AVG_CHARS_PER_TOKEN / avg_candidate_chars
            )
            candidates_this_round = max(1, candidates_this_round)  # At least 1

            end_idx = min(candidate_idx + candidates_this_round, len(candidate_states))
            round_candidates = candidate_states[candidate_idx:end_idx]
            candidate_idx = end_idx

            logger.info(
                f"   Round {round_num}: {len(round_candidates)} candidates "
                f"(remaining: {len(candidate_states) - candidate_idx})"
            )

            # Perform LLM analysis
            round_decisions = await self._llm_single_round_analysis(
                new_state=new_state,
                existing_relationships=[],  # No existing in later rounds
                candidate_states=round_candidates,
                previous_decisions_summary=previous_decisions_summary,
                round_num=round_num,
            )

            # Accumulate decisions
            all_decisions.extend(round_decisions)

            # Mark processed states
            for candidate in round_candidates:
                processed_state_ids.add(candidate.state.id)

            # Generate summary for next round
            if candidate_idx < len(candidate_states):
                previous_decisions_summary = self._generate_decisions_summary(
                    round_decisions, round_num
                )

            round_num += 1

        logger.info(f"   Completed in {round_num - 1} rounds: {len(all_decisions)} total decisions")

        return all_decisions

    def _generate_decisions_summary(self, decisions: list[dict[str, Any]], round_num: int) -> str:
        """Generate concise summary of decisions for next round

        Args:
            decisions: Decisions from current round
            round_num: Current round number

        Returns:
            Concise summary string
        """
        summary_parts = [f"**Round {round_num} Decisions**:"]

        kept = [d for d in decisions if d.get("action") == "keep_existing"]
        updated = [d for d in decisions if d.get("action") == "update_to_newer"]
        removed = [d for d in decisions if d.get("action") == "remove_existing"]
        created = [d for d in decisions if d.get("action") == "create_new"]

        if kept:
            summary_parts.append(f"- Kept {len(kept)} existing relationships")
        if updated:
            summary_parts.append(f"- Updated {len(updated)} to newer versions")
        if removed:
            summary_parts.append(f"- Removed {len(removed)} obsolete relationships")
        if created:
            summary_parts.append(
                f"- Created {len(created)} new: "
                + ", ".join(
                    f"{d.get('relationship_type', 'UNK')} → {d.get('target_topic', 'unknown')}"
                    for d in created[:5]  # Show first 5
                )
            )
            if len(created) > 5:
                summary_parts.append(f"  ... and {len(created) - 5} more")

        return "\n".join(summary_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 3.5 characters (mixed English/Chinese)
        return len(text) // 3

    async def _llm_single_round_analysis(
        self,
        new_state: MemoryStateDB,
        existing_relationships: list[dict[str, Any]],
        candidate_states: list[Any],
        previous_decisions_summary: str | None,
        round_num: int,
    ) -> list[dict[str, Any]]:
        """Single round of LLM analysis with context from previous rounds

        Args:
            new_state: The new memory state
            existing_relationships: Existing relationships (only in round 1)
            candidate_states: Candidates for this round
            previous_decisions_summary: Summary from previous rounds (if any)
            round_num: Current round number

        Returns:
            List of relationship decisions for this round
        """
        # Build prompt sections
        new_content_desc = (
            f"**NEW STATE**\nTopic: {new_state.topic}\nContent: {new_state.content}\n"
        )

        # Previous context (if not round 1)
        previous_context = ""
        if previous_decisions_summary:
            previous_context = f"\n**PREVIOUS ROUNDS**:\n{previous_decisions_summary}\n\n"

        # Existing relationships (only in round 1)
        existing_text = "None"
        if existing_relationships:
            existing_rel_desc = []
            for idx, rel in enumerate(existing_relationships):
                content = rel.get("target_content", "N/A")[:400]
                status_info = []

                if not rel.get("target_is_active", True):
                    status_info.append("INACTIVE")
                if rel.get("target_storage_tier") == "archived":
                    status_info.append("ARCHIVED")

                status_str = f" ({', '.join(status_info)})" if status_info else ""

                newer_version_note = ""
                if rel.get("has_newer_version"):
                    newer_version_note = (
                        f"\n    → NEWER v{rel['latest_version']} (active):\n"
                        f"       {rel.get('latest_content', 'N/A')[:250]}"
                    )

                existing_rel_desc.append(
                    f"E{idx + 1}. [{rel['rel_type']}] {rel['target_topic']} v{rel['target_version']}{status_str}\n"
                    f"    Content: {content}{newer_version_note}"
                )

            existing_text = "\n\n".join(existing_rel_desc)

        # Candidate states
        candidate_desc = []
        for idx, result in enumerate(candidate_states):
            candidate_desc.append(
                f"C{idx + 1}. {result.state.topic} v{result.state.version} (score: {result.score:.2f})\n"
                f"    Content: {result.state.content[:300]}"
            )

        candidates_text = "\n\n".join(candidate_desc) if candidate_desc else "None"

        # Build prompt
        round_context = f"Round {round_num}" if round_num > 1 else "Initial Analysis"
        prompt = f"""{new_content_desc}
{previous_context}
**TASK** ({round_context}): Analyze relationships for the new state.

{"**EXISTING RELATIONSHIPS** (from parent state - may include INACTIVE/ARCHIVED):" if existing_relationships else ""}
{existing_text if existing_relationships else ""}

**CANDIDATE STATES** (active states for potential relationships):
{candidates_text}

**INSTRUCTIONS**:

1. For EXISTING relationships (E1, E2, ...):
   - "update_to_newer": Replace with newer version IF dynamic AND valid
   - "keep_existing": Keep old version IF still valid (or version-bound)
   - "remove_existing": Delete IF no longer relevant

2. For CANDIDATE states (C1, C2, ...):
   - "create_new": Create relationship IF relevant
   - Skip if not relevant

3. **IMPORTANT**: You CAN have MULTIPLE relationships to different versions!
   Example: Keep RELATES_TO old version, CREATE UNAWARE_OF new version

4. Relationship types:
   **Dynamic** (update to latest): RELATES_TO, DEPENDS_ON, SIMILAR_TO, PART_OF, AWARE_OF, MONITORS
   **Version-bound** (historical): CAUSED_BY, CREATED_BY, INFLUENCED_BY, HIDDEN_FROM, UNAWARE_OF

**CRITICAL REQUIREMENTS**:
1. "reasoning" MUST be 1-5 words maximum
2. relationship_type MUST be EXACTLY one of: RELATES_TO, DEPENDS_ON, SIMILAR_TO, PART_OF, AWARE_OF, MONITORS, CAUSED_BY, CREATED_BY, INFLUENCED_BY, HIDDEN_FROM, UNAWARE_OF
3. DO NOT use any other relationship types (e.g., MANAGES, REPORTS_TO, FRIENDS_WITH are INVALID)
4. Respond with ONLY the JSON object below, NO additional text or explanations

Respond with JSON:
{{
  "existing": [
    {{"index": 1, "action": "keep_existing", "reasoning": "still valid"}},
    ...
  ],
  "new": [
    {{"candidate_index": 1, "relationship_type": "RELATES_TO", "reasoning": "same context"}},
    ...
  ]
}}
"""

        # Calculate max_tokens: input tokens + expected output tokens
        # Formula: input_tokens + (|partition| * 100 tokens per item * 1.5 buffer + 1000 for JSON)
        estimated_input_tokens = self._estimate_tokens(prompt)
        num_items = len(existing_relationships) + len(candidate_states)
        estimated_output_tokens = int(num_items * 100 * 1.5) + 1000

        max_tokens = estimated_input_tokens + estimated_output_tokens
        max_tokens = max(2000, min(max_tokens, 32000))  # Clamp between 2k-32k

        logger.debug(
            f"   Round {round_num}: {num_items} items, ~{estimated_input_tokens} input + "
            f"{estimated_output_tokens} output = {max_tokens} max_tokens"
        )

        # JSON schema
        schema = {
            "name": "relationship_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "existing": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "action": {
                                    "type": "string",
                                    "enum": ["keep_existing", "update_to_newer", "remove_existing"],
                                },
                                "reasoning": {"type": "string", "maxLength": 80},
                            },
                            "required": ["index", "action", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                    "new": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_index": {"type": "integer"},
                                "relationship_type": {
                                    "type": "string",
                                    "enum": [
                                        "RELATES_TO",
                                        "DEPENDS_ON",
                                        "SIMILAR_TO",
                                        "PART_OF",
                                        "AWARE_OF",
                                        "MONITORS",
                                        "CAUSED_BY",
                                        "CREATED_BY",
                                        "INFLUENCED_BY",
                                        "HIDDEN_FROM",
                                        "UNAWARE_OF",
                                    ],
                                },
                                "reasoning": {"type": "string", "maxLength": 80},
                            },
                            "required": ["candidate_index", "relationship_type", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["existing", "new"],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are an expert knowledge graph curator analyzing memory relationships. You MUST respond with ONLY valid JSON using the exact relationship types specified in the schema. Do NOT add any explanations or text outside the JSON object.",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=max_tokens,
            json_schema=schema,
        )

        # Call LLM with automatic retry
        result = await self._call_llm_with_retry(request, context_label=f"round {round_num}")
        if result is None:
            logger.error(f"Failed to get valid result after all retries in round {round_num}")
            return []

        # Truncate reasoning fields if too long (LLM sometimes ignores maxLength)
        for item in result.get("existing", []):
            if "reasoning" in item and len(item["reasoning"]) > 80:
                item["reasoning"] = item["reasoning"][:77] + "..."
        for item in result.get("new", []):
            if "reasoning" in item and len(item["reasoning"]) > 80:
                item["reasoning"] = item["reasoning"][:77] + "..."

        # Process decisions
        decisions = []

        # Existing relationships
        for decision in result.get("existing", []):
            # Ensure index is int (LLM might return string)
            idx = int(decision.get("index", 0)) - 1
            if 0 <= idx < len(existing_relationships):
                rel = existing_relationships[idx]
                decision_dict = {
                    "action": decision["action"],
                    "relationship": rel,
                    "reasoning": decision["reasoning"],
                }
                # Only set new_target_id if it exists and is valid
                if decision["action"] == "update_to_newer" and rel.get("latest_id"):
                    decision_dict["new_target_id"] = rel["latest_id"]
                    decision_dict["new_version"] = rel.get("latest_version")
                decisions.append(decision_dict)

        # New relationships
        for decision in result.get("new", []):
            # Ensure candidate_index is int (LLM might return string)
            cidx = int(decision.get("candidate_index", 0)) - 1
            if 0 <= cidx < len(candidate_states):
                candidate = candidate_states[cidx]
                rel_type = decision.get("relationship_type", "")

                # Validate relationship type against enum
                from tracingrag.core.models.graph import RelationshipType

                try:
                    RelationshipType(rel_type)
                except ValueError:
                    logger.warning(
                        f"Round {round_num}: Invalid relationship type '{rel_type}' for candidate {cidx + 1}, skipping"
                    )
                    continue

                decisions.append(
                    {
                        "action": "create_new",
                        "target_id": str(candidate.state.id),
                        "target_topic": candidate.state.topic,
                        "relationship_type": rel_type,
                        "confidence": 0.8,
                        "reasoning": decision["reasoning"],
                    }
                )

        return decisions

    async def _llm_comprehensive_analysis(
        self,
        new_state: MemoryStateDB,
        existing_relationships: list[dict[str, Any]],
        candidate_states: list[Any],  # QueryResult objects from embedding search
    ) -> list[dict[str, Any]]:
        """Comprehensive LLM analysis of all relationships (existing + potential new)

        Args:
            new_state: The new memory state
            existing_relationships: Enriched parent relationships (with version info)
            candidate_states: Similar states found via embedding search

        Returns:
            List of decisions for all relationships
        """
        # Smart truncation to fit context window
        # Target: Keep input under ~80k tokens (assuming 100k context with 20k for output)
        MAX_INPUT_TOKENS = 80000

        # Build prompt sections with dynamic content length
        new_content_desc = (
            f"**NEW STATE**\nTopic: {new_state.topic}\nContent: {new_state.content}\n"
        )

        # Section 1: Existing relationships (prioritize - these are mandatory decisions)
        existing_content_length = 500  # Start with generous length
        existing_newer_version_length = 300

        # Section 2: Candidates (can be truncated more aggressively)
        candidate_content_length = 300  # Start with medium length

        # Iterative truncation loop
        for attempt in range(3):
            existing_rel_desc = []
            for idx, rel in enumerate(existing_relationships):
                content = rel.get("target_content", "N/A")[:existing_content_length]
                status_info = []

                # Show version/status info
                if not rel.get("target_is_active", True):
                    status_info.append("INACTIVE")
                if rel.get("target_storage_tier") == "archived":
                    status_info.append("ARCHIVED")

                status_str = f" ({', '.join(status_info)})" if status_info else ""

                # Show newer version if available
                newer_version_note = ""
                if rel.get("has_newer_version"):
                    newer_version_note = (
                        f"\n    → NEWER v{rel['latest_version']} (active):\n"
                        f"       {rel.get('latest_content', 'N/A')[:existing_newer_version_length]}"
                    )

                existing_rel_desc.append(
                    f"E{idx + 1}. [{rel['rel_type']}] {rel['target_topic']} v{rel['target_version']}{status_str}\n"
                    f"    Content: {content}{newer_version_note}"
                )

            existing_text = "\n\n".join(existing_rel_desc) if existing_rel_desc else "None"

            # Candidate states
            candidate_desc = []
            for idx, result in enumerate(candidate_states):
                candidate_desc.append(
                    f"C{idx + 1}. {result.state.topic} v{result.state.version} (sim: {result.score:.2f})\n"
                    f"    Content: {result.state.content[:candidate_content_length]}"
                )

            candidates_text = "\n\n".join(candidate_desc) if candidate_desc else "None"

            # Build partial prompt to estimate tokens
            test_prompt = f"{new_content_desc}\n{existing_text}\n{candidates_text}"
            estimated_tokens = self._estimate_tokens(test_prompt)

            if estimated_tokens <= MAX_INPUT_TOKENS:
                logger.debug(
                    f"Context fit: {estimated_tokens} tokens (existing: {existing_content_length}ch, "
                    f"candidates: {candidate_content_length}ch)"
                )
                break

            # Adjust lengths for next iteration
            if attempt == 0:
                # First reduction: cut candidates aggressively
                candidate_content_length = 150
                existing_newer_version_length = 200
            elif attempt == 1:
                # Second reduction: cut both
                candidate_content_length = 100
                existing_content_length = 300
                existing_newer_version_length = 150
            else:
                # Last attempt: minimal content
                candidate_content_length = 80
                existing_content_length = 200
                existing_newer_version_length = 100
                logger.warning(
                    f"Context still large ({estimated_tokens} tokens), using minimal content lengths"
                )

        # Comprehensive prompt
        prompt = f"""{new_content_desc}

**TASK**: Analyze ALL relationships for the new state based on its content.

**EXISTING RELATIONSHIPS** (from parent state - may include INACTIVE/ARCHIVED states):
{existing_text}

**CANDIDATE STATES** (active states found via vector search - can be newer versions of existing):
{candidates_text}

**CRITICAL INSTRUCTIONS**:

1. For EACH EXISTING relationship (E1, E2, ...):
   - "update_to_newer": Replace with newer version IF relationship type is dynamic AND semantically valid
   - "keep_existing": Keep old version IF still valid (especially for version-bound relationships)
   - "remove_existing": Delete IF no longer relevant

2. For EACH CANDIDATE state (C1, C2, ...):
   - "create_new": Create relationship IF relevant (specify type)
   - Skip if not relevant

3. **IMPORTANT**: You CAN have MULTIPLE relationships to different versions of the same topic!
   - Example: A.v1 and B.v1 are friends
   - B.v2 betrays A (but A doesn't know)
   - When A evolves to v2:
     * KEEP: A.v2 → B.v1 (RELATES_TO) - A still thinks they're friends
     * CREATE: A.v2 → B.v2 (UNAWARE_OF) - A doesn't know about the betrayal
   - This allows modeling asymmetric knowledge!

4. Relationship types (choose appropriately):

   **Dynamic relationships** (update to latest version):
   - RELATES_TO: General semantic relationship
   - DEPENDS_ON: Current dependency/requirement
   - SIMILAR_TO: Semantically similar content
   - PART_OF: Hierarchical containment
   - AWARE_OF: Entity knows about target (current knowledge)
   - MONITORS: Entity actively tracks target

   **Version-bound relationships** (bind to specific version, record historical facts):
   - CAUSED_BY: Was caused by specific event/version
   - CREATED_BY: Was created by specific actor/version
   - INFLUENCED_BY: Was influenced at specific time
   - HIDDEN_FROM: Information hidden from entity (at creation time)
   - UNAWARE_OF: Entity doesn't know about target (at creation time)

**CRITICAL REQUIREMENTS**:
1. "reasoning" MUST be 1-5 words maximum (e.g., "still valid", "newer version", "no longer relevant")
2. relationship_type MUST be EXACTLY one of: RELATES_TO, DEPENDS_ON, SIMILAR_TO, PART_OF, AWARE_OF, MONITORS, CAUSED_BY, CREATED_BY, INFLUENCED_BY, HIDDEN_FROM, UNAWARE_OF
3. DO NOT use: MANAGES, REPORTS_TO, FRIENDS_WITH, or any other types not listed above
4. Respond with ONLY the JSON object below, NO additional text or explanations

Respond with JSON (only include actions to take, skip non-actions):
{{
  "existing": [
    {{"index": 1, "action": "keep_existing", "reasoning": "still valid"}},
    ...
  ],
  "new": [
    {{"candidate_index": 1, "relationship_type": "RELATES_TO", "reasoning": "same merchant"}},
    ...
  ]
}}
"""

        # Dynamic token calculation
        input_tokens = self._estimate_tokens(prompt)
        # Output: ~50 tokens per decision, plus overhead
        estimated_output = (len(existing_relationships) + len(candidate_states)) * 50 + 500
        max_tokens = max(
            4000, min(estimated_output * 2, 16000)
        )  # Between 4k-16k (generous to avoid truncation)

        logger.debug(
            f"Estimated input: {input_tokens} tokens, output: {estimated_output} tokens, using max_tokens={max_tokens}"
        )

        schema = {
            "name": "comprehensive_relationship_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "existing": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "action": {
                                    "type": "string",
                                    "enum": ["keep_existing", "update_to_newer", "remove_existing"],
                                },
                                "reasoning": {"type": "string", "maxLength": 80},
                            },
                            "required": ["index", "action", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                    "new": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_index": {"type": "integer"},
                                "relationship_type": {
                                    "type": "string",
                                    "enum": [
                                        # Dynamic relationships
                                        "RELATES_TO",
                                        "DEPENDS_ON",
                                        "SIMILAR_TO",
                                        "PART_OF",
                                        "AWARE_OF",
                                        "MONITORS",
                                        # Version-bound relationships
                                        "CAUSED_BY",
                                        "CREATED_BY",
                                        "INFLUENCED_BY",
                                        "HIDDEN_FROM",
                                        "UNAWARE_OF",
                                    ],
                                },
                                "reasoning": {"type": "string", "maxLength": 80},
                            },
                            "required": ["candidate_index", "relationship_type", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["existing", "new"],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are an expert knowledge graph curator. Maintain semantic accuracy and relevance. You MUST respond with ONLY valid JSON using the exact relationship types specified in the schema. Do NOT add any explanations or text outside the JSON object.",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=max_tokens,  # Dynamically calculated
            json_schema=schema,
        )

        # Call LLM with automatic retry
        result = await self._call_llm_with_retry(request, context_label="comprehensive analysis")
        if result is None:
            logger.error("Failed to get valid result after all retries for comprehensive analysis")
            # Fall back to keeping all existing relationships
            result = {"existing": [], "new": []}

        # Truncate reasoning fields if too long (LLM sometimes ignores maxLength)
        for item in result.get("existing", []):
            if "reasoning" in item and len(item["reasoning"]) > 80:
                item["reasoning"] = item["reasoning"][:77] + "..."
        for item in result.get("new", []):
            if "reasoning" in item and len(item["reasoning"]) > 80:
                item["reasoning"] = item["reasoning"][:77] + "..."

        all_decisions = []

        # Process existing relationship decisions
        for decision in result.get("existing", []):
            # Ensure index is int (LLM might return string)
            idx = int(decision.get("index", 0)) - 1
            if idx < 0 or idx >= len(existing_relationships):
                continue

            rel = existing_relationships[idx]
            action = decision["action"]

            decision_dict = {
                "action": action,
                "relationship": rel,
                "reasoning": decision["reasoning"],
            }

            if action == "update_to_newer" and rel.get("has_newer_version"):
                latest_id = rel.get("latest_id")
                if latest_id:
                    decision_dict["new_target_id"] = latest_id
                    decision_dict["new_version"] = rel.get("latest_version")

            all_decisions.append(decision_dict)

        # Add unchanged existing relationships as "keep_existing"
        analyzed_indices = {d["index"] - 1 for d in result.get("existing", [])}
        for idx, rel in enumerate(existing_relationships):
            if idx not in analyzed_indices:
                all_decisions.append(
                    {
                        "action": "keep_existing",
                        "relationship": rel,
                        "reasoning": "auto-kept (not analyzed)",
                    }
                )

        # Process new relationship decisions
        for decision in result.get("new", []):
            # Ensure candidate_index is int (LLM might return string)
            cidx = int(decision.get("candidate_index", 0)) - 1
            if cidx < 0 or cidx >= len(candidate_states):
                continue

            candidate = candidate_states[cidx]
            rel_type = decision.get("relationship_type", "")

            # Validate relationship type against enum
            from tracingrag.core.models.graph import RelationshipType

            try:
                RelationshipType(rel_type)
            except ValueError:
                logger.warning(
                    f"Comprehensive analysis: Invalid relationship type '{rel_type}' for candidate {cidx + 1}, skipping"
                )
                continue

            all_decisions.append(
                {
                    "action": "create_new",
                    "target_id": str(candidate.state.id),
                    "target_topic": candidate.state.topic,
                    "relationship_type": rel_type,
                    "reasoning": decision["reasoning"],
                    "confidence": candidate.score,
                }
            )

        return all_decisions

    async def create_initial_relationships(
        self,
        new_state: MemoryStateDB,
        similarity_threshold: float = 0.45,
    ) -> dict[str, Any]:
        """Create initial relationships for first-time memory creation

        Args:
            new_state: The newly created memory state (v1, no parent)
            similarity_threshold: Minimum embedding similarity score
            max_candidates: Maximum candidates to consider

        Returns:
            Dict with creation statistics: {created}
        """
        from tracingrag.services.retrieval import RetrievalService

        stats = {
            "created": 0,
        }

        logger.info(f"Creating initial relationships for {new_state.topic} v{new_state.version}")

        # Get total count of active states for comprehensive search
        from tracingrag.services.memory import MemoryService

        memory_service = MemoryService()
        total_states = await memory_service.count_states(storage_tier="active")
        search_limit = max(total_states, 1000)

        logger.info(f"   Total active states: {total_states}, search limit: {search_limit}")

        # Search for semantically related ACTIVE states
        retrieval_service = RetrievalService()
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=search_limit,  # Use all active states as limit
            score_threshold=similarity_threshold,
            latest_only=True,  # Only latest versions for initial relationships
            filter_conditions={"is_active": True},  # Only active states
        )

        # Filter out self
        candidates = [
            s
            for s in similar_states
            if s.state.id != new_state.id and s.state.topic != new_state.topic
        ]

        if not candidates:
            logger.info("   No candidates found for initial relationships")
            return stats

        logger.info(f"   Found {len(candidates)} candidates, analyzing with LLM...")

        # Build LLM prompt (simplified, no existing relationships to manage)
        candidates_desc = []
        for idx, result in enumerate(candidates):
            candidates_desc.append(
                f"C{idx + 1}. {result.state.topic} (similarity: {result.score:.2f})\n"
                f"    Content: {result.state.content[:300]}..."
            )

        candidates_text = "\n\n".join(candidates_desc)

        prompt = f"""**NEW MEMORY STATE** (first version, no prior relationships):
Topic: {new_state.topic}
Content: {new_state.content}

**CANDIDATE STATES** (semantically similar):
{candidates_text}

**TASK**: For EACH candidate, decide if a relationship should be created.

**Relationship types** (choose appropriately):

**Dynamic relationships** (update to latest version):
- RELATES_TO: General semantic relationship
- DEPENDS_ON: Current dependency/requirement
- SIMILAR_TO: Semantically similar content
- PART_OF: Hierarchical containment
- AWARE_OF: Entity knows about target
- MONITORS: Entity actively tracks target

**Version-bound relationships** (bind to specific version):
- CAUSED_BY: Was caused by specific event
- CREATED_BY: Was created by specific actor
- INFLUENCED_BY: Was influenced at specific time
- HIDDEN_FROM: Information hidden from entity
- UNAWARE_OF: Entity doesn't know about target

**CRITICAL REQUIREMENTS**:
1. "reasoning" MUST be 1-5 words maximum (e.g., "related merchant", "friend of Elena", "town guard")
2. relationship_type MUST be EXACTLY one of: RELATES_TO, DEPENDS_ON, SIMILAR_TO, PART_OF, AWARE_OF, MONITORS, CAUSED_BY, CREATED_BY, INFLUENCED_BY, HIDDEN_FROM, UNAWARE_OF
3. DO NOT use: MANAGES, REPORTS_TO, FRIENDS_WITH, WANTS_TO_POACH, or any other types not listed above
4. Respond with ONLY the JSON object below, NO additional text or explanations

Respond with JSON (only include relationships to create):
{{
  "new": [
    {{"candidate_index": 1, "relationship_type": "RELATES_TO", "reasoning": "same merchant"}},
    {{"candidate_index": 2, "relationship_type": "DEPENDS_ON", "reasoning": "Elena's friend"}},
    ...
  ]
}}
"""

        # Dynamic token calculation
        input_tokens = self._estimate_tokens(prompt)
        estimated_output = len(candidates) * 50 + 500  # ~50 tokens per decision
        max_tokens = max(2000, min(estimated_output * 2, 10000))  # Between 2k-10k (generous)

        logger.debug(
            f"Initial relationships: estimated input={input_tokens}, output={estimated_output}, max_tokens={max_tokens}"
        )

        schema = {
            "name": "initial_relationship_creation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "new": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_index": {"type": "integer"},
                                "relationship_type": {
                                    "type": "string",
                                    "enum": [
                                        # Dynamic relationships
                                        "RELATES_TO",
                                        "DEPENDS_ON",
                                        "SIMILAR_TO",
                                        "PART_OF",
                                        "AWARE_OF",
                                        "MONITORS",
                                        # Version-bound relationships
                                        "CAUSED_BY",
                                        "CREATED_BY",
                                        "INFLUENCED_BY",
                                        "HIDDEN_FROM",
                                        "UNAWARE_OF",
                                    ],
                                },
                                "reasoning": {"type": "string", "maxLength": 80},
                            },
                            "required": ["candidate_index", "relationship_type", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["new"],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are an expert knowledge graph curator. Carefully decide which relationships to create. You MUST respond with ONLY valid JSON using the exact relationship types specified in the schema. DO NOT add any explanations or text outside the JSON object.",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=max_tokens,  # Dynamically calculated
            json_schema=schema,
        )

        # Call LLM with automatic retry
        result = await self._call_llm_with_retry(request, context_label="initial relationships")
        if result is None:
            logger.error("Failed to get valid result after all retries for initial relationships")
            return stats

        new_relationships = result.get("new", [])

        # Truncate reasoning fields if too long (LLM sometimes ignores maxLength)
        for item in new_relationships:
            if "reasoning" in item and len(item["reasoning"]) > 80:
                item["reasoning"] = item["reasoning"][:77] + "..."

        # Create new relationships
        for decision in new_relationships:
            # Ensure candidate_index is int (LLM might return string)
            idx = int(decision.get("candidate_index", 0)) - 1
            if not (0 <= idx < len(candidates)):
                continue

            candidate = candidates[idx]
            rel_type = decision.get("relationship_type", "")

            # Validate relationship type against enum
            from tracingrag.core.models.graph import RelationshipType

            try:
                RelationshipType(rel_type)
            except ValueError:
                logger.warning(
                    f"Initial relationships: Invalid relationship type '{rel_type}' for candidate {idx + 1}, skipping"
                )
                continue

            # Create relationship
            await create_memory_relationship(
                source_id=new_state.id,
                target_id=candidate.state.id,
                relationship_type=rel_type,
                properties={
                    "confidence": candidate.score,
                    "reasoning": decision.get("reasoning", ""),
                    "initial_creation": True,
                },
            )

            stats["created"] += 1
            logger.info(f"   ✓ Created: {new_state.topic} -[{rel_type}]-> {candidate.state.topic}")

        logger.info(
            f"Initial relationship creation completed: {stats['created']} relationships created"
        )

        return stats


# Global instance
_relationship_manager = None


def get_relationship_manager() -> RelationshipManager:
    """Get or create the global relationship manager instance"""
    global _relationship_manager
    if _relationship_manager is None:
        _relationship_manager = RelationshipManager()
    return _relationship_manager
