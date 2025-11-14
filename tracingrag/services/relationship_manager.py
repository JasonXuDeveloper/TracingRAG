"""Intelligent relationship management for memory state evolution

This module handles smart relationship updates when memory states evolve,
ensuring relationships stay current and relevant across large-scale knowledge graphs.
"""

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB
from tracingrag.storage.neo4j_client import (
    create_memory_relationship,
    get_latest_version_of_topic,
    get_parent_relationships,
)
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


class RelationshipManager:
    """Manages intelligent relationship updates during memory evolution"""

    def __init__(self):
        self.llm_client = get_llm_client()

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

        # Step 2: Get total memory states count from MemoryService (unified method)
        from tracingrag.services.memory import MemoryService

        memory_service = MemoryService()
        total_states = await memory_service.count_states(latest_only=False)

        search_limit = max(total_states, 1000)  # Use actual count, minimum 1000
        logger.info(
            f"   Database has {total_states} states, using limit={search_limit} for full search"
        )

        # Step 3: Comprehensive semantic search (find all relevant states)
        retrieval_service = RetrievalService()
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=search_limit,  # Full search to not miss any relationships
            score_threshold=similarity_threshold,  # Use configured threshold (default 0.4)
            latest_only=True,  # Only latest versions (MAJOR OPTIMIZATION - no manual dedup needed!)
        )

        # Filter out self and same topic (no deduplication needed with latest_only=True)
        candidate_states = [
            s
            for s in similar_states
            if s.state.id != new_state.id and s.state.topic != new_state.topic
        ]

        logger.info(f"   Found {len(candidate_states)} unique candidate topics")

        # Step 4: Build a map of topic -> latest candidate for version checking
        candidate_topic_map = {result.state.topic: result for result in candidate_states}

        # Step 5: Enrich parent relationships with version update info
        enriched_parent_rels = []
        for rel in parent_relationships:
            # Check if this topic appears in candidates (which are already latest versions)
            if rel["target_topic"] in candidate_topic_map:
                candidate = candidate_topic_map[rel["target_topic"]]
                if candidate.state.version > rel["target_version"]:
                    rel["has_newer_version"] = True
                    rel["latest_version"] = candidate.state.version
                    rel["latest_id"] = str(candidate.state.id)
                else:
                    rel["has_newer_version"] = False
            else:
                # Target topic not in candidates - need to query DB
                latest = await get_latest_version_of_topic(rel["target_topic"])
                if latest and latest["version"] > rel["target_version"]:
                    rel["has_newer_version"] = True
                    rel["latest_version"] = latest["version"]
                    rel["latest_id"] = latest["id"]
                else:
                    rel["has_newer_version"] = False
            enriched_parent_rels.append(rel)

        # Step 6: Use LLM to analyze all relationships (batched if needed)
        relationship_decisions = await self._llm_comprehensive_analysis_batched(
            new_state=new_state,
            existing_relationships=enriched_parent_rels,
            candidate_states=candidate_states,
            batch_size=settings.relationship_update_llm_batch_size,
        )

        # Step 5: Apply decisions to graph
        for decision in relationship_decisions:
            action = decision["action"]

            if action == "keep_existing":
                # Keep existing relationship as-is
                rel = decision["relationship"]
                await create_memory_relationship(
                    source_id=new_state.id,
                    target_id=UUID(rel["target_id"]),
                    relationship_type=rel["rel_type"],
                    properties=rel.get("properties", {}),
                )
                stats["kept"] += 1

            elif action == "update_to_newer":
                # Update existing relationship to newer version of target
                rel = decision["relationship"]
                # Only update if we have the new target ID
                if "new_target_id" in decision:
                    await create_memory_relationship(
                        source_id=new_state.id,
                        target_id=UUID(decision["new_target_id"]),
                        relationship_type=rel["rel_type"],
                        properties=rel.get("properties", {}),
                    )
                    stats["updated"] += 1
                    logger.info(
                        f"   ✓ Updated: {rel['target_topic']} v{rel['target_version']} → v{decision['new_version']}"
                    )
                else:
                    # Fallback: keep existing if no newer version available
                    await create_memory_relationship(
                        source_id=new_state.id,
                        target_id=UUID(rel["target_id"]),
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
                logger.error(f"   ✗ Removed: {rel['target_topic']} (no longer relevant)")

            elif action == "create_new":
                # Create new relationship to a candidate
                await create_memory_relationship(
                    source_id=new_state.id,
                    target_id=UUID(decision["target_id"]),
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
        batch_size: int = 30,
    ) -> list[dict[str, Any]]:
        """Analyze relationships in batches to handle large numbers of candidates

        Args:
            new_state: The new memory state
            existing_relationships: Enriched parent relationships
            candidate_states: All deduplicated candidate states
            batch_size: Max candidates per LLM call

        Returns:
            Combined list of all relationship decisions
        """
        all_decisions = []

        # Always analyze existing relationships (usually small number)
        # Split candidates into batches
        total_batches = (len(candidate_states) + batch_size - 1) // batch_size

        logger.info(
            f"   Processing {len(candidate_states)} candidates in {total_batches} batch(es)"
        )

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(candidate_states))
            candidate_batch = candidate_states[start_idx:end_idx]

            logger.info(
                f"   Batch {batch_idx + 1}/{total_batches}: analyzing {len(candidate_batch)} candidates + {len(existing_relationships)} existing"
            )

            batch_decisions = await self._llm_comprehensive_analysis(
                new_state=new_state,
                existing_relationships=existing_relationships if batch_idx == 0 else [],
                candidate_states=candidate_batch,
            )

            all_decisions.extend(batch_decisions)

        return all_decisions

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 3.5 characters (mixed English/Chinese)
        return len(text) // 3

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
        # Fetch content for existing relationship targets
        async with get_session() as session:
            existing_contents = {}
            for rel in existing_relationships:
                target = await session.get(MemoryStateDB, UUID(rel["target_id"]))
                if target:
                    existing_contents[rel["target_id"]] = target.content[:250]

        # Build prompt sections
        new_content_desc = (
            f"**NEW STATE**\nTopic: {new_state.topic}\nContent: {new_state.content}\n"
        )

        # Section 1: Existing relationships to review
        existing_rel_desc = []
        for idx, rel in enumerate(existing_relationships):
            content = existing_contents.get(rel["target_id"], "N/A")
            version_note = ""
            if rel.get("has_newer_version"):
                version_note = f" → NEWER VERSION v{rel['latest_version']} available"

            existing_rel_desc.append(
                f"E{idx + 1}. [{rel['rel_type']}] {rel['target_topic']} v{rel['target_version']}{version_note}\n"
                f"    Content: {content}"
            )

        existing_text = "\n\n".join(existing_rel_desc) if existing_rel_desc else "None"

        # Section 2: ALL candidate states for potential new relationships
        # No limit - send all candidates in this batch
        candidate_desc = []
        for idx, result in enumerate(candidate_states):
            candidate_desc.append(
                f"C{idx + 1}. {result.state.topic} v{result.state.version} (sim: {result.score:.2f})\n"
                f"    Content: {result.state.content[:200]}"
            )

        candidates_text = "\n\n".join(candidate_desc) if candidate_desc else "None"

        # Comprehensive prompt
        prompt = f"""{new_content_desc}

**TASK**: Analyze ALL relationships for the new state based on its content.

**EXISTING RELATIONSHIPS** (from parent state - currently pointing to these):
{existing_text}

**CANDIDATE STATES** (all semantically related states found via vector search):
{candidates_text}

**INSTRUCTIONS**:

1. For EACH EXISTING relationship (E1, E2, ...):
   - "update_to_newer": If a newer version is available AND the relationship is still semantically valid
   - "keep_existing": If the relationship is still valid with current version
   - "remove_existing": If the new state's content no longer relates to this target

2. For EACH CANDIDATE state (C1, C2, ...):
   - "create_new": If the new state should have a relationship to this candidate (specify type)
   - Skip if not relevant

3. Relationship types:
   - RELATED_TO: General semantic relationship
   - DEPENDS_ON: The new state depends on/requires the target
   - SIMILAR_TO: Very similar content or concepts
   - CAUSED_BY: The new state is a result/effect of the target

**CRITICAL REQUIREMENT**: "reasoning" MUST be 1-5 words maximum. NO sentences. Examples:
- GOOD: "still valid"
- GOOD: "newer version available"
- GOOD: "no longer relevant"
- BAD: "The relationship is still semantically valid and relevant"

Respond with JSON (only include actions to take, skip non-actions):
{{
  "existing": [
    {{"index": 1, "action": "keep_existing", "reasoning": "still valid"}},
    ...
  ],
  "new": [
    {{"candidate_index": 1, "relationship_type": "RELATED_TO", "reasoning": "same merchant"}},
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
                                    "enum": ["RELATED_TO", "DEPENDS_ON", "SIMILAR_TO", "CAUSED_BY"],
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
            system_prompt="You are an expert knowledge graph curator. Maintain semantic accuracy and relevance.",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.2,
            max_tokens=max_tokens,  # Dynamically calculated
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)

            # Try to parse JSON with better error handling
            try:
                result = json.loads(response.content, strict=False)
            except json.JSONDecodeError as json_err:
                # Log the problematic JSON for debugging
                logger.error(f"JSON parsing failed: {json_err}")
                logger.error(f"Problematic JSON (first 1000 chars): {response.content[:1000]}")
                logger.error(f"Problematic JSON (last 500 chars): {response.content[-500:]}")

                # Try to fix common JSON issues
                fixed_content = response.content

                # 1. Remove leading text before JSON
                if "{" in fixed_content:
                    json_start = fixed_content.find("{")
                    fixed_content = fixed_content[json_start:]

                # 2. Remove trailing text after JSON
                if "}" in fixed_content:
                    json_end = fixed_content.rfind("}") + 1
                    fixed_content = fixed_content[:json_end]

                # 3. Fix trailing commas
                fixed_content = fixed_content.replace(",]", "]").replace(",}", "}")

                # 4. Try to fix unterminated strings by adding closing quotes and brackets
                # Count opening and closing braces/brackets
                open_braces = fixed_content.count("{")
                close_braces = fixed_content.count("}")
                open_brackets = fixed_content.count("[")
                close_brackets = fixed_content.count("]")

                # If JSON is incomplete, try to close it
                if open_braces > close_braces or open_brackets > close_brackets:
                    logger.warning(
                        f"Detected incomplete JSON: braces={open_braces}/{close_braces}, brackets={open_brackets}/{close_brackets}"
                    )
                    # Add missing closing quotes if string is unterminated
                    if fixed_content.count('"') % 2 != 0:
                        fixed_content += '"'
                    # Close missing brackets/braces
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
                    # If still fails, fall back to keeping all existing
                    raise json_err

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
                idx = decision["index"] - 1
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
                    decision_dict["new_target_id"] = rel["latest_id"]
                    decision_dict["new_version"] = rel["latest_version"]

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
                cidx = decision["candidate_index"] - 1
                if cidx < 0 or cidx >= len(candidate_states):
                    continue

                candidate = candidate_states[cidx]
                all_decisions.append(
                    {
                        "action": "create_new",
                        "target_id": str(candidate.state.id),
                        "target_topic": candidate.state.topic,
                        "relationship_type": decision["relationship_type"],
                        "reasoning": decision["reasoning"],
                        "confidence": candidate.score,
                    }
                )

            return all_decisions

        except Exception as e:
            # Fallback: keep all existing, create none new
            logger.error(f"LLM failed ({e}), keeping all existing relationships")
            return [
                {"action": "keep_existing", "relationship": rel, "reasoning": "fallback"}
                for rel in existing_relationships
            ]

    async def create_initial_relationships(
        self,
        new_state: MemoryStateDB,
        similarity_threshold: float = 0.4,
        max_candidates: int = 20,
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

        # Search for semantically related states
        retrieval_service = RetrievalService()
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=max_candidates,
            score_threshold=similarity_threshold,
            latest_only=True,  # Only latest versions for initial relationships
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

**Relationship types** (choose one):
   - RELATED_TO: General semantic relationship
   - DEPENDS_ON: The new state depends on/requires the target
   - SIMILAR_TO: Very similar content or concepts
   - CAUSED_BY: The new state is a result/effect of the target

**CRITICAL REQUIREMENT**: "reasoning" MUST be 1-5 words maximum. NO sentences. Examples:
- GOOD: "related merchant"
- GOOD: "friend of Elena"
- GOOD: "town guard"
- BAD: "Marcus is suspicious of Silas and has feelings for Elena"

Respond with JSON (only include relationships to create):
{{
  "new": [
    {{"candidate_index": 1, "relationship_type": "RELATED_TO", "reasoning": "same merchant"}},
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
                                    "enum": ["RELATED_TO", "DEPENDS_ON", "SIMILAR_TO", "CAUSED_BY"],
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
            system_prompt="You are an expert knowledge graph curator. Carefully decide which relationships to create.",
            user_message=prompt,
            context="",
            model=settings.analysis_model,
            temperature=0.2,
            max_tokens=max_tokens,  # Dynamically calculated
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)

            # Check if response is empty
            if not response.content or not response.content.strip():
                logger.error("Empty LLM response for initial relationships")
                return stats

            # Parse JSON with error handling
            try:
                result = json.loads(response.content, strict=False)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed: {json_err}")
                logger.error(f"Response content (first 1000 chars): {response.content[:1000]}")
                logger.error(f"Response content (last 500 chars): {response.content[-500:]}")

                # Try fixing common issues
                fixed_content = response.content

                # 1. Remove leading text before JSON (e.g., "Here is my analysis:")
                if "{" in fixed_content:
                    json_start = fixed_content.find("{")
                    fixed_content = fixed_content[json_start:]

                # 2. Remove trailing text after JSON
                if "}" in fixed_content:
                    json_end = fixed_content.rfind("}") + 1
                    fixed_content = fixed_content[:json_end]

                # 3. Fix trailing commas
                fixed_content = fixed_content.replace(",]", "]").replace(",}", "}")

                # 4. Try to fix unterminated strings by adding closing quotes and brackets
                open_braces = fixed_content.count("{")
                close_braces = fixed_content.count("}")
                open_brackets = fixed_content.count("[")
                close_brackets = fixed_content.count("]")

                if open_braces > close_braces or open_brackets > close_brackets:
                    logger.warning(
                        f"Incomplete JSON detected: braces={open_braces}/{close_braces}, brackets={open_brackets}/{close_brackets}"
                    )
                    # Add missing closing quotes if string is unterminated
                    if fixed_content.count('"') % 2 != 0:
                        fixed_content += '"'
                    # Close missing brackets/braces
                    while open_brackets > close_brackets:
                        fixed_content += "]"
                        close_brackets += 1
                    while open_braces > close_braces:
                        fixed_content += "}"
                        close_braces += 1

                try:
                    result = json.loads(fixed_content, strict=False)
                    logger.info("Successfully parsed after cleaning JSON")
                except json.JSONDecodeError:
                    logger.error("Failed to fix JSON, returning empty stats")
                    return stats

            new_relationships = result.get("new", [])

            # Truncate reasoning fields if too long (LLM sometimes ignores maxLength)
            for item in new_relationships:
                if "reasoning" in item and len(item["reasoning"]) > 80:
                    item["reasoning"] = item["reasoning"][:77] + "..."

            # Create new relationships
            for decision in new_relationships:
                idx = decision.get("candidate_index", 0) - 1
                if not (0 <= idx < len(candidates)):
                    continue

                candidate = candidates[idx]
                rel_type = decision.get("relationship_type")

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
                logger.info(
                    f"   ✓ Created: {new_state.topic} -[{rel_type}]-> {candidate.state.topic}"
                )

            logger.info(
                f"Initial relationship creation completed: {stats['created']} relationships created"
            )

        except Exception as e:
            logger.error(f"Failed to create initial relationships: {e}")

        return stats


# Global instance
_relationship_manager = None


def get_relationship_manager() -> RelationshipManager:
    """Get or create the global relationship manager instance"""
    global _relationship_manager
    if _relationship_manager is None:
        _relationship_manager = RelationshipManager()
    return _relationship_manager
