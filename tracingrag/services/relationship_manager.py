"""Intelligent relationship management for memory state evolution

This module handles smart relationship updates when memory states evolve,
ensuring relationships stay current and relevant across large-scale knowledge graphs.
"""

import json
from typing import Any
from uuid import UUID

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import get_llm_client
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB
from tracingrag.storage.neo4j_client import (
    create_memory_relationship,
    delete_memory_relationship,
    get_latest_version_of_topic,
    get_parent_relationships,
)


class RelationshipManager:
    """Manages intelligent relationship updates during memory evolution"""

    def __init__(self):
        self.llm_client = get_llm_client()

    async def update_relationships_on_evolution(
        self,
        new_state: MemoryStateDB,
        parent_state_id: UUID,
        similarity_threshold: float = 0.3,
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

        print(
            f"[RelationshipManager] Analyzing relationships for {new_state.topic} v{new_state.version}"
        )
        print(f"   Parent has {len(parent_relationships)} existing relationships")

        # Step 2: Get total memory states count for full database search
        from tracingrag.storage.database import get_session

        async with get_session() as session:
            from sqlalchemy import func, select

            from tracingrag.storage.models import MemoryStateDB

            result = await session.execute(select(func.count(MemoryStateDB.id)))
            total_states = result.scalar() or 0

        search_limit = max(total_states, 1000)  # Use actual count, minimum 1000
        print(f"   Database has {total_states} states, using limit={search_limit} for full search")

        # Step 3: Full vector DB search (comprehensive)
        retrieval_service = RetrievalService()
        similar_states = await retrieval_service.semantic_search(
            query=new_state.content,
            limit=search_limit,  # Dynamic limit based on actual database size
            score_threshold=similarity_threshold,
            latest_only=False,  # Get all versions first, then deduplicate
        )

        # Filter out self and same topic
        candidate_states_raw = [
            s
            for s in similar_states
            if s.state.id != new_state.id and s.state.topic != new_state.topic
        ]

        print(f"   Found {len(candidate_states_raw)} semantically related states from vector DB")

        # Step 3: Deduplicate - keep only latest version of each topic
        topic_to_latest = {}
        for result in candidate_states_raw:
            topic = result.state.topic
            if topic not in topic_to_latest:
                topic_to_latest[topic] = result
            else:
                # Keep the one with higher version
                if result.state.version > topic_to_latest[topic].state.version:
                    topic_to_latest[topic] = result

        candidate_states = list(topic_to_latest.values())
        print(
            f"   After deduplication (latest version only): {len(candidate_states)} unique topics"
        )

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
                await create_memory_relationship(
                    source_id=new_state.id,
                    target_id=UUID(decision["new_target_id"]),
                    relationship_type=rel["rel_type"],
                    properties=rel.get("properties", {}),
                )
                stats["updated"] += 1
                print(
                    f"   ✓ Updated: {rel['target_topic']} v{rel['target_version']} → v{decision['new_version']}"
                )

            elif action == "remove_existing":
                # Don't create this old relationship
                stats["deleted"] += 1
                rel = decision["relationship"]
                print(f"   ✗ Removed: {rel['target_topic']} (no longer relevant)")

            elif action == "create_new":
                # Create new relationship to a candidate
                await create_memory_relationship(
                    source_id=new_state.id,
                    target_id=UUID(decision["target_id"]),
                    relationship_type=decision["relationship_type"],
                    properties={"confidence": decision.get("confidence", 0.8)},
                )
                stats["created"] += 1
                print(
                    f"   + Created: {decision['relationship_type']} → {decision['target_topic']}"
                )

        print(
            f"[RelationshipManager] Summary: {stats['kept']} kept, {stats['updated']} updated, "
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

        print(f"   Processing {len(candidate_states)} candidates in {total_batches} batch(es)")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(candidate_states))
            candidate_batch = candidate_states[start_idx:end_idx]

            print(
                f"   Batch {batch_idx + 1}/{total_batches}: analyzing {len(candidate_batch)} candidates + {len(existing_relationships)} existing"
            )

            batch_decisions = await self._llm_comprehensive_analysis(
                new_state=new_state,
                existing_relationships=existing_relationships if batch_idx == 0 else [],
                candidate_states=candidate_batch,
            )

            all_decisions.extend(batch_decisions)

        return all_decisions

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
        new_content_desc = f"**NEW STATE**\nTopic: {new_state.topic}\nContent: {new_state.content}\n"

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

Respond with JSON (only include actions to take, skip non-actions):
{{
  "existing": [
    {{"index": 1, "action": "keep_existing"|"update_to_newer"|"remove_existing", "reasoning": "brief"}},
    ...
  ],
  "new": [
    {{"candidate_index": 1, "relationship_type": "RELATED_TO"|"DEPENDS_ON"|"SIMILAR_TO"|"CAUSED_BY", "reasoning": "brief"}},
    ...
  ]
}}
"""

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
                                "reasoning": {"type": "string"},
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
                                "reasoning": {"type": "string"},
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
            max_tokens=3000,
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)
            result = json.loads(response.content)

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
                    all_decisions.append({
                        "action": "keep_existing",
                        "relationship": rel,
                        "reasoning": "auto-kept (not analyzed)",
                    })

            # Process new relationship decisions
            for decision in result.get("new", []):
                cidx = decision["candidate_index"] - 1
                if cidx < 0 or cidx >= len(candidate_states):
                    continue

                candidate = candidate_states[cidx]
                all_decisions.append({
                    "action": "create_new",
                    "target_id": str(candidate.state.id),
                    "target_topic": candidate.state.topic,
                    "relationship_type": decision["relationship_type"],
                    "reasoning": decision["reasoning"],
                    "confidence": candidate.score,
                })

            return all_decisions

        except Exception as e:
            # Fallback: keep all existing, create none new
            print(f"[RelationshipManager] LLM failed ({e}), keeping all existing relationships")
            return [
                {"action": "keep_existing", "relationship": rel, "reasoning": "fallback"}
                for rel in existing_relationships
            ]


# Global instance
_relationship_manager = None


def get_relationship_manager() -> RelationshipManager:
    """Get or create the global relationship manager instance"""
    global _relationship_manager
    if _relationship_manager is None:
        _relationship_manager = RelationshipManager()
    return _relationship_manager
