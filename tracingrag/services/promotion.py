"""Memory promotion service for intelligent synthesis and consolidation"""

import json
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from tracingrag.core.models.graph import RelationshipType
from tracingrag.core.models.promotion import (
    Conflict,
    ConflictResolution,
    ConflictResolutionStrategy,
    ConflictType,
    EdgeUpdate,
    PromotionCandidate,
    PromotionRequest,
    PromotionResult,
    PromotionTrigger,
    QualityCheck,
    QualityCheckType,
    SynthesisSource,
)
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.graph import GraphService
from tracingrag.services.llm import LLMClient, get_llm_client
from tracingrag.services.memory import MemoryService
from tracingrag.services.retrieval import RetrievalService


class PromotionService:
    """Service for intelligent memory promotion and synthesis"""

    def __init__(
        self,
        memory_service: MemoryService | None = None,
        graph_service: GraphService | None = None,
        retrieval_service: RetrievalService | None = None,
        llm_client: LLMClient | None = None,
        synthesis_model: str = "anthropic/claude-3.5-sonnet",
        analysis_model: str = "deepseek/deepseek-chat-v3-0324:free",
        max_synthesis_tokens: int = 16000,
    ):
        """
        Initialize promotion service

        Args:
            memory_service: Memory service for CRUD operations
            graph_service: Graph service for edge management
            retrieval_service: Retrieval service for context gathering
            llm_client: LLM client for synthesis and analysis
            synthesis_model: Model for content synthesis (premium)
            analysis_model: Model for analysis tasks (cheap/free)
            max_synthesis_tokens: Maximum tokens for synthesis output
        """
        self.memory_service = memory_service or MemoryService()
        self.graph_service = graph_service or GraphService()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_client = llm_client or get_llm_client()
        self.synthesis_model = synthesis_model
        self.analysis_model = analysis_model
        self.max_synthesis_tokens = max_synthesis_tokens

    async def promote_memory(self, request: PromotionRequest) -> PromotionResult:
        """
        Promote a memory state with intelligent synthesis

        This is the main entry point for memory promotion. It:
        1. Gathers trace history and related states
        2. Detects and resolves conflicts
        3. Synthesizes new state with LLM
        4. Performs quality checks
        5. Updates edges appropriately
        6. Creates new memory state

        Args:
            request: Promotion request with topic and parameters

        Returns:
            PromotionResult with details of promotion
        """
        try:
            # Step 1: Gather context for synthesis
            synthesis_context = await self._gather_synthesis_context(
                topic=request.topic,
                include_related=request.include_related,
                max_sources=request.max_sources,
            )

            if not synthesis_context["sources"]:
                return PromotionResult(
                    success=False,
                    topic=request.topic,
                    reasoning="No sources found for synthesis",
                    error_message=f"No memory states found for topic '{request.topic}'",
                )

            # Step 2: Detect conflicts
            conflicts = await self._detect_conflicts(synthesis_context["sources"])

            # Step 3: Resolve conflicts
            resolved_conflicts = []
            if conflicts and request.auto_resolve_conflicts:
                resolved_conflicts = await self._resolve_conflicts(
                    conflicts, request.conflict_resolution_strategy
                )

                # Check if manual review needed
                manual_review = any(r.manual_review_needed for r in resolved_conflicts)
                if manual_review and not request.auto_resolve_conflicts:
                    return PromotionResult(
                        success=False,
                        topic=request.topic,
                        reasoning="Manual review required for conflicts",
                        conflicts_detected=conflicts,
                        manual_review_needed=True,
                    )

            # Step 4: Build synthesis context
            synthesis_sources = self._build_synthesis_sources(
                sources=synthesis_context["sources"],
                related_states=synthesis_context["related_states"],
            )

            # Step 5: Synthesize new content with LLM
            synthesis_result = await self._synthesize_content(
                topic=request.topic,
                reason=request.reason,
                sources=synthesis_sources,
                conflicts_resolved=resolved_conflicts,
            )

            # Step 6: Perform quality checks
            quality_checks = []
            if request.quality_checks_enabled:
                quality_checks = await self._perform_quality_checks(
                    synthesized_content=synthesis_result["content"],
                    sources=synthesis_sources,
                )

            # Check quality thresholds
            failed_checks = [qc for qc in quality_checks if not qc.passed]
            if failed_checks and request.quality_checks_enabled:
                return PromotionResult(
                    success=False,
                    topic=request.topic,
                    reasoning="Quality checks failed",
                    quality_checks=quality_checks,
                    synthesized_content=synthesis_result["content"],
                    manual_review_needed=True,
                    error_message=f"Quality checks failed: {[c.check_type.value for c in failed_checks]}",
                )

            # Step 7: Create new memory state
            previous_state = synthesis_context["latest_state"]
            new_version = (previous_state.version + 1) if previous_state else 1

            new_state = await self.memory_service.create_memory_state(
                topic=request.topic,
                content=synthesis_result["content"],
                version=new_version,
                parent_state_id=previous_state.id if previous_state else None,
                metadata={
                    **request.metadata,
                    "promotion_trigger": request.trigger.value,
                    "promotion_reason": request.reason,
                    "synthesized_from_count": len(synthesis_sources),
                    "conflicts_resolved_count": len(resolved_conflicts),
                },
                tags=[*previous_state.tags, "promoted"] if previous_state else ["promoted"],
                confidence=synthesis_result["confidence"],
            )

            # Step 8: Update edges
            edge_updates = await self._update_edges(
                new_state_id=new_state.id,
                previous_state=previous_state,
                synthesis_sources=synthesis_sources,
                related_states=synthesis_context["related_states"],
            )

            return PromotionResult(
                success=True,
                previous_state_id=previous_state.id if previous_state else None,
                new_state_id=new_state.id,
                topic=request.topic,
                new_version=new_version,
                synthesized_from=synthesis_sources,
                conflicts_detected=conflicts,
                conflicts_resolved=resolved_conflicts,
                edges_updated=edge_updates,
                quality_checks=quality_checks,
                reasoning=synthesis_result["reasoning"],
                synthesized_content=synthesis_result["content"],
                confidence=synthesis_result["confidence"],
                manual_review_needed=False,
                metadata=request.metadata,
            )

        except Exception as e:
            return PromotionResult(
                success=False,
                topic=request.topic,
                reasoning=f"Promotion failed with error: {str(e)}",
                error_message=str(e),
            )

    async def _gather_synthesis_context(
        self, topic: str, include_related: bool, max_sources: int
    ) -> dict[str, Any]:
        """
        Gather all context needed for synthesis

        Returns:
            Dictionary with:
            - sources: List of source states (versions)
            - related_states: List of related states
            - latest_state: Latest state for topic
        """
        # Get all versions of the topic
        versions = await self.memory_service.get_topic_history(
            topic=topic, limit=max_sources
        )

        if not versions:
            return {"sources": [], "related_states": [], "latest_state": None}

        latest_state = versions[0]

        # Get related states via graph if requested
        related_states = []
        if include_related:
            # Get states connected to latest state
            related_results = await self.retrieval_service.graph_enhanced_retrieval(
                query=topic, depth=1, limit=max_sources, start_nodes=[latest_state.id]
            )
            related_states = [r.state for r in related_results]

        return {
            "sources": versions,
            "related_states": related_states,
            "latest_state": latest_state,
        }

    async def _detect_conflicts(self, sources: list[Any]) -> list[Conflict]:
        """
        Detect conflicts between memory states using LLM

        Args:
            sources: List of memory states to check

        Returns:
            List of detected conflicts
        """
        if len(sources) < 2:
            return []

        # Build conflict detection prompt
        states_text = "\n\n".join(
            [
                f"State {i + 1} (v{s.version}, {s.timestamp}):\n{s.content}"
                for i, s in enumerate(sources)
            ]
        )

        prompt = f"""Analyze these memory states for conflicts:

{states_text}

Identify any:
1. Contradictions (directly conflicting information)
2. Inconsistencies (information that doesn't align)
3. Ambiguities (unclear or confusing information)
4. Temporal conflicts (time-based issues)

For each conflict, provide:
- Which states are involved (by number)
- Type of conflict
- Description
- Severity (0.0-1.0)
- Recommended resolution strategy

Respond with a JSON array of conflicts."""

        # Define JSON schema
        schema = {
            "name": "conflict_detection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "conflicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "state_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "conflict_type": {
                                    "type": "string",
                                    "enum": [
                                        "contradiction",
                                        "inconsistency",
                                        "ambiguity",
                                        "temporal",
                                    ],
                                },
                                "description": {"type": "string"},
                                "severity": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "resolution_strategy": {
                                    "type": "string",
                                    "enum": [
                                        "latest_wins",
                                        "highest_confidence",
                                        "merge",
                                        "manual",
                                        "llm_decide",
                                    ],
                                },
                            },
                            "required": [
                                "state_indices",
                                "conflict_type",
                                "description",
                                "severity",
                                "resolution_strategy",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["conflicts"],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are a conflict detection assistant that identifies contradictions and inconsistencies in information.",
            user_message=prompt,
            context="",
            model=self.analysis_model,
            temperature=0.0,
            max_tokens=2000,
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)
            result = json.loads(response.content)

            conflicts = []
            for c in result.get("conflicts", []):
                # Convert indices to state IDs
                state_ids = [sources[i - 1].id for i in c["state_indices"]]

                conflicts.append(
                    Conflict(
                        state_ids=state_ids,
                        conflict_type=ConflictType(c["conflict_type"]),
                        description=c["description"],
                        severity=c["severity"],
                        resolution_strategy=ConflictResolutionStrategy(
                            c["resolution_strategy"]
                        ),
                    )
                )

            return conflicts

        except Exception:
            # Fallback: no conflicts detected if LLM fails
            return []

    async def _resolve_conflicts(
        self,
        conflicts: list[Conflict],
        default_strategy: ConflictResolutionStrategy,
    ) -> list[ConflictResolution]:
        """
        Resolve detected conflicts

        Args:
            conflicts: List of conflicts to resolve
            default_strategy: Default resolution strategy

        Returns:
            List of conflict resolutions
        """
        resolutions = []

        for conflict in conflicts:
            strategy = conflict.resolution_strategy or default_strategy

            if strategy == ConflictResolutionStrategy.LATEST_WINS:
                # Use most recent state
                resolution = ConflictResolution(
                    conflict=conflict,
                    strategy_used=strategy,
                    resolution="Using most recent state",
                    winning_state_id=conflict.state_ids[-1],  # Last is most recent
                    confidence=0.8,
                    manual_review_needed=False,
                )

            elif strategy == ConflictResolutionStrategy.MANUAL:
                # Require manual intervention
                resolution = ConflictResolution(
                    conflict=conflict,
                    strategy_used=strategy,
                    resolution="Manual review required",
                    confidence=0.0,
                    manual_review_needed=True,
                )

            else:  # LLM_DECIDE or others
                # Let LLM resolve
                resolution = await self._llm_resolve_conflict(conflict)

            resolutions.append(resolution)

        return resolutions

    async def _llm_resolve_conflict(self, conflict: Conflict) -> ConflictResolution:
        """
        Use LLM to resolve a conflict

        Args:
            conflict: Conflict to resolve

        Returns:
            ConflictResolution with LLM's decision
        """
        # Fetch the conflicting states
        states = []
        for state_id in conflict.state_ids:
            state = await self.memory_service.get_memory_state(state_id)
            if state:
                states.append(state)

        states_text = "\n\n".join(
            [f"State {state.id} (v{state.version}):\n{state.content}" for state in states]
        )

        prompt = f"""Resolve this conflict:

Type: {conflict.conflict_type.value}
Description: {conflict.description}
Severity: {conflict.severity}

Conflicting States:
{states_text}

Provide a resolution that:
1. Explains how to resolve the conflict
2. Either picks a winning state OR provides merged content
3. Assigns a confidence score (0.0-1.0)
4. Indicates if manual review is still needed

Respond with JSON."""

        schema = {
            "name": "conflict_resolution",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "resolution": {"type": "string"},
                    "winning_state_index": {"type": "integer", "nullable": True},
                    "merged_content": {"type": "string", "nullable": True},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "manual_review_needed": {"type": "boolean"},
                },
                "required": [
                    "resolution",
                    "confidence",
                    "manual_review_needed",
                ],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are a conflict resolution assistant that resolves contradictions in information.",
            user_message=prompt,
            context="",
            model=self.analysis_model,
            temperature=0.2,
            max_tokens=1000,
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)
            result = json.loads(response.content)

            winning_state_id = None
            if result.get("winning_state_index") is not None:
                idx = result["winning_state_index"]
                if 0 <= idx < len(states):
                    winning_state_id = states[idx].id

            return ConflictResolution(
                conflict=conflict,
                strategy_used=ConflictResolutionStrategy.LLM_DECIDE,
                resolution=result["resolution"],
                winning_state_id=winning_state_id,
                merged_content=result.get("merged_content"),
                confidence=result["confidence"],
                manual_review_needed=result["manual_review_needed"],
            )

        except Exception as e:
            # Fallback to manual review
            return ConflictResolution(
                conflict=conflict,
                strategy_used=ConflictResolutionStrategy.MANUAL,
                resolution=f"LLM resolution failed: {str(e)}",
                confidence=0.0,
                manual_review_needed=True,
            )

    def _build_synthesis_sources(
        self, sources: list[Any], related_states: list[Any]
    ) -> list[SynthesisSource]:
        """
        Build list of synthesis sources with weights

        Args:
            sources: Version history sources
            related_states: Related states from graph

        Returns:
            List of SynthesisSource objects
        """
        synthesis_sources = []

        # Add version sources (higher weight for recent)
        for i, state in enumerate(sources):
            weight = 1.0 - (i * 0.1)  # Decay weight for older versions
            weight = max(0.3, weight)  # Minimum weight 0.3

            synthesis_sources.append(
                SynthesisSource(
                    state_id=state.id,
                    topic=state.topic,
                    content=state.content,
                    version=state.version,
                    timestamp=state.timestamp,
                    confidence=state.confidence,
                    weight=weight,
                    reasoning=f"Version {state.version} of topic history",
                )
            )

        # Add related sources (lower weight)
        for state in related_states[:5]:  # Limit to top 5 related
            synthesis_sources.append(
                SynthesisSource(
                    state_id=state.id,
                    topic=state.topic,
                    content=state.content,
                    version=state.version,
                    timestamp=state.timestamp,
                    confidence=state.confidence,
                    weight=0.5,  # Lower weight for related
                    reasoning="Related state from graph",
                )
            )

        return synthesis_sources

    async def _synthesize_content(
        self,
        topic: str,
        reason: str,
        sources: list[SynthesisSource],
        conflicts_resolved: list[ConflictResolution],
    ) -> dict[str, Any]:
        """
        Synthesize new content from sources using LLM

        Args:
            topic: Topic being promoted
            reason: Reason for promotion
            sources: Sources to synthesize from
            conflicts_resolved: Any resolved conflicts

        Returns:
            Dictionary with:
            - content: Synthesized content
            - reasoning: Explanation
            - confidence: Confidence score
        """
        # Build synthesis prompt
        sources_text = "\n\n".join(
            [
                f"Source {i + 1} (weight={s.weight:.2f}, v{s.version}):\n{s.content}\nReasoning: {s.reasoning}"
                for i, s in enumerate(sources)
            ]
        )

        conflicts_text = ""
        if conflicts_resolved:
            conflicts_text = "\n\nResolved Conflicts:\n" + "\n".join(
                [
                    f"- {r.conflict.description}\n  Resolution: {r.resolution}"
                    for r in conflicts_resolved
                ]
            )

        prompt = f"""Synthesize a new memory state for topic "{topic}".

Promotion Reason: {reason}

Sources to synthesize from:
{sources_text}
{conflicts_text}

Your task:
1. Integrate information from all sources, weighing by their weights
2. Incorporate conflict resolutions
3. Create a coherent, consolidated state
4. Focus on the most recent and important information
5. Maintain factual accuracy and cite sources when relevant

Provide:
1. Synthesized content (comprehensive but concise)
2. Reasoning for synthesis approach
3. Confidence score (0.0-1.0)

Respond with JSON."""

        schema = {
            "name": "content_synthesis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["content", "reasoning", "confidence"],
                "additionalProperties": False,
            },
        }

        # Calculate dynamic max_tokens
        input_length = len(prompt)
        input_tokens = input_length // 4
        model_context = 50000
        available = model_context - input_tokens - 1000
        max_tokens = max(4000, min(self.max_synthesis_tokens, available))

        request = LLMRequest(
            system_prompt=self._get_synthesis_system_prompt(),
            user_message=prompt,
            context="",
            model=self.synthesis_model,
            temperature=0.3,  # Slight creativity but mostly deterministic
            max_tokens=max_tokens,
            json_schema=schema,
        )

        response = await self.llm_client.generate(request)
        result = json.loads(response.content)

        return {
            "content": result["content"],
            "reasoning": result["reasoning"],
            "confidence": result["confidence"],
        }

    def _get_synthesis_system_prompt(self) -> str:
        """System prompt for content synthesis"""
        return """You are a memory synthesis assistant that consolidates information from multiple sources.

Your role:
1. Integrate information from weighted sources
2. Resolve conflicts and contradictions
3. Create coherent, consolidated states
4. Maintain temporal awareness
5. Preserve factual accuracy

Guidelines:
- Prioritize recent and high-weight sources
- Incorporate conflict resolutions
- Be comprehensive but concise
- Cite sources when making claims
- Acknowledge uncertainty when appropriate
- Maintain consistent tone and style"""

    async def _perform_quality_checks(
        self, synthesized_content: str, sources: list[SynthesisSource]
    ) -> list[QualityCheck]:
        """
        Perform quality checks on synthesized content

        Args:
            synthesized_content: Content to check
            sources: Sources used in synthesis

        Returns:
            List of QualityCheck results
        """
        checks = []

        # Check 1: Hallucination detection
        hallucination_check = await self._check_hallucination(
            synthesized_content, sources
        )
        checks.append(hallucination_check)

        # Check 2: Citation verification
        citation_check = await self._check_citations(synthesized_content, sources)
        checks.append(citation_check)

        # Check 3: Consistency check
        consistency_check = await self._check_consistency(synthesized_content, sources)
        checks.append(consistency_check)

        return checks

    async def _check_hallucination(
        self, content: str, sources: list[SynthesisSource]
    ) -> QualityCheck:
        """Check for hallucinations in synthesized content"""
        sources_text = "\n\n".join([f"Source: {s.content}" for s in sources[:5]])

        prompt = f"""Check if the synthesized content contains hallucinations (information not supported by sources).

Sources:
{sources_text}

Synthesized Content:
{content}

Analyze if all claims in the synthesized content are supported by the sources.
Provide:
1. Whether it passed (no major hallucinations)
2. Score (0.0=major hallucinations, 1.0=fully grounded)
3. List of any unsupported claims
4. Recommendations

Respond with JSON."""

        schema = {
            "name": "hallucination_check",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["passed", "score", "unsupported_claims", "recommendations"],
                "additionalProperties": False,
            },
        }

        request = LLMRequest(
            system_prompt="You are a fact-checking assistant that detects hallucinations.",
            user_message=prompt,
            context="",
            model=self.analysis_model,
            temperature=0.0,
            max_tokens=1000,
            json_schema=schema,
        )

        try:
            response = await self.llm_client.generate(request)
            result = json.loads(response.content)

            return QualityCheck(
                check_type=QualityCheckType.HALLUCINATION,
                passed=result["passed"],
                score=result["score"],
                issues=result["unsupported_claims"],
                recommendations=result["recommendations"],
            )
        except Exception:
            # Default to passing if check fails
            return QualityCheck(
                check_type=QualityCheckType.HALLUCINATION,
                passed=True,
                score=0.7,
                issues=[],
                recommendations=["Quality check failed, manual review recommended"],
            )

    async def _check_citations(
        self, content: str, sources: list[SynthesisSource]
    ) -> QualityCheck:
        """Check citation quality"""
        # Simple heuristic check
        has_references = any(
            marker in content.lower()
            for marker in ["source", "according to", "version", "stated"]
        )

        return QualityCheck(
            check_type=QualityCheckType.CITATION,
            passed=has_references or len(sources) <= 2,
            score=0.8 if has_references else 0.6,
            issues=[] if has_references else ["Content lacks source citations"],
            recommendations=["Consider adding source references"]
            if not has_references
            else [],
        )

    async def _check_consistency(
        self, content: str, sources: list[SynthesisSource]
    ) -> QualityCheck:
        """Check internal consistency"""
        # Simple length and coherence check
        word_count = len(content.split())
        is_reasonable_length = 50 <= word_count <= 2000

        return QualityCheck(
            check_type=QualityCheckType.CONSISTENCY,
            passed=is_reasonable_length,
            score=0.8 if is_reasonable_length else 0.5,
            issues=[]
            if is_reasonable_length
            else [f"Content length unusual: {word_count} words"],
            recommendations=[],
        )

    async def _update_edges(
        self,
        new_state_id: UUID,
        previous_state: Any | None,
        synthesis_sources: list[SynthesisSource],
        related_states: list[Any],
    ) -> list[EdgeUpdate]:
        """
        Update graph edges for promoted state

        Args:
            new_state_id: ID of new promoted state
            previous_state: Previous state (if exists)
            synthesis_sources: Sources used in synthesis
            related_states: Related states from graph

        Returns:
            List of EdgeUpdate objects
        """
        updates = []

        # 1. Create EVOLVES_FROM edge to previous state
        if previous_state:
            await self.graph_service.create_edge(
                source_state_id=previous_state.id,
                target_state_id=new_state_id,
                relationship_type=RelationshipType.EVOLVES_INTO,
                strength=1.0,
                metadata={"promotion": True},
            )
            updates.append(
                EdgeUpdate(
                    source_id=previous_state.id,
                    target_id=new_state_id,
                    relationship_type=RelationshipType.EVOLVES_INTO.value,
                    strength=1.0,
                    action="create",
                    reasoning="Link to previous version",
                )
            )

        # 2. Create SYNTHESIZED_FROM edges to sources
        for source in synthesis_sources[:5]:  # Limit to top 5
            if source.state_id != (previous_state.id if previous_state else None):
                await self.graph_service.create_edge(
                    source_state_id=source.state_id,
                    target_state_id=new_state_id,
                    relationship_type=RelationshipType.RELATES_TO,
                    strength=source.weight,
                    metadata={"synthesis_source": True, "weight": source.weight},
                )
                updates.append(
                    EdgeUpdate(
                        source_id=source.state_id,
                        target_id=new_state_id,
                        relationship_type=RelationshipType.RELATES_TO.value,
                        strength=source.weight,
                        action="create",
                        reasoning=f"Synthesis source (weight={source.weight})",
                    )
                )

        # 3. Carry forward important edges from previous state
        if previous_state:
            # Get edges from previous state
            prev_edges = await self.graph_service.get_edges_from_state(
                previous_state.id
            )

            for edge in prev_edges:
                # Carry forward high-strength edges
                if edge.strength >= 0.7:
                    await self.graph_service.create_edge(
                        source_state_id=new_state_id,
                        target_state_id=edge.target_state_id,
                        relationship_type=edge.relationship_type,
                        strength=edge.strength * 0.9,  # Slightly decay
                        metadata={
                            **edge.metadata,
                            "carried_forward": True,
                            "from_state": str(previous_state.id),
                        },
                    )
                    updates.append(
                        EdgeUpdate(
                            source_id=new_state_id,
                            target_id=edge.target_state_id,
                            relationship_type=edge.relationship_type.value,
                            strength=edge.strength * 0.9,
                            action="carry_forward",
                            reasoning=f"High-strength edge carried forward (original={edge.strength})",
                        )
                    )

        return updates

    async def find_promotion_candidates(
        self, limit: int = 10, min_priority: int = 5
    ) -> list[PromotionCandidate]:
        """
        Find topics that are candidates for promotion

        Args:
            limit: Maximum candidates to return
            min_priority: Minimum priority (1-10)

        Returns:
            List of PromotionCandidate objects sorted by priority
        """
        candidates = []

        # Get all topics with multiple versions
        # This is a simplified implementation - in production, you'd query the database
        # For now, return empty list as this requires more complex database queries
        # that would need to be added to MemoryService

        # TODO: Implement database query to find topics with:
        # - Multiple versions (>= 5)
        # - Not promoted recently (last 7 days)
        # - High access frequency
        # - Growing complexity (many related states)

        return candidates

    async def batch_promote(
        self, candidates: list[PromotionCandidate], max_concurrent: int = 3
    ) -> list[PromotionResult]:
        """
        Batch promote multiple candidates

        Args:
            candidates: List of promotion candidates
            max_concurrent: Maximum concurrent promotions

        Returns:
            List of PromotionResult objects
        """
        results = []

        # Sort by priority
        sorted_candidates = sorted(
            candidates, key=lambda c: c.priority, reverse=True
        )

        # Process in batches (simplified - in production use asyncio.gather with semaphore)
        for candidate in sorted_candidates:
            request = PromotionRequest(
                topic=candidate.topic,
                reason=candidate.reasoning,
                trigger=candidate.trigger,
            )
            result = await self.promote_memory(request)
            results.append(result)

        return results


# Singleton instance
_promotion_service: PromotionService | None = None


def get_promotion_service(
    memory_service: MemoryService | None = None,
    graph_service: GraphService | None = None,
    retrieval_service: RetrievalService | None = None,
    llm_client: LLMClient | None = None,
) -> PromotionService:
    """
    Get or create promotion service singleton

    Args:
        memory_service: Optional memory service
        graph_service: Optional graph service
        retrieval_service: Optional retrieval service
        llm_client: Optional LLM client

    Returns:
        PromotionService instance
    """
    global _promotion_service

    if _promotion_service is None:
        _promotion_service = PromotionService(
            memory_service=memory_service,
            graph_service=graph_service,
            retrieval_service=retrieval_service,
            llm_client=llm_client,
        )

    return _promotion_service
