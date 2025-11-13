"""Memory management agent for intelligent memory operations"""

import json
from typing import Any

from tracingrag.agents.models import MemorySuggestion
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.graph import GraphService
from tracingrag.services.llm import LLMClient, get_llm_client
from tracingrag.services.memory import MemoryService


class MemoryManagerAgent:
    """Agent for managing memory states and suggesting operations"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory_service: MemoryService | None = None,
        graph_service: GraphService | None = None,
        manager_model: str = "deepseek/deepseek-chat-v3-0324:free",
    ):
        """
        Initialize memory manager agent

        Args:
            llm_client: LLM client for analysis
            memory_service: Memory service for operations
            graph_service: Graph service for relationships
            manager_model: Model to use (free/cheap recommended)
        """
        self.llm_client = llm_client or get_llm_client()
        self.memory_service = memory_service or MemoryService()
        self.graph_service = graph_service or GraphService()
        self.manager_model = manager_model

    async def analyze_memory_state(
        self, topic: str, max_versions: int = 10
    ) -> list[MemorySuggestion]:
        """
        Analyze memory state and generate suggestions

        Args:
            topic: Topic to analyze
            max_versions: Maximum versions to consider

        Returns:
            List of suggestions
        """
        # Get memory history
        versions = await self.memory_service.get_topic_history(
            topic=topic, limit=max_versions
        )

        if not versions:
            return []

        # Get latest version
        latest = versions[0]

        suggestions: list[MemorySuggestion] = []

        # Check for promotion need
        promotion_suggestion = await self._check_promotion_need(topic, versions)
        if promotion_suggestion:
            suggestions.append(promotion_suggestion)

        # Check for connection opportunities
        connection_suggestions = await self._suggest_connections(latest)
        suggestions.extend(connection_suggestions)

        # Check for conflicts
        conflict_suggestions = await self._detect_conflicts(topic, versions)
        suggestions.extend(conflict_suggestions)

        return suggestions

    async def _check_promotion_need(
        self, topic: str, versions: list[Any]
    ) -> MemorySuggestion | None:
        """
        Check if memory needs promotion/consolidation

        Args:
            topic: Topic name
            versions: Version history

        Returns:
            Suggestion if promotion needed, None otherwise
        """
        if len(versions) < 3:
            return None  # Not enough history

        # Build context about the topic's evolution
        context = self._build_version_context(versions)

        # Define JSON schema for promotion analysis
        analysis_schema = {
            "name": "promotion_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "needs_promotion": {
                        "type": "boolean",
                        "description": "Whether promotion is needed",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Explanation of the decision",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score",
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Priority 1-10",
                    },
                },
                "required": ["needs_promotion", "rationale", "confidence", "priority"],
                "additionalProperties": False,
            },
        }

        prompt = f"""Analyze if this memory topic needs promotion (consolidation into a summary).

Topic: {topic}
Number of Versions: {len(versions)}

{context}

Criteria for Promotion:
- Multiple significant changes over time
- Information could be synthesized into clearer summary
- Reduces cognitive load for retrieval
- Maintains important details while removing noise

Respond with your analysis."""

        request = LLMRequest(
            system_prompt=self._get_manager_system_prompt(),
            user_message=prompt,
            context="",
            model=self.manager_model,
            temperature=0.0,
            max_tokens=500,
            json_schema=analysis_schema,
            metadata={"task": "promotion_analysis"},
        )

        response = await self.llm_client.generate(request)

        try:
            analysis = json.loads(response.content)

            if analysis["needs_promotion"]:
                return MemorySuggestion(
                    suggestion_type="promotion",
                    target_states=[v.id for v in versions],
                    rationale=analysis["rationale"],
                    confidence=analysis["confidence"],
                    priority=analysis["priority"],
                    metadata={"version_count": len(versions)},
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    async def _suggest_connections(self, state: Any) -> list[MemorySuggestion]:
        """
        Suggest new connections for a memory state

        Args:
            state: Memory state to analyze

        Returns:
            List of connection suggestions
        """
        # For now, use simple heuristic: check if state has few connections
        # In production, would use LLM to analyze content and suggest semantic connections

        try:
            # Get existing edges
            edges = await self.graph_service.get_edges_for_state(state.id)

            if len(edges) < 2:
                # Suggest finding connections
                return [
                    MemorySuggestion(
                        suggestion_type="connection",
                        target_states=[state.id],
                        rationale=f"Topic '{state.topic}' has few connections ({len(edges)}). Consider linking to related concepts.",
                        confidence=0.6,
                        priority=5,
                        metadata={"current_connections": len(edges)},
                    )
                ]
        except Exception:
            pass

        return []

    async def _detect_conflicts(
        self, topic: str, versions: list[Any]
    ) -> list[MemorySuggestion]:
        """
        Detect potential conflicts in memory versions

        Args:
            topic: Topic name
            versions: Version history

        Returns:
            List of conflict suggestions
        """
        if len(versions) < 2:
            return []

        # Check for decreasing confidence over versions
        confidences = [v.confidence for v in versions]

        if len(confidences) >= 2:
            recent_avg = sum(confidences[:2]) / 2
            older_avg = sum(confidences[2:]) / len(confidences[2:]) if len(confidences) > 2 else 1.0

            if recent_avg < older_avg - 0.2:  # Significant drop
                return [
                    MemorySuggestion(
                        suggestion_type="conflict",
                        target_states=[v.id for v in versions[:3]],
                        rationale=f"Confidence declining in recent versions for topic '{topic}'. May indicate conflicting information or uncertainty.",
                        confidence=0.7,
                        priority=7,
                        metadata={
                            "recent_confidence": recent_avg,
                            "older_confidence": older_avg,
                        },
                    )
                ]

        return []

    def _build_version_context(self, versions: list[Any], max_length: int = 500) -> str:
        """
        Build context string from version history

        Args:
            versions: Version history
            max_length: Maximum context length per version

        Returns:
            Formatted context string
        """
        lines = ["Version History (newest first):"]

        for v in versions[:5]:  # Limit to 5 most recent
            content_preview = (
                v.content[:max_length] + "..."
                if len(v.content) > max_length
                else v.content
            )
            lines.append(
                f"- v{v.version} ({v.timestamp.date()}): {content_preview}"
            )

        return "\n".join(lines)

    def _get_manager_system_prompt(self) -> str:
        """Get system prompt for memory manager"""
        return """You are a memory management assistant for a temporal knowledge graph system.

Your role is to analyze memory states and suggest operations:
- **Promotion**: When to consolidate multiple versions into summaries
- **Connections**: When to link related concepts
- **Conflicts**: When to flag inconsistencies or contradictions

Key Principles:
1. **Preserve Information**: Never suggest losing important details
2. **Reduce Noise**: Help consolidate redundant or outdated information
3. **Enhance Discovery**: Suggest connections that aid retrieval
4. **Flag Uncertainty**: Detect declining confidence or conflicts

Always respond with structured JSON following the required schema."""
