"""Query planning agent for intelligent retrieval"""

import json
import time
from typing import Any

from tracingrag.agents.models import AgentAction, AgentStep, RetrievalPlan
from tracingrag.agents.tools import AgentTools
from tracingrag.core.models.rag import LLMRequest
from tracingrag.services.llm import LLMClient, get_llm_client
from tracingrag.services.query_analyzer import QueryAnalyzer, get_query_analyzer


class QueryPlannerAgent:
    """Agent for analyzing queries and creating retrieval plans"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        tools: AgentTools | None = None,
        planner_model: str = "deepseek/deepseek-chat-v3-0324:free",
    ):
        """
        Initialize query planner agent

        Args:
            llm_client: LLM client for generation
            query_analyzer: Query analyzer for intent detection
            tools: Agent tools for execution
            planner_model: Model to use for planning (free/cheap recommended)
        """
        self.llm_client = llm_client or get_llm_client()
        self.query_analyzer = query_analyzer or get_query_analyzer()
        self.tools = tools or AgentTools()
        self.planner_model = planner_model

    async def analyze_and_plan(self, query: str) -> tuple[dict[str, Any], RetrievalPlan]:
        """
        Analyze query and create retrieval plan

        Args:
            query: User query

        Returns:
            Tuple of (query_analysis, retrieval_plan)
        """
        # Step 1: Analyze query
        analysis = await self.query_analyzer.analyze_query(query, use_llm=True)

        # Step 2: Generate retrieval plan with LLM
        plan = await self._generate_plan(query, analysis)

        return analysis, plan

    async def _generate_plan(self, query: str, analysis: dict[str, Any]) -> RetrievalPlan:
        """
        Generate retrieval plan using LLM

        Args:
            query: User query
            analysis: Query analysis result

        Returns:
            Retrieval plan
        """
        # Define JSON schema for plan
        plan_schema = {
            "name": "retrieval_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "vector_search",
                                        "graph_traversal",
                                        "trace_history",
                                    ],
                                    "description": "Action to execute",
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Parameters for the action",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "Why this step is needed",
                                },
                            },
                            "required": ["action", "parameters", "rationale"],
                            "additionalProperties": False,
                        },
                        "description": "Ordered steps to execute",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Overall rationale for the plan",
                    },
                    "estimated_complexity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Complexity score 1-10",
                    },
                },
                "required": ["steps", "rationale", "estimated_complexity"],
                "additionalProperties": False,
            },
        }

        # Build prompt
        prompt = f"""Create a retrieval plan for this query.

Query: "{query}"

Query Analysis:
- Type: {analysis['query_type'].value}
- Needs History: {analysis['needs_history']}
- Needs Graph: {analysis['needs_graph']}
- Time Scope: {analysis['time_scope']}
- Entities: {', '.join(analysis['entities']) if analysis['entities'] else 'None'}

Available Actions:
1. **vector_search**: Semantic search for similar content
   Parameters: query (string), limit (int), score_threshold (float)

2. **graph_traversal**: Traverse knowledge graph from a starting node
   Parameters: start_node_id (string UUID), depth (int), limit (int)

3. **trace_history**: Get version history for a topic
   Parameters: topic (string), limit (int)

Guidelines:
- For simple queries: 1-2 steps (vector search usually sufficient)
- For historical queries: Use trace_history
- For relationship queries: Use graph_traversal after finding initial nodes
- For complex queries: Chain multiple actions
- Be efficient: Fewer steps are better if they achieve the goal

Create a step-by-step retrieval plan."""

        # Request from LLM
        request = LLMRequest(
            system_prompt=self._get_planner_system_prompt(),
            user_message=prompt,
            context="",
            model=self.planner_model,
            temperature=0.0,  # Deterministic planning
            max_tokens=1000,
            json_schema=plan_schema,
            metadata={"task": "retrieval_planning"},
        )

        response = await self.llm_client.generate(request)

        # Parse plan
        try:
            plan_data = json.loads(response.content)
            return RetrievalPlan(
                query=query,
                steps=plan_data["steps"],
                rationale=plan_data["rationale"],
                estimated_complexity=plan_data["estimated_complexity"],
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback to simple plan
            return RetrievalPlan(
                query=query,
                steps=[
                    {
                        "action": "vector_search",
                        "parameters": {"query": query, "limit": 10},
                        "rationale": "Fallback: Simple semantic search",
                    }
                ],
                rationale="Fallback plan due to parsing error",
                estimated_complexity=1,
            )

    async def execute_plan(self, plan: RetrievalPlan) -> tuple[list[AgentStep], dict[str, Any]]:
        """
        Execute retrieval plan

        Args:
            plan: Retrieval plan to execute

        Returns:
            Tuple of (executed_steps, results)
        """
        executed_steps: list[AgentStep] = []
        all_results: dict[str, Any] = {
            "retrieved_states": [],
            "context": {},
        }

        for i, step_def in enumerate(plan.steps):
            start_time = time.time()

            action_name = step_def["action"]
            parameters = step_def["parameters"]

            step = AgentStep(
                action=AgentAction(action_name),
                parameters=parameters,
            )

            try:
                # Execute action
                if action_name == "vector_search":
                    result = await self.tools.vector_search(**parameters)
                elif action_name == "graph_traversal":
                    result = await self.tools.graph_traversal(**parameters)
                elif action_name == "trace_history":
                    result = await self.tools.trace_history(**parameters)
                else:
                    result = {"success": False, "error": f"Unknown action: {action_name}"}

                step.result = result
                step.duration_ms = (time.time() - start_time) * 1000

                # Accumulate results
                if result.get("success"):
                    if "results" in result:
                        all_results["retrieved_states"].extend(result["results"])
                    if "versions" in result:
                        all_results["retrieved_states"].extend(result["versions"])

            except Exception as e:
                step.error = str(e)
                step.duration_ms = (time.time() - start_time) * 1000

            executed_steps.append(step)

        return executed_steps, all_results

    async def should_replan(
        self, query: str, current_results: dict[str, Any], attempt: int
    ) -> tuple[bool, str]:
        """
        Determine if replanning is needed

        Args:
            query: Original query
            current_results: Results from current plan
            attempt: Current attempt number

        Returns:
            Tuple of (should_replan, reason)
        """
        # Simple heuristics for now
        results_count = len(current_results.get("retrieved_states", []))

        if results_count == 0 and attempt < 3:
            return True, "No results retrieved, need different strategy"

        if results_count < 3 and attempt < 2:
            return True, "Too few results, try broader search"

        return False, "Sufficient results or max attempts reached"

    def _get_planner_system_prompt(self) -> str:
        """Get system prompt for planner"""
        return """You are a retrieval planning assistant for a temporal knowledge graph system.

Your job is to analyze queries and create efficient, step-by-step retrieval plans.

Key Capabilities:
- **Temporal Tracking**: Memory states evolve over time with version history
- **Graph Relationships**: States are connected via typed relationships
- **Semantic Search**: Vector-based similarity search

Planning Principles:
1. **Efficiency**: Use the minimum steps needed
2. **Specificity**: Target the right information with appropriate actions
3. **Chaining**: Use results from one step to inform the next
4. **Fallback**: Provide alternative strategies if first approach fails

Always respond with valid JSON following the required schema."""
