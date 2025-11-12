"""Tests for agent system"""

from uuid import uuid4

import pytest

from tracingrag.agents.models import (
    AgentAction,
    AgentResult,
    AgentState,
    AgentStep,
    MemorySuggestion,
    RetrievalPlan,
)
from tracingrag.agents.query_planner import QueryPlannerAgent
from tracingrag.agents.service import AgentService
from tracingrag.agents.tools import AgentTools


class TestAgentModels:
    """Tests for agent data models"""

    def test_agent_action_enum(self):
        """Test AgentAction enum"""
        assert hasattr(AgentAction, "VECTOR_SEARCH")
        assert hasattr(AgentAction, "GRAPH_TRAVERSAL")
        assert hasattr(AgentAction, "TRACE_HISTORY")
        assert AgentAction.VECTOR_SEARCH.value == "vector_search"

    def test_retrieval_plan_creation(self):
        """Test RetrievalPlan model creation"""
        plan = RetrievalPlan(
            query="What is the status?",
            steps=[
                {
                    "action": "vector_search",
                    "parameters": {"query": "status", "limit": 10},
                    "rationale": "Search for status information",
                }
            ],
            rationale="Simple status query",
            estimated_complexity=1,
        )

        assert plan.query == "What is the status?"
        assert len(plan.steps) == 1
        assert plan.steps[0]["action"] == "vector_search"
        assert plan.estimated_complexity == 1

    def test_agent_step_creation(self):
        """Test AgentStep model creation"""
        step = AgentStep(
            action=AgentAction.VECTOR_SEARCH,
            parameters={"query": "test", "limit": 5},
            result={"success": True, "count": 3},
        )

        assert step.action == AgentAction.VECTOR_SEARCH
        assert step.parameters["query"] == "test"
        assert step.result["success"] is True
        assert step.error is None

    def test_agent_state_creation(self):
        """Test AgentState model creation"""
        state = AgentState(
            query="Test query",
            query_type="status",
            max_iterations=3,
        )

        assert state.query == "Test query"
        assert state.query_type == "status"
        assert state.current_step == 0
        assert state.iteration == 0
        assert len(state.steps) == 0
        assert not state.needs_replanning

    def test_memory_suggestion_creation(self):
        """Test MemorySuggestion model creation"""
        suggestion = MemorySuggestion(
            suggestion_type="promotion",
            target_states=[uuid4(), uuid4()],
            rationale="Multiple versions should be consolidated",
            confidence=0.85,
            priority=7,
        )

        assert suggestion.suggestion_type == "promotion"
        assert len(suggestion.target_states) == 2
        assert suggestion.confidence == 0.85
        assert suggestion.priority == 7

    def test_agent_result_creation(self):
        """Test AgentResult model creation"""
        result = AgentResult(
            query="Test query",
            answer="Test answer",
            reasoning_steps=[],
            sources=[uuid4()],
            confidence=0.9,
            replanning_count=0,
            total_time_ms=150.5,
        )

        assert result.query == "Test query"
        assert result.answer == "Test answer"
        assert result.confidence == 0.9
        assert result.replanning_count == 0
        assert len(result.sources) == 1


class TestAgentTools:
    """Tests for agent tools"""

    @pytest.fixture
    def agent_tools(self):
        """Create agent tools instance"""
        return AgentTools()

    def test_agent_tools_instantiation(self, agent_tools):
        """Test that agent tools can be instantiated"""
        assert agent_tools is not None
        assert isinstance(agent_tools, AgentTools)
        assert agent_tools.retrieval_service is not None
        assert agent_tools.memory_service is not None
        assert agent_tools.graph_service is not None

    def test_get_tool_definitions(self, agent_tools):
        """Test getting tool definitions"""
        definitions = agent_tools.get_tool_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) == 5  # 5 tools

        # Check that all tools have required fields
        for tool_def in definitions:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "parameters" in tool_def
            assert "function" in tool_def

        # Check specific tools exist
        tool_names = [t["name"] for t in definitions]
        assert "vector_search" in tool_names
        assert "graph_traversal" in tool_names
        assert "trace_history" in tool_names
        assert "create_memory" in tool_names
        assert "create_edge" in tool_names


class TestQueryPlannerAgent:
    """Tests for query planner agent"""

    @pytest.fixture
    def query_planner(self):
        """Create query planner instance"""
        return QueryPlannerAgent()

    def test_query_planner_instantiation(self, query_planner):
        """Test that query planner can be instantiated"""
        assert query_planner is not None
        assert isinstance(query_planner, QueryPlannerAgent)
        assert query_planner.llm_client is not None
        assert query_planner.query_analyzer is not None
        assert query_planner.tools is not None

    @pytest.mark.asyncio
    async def test_analyze_with_rules(self, query_planner):
        """Test that analyze query works with rule-based fallback"""
        query = "What is the current status?"

        # Use rule-based analysis (no LLM call)
        analysis = await query_planner.query_analyzer.analyze_query(
            query, use_llm=False
        )

        # Check analysis
        assert "query_type" in analysis
        assert "needs_history" in analysis
        assert "needs_graph" in analysis
        assert analysis["query_type"].value == "status"

    def test_system_prompt_defined(self, query_planner):
        """Test that system prompt is defined"""
        prompt = query_planner._get_planner_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "retrieval" in prompt.lower()


class TestAgentService:
    """Tests for agent service"""

    @pytest.fixture
    def agent_service(self):
        """Create agent service instance"""
        return AgentService()

    def test_agent_service_instantiation(self, agent_service):
        """Test that agent service can be instantiated"""
        assert agent_service is not None
        assert isinstance(agent_service, AgentService)
        assert agent_service.llm_client is not None
        assert agent_service.tools is not None
        assert agent_service.query_planner is not None
        assert agent_service.memory_manager is not None

    def test_extract_source_ids(self, agent_service):
        """Test extracting source IDs from state"""
        state_id1 = uuid4()
        state_id2 = uuid4()

        state = AgentState(
            query="test",
            retrieved_states=[
                {"id": str(state_id1), "content": "test1"},
                {"id": str(state_id2), "content": "test2"},
                {"content": "no id"},  # Should be skipped
            ],
        )

        ids = agent_service._extract_source_ids(state)

        assert len(ids) == 2
        assert state_id1 in ids
        assert state_id2 in ids

    def test_calculate_confidence(self, agent_service):
        """Test confidence calculation"""
        state = AgentState(
            query="test",
            iteration=0,
            retrieved_states=[
                {"id": str(uuid4()), "score": 0.8},
                {"id": str(uuid4()), "score": 0.9},
                {"id": str(uuid4()), "score": 0.7},
            ],
        )

        confidence = agent_service._calculate_confidence(state)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be relatively high with good scores


class TestAgentIntegration:
    """Integration tests for agent system"""

    def test_agent_models_import(self):
        """Test that agent models can be imported"""
        from tracingrag.agents import (
            AgentAction,
            AgentResult,
            AgentState,
            AgentStep,
            MemorySuggestion,
            RetrievalPlan,
        )

        assert AgentAction is not None
        assert AgentState is not None
        assert AgentStep is not None
        assert AgentResult is not None
        assert RetrievalPlan is not None
        assert MemorySuggestion is not None

    def test_agent_services_import(self):
        """Test that agent services can be imported"""
        from tracingrag.agents import (
            AgentService,
            AgentTools,
            MemoryManagerAgent,
            QueryPlannerAgent,
            query_with_agent,
        )

        assert AgentTools is not None
        assert QueryPlannerAgent is not None
        assert MemoryManagerAgent is not None
        assert AgentService is not None
        assert query_with_agent is not None

    @pytest.mark.asyncio
    async def test_end_to_end_agent_imports(self):
        """Test complete import chain for agents"""
        from tracingrag.agents import AgentService, AgentState

        # Create service
        service = AgentService()
        assert service is not None

        # Create state
        state = AgentState(query="test query")
        assert state is not None
        assert state.query == "test query"
