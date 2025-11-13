"""Tests for memory promotion service"""

from datetime import datetime
from uuid import uuid4

import pytest

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
from tracingrag.services.promotion import PromotionService


class TestPromotionModels:
    """Test promotion data models"""

    def test_promotion_request_creation(self):
        """Test creating a promotion request"""
        request = PromotionRequest(
            topic="test_topic",
            reason="Testing promotion",
            trigger=PromotionTrigger.MANUAL,
        )

        assert request.topic == "test_topic"
        assert request.reason == "Testing promotion"
        assert request.trigger == PromotionTrigger.MANUAL
        assert request.include_related is True
        assert request.conflict_resolution_strategy == ConflictResolutionStrategy.LLM_DECIDE

    def test_conflict_creation(self):
        """Test creating a conflict"""
        state_ids = [uuid4(), uuid4()]
        conflict = Conflict(
            state_ids=state_ids,
            conflict_type=ConflictType.CONTRADICTION,
            description="States contradict each other",
            severity=0.8,
        )

        assert len(conflict.state_ids) == 2
        assert conflict.conflict_type == ConflictType.CONTRADICTION
        assert conflict.severity == 0.8
        assert conflict.resolution_strategy == ConflictResolutionStrategy.LLM_DECIDE

    def test_synthesis_source_creation(self):
        """Test creating a synthesis source"""
        source = SynthesisSource(
            state_id=uuid4(),
            topic="test",
            content="Test content",
            version=1,
            timestamp=datetime.utcnow(),
            confidence=0.9,
            weight=0.8,
            reasoning="High quality source",
        )

        assert source.weight == 0.8
        assert source.confidence == 0.9
        assert source.reasoning == "High quality source"

    def test_promotion_result_creation(self):
        """Test creating a promotion result"""
        result = PromotionResult(
            success=True,
            topic="test_topic",
            new_version=2,
            reasoning="Successfully promoted",
            confidence=0.85,
        )

        assert result.success is True
        assert result.new_version == 2
        assert result.confidence == 0.85
        assert result.manual_review_needed is False

    def test_quality_check_creation(self):
        """Test creating a quality check"""
        check = QualityCheck(
            check_type=QualityCheckType.HALLUCINATION,
            passed=True,
            score=0.9,
            issues=[],
            recommendations=["Good quality content"],
        )

        assert check.check_type == QualityCheckType.HALLUCINATION
        assert check.passed is True
        assert check.score == 0.9

    def test_edge_update_creation(self):
        """Test creating an edge update"""
        update = EdgeUpdate(
            source_id=uuid4(),
            target_id=uuid4(),
            relationship_type="EVOLVES_INTO",
            strength=1.0,
            action="create",
            reasoning="Evolution edge",
        )

        assert update.strength == 1.0
        assert update.action == "create"
        assert update.relationship_type == "EVOLVES_INTO"

    def test_promotion_candidate_creation(self):
        """Test creating a promotion candidate"""
        candidate = PromotionCandidate(
            topic="test_topic",
            trigger=PromotionTrigger.AUTO_VERSION_COUNT,
            priority=8,
            reasoning="Too many versions",
            current_version_count=10,
        )

        assert candidate.priority == 8
        assert candidate.trigger == PromotionTrigger.AUTO_VERSION_COUNT
        assert candidate.current_version_count == 10

    def test_conflict_resolution_creation(self):
        """Test creating a conflict resolution"""
        conflict = Conflict(
            state_ids=[uuid4()],
            conflict_type=ConflictType.AMBIGUITY,
            description="Unclear information",
            severity=0.5,
        )

        resolution = ConflictResolution(
            conflict=conflict,
            strategy_used=ConflictResolutionStrategy.MANUAL,
            resolution="Manual review needed",
            confidence=0.0,
            manual_review_needed=True,
        )

        assert resolution.strategy_used == ConflictResolutionStrategy.MANUAL
        assert resolution.manual_review_needed is True
        assert resolution.confidence == 0.0


class TestPromotionService:
    """Test promotion service functionality"""

    @pytest.fixture
    def promotion_service(self):
        """Create a promotion service instance"""
        return PromotionService()

    def test_promotion_service_creation(self, promotion_service):
        """Test that promotion service can be instantiated"""
        assert promotion_service is not None
        assert isinstance(promotion_service, PromotionService)
        assert promotion_service.synthesis_model == "anthropic/claude-3.5-sonnet"
        assert promotion_service.analysis_model == "deepseek/deepseek-chat-v3-0324:free"

    def test_build_synthesis_sources(self, promotion_service):
        """Test building synthesis sources from states"""
        from tracingrag.storage.models import MemoryStateDB

        # Create mock states
        states = [
            MemoryStateDB(
                id=uuid4(),
                topic="test",
                content=f"Content {i}",
                version=i + 1,
                timestamp=datetime.utcnow(),
                embedding=[0.1] * 768,
                confidence=0.9,
            )
            for i in range(3)
        ]

        sources = promotion_service._build_synthesis_sources(
            sources=states, related_states=[]
        )

        assert len(sources) == 3
        assert sources[0].weight == 1.0  # Most recent
        assert sources[1].weight == 0.9  # Second most recent
        assert sources[2].weight == 0.8  # Third most recent
        assert all(s.version > 0 for s in sources)

    def test_build_synthesis_sources_with_related(self, promotion_service):
        """Test building synthesis sources with related states"""
        from tracingrag.storage.models import MemoryStateDB

        # Create mock states
        version_states = [
            MemoryStateDB(
                id=uuid4(),
                topic="test",
                content=f"Version {i}",
                version=i + 1,
                timestamp=datetime.utcnow(),
                embedding=[0.1] * 768,
                confidence=0.9,
            )
            for i in range(2)
        ]

        related_states = [
            MemoryStateDB(
                id=uuid4(),
                topic=f"related_{i}",
                content=f"Related {i}",
                version=1,
                timestamp=datetime.utcnow(),
                embedding=[0.1] * 768,
                confidence=0.8,
            )
            for i in range(3)
        ]

        sources = promotion_service._build_synthesis_sources(
            sources=version_states, related_states=related_states
        )

        # Should have version states + related states
        assert len(sources) == 5
        # Related states should have lower weight
        related_sources = [s for s in sources if "related_" in s.topic]
        assert all(s.weight == 0.5 for s in related_sources)

    def test_get_synthesis_system_prompt(self, promotion_service):
        """Test that synthesis system prompt is defined"""
        prompt = promotion_service._get_synthesis_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "synthesis" in prompt.lower()
        assert "consolidate" in prompt.lower()

    @pytest.mark.asyncio
    async def test_check_citations_with_references(self, promotion_service):
        """Test citation check with content that has references"""

        sources = [
            SynthesisSource(
                state_id=uuid4(),
                topic="test",
                content="Source content",
                version=1,
                timestamp=datetime.utcnow(),
                confidence=0.9,
                weight=1.0,
            )
        ]

        content_with_citations = "According to source 1, this is the case."
        check = await promotion_service._check_citations(content_with_citations, sources)

        assert check.check_type == QualityCheckType.CITATION
        assert check.passed is True
        assert check.score >= 0.8

    @pytest.mark.asyncio
    async def test_check_citations_without_references(self, promotion_service):
        """Test citation check with content lacking references"""
        sources = [
            SynthesisSource(
                state_id=uuid4(),
                topic="test",
                content="Source content",
                version=1,
                timestamp=datetime.utcnow(),
                confidence=0.9,
                weight=1.0,
            )
            for _ in range(3)
        ]

        content_without_citations = "This is just some content without any references."
        check = await promotion_service._check_citations(
            content_without_citations, sources
        )

        assert check.check_type == QualityCheckType.CITATION
        # Should fail when there are multiple sources but no citations
        assert check.passed is False
        assert len(check.issues) > 0

    @pytest.mark.asyncio
    async def test_check_consistency_reasonable_length(self, promotion_service):
        """Test consistency check with reasonable content length"""
        sources = []
        content = " ".join(["word"] * 100)  # 100 words

        check = await promotion_service._check_consistency(content, sources)

        assert check.check_type == QualityCheckType.CONSISTENCY
        assert check.passed is True
        assert check.score >= 0.8

    @pytest.mark.asyncio
    async def test_check_consistency_unreasonable_length(self, promotion_service):
        """Test consistency check with unreasonable content length"""
        sources = []
        content = "short"  # Too short

        check = await promotion_service._check_consistency(content, sources)

        assert check.check_type == QualityCheckType.CONSISTENCY
        assert check.passed is False
        assert len(check.issues) > 0


class TestPromotionIntegration:
    """Integration tests for promotion service"""

    @pytest.fixture
    def promotion_service(self):
        """Create a promotion service instance"""
        return PromotionService()

    @pytest.mark.asyncio
    async def test_promote_memory_no_states(self, promotion_service):
        """Test promotion when no states exist for topic"""
        request = PromotionRequest(
            topic="nonexistent_topic",
            reason="Testing",
            trigger=PromotionTrigger.MANUAL,
        )

        result = await promotion_service.promote_memory(request)

        # Should fail (either no states found or database config error in test env)
        assert result.success is False
        assert result.topic == "nonexistent_topic"
        # Error message should indicate failure
        assert result.error_message is not None and len(result.error_message) > 0

    def test_promotion_trigger_types(self):
        """Test all promotion trigger types are defined"""
        triggers = [
            PromotionTrigger.MANUAL,
            PromotionTrigger.AUTO_VERSION_COUNT,
            PromotionTrigger.AUTO_TIME_BASED,
            PromotionTrigger.AUTO_COMPLEXITY,
            PromotionTrigger.AUTO_CONFLICT,
            PromotionTrigger.AUTO_RELATED_GROWTH,
        ]

        for trigger in triggers:
            assert isinstance(trigger, PromotionTrigger)
            assert trigger.value is not None

    def test_conflict_types(self):
        """Test all conflict types are defined"""
        conflict_types = [
            ConflictType.CONTRADICTION,
            ConflictType.INCONSISTENCY,
            ConflictType.AMBIGUITY,
            ConflictType.TEMPORAL,
        ]

        for ct in conflict_types:
            assert isinstance(ct, ConflictType)
            assert ct.value is not None

    def test_conflict_resolution_strategies(self):
        """Test all resolution strategies are defined"""
        strategies = [
            ConflictResolutionStrategy.LATEST_WINS,
            ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            ConflictResolutionStrategy.MERGE,
            ConflictResolutionStrategy.MANUAL,
            ConflictResolutionStrategy.LLM_DECIDE,
        ]

        for strategy in strategies:
            assert isinstance(strategy, ConflictResolutionStrategy)
            assert strategy.value is not None

    def test_quality_check_types(self):
        """Test all quality check types are defined"""
        check_types = [
            QualityCheckType.HALLUCINATION,
            QualityCheckType.CITATION,
            QualityCheckType.CONSISTENCY,
            QualityCheckType.COMPLETENESS,
            QualityCheckType.RELEVANCE,
        ]

        for ct in check_types:
            assert isinstance(ct, QualityCheckType)
            assert ct.value is not None

    @pytest.mark.asyncio
    async def test_find_promotion_candidates(self, promotion_service):
        """Test finding promotion candidates"""
        candidates = await promotion_service.find_promotion_candidates(limit=10)

        # Should return list (may be empty in test environment)
        assert isinstance(candidates, list)
        assert all(isinstance(c, PromotionCandidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_batch_promote_empty_list(self, promotion_service):
        """Test batch promotion with empty candidate list"""
        results = await promotion_service.batch_promote(candidates=[])

        assert isinstance(results, list)
        assert len(results) == 0


class TestEdgeManagement:
    """Test edge management during promotion"""

    @pytest.fixture
    def promotion_service(self):
        """Create a promotion service instance"""
        return PromotionService()

    def test_edge_update_model(self):
        """Test EdgeUpdate model"""
        update = EdgeUpdate(
            source_id=uuid4(),
            target_id=uuid4(),
            relationship_type="RELATES_TO",
            strength=0.7,
            action="carry_forward",
            reasoning="Important edge",
        )

        assert update.strength == 0.7
        assert update.action == "carry_forward"
        assert "Important" in update.reasoning


class TestQualityControl:
    """Test quality control mechanisms"""

    @pytest.fixture
    def promotion_service(self):
        """Create a promotion service instance"""
        return PromotionService()

    @pytest.mark.asyncio
    async def test_perform_quality_checks(self, promotion_service):
        """Test performing multiple quality checks"""
        sources = [
            SynthesisSource(
                state_id=uuid4(),
                topic="test",
                content="Source content here",
                version=1,
                timestamp=datetime.utcnow(),
                confidence=0.9,
                weight=1.0,
            )
        ]

        content = "According to the source, " + " ".join(["content"] * 50)

        checks = await promotion_service._perform_quality_checks(content, sources)

        # Should perform multiple checks
        assert len(checks) >= 2
        assert all(isinstance(c, QualityCheck) for c in checks)

        # Should include hallucination and citation checks
        check_types = [c.check_type for c in checks]
        assert QualityCheckType.HALLUCINATION in check_types
        assert QualityCheckType.CITATION in check_types


class TestAutomaticPromotion:
    """Test automatic promotion features"""

    def test_promotion_mode_enum(self):
        """Test promotion mode enum values"""
        from tracingrag.core.models.promotion import PromotionMode

        assert PromotionMode.MANUAL == "manual"
        assert PromotionMode.AUTOMATIC == "automatic"

    def test_promotion_policy_creation(self):
        """Test creating a promotion policy"""
        from tracingrag.core.models.promotion import PromotionMode, PromotionPolicy

        policy = PromotionPolicy(
            mode=PromotionMode.AUTOMATIC,
            version_count_threshold=10,
            confidence_threshold=0.9,
        )

        assert policy.mode == PromotionMode.AUTOMATIC
        assert policy.version_count_threshold == 10
        assert policy.confidence_threshold == 0.9
        assert policy.use_llm_evaluation is True

    def test_promotion_policy_defaults(self):
        """Test promotion policy default values"""
        from tracingrag.core.models.promotion import PromotionMode, PromotionPolicy

        policy = PromotionPolicy()

        assert policy.mode == PromotionMode.MANUAL
        assert policy.version_count_threshold == 5
        assert policy.time_threshold_days == 7
        assert policy.confidence_threshold == 0.8
        assert policy.use_llm_evaluation is True

    def test_promotion_evaluation_creation(self):
        """Test creating a promotion evaluation"""
        from tracingrag.core.models.promotion import (
            PromotionEvaluation,
            PromotionTrigger,
        )

        eval = PromotionEvaluation(
            topic="test_topic",
            should_promote=True,
            confidence=0.9,
            priority=8,
            trigger=PromotionTrigger.AUTO_VERSION_COUNT,
            reasoning="High version count with significant changes",
            metrics={"version_count": 10},
        )

        assert eval.should_promote is True
        assert eval.confidence == 0.9
        assert eval.priority == 8

    @pytest.mark.asyncio
    async def test_promotion_service_with_policy(self):
        """Test promotion service instantiation with policy"""
        from tracingrag.core.models.promotion import PromotionMode, PromotionPolicy
        from tracingrag.services.promotion import PromotionService

        policy = PromotionPolicy(mode=PromotionMode.AUTOMATIC)
        service = PromotionService(policy=policy)

        assert service.policy.mode == PromotionMode.AUTOMATIC
        assert service.policy.version_count_threshold == 5

    @pytest.mark.asyncio
    async def test_evaluate_after_insertion_manual_mode(self):
        """Test that evaluate_after_insertion returns None in manual mode"""
        from uuid import uuid4

        from tracingrag.core.models.promotion import PromotionMode, PromotionPolicy
        from tracingrag.services.promotion import PromotionService

        policy = PromotionPolicy(mode=PromotionMode.MANUAL)
        service = PromotionService(policy=policy)

        result = await service.evaluate_after_insertion(
            topic="test_topic", new_state_id=uuid4()
        )

        # Should return None in manual mode
        assert result is None
