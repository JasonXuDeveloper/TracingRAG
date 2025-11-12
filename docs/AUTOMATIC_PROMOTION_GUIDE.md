# Automatic Memory Promotion - Usage Guide

This guide explains how to use TracingRAG's automatic memory promotion feature with LLM-based decision making.

## Overview

TracingRAG supports two promotion modes:
- **Manual**: Users explicitly request promotion (default, safest)
- **Automatic**: LLM evaluates and promotes automatically

## Quick Start

### 1. Manual Mode (Default)

```python
from tracingrag.services.promotion import PromotionService
from tracingrag.core.models.promotion import PromotionRequest

# Create service (defaults to manual mode)
promotion_service = PromotionService()

# Explicitly promote when needed
result = await promotion_service.promote_memory(
    PromotionRequest(
        topic="project_alpha",
        reason="Consolidating 10 versions of design changes"
    )
)

print(f"Success: {result.success}")
print(f"New version: {result.new_version}")
print(f"Reasoning: {result.reasoning}")
```

### 2. Automatic Mode with LLM Evaluation

```python
from tracingrag.services.promotion import PromotionService
from tracingrag.core.models.promotion import PromotionPolicy, PromotionMode

# Configure automatic promotion
policy = PromotionPolicy(
    mode=PromotionMode.AUTOMATIC,
    version_count_threshold=5,  # Promote after 5+ versions
    time_threshold_days=7,      # Wait 7 days since last
    confidence_threshold=0.8,    # LLM must be 80%+ confident
    use_llm_evaluation=True,     # Use LLM to decide
)

# Create service with policy
promotion_service = PromotionService(policy=policy)

# After inserting new memory, check if promotion needed
from tracingrag.services.memory import MemoryService

memory_service = MemoryService()
new_state = await memory_service.create_memory_state(
    topic="project_alpha",
    content="Updated API design after feedback from review"
)

# Evaluate if promotion needed (auto-promotes if thresholds met)
evaluation = await promotion_service.evaluate_after_insertion(
    topic="project_alpha",
    new_state_id=new_state.id
)

if evaluation:
    print(f"Promotion triggered: {evaluation.should_promote}")
    print(f"Confidence: {evaluation.confidence}")
    print(f"Reasoning: {evaluation.reasoning}")
```

## Configuration Options

### PromotionPolicy Parameters

```python
policy = PromotionPolicy(
    # Mode
    mode=PromotionMode.MANUAL,  # or AUTOMATIC

    # Detection thresholds
    version_count_threshold=5,    # Min versions before considering
    time_threshold_days=7,        # Days since last promotion
    confidence_threshold=0.8,     # Min LLM confidence for auto-promotion

    # LLM evaluation
    use_llm_evaluation=True,      # Use LLM to decide
    evaluation_model="deepseek/deepseek-chat-v3-0324:free",  # Cheap model

    # Safety
    require_approval_for_conflicts=True,  # Ask user if conflicts found
    notify_on_auto_promotion=True,         # Notify after auto-promotion
    dry_run=False,                         # Test without executing

    # Resource limits
    max_concurrent_promotions=3,     # Batch limit
    max_candidates_per_scan=20,      # Max to evaluate per scan
)
```

## Use Cases

### Use Case 1: High-Volume Knowledge Base

```python
# Scenario: Customer support system with 1000s of tickets/day
# Need: Automatic consolidation of similar issues

policy = PromotionPolicy(
    mode=PromotionMode.AUTOMATIC,
    version_count_threshold=10,  # Wait for more data
    confidence_threshold=0.9,    # Be very confident
    use_llm_evaluation=True,
)

service = PromotionService(policy=policy)

# Hook into your ticket creation flow:
async def on_ticket_created(ticket):
    await memory_service.create_memory_state(
        topic=f"issue_{ticket.category}",
        content=ticket.description
    )

    # Auto-evaluate and promote if needed
    await service.evaluate_after_insertion(
        topic=f"issue_{ticket.category}",
        new_state_id=ticket.state_id
    )
```

### Use Case 2: Personal Knowledge Management

```python
# Scenario: Personal notes/journal
# Need: Manual control, but with suggestions

policy = PromotionPolicy(
    mode=PromotionMode.MANUAL,  # User decides
    use_llm_evaluation=True,    # But get LLM suggestions
)

service = PromotionService(policy=policy)

# Find promotion suggestions
candidates = await service.find_promotion_candidates(limit=10)

for candidate in candidates:
    print(f"\nSuggested promotion for: {candidate.topic}")
    print(f"Reason: {candidate.reasoning}")
    print(f"Priority: {candidate.priority}/10")
    print(f"Confidence: {candidate.confidence:.2f}")

    # User decides
    if input("Promote? (y/n): ").lower() == 'y':
        result = await service.promote_memory(
            PromotionRequest(
                topic=candidate.topic,
                reason=candidate.reasoning
            )
        )
```

### Use Case 3: IoT Data Consolidation

```python
# Scenario: Sensor data aggregation
# Need: Automatic hourly → daily → weekly rollups

policy = PromotionPolicy(
    mode=PromotionMode.AUTOMATIC,
    version_count_threshold=24,  # 24 hourly readings
    use_llm_evaluation=False,    # Rule-based is fine
    confidence_threshold=0.7,
)

service = PromotionService(policy=policy)

# Periodic job (every hour)
async def consolidate_sensor_data():
    candidates = await service.find_promotion_candidates()
    results = await service.batch_promote(candidates)
    print(f"Promoted {len(results)} sensor topics")
```

## LLM Evaluation Process

When `use_llm_evaluation=True`, the LLM considers:

1. **Version Count**: Are there enough versions to warrant consolidation?
2. **Significance**: Is there substantial new information?
3. **Coherence**: Would consolidation improve clarity?
4. **Stability**: Is the information stable enough?

Example LLM decision:
```python
evaluation = await service.evaluate_promotion_need(
    topic="api_design",
    trigger=PromotionTrigger.AUTO_VERSION_COUNT,
    metrics={"version_count": 8}
)

# LLM returns:
{
    "should_promote": True,
    "confidence": 0.85,
    "priority": 7,
    "reasoning": "8 versions with iterative refinements to API endpoints.
                  Clear evolution from REST to GraphQL. Consolidation would
                  provide single source of truth for current API design."
}
```

## Finding Promotion Candidates

```python
# Find topics needing promotion (with database queries)
candidates = await service.find_promotion_candidates(
    limit=20,
    min_priority=5  # Only show priority 5+
)

for candidate in candidates:
    print(f"Topic: {candidate.topic}")
    print(f"Versions: {candidate.current_version_count}")
    print(f"Trigger: {candidate.trigger.value}")
    print(f"Priority: {candidate.priority}/10")
    print(f"Confidence: {candidate.confidence:.2%}")
    print(f"Reasoning: {candidate.reasoning}\n")
```

## Safety Features

### 1. Dry Run Mode
```python
policy = PromotionPolicy(
    mode=PromotionMode.AUTOMATIC,
    dry_run=True  # Simulate without executing
)

# Evaluations happen, but no actual promotions
evaluation = await service.evaluate_after_insertion(topic, state_id)
print(f"Would promote: {evaluation.should_promote}")
```

### 2. Conflict Detection
```python
# If conflicts detected during promotion
result = await service.promote_memory(request)

if result.manual_review_needed:
    print("Manual review required:")
    for conflict in result.conflicts_detected:
        print(f"- {conflict.description}")
    for resolution in result.conflicts_resolved:
        print(f"  Resolved: {resolution.resolution}")
```

### 3. Quality Checks
```python
result = await service.promote_memory(request)

for check in result.quality_checks:
    print(f"{check.check_type.value}: {'PASSED' if check.passed else 'FAILED'}")
    if not check.passed:
        for issue in check.issues:
            print(f"  - {issue}")
```

## Integration with Memory Service

Hook promotion evaluation into memory creation:

```python
class EnhancedMemoryService(MemoryService):
    def __init__(self, promotion_service: PromotionService):
        super().__init__()
        self.promotion_service = promotion_service

    async def create_memory_state(self, topic: str, content: str, **kwargs):
        # Create state as normal
        state = await super().create_memory_state(topic, content, **kwargs)

        # Check if promotion needed (if automatic mode)
        evaluation = await self.promotion_service.evaluate_after_insertion(
            topic=topic,
            new_state_id=state.id
        )

        if evaluation:
            print(f"Auto-promotion triggered for {topic}")

        return state
```

## Best Practices

1. **Start with Manual Mode**: Test and understand before enabling automatic

2. **Conservative Thresholds**: Start with high thresholds (confidence=0.9, versions=10)

3. **Use Dry Run**: Test automatic promotion with `dry_run=True` first

4. **Monitor Results**: Track promotion success rates and quality

5. **Cost Optimization**: Use cheap models (DeepSeek) for evaluation, premium (Claude) for synthesis

6. **Gradual Rollout**: Start with low-risk topics, expand based on results

## Troubleshooting

### Promotions Not Triggering

```python
# Check policy configuration
print(f"Mode: {service.policy.mode}")
print(f"Enabled triggers: {service.policy.enabled_triggers}")
print(f"Thresholds: versions={service.policy.version_count_threshold}, "
      f"confidence={service.policy.confidence_threshold}")

# Check candidates
candidates = await service.find_promotion_candidates()
print(f"Found {len(candidates)} candidates")
```

### Low Quality Promotions

```python
# Increase confidence threshold
policy.confidence_threshold = 0.95  # Very conservative

# Enable all quality checks
policy.require_approval_for_conflicts = True

# Review LLM evaluations
evaluation = await service.evaluate_promotion_need(topic, trigger)
print(f"LLM reasoning: {evaluation.reasoning}")
```

## Performance Considerations

- **LLM Evaluation**: Adds ~500ms per topic evaluation
- **Database Queries**: `find_promotion_candidates()` runs complex aggregations
- **Batch Processing**: Limit `max_concurrent_promotions` based on resources
- **Cost**: DeepSeek free tier = $0, but rate-limited

## Summary

- **Manual Mode** (default): Safe, explicit control
- **Automatic Mode**: LLM-driven, configurable thresholds
- **LLM Evaluation**: Intelligent decisions based on content analysis
- **Safety**: Dry run, conflict detection, quality checks
- **Flexible**: Per-topic policies, gradual rollout, monitoring
