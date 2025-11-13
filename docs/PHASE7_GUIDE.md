# Phase 7: Caching & Consolidation - Usage Guide

This guide explains how to use TracingRAG's Redis caching layer and hierarchical consolidation system.

## Overview

Phase 7 adds two major performance and scalability features:

1. **Redis Caching Layer**: Multi-tier caching for embeddings, query results, and working memory
2. **Hierarchical Consolidation**: Automatic memory consolidation at daily/weekly/monthly intervals

## Table of Contents

- [Redis Caching](#redis-caching)
  - [Setup](#setup)
  - [Using the Cache Service](#using-the-cache-service)
  - [Enhanced Embedding Service](#enhanced-embedding-service)
  - [Cache Invalidation](#cache-invalidation)
  - [Cache Statistics](#cache-statistics)
- [Hierarchical Consolidation](#hierarchical-consolidation)
  - [Configuration](#configuration)
  - [Running Consolidation](#running-consolidation)
  - [Drill-Down Queries](#drill-down-queries)
  - [Automation](#automation)

---

## Redis Caching

### Setup

1. **Start Redis** (already configured in docker-compose.yml):
```bash
docker-compose up -d redis
```

2. **Configure Redis URL** (optional, defaults to localhost):
```bash
# .env
REDIS_URL=redis://localhost:6379/0
```

3. **Verify Connection**:
```python
from tracingrag.services.cache import get_cache_service

cache = get_cache_service()
stats = await cache.get_stats()
print(f"Redis version: {stats['redis_version']}")
```

### Using the Cache Service

#### Basic Operations

```python
from tracingrag.services.cache import get_cache_service

cache = get_cache_service()

# Set a value (with optional TTL)
await cache.set("my_key", "my_value", ttl=3600)  # 1 hour

# Get a value
value = await cache.get("my_key")
print(value)  # "my_value"

# Check if key exists
exists = await cache.exists("my_key")

# Get TTL
ttl_seconds = await cache.ttl("my_key")

# Delete a key
await cache.delete("my_key")

# Delete by pattern
await cache.delete_pattern("prefix:*")
```

#### Embedding Cache

The embedding cache speeds up repeated embedding generation:

```python
# Cache embedding
text = "Hello, world!"
model = "all-mpnet-base-v2"
embedding = [0.1, 0.2, 0.3, ...]  # 768-dim vector

await cache.set_embedding(text, model, embedding, ttl=timedelta(days=7))

# Retrieve cached embedding
cached = await cache.get_embedding(text, model)

# Invalidate embeddings for a model
await cache.invalidate_embeddings(model)

# Invalidate all embeddings
await cache.invalidate_embeddings()
```

**Performance Impact**: Avoids re-computing embeddings for the same text. Typical speedup: 100-1000x for cached embeddings.

#### Query Result Cache

Cache query results for faster response times:

```python
# Cache query result
query = "What is the status of project alpha?"
params = {"limit": 10, "include_history": True}
result = {
    "answer": "Project alpha is in testing phase",
    "confidence": 0.92,
    "sources": [...]
}

await cache.set_query_result(query, params, result, ttl=timedelta(hours=1))

# Retrieve cached result
cached = await cache.get_query_result(query, params)

# Invalidate all query caches
await cache.invalidate_queries()
```

**Performance Impact**: Returns cached results instantly. Typical speedup: 10-50x for identical queries.

#### Working Memory Cache

Store active context for session-based operations:

```python
# Set working memory for a session
context_id = "session_123"
context = {
    "active_topic": "project_alpha",
    "recent_queries": ["status", "timeline", "team"],
    "user_preferences": {"detail_level": "high"}
}

await cache.set_working_memory(context_id, context, ttl=timedelta(minutes=30))

# Retrieve working memory
cached_context = await cache.get_working_memory(context_id)

# Invalidate specific session
await cache.invalidate_working_memory(context_id)

# Invalidate all sessions
await cache.invalidate_working_memory()
```

#### Latest State Cache

Cache latest state for O(1) topic lookups:

```python
# Cache latest state
topic = "project_alpha"
state = {
    "id": "abc123",
    "content": "Latest version of project alpha status",
    "version": 5,
    "timestamp": "2025-01-15T10:30:00Z"
}

await cache.set_latest_state(topic, state, ttl=timedelta(hours=24))

# Retrieve latest state
latest = await cache.get_latest_state(topic)

# Invalidate specific topic
await cache.invalidate_latest_state(topic)

# Invalidate all latest states
await cache.invalidate_latest_state()
```

### Enhanced Embedding Service

The `EmbeddingService` class integrates Redis caching automatically:

```python
from tracingrag.services.embedding import EmbeddingService

# Create service with Redis caching enabled
embedding_service = EmbeddingService(use_redis_cache=True)

# Generate embedding (automatically caches)
text = "This is a test"
embedding = await embedding_service.embed(text)

# Subsequent calls return cached value
cached_embedding = await embedding_service.embed(text)  # Instant!

# Batch embeddings with caching
texts = ["text 1", "text 2", "text 3"]
embeddings = await embedding_service.embed_batch(texts)

# Clear caches
await embedding_service.clear_cache(redis_only=False)
```

**Two-Tier Caching**:
1. **In-Memory Cache**: Fast access for recent embeddings (LRU, 1000 items)
2. **Redis Cache**: Persistent storage for long-term caching (7-day TTL)

The service tries Redis first, falls back to in-memory, and generates if not found.

### Cache Invalidation

Strategic cache invalidation ensures data consistency:

```python
# When a memory state is updated
await cache.invalidate_latest_state(topic)
await cache.invalidate_queries()  # Query results may be stale

# When embeddings model changes
await cache.invalidate_embeddings(old_model)

# When a session ends
await cache.invalidate_working_memory(session_id)

# Pattern-based invalidation
await cache.delete_pattern("query:*")  # All queries
await cache.delete_pattern("embedding:model-v1:*")  # Specific model
```

### Cache Statistics

Monitor cache performance:

```python
stats = await cache.get_stats()

print(f"Redis version: {stats['redis_version']}")
print(f"Memory used: {stats['used_memory_mb']:.2f} MB")
print(f"Total keys: {stats['total_keys']}")
print(f"Embedding keys: {stats['embedding_keys']}")
print(f"Query keys: {stats['query_keys']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Cache Warming

Pre-populate caches for better initial performance:

```python
# Warm embedding cache
common_texts = [
    "project status",
    "team members",
    "upcoming deadlines",
    # ... more common phrases
]

model = "all-mpnet-base-v2"
count = await cache.warm_embeddings(common_texts, model, batch_size=100)
print(f"Warmed {count} embeddings")

# Warm latest state cache
active_topics = ["project_alpha", "project_beta", "team_updates"]
count = await cache.warm_latest_states(active_topics)
print(f"Warmed {count} latest states")
```

---

## Hierarchical Consolidation

Hierarchical consolidation automatically summarizes memory states at different time scales, mimicking human memory consolidation during sleep.

### Configuration

```python
from tracingrag.services.consolidation import (
    ConsolidationService,
    ConsolidationConfig,
    ConsolidationLevel,
)

# Create configuration
config = ConsolidationConfig(
    # Thresholds
    daily_threshold=10,  # Min states for daily consolidation
    weekly_threshold=5,   # Min daily summaries for weekly
    monthly_threshold=4,  # Min weekly summaries for monthly

    # Schedules (UTC)
    daily_schedule_hour=2,    # 2 AM UTC
    weekly_schedule_day=6,    # Sunday (0=Monday)
    monthly_schedule_day=1,   # 1st of month

    # LLM model for consolidation
    consolidation_model="anthropic/claude-3.5-sonnet",

    # Enabled levels
    enabled_levels=[
        ConsolidationLevel.DAILY,
        ConsolidationLevel.WEEKLY,
        ConsolidationLevel.MONTHLY,
    ],
)

# Create service
consolidation = ConsolidationService(config=config)
```

### Running Consolidation

#### Daily Consolidation

Consolidates all states from a single day into a daily summary:

```python
from datetime import datetime, timedelta

# Consolidate yesterday (default)
results = await consolidation.run_daily_consolidation()

# Consolidate specific date
date = datetime(2025, 1, 15)
results = await consolidation.run_daily_consolidation(date)

# Process results
for result in results:
    if result.success:
        print(f"✅ Consolidated {result.topic}")
        print(f"   States: {result.states_consolidated}")
        print(f"   New ID: {result.new_state_id}")
        print(f"   Summary: {result.summary[:100]}...")
    else:
        print(f"❌ Failed: {result.topic}")
        print(f"   Error: {result.error_message}")
```

**Example**:
- **Input**: 15 states for `project_alpha` on 2025-01-15
- **Output**: 1 state `project_alpha:daily:2025-01-15` with consolidated summary
- **Tags**: `daily_summary`, `consolidated`

#### Weekly Consolidation

Consolidates daily summaries from a week into a weekly summary:

```python
# Consolidate last week (default)
results = await consolidation.run_weekly_consolidation()

# Consolidate specific week
end_of_week = datetime(2025, 1, 19)  # Sunday
results = await consolidation.run_weekly_consolidation(end_of_week)
```

**Example**:
- **Input**: 7 daily summaries for `project_alpha` (Mon-Sun)
- **Output**: 1 state `project_alpha:weekly:2025-W03` with weekly overview
- **Tags**: `weekly_summary`, `consolidated`

#### Monthly Consolidation

Consolidates weekly summaries from a month into a monthly summary:

```python
# Consolidate last month (default)
results = await consolidation.run_monthly_consolidation()

# Consolidate specific month
month_date = datetime(2025, 1, 15)
results = await consolidation.run_monthly_consolidation(month_date)
```

**Example**:
- **Input**: 4-5 weekly summaries for `project_alpha` in January
- **Output**: 1 state `project_alpha:monthly:2025-01` with monthly report
- **Tags**: `monthly_summary`, `consolidated`

#### Run All Consolidations

```python
# Run all enabled consolidation levels
all_results = await consolidation.run_all_consolidations()

for level, results in all_results.items():
    print(f"\n{level.value.upper()} Consolidation:")
    print(f"  Consolidated {len(results)} topics")
    success_count = sum(1 for r in results if r.success)
    print(f"  Success rate: {success_count}/{len(results)}")
```

### Finding Consolidation Candidates

Manually identify topics that need consolidation:

```python
# Find daily candidates
daily_candidates = await consolidation.find_daily_candidates()

for candidate in daily_candidates:
    print(f"Topic: {candidate.topic}")
    print(f"  States: {candidate.state_count}")
    print(f"  Period: {candidate.earliest_timestamp} to {candidate.latest_timestamp}")
    print(f"  Should consolidate: {candidate.should_consolidate}")
    print(f"  Reasoning: {candidate.reasoning}")

# Find all candidates
all_candidates = await consolidation.find_all_candidates()

for level, candidates in all_candidates.items():
    print(f"\n{level.value}: {len(candidates)} candidates")
```

### Drill-Down Queries

Retrieve detailed states from a consolidated summary:

```python
# Get details for a daily summary
consolidated_topic = "project_alpha:daily:2025-01-15"
details = await consolidation.get_detailed_states(consolidated_topic)

print(f"Daily summary consolidates {len(details)} detailed states:")
for state in details:
    print(f"  - v{state.version}: {state.content[:50]}... ({state.timestamp})")

# Get details for weekly summary
weekly_topic = "project_alpha:weekly:2025-W03"
weekly_details = await consolidation.get_detailed_states(weekly_topic)

# Get details for monthly summary
monthly_topic = "project_alpha:monthly:2025-01"
monthly_details = await consolidation.get_detailed_states(monthly_topic)
```

**Use Cases**:
- View summary → drill down to see original details
- Navigate from monthly → weekly → daily → original states
- Audit what was included in a consolidation

### Automation

#### Scheduled Consolidation

Use a task scheduler (e.g., cron, APScheduler) to run consolidation automatically:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tracingrag.services.consolidation import ConsolidationService

scheduler = AsyncIOScheduler()
consolidation = ConsolidationService()

# Run daily consolidation at 2 AM UTC
@scheduler.scheduled_job('cron', hour=2, minute=0)
async def daily_consolidation_job():
    print("Running daily consolidation...")
    results = await consolidation.run_daily_consolidation()
    print(f"Consolidated {len(results)} topics")

# Run weekly consolidation on Sunday at 3 AM UTC
@scheduler.scheduled_job('cron', day_of_week=6, hour=3, minute=0)
async def weekly_consolidation_job():
    print("Running weekly consolidation...")
    results = await consolidation.run_weekly_consolidation()
    print(f"Consolidated {len(results)} topics")

# Run monthly consolidation on 1st of month at 4 AM UTC
@scheduler.scheduled_job('cron', day=1, hour=4, minute=0)
async def monthly_consolidation_job():
    print("Running monthly consolidation...")
    results = await consolidation.run_monthly_consolidation()
    print(f"Consolidated {len(results)} topics")

scheduler.start()
```

#### Manual Trigger

Create an API endpoint for manual consolidation:

```python
# In your FastAPI app
@app.post("/api/v1/consolidate/{level}")
async def trigger_consolidation(level: ConsolidationLevel):
    consolidation = ConsolidationService()

    if level == ConsolidationLevel.DAILY:
        results = await consolidation.run_daily_consolidation()
    elif level == ConsolidationLevel.WEEKLY:
        results = await consolidation.run_weekly_consolidation()
    else:  # MONTHLY
        results = await consolidation.run_monthly_consolidation()

    return {
        "level": level.value,
        "topics_consolidated": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
    }
```

---

## Best Practices

### Caching

1. **TTL Selection**:
   - Embeddings: 7 days (stable, rarely change)
   - Query results: 1 hour (may change frequently)
   - Working memory: 30 minutes (session-based)
   - Latest states: 24 hours (updated on new versions)

2. **Cache Invalidation**:
   - Invalidate on write operations
   - Invalidate query cache when states change
   - Invalidate latest state cache when new versions created

3. **Monitoring**:
   - Track hit rates (aim for >80%)
   - Monitor memory usage
   - Alert on low hit rates or high memory

4. **Cost Optimization**:
   - Use Redis for persistent caching
   - Use in-memory for hot paths
   - Warm caches during off-peak hours

### Consolidation

1. **Thresholds**:
   - Start conservative (10+ states for daily)
   - Adjust based on activity patterns
   - Higher thresholds for high-volume topics

2. **Scheduling**:
   - Run during low-traffic periods (2-4 AM)
   - Stagger different levels (daily at 2 AM, weekly at 3 AM)
   - Allow time between levels for processing

3. **Quality Control**:
   - Review consolidated summaries regularly
   - Adjust LLM model based on quality
   - Use Claude for high-quality synthesis

4. **Storage**:
   - Consolidation reduces storage growth
   - Keep original states for drill-down
   - Archive old detailed states if needed

---

## Troubleshooting

### Redis Connection Issues

```python
# Check Redis connection
try:
    stats = await cache.get_stats()
    print("✅ Redis connected")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

**Solutions**:
- Verify Redis is running: `docker ps | grep redis`
- Check `REDIS_URL` in `.env`
- Ensure port 6379 is not blocked

### Low Cache Hit Rate

```python
stats = await cache.get_stats()
if stats['hit_rate'] < 0.5:
    print("⚠️ Low cache hit rate!")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
```

**Solutions**:
- Increase TTLs if data is stable
- Warm cache with common queries/embeddings
- Review invalidation strategy (may be too aggressive)

### Consolidation Not Triggering

```python
# Check candidates
candidates = await consolidation.find_daily_candidates()
if not candidates:
    print("No candidates found for consolidation")
    print("Possible reasons:")
    print("- Not enough states (need 10+)")
    print("- Already consolidated")
    print("- Wrong date range")
```

**Solutions**:
- Lower thresholds in `ConsolidationConfig`
- Check that states have correct timestamps
- Verify consolidation tags aren't already present

### Memory Growth

```python
stats = await cache.get_stats()
if stats['used_memory_mb'] > 1000:  # 1 GB
    print("⚠️ High memory usage!")
    print("Consider:")
    print("- Reducing TTLs")
    print("- Clearing old caches")
    print("- Increasing Redis memory limit")
```

**Solutions**:
- Run `await cache.delete_pattern("query:*")` to clear query cache
- Reduce embedding cache TTL
- Configure Redis `maxmemory` and `maxmemory-policy`

---

## Summary

**Redis Caching**:
- Speeds up embeddings (100-1000x)
- Speeds up queries (10-50x)
- Reduces database load
- Enables horizontal scaling

**Hierarchical Consolidation**:
- Automatically summarizes at daily/weekly/monthly levels
- Reduces information overload
- Mimics human memory consolidation
- Enables drill-down from summaries to details

Together, these features make TracingRAG performant and scalable for production use.
