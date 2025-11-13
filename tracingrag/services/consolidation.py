"""Hierarchical memory consolidation service

This service implements sleep-like consolidation that automatically
summarizes and rolls up memory states at daily, weekly, and monthly intervals.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select

from tracingrag.core.models.promotion import (
    PromotionRequest,
    PromotionTrigger,
)
from tracingrag.services.llm import LLMClient
from tracingrag.services.memory import MemoryService
from tracingrag.services.promotion import PromotionService
from tracingrag.storage.database import get_session
from tracingrag.storage.models import MemoryStateDB


class ConsolidationLevel(str, Enum):
    """Consolidation level"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ConsolidationConfig(BaseModel):
    """Configuration for consolidation"""

    daily_threshold: int = Field(default=10, description="Min states for daily consolidation")
    weekly_threshold: int = Field(default=5, description="Min daily summaries for weekly")
    monthly_threshold: int = Field(default=4, description="Min weekly summaries for monthly")

    daily_schedule_hour: int = Field(default=2, ge=0, le=23, description="Hour to run daily (UTC)")
    weekly_schedule_day: int = Field(default=6, ge=0, le=6, description="Day to run weekly (0=Monday)")
    monthly_schedule_day: int = Field(default=1, ge=1, le=28, description="Day of month for monthly")

    consolidation_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="LLM model for consolidation synthesis",
    )

    enabled_levels: list[ConsolidationLevel] = Field(
        default=[
            ConsolidationLevel.DAILY,
            ConsolidationLevel.WEEKLY,
            ConsolidationLevel.MONTHLY,
        ],
        description="Enabled consolidation levels",
    )


class ConsolidationCandidate(BaseModel):
    """Candidate for consolidation"""

    topic: str
    level: ConsolidationLevel
    state_count: int
    earliest_timestamp: datetime
    latest_timestamp: datetime
    should_consolidate: bool
    reasoning: str


class ConsolidationResult(BaseModel):
    """Result of consolidation operation"""

    success: bool
    topic: str
    level: ConsolidationLevel
    states_consolidated: int
    new_state_id: UUID | None = None
    summary: str | None = None
    error_message: str | None = None


class ConsolidationService:
    """Service for hierarchical memory consolidation"""

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        memory_service: MemoryService | None = None,
        promotion_service: PromotionService | None = None,
        llm_client: LLMClient | None = None,
    ):
        """Initialize consolidation service

        Args:
            config: Consolidation configuration
            memory_service: Memory service instance
            promotion_service: Promotion service instance
            llm_client: LLM client instance
        """
        self.config = config or ConsolidationConfig()
        self.memory_service = memory_service or MemoryService()
        self.promotion_service = promotion_service or PromotionService()
        self.llm_client = llm_client or LLMClient()

    # ========================================================================
    # Finding Consolidation Candidates
    # ========================================================================

    async def find_daily_candidates(self, date: datetime | None = None) -> list[ConsolidationCandidate]:
        """Find topics that need daily consolidation

        Args:
            date: Date to consolidate (default: yesterday)

        Returns:
            List of consolidation candidates
        """
        if date is None:
            date = datetime.utcnow() - timedelta(days=1)

        # Get start and end of day
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        candidates = []

        async with get_session() as session:
            # Find topics with states created during the day
            query = (
                select(
                    MemoryStateDB.topic,
                    func.count(MemoryStateDB.id).label("state_count"),
                    func.min(MemoryStateDB.timestamp).label("earliest"),
                    func.max(MemoryStateDB.timestamp).label("latest"),
                )
                .where(
                    and_(
                        MemoryStateDB.timestamp >= start_of_day,
                        MemoryStateDB.timestamp < end_of_day,
                        ~MemoryStateDB.tags.contains(["daily_summary"]),
                    )
                )
                .group_by(MemoryStateDB.topic)
                .having(func.count(MemoryStateDB.id) >= self.config.daily_threshold)
            )

            result = await session.execute(query)
            rows = result.all()

            for row in rows:
                # Check if daily summary already exists for this date
                existing_query = select(MemoryStateDB).where(
                    and_(
                        MemoryStateDB.topic == f"{row.topic}:daily:{start_of_day.date()}",
                        MemoryStateDB.tags.contains(["daily_summary"]),
                    )
                )
                existing = await session.execute(existing_query)
                if existing.scalar():
                    continue  # Already consolidated

                candidates.append(
                    ConsolidationCandidate(
                        topic=row.topic,
                        level=ConsolidationLevel.DAILY,
                        state_count=row.state_count,
                        earliest_timestamp=row.earliest,
                        latest_timestamp=row.latest,
                        should_consolidate=True,
                        reasoning=f"Found {row.state_count} states for {start_of_day.date()}",
                    )
                )

        return candidates

    async def find_weekly_candidates(self, date: datetime | None = None) -> list[ConsolidationCandidate]:
        """Find topics that need weekly consolidation

        Args:
            date: End date of week (default: last Sunday)

        Returns:
            List of consolidation candidates
        """
        if date is None:
            today = datetime.utcnow()
            # Go back to last Sunday
            days_since_sunday = (today.weekday() + 1) % 7
            date = today - timedelta(days=days_since_sunday if days_since_sunday > 0 else 7)

        # Get start and end of week
        start_of_week = date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6)
        end_of_week = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        candidates = []

        async with get_session() as session:
            # Find topics with daily summaries for the week
            query = (
                select(
                    MemoryStateDB.topic,
                    func.count(MemoryStateDB.id).label("summary_count"),
                    func.min(MemoryStateDB.timestamp).label("earliest"),
                    func.max(MemoryStateDB.timestamp).label("latest"),
                )
                .where(
                    and_(
                        MemoryStateDB.timestamp >= start_of_week,
                        MemoryStateDB.timestamp <= end_of_week,
                        MemoryStateDB.tags.contains(["daily_summary"]),
                    )
                )
                .group_by(MemoryStateDB.topic)
                .having(func.count(MemoryStateDB.id) >= self.config.weekly_threshold)
            )

            result = await session.execute(query)
            rows = result.all()

            for row in rows:
                # Extract base topic (remove :daily:date suffix)
                base_topic = row.topic.split(":daily:")[0] if ":daily:" in row.topic else row.topic

                # Check if weekly summary already exists
                week_id = f"{start_of_week.isocalendar()[0]}-W{start_of_week.isocalendar()[1]}"
                existing_query = select(MemoryStateDB).where(
                    and_(
                        MemoryStateDB.topic == f"{base_topic}:weekly:{week_id}",
                        MemoryStateDB.tags.contains(["weekly_summary"]),
                    )
                )
                existing = await session.execute(existing_query)
                if existing.scalar():
                    continue  # Already consolidated

                candidates.append(
                    ConsolidationCandidate(
                        topic=base_topic,
                        level=ConsolidationLevel.WEEKLY,
                        state_count=row.summary_count,
                        earliest_timestamp=row.earliest,
                        latest_timestamp=row.latest,
                        should_consolidate=True,
                        reasoning=f"Found {row.summary_count} daily summaries for week {week_id}",
                    )
                )

        return candidates

    async def find_monthly_candidates(self, date: datetime | None = None) -> list[ConsolidationCandidate]:
        """Find topics that need monthly consolidation

        Args:
            date: Date in month to consolidate (default: last month)

        Returns:
            List of consolidation candidates
        """
        if date is None:
            today = datetime.utcnow()
            # Go to first day of this month, then back one day to get last month
            first_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            date = first_of_month - timedelta(days=1)

        # Get start and end of month
        start_of_month = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Get last day of month
        if date.month == 12:
            end_of_month = date.replace(year=date.year + 1, month=1, day=1) - timedelta(seconds=1)
        else:
            end_of_month = date.replace(month=date.month + 1, day=1) - timedelta(seconds=1)

        candidates = []

        async with get_session() as session:
            # Find topics with weekly summaries for the month
            query = (
                select(
                    MemoryStateDB.topic,
                    func.count(MemoryStateDB.id).label("summary_count"),
                    func.min(MemoryStateDB.timestamp).label("earliest"),
                    func.max(MemoryStateDB.timestamp).label("latest"),
                )
                .where(
                    and_(
                        MemoryStateDB.timestamp >= start_of_month,
                        MemoryStateDB.timestamp <= end_of_month,
                        MemoryStateDB.tags.contains(["weekly_summary"]),
                    )
                )
                .group_by(MemoryStateDB.topic)
                .having(func.count(MemoryStateDB.id) >= self.config.monthly_threshold)
            )

            result = await session.execute(query)
            rows = result.all()

            for row in rows:
                # Extract base topic
                base_topic = row.topic.split(":weekly:")[0] if ":weekly:" in row.topic else row.topic

                # Check if monthly summary already exists
                month_id = f"{date.year}-{date.month:02d}"
                existing_query = select(MemoryStateDB).where(
                    and_(
                        MemoryStateDB.topic == f"{base_topic}:monthly:{month_id}",
                        MemoryStateDB.tags.contains(["monthly_summary"]),
                    )
                )
                existing = await session.execute(existing_query)
                if existing.scalar():
                    continue  # Already consolidated

                candidates.append(
                    ConsolidationCandidate(
                        topic=base_topic,
                        level=ConsolidationLevel.MONTHLY,
                        state_count=row.summary_count,
                        earliest_timestamp=row.earliest,
                        latest_timestamp=row.latest,
                        should_consolidate=True,
                        reasoning=f"Found {row.summary_count} weekly summaries for {month_id}",
                    )
                )

        return candidates

    async def find_all_candidates(self) -> dict[ConsolidationLevel, list[ConsolidationCandidate]]:
        """Find all consolidation candidates across all levels

        Returns:
            Dictionary mapping level to candidates
        """
        results = {}

        if ConsolidationLevel.DAILY in self.config.enabled_levels:
            results[ConsolidationLevel.DAILY] = await self.find_daily_candidates()

        if ConsolidationLevel.WEEKLY in self.config.enabled_levels:
            results[ConsolidationLevel.WEEKLY] = await self.find_weekly_candidates()

        if ConsolidationLevel.MONTHLY in self.config.enabled_levels:
            results[ConsolidationLevel.MONTHLY] = await self.find_monthly_candidates()

        return results

    # ========================================================================
    # Consolidation Execution
    # ========================================================================

    async def consolidate(
        self, candidate: ConsolidationCandidate
    ) -> ConsolidationResult:
        """Execute consolidation for a candidate

        Args:
            candidate: Consolidation candidate

        Returns:
            Consolidation result
        """
        try:
            # Determine new topic name
            if candidate.level == ConsolidationLevel.DAILY:
                date_str = candidate.earliest_timestamp.date().isoformat()
                new_topic = f"{candidate.topic}:daily:{date_str}"
                summary_tag = "daily_summary"
            elif candidate.level == ConsolidationLevel.WEEKLY:
                week = candidate.earliest_timestamp.isocalendar()
                week_id = f"{week[0]}-W{week[1]:02d}"
                new_topic = f"{candidate.topic}:weekly:{week_id}"
                summary_tag = "weekly_summary"
            else:  # MONTHLY
                month_id = f"{candidate.earliest_timestamp.year}-{candidate.earliest_timestamp.month:02d}"
                new_topic = f"{candidate.topic}:monthly:{month_id}"
                summary_tag = "monthly_summary"

            # Use promotion service to consolidate
            request = PromotionRequest(
                topic=candidate.topic,
                reason=f"{candidate.level.value} consolidation: {candidate.reasoning}",
                trigger=PromotionTrigger.AUTO_TIME_BASED,
                include_related=False,  # Focus on time range only
                max_sources=50,  # Allow more sources for consolidation
            )

            result = await self.promotion_service.promote_memory(request)

            if result.success and result.new_state_id:
                # Update the new state to use consolidation topic and add summary tag
                state = await self.memory_service.get_memory_state(result.new_state_id)
                if state:
                    # Create new state with consolidation topic
                    consolidated_state = await self.memory_service.create_memory_state(
                        topic=new_topic,
                        content=state.content,
                        parent_state_id=result.previous_state_id,
                        metadata={
                            "consolidation_level": candidate.level.value,
                            "original_topic": candidate.topic,
                            "states_consolidated": candidate.state_count,
                            "period_start": candidate.earliest_timestamp.isoformat(),
                            "period_end": candidate.latest_timestamp.isoformat(),
                        },
                        tags=[summary_tag, "consolidated"],
                        confidence=result.confidence,
                        source="consolidation_service",
                    )

                    return ConsolidationResult(
                        success=True,
                        topic=candidate.topic,
                        level=candidate.level,
                        states_consolidated=candidate.state_count,
                        new_state_id=consolidated_state.id,
                        summary=state.content,
                    )

            return ConsolidationResult(
                success=False,
                topic=candidate.topic,
                level=candidate.level,
                states_consolidated=0,
                error_message=result.error_message or "Promotion failed",
            )

        except Exception as e:
            return ConsolidationResult(
                success=False,
                topic=candidate.topic,
                level=candidate.level,
                states_consolidated=0,
                error_message=str(e),
            )

    async def consolidate_batch(
        self, candidates: list[ConsolidationCandidate]
    ) -> list[ConsolidationResult]:
        """Consolidate multiple candidates

        Args:
            candidates: List of candidates to consolidate

        Returns:
            List of consolidation results
        """
        tasks = [self.consolidate(candidate) for candidate in candidates]
        return await asyncio.gather(*tasks)

    async def run_daily_consolidation(self, date: datetime | None = None) -> list[ConsolidationResult]:
        """Run daily consolidation

        Args:
            date: Date to consolidate (default: yesterday)

        Returns:
            List of consolidation results
        """
        candidates = await self.find_daily_candidates(date)
        return await self.consolidate_batch(candidates)

    async def run_weekly_consolidation(self, date: datetime | None = None) -> list[ConsolidationResult]:
        """Run weekly consolidation

        Args:
            date: End date of week (default: last Sunday)

        Returns:
            List of consolidation results
        """
        candidates = await self.find_weekly_candidates(date)
        return await self.consolidate_batch(candidates)

    async def run_monthly_consolidation(self, date: datetime | None = None) -> list[ConsolidationResult]:
        """Run monthly consolidation

        Args:
            date: Date in month to consolidate (default: last month)

        Returns:
            List of consolidation results
        """
        candidates = await self.find_monthly_candidates(date)
        return await self.consolidate_batch(candidates)

    async def run_all_consolidations(self) -> dict[ConsolidationLevel, list[ConsolidationResult]]:
        """Run all enabled consolidations

        Returns:
            Dictionary mapping level to results
        """
        results = {}

        if ConsolidationLevel.DAILY in self.config.enabled_levels:
            results[ConsolidationLevel.DAILY] = await self.run_daily_consolidation()

        if ConsolidationLevel.WEEKLY in self.config.enabled_levels:
            results[ConsolidationLevel.WEEKLY] = await self.run_weekly_consolidation()

        if ConsolidationLevel.MONTHLY in self.config.enabled_levels:
            results[ConsolidationLevel.MONTHLY] = await self.run_monthly_consolidation()

        return results

    # ========================================================================
    # Drill-Down Queries
    # ========================================================================

    async def get_detailed_states(
        self, consolidated_topic: str
    ) -> list[MemoryStateDB]:
        """Get detailed states that were consolidated into a summary

        Args:
            consolidated_topic: Topic of consolidated summary

        Returns:
            List of detailed memory states
        """
        # Parse consolidated topic to get original topic and time range
        parts = consolidated_topic.split(":")
        if len(parts) < 3:
            return []

        original_topic = parts[0]
        level = parts[1]  # daily, weekly, or monthly
        period = parts[2]

        # Get the consolidated state to extract metadata
        consolidated_state = await self.memory_service.get_latest_state(consolidated_topic)
        if not consolidated_state or not consolidated_state.custom_metadata:
            return []

        metadata = consolidated_state.custom_metadata
        period_start = datetime.fromisoformat(metadata.get("period_start", ""))
        period_end = datetime.fromisoformat(metadata.get("period_end", ""))

        # Query detailed states
        async with get_session() as session:
            query = (
                select(MemoryStateDB)
                .where(
                    and_(
                        MemoryStateDB.topic == original_topic,
                        MemoryStateDB.timestamp >= period_start,
                        MemoryStateDB.timestamp <= period_end,
                    )
                )
                .order_by(MemoryStateDB.timestamp)
            )

            result = await session.execute(query)
            return list(result.scalars().all())
