"""Experiential memory (Tier 2) — stores lessons learned, outcomes, patterns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class Learning:
    content: str
    context: str = ""
    applicable_to: list[str] = field(default_factory=list)
    source_task_id: str = ""


@dataclass
class SuccessPattern:
    pattern: str
    task_types: list[str] = field(default_factory=list)
    effectiveness: float = 0.7
    usage_count: int = 1


@dataclass
class BestPractice:
    practice: str
    domain: str = ""
    evidence: list[str] = field(default_factory=list)
    effectiveness: float = 0.7


@dataclass
class EffectivenessTrend:
    data_points: list[dict] = field(default_factory=list)
    trend_direction: str = "stable"  # improving, stable, declining
    avg_effectiveness: float = 0.5


class ExperientialMemory:
    """Experiential memory — stores lessons, outcomes, success/failure patterns.

    Learns from task execution to improve future performance.
    Detects recurring patterns and reinforces effective strategies.
    """

    def __init__(self, memory_manager: MemoryManager, lieutenant_id: str = ""):
        self.mm = memory_manager
        self.lieutenant_id = lieutenant_id

    def store_outcome(
        self,
        task_id: str,
        outcome: str,
        success: bool,
        learnings: list[str] | None = None,
        task_type: str = "",
    ) -> list[dict]:
        """Store a task outcome with extracted learnings."""
        created = []

        # Store the outcome
        created.append(self.mm.store(
            content=f"{'SUCCESS' if success else 'FAILURE'}: {outcome[:2000]}",
            memory_type="experiential",
            lieutenant_id=self.lieutenant_id,
            title=f"Outcome: {'success' if success else 'failure'}",
            category="outcome",
            importance=0.6 if success else 0.7,
            tags=["outcome", "success" if success else "failure", task_type] if task_type else ["outcome"],
            source_task_id=task_id,
            metadata={"success": success, "task_type": task_type},
        ))

        # Store individual learnings
        for learning in (learnings or []):
            created.append(self.store_lesson(Learning(
                content=learning,
                context=outcome[:200],
                source_task_id=task_id,
            )))

        return created

    def store_lesson(self, lesson: Learning) -> dict:
        """Store a lesson learned."""
        return self.mm.store(
            content=lesson.content,
            memory_type="experiential",
            lieutenant_id=self.lieutenant_id,
            title=f"Lesson: {lesson.content[:80]}",
            category="lesson",
            importance=0.7,
            tags=["lesson"] + lesson.applicable_to,
            source_task_id=lesson.source_task_id,
            metadata={"context": lesson.context, "applicable_to": lesson.applicable_to},
        )

    def store_failure(
        self,
        task_id: str,
        error: str,
        root_cause: str = "",
        fix: str = "",
    ) -> dict:
        """Store a failure record with root cause and fix."""
        content = f"Failure: {error}"
        if root_cause:
            content += f"\nRoot cause: {root_cause}"
        if fix:
            content += f"\nFix: {fix}"

        return self.mm.store(
            content=content,
            memory_type="experiential",
            lieutenant_id=self.lieutenant_id,
            title=f"Failure: {error[:80]}",
            category="failure",
            importance=0.8,  # Failures are valuable
            tags=["failure", "error"],
            source_task_id=task_id,
            metadata={"error": error, "root_cause": root_cause, "fix": fix},
        )

    def store_success_pattern(self, pattern: SuccessPattern) -> dict:
        """Store a recurring success pattern."""
        return self.mm.store(
            content=f"Success pattern: {pattern.pattern}",
            memory_type="experiential",
            lieutenant_id=self.lieutenant_id,
            title=f"Pattern: {pattern.pattern[:80]}",
            category="pattern",
            importance=0.75,
            tags=["pattern", "success"] + pattern.task_types,
            metadata={
                "effectiveness": pattern.effectiveness,
                "usage_count": pattern.usage_count,
                "task_types": pattern.task_types,
            },
        )

    def query_lessons(self, task_type: str = "", domain: str = "", limit: int = 10) -> list[dict]:
        """Query relevant lessons for a task."""
        query = f"lesson {task_type} {domain}".strip()
        return self.mm.recall(
            query=query,
            memory_types=["experiential"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def query_failures(self, error_type: str = "", limit: int = 10) -> list[dict]:
        """Query past failures for a type of error."""
        return self.mm.recall(
            query=f"failure {error_type}".strip(),
            memory_types=["experiential"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def query_success_patterns(self, task_type: str = "", limit: int = 10) -> list[dict]:
        """Query success patterns for a task type."""
        return self.mm.recall(
            query=f"pattern success {task_type}".strip(),
            memory_types=["experiential"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def extract_learnings(self, task_result: dict) -> list[Learning]:
        """Auto-extract learnings from a task result.

        Uses heuristics to identify lessons from the task content.
        """
        content = task_result.get("content", "")
        success = task_result.get("success", False)
        task_id = task_result.get("task_id", "")
        learnings = []

        # Extract from quality feedback
        issues = task_result.get("quality_details", {}).get("issues", [])
        for issue in issues:
            if isinstance(issue, dict):
                desc = issue.get("description", "")
            else:
                desc = str(issue)
            if desc:
                learnings.append(Learning(
                    content=f"Quality issue to avoid: {desc}",
                    source_task_id=task_id,
                    applicable_to=["quality"],
                ))

        suggestions = task_result.get("quality_details", {}).get("suggestions", [])
        for suggestion in suggestions:
            learnings.append(Learning(
                content=f"Improvement suggestion: {suggestion}",
                source_task_id=task_id,
                applicable_to=["improvement"],
            ))

        if not success and task_result.get("error"):
            learnings.append(Learning(
                content=f"Error to watch for: {task_result['error']}",
                source_task_id=task_id,
                applicable_to=["error_prevention"],
            ))

        return learnings

    def get_best_practices(self, domain: str = "", limit: int = 10) -> list[BestPractice]:
        """Get best practices derived from experiential memory."""
        patterns = self.query_success_patterns(domain, limit=limit)
        return [
            BestPractice(
                practice=p.get("content", ""),
                domain=domain,
                effectiveness=p.get("importance", 0.5),
            )
            for p in patterns
        ]

    def get_effectiveness_trend(self, limit: int = 20) -> EffectivenessTrend:
        """Get effectiveness trend over recent outcomes."""
        outcomes = self.mm.recall(
            query="outcome",
            memory_types=["experiential"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

        data_points = []
        for o in outcomes:
            meta = o.get("metadata", {}) if isinstance(o.get("metadata"), dict) else {}
            data_points.append({
                "success": meta.get("success", False),
                "importance": o.get("importance", 0.5),
                "created_at": o.get("created_at"),
            })

        if len(data_points) < 3:
            return EffectivenessTrend(data_points=data_points, trend_direction="insufficient_data")

        # Calculate trend
        recent = data_points[:len(data_points)//2]
        older = data_points[len(data_points)//2:]

        recent_rate = sum(1 for d in recent if d.get("success")) / len(recent) if recent else 0
        older_rate = sum(1 for d in older if d.get("success")) / len(older) if older else 0

        if recent_rate > older_rate + 0.1:
            direction = "improving"
        elif recent_rate < older_rate - 0.1:
            direction = "declining"
        else:
            direction = "stable"

        return EffectivenessTrend(
            data_points=data_points,
            trend_direction=direction,
            avg_effectiveness=(recent_rate + older_rate) / 2,
        )
