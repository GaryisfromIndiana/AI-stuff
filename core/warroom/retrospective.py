"""Retrospective engine — post-directive analysis and learning extraction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Retrospective:
    """Full retrospective analysis."""
    what_went_well: list[str] = field(default_factory=list)
    what_went_wrong: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    effectiveness_score: float = 0.0
    plan_accuracy: float = 0.0
    cost_usd: float = 0.0


@dataclass
class PerformanceAnalysis:
    """Analysis of directive performance."""
    total_tasks: int = 0
    success_rate: float = 0.0
    avg_quality: float = 0.0
    total_cost: float = 0.0
    total_duration: float = 0.0
    efficiency_score: float = 0.0
    quality_score: float = 0.0
    cost_efficiency: float = 0.0
    time_accuracy: float = 0.0


@dataclass
class PlanVsActual:
    """Comparison of planned vs actual execution."""
    planned_tasks: int = 0
    actual_tasks: int = 0
    planned_waves: int = 0
    actual_waves: int = 0
    planned_cost: float = 0.0
    actual_cost: float = 0.0
    variance_analysis: list[str] = field(default_factory=list)


@dataclass
class Lesson:
    """A specific lesson learned."""
    lesson: str
    category: str = "general"  # process, quality, cost, communication, technical
    applicable_to: list[str] = field(default_factory=list)
    severity: str = "medium"
    actionable: bool = True


@dataclass
class Improvement:
    """A specific improvement suggestion."""
    description: str
    category: str = "general"
    impact: str = "medium"
    effort: str = "medium"
    priority: str = "medium"


@dataclass
class TrendPoint:
    """A data point for trend analysis."""
    directive_id: str = ""
    effectiveness: float = 0.0
    quality: float = 0.0
    cost: float = 0.0
    timestamp: str = ""


@dataclass
class OutcomeScore:
    """Scoring of a directive outcome."""
    overall: float = 0.0
    completeness: float = 0.0
    quality: float = 0.0
    efficiency: float = 0.0
    timeliness: float = 0.0
    cost_effectiveness: float = 0.0


class RetrospectiveEngine:
    """Post-directive retrospective analysis and learning extraction.

    Analyzes what went well, what went wrong, extracts lessons,
    and feeds findings back into lieutenant memory for continuous improvement.
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id
        self._router = None

    def _get_router(self):
        if self._router is None:
            from llm.router import ModelRouter
            self._router = ModelRouter()
        return self._router

    def run_retrospective(
        self,
        directive_title: str,
        results: dict,
    ) -> Retrospective:
        """Run a full retrospective on a completed directive.

        Args:
            directive_title: Title of the directive.
            results: Execution results.

        Returns:
            Retrospective analysis.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()
        results_str = json.dumps(results, indent=2, default=str)[:6000]

        prompt = f"""Conduct a thorough retrospective on this completed directive.

Directive: {directive_title}

Results:
{results_str}

Analyze:
1. What went well — successful strategies, good decisions, effective collaboration
2. What went wrong — failures, bottlenecks, missed opportunities
3. Lessons learned — specific, actionable takeaways
4. Improvements — concrete changes for next time
5. Action items — immediate steps to implement improvements

Be specific and actionable. Focus on patterns, not individual incidents.

Respond as JSON:
{{
    "what_went_well": ["..."],
    "what_went_wrong": ["..."],
    "lessons_learned": ["..."],
    "improvements": ["..."],
    "action_items": [{{"description": "...", "priority": "high", "category": "process"}}],
    "effectiveness_score": 0.7,
    "plan_accuracy": 0.7
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                system_prompt="You are a retrospective facilitator. Be honest, specific, and constructive.",
                temperature=0.3,
                max_tokens=2500,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis", complexity="complex"))

            from llm.schemas import safe_json_loads
            data = safe_json_loads(response.content)

            return Retrospective(
                what_went_well=data.get("what_went_well", []),
                what_went_wrong=data.get("what_went_wrong", []),
                lessons_learned=data.get("lessons_learned", []),
                improvements=data.get("improvements", []),
                action_items=data.get("action_items", []),
                effectiveness_score=float(data.get("effectiveness_score", 0.5)),
                plan_accuracy=float(data.get("plan_accuracy", 0.5)),
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Retrospective failed: %s", e)
            return Retrospective(what_went_wrong=[f"Retrospective generation failed: {e}"])

    def analyze_performance(self, task_results: list[dict]) -> PerformanceAnalysis:
        """Analyze performance metrics from task results.

        Args:
            task_results: List of task result dicts.

        Returns:
            PerformanceAnalysis.
        """
        if not task_results:
            return PerformanceAnalysis()

        total = len(task_results)
        succeeded = sum(1 for t in task_results if t.get("success"))
        qualities = [t.get("quality_score", 0) for t in task_results if t.get("quality_score")]
        costs = [t.get("cost_usd", 0) for t in task_results]
        durations = [t.get("execution_time", 0) for t in task_results if t.get("execution_time")]

        success_rate = succeeded / total if total > 0 else 0
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        total_cost = sum(costs)
        total_duration = sum(durations)

        # Efficiency: quality per dollar spent
        cost_efficiency = avg_quality / max(total_cost, 0.001) if total_cost > 0 else 0

        return PerformanceAnalysis(
            total_tasks=total,
            success_rate=success_rate,
            avg_quality=avg_quality,
            total_cost=total_cost,
            total_duration=total_duration,
            efficiency_score=success_rate * 0.5 + avg_quality * 0.5,
            quality_score=avg_quality,
            cost_efficiency=min(1.0, cost_efficiency / 10),  # Normalize
        )

    def extract_lessons(self, results: dict) -> list[Lesson]:
        """Extract specific lessons from directive results.

        Args:
            results: Directive execution results.

        Returns:
            List of lessons.
        """
        lessons = []

        wave_results = results.get("wave_results", [])
        for wave in wave_results:
            success_rate = wave.get("success_rate", 0)
            if success_rate < 0.5:
                lessons.append(Lesson(
                    lesson=f"Wave {wave.get('wave_number', '?')} had low success rate ({success_rate:.0%}). "
                           f"Consider breaking tasks into smaller pieces or improving prompts.",
                    category="quality",
                    severity="high",
                ))
            elif success_rate == 1.0:
                lessons.append(Lesson(
                    lesson=f"Wave {wave.get('wave_number', '?')} achieved 100% success. "
                           f"The task decomposition worked well for this wave structure.",
                    category="process",
                    severity="low",
                ))

        total_cost = results.get("total_cost", 0)
        if total_cost > 10:
            lessons.append(Lesson(
                lesson=f"Directive cost ${total_cost:.2f}. Consider using more cost-efficient models for simple tasks.",
                category="cost",
                severity="medium",
                applicable_to=["cost_optimization"],
            ))

        retro = results.get("retrospective", {})
        for item in retro.get("lessons_learned", []):
            if isinstance(item, str):
                lessons.append(Lesson(lesson=item, category="general"))

        return lessons

    def identify_improvements(self, results: dict) -> list[Improvement]:
        """Identify specific improvements from results.

        Args:
            results: Directive results.

        Returns:
            List of improvements.
        """
        improvements = []

        retro = results.get("retrospective", {})
        for item in retro.get("improvements", []):
            if isinstance(item, str):
                improvements.append(Improvement(description=item))

        # Analyze patterns
        wave_results = results.get("wave_results", [])
        if len(wave_results) > 3:
            improvements.append(Improvement(
                description="Consider consolidating waves — many small waves add overhead",
                category="process",
                impact="medium",
                effort="low",
            ))

        return improvements

    def generate_action_items(self, retrospective: Retrospective) -> list[dict]:
        """Generate action items from a retrospective.

        Args:
            retrospective: Completed retrospective.

        Returns:
            List of action item dicts.
        """
        items = list(retrospective.action_items)

        # Generate additional items from lessons
        for lesson in retrospective.lessons_learned:
            items.append({
                "description": f"Implement lesson: {lesson[:100]}",
                "priority": "medium",
                "category": "improvement",
                "source": "retrospective",
            })

        return items

    def compare_to_plan(self, plan: dict, results: dict) -> PlanVsActual:
        """Compare planned execution to actual results.

        Args:
            plan: Original execution plan.
            results: Actual execution results.

        Returns:
            PlanVsActual comparison.
        """
        planned_waves = len(plan.get("waves", []))
        planned_tasks = sum(len(w.get("tasks", [])) for w in plan.get("waves", []))

        actual_waves = len(results.get("wave_results", []))
        actual_tasks = sum(
            len(w.get("tasks", []))
            for w in results.get("wave_results", [])
        )

        variance = []
        if actual_tasks != planned_tasks:
            variance.append(f"Tasks: planned {planned_tasks}, actual {actual_tasks}")
        if actual_waves != planned_waves:
            variance.append(f"Waves: planned {planned_waves}, actual {actual_waves}")

        return PlanVsActual(
            planned_tasks=planned_tasks,
            actual_tasks=actual_tasks,
            planned_waves=planned_waves,
            actual_waves=actual_waves,
            actual_cost=results.get("total_cost", 0),
            variance_analysis=variance,
        )

    def score_directive_outcome(self, results: dict) -> OutcomeScore:
        """Score the overall outcome of a directive.

        Args:
            results: Directive results.

        Returns:
            OutcomeScore.
        """
        wave_results = results.get("wave_results", [])
        all_tasks = []
        for wave in wave_results:
            all_tasks.extend(wave.get("tasks", []))

        if not all_tasks:
            return OutcomeScore()

        success_count = sum(1 for t in all_tasks if t.get("success"))
        total = len(all_tasks)
        qualities = [t.get("quality_score", 0) for t in all_tasks if t.get("quality_score")]

        completeness = success_count / total if total > 0 else 0
        quality = sum(qualities) / len(qualities) if qualities else 0
        efficiency = completeness * quality  # Combined metric

        return OutcomeScore(
            overall=(completeness * 0.4 + quality * 0.4 + efficiency * 0.2),
            completeness=completeness,
            quality=quality,
            efficiency=efficiency,
        )

    def feed_back_to_memory(
        self,
        retrospective: Retrospective,
        lieutenant_ids: list[str],
    ) -> int:
        """Store retrospective lessons in lieutenant memory.

        Args:
            retrospective: Completed retrospective.
            lieutenant_ids: Lieutenants who participated.

        Returns:
            Number of memories stored.
        """
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.empire_id)
        stored = 0

        for lesson in retrospective.lessons_learned:
            for lt_id in lieutenant_ids:
                mm.store(
                    content=f"[Retrospective] {lesson}",
                    memory_type="experiential",
                    lieutenant_id=lt_id,
                    title="Retrospective lesson",
                    category="retrospective",
                    importance=0.75,
                    tags=["retrospective", "lesson"],
                    source_type="retrospective",
                )
                stored += 1

        for improvement in retrospective.improvements:
            mm.store(
                content=f"[Improvement] {improvement}",
                memory_type="design",
                category="improvement",
                importance=0.7,
                tags=["improvement", "retrospective"],
                source_type="retrospective",
            )
            stored += 1

        logger.info("Fed %d retrospective memories back to %d lieutenants", stored, len(lieutenant_ids))
        return stored

    def get_retrospective_trends(self, limit: int = 10) -> list[TrendPoint]:
        """Get trends across recent retrospectives.

        Args:
            limit: Number of recent directives to analyze.

        Returns:
            List of trend points.
        """
        try:
            from db.engine import get_session
            from db.repositories.directive import DirectiveRepository

            session = get_session()
            repo = DirectiveRepository(session)
            completed = repo.get_completed(self.empire_id, days=90, limit=limit)

            points = []
            for d in completed:
                points.append(TrendPoint(
                    directive_id=d.id,
                    effectiveness=d.quality_score or 0,
                    quality=d.quality_score or 0,
                    cost=d.total_cost_usd,
                    timestamp=d.completed_at.isoformat() if d.completed_at else "",
                ))

            return points

        except Exception as e:
            logger.warning("Could not get trends: %s", e)
            return []
