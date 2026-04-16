"""Wave execution — executes tasks in ordered waves with dependency management."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WaveTask:
    """A task within a wave."""
    title: str = ""
    description: str = ""
    assigned_lieutenant: str = ""
    task_type: str = "general"
    estimated_tokens: int = 2000
    dependencies: list[str] = field(default_factory=list)


@dataclass
class WaveResult:
    """Result of executing one wave."""
    wave_number: int = 0
    task_results: list[dict] = field(default_factory=list)
    success_rate: float = 0.0
    cost: float = 0.0
    duration_seconds: float = 0.0
    tasks_total: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0


@dataclass
class WaveStatus:
    """Current status of a wave."""
    wave_number: int = 0
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    in_progress: int = 0
    status: str = "pending"


class WaveExecutor:
    """Executes tasks in ordered waves with dependency management.

    Tasks within a wave can execute in parallel (they have no
    inter-wave dependencies). Waves must execute sequentially.
    Failed tasks in a wave get one more retry with successful
    sibling outputs as context.
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def execute_wave(
        self,
        wave_data: dict,
        context: dict | None = None,
        previous_results: list[dict] | None = None,
    ) -> WaveResult:
        """Execute a single wave of tasks.

        Args:
            wave_data: Wave data with tasks.
            context: Directive context.
            previous_results: Results from previous waves.

        Returns:
            WaveResult.
        """
        start_time = time.time()
        wave_number = wave_data.get("wave_number", 1)
        tasks = wave_data.get("tasks", [])

        logger.info("Executing wave %d with %d tasks", wave_number, len(tasks))

        # Execute each task
        task_results = []
        total_cost = 0.0

        for task_data in tasks:
            result = self._execute_task(task_data, context, previous_results)
            task_results.append(result)
            total_cost += result.get("cost_usd", 0)

        # Retry failed tasks with sibling context
        failed_tasks = [t for t in task_results if not t.get("success")]
        succeeded_tasks = [t for t in task_results if t.get("success")]

        if failed_tasks and succeeded_tasks:
            logger.info("Retrying %d failed tasks with sibling context", len(failed_tasks))
            for i, failed in enumerate(failed_tasks):
                retry_result = self._retry_with_siblings(
                    failed,
                    succeeded_tasks,
                    context,
                )
                if retry_result.get("success"):
                    # Replace failed result with successful retry
                    idx = task_results.index(failed)
                    task_results[idx] = retry_result
                    total_cost += retry_result.get("cost_usd", 0)

        succeeded = sum(1 for t in task_results if t.get("success"))
        duration = time.time() - start_time

        return WaveResult(
            wave_number=wave_number,
            task_results=task_results,
            success_rate=succeeded / max(len(task_results), 1),
            cost=total_cost,
            duration_seconds=duration,
            tasks_total=len(task_results),
            tasks_succeeded=succeeded,
            tasks_failed=len(task_results) - succeeded,
        )

    def execute_all_waves(
        self,
        waves: list[dict],
        context: dict | None = None,
    ) -> list[dict]:
        """Execute all waves in sequence.

        Args:
            waves: List of wave data dicts.
            context: Directive context.

        Returns:
            List of wave result dicts.
        """
        results = []
        previous_results: list[dict] = []

        for wave_data in waves:
            wave_result = self.execute_wave(wave_data, context, previous_results)

            result_dict = {
                "wave_number": wave_result.wave_number,
                "task_results": wave_result.task_results,
                "success_rate": wave_result.success_rate,
                "cost": wave_result.cost,
                "duration": wave_result.duration_seconds,
                "tasks_total": wave_result.tasks_total,
                "tasks_succeeded": wave_result.tasks_succeeded,
                "tasks_failed": wave_result.tasks_failed,
            }
            results.append(result_dict)

            # Collect results for next wave context
            for task_result in wave_result.task_results:
                if task_result.get("success"):
                    previous_results.append(task_result)

        return results

    def plan_waves(
        self,
        tasks: list[dict],
        dependencies: list[dict] | None = None,
    ) -> list[dict]:
        """Plan tasks into waves based on dependencies.

        Args:
            tasks: List of task dicts with optional dependency info.
            dependencies: Explicit dependency list.

        Returns:
            List of wave dicts.
        """
        if not tasks:
            return []

        # Build dependency graph
        task_titles = {t.get("title", f"task_{i}"): i for i, t in enumerate(tasks)}
        in_degree = {t.get("title", f"task_{i}"): 0 for i, t in enumerate(tasks)}
        adj: dict[str, list[str]] = {t.get("title", f"task_{i}"): [] for i, t in enumerate(tasks)}

        for task in tasks:
            title = task.get("title", "")
            for dep in task.get("dependencies", []):
                if dep in task_titles and dep != title:
                    adj[dep].append(title)
                    in_degree[title] = in_degree.get(title, 0) + 1

        # Topological sort into waves
        waves = []
        remaining = set(task_titles.keys())

        while remaining:
            # Find tasks with no unresolved dependencies
            wave_tasks = [t for t in remaining if in_degree.get(t, 0) == 0]

            if not wave_tasks:
                # Circular dependency — just add remaining
                wave_tasks = list(remaining)

            wave_data = {
                "wave_number": len(waves) + 1,
                "tasks": [tasks[task_titles[t]] for t in wave_tasks if t in task_titles],
            }
            waves.append(wave_data)

            # Remove wave tasks and update dependencies
            for t in wave_tasks:
                remaining.discard(t)
                for neighbor in adj.get(t, []):
                    in_degree[neighbor] = max(0, in_degree.get(neighbor, 0) - 1)

        return waves

    def replan_waves(
        self,
        remaining_tasks: list[dict],
        failed_tasks: list[dict],
    ) -> list[dict]:
        """Replan waves after failures.

        Args:
            remaining_tasks: Tasks still to execute.
            failed_tasks: Tasks that failed and need retry.

        Returns:
            New wave plan.
        """
        # Put failed tasks first (they need retry with fresh context)
        all_tasks = failed_tasks + remaining_tasks
        return self.plan_waves(all_tasks)

    def _execute_task(
        self,
        task_data: dict,
        context: dict | None,
        previous_results: list[dict] | None,
    ) -> dict:
        """Execute a single task within a wave."""
        from core.ace.engine import TaskInput
        from core.lieutenant.manager import LieutenantManager

        lt_manager = LieutenantManager(self.empire_id)

        title = task_data.get("title", "")
        description = task_data.get("description", "")
        assigned = task_data.get("assigned_to", "")

        # Find lieutenant
        lt = None
        if assigned:
            lt = lt_manager.get_lieutenant(assigned)
            if not lt:
                lt = lt_manager.find_best_lieutenant(description)
        else:
            lt = lt_manager.find_best_lieutenant(description)

        if not lt:
            return {
                "title": title,
                "success": False,
                "error": "No available lieutenant",
                "cost_usd": 0.0,
            }

        # Build task with context from previous waves
        full_description = description
        if previous_results:
            prev_context = "\n".join(
                f"- [{r.get('title', 'Previous')}]: {r.get('content', '')[:200]}"
                for r in previous_results[-5:]
            )
            full_description += f"\n\n## Context from Previous Tasks\n{prev_context}"

        task = TaskInput(
            title=title,
            description=full_description,
            task_type=task_data.get("task_type", "general"),
            max_tokens=task_data.get("estimated_tokens", 4096),
        )

        try:
            result = lt.execute_task(task)

            # Record in DB
            self._record_task(
                directive_id=context.get("directive_id", "") if context else "",
                lieutenant_id=lt.id,
                task_data=task_data,
                result=result,
            )

            return {
                "title": title,
                "success": result.success,
                "content": result.content,
                "quality_score": result.quality_score,
                "cost_usd": result.cost_usd,
                "model_used": result.model_used,
                "lieutenant": lt.name,
                "execution_time": result.execution_time_seconds,
            }

        except Exception as e:
            logger.error("Task execution failed: %s", e)
            return {
                "title": title,
                "success": False,
                "error": str(e),
                "cost_usd": 0.0,
            }

    def _retry_with_siblings(
        self,
        failed_task: dict,
        successful_siblings: list[dict],
        context: dict | None,
    ) -> dict:
        """Retry a failed task using successful sibling outputs as context."""
        sibling_context = "\n\n".join(
            f"## Successful sibling: {s.get('title', '')}\n{s.get('content', '')[:500]}"
            for s in successful_siblings[:3]
        )

        enriched_task = dict(failed_task)
        original_desc = enriched_task.get("description", "")
        enriched_task["description"] = (
            f"{original_desc}\n\n"
            f"## Previous Attempt Failed\n"
            f"Error: {failed_task.get('error', 'Quality too low')}\n\n"
            f"## Context from Successful Sibling Tasks\n"
            f"{sibling_context}"
        )

        return self._execute_task(enriched_task, context, successful_siblings)

    def _record_task(
        self,
        directive_id: str,
        lieutenant_id: str,
        task_data: dict,
        result: Any,
    ) -> None:
        """Record a task execution in the database."""
        try:
            from db.engine import session_scope
            from db.models import Task

            with session_scope() as session:
                task = Task(
                    directive_id=directive_id or None,
                    lieutenant_id=lieutenant_id,
                    title=task_data.get("title", ""),
                    description=task_data.get("description", "")[:5000],
                    status="completed" if result.success else "failed",
                    task_type=task_data.get("task_type", "general"),
                    wave_number=task_data.get("wave_number", 0),
                    model_used=result.model_used,
                    tokens_input=result.tokens_input,
                    tokens_output=result.tokens_output,
                    cost_usd=result.cost_usd,
                    quality_score=result.quality_score,
                    execution_time_seconds=result.execution_time_seconds,
                    output_json={"content": result.content[:10000]},
                    error_log_json=result.error_log if hasattr(result, "error_log") else [],
                )
                session.add(task)

        except Exception as e:
            logger.warning("Failed to record task: %s", e)

    def get_wave_status(self, directive_id: str, wave_number: int) -> WaveStatus:
        """Get status of a specific wave."""
        try:
            from db.engine import get_session
            from db.repositories.task import TaskRepository

            session = get_session()
            repo = TaskRepository(session)
            tasks = repo.get_by_wave(directive_id, wave_number)

            completed = sum(1 for t in tasks if t.status == "completed")
            failed = sum(1 for t in tasks if t.status == "failed")
            in_progress = sum(1 for t in tasks if t.status in ("executing", "planning"))

            return WaveStatus(
                wave_number=wave_number,
                total_tasks=len(tasks),
                completed=completed,
                failed=failed,
                in_progress=in_progress,
                status="completed" if completed == len(tasks) else "in_progress" if in_progress else "pending",
            )

        except Exception:
            return WaveStatus(wave_number=wave_number)
