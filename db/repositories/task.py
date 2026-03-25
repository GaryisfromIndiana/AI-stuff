"""Task-specific repository with execution tracking and cost aggregation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select, func, and_, desc, asc

from db.models import Task
from db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class TaskRepository(BaseRepository[Task]):
    """Repository for Task entities with execution and cost queries."""

    model_class = Task

    def get_by_directive(
        self,
        directive_id: str,
        status: str | None = None,
        wave: int | None = None,
    ) -> list[Task]:
        """Get tasks for a directive with optional filters."""
        filters: dict[str, Any] = {"directive_id": directive_id}
        if status:
            filters["status"] = status
        if wave is not None:
            filters["wave_number"] = wave
        return self.find(filters=filters, order_by="wave_number", order_dir="asc")

    def get_by_lieutenant(
        self,
        lieutenant_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """Get tasks assigned to a lieutenant."""
        filters: dict[str, Any] = {"lieutenant_id": lieutenant_id}
        if status:
            filters["status"] = status
        return self.find(filters=filters, limit=limit)

    def get_by_wave(self, directive_id: str, wave_number: int) -> list[Task]:
        """Get all tasks in a specific wave."""
        return self.find(
            filters={"directive_id": directive_id, "wave_number": wave_number},
            order_by="created_at",
            order_dir="asc",
        )

    def get_failed_retryable(self, directive_id: str) -> list[Task]:
        """Get failed tasks that can be retried."""
        stmt = (
            select(Task)
            .where(and_(
                Task.directive_id == directive_id,
                Task.status == "failed",
                Task.retry_count < Task.max_retries,
            ))
            .order_by(asc(Task.wave_number))
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_pending_by_priority(self, lieutenant_id: str | None = None, limit: int = 20) -> list[Task]:
        """Get pending tasks ordered by priority."""
        stmt = select(Task).where(Task.status == "pending")
        if lieutenant_id:
            stmt = stmt.where(Task.lieutenant_id == lieutenant_id)
        stmt = stmt.order_by(asc(Task.priority), asc(Task.created_at)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def start_task(self, task_id: str, model: str | None = None) -> Task | None:
        """Mark task as started."""
        updates: dict[str, Any] = {
            "status": "executing",
            "started_at": datetime.now(timezone.utc),
            "pipeline_stage": "executing",
        }
        if model:
            updates["model_used"] = model
        return self.update(task_id, **updates)

    def complete_task(
        self,
        task_id: str,
        output: dict | None = None,
        quality_score: float | None = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        model_used: str | None = None,
    ) -> Task | None:
        """Mark task as completed with results."""
        now = datetime.now(timezone.utc)
        task = self.get(task_id)
        if task is None:
            return None

        execution_time = None
        if task.started_at:
            execution_time = (now - task.started_at).total_seconds()

        updates: dict[str, Any] = {
            "status": "completed",
            "completed_at": now,
            "pipeline_stage": "completed",
            "execution_time_seconds": execution_time,
            "tokens_input": task.tokens_input + tokens_input,
            "tokens_output": task.tokens_output + tokens_output,
            "cost_usd": task.cost_usd + cost_usd,
        }
        if output:
            updates["output_json"] = output
        if quality_score is not None:
            updates["quality_score"] = quality_score
        if model_used:
            updates["model_used"] = model_used

        return self.update(task_id, **updates)

    def fail_task(self, task_id: str, error: str, increment_retry: bool = True) -> Task | None:
        """Mark task as failed."""
        task = self.get(task_id)
        if task is None:
            return None

        error_log = list(task.error_log_json or [])
        error_log.append({
            "attempt": task.retry_count + 1,
            "error": error,
            "model": task.model_used,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        updates: dict[str, Any] = {
            "status": "failed",
            "last_error": error,
            "error_log_json": error_log,
            "completed_at": datetime.now(timezone.utc),
        }
        if increment_retry:
            updates["retry_count"] = task.retry_count + 1

        return self.update(task_id, **updates)

    def mark_retrying(self, task_id: str) -> Task | None:
        """Mark task as retrying."""
        return self.update(
            task_id,
            status="retrying",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
        )

    def get_cost_aggregation(
        self,
        empire_id: str | None = None,
        days: int = 30,
    ) -> dict:
        """Get aggregated cost data."""
        since = datetime.now(timezone.utc) - timedelta(days=days)

        stmt = (
            select(
                Task.model_used,
                Task.provider,
                func.count(Task.id).label("count"),
                func.sum(Task.cost_usd).label("total_cost"),
                func.sum(Task.tokens_input).label("total_input"),
                func.sum(Task.tokens_output).label("total_output"),
                func.avg(Task.quality_score).label("avg_quality"),
                func.avg(Task.execution_time_seconds).label("avg_time"),
            )
            .where(Task.created_at >= since)
            .group_by(Task.model_used, Task.provider)
        )
        results = self.session.execute(stmt).all()

        aggregation = {
            "by_model": {},
            "total_cost": 0.0,
            "total_tasks": 0,
            "total_tokens": 0,
            "period_days": days,
        }

        for row in results:
            model, provider, count, cost, t_in, t_out, avg_q, avg_t = row
            key = model or "unknown"
            aggregation["by_model"][key] = {
                "provider": provider,
                "count": count,
                "total_cost": float(cost or 0),
                "total_tokens_input": int(t_in or 0),
                "total_tokens_output": int(t_out or 0),
                "avg_quality": float(avg_q or 0),
                "avg_execution_time": float(avg_t or 0),
            }
            aggregation["total_cost"] += float(cost or 0)
            aggregation["total_tasks"] += count
            aggregation["total_tokens"] += int(t_in or 0) + int(t_out or 0)

        return aggregation

    def get_performance_stats(
        self,
        lieutenant_id: str | None = None,
        days: int = 7,
    ) -> dict:
        """Get task performance statistics."""
        since = datetime.now(timezone.utc) - timedelta(days=days)

        stmt = select(Task).where(Task.created_at >= since)
        if lieutenant_id:
            stmt = stmt.where(Task.lieutenant_id == lieutenant_id)

        tasks = list(self.session.execute(stmt).scalars().all())

        completed = [t for t in tasks if t.status == "completed"]
        failed = [t for t in tasks if t.status == "failed"]

        return {
            "total": len(tasks),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(tasks) if tasks else 0.0,
            "avg_quality": (
                sum(t.quality_score for t in completed if t.quality_score) / len(completed)
                if completed else 0.0
            ),
            "avg_execution_time": (
                sum(t.execution_time_seconds for t in completed if t.execution_time_seconds)
                / len(completed)
                if completed else 0.0
            ),
            "total_cost": sum(t.cost_usd for t in tasks),
            "avg_retries": (
                sum(t.retry_count for t in tasks) / len(tasks) if tasks else 0.0
            ),
            "period_days": days,
        }

    def get_wave_summary(self, directive_id: str) -> list[dict]:
        """Get summary of each wave in a directive."""
        stmt = (
            select(
                Task.wave_number,
                func.count(Task.id).label("total"),
                func.sum(func.cast(Task.status == "completed", func.integer())).label("completed"),
                func.sum(func.cast(Task.status == "failed", func.integer())).label("failed"),
                func.sum(Task.cost_usd).label("cost"),
                func.avg(Task.quality_score).label("avg_quality"),
            )
            .where(Task.directive_id == directive_id)
            .group_by(Task.wave_number)
            .order_by(asc(Task.wave_number))
        )
        results = self.session.execute(stmt).all()

        return [
            {
                "wave": row[0],
                "total": row[1],
                "completed": int(row[2] or 0),
                "failed": int(row[3] or 0),
                "cost": float(row[4] or 0),
                "avg_quality": float(row[5] or 0),
            }
            for row in results
        ]

    def get_recent(self, limit: int = 20, status: str | None = None) -> list[Task]:
        """Get most recent tasks."""
        filters = {"status": status} if status else None
        return self.find(filters=filters, limit=limit, order_by="created_at", order_dir="desc")

    def get_by_type(self, task_type: str, limit: int = 50) -> list[Task]:
        """Get tasks by type."""
        return self.find(filters={"task_type": task_type}, limit=limit)

    def get_subtasks(self, parent_task_id: str) -> list[Task]:
        """Get subtasks of a parent task."""
        return self.find(filters={"parent_task_id": parent_task_id}, order_by="created_at", order_dir="asc")

    def cleanup_stale(self, hours: int = 24) -> int:
        """Mark stale executing tasks as failed.

        Args:
            hours: Hours after which an executing task is considered stale.

        Returns:
            Number of tasks marked as failed.
        """
        threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.update_where(
            filters={"status": "executing"},
            status="failed",
            last_error="Task timed out (stale execution)",
            completed_at=datetime.now(timezone.utc),
        )
