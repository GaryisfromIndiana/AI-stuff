"""Directive-specific repository with pipeline and wave queries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, asc, desc, func, select

from db.models import Directive, Task
from db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class DirectiveRepository(BaseRepository[Directive]):
    """Repository for Directive entities."""

    model_class = Directive

    def get_by_empire(
        self,
        empire_id: str,
        status: str | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[Directive]:
        """Get directives for an empire with optional filters."""
        filters: dict[str, Any] = {"empire_id": empire_id}
        if status:
            filters["status"] = status
        if source:
            filters["source"] = source
        return self.find(filters=filters, limit=limit, order_by="created_at", order_dir="desc")

    def get_active(self, empire_id: str) -> list[Directive]:
        """Get all active directives (planning, executing, reviewing)."""
        stmt = (
            select(Directive)
            .where(and_(
                Directive.empire_id == empire_id,
                Directive.status.in_(["planning", "executing", "reviewing"]),
            ))
            .order_by(asc(Directive.priority), desc(Directive.created_at))
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_pending(self, empire_id: str, limit: int = 20) -> list[Directive]:
        """Get pending directives ready for execution, ordered by priority."""
        stmt = (
            select(Directive)
            .where(and_(
                Directive.empire_id == empire_id,
                Directive.status == "pending",
            ))
            .order_by(asc(Directive.priority), asc(Directive.created_at))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_completed(
        self,
        empire_id: str,
        days: int = 30,
        limit: int = 50,
    ) -> list[Directive]:
        """Get recently completed directives."""
        since = datetime.now(UTC) - timedelta(days=days)
        stmt = (
            select(Directive)
            .where(and_(
                Directive.empire_id == empire_id,
                Directive.status == "completed",
                Directive.completed_at >= since,
            ))
            .order_by(desc(Directive.completed_at))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_failed(self, empire_id: str, limit: int = 20) -> list[Directive]:
        """Get failed directives."""
        return self.find(
            filters={"empire_id": empire_id, "status": "failed"},
            limit=limit,
        )

    def get_with_tasks(self, directive_id: str) -> dict:
        """Get directive with all its tasks grouped by wave.

        Returns:
            Dict with directive and tasks_by_wave.
        """
        directive = self.get(directive_id)
        if directive is None:
            return {}

        stmt = (
            select(Task)
            .where(Task.directive_id == directive_id)
            .order_by(asc(Task.wave_number), asc(Task.created_at))
        )
        tasks = list(self.session.execute(stmt).scalars().all())

        waves: dict[int, list[Task]] = {}
        for task in tasks:
            waves.setdefault(task.wave_number, []).append(task)

        return {
            "directive": directive,
            "tasks": tasks,
            "tasks_by_wave": waves,
            "total_tasks": len(tasks),
            "wave_count": len(waves),
        }

    def get_progress(self, directive_id: str) -> dict:
        """Get execution progress for a directive.

        Returns:
            Dict with task counts by status, completion percentage, etc.
        """
        stmt = (
            select(
                Task.status,
                func.count(Task.id).label("count"),
            )
            .where(Task.directive_id == directive_id)
            .group_by(Task.status)
        )
        results = self.session.execute(stmt).all()

        by_status = {row[0]: row[1] for row in results}
        total = sum(by_status.values())
        completed = by_status.get("completed", 0)
        failed = by_status.get("failed", 0)
        in_progress = by_status.get("executing", 0) + by_status.get("planning", 0)
        pending = by_status.get("pending", 0)

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "completion_percent": (completed / total * 100) if total > 0 else 0.0,
            "success_rate": (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0.0,
            "by_status": by_status,
        }

    def get_cost_summary(self, directive_id: str) -> dict:
        """Get cost summary for a directive."""
        stmt = (
            select(
                func.sum(Task.cost_usd).label("total_cost"),
                func.sum(Task.tokens_input).label("total_input"),
                func.sum(Task.tokens_output).label("total_output"),
                func.count(Task.id).label("task_count"),
            )
            .where(Task.directive_id == directive_id)
        )
        row = self.session.execute(stmt).one()

        # By model
        stmt_model = (
            select(
                Task.model_used,
                func.sum(Task.cost_usd).label("cost"),
                func.count(Task.id).label("count"),
            )
            .where(Task.directive_id == directive_id)
            .group_by(Task.model_used)
        )
        model_rows = self.session.execute(stmt_model).all()

        return {
            "total_cost_usd": float(row[0] or 0),
            "total_tokens_input": int(row[1] or 0),
            "total_tokens_output": int(row[2] or 0),
            "task_count": row[3],
            "by_model": {r[0]: {"cost": float(r[1] or 0), "count": r[2]} for r in model_rows if r[0]},
        }

    def get_timeline(self, directive_id: str) -> list[dict]:
        """Get execution timeline for a directive.

        Returns:
            List of timeline events.
        """
        directive = self.get(directive_id)
        if directive is None:
            return []

        events = []

        # Directive creation
        events.append({
            "time": directive.created_at.isoformat() if directive.created_at else None,
            "event": "directive_created",
            "details": {"title": directive.title, "source": directive.source},
        })

        if directive.started_at:
            events.append({
                "time": directive.started_at.isoformat(),
                "event": "directive_started",
                "details": {},
            })

        # Task events
        stmt = (
            select(Task)
            .where(Task.directive_id == directive_id)
            .order_by(asc(Task.created_at))
        )
        tasks = self.session.execute(stmt).scalars().all()

        for task in tasks:
            events.append({
                "time": task.created_at.isoformat() if task.created_at else None,
                "event": "task_created",
                "details": {
                    "task_id": task.id,
                    "title": task.title,
                    "wave": task.wave_number,
                },
            })
            if task.completed_at:
                events.append({
                    "time": task.completed_at.isoformat(),
                    "event": f"task_{task.status}",
                    "details": {
                        "task_id": task.id,
                        "title": task.title,
                        "quality": task.quality_score,
                        "cost": task.cost_usd,
                    },
                })

        if directive.completed_at:
            events.append({
                "time": directive.completed_at.isoformat(),
                "event": f"directive_{directive.status}",
                "details": {"quality": directive.quality_score, "cost": directive.total_cost_usd},
            })

        events.sort(key=lambda e: e["time"] or "")
        return events

    def start_directive(self, directive_id: str) -> Directive | None:
        """Mark directive as started."""
        return self.update(
            directive_id,
            status="planning",
            started_at=datetime.now(UTC),
            pipeline_stage="planning",
        )

    def complete_directive(self, directive_id: str, results: dict | None = None) -> Directive | None:
        """Mark directive as completed."""
        updates: dict[str, Any] = {
            "status": "completed",
            "completed_at": datetime.now(UTC),
            "pipeline_stage": "delivered",
        }
        if results:
            updates["results_json"] = results
        return self.update(directive_id, **updates)

    def fail_directive(self, directive_id: str, error: str = "") -> Directive | None:
        """Mark directive as failed."""
        return self.update(
            directive_id,
            status="failed",
            completed_at=datetime.now(UTC),
            results_json={"error": error},
        )

    def get_stats(self, empire_id: str, days: int = 30) -> dict:
        """Get directive statistics for an empire."""
        since = datetime.now(UTC) - timedelta(days=days)

        stmt = (
            select(
                Directive.status,
                func.count(Directive.id).label("count"),
                func.sum(Directive.total_cost_usd).label("cost"),
                func.avg(Directive.quality_score).label("avg_quality"),
            )
            .where(and_(
                Directive.empire_id == empire_id,
                Directive.created_at >= since,
            ))
            .group_by(Directive.status)
        )
        results = self.session.execute(stmt).all()

        stats: dict[str, Any] = {
            "by_status": {},
            "total": 0,
            "total_cost": 0.0,
            "avg_quality": 0.0,
            "period_days": days,
        }

        for row in results:
            status, count, cost, avg_q = row
            stats["by_status"][status] = {
                "count": count,
                "cost": float(cost or 0),
                "avg_quality": float(avg_q or 0),
            }
            stats["total"] += count
            stats["total_cost"] += float(cost or 0)

        return stats
