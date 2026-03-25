"""Empire-specific repository with cross-empire queries and health overview."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select, func, and_, desc

from db.models import (
    Empire, Lieutenant, Directive, Task, KnowledgeEntity,
    MemoryEntry, BudgetLog, HealthCheck, CrossEmpireSync,
)
from db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class EmpireRepository(BaseRepository[Empire]):
    """Repository for Empire entities with cross-empire operations."""

    model_class = Empire

    def get_by_name(self, name: str) -> Empire | None:
        return self.find_one({"name": name})

    def get_active(self) -> list[Empire]:
        return self.find(filters={"status": "active"}, order_by="name", order_dir="asc")

    def get_by_domain(self, domain: str) -> list[Empire]:
        return self.find(filters={"domain": domain, "status": "active"})

    def get_health_overview(self, empire_id: str) -> dict:
        """Get comprehensive health overview for an empire."""
        empire = self.get(empire_id)
        if not empire:
            return {}

        lt_count = self.session.execute(
            select(func.count(Lieutenant.id)).where(Lieutenant.empire_id == empire_id)
        ).scalar() or 0

        active_lt = self.session.execute(
            select(func.count(Lieutenant.id)).where(and_(
                Lieutenant.empire_id == empire_id, Lieutenant.status == "active"
            ))
        ).scalar() or 0

        active_directives = self.session.execute(
            select(func.count(Directive.id)).where(and_(
                Directive.empire_id == empire_id,
                Directive.status.in_(["planning", "executing", "reviewing"]),
            ))
        ).scalar() or 0

        total_tasks = self.session.execute(
            select(func.count(Task.id))
            .join(Directive, Task.directive_id == Directive.id)
            .where(Directive.empire_id == empire_id)
        ).scalar() or 0

        knowledge_count = self.session.execute(
            select(func.count(KnowledgeEntity.id)).where(KnowledgeEntity.empire_id == empire_id)
        ).scalar() or 0

        memory_count = self.session.execute(
            select(func.count(MemoryEntry.id)).where(MemoryEntry.empire_id == empire_id)
        ).scalar() or 0

        # Recent health checks
        recent_checks = self.session.execute(
            select(HealthCheck)
            .where(HealthCheck.empire_id == empire_id)
            .order_by(desc(HealthCheck.created_at))
            .limit(10)
        ).scalars().all()

        unhealthy = [c for c in recent_checks if c.status == "unhealthy"]

        return {
            "empire": {"id": empire.id, "name": empire.name, "status": empire.status},
            "lieutenants": {"total": lt_count, "active": active_lt},
            "directives": {"active": active_directives},
            "tasks": {"total": total_tasks},
            "knowledge": {"entities": knowledge_count},
            "memory": {"entries": memory_count},
            "health": {
                "status": "unhealthy" if unhealthy else "healthy",
                "recent_issues": len(unhealthy),
                "last_check": recent_checks[0].created_at.isoformat() if recent_checks else None,
            },
            "cost": {"total_usd": empire.total_cost_usd},
        }

    def get_daily_spend(self, empire_id: str, days: int = 30) -> list[dict]:
        """Get daily spend over a period."""
        stmt = (
            select(
                BudgetLog.cost_date,
                func.sum(BudgetLog.cost_usd).label("total"),
                func.count(BudgetLog.id).label("requests"),
            )
            .where(BudgetLog.empire_id == empire_id)
            .group_by(BudgetLog.cost_date)
            .order_by(desc(BudgetLog.cost_date))
            .limit(days)
        )
        results = self.session.execute(stmt).all()
        return [
            {"date": row[0], "cost": float(row[1]), "requests": row[2]}
            for row in reversed(results)
        ]

    def get_network_stats(self) -> dict:
        """Get statistics across all empires in the network."""
        empires = self.get_active()

        total_lt = self.session.execute(
            select(func.count(Lieutenant.id)).where(Lieutenant.status == "active")
        ).scalar() or 0

        total_knowledge = self.session.execute(
            select(func.count(KnowledgeEntity.id))
        ).scalar() or 0

        total_cost = self.session.execute(
            select(func.coalesce(func.sum(Empire.total_cost_usd), 0.0))
        ).scalar() or 0.0

        total_tasks = self.session.execute(
            select(func.coalesce(func.sum(Empire.total_tasks_completed), 0))
        ).scalar() or 0

        return {
            "total_empires": len(empires),
            "total_lieutenants": total_lt,
            "total_knowledge_entities": total_knowledge,
            "total_tasks_completed": total_tasks,
            "total_cost_usd": float(total_cost),
            "empires": [
                {"id": e.id, "name": e.name, "domain": e.domain, "status": e.status}
                for e in empires
            ],
        }

    def get_sync_status(self, empire_id: str) -> list[dict]:
        """Get cross-empire sync status."""
        stmt = (
            select(CrossEmpireSync)
            .where(
                (CrossEmpireSync.source_empire_id == empire_id) |
                (CrossEmpireSync.target_empire_id == empire_id)
            )
            .order_by(desc(CrossEmpireSync.created_at))
            .limit(20)
        )
        syncs = list(self.session.execute(stmt).scalars().all())

        return [
            {
                "id": s.id,
                "source": s.source_empire_id,
                "target": s.target_empire_id,
                "type": s.sync_type,
                "status": s.status,
                "entities_synced": s.entities_synced,
                "conflicts": s.conflicts_found,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in syncs
        ]

    def increment_stats(
        self,
        empire_id: str,
        tasks_completed: int = 0,
        cost_usd: float = 0.0,
        knowledge_entries: int = 0,
    ) -> None:
        """Increment empire-level statistics."""
        empire = self.get(empire_id)
        if empire:
            empire.total_tasks_completed += tasks_completed
            empire.total_cost_usd += cost_usd
            empire.total_knowledge_entries += knowledge_entries
            self.session.flush()

    def get_capability_map(self) -> dict:
        """Get a map of capabilities across all empires."""
        stmt = (
            select(
                Empire.id,
                Empire.name,
                Empire.domain,
                Lieutenant.domain.label("lt_domain"),
                func.count(Lieutenant.id).label("lt_count"),
                func.avg(Lieutenant.performance_score).label("avg_perf"),
            )
            .join(Lieutenant, Lieutenant.empire_id == Empire.id)
            .where(and_(Empire.status == "active", Lieutenant.status == "active"))
            .group_by(Empire.id, Empire.name, Empire.domain, Lieutenant.domain)
        )
        results = self.session.execute(stmt).all()

        cap_map: dict[str, dict] = {}
        for row in results:
            emp_id, emp_name, emp_domain, lt_domain, lt_count, avg_perf = row
            if emp_id not in cap_map:
                cap_map[emp_id] = {
                    "name": emp_name,
                    "domain": emp_domain,
                    "capabilities": {},
                }
            cap_map[emp_id]["capabilities"][lt_domain] = {
                "lieutenant_count": lt_count,
                "avg_performance": float(avg_perf or 0),
            }

        return cap_map
