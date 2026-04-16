"""Evolution-specific repository with proposal tracking and cycle history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, select

from db.models import EvolutionCycle, EvolutionProposal
from db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class EvolutionRepository(BaseRepository[EvolutionProposal]):
    """Repository for evolution proposals and cycles."""

    model_class = EvolutionProposal

    # ── Proposal queries ───────────────────────────────────────────────

    def get_by_empire(
        self,
        empire_id: str,
        status: str | None = None,
        proposal_type: str | None = None,
        limit: int = 50,
    ) -> list[EvolutionProposal]:
        """Get proposals for an empire."""
        filters: dict[str, Any] = {"empire_id": empire_id}
        if status:
            filters["review_status"] = status
        if proposal_type:
            filters["proposal_type"] = proposal_type
        return self.find(filters=filters, limit=limit)

    def get_pending(self, empire_id: str, limit: int = 20) -> list[EvolutionProposal]:
        """Get pending proposals awaiting review."""
        return self.find(
            filters={"empire_id": empire_id, "review_status": "pending"},
            limit=limit,
            order_by="created_at",
            order_dir="asc",
        )

    def get_approved(self, empire_id: str, unapplied_only: bool = False) -> list[EvolutionProposal]:
        """Get approved proposals."""
        filters: dict[str, Any] = {"empire_id": empire_id, "review_status": "approved"}
        if unapplied_only:
            filters["applied"] = False
        return self.find(filters=filters)

    def get_by_lieutenant(self, lieutenant_id: str, limit: int = 20) -> list[EvolutionProposal]:
        """Get proposals from a specific lieutenant."""
        return self.find(filters={"lieutenant_id": lieutenant_id}, limit=limit)

    def get_by_cycle(self, cycle_id: str) -> list[EvolutionProposal]:
        """Get all proposals in a cycle."""
        return self.find(filters={"cycle_id": cycle_id}, order_by="created_at", order_dir="asc")

    def approve_proposal(self, proposal_id: str, notes: str = "", reviewer_model: str = "") -> EvolutionProposal | None:
        """Mark proposal as approved."""
        return self.update(
            proposal_id,
            review_status="approved",
            review_notes=notes,
            reviewer_model=reviewer_model,
        )

    def reject_proposal(self, proposal_id: str, notes: str = "", reviewer_model: str = "") -> EvolutionProposal | None:
        """Mark proposal as rejected."""
        return self.update(
            proposal_id,
            review_status="rejected",
            review_notes=notes,
            reviewer_model=reviewer_model,
        )

    def mark_applied(self, proposal_id: str, result: dict | None = None) -> EvolutionProposal | None:
        """Mark proposal as applied."""
        return self.update(
            proposal_id,
            applied=True,
            applied_at=datetime.now(UTC),
            application_result_json=result or {},
        )

    def mark_rolled_back(self, proposal_id: str) -> EvolutionProposal | None:
        """Mark proposal as rolled back."""
        return self.update(proposal_id, rolled_back=True)

    def get_proposal_stats(self, empire_id: str, days: int = 30) -> dict:
        """Get proposal statistics."""
        since = datetime.now(UTC) - timedelta(days=days)

        stmt = (
            select(
                EvolutionProposal.review_status,
                func.count(EvolutionProposal.id).label("count"),
            )
            .where(and_(
                EvolutionProposal.empire_id == empire_id,
                EvolutionProposal.created_at >= since,
            ))
            .group_by(EvolutionProposal.review_status)
        )
        status_results = self.session.execute(stmt).all()

        by_type_stmt = (
            select(
                EvolutionProposal.proposal_type,
                func.count(EvolutionProposal.id).label("count"),
            )
            .where(and_(
                EvolutionProposal.empire_id == empire_id,
                EvolutionProposal.created_at >= since,
            ))
            .group_by(EvolutionProposal.proposal_type)
        )
        type_results = self.session.execute(by_type_stmt).all()

        by_status = {row[0]: row[1] for row in status_results}
        total = sum(by_status.values())
        approved = by_status.get("approved", 0)
        applied_count = self.count({
            "empire_id": empire_id,
            "applied": True,
        })

        return {
            "total": total,
            "by_status": by_status,
            "by_type": {row[0]: row[1] for row in type_results},
            "approval_rate": approved / total if total > 0 else 0.0,
            "applied_count": applied_count,
            "period_days": days,
        }

    # ── Cycle queries ──────────────────────────────────────────────────

    def create_cycle(self, empire_id: str, cycle_number: int | None = None) -> EvolutionCycle:
        """Create a new evolution cycle."""
        if cycle_number is None:
            last = self.get_latest_cycle(empire_id)
            cycle_number = (last.cycle_number + 1) if last else 1

        cycle = EvolutionCycle(
            empire_id=empire_id,
            cycle_number=cycle_number,
            status="collecting",
        )
        self.session.add(cycle)
        self.session.flush()
        return cycle

    def get_cycle(self, cycle_id: str) -> EvolutionCycle | None:
        """Get an evolution cycle by ID."""
        return self.session.get(EvolutionCycle, cycle_id)

    def get_active_cycle(self, empire_id: str) -> EvolutionCycle | None:
        """Get the currently active cycle for an empire."""
        stmt = (
            select(EvolutionCycle)
            .where(and_(
                EvolutionCycle.empire_id == empire_id,
                EvolutionCycle.status.in_(["collecting", "reviewing", "executing", "learning"]),
            ))
            .order_by(desc(EvolutionCycle.created_at))
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_latest_cycle(self, empire_id: str) -> EvolutionCycle | None:
        """Get the most recent cycle for an empire."""
        stmt = (
            select(EvolutionCycle)
            .where(EvolutionCycle.empire_id == empire_id)
            .order_by(desc(EvolutionCycle.cycle_number))
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_cycle_history(self, empire_id: str, limit: int = 20) -> list[EvolutionCycle]:
        """Get cycle history for an empire."""
        stmt = (
            select(EvolutionCycle)
            .where(EvolutionCycle.empire_id == empire_id)
            .order_by(desc(EvolutionCycle.cycle_number))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def update_cycle(self, cycle_id: str, **kwargs: Any) -> EvolutionCycle | None:
        """Update a cycle."""
        cycle = self.session.get(EvolutionCycle, cycle_id)
        if cycle is None:
            return None
        for key, value in kwargs.items():
            if hasattr(cycle, key):
                setattr(cycle, key, value)
        self.session.flush()
        return cycle

    def complete_cycle(self, cycle_id: str, learnings: list | None = None, summary: str = "") -> EvolutionCycle | None:
        """Mark a cycle as completed."""
        return self.update_cycle(
            cycle_id,
            status="completed",
            completed_at=datetime.now(UTC),
            learnings_json=learnings or [],
            summary=summary,
        )

    def get_cycle_stats(self, empire_id: str) -> dict:
        """Get evolution cycle statistics."""
        stmt = (
            select(
                func.count(EvolutionCycle.id).label("total_cycles"),
                func.avg(EvolutionCycle.proposals_count).label("avg_proposals"),
                func.avg(EvolutionCycle.approved_count).label("avg_approved"),
                func.avg(EvolutionCycle.applied_count).label("avg_applied"),
                func.sum(EvolutionCycle.total_cost_usd).label("total_cost"),
            )
            .where(EvolutionCycle.empire_id == empire_id)
        )
        row = self.session.execute(stmt).one()

        return {
            "total_cycles": row[0] or 0,
            "avg_proposals_per_cycle": float(row[1] or 0),
            "avg_approved_per_cycle": float(row[2] or 0),
            "avg_applied_per_cycle": float(row[3] or 0),
            "total_cost": float(row[4] or 0),
            "avg_approval_rate": (
                float(row[2] or 0) / float(row[1] or 1) if row[1] else 0.0
            ),
        }

    def should_run_cycle(self, empire_id: str, cooldown_hours: int = 2) -> bool:
        """Check if enough time has passed since the last cycle.

        Args:
            empire_id: Empire ID.
            cooldown_hours: Minimum hours between cycles.

        Returns:
            True if a new cycle can be started.
        """
        active = self.get_active_cycle(empire_id)
        if active:
            return False

        latest = self.get_latest_cycle(empire_id)
        if latest is None:
            return True

        if latest.completed_at is None:
            return False

        completed = latest.completed_at
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=UTC)
        threshold = completed + timedelta(hours=cooldown_hours)
        return datetime.now(UTC) >= threshold

    def get_improvement_trend(self, empire_id: str, limit: int = 10) -> list[dict]:
        """Get trend data showing improvement over cycles."""
        cycles = self.get_cycle_history(empire_id, limit=limit)
        cycles.reverse()  # Oldest first

        return [
            {
                "cycle_number": c.cycle_number,
                "proposals": c.proposals_count,
                "approved": c.approved_count,
                "applied": c.applied_count,
                "approval_rate": c.approval_rate,
                "cost": c.total_cost_usd,
                "completed_at": c.completed_at.isoformat() if c.completed_at else None,
            }
            for c in cycles
        ]
