"""Dashboard routes — main overview page."""

from __future__ import annotations

import logging

from flask import Blueprint, render_template, jsonify, current_app

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """Main dashboard page."""
    empire_id = current_app.config.get("EMPIRE_ID", "")

    try:
        from db.engine import get_session
        from db.repositories.empire import EmpireRepository
        from db.repositories.lieutenant import LieutenantRepository
        from db.repositories.directive import DirectiveRepository

        session = get_session()

        # Empire overview
        empire_repo = EmpireRepository(session)
        health = empire_repo.get_health_overview(empire_id)

        # Recent directives
        dir_repo = DirectiveRepository(session)
        active_directives = dir_repo.get_active(empire_id)
        recent_completed = dir_repo.get_completed(empire_id, days=7, limit=5)

        # Lieutenant fleet
        lt_repo = LieutenantRepository(session)
        fleet = lt_repo.get_fleet_summary(empire_id)

        # Budget
        from core.routing.budget import BudgetManager
        bm = BudgetManager(empire_id)
        budget = bm.get_budget_report(days=30)

        context = {
            "health": health,
            "active_directives": [
                {"id": d.id, "title": d.title, "status": d.status, "priority": d.priority}
                for d in active_directives
            ],
            "recent_completed": [
                {"id": d.id, "title": d.title, "quality": d.quality_score, "cost": d.total_cost_usd}
                for d in recent_completed
            ],
            "fleet": fleet,
            "budget": {
                "daily_spend": budget.daily_spend,
                "monthly_spend": budget.monthly_spend,
                "daily_remaining": budget.daily_remaining,
                "monthly_remaining": budget.monthly_remaining,
                "alerts": [{"message": a.message, "severity": a.severity} for a in budget.alerts],
            },
        }

        return render_template("dashboard.html", **context)

    except Exception as e:
        logger.error("Dashboard error: %s", e)
        return render_template("dashboard.html", error=str(e))


@dashboard_bp.route("/api/dashboard/stats")
def dashboard_stats():
    """Dashboard stats API endpoint."""
    empire_id = current_app.config.get("EMPIRE_ID", "")

    try:
        from db.engine import get_session
        from db.repositories.empire import EmpireRepository
        session = get_session()
        repo = EmpireRepository(session)

        health = repo.get_health_overview(empire_id)
        network = repo.get_network_stats()

        return jsonify({
            "health": health,
            "network": network,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
