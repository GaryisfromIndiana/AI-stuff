"""Settings routes."""

from __future__ import annotations

import logging
from flask import Blueprint, render_template, jsonify, request, current_app

logger = logging.getLogger(__name__)
settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/")
def settings_page():
    """Settings overview page."""
    try:
        from config.settings import get_settings
        s = get_settings()
        return render_template("settings/index.html", settings={
            "empire_id": s.empire_id,
            "empire_name": s.empire_name,
            "debug": s.debug,
            "log_level": s.log_level,
            "db_url": s.db_url.split("?")[0],
            "scheduler": {
                "tick_interval": s.scheduler.tick_interval_seconds,
                "learning_cycle_hours": s.scheduler.learning_cycle_hours,
                "evolution_cycle_hours": s.scheduler.evolution_cycle_hours,
                "health_check_minutes": s.scheduler.health_check_interval_minutes,
            },
            "budget": {
                "daily_limit": s.budget.daily_limit_usd,
                "monthly_limit": s.budget.monthly_limit_usd,
                "per_task_limit": s.budget.per_task_limit_usd,
                "alert_threshold": s.budget.alert_threshold_percent,
            },
            "quality": {
                "min_confidence": s.quality.min_confidence_score,
                "min_completeness": s.quality.min_completeness_score,
                "require_citations": s.quality.require_source_citations,
            },
            "ace": {
                "planning_model": s.ace.default_planning_model,
                "execution_model": s.ace.default_execution_model,
                "critic_model": s.ace.default_critic_model,
                "max_iterations": s.ace.max_pipeline_iterations,
            },
        })
    except Exception as e:
        return render_template("settings/index.html", settings={}, error=str(e))


@settings_bp.route("/models")
def available_models():
    """List available LLM models."""
    from config.settings import MODEL_CATALOG
    models = []
    for key, config in MODEL_CATALOG.items():
        models.append({
            "key": key,
            "model_id": config.model_id,
            "provider": config.provider,
            "tier": config.tier,
            "cost_input": config.cost_per_1k_input,
            "cost_output": config.cost_per_1k_output,
            "capabilities": config.capabilities,
        })
    return jsonify(models)


@settings_bp.route("/health")
def system_health():
    """System health check."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.scheduler.health import HealthChecker
    checker = HealthChecker(empire_id)
    report = checker.run_all_checks()
    return jsonify(report)


@settings_bp.route("/scheduler")
def scheduler_status():
    """Scheduler status."""
    return jsonify({"status": "Use CLI to view scheduler status"})


@settings_bp.route("/db/stats")
def db_stats():
    """Database statistics."""
    from db.engine import get_db_stats
    return jsonify(get_db_stats())


@settings_bp.route("/db/migrate", methods=["POST"])
def run_migrations():
    """Run database migrations."""
    try:
        from db.migrations import MigrationRunner
        runner = MigrationRunner()
        applied = runner.migrate()
        return jsonify({
            "applied": len(applied),
            "migrations": [{"version": m.version, "name": m.name} for m in applied],
            "status": runner.get_status(),
        })
    except Exception as e:
        logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
