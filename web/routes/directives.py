"""Directive management routes."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, render_template, request

logger = logging.getLogger(__name__)

directives_bp = Blueprint("directives", __name__)


@directives_bp.route("/")
def list_directives():
    """List all directives."""
    empire_id = current_app.config.get("EMPIRE_ID", "")

    try:
        from core.directives.manager import DirectiveManager
        dm = DirectiveManager(empire_id)

        status_filter = request.args.get("status")
        directives = dm.list_directives(status=status_filter)
        stats = dm.get_stats()

        return render_template(
            "directives/list.html",
            directives=directives,
            stats=stats,
        )
    except Exception as e:
        return render_template("directives/list.html", directives=[], error=str(e))


@directives_bp.route("/<directive_id>")
def directive_detail(directive_id: str):
    """Directive detail page."""
    empire_id = current_app.config.get("EMPIRE_ID", "")

    try:
        from core.directives.manager import DirectiveManager
        dm = DirectiveManager(empire_id)

        directive = dm.get_directive(directive_id)
        if not directive:
            return "Directive not found", 404

        progress = dm.get_progress(directive_id)
        cost = dm.get_cost_summary(directive_id)

        from db.engine import repo_scope
        from db.repositories.directive import DirectiveRepository

        with repo_scope(DirectiveRepository) as dir_repo:
            timeline = dir_repo.get_timeline(directive_id)
            tasks_by_wave = dir_repo.get_with_tasks(directive_id)

            return render_template(
                "directives/detail.html",
                directive=directive,
                progress=progress.__dict__,
                cost=cost.__dict__,
                timeline=timeline,
                tasks_by_wave=tasks_by_wave.get("tasks_by_wave", {}),
            )
    except Exception as e:
        return str(e), 500


@directives_bp.route("/create", methods=["POST"])
def create_directive():
    """Create a new directive."""
    empire_id = current_app.config.get("EMPIRE_ID", "")

    try:
        data = request.get_json(silent=True) or {}
        from core.directives.manager import DirectiveManager
        dm = DirectiveManager(empire_id)

        result = dm.create_directive(
            title=data.get("title", ""),
            description=data.get("description", ""),
            priority=data.get("priority", 5),
            source=data.get("source", "human"),
        )

        return jsonify(result), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@directives_bp.route("/<directive_id>/execute", methods=["POST"])
def execute_directive(directive_id: str):
    """Execute a directive."""
    empire_id = current_app.config.get("EMPIRE_ID", "")

    try:
        from core.directives.manager import DirectiveManager
        dm = DirectiveManager(empire_id)
        result = dm.execute_directive(directive_id)
        return jsonify(result)

    except Exception as e:
        logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@directives_bp.route("/<directive_id>/cancel", methods=["POST"])
def cancel_directive(directive_id: str):
    """Cancel a directive."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager(empire_id)
    success = dm.cancel_directive(directive_id)
    return jsonify({"success": success})


@directives_bp.route("/<directive_id>/progress")
def directive_progress(directive_id: str):
    """Get directive progress."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager(empire_id)
    progress = dm.get_progress(directive_id)
    return jsonify(progress.__dict__)
