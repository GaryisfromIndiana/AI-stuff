"""Memory browsing routes."""

from __future__ import annotations

import logging
from flask import Blueprint, render_template, jsonify, request, current_app

logger = logging.getLogger(__name__)
memory_bp = Blueprint("memory", __name__)


@memory_bp.route("/")
def memory_overview():
    """Memory system overview."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    try:
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        stats = mm.get_stats()
        recent = mm.recall(limit=10)
        return render_template("memory/overview.html", stats=stats.__dict__, recent=recent)
    except Exception as e:
        return render_template("memory/overview.html", stats={}, recent=[], error=str(e))


@memory_bp.route("/search")
def memory_search():
    """Search memories."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    query = request.args.get("q", "")
    memory_types = request.args.getlist("type")
    try:
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        results = mm.search(query=query, memory_types=memory_types or None, limit=30)
        return render_template("memory/search.html", query=query, results=results, types=memory_types)
    except Exception as e:
        return render_template("memory/search.html", query=query, results=[], error=str(e))


@memory_bp.route("/by-type/<memory_type>")
def memories_by_type(memory_type: str):
    """List memories by type."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    try:
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        memories = mm.recall(memory_types=[memory_type], limit=50)
        return jsonify(memories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@memory_bp.route("/by-lieutenant/<lieutenant_id>")
def memories_by_lieutenant(lieutenant_id: str):
    """List memories for a lieutenant."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    try:
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        memories = mm.recall(lieutenant_id=lieutenant_id, limit=50)
        return jsonify(memories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@memory_bp.route("/decay", methods=["POST"])
def run_decay():
    """Manually trigger memory decay."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.memory.manager import MemoryManager
    mm = MemoryManager(empire_id)
    decayed = mm.decay()
    return jsonify({"decayed": decayed})


@memory_bp.route("/cleanup", methods=["POST"])
def run_cleanup():
    """Manually trigger memory cleanup."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.memory.manager import MemoryManager
    mm = MemoryManager(empire_id)
    result = mm.cleanup()
    return jsonify(result)


@memory_bp.route("/consolidate", methods=["POST"])
def run_consolidate():
    """Manually trigger memory consolidation."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.memory.manager import MemoryManager
    mm = MemoryManager(empire_id)
    promoted = mm.consolidate()
    return jsonify({"promoted": promoted})
