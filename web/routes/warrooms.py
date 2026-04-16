"""War Room routes."""

from __future__ import annotations

import logging
from flask import Blueprint, render_template, jsonify, request, current_app

logger = logging.getLogger(__name__)
warrooms_bp = Blueprint("warrooms", __name__)


@warrooms_bp.route("/")
def list_warrooms():
    """List war room sessions."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    try:
        from db.engine import read_session
        from db.models import WarRoom, Lieutenant
        from sqlalchemy import select, desc

        with read_session() as session:
            stmt = select(WarRoom).where(WarRoom.empire_id == empire_id).order_by(desc(WarRoom.created_at)).limit(20)
            warrooms = list(session.execute(stmt).scalars().all())
            # Active lieutenants for the "New Session" modal
            lt_stmt = select(Lieutenant).where(Lieutenant.empire_id == empire_id, Lieutenant.status == "active")
            active_lts = [{"id": lt.id, "name": lt.name, "domain": lt.domain} for lt in session.execute(lt_stmt).scalars().all()]
            return render_template("warrooms/list.html", warrooms=[
                {"id": w.id, "status": w.status, "type": w.session_type, "participants": len(w.participants_json or []),
                 "cost": w.total_cost_usd, "created_at": w.created_at.isoformat() if w.created_at else None}
                for w in warrooms
            ], active_lieutenants=active_lts)
    except Exception as e:
        return render_template("warrooms/list.html", warrooms=[], active_lieutenants=[], error=str(e))


@warrooms_bp.route("/<session_id>")
def warroom_detail(session_id: str):
    """War room session detail."""
    try:
        from db.engine import read_session
        from db.models import WarRoom, Lieutenant

        with read_session() as session:
            warroom = session.get(WarRoom, session_id)
            if not warroom:
                return "War room not found", 404

            # Resolve participant IDs to names + domains
            participant_ids = warroom.participants_json or []
            participants = []
            for pid in participant_ids:
                if isinstance(pid, dict):
                    participants.append(pid)
                    continue
                lt = session.get(Lieutenant, pid)
                if lt:
                    participants.append({"id": lt.id, "name": lt.name, "domain": lt.domain})
                else:
                    participants.append({"id": pid, "name": pid[:12], "domain": "unknown"})

            return render_template("warrooms/detail.html", warroom={
                "id": warroom.id,
                "title": warroom.title or warroom.debate_topic or f"Session {warroom.id[:8]}",
                "status": warroom.status,
                "type": warroom.session_type,
                "participants": participants,
                "synthesis": warroom.synthesis_json,
                "action_items": warroom.action_items_json,
                "transcript": warroom.transcript_json or [],
                "cost": warroom.total_cost_usd,
                "debate_topic": warroom.debate_topic,
                "consensus_score": warroom.consensus_score,
                "consensus_reached": warroom.consensus_reached,
                "round_count": warroom.debate_round_count,
                "created_at": warroom.created_at.isoformat() if warroom.created_at else None,
                "completed_at": warroom.completed_at.isoformat() if warroom.completed_at else None,
            })
    except Exception as e:
        return str(e), 500


@warrooms_bp.route("/create", methods=["POST"])
def create_warroom():
    """Create and start a new war room session."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json(silent=True) or {}
    try:
        from core.warroom.session import WarRoomSession
        session = WarRoomSession(empire_id=empire_id, session_type=data.get("type", "planning"))
        for lt in data.get("participants", []):
            session.add_participant(lt.get("id", ""), lt.get("name", ""))
        if data.get("topic"):
            result = session.start_debate(data["topic"])
            summary = session.close_session()
            return jsonify({"session_id": session.session_id, "result": result, "summary": summary.__dict__}), 201
        return jsonify({"session_id": session.session_id, "status": "created"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400
