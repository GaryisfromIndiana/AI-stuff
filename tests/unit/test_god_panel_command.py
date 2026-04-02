"""Unit tests for the GodPanelCommand model."""

from __future__ import annotations

from datetime import datetime, timezone

from db.models import GodPanelCommand


def test_to_dict_contains_all_fields():
    """to_dict() should include all command fields."""
    now = datetime.now(timezone.utc)
    cmd = GodPanelCommand(
        id="abc123",
        empire_id="emp-1",
        command="research transformers",
        action="RESEARCH",
        topic="transformers",
        status="completed",
        cost_usd=0.05,
        started_at=now,
        completed_at=now,
        result_json={"status": "ok"},
        error=None,
    )

    d = cmd.to_dict()
    assert d["id"] == "abc123"
    assert d["command"] == "research transformers"
    assert d["action"] == "RESEARCH"
    assert d["topic"] == "transformers"
    assert d["status"] == "completed"
    assert d["cost"] == 0.05
    assert d["result"] == {"status": "ok"}
    assert d["error"] is None
    assert d["started_at"] is not None
    assert d["completed_at"] is not None


def test_to_dict_handles_none_dates():
    """to_dict() should handle None datetimes gracefully."""
    cmd = GodPanelCommand(
        id="xyz",
        empire_id="emp-1",
        command="test",
        action="STATUS",
        topic="",
        status="accepted",
    )
    # Simulate no started_at set
    cmd.started_at = None
    cmd.completed_at = None

    d = cmd.to_dict()
    assert d["started_at"] is None
    assert d["completed_at"] is None


def test_repr():
    """__repr__ should be readable."""
    cmd = GodPanelCommand(
        id="test-id",
        empire_id="emp-1",
        command="test",
        action="SWEEP",
        topic="",
        status="running",
    )
    r = repr(cmd)
    assert "test-id" in r
    assert "SWEEP" in r
    assert "running" in r
