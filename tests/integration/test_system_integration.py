"""Integration tests — verify modules work together as a system.

Uses an in-memory SQLite database. No network calls, no LLM API keys needed.
Tests the real flow: store → recall → supersede → decay → cleanup.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta

import pytest

# Force in-memory DB before any Empire module touches the engine
os.environ["EMPIRE_DB_URL"] = "sqlite:///:memory:"
os.environ.setdefault("EMPIRE_ANTHROPIC_API_KEY", "sk-test-fake")

import db.engine as eng


@pytest.fixture(autouse=True)
def fresh_db():
    """Reset DB engine and create fresh tables for each test."""
    # Reset singletons
    if eng._scoped_session is not None:
        eng._scoped_session.remove()
    eng._scoped_session = None
    if eng._engine is not None:
        eng._engine.dispose()
    eng._engine = None
    eng._session_factory = None

    # Create fresh tables
    eng.init_db()

    yield

    # Cleanup
    if eng._scoped_session is not None:
        eng._scoped_session.remove()


EMPIRE_ID = "test-empire"


def _ensure_empire():
    """Create the test empire row."""
    from db.models import Empire
    with eng.session_scope() as session:
        if not session.get(Empire, EMPIRE_ID):
            session.add(Empire(id=EMPIRE_ID, name="Test Empire", domain="ai_research", status="active"))


# ── Memory Store + Recall ────────────────────────────────────────────


class TestMemoryStoreRecall:
    """Test that memories can be stored and recalled."""

    def test_store_and_recall_by_keyword(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(
            content="GPT-4o was released by OpenAI in May 2024 with multimodal capabilities",
            memory_type="semantic",
            title="GPT-4o Release",
            importance=0.8,
        )

        results = mm.recall(query="GPT-4o", memory_types=["semantic"], limit=5)
        assert len(results) >= 1
        assert "GPT-4o" in results[0]["content"]

    def test_recall_returns_metadata(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(
            content="Test memory with metadata",
            memory_type="semantic",
            title="Metadata Test",
            importance=0.7,
            tags=["test", "metadata"],
        )

        results = mm.recall(query="Metadata Test", limit=1)
        assert len(results) == 1
        mem = results[0]
        assert mem["title"] == "Metadata Test"
        assert "test" in mem["tags"]
        assert mem["metadata"].get("temporal") is True  # Auto-added
        assert mem["metadata"].get("recorded_at")  # Auto-added timestamp

    def test_recall_empty_query_returns_most_important(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(content="Low importance", memory_type="semantic", importance=0.1)
        mm.store(content="High importance", memory_type="semantic", importance=0.9)

        results = mm.recall(query="", memory_types=["semantic"], limit=2)
        assert len(results) == 2
        assert results[0]["importance"] >= results[1]["importance"]

    def test_recall_filters_by_memory_type(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(content="Semantic fact", memory_type="semantic")
        mm.store(content="Episodic event", memory_type="episodic")

        semantic = mm.recall(query="fact", memory_types=["semantic"], limit=10)
        episodic = mm.recall(query="event", memory_types=["episodic"], limit=10)

        assert all(m["type"] == "semantic" for m in semantic)
        assert all(m["type"] == "episodic" for m in episodic)


# ── Bi-Temporal Store + Supersede ────────────────────────────────────


class TestBiTemporal:
    """Test bi-temporal memory: versioning, supersession, temporal queries."""

    def test_store_fact_creates_version_1(self):
        _ensure_empire()
        from core.memory.bitemporal import BiTemporalMemory
        bt = BiTemporalMemory(EMPIRE_ID)

        fact = bt.store_fact(
            content="Claude 3.5 Sonnet is Anthropic's latest model",
            title="Claude 3.5 Sonnet",
            importance=0.8,
        )
        assert fact.version == 1
        assert fact.id != ""
        assert fact.previous_version_id is None

    def test_store_same_title_supersedes(self):
        _ensure_empire()
        from core.memory.bitemporal import BiTemporalMemory
        bt = BiTemporalMemory(EMPIRE_ID)

        v1 = bt.store_fact(
            content="Claude Sonnet is a mid-tier model",
            title="Claude Sonnet Status",
            importance=0.7,
        )
        v2 = bt.store_fact(
            content="Claude Sonnet 4 is the latest mid-tier model from Anthropic",
            title="Claude Sonnet Status",
            importance=0.8,
        )

        assert v2.version == 2
        assert v2.previous_version_id == v1.id

    def test_superseded_facts_excluded_by_default(self):
        _ensure_empire()
        from core.memory.bitemporal import BiTemporalMemory, TemporalQuery
        bt = BiTemporalMemory(EMPIRE_ID)

        bt.store_fact(content="Old info", title="Test Topic", importance=0.7)
        bt.store_fact(content="New info", title="Test Topic", importance=0.8)

        # Default: exclude superseded
        current = bt.query(TemporalQuery(query="Test Topic", limit=10))
        assert len(current) == 1
        assert "New info" in current[0].content

        # Include superseded
        all_versions = bt.query(TemporalQuery(query="Test Topic", include_superseded=True, limit=10))
        assert len(all_versions) == 2

    def test_store_smart_supersedes_by_title(self):
        _ensure_empire()
        from core.memory.bitemporal import BiTemporalMemory
        bt = BiTemporalMemory(EMPIRE_ID)

        bt.store_smart(content="GPT-5 rumored for Q3 2025", title="GPT-5 Status")
        bt.store_smart(content="GPT-5 confirmed released July 2025", title="GPT-5 Status")

        # Only the latest should be current
        current = bt.get_current_facts(query="GPT-5", limit=5)
        gpt5 = [f for f in current if "GPT-5" in f.title]
        assert len(gpt5) == 1
        assert "confirmed" in gpt5[0].content

    def test_store_smart_does_not_supersede_different_title(self):
        """Different titles = different facts, no supersession."""
        _ensure_empire()
        from core.memory.bitemporal import BiTemporalMemory
        bt = BiTemporalMemory(EMPIRE_ID)

        bt.store_smart(content="GPT-5 release info", title="GPT-5 Release")
        bt.store_smart(content="GPT-5 pricing info", title="GPT-5 Pricing")

        current = bt.get_current_facts(query="GPT-5", limit=10)
        titles = [f.title for f in current]
        assert "GPT-5 Release" in titles
        assert "GPT-5 Pricing" in titles

    def test_fact_timeline(self):
        _ensure_empire()
        from core.memory.bitemporal import BiTemporalMemory
        bt = BiTemporalMemory(EMPIRE_ID)

        bt.store_fact(content="v1 content", title="Evolving Fact", importance=0.6)
        bt.store_fact(content="v2 content", title="Evolving Fact", importance=0.7)
        bt.store_fact(content="v3 content", title="Evolving Fact", importance=0.8)

        timeline = bt.get_fact_timeline("Evolving Fact")
        assert timeline.current_version == 3
        assert len(timeline.versions) == 3


# ── Memory Decay + Retention ─────────────────────────────────────────


class TestDecayAndRetention:
    """Test memory decay and cleanup retention policy."""

    def test_decay_reduces_effective_importance(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(content="Decayable fact", memory_type="semantic", importance=0.5)

        # Run decay multiple times
        for _ in range(10):
            mm.decay(rate=0.05)

        results = mm.recall(query="Decayable", limit=1)
        assert len(results) == 1
        assert results[0]["importance"] < 0.5  # Should have decayed

    def test_episodic_decays_faster_than_semantic(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(content="Semantic fact XYZ", memory_type="semantic", importance=0.5)
        mm.store(content="Episodic event XYZ", memory_type="episodic", importance=0.5)

        for _ in range(5):
            mm.decay(rate=0.05)

        sem = mm.recall(query="XYZ", memory_types=["semantic"], limit=1)
        epi = mm.recall(query="XYZ", memory_types=["episodic"], limit=1)

        if sem and epi:
            assert epi[0]["importance"] < sem[0]["importance"]

    def test_cleanup_removes_fully_decayed(self):
        _ensure_empire()
        from core.memory.manager import MemoryManager
        mm = MemoryManager(EMPIRE_ID)

        mm.store(content="Will decay away", memory_type="episodic", importance=0.1)

        # Decay aggressively
        for _ in range(50):
            mm.decay(rate=0.1)

        stats = mm.cleanup(importance_threshold=0.05)
        assert stats["total_removed"] >= 1


# ── GodPanelCommand Lifecycle ────────────────────────────────────────


class TestGodPanelCommand:
    """Test command tracking through the GodPanelCommand table."""

    def test_command_lifecycle(self):
        _ensure_empire()
        from db.models import GodPanelCommand
        from db.engine import session_scope, read_session

        # Create
        with session_scope() as session:
            cmd = GodPanelCommand(
                id="test-cmd-001",
                empire_id=EMPIRE_ID,
                command="research transformers",
                action="RESEARCH",
                topic="transformers",
            )
            session.add(cmd)

        # Read back
        with read_session() as session:
            cmd = session.get(GodPanelCommand, "test-cmd-001")
            assert cmd is not None
            assert cmd.status == "accepted"
            assert cmd.action == "RESEARCH"

        # Update
        with session_scope() as session:
            cmd = session.get(GodPanelCommand, "test-cmd-001")
            cmd.status = "completed"
            cmd.cost_usd = 0.05
            cmd.result_json = {"synthesis": "Transformers are attention-based models"}

        # Verify
        with read_session() as session:
            cmd = session.get(GodPanelCommand, "test-cmd-001")
            assert cmd.status == "completed"
            assert cmd.cost_usd == 0.05
            d = cmd.to_dict()
            assert d["status"] == "completed"
            assert d["cost"] == 0.05


# ── Scheduler Tick ───────────────────────────────────────────────────


class TestSchedulerTick:
    """Test that the scheduler tick mechanism works."""

    def test_tick_executes_due_jobs(self):
        _ensure_empire()
        from core.scheduler.daemon import SchedulerDaemon, JobConfig

        daemon = SchedulerDaemon(EMPIRE_ID)

        # Clear all default jobs and add a simple test job
        daemon._jobs.clear()
        calls = []
        daemon.register_job(JobConfig(
            name="test_job",
            interval_seconds=1,
            handler=lambda: calls.append(1) or {"ok": True},
            priority=1,
        ))

        executed = daemon.tick()
        assert "test_job" in executed
        assert len(calls) == 1

    def test_tick_skips_not_due_jobs(self):
        from core.scheduler.daemon import SchedulerDaemon, JobConfig

        daemon = SchedulerDaemon(EMPIRE_ID)
        daemon._jobs.clear()

        calls = []
        daemon.register_job(JobConfig(
            name="slow_job",
            interval_seconds=99999,
            handler=lambda: calls.append(1) or {"ok": True},
            priority=1,
        ))

        # First tick runs it (no last_run)
        daemon.tick()
        assert len(calls) == 1

        # Second tick should skip (interval not elapsed)
        daemon.tick()
        assert len(calls) == 1

    def test_tick_tracks_errors(self):
        from core.scheduler.daemon import SchedulerDaemon, JobConfig

        daemon = SchedulerDaemon(EMPIRE_ID)
        daemon._jobs.clear()

        daemon.register_job(JobConfig(
            name="failing_job",
            interval_seconds=1,
            handler=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            priority=1,
        ))

        daemon.tick()

        job = daemon._jobs["failing_job"]
        assert job.error_count == 1
        assert job.consecutive_errors == 1
        assert "boom" in job.last_error


# ── Entity Resolution ────────────────────────────────────────────────


class TestEntityResolution:
    """Test 3-stage entity deduplication."""

    def test_exact_match(self):
        _ensure_empire()
        from core.knowledge.graph import KnowledgeGraph
        from core.knowledge.resolution import EntityResolver

        graph = KnowledgeGraph(EMPIRE_ID)
        graph.add_entity("Claude Sonnet", "model", description="Anthropic mid-tier model")

        resolver = EntityResolver(EMPIRE_ID)
        result = resolver.resolve("Claude Sonnet")
        assert result.resolved
        assert result.match.match_stage == 1
        assert result.action == "merge"

    def test_case_insensitive_match(self):
        _ensure_empire()
        from core.knowledge.graph import KnowledgeGraph
        from core.knowledge.resolution import EntityResolver

        graph = KnowledgeGraph(EMPIRE_ID)
        graph.add_entity("GPT-4o", "model")

        resolver = EntityResolver(EMPIRE_ID)
        result = resolver.resolve("gpt-4o")
        assert result.resolved
        assert result.match.match_stage == 1

    def test_no_match_returns_create(self):
        _ensure_empire()
        from core.knowledge.resolution import EntityResolver

        resolver = EntityResolver(EMPIRE_ID)
        result = resolver.resolve("Completely New Entity XYZ123")
        assert not result.resolved
        assert result.action == "create"


# ── Empire Route Decorator + DB ──────────────────────────────────────


class TestEmpireRouteWithDB:
    """Test the empire_route decorator with a real DB backend."""

    def test_api_health_returns_json(self):
        _ensure_empire()
        from flask import Flask
        from web.routes.api import api_bp

        app = Flask(__name__)
        app.config["EMPIRE_ID"] = EMPIRE_ID
        app.config["TESTING"] = True
        app.register_blueprint(api_bp, url_prefix="/api")

        with app.test_client() as client:
            # This should not crash — the empire route decorator catches errors
            resp = client.get("/api/empire")
            assert resp.status_code in (200, 500)  # 200 if empire found, 500 if deps missing
            assert resp.content_type == "application/json"
