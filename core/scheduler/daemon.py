"""Scheduler daemon — the 60-second tick that drives everything autonomously."""

from __future__ import annotations

import logging
import signal
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for a scheduled job."""
    name: str
    interval_seconds: int
    handler: Callable[[], dict]
    enabled: bool = True
    priority: int = 5  # 1 = highest
    description: str = ""
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    avg_duration_ms: float = 0.0
    metadata_json: dict = field(default_factory=dict)

    @property
    def job_type(self) -> str:
        return self.name


@dataclass
class DaemonStatus:
    """Status of the scheduler daemon."""
    running: bool = False
    uptime_seconds: float = 0.0
    jobs_registered: int = 0
    jobs_active: int = 0
    last_tick: str = ""
    total_ticks: int = 0
    total_job_runs: int = 0
    errors: int = 0


@dataclass
class ScheduledRun:
    """Information about a scheduled job run."""
    job_name: str
    next_run: str
    interval_seconds: int
    last_run: str = ""
    status: str = "pending"


class SchedulerDaemon:
    """The autonomous scheduler daemon.

    Ticks every 60 seconds (configurable), checking which jobs are due
    and executing them. Drives learning cycles, evolution runs,
    health checks, and knowledge maintenance without human intervention.
    """

    def __init__(self, empire_id: str = "", tick_interval: int = 60):
        self.empire_id = empire_id
        self.tick_interval = tick_interval
        self._jobs: dict[str, JobConfig] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._tick_count = 0
        self._total_job_runs = 0
        self._total_errors = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Register default jobs
        self._register_default_jobs()

    def _register_default_jobs(self) -> None:
        """Register the default set of recurring jobs."""
        try:
            from config.settings import get_settings
            s = get_settings().scheduler
        except Exception:
            class DefaultScheduler:
                health_check_interval_minutes = 5
                learning_cycle_hours = 6
                evolution_cycle_hours = 12
                knowledge_maintenance_hours = 4
            s = DefaultScheduler()

        # (name, interval_seconds, priority, description)
        jobs = [
            # ── Core system jobs ─────────────────────────────────────
            ("health_check",         s.health_check_interval_minutes * 60, 1, "System health checks"),
            ("budget_check",         900,                                  2, "Budget limit checking"),
            ("directive_check",      300,                                  3, "Check for pending directives"),
            ("memory_decay",         3600,                                 3, "Apply memory decay"),
            # ── Knowledge & research ─────────────────────────────────
            ("knowledge_maintenance", s.knowledge_maintenance_hours * 3600, 4, "Knowledge graph maintenance"),
            ("intelligence_sweep",   43200,                                4, "Proactive discovery across AI sources"),
            ("autonomous_research",  21600,                                4, "Gap-driven autonomous research"),
            ("learning_cycle",       s.learning_cycle_hours * 3600,        5, "Lieutenant learning cycles"),
            ("quality_scoring",      21600,                                5, "8-dimension entity quality scoring"),
            ("duplicate_resolution", 14400,                                5, "3-stage fuzzy entity deduplication"),
            ("cross_synthesis",      28800,                                5, "Synthesize overlapping knowledge across domains"),
            ("autonomous_warroom",   21600,                                5, "Auto-detect cross-domain topics and run debates"),
            # ── Evolution & maintenance ──────────────────────────────
            ("evolution_cycle",      s.evolution_cycle_hours * 3600,        6, "Self-improvement evolution cycle"),
            ("memory_compression",   43200,                                6, "LLM-powered memory compression"),
            ("llm_audit",            43200,                                6, "Deep LLM audit for contaminated entities"),
            ("iterative_deepening",  28800,                                6, "Deepen high-signal shallow research"),
            ("content_generation",   86400,                                7, "Auto-generate research digest"),
            ("auto_spawn",           86400,                                7, "Auto-spawn lieutenants for uncovered clusters"),
            ("shallow_enrichment",   21600,                                7, "Enrich low-detail knowledge graph entities"),
            ("cleanup",              86400,                                8, "Archive and cleanup old data"),
            ("embedding_backfill",   3600,                                 8, "Backfill embeddings for memories and entities"),
        ]

        for name, interval, priority, description in jobs:
            handler = getattr(self, f"_run_{name}", None)
            if handler:
                self.register_job(JobConfig(
                    name=name, interval_seconds=interval, handler=handler,
                    priority=priority, description=description,
                ))

    def register_job(self, job: JobConfig) -> None:
        """Register a recurring job."""
        with self._lock:
            self._jobs[job.name] = job
        logger.debug("Registered job: %s (interval=%ds)", job.name, job.interval_seconds)

    def unregister_job(self, job_name: str) -> None:
        """Unregister a job."""
        with self._lock:
            self._jobs.pop(job_name, None)

    def _sync_from_db(self) -> int:
        """Load persisted job state from the scheduler_jobs table.

        Restores last_run, run_count, error_count so the daemon resumes
        where it left off after a process restart instead of re-running everything.

        Returns number of jobs synced.
        """
        synced = 0
        try:
            from db.engine import session_scope
            from db.models import SchedulerJob
            from sqlalchemy import select, and_

            with session_scope() as session:
                stmt = select(SchedulerJob).where(SchedulerJob.empire_id == self.empire_id)
                db_jobs = {j.job_type: j for j in session.execute(stmt).scalars().all()}

                with self._lock:
                    for job in self._jobs.values():
                        db_job = db_jobs.get(job.job_type)
                        if db_job and db_job.last_run_at:
                            job.last_run = db_job.last_run_at
                            job.run_count = db_job.run_count or 0
                            job.error_count = db_job.error_count or 0
                            job.consecutive_errors = db_job.consecutive_errors or 0
                            job.enabled = db_job.enabled
                            synced += 1

            if synced:
                logger.info("Synced %d jobs from DB — resuming from last known state", synced)
        except Exception as e:
            logger.debug("DB sync on startup skipped: %s", e)
        return synced

    def _sync_to_db(self) -> None:
        """Persist current job state to the scheduler_jobs table.

        Called after each tick so job state survives process restarts.
        """
        try:
            from db.engine import session_scope
            from db.models import SchedulerJob
            from sqlalchemy import select

            with session_scope() as session:
                with self._lock:
                    for job in self._jobs.values():
                        # Upsert: find existing or create
                        existing = session.execute(
                            select(SchedulerJob).where(
                                SchedulerJob.empire_id == self.empire_id,
                                SchedulerJob.job_type == job.job_type,
                            )
                        ).scalar_one_or_none()

                        now = datetime.now(timezone.utc)
                        next_run = None
                        if job.last_run:
                            from datetime import timedelta
                            next_run = job.last_run + timedelta(seconds=job.interval_seconds)

                        if existing:
                            existing.last_run_at = job.last_run
                            existing.next_run_at = next_run
                            existing.run_count = job.run_count
                            existing.success_count = job.run_count - job.error_count
                            existing.error_count = job.error_count
                            existing.consecutive_errors = job.consecutive_errors
                            existing.last_error = job.last_error or None
                            existing.avg_duration_ms = job.avg_duration_ms or None
                            existing.enabled = job.enabled
                            existing.status = "active" if job.enabled else "disabled"
                        else:
                            db_job = SchedulerJob(
                                empire_id=self.empire_id,
                                job_type=job.job_type,
                                name=job.name,
                                description=job.description,
                                status="active" if job.enabled else "disabled",
                                enabled=job.enabled,
                                interval_seconds=job.interval_seconds,
                                priority=job.priority,
                                last_run_at=job.last_run,
                                next_run_at=next_run,
                                run_count=job.run_count,
                                success_count=job.run_count - job.error_count,
                                error_count=job.error_count,
                                consecutive_errors=job.consecutive_errors,
                                last_error=job.last_error or None,
                                avg_duration_ms=job.avg_duration_ms or None,
                            )
                            session.add(db_job)
        except Exception as e:
            logger.debug("DB sync after tick failed: %s", e)

    def start(self) -> None:
        """Start the scheduler daemon in a background thread.

        On startup, syncs job state from the DB so the daemon resumes where
        it left off after a process restart. Only staggers jobs if this is a
        fresh start (no DB state).
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._start_time = time.time()
        self._stop_event.clear()

        # Try to restore job state from DB (survives process restarts)
        synced = self._sync_from_db()

        if synced == 0:
            # Fresh start — stagger jobs so they don't all fire on first tick
            now = datetime.now(timezone.utc)
            immediate_jobs = {"health_check", "budget_check", "directive_check"}
            with self._lock:
                for job in self._jobs.values():
                    if job.name not in immediate_jobs:
                        job.last_run = now  # Will wait full interval before first run

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="empire-scheduler")
        self._thread.start()

        logger.info("Scheduler daemon started (tick_interval=%ds, jobs=%d, synced=%d)", self.tick_interval, len(self._jobs), synced)

    def stop(self) -> None:
        """Stop the scheduler daemon gracefully."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        logger.info("Scheduler daemon stopped")

    def _run_loop(self) -> None:
        """Main scheduler loop — runs until stopped."""
        while self._running and not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as e:
                logger.error("Scheduler tick error: %s", e)
                self._total_errors += 1

            self._stop_event.wait(timeout=self.tick_interval)

    def _try_acquire_tick_lock(self) -> bool:
        """Try to acquire a Postgres advisory lock for this tick.

        Returns True if this worker should run the tick.
        On SQLite or connection failure, always returns True.
        """
        try:
            from db.engine import get_engine
            engine = get_engine()
            if "postgresql" not in str(engine.url):
                return True
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text("SELECT pg_try_advisory_lock(42)"))
                acquired = result.scalar()
                if acquired:
                    conn.execute(text("SELECT pg_advisory_unlock(42)"))
                return bool(acquired)
        except Exception:
            return True

    def tick(self) -> list[str]:
        """Execute a single scheduler tick.

        Uses Postgres advisory lock so only one worker runs jobs per tick.

        Returns:
            List of job names that were executed.
        """
        if not self._try_acquire_tick_lock():
            return []

        with self._lock:
            self._tick_count += 1
        now = datetime.now(timezone.utc)
        executed = []

        # Sort jobs by priority
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.priority)

        for job in jobs:
            with self._lock:
                if not job.enabled:
                    continue

                # Check if job is due
                if job.last_run is not None and (now - job.last_run).total_seconds() < job.interval_seconds:
                    continue

            # Run handler WITHOUT lock (can be slow)
            try:
                start = time.time()
                result = job.handler()
                duration_ms = (time.time() - start) * 1000

                with self._lock:
                    job.last_run = now
                    job.run_count += 1
                    job.consecutive_errors = 0
                    job.avg_duration_ms = (job.avg_duration_ms * 0.9 + duration_ms * 0.1) if job.avg_duration_ms else duration_ms
                    self._total_job_runs += 1
                executed.append(job.name)

                logger.debug("Job %s completed in %.1fms", job.name, duration_ms)

            except Exception as e:
                with self._lock:
                    job.error_count += 1
                    job.consecutive_errors += 1
                    job.last_error = str(e)
                    self._total_errors += 1
                logger.error("Job %s failed: %s", job.name, e)

                with self._lock:
                    # Disable job after 20 consecutive errors (auto-re-enables after 10 ticks)
                    if job.consecutive_errors >= 20:
                        job.enabled = False
                        job.metadata_json = job.metadata_json or {}
                        job.metadata_json["disabled_at_tick"] = self._tick_count
                        logger.warning("Job %s disabled after %d consecutive errors", job.name, job.consecutive_errors)

        # Auto-re-enable disabled jobs after 10 ticks (~50 min)
        with self._lock:
            for job in self._jobs.values():
                if not job.enabled:
                    meta = getattr(job, "metadata_json", None) or {}
                    disabled_at = meta.get("disabled_at_tick", 0)
                    if self._tick_count - disabled_at >= 10:
                        job.enabled = True
                        job.consecutive_errors = 0
                        logger.info("Job %s auto-re-enabled after cooldown", job.name)

        # Persist job state to DB so it survives process restarts
        if executed:
            self._sync_to_db()

        return executed

    def force_run(self, job_name: str) -> dict:
        """Immediately run a specific job."""
        with self._lock:
            job = self._jobs.get(job_name)

        if not job:
            return {"error": f"Job not found: {job_name}"}

        try:
            start = time.time()
            result = job.handler()
            duration = time.time() - start
            with self._lock:
                job.last_run = datetime.now(timezone.utc)
                job.run_count += 1
            return {"job": job_name, "duration_seconds": duration, "result": result}
        except Exception as e:
            return {"job": job_name, "error": str(e)}

    def pause_job(self, job_name: str) -> bool:
        """Pause a job."""
        with self._lock:
            if job_name in self._jobs:
                self._jobs[job_name].enabled = False
                return True
        return False

    def resume_job(self, job_name: str) -> bool:
        """Resume a paused job."""
        with self._lock:
            if job_name in self._jobs:
                self._jobs[job_name].enabled = True
                self._jobs[job_name].consecutive_errors = 0
                return True
        return False

    def get_status(self) -> DaemonStatus:
        """Get daemon status."""
        uptime = time.time() - self._start_time if self._start_time else 0
        with self._lock:
            active_jobs = sum(1 for j in self._jobs.values() if j.enabled)

        return DaemonStatus(
            running=self._running,
            uptime_seconds=uptime,
            jobs_registered=len(self._jobs),
            jobs_active=active_jobs,
            last_tick=datetime.now(timezone.utc).isoformat(),
            total_ticks=self._tick_count,
            total_job_runs=self._total_job_runs,
            errors=self._total_errors,
        )

    def get_next_runs(self) -> list[ScheduledRun]:
        """Get upcoming scheduled runs."""
        runs = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for job in self._jobs.values():
                if not job.enabled:
                    continue

                if job.last_run:
                    next_run = job.last_run + timedelta(seconds=job.interval_seconds)
                else:
                    next_run = now

                runs.append(ScheduledRun(
                    job_name=job.name,
                    next_run=next_run.isoformat(),
                    interval_seconds=job.interval_seconds,
                    last_run=job.last_run.isoformat() if job.last_run else "",
                    status="active" if job.enabled else "paused",
                ))

        runs.sort(key=lambda r: r.next_run)
        return runs

    def get_job_status(self, job_name: str) -> dict:
        """Get status of a specific job."""
        with self._lock:
            job = self._jobs.get(job_name)
            if not job:
                return {"error": "Job not found"}
            return {
                "name": job.name,
                "type": job.job_type,
                "enabled": job.enabled,
                "interval_seconds": job.interval_seconds,
                "run_count": job.run_count,
                "error_count": job.error_count,
                "consecutive_errors": job.consecutive_errors,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_error": job.last_error,
                "avg_duration_ms": job.avg_duration_ms,
            }

    # ── Job handlers ───────────────────────────────────────────────────

    def _run_health_check(self) -> dict:
        """Run system health checks."""
        from core.scheduler.health import HealthChecker
        checker = HealthChecker(self.empire_id)
        report = checker.run_all_checks()
        return {"status": report.get("overall_status", "unknown"), "checks": len(report.get("checks", []))}

    def _run_memory_decay(self) -> dict:
        """Apply memory decay."""
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        decayed = mm.decay()
        return {"decayed": decayed}

    def _run_knowledge_maintenance(self) -> dict:
        """Run knowledge maintenance."""
        from core.knowledge.maintenance import KnowledgeMaintainer
        maintainer = KnowledgeMaintainer(self.empire_id)
        report = maintainer.run_maintenance()
        return {"health_score": report.health_score, "entities": report.entity_count}

    def _run_learning_cycle(self) -> dict:
        """Run lieutenant learning cycles."""
        from core.lieutenant.manager import LieutenantManager
        manager = LieutenantManager(self.empire_id)
        return manager.run_all_learning_cycles()

    def _run_evolution_cycle(self) -> dict:
        """Run evolution cycle."""
        from core.evolution.cycle import EvolutionCycleManager
        ecm = EvolutionCycleManager(self.empire_id)
        if ecm.should_run_cycle():
            result = ecm.run_full_cycle()
            return {"proposals": result.proposals_collected, "applied": result.applied}
        return {"skipped": True, "reason": "cooldown"}

    def _run_budget_check(self) -> dict:
        """Check budget limits."""
        from core.routing.budget import BudgetManager
        bm = BudgetManager(self.empire_id)
        return {
            "daily_spend": bm.get_daily_spend(),
            "monthly_spend": bm.get_monthly_spend(),
            "over_budget": bm.is_over_budget(),
        }

    def _run_directive_check(self) -> dict:
        """Check for pending directives."""
        from core.directives.manager import DirectiveManager
        dm = DirectiveManager(self.empire_id)
        pending = dm.list_directives(status="pending")
        return {"pending_count": len(pending)}

    def _run_cleanup(self) -> dict:
        """Archive old data and cleanup."""
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        cleanup = mm.cleanup()
        return cleanup

    def _run_intelligence_sweep(self) -> dict:
        """Proactive discovery across AI sources."""
        try:
            from core.search.sweep import IntelligenceSweep
            sweep = IntelligenceSweep(self.empire_id)
            results = sweep.run_sweep()
            return {"discoveries": len(results) if isinstance(results, list) else 0}
        except Exception as e:
            logger.warning("Intelligence sweep failed: %s", e)
            return {"error": str(e)}

    def _run_quality_scoring(self) -> dict:
        """Score entity quality across 8 dimensions."""
        try:
            from core.knowledge.quality import EntityQualityScorer
            scorer = EntityQualityScorer(self.empire_id)
            scored = scorer.score_all()
            return {"scored": scored}
        except Exception as e:
            logger.warning("Quality scoring failed: %s", e)
            return {"error": str(e)}

    def _run_duplicate_resolution(self) -> dict:
        """3-stage fuzzy entity deduplication."""
        try:
            from core.knowledge.resolution import EntityResolver
            resolver = EntityResolver(self.empire_id)
            merged = resolver.resolve_all()
            return {"merged": merged}
        except Exception as e:
            logger.warning("Duplicate resolution failed: %s", e)
            return {"error": str(e)}

    def _run_memory_compression(self) -> dict:
        """LLM-powered memory compression."""
        try:
            from core.memory.compression import MemoryCompressor
            compressor = MemoryCompressor(self.empire_id)
            compressed = compressor.compress_old_memories()
            return {"compressed": compressed}
        except Exception as e:
            logger.warning("Memory compression failed: %s", e)
            return {"error": str(e)}

    def _run_autonomous_research(self) -> dict:
        """Closed-loop autonomous research via AutoResearcher.

        Full pipeline: detect gaps → generate questions → search → scrape →
        extract entities → synthesize → update strategy tracker.
        """
        try:
            from core.research.autoresearcher import AutoResearcher

            researcher = AutoResearcher(self.empire_id)
            result = researcher.run_cycle()

            return {
                "cycle_id": result.cycle_id,
                "gaps_detected": result.gaps_detected,
                "questions_generated": result.questions_generated,
                "questions_researched": result.questions_researched,
                "total_findings": result.total_findings,
                "novel_findings": result.novel_findings,
                "entities_extracted": result.entities_extracted,
                "memories_stored": result.memories_stored,
                "synthesis_reports": result.synthesis_reports,
                "domains_covered": result.domains_covered,
                "cost_usd": result.total_cost_usd,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors[:5],
            }
        except Exception as e:
            logger.warning("Autonomous research failed: %s", e)
            return {"error": str(e)}

    def _run_content_generation(self) -> dict:
        """Auto-generate a research digest from recent findings."""
        try:
            from core.content.generator import ContentGenerator
            gen = ContentGenerator(self.empire_id)
            report = gen.generate_weekly_digest()
            return {"report_id": report.get("id", ""), "sections": report.get("sections", 0)}
        except Exception as e:
            logger.warning("Content generation failed: %s", e)
            return {"error": str(e)}

    def _run_llm_audit(self) -> dict:
        """Deep LLM audit for contaminated/hallucinated entities."""
        try:
            from core.knowledge.maintenance import KnowledgeMaintainer
            maintainer = KnowledgeMaintainer(self.empire_id)
            return maintainer.deep_llm_audit(batch_size=20)
        except Exception as e:
            logger.warning("LLM audit failed: %s", e)
            return {"error": str(e)}

    def _run_auto_spawn(self) -> dict:
        """Auto-spawn lieutenants for uncovered topic clusters."""
        try:
            from core.knowledge.maintenance import KnowledgeMaintainer
            maintainer = KnowledgeMaintainer(self.empire_id)
            spawned = maintainer.auto_spawn_lieutenants(max_spawns=2)
            return {"spawned": len(spawned), "details": spawned}
        except Exception as e:
            logger.warning("Auto-spawn failed: %s", e)
            return {"error": str(e)}

    def _run_iterative_deepening(self) -> dict:
        """Detect high-signal topics and deepen research."""
        try:
            from core.research.deepening import IterativeDeepener
            deepener = IterativeDeepener(self.empire_id)
            results = deepener.run_deepening_cycle(max_topics=3)
            return {
                "topics_deepened": len(results),
                "new_entities": sum(r.new_entities for r in results),
                "new_relations": sum(r.new_relations for r in results),
                "topics": [r.topic for r in results],
            }
        except Exception as e:
            logger.warning("Iterative deepening failed: %s", e)
            return {"error": str(e)}

    def _run_shallow_enrichment(self) -> dict:
        """Find and enrich low-detail entities."""
        try:
            from core.research.enrichment import ShallowEnricher
            enricher = ShallowEnricher(self.empire_id)
            result = enricher.run_enrichment_cycle(max_entities=10)
            return {
                "scanned": result.entities_scanned,
                "enriched": result.enriched,
                "descriptions_improved": result.descriptions_improved,
                "fields_added": result.fields_added,
            }
        except Exception as e:
            logger.warning("Shallow enrichment failed: %s", e)
            return {"error": str(e)}

    def _run_cross_synthesis(self) -> dict:
        """Find cross-domain overlaps and synthesize insights."""
        try:
            from core.research.cross_synthesis import CrossLieutenantSynthesizer
            synthesizer = CrossLieutenantSynthesizer(self.empire_id)
            result = synthesizer.run_synthesis_cycle(max_syntheses=3)
            return {
                "overlaps_detected": result.overlaps_detected,
                "syntheses_produced": result.syntheses_produced,
                "insights": result.total_insights,
                "cost_usd": result.total_cost_usd,
            }
        except Exception as e:
            logger.warning("Cross-lieutenant synthesis failed: %s", e)
            return {"error": str(e)}

    def _run_autonomous_warroom(self) -> dict:
        """Auto-detect cross-domain topics and run lieutenant debates."""
        try:
            from core.warroom.session import run_autonomous_debate
            return run_autonomous_debate(self.empire_id)
        except Exception as e:
            logger.warning("Autonomous war room failed: %s", e)
            return {"error": str(e)}

    def _run_embedding_backfill(self) -> dict:
        """Backfill embeddings for memories and KG entities that lack them."""
        from core.memory.embeddings import backfill_embeddings
        return backfill_embeddings(self.empire_id)
