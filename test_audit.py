"""Full system audit — tests every subsystem and reports bugs."""

import sys
import traceback

# Reset DB engine to avoid stale connections from prior processes
import db.engine as _eng
_eng._engine = None
_eng._session_factory = None
_eng._scoped_session = None

results = []

def test(name, fn):
    try:
        fn()
        results.append(("PASS", name, ""))
        print(f"  PASS: {name}")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  FAIL: {name} — {e}")
        traceback.print_exc()


# ── 1. Config ──────────────────────────────────────────────────────────

def test_config():
    from config.settings import get_settings, MODEL_CATALOG
    s = get_settings()
    assert s.empire_name, "No empire name"
    assert s.db_url, "No DB URL"
    assert len(MODEL_CATALOG) >= 5, f"Only {len(MODEL_CATALOG)} models"
    for key, m in MODEL_CATALOG.items():
        assert m.model_id, f"Model {key} has no model_id"
        assert m.provider in ("anthropic", "openai"), f"Unknown provider: {m.provider}"

print("=== 1. Config ===")
test("Settings load", test_config)


# ── 2. Database ────────────────────────────────────────────────────────

def test_db_connection():
    from db.engine import check_connection
    assert check_connection(), "DB connection failed"

def test_db_models():
    from db.engine import read_session
    from db.models import Empire, Lieutenant, KnowledgeEntity, MemoryEntry, Task
    with read_session() as session:
        assert session.query(Empire).count() >= 1, "No empires"
        assert session.query(Lieutenant).count() >= 1, "No lieutenants"

def test_db_manager():
    from db.engine import DatabaseManager
    dm = DatabaseManager()
    assert dm.check_health(), "DB manager health check failed"
    stats = dm.get_stats()
    assert "tables" in stats, "No tables in stats"

print("\n=== 2. Database ===")
test("Connection", test_db_connection)
test("Models", test_db_models)
test("DatabaseManager", test_db_manager)


# ── 3. Repositories ───────────────────────────────────────────────────

def test_lieutenant_repo():
    from db.engine import get_session
    from db.repositories.lieutenant import LieutenantRepository
    session = get_session()
    repo = LieutenantRepository(session)
    lts = repo.get_by_empire("empire-alpha")
    assert len(lts) == 6, f"Expected 6 lieutenants, got {len(lts)}"
    fleet = repo.get_fleet_summary("empire-alpha")
    assert fleet["total_lieutenants"] == 6

def test_directive_repo():
    from db.engine import get_session
    from db.repositories.directive import DirectiveRepository
    session = get_session()
    repo = DirectiveRepository(session)
    dirs = repo.get_by_empire("empire-alpha")
    assert isinstance(dirs, list)  # May be empty on fresh DB
    stats = repo.get_stats("empire-alpha")
    assert "by_status" in stats or "total" in stats

def test_task_repo():
    from db.engine import get_session
    from db.repositories.task import TaskRepository
    session = get_session()
    repo = TaskRepository(session)
    tasks = repo.get_recent(limit=5)
    assert isinstance(tasks, list)
    stats = repo.get_performance_stats(days=7)
    assert "total" in stats

def test_knowledge_repo():
    from db.engine import get_session
    from db.repositories.knowledge import KnowledgeRepository
    session = get_session()
    repo = KnowledgeRepository(session)
    stats = repo.get_graph_stats("empire-alpha")
    assert "entity_count" in stats  # May be 0 on fresh DB

def test_memory_repo():
    from db.engine import get_session
    from db.repositories.memory import MemoryRepository
    session = get_session()
    repo = MemoryRepository(session)
    stats = repo.get_stats("empire-alpha")
    assert "total" in stats  # May be 0 on fresh DB

def test_empire_repo():
    from db.engine import get_session
    from db.repositories.empire import EmpireRepository
    session = get_session()
    repo = EmpireRepository(session)
    health = repo.get_health_overview("empire-alpha")
    assert health["empire"]["status"] == "active"

def test_evolution_repo():
    from db.engine import get_session
    from db.repositories.evolution import EvolutionRepository
    session = get_session()
    repo = EvolutionRepository(session)
    # Should not error even with no data
    stats = repo.get_proposal_stats("empire-alpha")
    assert "total" in stats

print("\n=== 3. Repositories ===")
test("LieutenantRepo", test_lieutenant_repo)
test("DirectiveRepo", test_directive_repo)
test("TaskRepo", test_task_repo)
test("KnowledgeRepo", test_knowledge_repo)
test("MemoryRepo", test_memory_repo)
test("EmpireRepo", test_empire_repo)
test("EvolutionRepo", test_evolution_repo)


# ── 4. LLM Layer ──────────────────────────────────────────────────────

def test_llm_base():
    from llm.base import LLMRequest, LLMResponse, LLMMessage, ToolDefinition, RateLimiter, estimate_tokens
    msg = LLMMessage.user("test")
    assert msg.role == "user"
    req = LLMRequest(messages=[msg], model="test")
    assert not req.has_tools
    assert estimate_tokens("hello world test") >= 1
    rl = RateLimiter()
    assert rl.can_proceed()

def test_llm_anthropic_import():
    from llm.anthropic import AnthropicClient
    # Don't instantiate (needs API key in env), just check import
    assert AnthropicClient.provider_name == "anthropic"

def test_llm_router_import():
    from llm.router import ModelRouter, TaskMetadata, RoutingDecision
    meta = TaskMetadata(task_type="research", complexity="moderate")
    assert meta.task_type == "research"

def test_llm_schemas():
    from llm.schemas import (
        PlanningOutput, AnalysisOutput, CriticOutput, EntityExtractionOutput,
        DebateOutput, SynthesisOutput, ProposalOutput, ReviewOutput,
        parse_llm_output, pydantic_to_tool_schema, SCHEMA_REGISTRY,
    )
    assert len(SCHEMA_REGISTRY) >= 10
    schema = pydantic_to_tool_schema(CriticOutput, "test")
    assert "properties" in schema

    # Test JSON parsing
    result = parse_llm_output('{"summary": "test", "findings": []}', AnalysisOutput)
    assert result is not None
    assert result.summary == "test"

    # Test markdown extraction
    md = '```json\n{"summary": "from_md"}\n```'
    result2 = parse_llm_output(md, AnalysisOutput)
    assert result2 is not None

print("\n=== 4. LLM Layer ===")
test("Base classes", test_llm_base)
test("Anthropic client", test_llm_anthropic_import)
test("Router", test_llm_router_import)
test("Schemas", test_llm_schemas)


# ── 5. Core: ACE Engine ───────────────────────────────────────────────

def test_ace_imports():
    from core.ace.engine import ACEEngine, ACEContext, TaskInput, TaskResult
    from core.ace.pipeline import Pipeline, PipelineContext, PipelineConfig
    from core.ace.planner import Planner, Plan, SubTask, ComplexityEstimate
    from core.ace.executor import Executor, ExecutionResult
    from core.ace.critic import Critic, CriticEvaluation, RetryDecision
    from core.ace.quality_gates import QualityGateChain, GateResult
    from core.ace.tools import ToolRegistry

def test_ace_context():
    from core.ace.engine import ACEContext
    ctx = ACEContext(
        persona_prompt="You are a test agent.",
        domain_context="Testing",
        memories=["mem1", "mem2"],
        knowledge=["fact1"],
    )
    prompt = ctx.build_system_prompt()
    assert "test agent" in prompt
    assert "mem1" in prompt

def test_planner_complexity():
    from core.ace.planner import Planner
    p = Planner()
    est = p.estimate_complexity("Analyze market trends", "Comprehensive evaluation of global equity markets")
    assert est.level in ("simple", "moderate", "complex", "expert")
    assert est.estimated_tokens > 0

def test_quality_gates():
    from core.ace.quality_gates import QualityGateChain, ConfidenceGate, ContentLengthGate
    chain = QualityGateChain.create_default()
    result = chain.run("This is a short test output.", {"overall_score": 0.8, "confidence": 0.9, "completeness": 0.8, "coherence": 0.85})
    assert result.total_gates > 0
    assert isinstance(result.pass_rate, float)

print("\n=== 5. ACE Engine ===")
test("Imports", test_ace_imports)
test("Context building", test_ace_context)
test("Complexity estimation", test_planner_complexity)
test("Quality gates", test_quality_gates)


# ── 6. Core: Memory ───────────────────────────────────────────────────

def test_memory_manager():
    from core.memory.manager import MemoryManager
    mm = MemoryManager("empire-alpha")
    stats = mm.get_stats()
    assert stats.total_count >= 0

def test_memory_tiers():
    from core.memory.manager import MemoryManager
    from core.memory.semantic import SemanticMemory
    from core.memory.experiential import ExperientialMemory
    from core.memory.design import DesignMemory
    from core.memory.episodic import EpisodicMemory
    mm = MemoryManager("empire-alpha")
    sem = SemanticMemory(mm)
    exp = ExperientialMemory(mm)
    des = DesignMemory(mm)
    epi = EpisodicMemory(mm)
    # Should not error
    facts = sem.query_facts("test")
    lessons = exp.query_lessons("test")

def test_memory_store_recall():
    # Force fresh engine to avoid stale session locks
    import db.engine as _eng
    _eng._engine = None
    _eng._session_factory = None
    _eng._scoped_session = None

    from core.memory.manager import MemoryManager
    mm = MemoryManager("empire-alpha")
    mm._memory_repo = None
    entry = mm.store(
        content="Test fact: Claude Sonnet 4 was released in 2025",
        memory_type="semantic",
        title="Test memory",
        importance=0.8,
        tags=["test"],
    )
    assert "id" in entry
    recalled = mm.recall(query="Claude Sonnet", memory_types=["semantic"], limit=5)
    assert len(recalled) >= 1

print("\n=== 6. Memory ===")
test("Manager", test_memory_manager)
test("Tier imports", test_memory_tiers)
test("Store & recall", test_memory_store_recall)


# ── 7. Core: Knowledge ────────────────────────────────────────────────

def test_knowledge_graph():
    from core.knowledge.graph import KnowledgeGraph
    graph = KnowledgeGraph("empire-alpha")
    stats = graph.get_stats()
    assert stats.entity_count >= 0  # May be 0 on fresh DB

def test_knowledge_search():
    from core.knowledge.graph import KnowledgeGraph
    graph = KnowledgeGraph("empire-alpha")
    results = graph.find_entities(query="agent", limit=5)
    # Should find something from the directive run
    assert isinstance(results, list)

def test_knowledge_maintenance():
    from core.knowledge.maintenance import KnowledgeMaintainer
    m = KnowledgeMaintainer("empire-alpha")
    report = m.generate_knowledge_report()
    assert report.entity_count >= 0

print("\n=== 7. Knowledge ===")
test("Graph", test_knowledge_graph)
test("Search", test_knowledge_search)
test("Maintenance", test_knowledge_maintenance)


# ── 8. Core: Lieutenant ───────────────────────────────────────────────

def test_lieutenant_manager():
    from core.lieutenant.manager import LieutenantManager
    mgr = LieutenantManager("empire-alpha")
    lts = mgr.list_lieutenants()
    assert len(lts) == 6
    stats = mgr.get_fleet_stats()
    assert stats.total == 6

def test_persona_templates():
    from core.lieutenant.persona import PERSONA_TEMPLATES, list_persona_templates, create_persona, PersonaBuilder
    assert len(PERSONA_TEMPLATES) >= 5
    names = list_persona_templates()
    assert "financial_modeler" in names
    p = create_persona("research_analyst")
    assert p.domain == "research"
    prompt = p.build_system_prompt()
    assert len(prompt) > 20

    # Builder
    built = PersonaBuilder().with_name("Test").with_domain("test").with_expertise("testing").build()
    assert built.name == "Test"

def test_lieutenant_registry():
    from core.lieutenant.registry import LieutenantRegistry
    reg = LieutenantRegistry()
    reg.register("lt1", "emp1", "Test Lt", "testing", ["analysis"])
    found = reg.find_by_domain("testing")
    assert len(found) == 1
    stats = reg.get_registry_stats()
    assert stats.total_lieutenants == 1

def test_workload_balancer():
    from core.lieutenant.workload import WorkloadBalancer
    wb = WorkloadBalancer("empire-alpha")
    report = wb.get_workload_report()
    assert report.total_lieutenants >= 0

print("\n=== 8. Lieutenant ===")
test("Manager", test_lieutenant_manager)
test("Personas", test_persona_templates)
test("Registry", test_lieutenant_registry)
test("Workload", test_workload_balancer)


# ── 9. Core: Directives ───────────────────────────────────────────────

def test_directive_manager():
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager("empire-alpha")
    dirs = dm.list_directives()
    assert isinstance(dirs, list)
    stats = dm.get_stats()
    assert isinstance(stats, dict)

print("\n=== 9. Directives ===")
test("Manager", test_directive_manager)


# ── 10. Core: War Room ────────────────────────────────────────────────

def test_warroom_session():
    from core.warroom.session import WarRoomSession, SessionState
    s = WarRoomSession(empire_id="empire-alpha", session_type="planning")
    assert s.state == SessionState.CREATED
    s.add_participant("lt1", "Test", "testing")
    assert len(s.participants) == 1

def test_debate_strategies():
    from core.warroom.debate_strategies import get_strategy, list_strategies, DebateStrategy
    strategies = list_strategies()
    assert len(strategies) >= 4
    s = get_strategy(DebateStrategy.COLLABORATIVE)
    assert s is not None

print("\n=== 10. War Room ===")
test("Session", test_warroom_session)
test("Strategies", test_debate_strategies)


# ── 11. Core: Evolution ───────────────────────────────────────────────

def test_evolution_cycle():
    from core.evolution.cycle import EvolutionCycleManager
    ecm = EvolutionCycleManager("empire-alpha")
    # should_run_cycle returns bool (may be False if a cycle exists)
    can_run = ecm.should_run_cycle()
    assert isinstance(can_run, bool)
    stats = ecm.get_stats()
    assert stats.total_cycles >= 0

def test_evolution_proposer():
    from core.evolution.proposer import UpgradeProposer, EvolutionProposal, ProposalType
    p = UpgradeProposer("empire-alpha")
    prop = EvolutionProposal(title="Improve memory consolidation", description="Test proposal with enough detail for validation", confidence=0.7)
    v = p.validate_proposal(prop)
    assert v.valid, f"Validation failed: {v.issues}"

def test_evolution_reviewer():
    from core.evolution.reviewer import ProposalReviewer, ReviewAction
    assert ReviewAction.APPROVE == "approve"

def test_evolution_executor():
    from core.evolution.executor import EvolutionExecutor
    ex = EvolutionExecutor("empire-alpha")
    history = ex.get_execution_history()
    assert isinstance(history, list)

print("\n=== 11. Evolution ===")
test("Cycle manager", test_evolution_cycle)
test("Proposer", test_evolution_proposer)
test("Reviewer", test_evolution_reviewer)
test("Executor", test_evolution_executor)


# ── 12. Core: Scheduler ───────────────────────────────────────────────

def test_scheduler():
    from core.scheduler.daemon import SchedulerDaemon
    d = SchedulerDaemon("empire-alpha", tick_interval=60)
    status = d.get_status()
    assert status.jobs_registered >= 7
    runs = d.get_next_runs()
    assert len(runs) >= 7

def test_scheduler_jobs():
    from core.scheduler.jobs import JOB_REGISTRY, get_all_jobs, get_job
    assert len(JOB_REGISTRY) >= 8
    jobs = get_all_jobs("empire-alpha")
    assert len(jobs) >= 8
    hc = get_job("health_check", "empire-alpha")
    assert hc is not None
    assert hc.name == "health_check"

def test_health_checker():
    from core.scheduler.health import HealthChecker
    hc = HealthChecker("empire-alpha")
    report = hc.run_all_checks()
    assert "overall_status" in report
    assert "checks" in report

print("\n=== 12. Scheduler ===")
test("Daemon", test_scheduler)
test("Jobs", test_scheduler_jobs)
test("Health checker", test_health_checker)


# ── 13. Core: Retry ───────────────────────────────────────────────────

def test_retry():
    from core.retry.ralph_wiggum import RalphWiggumRetry, ErrorClass
    rw = RalphWiggumRetry(max_retries=3)
    assert rw._classify_error("rate limit exceeded") == ErrorClass.RATE_LIMIT
    assert rw._classify_error("server error 500") == ErrorClass.TRANSIENT
    assert rw._classify_error("invalid request format") == ErrorClass.PERMANENT
    assert rw._classify_error("quality below threshold") == ErrorClass.QUALITY_FAILURE
    backoff = rw._calculate_backoff(0)
    assert backoff > 0

print("\n=== 13. Retry ===")
test("Ralph Wiggum", test_retry)


# ── 14. Core: Routing & Budget ─────────────────────────────────────────

def test_budget():
    from core.routing.budget import BudgetManager
    bm = BudgetManager("empire-alpha")
    check = bm.check_budget(0.01)
    assert check.allowed
    assert bm.get_daily_spend() >= 0
    assert bm.get_monthly_spend() >= 0
    alerts = bm.get_budget_alerts()
    assert isinstance(alerts, list)

def test_pricing():
    from core.routing.pricing import PricingEngine
    pe = PricingEngine()
    cost = pe.calculate_cost("claude-sonnet-4", 1000, 500)
    assert cost > 0
    est = pe.estimate_task_cost("analysis", "moderate", "claude-sonnet-4", 4000)
    assert est.estimated_cost_usd > 0
    models = pe.compare_models("research", "moderate")
    assert len(models) >= 1

print("\n=== 14. Routing & Budget ===")
test("Budget manager", test_budget)
test("Pricing engine", test_pricing)


# ── 15. Core: Replication ──────────────────────────────────────────────

def test_empire_generator():
    from core.replication.generator import EmpireGenerator, EMPIRE_TEMPLATES
    gen = EmpireGenerator()
    templates = gen.get_templates()
    assert len(templates) >= 5
    assert any(t["key"] == "finance" for t in templates)

def test_empire_registry():
    from core.replication.registry import EmpireRegistry
    reg = EmpireRegistry()
    reg.register_empire("test-emp", "Test Empire", "testing")
    found = reg.find_empire_by_domain("testing")
    assert len(found) == 1
    routing = reg.route_directive("test task for testing domain", "testing")
    assert routing.empire_id == "test-emp"

print("\n=== 15. Replication ===")
test("Generator", test_empire_generator)
test("Registry", test_empire_registry)


# ── 16. Utils ──────────────────────────────────────────────────────────

def test_utils_text():
    from utils.text import truncate, estimate_tokens, format_cost, format_tokens, format_duration, chunk_text, slugify
    assert truncate("hello world", 5) == "he..."
    assert estimate_tokens("hello world test string") >= 1
    assert format_cost(0.005) == "$0.0050"  # 0.005 >= 0.001 so 4 decimal
    assert format_tokens(1500) == "1.5K"
    assert format_duration(90) == "1.5m"
    chunks = chunk_text("a" * 10000, chunk_size=4000)
    assert len(chunks) >= 3
    assert slugify("Hello World!") == "hello-world"

def test_utils_crypto():
    from utils.crypto import generate_id, generate_secret_key, hash_content, mask_api_key, generate_token, validate_token
    assert len(generate_id()) == 16
    assert len(generate_id("emp_")) > 16
    assert len(generate_secret_key()) > 20
    assert hash_content("test") == hash_content("test")
    assert mask_api_key("sk-ant-very-long-key-here") == "sk-a...here"
    token = generate_token("payload", 3600)
    valid, payload = validate_token(token)
    assert valid
    assert payload == "payload"

def test_utils_events():
    from utils.events import EventBus, Event, EventTypes
    bus = EventBus()
    received = []
    bus.subscribe("test.event", lambda e: received.append(e))
    bus.emit("test.event", data={"key": "value"})
    assert len(received) == 1
    assert received[0].data["key"] == "value"
    # Wildcard
    bus.subscribe("test.*", lambda e: received.append(e))
    bus.emit("test.other", data={"wild": True})
    assert len(received) == 2  # wildcard only (no exact match for test.other)

def test_utils_metrics():
    from utils.metrics import MetricsCollector, Counter, Gauge, Histogram
    mc = MetricsCollector()
    c = mc.counter("test.count")
    c.increment(5)
    assert c.value == 5
    g = mc.gauge("test.gauge", 10)
    g.set(42)
    assert g.value == 42
    h = mc.histogram("test.hist")
    h.observe(1.0)
    h.observe(2.0)
    h.observe(3.0)
    assert h.avg == 2.0
    assert h.min == 1.0
    assert h.max == 3.0

def test_utils_validators():
    from utils.validators import Validator, validate_directive
    v = validate_directive({"title": "Test", "description": "A test directive with enough detail", "priority": 3, "source": "human"})
    assert v.is_valid
    v2 = validate_directive({"title": "", "description": ""})
    assert not v2.is_valid

print("\n=== 16. Utils ===")
test("Text", test_utils_text)
test("Crypto", test_utils_crypto)
test("Events", test_utils_events)
test("Metrics", test_utils_metrics)
test("Validators", test_utils_validators)


# ── 17. Migrations ─────────────────────────────────────────────────────

def test_migrations():
    from db.migrations import MigrationRunner
    runner = MigrationRunner()
    status = runner.get_status()
    assert status["current_version"] >= 5
    integrity = runner.verify_integrity()
    assert integrity["valid"], f"Missing tables: {integrity['missing_tables']}"

print("\n=== 17. Migrations ===")
test("Schema integrity", test_migrations)


# ── 18. Search Infrastructure ──────────────────────────────────────────

def test_credibility():
    from core.search.credibility import CredibilityScorer, get_source_tiers
    scorer = CredibilityScorer()
    s1 = scorer.score("https://arxiv.org/abs/1234")
    assert s1.score > 0.9, f"arxiv should be >0.9, got {s1.score}"
    assert s1.tier == "authoritative"
    s2 = scorer.score("https://medium.com/foo")
    assert s2.score < 0.5, f"medium should be <0.5, got {s2.score}"
    s3 = scorer.score("https://anthropic.com/research")
    assert s3.score > 0.9
    assert s3.tier == "primary"
    # Unknown domain
    s4 = scorer.score("https://random-unknown-blog.xyz/post")
    assert 0.3 <= s4.score <= 0.6
    # Ranking
    ranked = scorer.rank_urls(["https://medium.com/x", "https://arxiv.org/y", "https://anthropic.com/z"])
    assert ranked[0][1].score > ranked[2][1].score
    # Tiers
    tiers = get_source_tiers()
    assert "primary" in tiers
    assert len(tiers["primary"]) >= 5

def test_cache():
    from core.search.cache import ScrapeCache
    cache = ScrapeCache("empire-alpha")
    # Miss
    assert cache.get("https://example.com/nonexistent") is None
    stats = cache.get_stats()
    assert stats.misses >= 1

def test_feeds_import():
    from core.search.feeds import FeedReader, AI_FEEDS
    reader = FeedReader("empire-alpha")
    feeds = reader.list_feeds()
    assert len(feeds) >= 10
    assert any(f["name"] == "Anthropic Research" for f in feeds)
    assert any(f["domain"] == "arxiv.org" for f in feeds)

def test_scraper_import():
    from core.search.scraper import WebScraper, ScrapedPage
    scraper = WebScraper("empire-alpha")
    # Just test import and init — don't actually fetch
    page = ScrapedPage(url="https://test.com", success=False)
    assert not page.success
    formatted = scraper.format_for_prompt(page)
    assert "Failed" in formatted

def test_web_searcher_import():
    from core.search.web import WebSearcher
    searcher = WebSearcher("empire-alpha")
    assert searcher.empire_id == "empire-alpha"

print("\n=== 18. Search Infrastructure ===")
test("Credibility scorer", test_credibility)
test("Scrape cache", test_cache)
test("RSS feeds config", test_feeds_import)
test("Scraper", test_scraper_import)
test("Web searcher", test_web_searcher_import)


# ── Summary ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")

if failed:
    print("\nFAILURES:")
    for status, name, err in results:
        if status == "FAIL":
            print(f"  FAIL: {name} — {err}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
