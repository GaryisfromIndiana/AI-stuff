"""REST API routes for programmatic access to Empire."""

from __future__ import annotations

import logging
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)


# ── Empire ─────────────────────────────────────────────────────────────

@api_bp.route("/empire")
def get_empire():
    """Get current empire info."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    try:
        from db.engine import get_session
        from db.repositories.empire import EmpireRepository
        session = get_session()
        repo = EmpireRepository(session)
        health = repo.get_health_overview(empire_id)
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/empire/network")
def get_network():
    """Get network stats across all empires."""
    try:
        from db.engine import get_session
        from db.repositories.empire import EmpireRepository
        session = get_session()
        repo = EmpireRepository(session)
        return jsonify(repo.get_network_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Lieutenants ────────────────────────────────────────────────────────

@api_bp.route("/lieutenants")
def api_list_lieutenants():
    """List all lieutenants."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.lieutenant.manager import LieutenantManager
    manager = LieutenantManager(empire_id)
    return jsonify(manager.list_lieutenants(
        status=request.args.get("status"),
        domain=request.args.get("domain"),
    ))


@api_bp.route("/lieutenants", methods=["POST"])
def api_create_lieutenant():
    """Create a lieutenant."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json()
    from core.lieutenant.manager import LieutenantManager
    manager = LieutenantManager(empire_id)
    lt = manager.create_lieutenant(
        name=data.get("name", ""),
        template=data.get("template", ""),
        domain=data.get("domain", ""),
    )
    return jsonify({"id": lt.id, "name": lt.name, "domain": lt.domain}), 201


@api_bp.route("/lieutenants/<lt_id>/task", methods=["POST"])
def api_lieutenant_task(lt_id: str):
    """Submit a task to a lieutenant."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json()
    from core.lieutenant.manager import LieutenantManager
    from core.ace.engine import TaskInput
    manager = LieutenantManager(empire_id)
    lt = manager.get_lieutenant(lt_id)
    if not lt:
        return jsonify({"error": "Lieutenant not found"}), 404
    task = TaskInput(title=data.get("title", ""), description=data.get("description", ""), task_type=data.get("type", "general"))
    result = lt.execute_task(task)
    return jsonify(result.to_dict())


# ── Directives ─────────────────────────────────────────────────────────

@api_bp.route("/directives")
def api_list_directives():
    """List directives."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager(empire_id)
    return jsonify(dm.list_directives(status=request.args.get("status")))


@api_bp.route("/directives", methods=["POST"])
def api_create_directive():
    """Create a directive."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json()
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager(empire_id)
    result = dm.create_directive(
        title=data.get("title", ""),
        description=data.get("description", ""),
        priority=data.get("priority", 5),
    )
    return jsonify(result), 201


@api_bp.route("/directives/<directive_id>/execute", methods=["POST"])
def api_execute_directive(directive_id: str):
    """Execute a directive."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager(empire_id)
    return jsonify(dm.execute_directive(directive_id))


@api_bp.route("/directives/<directive_id>/progress")
def api_directive_progress(directive_id: str):
    """Get directive progress."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.directives.manager import DirectiveManager
    dm = DirectiveManager(empire_id)
    return jsonify(dm.get_progress(directive_id).__dict__)


@api_bp.route("/directives/<directive_id>/report")
def api_directive_report(directive_id: str):
    """Get the full output/report from a completed directive."""
    try:
        from db.engine import get_session
        from db.repositories.directive import DirectiveRepository
        from db.repositories.task import TaskRepository

        session = get_session()
        dir_repo = DirectiveRepository(session)
        task_repo = TaskRepository(session)

        directive = dir_repo.get(directive_id)
        if not directive:
            return jsonify({"error": "Directive not found"}), 404

        # Get all tasks with their output
        tasks = task_repo.get_by_directive(directive_id)

        # Build report
        report_sections = []
        for task in tasks:
            output = task.output_json or {}
            content = output.get("content", "")
            if content:
                report_sections.append({
                    "title": task.title,
                    "wave": task.wave_number,
                    "lieutenant": task.lieutenant_id,
                    "status": task.status,
                    "quality_score": task.quality_score,
                    "cost_usd": task.cost_usd,
                    "content": content,
                })

        # Get war room synthesis
        from db.models import WarRoom
        from sqlalchemy import select, desc
        war_rooms = list(session.execute(
            select(WarRoom).where(WarRoom.directive_id == directive_id).order_by(desc(WarRoom.created_at))
        ).scalars().all())

        synthesis = {}
        if war_rooms:
            synthesis = war_rooms[0].synthesis_json or {}

        return jsonify({
            "directive": {
                "id": directive.id,
                "title": directive.title,
                "description": directive.description,
                "status": directive.status,
                "total_cost": directive.total_cost_usd,
                "quality_score": directive.quality_score,
                "created_at": directive.created_at.isoformat() if directive.created_at else None,
                "completed_at": directive.completed_at.isoformat() if directive.completed_at else None,
            },
            "sections": report_sections,
            "total_sections": len(report_sections),
            "war_room_synthesis": synthesis,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/reports/latest")
def api_latest_reports():
    """Get the latest research reports."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    try:
        from db.engine import get_session
        from db.repositories.directive import DirectiveRepository
        session = get_session()
        repo = DirectiveRepository(session)
        completed = repo.get_completed(empire_id, days=30, limit=10)

        reports = []
        for d in completed:
            reports.append({
                "id": d.id,
                "title": d.title,
                "status": d.status,
                "quality_score": d.quality_score,
                "total_cost": d.total_cost_usd,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "completed_at": d.completed_at.isoformat() if d.completed_at else None,
                "report_url": f"/api/directives/{d.id}/report",
            })

        # Also include recent research from memory
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        research_memories = mm.recall(
            query="research synthesis",
            memory_types=["semantic"],
            limit=10,
        )

        return jsonify({
            "directive_reports": reports,
            "research_entries": research_memories,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Knowledge ──────────────────────────────────────────────────────────

@api_bp.route("/knowledge/stats")
def api_knowledge_stats():
    """Get knowledge graph stats."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.knowledge.graph import KnowledgeGraph
    graph = KnowledgeGraph(empire_id)
    return jsonify(graph.get_stats().__dict__)


@api_bp.route("/knowledge/entities")
def api_knowledge_entities():
    """Search knowledge entities."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.knowledge.graph import KnowledgeGraph
    graph = KnowledgeGraph(empire_id)
    entities = graph.find_entities(
        query=request.args.get("q", ""),
        entity_type=request.args.get("type", ""),
        limit=int(request.args.get("limit", 20)),
    )
    return jsonify([{"name": e.name, "type": e.entity_type, "confidence": e.confidence, "importance": e.importance} for e in entities])


@api_bp.route("/knowledge/entity/<entity_name>/neighbors")
def api_entity_neighbors(entity_name: str):
    """Get entity neighbors."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.knowledge.graph import KnowledgeGraph
    graph = KnowledgeGraph(empire_id)
    neighbors = graph.get_neighbors(entity_name, max_depth=int(request.args.get("depth", 2)))
    return jsonify([{"name": n.name, "type": n.entity_type, "depth": n.depth} for n in neighbors])


# ── Memory ─────────────────────────────────────────────────────────────

@api_bp.route("/memory/stats")
def api_memory_stats():
    """Get memory stats."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.memory.manager import MemoryManager
    mm = MemoryManager(empire_id)
    return jsonify(mm.get_stats().__dict__)


@api_bp.route("/memory/search")
def api_memory_search():
    """Search memories."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.memory.manager import MemoryManager
    mm = MemoryManager(empire_id)
    return jsonify(mm.search(
        query=request.args.get("q", ""),
        memory_types=request.args.getlist("type") or None,
        limit=int(request.args.get("limit", 20)),
    ))


# ── Evolution ──────────────────────────────────────────────────────────

@api_bp.route("/evolution/stats")
def api_evolution_stats():
    """Get evolution stats."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.evolution.cycle import EvolutionCycleManager
    ecm = EvolutionCycleManager(empire_id)
    return jsonify(ecm.get_stats().__dict__)


@api_bp.route("/evolution/run", methods=["POST"])
def api_run_evolution():
    """Run evolution cycle."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.evolution.cycle import EvolutionCycleManager
    ecm = EvolutionCycleManager(empire_id)
    result = ecm.run_full_cycle()
    return jsonify({"proposals": result.proposals_collected, "approved": result.approved, "applied": result.applied})


# ── Budget ─────────────────────────────────────────────────────────────

@api_bp.route("/budget")
def api_budget():
    """Get budget summary."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.routing.budget import BudgetManager
    bm = BudgetManager(empire_id)
    report = bm.get_budget_report()
    return jsonify({
        "daily_spend": report.daily_spend, "monthly_spend": report.monthly_spend,
        "daily_remaining": report.daily_remaining, "monthly_remaining": report.monthly_remaining,
        "alerts": [{"message": a.message, "severity": a.severity} for a in report.alerts],
    })


# ── Health ─────────────────────────────────────────────────────────────

@api_bp.route("/health")
def api_health():
    """System health check."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    from core.scheduler.health import HealthChecker
    checker = HealthChecker(empire_id)
    return jsonify(checker.run_all_checks())


# ── Replication ────────────────────────────────────────────────────────

@api_bp.route("/empires/generate", methods=["POST"])
def api_generate_empire():
    """Generate a new empire."""
    data = request.get_json()
    from core.replication.generator import EmpireGenerator
    gen = EmpireGenerator()
    result = gen.generate_empire(
        name=data.get("name", ""),
        template=data.get("template", ""),
        domain=data.get("domain", "general"),
        description=data.get("description", ""),
    )
    return jsonify({
        "empire_id": result.empire_id, "lieutenants": result.lieutenants_created,
        "ready": result.launch_ready,
    }), 201


@api_bp.route("/empires/templates")
def api_empire_templates():
    """Get empire templates."""
    from core.replication.generator import EmpireGenerator
    gen = EmpireGenerator()
    return jsonify(gen.get_templates())


# ── Web Search ─────────────────────────────────────────────────────────

@api_bp.route("/search/web")
def api_web_search():
    """Search the web."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query parameter 'q' required"}), 400
    max_results = int(request.args.get("limit", 10))
    from core.search.web import WebSearcher
    searcher = WebSearcher(current_app.config.get("EMPIRE_ID", ""))
    result = searcher.search_and_summarize(query, max_results=max_results)
    return jsonify(result)


@api_bp.route("/search/news")
def api_news_search():
    """Search news articles."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query parameter 'q' required"}), 400
    max_results = int(request.args.get("limit", 10))
    time_range = request.args.get("range", "w")
    from core.search.web import WebSearcher
    searcher = WebSearcher(current_app.config.get("EMPIRE_ID", ""))
    response = searcher.search_news(query, max_results=max_results, time_range=time_range)
    return jsonify({
        "query": query,
        "results": [{"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source, "published": r.published} for r in response.results],
        "total": response.total_results,
    })


@api_bp.route("/search/ai")
def api_ai_search():
    """Search for AI-specific news and developments."""
    topic = request.args.get("topic", "")
    max_results = int(request.args.get("limit", 10))
    from core.search.web import WebSearcher
    searcher = WebSearcher(current_app.config.get("EMPIRE_ID", ""))
    response = searcher.search_ai_news(topic, max_results=max_results)
    return jsonify({
        "topic": topic,
        "results": [{"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source, "published": r.published} for r in response.results],
        "total": response.total_results,
    })


@api_bp.route("/search/papers")
def api_paper_search():
    """Search for AI research papers."""
    topic = request.args.get("topic", "")
    if not topic:
        return jsonify({"error": "Query parameter 'topic' required"}), 400
    max_results = int(request.args.get("limit", 10))
    from core.search.web import WebSearcher
    searcher = WebSearcher(current_app.config.get("EMPIRE_ID", ""))
    response = searcher.search_ai_papers(topic, max_results=max_results)
    return jsonify({
        "topic": topic,
        "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in response.results],
        "total": response.total_results,
    })


@api_bp.route("/search/store", methods=["POST"])
def api_search_and_store():
    """Search the web and store findings in knowledge graph + memory."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query required"}), 400
    from core.search.web import WebSearcher
    searcher = WebSearcher(empire_id)
    result = searcher.search_and_store(query, max_results=data.get("max_results", 5))
    return jsonify(result)


@api_bp.route("/scrape", methods=["POST"])
def api_scrape_url():
    """Scrape a URL and extract content."""
    data = request.get_json()
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "URL required"}), 400
    from core.search.scraper import WebScraper
    scraper = WebScraper(current_app.config.get("EMPIRE_ID", ""))
    page = scraper.scrape_url(url)
    return jsonify({
        "url": url, "success": page.success, "title": page.title,
        "content": page.content[:10000], "word_count": page.word_count,
        "author": page.author, "date": page.date, "domain": page.domain,
        "error": page.error,
    })


@api_bp.route("/scrape/store", methods=["POST"])
def api_scrape_and_store():
    """Scrape a URL and store in knowledge + memory."""
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json()
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "URL required"}), 400
    from core.search.scraper import WebScraper
    scraper = WebScraper(empire_id)
    return jsonify(scraper.scrape_and_store(url))


@api_bp.route("/research", methods=["POST"])
def api_research_topic():
    """Search, scrape, and synthesize research on a topic.

    Full pipeline: search → scrape top results → LLM synthesis → store.
    """
    empire_id = current_app.config.get("EMPIRE_ID", "")
    data = request.get_json()
    topic = data.get("topic", "")
    if not topic:
        return jsonify({"error": "Topic required"}), 400

    max_sources = data.get("max_sources", 3)

    from core.search.scraper import WebScraper
    from core.search.web import WebSearcher

    scraper = WebScraper(empire_id)
    searcher = WebSearcher(empire_id)

    # 1. Search for sources
    news = searcher.search_ai_news(topic, max_results=max_sources)
    urls = [r.url for r in news.results if r.url]

    # 2. Scrape top results
    scraped = []
    for url in urls[:max_sources]:
        page = scraper.scrape_url(url)
        if page.success:
            scraped.append(page)

    if not scraped:
        return jsonify({"topic": topic, "success": False, "error": "No sources could be scraped"})

    # 3. Build research context
    source_texts = []
    for page in scraped:
        source_texts.append(f"## {page.title}\nSource: {page.domain} | {page.date}\n\n{page.content[:3000]}")

    combined = "\n\n---\n\n".join(source_texts)

    # 4. LLM synthesis
    from llm.base import LLMRequest, LLMMessage
    from llm.router import ModelRouter, TaskMetadata

    router = ModelRouter()
    prompt = f"""Synthesize this research on: {topic}

Sources:
{combined}

Provide:
1. Key findings across all sources
2. What's new/significant
3. What this means for the AI landscape
4. Questions for follow-up research

Be specific and cite which source each finding comes from.
"""
    try:
        request_llm = LLMRequest(
            messages=[LLMMessage.user(prompt)],
            system_prompt="You are an AI research analyst. Synthesize sources accurately and identify what matters.",
            temperature=0.3,
            max_tokens=3000,
        )
        response = router.execute(request_llm, TaskMetadata(task_type="analysis", complexity="complex"))

        # 5. Store in memory
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        mm.store(
            content=f"Research: {topic}\n\n{response.content[:5000]}",
            memory_type="semantic",
            title=f"Research: {topic}",
            category="research",
            importance=0.75,
            tags=["research", "synthesis", topic.lower().replace(" ", "_")],
            source_type="research",
        )

        return jsonify({
            "topic": topic,
            "success": True,
            "synthesis": response.content,
            "sources": [{"title": p.title, "url": p.url, "domain": p.domain, "words": p.word_count} for p in scraped],
            "source_count": len(scraped),
            "cost_usd": response.cost_usd,
        })

    except Exception as e:
        return jsonify({"topic": topic, "success": False, "error": str(e)}), 500
