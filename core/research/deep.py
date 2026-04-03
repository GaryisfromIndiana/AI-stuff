"""Deep research orchestration — the core compounding loop.

Flow:
1. Check prior knowledge (what does Empire already know?)
2. Research pipeline (search → scrape → synthesize)
3. Final synthesis (merge new findings with prior knowledge)
4. Store in bi-temporal memory (with supersession)
5. Extract entities → store in knowledge graph (MANDATORY)

Every research cycle should leave the system smarter than before.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Domain -> (lieutenant name, focus area)
DOMAIN_ROLES = {
    "models": ("Model Intelligence", "LLM releases, benchmarks, pricing, capabilities, architecture comparisons"),
    "research": ("Research Scout", "AI papers, training techniques, alignment research, scaling laws"),
    "agents": ("Agent Systems", "multi-agent architectures, tool use, frameworks, MCP, orchestration"),
    "tooling": ("Tooling & Infra", "APIs, inference engines, vector DBs, deployment, MLOps"),
    "industry": ("Industry & Strategy", "company strategy, funding rounds, enterprise AI adoption, market dynamics"),
    "open_source": ("Open Source", "open weight models, HuggingFace releases, local inference, community projects"),
}


def execute_deep_research(
    empire_id: str,
    topic: str,
    description: str,
    lieutenant_domains: list[str],
    prior_knowledge: str,
    build_on_existing: bool,
    priority: int,
) -> dict:
    """Execute deep research with compounding knowledge.

    This is the central research function. Every call should:
    - Check what Empire already knows (prior knowledge)
    - Research new information (web search + scrape)
    - Synthesize everything into a brief
    - Store the synthesis as a bi-temporal fact
    - Extract entities and store them in the knowledge graph

    The knowledge graph and memory get richer with every cycle.
    """
    result = {"status": "completed", "research_cost": 0.0}

    # ── 1. Check prior knowledge ──────────────────────────────────
    # What does Empire already know about this topic?
    # This makes research BUILD ON existing knowledge instead of starting fresh.
    if not prior_knowledge:
        prior_knowledge = _gather_prior_knowledge(empire_id, topic)
        if prior_knowledge:
            build_on_existing = True
            result["prior_knowledge_found"] = True
            logger.info("Found prior knowledge on '%s' — building on it", topic[:50])

    # ── 2. Research pipeline ──────────────────────────────────────
    try:
        from core.research.pipeline import ResearchPipeline
        pipeline = ResearchPipeline(empire_id)
        depth = "deep" if priority >= 7 else "standard"
        pipe_result = pipeline.run(topic, depth=depth)

        result["pipeline"] = {
            "stages": len(pipe_result.stages),
            "entities": pipe_result.total_entities,
            "relations": pipe_result.total_relations,
            "success": pipe_result.success,
        }
        result["research_cost"] += pipe_result.cost_usd
        raw_synthesis = pipe_result.synthesis or ""
    except Exception as e:
        logger.warning("Pipeline failed, falling back to basic search: %s", e)
        from core.search.web import WebSearcher
        searcher = WebSearcher(empire_id)
        search_result = searcher.research_topic(topic, depth="deep")
        raw_synthesis = search_result.get("synthesis", "")
        result["pipeline"] = {"stages": 0, "fallback": True}
        result["research_cost"] += search_result.get("cost_usd", 0)

    if not raw_synthesis:
        result["synthesis"] = "Research produced no synthesis."
        return result

    # ── 3. Final synthesis (merge new + prior) ────────────────────
    from llm.router import ModelRouter, TaskMetadata
    from llm.base import LLMRequest, LLMMessage

    router = ModelRouter(empire_id)

    synthesis_parts = [f"## New Research Findings\n{raw_synthesis[:4000]}"]
    if prior_knowledge and build_on_existing:
        synthesis_parts.append(f"\n## What Empire Already Knew\n{prior_knowledge[:1500]}")

    combined = "\n".join(synthesis_parts)

    try:
        final_prompt = (
            f"Synthesize all inputs about '{topic}' into an intelligence brief.\n\n"
            f"Structure:\n"
            f"1. **Executive Summary** (2-3 sentences)\n"
            f"2. **Key Findings** (bullet points — cite sources)\n"
            f"3. **What's New** (what changed vs prior knowledge, if any)\n"
            f"4. **Knowledge Gaps** (what to investigate next)\n\n"
            f"Inputs:\n{combined[:8000]}"
        )

        final_response = router.execute(
            LLMRequest(
                messages=[LLMMessage.user(final_prompt)],
                system_prompt="You are a research analyst. Be specific, cite sources, distinguish new from known.",
                max_tokens=1500, temperature=0.3,
            ),
            TaskMetadata(task_type="synthesis", complexity="complex"),
        )
        result["synthesis"] = final_response.content
        result["research_cost"] += final_response.cost_usd

    except Exception as e:
        logger.warning("Final synthesis failed: %s", e)
        result["synthesis"] = raw_synthesis[:2000]

    # ── 4. Store in bi-temporal memory ────────────────────────────
    synthesis_text = result.get("synthesis", "")
    try:
        from core.memory.bitemporal import BiTemporalMemory
        BiTemporalMemory(empire_id).store_smart(
            content=f"Research: {topic}\n\n{synthesis_text}",
            title=f"Research: {topic[:60]}",
            category="research",
            importance=0.85,
            source="research_pipeline",
            tags=["research", "synthesis"] + [d for d in lieutenant_domains[:3]],
        )
    except Exception as e:
        logger.debug("Failed to store research memory: %s", e)

    # ── 5. Extract entities → Knowledge Graph (MANDATORY) ─────────
    # This is what makes knowledge compound. Every synthesis should
    # produce entities that land in the KG for future research to build on.
    extraction_result = _extract_and_store_entities(empire_id, synthesis_text, topic, router)
    result["entities_extracted"] = extraction_result.get("extracted", 0)
    result["entities_stored"] = extraction_result.get("stored", 0)
    result["relations_stored"] = extraction_result.get("relations", 0)
    result["research_cost"] += extraction_result.get("cost_usd", 0)

    return result


def _gather_prior_knowledge(empire_id: str, topic: str) -> str:
    """Check what Empire already knows about a topic.

    Queries both memory and knowledge graph. Returns a formatted string
    for injection into the research prompt.
    """
    parts = []

    # Check memories
    try:
        from core.memory.manager import MemoryManager
        mm = MemoryManager(empire_id)
        memories = mm.recall(query=topic, memory_types=["semantic"], limit=5)
        if memories:
            mem_parts = []
            for m in memories:
                title = m.get("title", "")
                content = m.get("content", "")[:200]
                if content:
                    mem_parts.append(f"- {title}: {content}" if title else f"- {content}")
            if mem_parts:
                parts.append("Known facts:\n" + "\n".join(mem_parts[:5]))
    except Exception as e:
        logger.debug("Prior memory lookup failed: %s", e)

    # Check knowledge graph
    try:
        from core.knowledge.graph import KnowledgeGraph
        graph = KnowledgeGraph(empire_id)
        entities = graph.find_entities(query=topic, limit=5)
        if entities:
            ent_parts = [
                f"- {e.name} ({e.entity_type}): {e.description[:100]}"
                for e in entities if e.description
            ]
            if ent_parts:
                parts.append("Known entities:\n" + "\n".join(ent_parts[:5]))
    except Exception as e:
        logger.debug("Prior KG lookup failed: %s", e)

    return "\n\n".join(parts)


def _extract_and_store_entities(
    empire_id: str,
    synthesis: str,
    topic: str,
    router,
) -> dict:
    """Extract entities from synthesis and store them in the knowledge graph.

    This is MANDATORY — it's what makes the KG grow and knowledge compound.
    """
    if not synthesis or len(synthesis) < 50:
        return {"extracted": 0, "stored": 0, "relations": 0, "cost_usd": 0}

    try:
        from core.knowledge.entities import EntityExtractor
        from core.knowledge.graph import KnowledgeGraph

        extractor = EntityExtractor(router=router)
        extraction = extractor.extract_from_text(synthesis[:4000], context=f"Topic: {topic}")

        if not extraction.entities:
            return {"extracted": 0, "stored": 0, "relations": 0, "cost_usd": extraction.cost_usd}

        graph = KnowledgeGraph(empire_id)
        stored = 0
        for e in extraction.entities:
            try:
                graph.add_entity(
                    name=e.get("name", ""),
                    entity_type=e.get("entity_type", "concept"),
                    description=e.get("description", ""),
                    confidence=e.get("confidence", 0.7),
                )
                stored += 1
            except Exception:
                pass

        rel_stored = 0
        for r in extraction.relations:
            try:
                graph.add_relation(
                    source_name=r.get("source", ""),
                    target_name=r.get("target", ""),
                    relation_type=r.get("type", "related_to"),
                    confidence=r.get("confidence", 0.7),
                )
                rel_stored += 1
            except Exception:
                pass

        logger.info(
            "Extracted %d entities, %d relations from '%s' (stored %d/%d)",
            len(extraction.entities), len(extraction.relations),
            topic[:40], stored, len(extraction.entities),
        )

        return {
            "extracted": len(extraction.entities),
            "stored": stored,
            "relations": rel_stored,
            "cost_usd": extraction.cost_usd,
        }

    except Exception as e:
        logger.warning("Entity extraction failed for '%s': %s", topic[:40], e)
        return {"extracted": 0, "stored": 0, "relations": 0, "cost_usd": 0}


def execute_autonomous_gap_research(empire_id: str, priority: int = 5) -> dict:
    """Empire finds its own knowledge gaps and researches to fill them."""
    from core.knowledge.graph import KnowledgeGraph
    from core.routing.budget import BudgetManager

    rounds = min(priority, 5)
    total_cost = 0.0
    topics_researched = []

    DOMAINS = {
        "models": "Latest LLM releases, benchmarks, pricing, architecture comparisons",
        "research": "Recent AI papers, training techniques, alignment research, scaling laws",
        "agents": "Multi-agent frameworks, tool use patterns, MCP developments, orchestration",
        "tooling": "Inference engines, vector databases, deployment tools, MLOps platforms",
        "industry": "AI company strategy, funding rounds, enterprise adoption trends",
        "open_source": "Open weight model releases, HuggingFace trends, local inference",
    }

    round_num = 0
    for round_num in range(1, rounds + 1):
        bm = BudgetManager(empire_id)
        check = bm.check_budget(estimated_cost=0.10)
        if check.remaining_daily < 0.50:
            logger.info("Autoresearch stopping at round %d — budget low", round_num)
            break

        graph = KnowledgeGraph(empire_id)
        domain_counts = {}
        for domain in DOMAINS:
            try:
                entities = graph.find_entities(query=domain, limit=100)
                domain_counts[domain] = len(entities) if entities else 0
            except Exception:
                domain_counts[domain] = 0

        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1])
        weak = sorted_domains[:2]

        for domain, count in weak:
            try:
                from llm.router import ModelRouter, TaskMetadata
                from llm.base import LLMRequest, LLMMessage

                router = ModelRouter(empire_id)
                topic_prompt = (
                    f"You are an AI research director. The '{domain}' knowledge domain has {count} entries, "
                    f"which is {'very low' if count < 20 else 'moderate'}.\n\n"
                    f"Domain focus: {DOMAINS[domain]}\n\n"
                    f"Already researched: {[t['topic'] for t in topics_researched]}\n\n"
                    f"Generate ONE specific, timely research topic for early 2025. "
                    f"Do NOT repeat already-researched topics.\n\n"
                    f"Respond with ONLY the topic — one sentence, no explanation."
                )

                resp = router.execute(
                    LLMRequest(messages=[LLMMessage.user(topic_prompt)], max_tokens=100, temperature=0.7),
                    TaskMetadata(task_type="planning", complexity="simple"),
                )
                topic = resp.content.strip().strip('"')
                total_cost += resp.cost_usd

                if not topic:
                    continue

                logger.info("Autoresearch round %d: %s (domain=%s)", round_num, topic[:60], domain)
                research_result = execute_deep_research(
                    empire_id, topic, f"Autonomous gap research for {domain}: {topic}",
                    [domain], "", False, priority,
                )
                total_cost += research_result.get("research_cost", 0)

                topics_researched.append({
                    "round": round_num,
                    "domain": domain,
                    "topic": topic,
                    "entities_before": count,
                    "entities_extracted": research_result.get("entities_extracted", 0),
                    "success": research_result.get("status") == "completed",
                })

            except Exception as e:
                logger.warning("Autoresearch failed for %s: %s", domain, e)
                topics_researched.append({
                    "round": round_num, "domain": domain,
                    "topic": f"Failed: {e}", "success": False,
                })

    return {
        "status": "completed",
        "rounds_completed": min(round_num, rounds) if topics_researched else 0,
        "topics_researched": topics_researched,
        "total_cost": total_cost,
        "domains_covered": list(set(t["domain"] for t in topics_researched)),
    }
