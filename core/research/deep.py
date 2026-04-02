"""Deep research orchestration — pipeline + lieutenant perspectives + synthesis.

Extracted from god_panel.py so it can be called from any coordination layer
(God Panel, scheduler, API, CLI) without importing web routes.
"""

from __future__ import annotations

import json
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
    """Execute deep research with lieutenant perspectives.

    Flow:
    1. Run research pipeline (search -> scrape -> extract -> synthesize)
    2. Get lieutenant perspectives on the findings
    3. Synthesize everything into a final brief
    4. Store compounded knowledge
    """
    result = {"status": "completed", "research_cost": 0.0}

    # 1. Research pipeline
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
        logger.warning("Pipeline failed in deep research: %s", e)
        from core.search.web import WebSearcher
        searcher = WebSearcher(empire_id)
        search_result = searcher.research_topic(topic, depth="deep")
        raw_synthesis = search_result.get("synthesis", "")
        result["pipeline"] = {"stages": 0, "fallback": True}
        result["research_cost"] += search_result.get("cost_usd", 0)

    if not raw_synthesis:
        result["synthesis"] = "Research produced no synthesis."
        return result

    # 2. Lieutenant perspectives
    lieutenant_insights = []
    if lieutenant_domains:
        try:
            from llm.router import ModelRouter, TaskMetadata
            from llm.base import LLMRequest, LLMMessage

            router = ModelRouter(empire_id)

            for domain in lieutenant_domains[:3]:
                lt_name, lt_focus = DOMAIN_ROLES.get(domain, (domain.title(), domain))

                lt_prompt = (
                    f"You are {lt_name}, Empire's specialist in {lt_focus}.\n\n"
                    f"Research findings on '{topic}':\n{raw_synthesis[:3000]}\n\n"
                    f"From your domain perspective ({domain}), provide:\n"
                    f"1. What stands out as most significant?\n"
                    f"2. What's missing or needs deeper investigation?\n"
                    f"3. How does this connect to your domain?\n\n"
                    f"Be concise (3-5 sentences max)."
                )

                try:
                    lt_response = router.execute(
                        LLMRequest(messages=[LLMMessage.user(lt_prompt)], max_tokens=300, temperature=0.3),
                        TaskMetadata(task_type="analysis", complexity="moderate"),
                    )
                    lieutenant_insights.append({
                        "lieutenant": lt_name,
                        "domain": domain,
                        "perspective": lt_response.content,
                    })
                    result["research_cost"] += lt_response.cost_usd
                except Exception as e:
                    logger.debug("Lieutenant %s perspective failed: %s", domain, e)

        except Exception as e:
            logger.debug("Lieutenant perspectives failed: %s", e)

    result["lieutenant_perspectives"] = lieutenant_insights

    # 3. Final synthesis with all inputs
    try:
        from llm.router import ModelRouter, TaskMetadata
        from llm.base import LLMRequest, LLMMessage

        router = ModelRouter(empire_id)

        synthesis_parts = [f"## Research Findings\n{raw_synthesis[:4000]}"]

        if prior_knowledge and build_on_existing:
            synthesis_parts.append(f"\n## Empire's Prior Knowledge\n{prior_knowledge[:1500]}")

        if lieutenant_insights:
            lt_section = "\n## Lieutenant Perspectives\n"
            for lt in lieutenant_insights:
                lt_section += f"\n**{lt['lieutenant']}** ({lt['domain']}):\n{lt['perspective']}\n"
            synthesis_parts.append(lt_section)

        combined = "\n".join(synthesis_parts)

        final_prompt = (
            f"You are Empire's Chief of Staff. Synthesize all inputs about '{topic}' "
            f"into a final intelligence brief.\n\n"
            f"Structure:\n"
            f"1. **Executive Summary** (2-3 sentences)\n"
            f"2. **Key Findings** (bullet points)\n"
            f"3. **Lieutenant Insights** (what the specialists flagged)\n"
            f"4. **Knowledge Gaps** (what to investigate next)\n"
            f"5. **Strategic Implications** (what this means for AI)\n\n"
            f"Inputs:\n{combined[:8000]}"
        )

        final_response = router.execute(
            LLMRequest(messages=[LLMMessage.user(final_prompt)], max_tokens=1500, temperature=0.3),
            TaskMetadata(task_type="synthesis", complexity="complex"),
        )

        result["synthesis"] = final_response.content
        result["research_cost"] += final_response.cost_usd

        # 4. Store the compounded knowledge
        try:
            from core.memory.bitemporal import BiTemporalMemory
            BiTemporalMemory(empire_id).store_smart(
                content=f"God Panel Research: {topic}\n\n{final_response.content}",
                title=f"Research: {topic[:60]}",
                category="god_panel_research",
                importance=0.85,
                tags=["god_panel", "research", "synthesis"],
            )
        except Exception as e:
            logger.debug("Failed to store research memory: %s", e)

    except Exception as e:
        logger.warning("Final synthesis failed: %s", e)
        result["synthesis"] = raw_synthesis[:2000]

    return result


def execute_autonomous_gap_research(empire_id: str, priority: int = 5) -> dict:
    """Empire finds its own knowledge gaps and researches to fill them.

    Flow:
    1. Scan KG for domains with least coverage
    2. Generate research topics for the weakest areas
    3. Run deep research on each topic
    """
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
            logger.info("Autoresearch stopping at round %d — budget low ($%.2f remaining)", round_num, check.remaining_daily)
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
                    f"Already researched topics this session: {[t['topic'] for t in topics_researched]}\n\n"
                    f"Generate ONE specific, timely research topic that would fill the biggest gap. "
                    f"Focus on developments from early 2025. Do NOT repeat already-researched topics.\n\n"
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

                logger.info("Autoresearch round %d: %s (domain=%s, entities=%d)", round_num, topic[:60], domain, count)
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
                    "success": research_result.get("status") == "completed",
                })

            except Exception as e:
                logger.warning("Autoresearch failed for %s: %s", domain, e)
                topics_researched.append({
                    "round": round_num,
                    "domain": domain,
                    "topic": f"Failed: {e}",
                    "success": False,
                })

    return {
        "status": "completed",
        "rounds_completed": min(round_num, rounds) if topics_researched else 0,
        "topics_researched": topics_researched,
        "total_cost": total_cost,
        "domains_covered": list(set(t["domain"] for t in topics_researched)),
    }
