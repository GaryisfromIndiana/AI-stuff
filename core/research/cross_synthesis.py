"""Cross-Lieutenant Synthesis — finds overlapping knowledge across domains and generates insights.

Lieutenants research independently. This module detects where their domains
overlap (e.g., "MCP" appears in both agents + tooling research) and generates
cross-cutting insights that no single lieutenant would produce alone.

Runs as a scheduled job and can be triggered from the God Panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Minimum overlap to consider two domains connected on a topic
MIN_SHARED_ENTITIES = 2
# Maximum cross-synthesis reports per cycle
MAX_SYNTHESES_PER_CYCLE = 3


@dataclass
class DomainOverlap:
    """An overlap detected between lieutenant domains."""
    topic: str
    domains: list[str] = field(default_factory=list)
    shared_entities: list[dict] = field(default_factory=list)
    domain_specific: dict[str, list[dict]] = field(default_factory=dict)
    overlap_score: float = 0.0


@dataclass
class CrossSynthesisResult:
    """Result of a cross-lieutenant synthesis."""
    topic: str
    domains: list[str] = field(default_factory=list)
    synthesis: str = ""
    insights: list[str] = field(default_factory=list)
    connections_found: int = 0
    entities_involved: int = 0
    cost_usd: float = 0.0


@dataclass
class CrossSynthesisCycleResult:
    """Result of a full cross-synthesis cycle."""
    overlaps_detected: int = 0
    syntheses_produced: int = 0
    total_insights: int = 0
    total_cost_usd: float = 0.0
    results: list[CrossSynthesisResult] = field(default_factory=list)


class CrossLieutenantSynthesizer:
    """Detects cross-domain overlaps and generates unified insights.

    Flow:
    1. Scan KG entities grouped by tags/source to identify domain associations
    2. Find entities that appear across multiple domain contexts
    3. For each significant overlap, gather domain-specific perspectives
    4. LLM synthesis: what do these overlapping findings mean together?
    5. Store the cross-cutting insight as high-importance memory
    """

    # Lieutenant domains and their associated tag/topic patterns
    DOMAIN_KEYWORDS = {
        "models": ["llm", "model", "benchmark", "gpt", "claude", "gemini", "llama", "pricing", "capabilities"],
        "research": ["paper", "arxiv", "training", "alignment", "scaling", "architecture", "attention"],
        "agents": ["agent", "multi-agent", "tool_use", "mcp", "framework", "orchestration", "autonomous"],
        "tooling": ["api", "inference", "vllm", "deployment", "vector", "mlops", "infrastructure"],
        "industry": ["funding", "enterprise", "strategy", "acquisition", "startup", "market", "adoption"],
        "open_source": ["open_source", "huggingface", "local", "open_weight", "community", "fine_tuning"],
    }

    DOMAIN_NAMES = {
        "models": "Model Intelligence",
        "research": "Research Scout",
        "agents": "Agent Systems",
        "tooling": "Tooling & Infra",
        "industry": "Industry & Strategy",
        "open_source": "Open Source",
    }

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def detect_overlaps(self, min_shared: int = MIN_SHARED_ENTITIES) -> list[DomainOverlap]:
        """Scan the knowledge graph for cross-domain entity overlaps.

        Groups entities by which lieutenant domains they're relevant to
        (based on tags, entity type, description keywords), then finds
        topics where 2+ domains share entities.
        """
        try:
            from core.knowledge.graph import KnowledgeGraph

            graph = KnowledgeGraph(self.empire_id)
            all_entities = graph.find_entities(limit=500)

            if not all_entities:
                return []

            # Map each entity to the domains it's relevant to
            entity_domains: dict[str, list[str]] = {}  # entity_name -> [domains]
            entity_info: dict[str, dict] = {}  # entity_name -> entity data

            for entity in all_entities:
                name = entity.name.lower()
                desc = (entity.description or "").lower()
                tags = []
                # Tags might be on the entity object or in attributes
                if hasattr(entity, "tags") and entity.tags:
                    tags = [t.lower() for t in entity.tags]

                # Determine which domains this entity belongs to
                domains = set()
                combined_text = f"{name} {desc} {' '.join(tags)}"

                for domain, keywords in self.DOMAIN_KEYWORDS.items():
                    for kw in keywords:
                        if kw in combined_text:
                            domains.add(domain)
                            break

                if len(domains) >= 2:
                    entity_domains[entity.name] = list(domains)
                    entity_info[entity.name] = {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "description": (entity.description or "")[:200],
                        "confidence": entity.confidence,
                        "domains": list(domains),
                    }

            if not entity_domains:
                logger.info("No cross-domain entities found")
                return []

            # Cluster overlapping entities by domain pairs
            domain_pair_entities: dict[tuple, list[str]] = {}
            for entity_name, domains in entity_domains.items():
                domains_sorted = sorted(domains)
                for i in range(len(domains_sorted)):
                    for j in range(i + 1, len(domains_sorted)):
                        pair = (domains_sorted[i], domains_sorted[j])
                        if pair not in domain_pair_entities:
                            domain_pair_entities[pair] = []
                        domain_pair_entities[pair].append(entity_name)

            # Build overlaps for pairs with enough shared entities
            overlaps = []
            for (d1, d2), entities in domain_pair_entities.items():
                if len(entities) < min_shared:
                    continue

                # Build a topic label from the most common entity types/names
                shared = [entity_info[e] for e in entities if e in entity_info]
                if not shared:
                    continue

                # Use the most confident entity's description as topic seed
                shared.sort(key=lambda x: x["confidence"], reverse=True)
                topic_entities = [e["name"] for e in shared[:5]]
                topic = f"{', '.join(topic_entities[:3])}"

                overlap = DomainOverlap(
                    topic=topic,
                    domains=[d1, d2],
                    shared_entities=shared,
                    overlap_score=len(entities) / max(len(all_entities), 1) * 10 + len(entities) * 0.5,
                )
                overlaps.append(overlap)

            # Sort by overlap score descending
            overlaps.sort(key=lambda x: x.overlap_score, reverse=True)
            logger.info("Detected %d cross-domain overlaps", len(overlaps))
            return overlaps

        except Exception as e:
            logger.error("Failed to detect overlaps: %s", e)
            return []

    def synthesize_overlap(self, overlap: DomainOverlap) -> CrossSynthesisResult:
        """Generate a cross-cutting synthesis for a detected overlap."""
        result = CrossSynthesisResult(
            topic=overlap.topic,
            domains=overlap.domains,
            entities_involved=len(overlap.shared_entities),
        )

        try:
            from llm.router import ModelRouter, TaskMetadata
            from llm.base import LLMRequest, LLMMessage

            router = ModelRouter(self.empire_id)

            # Build context from shared entities
            entity_descriptions = []
            for e in overlap.shared_entities[:10]:
                entity_descriptions.append(
                    f"- **{e['name']}** ({e['type']}): {e['description']}"
                )

            domain_names = [self.DOMAIN_NAMES.get(d, d) for d in overlap.domains]

            prompt = (
                f"You are Empire's Chief of Staff performing cross-domain analysis.\n\n"
                f"Two specialist teams have independently discovered overlapping knowledge:\n"
                f"- **{domain_names[0]}** (focuses on {self.DOMAIN_KEYWORDS.get(overlap.domains[0], ['general'])[0]})\n"
                f"- **{domain_names[1]}** (focuses on {self.DOMAIN_KEYWORDS.get(overlap.domains[1], ['general'])[0]})\n\n"
                f"Shared entities found across both domains:\n"
                f"{'chr(10)'.join(entity_descriptions)}\n\n"
                f"Provide:\n"
                f"1. **Cross-Domain Insight**: What does this overlap reveal that neither "
                f"domain would see alone? (2-3 sentences)\n"
                f"2. **Connections**: How do these domains interact on this topic? "
                f"What causal links or dependencies exist?\n"
                f"3. **Blind Spots**: What might both domains be missing by looking "
                f"at this from only their perspective?\n"
                f"4. **Action Items**: What should Empire research next to deepen "
                f"understanding of this cross-domain topic?\n\n"
                f"Be specific and analytical. Reference the actual entities."
            )

            response = router.execute(
                LLMRequest(
                    messages=[LLMMessage.user(prompt)],
                    max_tokens=800,
                    temperature=0.3,
                ),
                TaskMetadata(task_type="synthesis", complexity="complex"),
            )

            result.synthesis = response.content
            result.cost_usd = response.cost_usd

            # Extract action items as insights
            lines = response.content.split("\n")
            in_actions = False
            for line in lines:
                stripped = line.strip()
                if "action item" in stripped.lower() or "research next" in stripped.lower():
                    in_actions = True
                    continue
                if in_actions and stripped.startswith(("-", "*", "•")):
                    result.insights.append(stripped.lstrip("-*• "))
                elif in_actions and stripped and not stripped.startswith("#"):
                    result.insights.append(stripped)

            result.connections_found = len(overlap.shared_entities)

            # Store as high-importance cross-domain memory
            try:
                from core.memory.manager import MemoryManager
                mm = MemoryManager(self.empire_id)
                mm.store(
                    content=(
                        f"Cross-Domain Synthesis: {overlap.topic}\n"
                        f"Domains: {', '.join(domain_names)}\n"
                        f"Entities: {', '.join(e['name'] for e in overlap.shared_entities[:5])}\n\n"
                        f"{response.content}"
                    ),
                    memory_type="semantic",
                    title=f"Cross-Synthesis: {overlap.topic[:60]}",
                    category="cross_lieutenant_synthesis",
                    importance=0.9,
                    tags=["cross_synthesis"] + overlap.domains,
                    source_type="cross_synthesis",
                    metadata={
                        "domains": overlap.domains,
                        "entities_involved": len(overlap.shared_entities),
                        "overlap_score": overlap.overlap_score,
                    },
                )
            except Exception as e:
                logger.debug("Failed to store cross-synthesis memory: %s", e)

            # Create relations between shared entities if they don't exist
            try:
                from core.knowledge.graph import KnowledgeGraph
                graph = KnowledgeGraph(self.empire_id)
                entities = overlap.shared_entities
                for i in range(min(len(entities), 5)):
                    for j in range(i + 1, min(len(entities), 5)):
                        graph.add_relation(
                            source_name=entities[i]["name"],
                            target_name=entities[j]["name"],
                            relation_type="cross_domain_connection",
                            confidence=0.7,
                        )
            except Exception as e:
                logger.debug("Failed to add cross-domain relations: %s", e)

        except Exception as e:
            logger.error("Cross-synthesis failed for '%s': %s", overlap.topic, e)

        return result

    def run_synthesis_cycle(
        self,
        max_syntheses: int = MAX_SYNTHESES_PER_CYCLE,
        min_shared: int = MIN_SHARED_ENTITIES,
    ) -> CrossSynthesisCycleResult:
        """Run a full cross-synthesis cycle.

        1. Detect overlaps across all domains
        2. Synthesize the top N overlaps
        3. Store insights and relations

        Args:
            max_syntheses: Maximum number of syntheses to produce.
            min_shared: Minimum shared entities to consider an overlap.

        Returns:
            CrossSynthesisCycleResult with all findings.
        """
        cycle_result = CrossSynthesisCycleResult()

        # Check if we ran recently (cooldown: 6 hours)
        try:
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)
            recent = mm.recall(
                query="cross_lieutenant_synthesis",
                memory_types=["semantic"],
                limit=1,
            )
            if recent:
                last = recent[0]
                created = getattr(last, "created_at", None)
                if not isinstance(last, dict):
                    created = getattr(last, "created_at", None)
                else:
                    created = last.get("created_at")

                if created and isinstance(created, datetime):
                    if datetime.now(timezone.utc) - created < timedelta(hours=6):
                        logger.info("Cross-synthesis ran recently, skipping (6h cooldown)")
                        return cycle_result
        except Exception:
            pass

        overlaps = self.detect_overlaps(min_shared=min_shared)
        cycle_result.overlaps_detected = len(overlaps)

        if not overlaps:
            logger.info("No cross-domain overlaps found — KG may need more entities")
            return cycle_result

        for overlap in overlaps[:max_syntheses]:
            logger.info(
                "Synthesizing overlap: %s (domains: %s, %d shared entities)",
                overlap.topic, overlap.domains, len(overlap.shared_entities),
            )

            synthesis = self.synthesize_overlap(overlap)
            cycle_result.results.append(synthesis)
            cycle_result.syntheses_produced += 1
            cycle_result.total_insights += len(synthesis.insights)
            cycle_result.total_cost_usd += synthesis.cost_usd

        logger.info(
            "Cross-synthesis cycle complete: %d overlaps, %d syntheses, %d insights, $%.4f",
            cycle_result.overlaps_detected,
            cycle_result.syntheses_produced,
            cycle_result.total_insights,
            cycle_result.total_cost_usd,
        )

        return cycle_result
