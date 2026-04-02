"""Shallow Enrichment Loop — finds low-detail entities and enriches them.

Scans the knowledge graph for entities with sparse attributes, short descriptions,
or missing schema fields, then runs targeted research to fill the gaps. This
turns shallow stubs into rich, well-attributed knowledge entries.

Enrichment targets:
  - Entities with < 50 char descriptions
  - Entities missing required schema fields
  - Entities with completeness score < 0.4
  - Entities with confidence < 0.5 that could be verified
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentTarget:
    """An entity that needs enrichment."""
    entity_id: str
    entity_name: str
    entity_type: str
    description: str = ""
    missing_fields: list[str] = field(default_factory=list)
    completeness: float = 0.0
    priority: float = 0.0
    reason: str = ""


@dataclass
class EnrichmentResult:
    """Result of an enrichment cycle."""
    entities_scanned: int = 0
    targets_found: int = 0
    enriched: int = 0
    fields_added: int = 0
    descriptions_improved: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ShallowEnricher:
    """Finds low-detail knowledge graph entities and enriches them via research.

    Runs as a scheduled job. Each cycle:
    1. Scans for entities with low completeness
    2. Prioritizes by importance (high-importance stubs first)
    3. Runs targeted web search for each entity
    4. Uses LLM to extract missing attributes
    5. Updates the entity in the knowledge graph
    """

    MIN_DESCRIPTION_LENGTH = 50     # Entities shorter than this need enrichment
    MIN_COMPLETENESS = 0.4          # Below this = needs enrichment
    MAX_ENRICH_PER_CYCLE = 10       # Don't enrich more than this per run
    COOLDOWN_HOURS = 24             # Don't re-enrich same entity within this window

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def find_targets(self, max_targets: int = 20) -> list[EnrichmentTarget]:
        """Scan the knowledge graph for entities needing enrichment.

        Args:
            max_targets: Maximum targets to return.

        Returns:
            List of EnrichmentTarget sorted by priority.
        """
        from db.engine import get_session
        from db.repositories.knowledge import KnowledgeRepository
        from core.knowledge.schemas import ENTITY_SCHEMAS

        session = get_session()
        try:
            repo = KnowledgeRepository(session)
            entities = repo.get_by_empire(self.empire_id, limit=2000)

            targets = []
            for entity in entities:
                # Skip recently enriched
                attrs = entity.attributes_json if hasattr(entity, "attributes_json") else {}
                if not isinstance(attrs, dict):
                    attrs = {}

                last_enriched = attrs.get("last_enriched")
                if last_enriched:
                    try:
                        enriched_at = datetime.fromisoformat(last_enriched)
                        if enriched_at > datetime.now(timezone.utc) - timedelta(hours=self.COOLDOWN_HOURS):
                            continue
                    except (ValueError, TypeError):
                        pass

                # Check description length
                desc_len = len(entity.description or "")
                short_desc = desc_len < self.MIN_DESCRIPTION_LENGTH

                # Check schema completeness
                missing_fields = []
                completeness = 1.0
                schema = ENTITY_SCHEMAS.get(entity.entity_type)
                if schema:
                    all_fields = schema.all_field_names
                    if all_fields:
                        filled = sum(1 for f in all_fields if attrs.get(f) is not None)
                        completeness = filled / len(all_fields)
                        if completeness < self.MIN_COMPLETENESS:
                            missing_fields = [f for f in all_fields if attrs.get(f) is None]

                # Determine if this entity needs enrichment
                needs_enrichment = short_desc or completeness < self.MIN_COMPLETENESS

                if not needs_enrichment:
                    continue

                # Calculate priority: high-importance stubs get enriched first
                importance = entity.importance_score or 0.5
                access_count = entity.access_count or 0
                priority = (
                    importance * 0.4 +
                    (1.0 - completeness) * 0.3 +
                    min(1.0, access_count / 10.0) * 0.2 +
                    (0.1 if short_desc else 0.0)
                )

                reason_parts = []
                if short_desc:
                    reason_parts.append(f"short description ({desc_len} chars)")
                if completeness < self.MIN_COMPLETENESS:
                    reason_parts.append(f"low completeness ({completeness:.0%})")
                if missing_fields:
                    reason_parts.append(f"{len(missing_fields)} missing fields")

                targets.append(EnrichmentTarget(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.entity_type or "concept",
                    description=entity.description or "",
                    missing_fields=missing_fields[:10],
                    completeness=completeness,
                    priority=priority,
                    reason="; ".join(reason_parts),
                ))

            targets.sort(key=lambda t: t.priority, reverse=True)
            return targets[:max_targets]

        finally:
            session.close()

    def enrich_entity(self, target: EnrichmentTarget) -> dict:
        """Enrich a single entity with web research + LLM extraction.

        Args:
            target: The entity to enrich.

        Returns:
            Dict with enrichment results.
        """
        from core.search.web import WebSearcher
        from core.knowledge.graph import KnowledgeGraph
        from core.knowledge.schemas import ENTITY_SCHEMAS

        searcher = WebSearcher(self.empire_id)
        graph = KnowledgeGraph(self.empire_id)

        result = {"entity": target.entity_name, "fields_added": 0, "description_improved": False}

        # Step 1: Search for this entity
        query = self._build_search_query(target)
        search_data = searcher.search_and_summarize(query, max_results=5)

        if not search_data.get("found"):
            logger.debug("No search results for enrichment of '%s'", target.entity_name)
            return result

        summary = search_data.get("summary", "")
        if not summary:
            return result

        # Step 2: Use LLM to extract missing attributes
        try:
            from llm.router import ModelRouter, TaskMetadata
            from llm.base import LLMRequest, LLMMessage
            router = ModelRouter()

            schema = ENTITY_SCHEMAS.get(target.entity_type)
            field_descriptions = ""
            if schema and target.missing_fields:
                field_descriptions = "\n".join(
                    f"- {f.name}: {f.description} (type: {f.field_type})"
                    for f in schema.fields
                    if f.name in target.missing_fields
                )

            prompt = (
                f"You are enriching a knowledge graph entity.\n\n"
                f"Entity: {target.entity_name}\n"
                f"Type: {target.entity_type}\n"
                f"Current description: {target.description[:200]}\n\n"
                f"Research findings:\n{summary[:4000]}\n\n"
                f"Tasks:\n"
                f"1. Write a comprehensive, factual description (2-3 sentences) for this entity.\n"
            )

            if field_descriptions:
                prompt += (
                    f"2. Fill in these missing attributes based on the research:\n"
                    f"{field_descriptions}\n\n"
                )

            prompt += (
                "Respond in this exact JSON format:\n"
                '{"description": "improved description", "attributes": {"field_name": "value", ...}}\n'
                "Only include attributes you can confidently fill from the research. "
                "Do not hallucinate or guess."
            )

            response = router.execute(
                LLMRequest(
                    messages=[LLMMessage.user(prompt)],
                    max_tokens=500,
                    temperature=0.2,
                ),
                TaskMetadata(task_type="extraction", complexity="simple"),
            )

            result["cost_usd"] = response.cost_usd

            # Parse response
            from llm.schemas import safe_json_loads
            enrichment = safe_json_loads(response.content)
            if not enrichment:
                return result

            # Step 3: Apply enrichment to knowledge graph
            new_desc = enrichment.get("description", "")
            new_attrs = enrichment.get("attributes", {})

            # Only update description if it's better (longer and meaningful)
            if new_desc and len(new_desc) > len(target.description) + 20:
                graph.update_entity(
                    name=target.entity_name,
                    description=new_desc,
                )
                result["description_improved"] = True

            # Add new attributes
            if new_attrs and isinstance(new_attrs, dict):
                # Merge with existing attributes
                new_attrs["last_enriched"] = datetime.now(timezone.utc).isoformat()
                new_attrs["enrichment_source"] = "shallow_enricher"
                graph.update_entity_attributes(
                    name=target.entity_name,
                    attributes=new_attrs,
                )
                result["fields_added"] = len([v for v in new_attrs.values() if v])

            # Boost confidence slightly for enriched entities
            graph.boost_confidence(target.entity_name, amount=0.05)

        except Exception as e:
            logger.warning("LLM enrichment failed for '%s': %s", target.entity_name, e)
            result["error"] = str(e)

        return result

    def run_enrichment_cycle(self, max_entities: int = 0) -> EnrichmentResult:
        """Full enrichment cycle: find targets and enrich them.

        Args:
            max_entities: Override for max entities per cycle (0 = use default).

        Returns:
            EnrichmentResult with stats.
        """
        start = time.time()
        limit = max_entities or self.MAX_ENRICH_PER_CYCLE
        result = EnrichmentResult()

        targets = self.find_targets(max_targets=limit * 2)
        result.entities_scanned = len(targets)
        result.targets_found = len(targets)

        if not targets:
            logger.info("No enrichment targets found — knowledge graph is well-populated")
            return result

        for target in targets[:limit]:
            try:
                enrichment = self.enrich_entity(target)
                if enrichment.get("description_improved") or enrichment.get("fields_added", 0) > 0:
                    result.enriched += 1
                if enrichment.get("description_improved"):
                    result.descriptions_improved += 1
                result.fields_added += enrichment.get("fields_added", 0)
                result.cost_usd += enrichment.get("cost_usd", 0)
            except Exception as e:
                result.errors.append(f"{target.entity_name}: {e}")
                logger.warning("Enrichment failed for '%s': %s", target.entity_name, e)

        result.duration_seconds = time.time() - start

        logger.info(
            "Enrichment cycle: %d scanned, %d enriched, %d descriptions improved, %d fields added (%.1fs, $%.4f)",
            result.entities_scanned, result.enriched, result.descriptions_improved,
            result.fields_added, result.duration_seconds, result.cost_usd,
        )
        return result

    def _build_search_query(self, target: EnrichmentTarget) -> str:
        """Build an optimal search query for enriching an entity."""
        name = target.entity_name
        etype = target.entity_type

        # Type-specific query templates
        type_queries = {
            "ai_model": f"{name} AI model specifications capabilities release",
            "company": f"{name} company AI products funding",
            "research_lab": f"{name} research lab publications focus areas",
            "paper": f"{name} research paper findings methodology",
            "technique": f"{name} technique how it works applications",
            "framework": f"{name} framework features documentation",
            "benchmark": f"{name} benchmark results leaderboard",
            "person": f"{name} AI researcher contributions",
            "product": f"{name} product features pricing",
            "dataset": f"{name} dataset size contents use cases",
        }

        return type_queries.get(etype, f"{name} {etype} details information 2026")
