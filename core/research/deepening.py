"""Iterative Deepening Loop — shallow finds automatically trigger deeper research.

When the autonomous research job or intelligence sweep discovers high-signal topics,
this module detects them and queues deeper research passes. Each pass extracts more
entities, finds more relations, and builds out the knowledge graph around that topic.

Depth levels:
  0 (shallow) — news scan, snippet storage, basic entity extraction
  1 (medium)  — scrape top sources, detailed entity extraction, relation mapping
  2 (deep)    — multi-query research, cross-reference, synthesis, gap filling
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeepeningCandidate:
    """A topic that warrants deeper research."""
    topic: str
    entity_names: list[str] = field(default_factory=list)
    current_depth: int = 0
    signal_score: float = 0.0
    trigger_reason: str = ""


@dataclass
class DeepeningResult:
    """Result of a deepening pass."""
    topic: str
    depth: int = 0
    new_entities: int = 0
    new_relations: int = 0
    new_memories: int = 0
    queries_run: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class IterativeDeepener:
    """Detects high-signal shallow findings and triggers deeper research.

    Signal detection heuristics:
      - Entity cluster density: 3+ entities extracted from a single search = high signal
      - Novel entity ratio: >50% of extracted entities are new = unexplored territory
      - High-confidence extraction: avg confidence > 0.7 = reliable topic
      - Relation density: entities with 2+ new relations = interconnected topic
      - Recent recurrence: topic appears in 2+ recent searches = trending
    """

    # Thresholds for triggering deepening
    MIN_CLUSTER_SIZE = 3          # At least 3 entities from a search
    MIN_NOVEL_RATIO = 0.4         # 40% of entities must be new
    MIN_AVG_CONFIDENCE = 0.5      # Average confidence of extracted entities
    MIN_SIGNAL_SCORE = 0.6        # Composite signal score threshold
    MAX_DEPTH = 2                 # Maximum research depth
    COOLDOWN_HOURS = 12           # Don't re-deepen same topic within this window

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def detect_candidates(self, max_candidates: int = 5) -> list[DeepeningCandidate]:
        """Scan recent knowledge for topics that warrant deeper research.

        Looks at entities added in the last 24 hours and scores them for
        signal strength. Topics with high signal get flagged for deepening.

        Args:
            max_candidates: Maximum candidates to return.

        Returns:
            List of DeepeningCandidate sorted by signal score.
        """
        from db.engine import get_session
        from db.repositories.knowledge import KnowledgeRepository

        session = get_session()
        try:
            repo = KnowledgeRepository(session)
            cutoff = datetime.now(UTC) - timedelta(hours=24)

            # Get recently added entities
            all_entities = repo.get_by_empire(self.empire_id, limit=5000)
            recent = [e for e in all_entities if e.created_at and e.created_at >= cutoff]

            if len(recent) < self.MIN_CLUSTER_SIZE:
                return []

            # Group recent entities by related topics using type + keyword clustering
            clusters = self._cluster_entities(recent)

            candidates = []
            for topic, entities in clusters.items():
                if len(entities) < self.MIN_CLUSTER_SIZE:
                    continue

                # Check if already deepened recently
                if self._recently_deepened(topic, repo, all_entities):
                    continue

                # Calculate signal score
                signal = self._calculate_signal(entities, all_entities)
                if signal < self.MIN_SIGNAL_SCORE:
                    continue

                # Determine current depth from entity metadata
                current_depth = self._get_current_depth(entities)
                if current_depth >= self.MAX_DEPTH:
                    continue

                candidates.append(DeepeningCandidate(
                    topic=topic,
                    entity_names=[e.name for e in entities[:20]],
                    current_depth=current_depth,
                    signal_score=signal,
                    trigger_reason=self._explain_signal(entities, signal),
                ))

            # Sort by signal score, return top N
            candidates.sort(key=lambda c: c.signal_score, reverse=True)
            return candidates[:max_candidates]

        finally:
            session.close()

    def deepen(self, candidate: DeepeningCandidate) -> DeepeningResult:
        """Execute a deeper research pass on a topic.

        Args:
            candidate: The topic to deepen.

        Returns:
            DeepeningResult with counts of new knowledge added.
        """
        start = time.time()
        target_depth = candidate.current_depth + 1
        result = DeepeningResult(topic=candidate.topic, depth=target_depth)

        logger.info(
            "Deepening research: '%s' (depth %d → %d, signal=%.2f)",
            candidate.topic, candidate.current_depth, target_depth, candidate.signal_score,
        )

        if target_depth == 1:
            result = self._deepen_medium(candidate, result)
        elif target_depth >= 2:
            result = self._deepen_deep(candidate, result)

        result.duration_seconds = time.time() - start

        # Mark topic as deepened
        self._mark_deepened(candidate.topic, target_depth)

        logger.info(
            "Deepening complete: '%s' depth=%d, +%d entities, +%d relations (%.1fs)",
            candidate.topic, target_depth, result.new_entities,
            result.new_relations, result.duration_seconds,
        )
        return result

    def run_deepening_cycle(self, max_topics: int = 3) -> list[DeepeningResult]:
        """Full cycle: detect candidates and deepen the best ones.

        Args:
            max_topics: Maximum topics to deepen per cycle.

        Returns:
            List of DeepeningResults.
        """
        candidates = self.detect_candidates(max_candidates=max_topics)
        if not candidates:
            logger.info("No deepening candidates found — knowledge is well-explored")
            return []

        results = []
        for candidate in candidates:
            try:
                result = self.deepen(candidate)
                results.append(result)
            except Exception as e:
                logger.warning("Deepening failed for '%s': %s", candidate.topic, e)
                results.append(DeepeningResult(topic=candidate.topic))

        return results

    # ── Depth 1: Medium deepening ────────────────────────────────────

    def _deepen_medium(self, candidate: DeepeningCandidate, result: DeepeningResult) -> DeepeningResult:
        """Medium depth: scrape top sources, detailed extraction, relation mapping."""
        from core.knowledge.entities import EntityExtractor
        from core.knowledge.graph import KnowledgeGraph
        from core.memory.manager import MemoryManager
        from core.search.web import WebSearcher

        searcher = WebSearcher(self.empire_id)
        extractor = EntityExtractor()
        graph = KnowledgeGraph(self.empire_id)
        mm = MemoryManager(self.empire_id)

        # Run multiple search queries to broaden coverage
        queries = self._generate_medium_queries(candidate)
        for query in queries:
            try:
                search_data = searcher.search_and_summarize(query, max_results=5)
                result.queries_run += 1

                if not search_data.get("found"):
                    continue

                summary = search_data.get("summary", "")
                if not summary:
                    continue

                # Store enriched memory
                mm.store(
                    content=f"Deep research (depth 1): {candidate.topic}\nQuery: {query}\n\n{summary[:3000]}",
                    memory_type="semantic",
                    title=f"Deepened: {candidate.topic[:60]}",
                    category="iterative_deepening",
                    importance=0.7,
                    tags=["deepening", "depth_1", candidate.topic.replace(" ", "_")[:30]],
                    source_type="deepening",
                    metadata={"depth": 1, "query": query, "topic": candidate.topic},
                )
                result.new_memories += 1

                # Extract entities with more detail
                extraction = extractor.extract_from_text(
                    summary[:5000],
                    context=f"Deep research on: {candidate.topic}. Extract detailed entities with relations.",
                    max_entities=15,
                )

                if extraction.entities:
                    for entity in extraction.entities:
                        graph.add_entity(
                            name=entity.get("name", ""),
                            entity_type=entity.get("entity_type", "concept"),
                            description=entity.get("description", ""),
                            confidence=entity.get("confidence", 0.65),
                            tags=["deepening", "depth_1"],
                            attributes={"deepening_depth": 1, "deepening_topic": candidate.topic},
                        )
                    result.new_entities += len(extraction.entities)

                if extraction.relations:
                    for rel in extraction.relations:
                        graph.add_relation(
                            source_name=rel.get("source", ""),
                            target_name=rel.get("target", ""),
                            relation_type=rel.get("type", "related_to"),
                            confidence=rel.get("confidence", 0.6),
                        )
                    result.new_relations += len(extraction.relations)

            except Exception as e:
                logger.warning("Medium deepening query failed for '%s': %s", query, e)

        return result

    # ── Depth 2: Deep deepening ──────────────────────────────────────

    def _deepen_deep(self, candidate: DeepeningCandidate, result: DeepeningResult) -> DeepeningResult:
        """Deep: multi-query research, cross-reference, synthesis, gap filling."""
        from core.knowledge.entities import EntityExtractor
        from core.knowledge.graph import KnowledgeGraph
        from core.memory.manager import MemoryManager
        from core.search.web import WebSearcher

        searcher = WebSearcher(self.empire_id)
        extractor = EntityExtractor()
        graph = KnowledgeGraph(self.empire_id)
        mm = MemoryManager(self.empire_id)

        # Generate deep queries — more specific, cross-referencing
        queries = self._generate_deep_queries(candidate)

        all_summaries = []
        for query in queries:
            try:
                search_data = searcher.search_and_summarize(query, max_results=8)
                result.queries_run += 1

                if search_data.get("found"):
                    summary = search_data.get("summary", "")
                    all_summaries.append(summary)

                    # Extract with maximum detail
                    extraction = extractor.extract_from_text(
                        summary[:5000],
                        context=(
                            f"Deep research (depth 2) on: {candidate.topic}. "
                            f"Extract ALL entities with full detail including relations, "
                            f"temporal information, and cross-references."
                        ),
                        max_entities=20,
                    )

                    if extraction.entities:
                        for entity in extraction.entities:
                            graph.add_entity(
                                name=entity.get("name", ""),
                                entity_type=entity.get("entity_type", "concept"),
                                description=entity.get("description", ""),
                                confidence=entity.get("confidence", 0.7),
                                tags=["deepening", "depth_2"],
                                attributes={"deepening_depth": 2, "deepening_topic": candidate.topic},
                            )
                        result.new_entities += len(extraction.entities)

                    if extraction.relations:
                        for rel in extraction.relations:
                            graph.add_relation(
                                source_name=rel.get("source", ""),
                                target_name=rel.get("target", ""),
                                relation_type=rel.get("type", "related_to"),
                                confidence=rel.get("confidence", 0.65),
                            )
                        result.new_relations += len(extraction.relations)

            except Exception as e:
                logger.warning("Deep query failed for '%s': %s", query, e)

        # Synthesize all findings into a comprehensive memory
        if all_summaries:
            try:
                from llm.base import LLMMessage, LLMRequest
                from llm.router import ModelRouter, TaskMetadata
                router = ModelRouter()

                combined = "\n\n---\n\n".join(s[:2000] for s in all_summaries[:5])
                synth_prompt = (
                    f"Synthesize these research findings about '{candidate.topic}' into a "
                    f"comprehensive summary. Focus on key insights, relationships between "
                    f"entities, and implications.\n\n{combined[:8000]}"
                )

                response = router.execute(
                    LLMRequest(
                        messages=[LLMMessage.user(synth_prompt)],
                        max_tokens=1000,
                        temperature=0.3,
                    ),
                    TaskMetadata(task_type="synthesis", complexity="moderate"),
                )

                synthesis = response.content
                result.cost_usd = response.cost_usd

                mm.store(
                    content=f"Deep Synthesis (depth 2): {candidate.topic}\n\n{synthesis}",
                    memory_type="semantic",
                    title=f"Synthesis: {candidate.topic[:60]}",
                    category="iterative_deepening",
                    importance=0.85,
                    tags=["deepening", "depth_2", "synthesis"],
                    source_type="deepening",
                    metadata={"depth": 2, "topic": candidate.topic, "queries": len(queries)},
                )
                result.new_memories += 1

            except Exception as e:
                logger.warning("Synthesis failed for '%s': %s", candidate.topic, e)

        return result

    # ── Helper methods ───────────────────────────────────────────────

    def _cluster_entities(self, entities: list) -> dict[str, list]:
        """Cluster entities by topic using type + name keyword analysis."""
        clusters: dict[str, list] = {}

        # Group by entity type first
        for entity in entities:
            etype = entity.entity_type or "concept"
            if etype not in clusters:
                clusters[etype] = []
            clusters[etype].append(entity)

        # Also cluster by keyword extraction from names/descriptions
        keyword_map = {}
        for entity in entities:
            text = f"{entity.name} {entity.description or ''}".lower()
            # Extract significant words (3+ chars, not stopwords)
            words = [w for w in text.split() if len(w) >= 4 and w not in _STOP_WORDS]
            for word in words[:5]:  # Top 5 keywords per entity
                if word not in keyword_map:
                    keyword_map[word] = []
                keyword_map[word].append(entity)

        # Merge keyword clusters with 3+ entities into the main clusters
        for keyword, ents in keyword_map.items():
            if len(ents) >= 3:
                cluster_key = f"topic:{keyword}"
                clusters[cluster_key] = ents

        return clusters

    def _calculate_signal(self, cluster_entities: list, all_entities: list) -> float:
        """Calculate signal strength for an entity cluster."""
        if not cluster_entities:
            return 0.0

        # 1. Cluster density (more entities = higher signal)
        density = min(1.0, len(cluster_entities) / 10.0)

        # 2. Average confidence
        avg_conf = sum(e.confidence for e in cluster_entities) / len(cluster_entities)

        # 3. Novel ratio — what fraction are genuinely new vs updates to existing?
        all_names = {e.name.lower() for e in all_entities}
        recent_names = {e.name.lower() for e in cluster_entities}
        # Entities that were only seen in this batch = novel
        existing_before = all_names - recent_names
        novel_count = sum(1 for e in cluster_entities if e.name.lower() not in existing_before)
        novel_ratio = novel_count / len(cluster_entities) if cluster_entities else 0

        # 4. Description richness — do entities have substantial descriptions?
        desc_lengths = [len(e.description or "") for e in cluster_entities]
        avg_desc = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
        richness = min(1.0, avg_desc / 200.0)  # 200+ chars = rich

        # Composite signal
        signal = (
            density * 0.25 +
            avg_conf * 0.25 +
            novel_ratio * 0.30 +
            richness * 0.20
        )

        return round(signal, 3)

    def _explain_signal(self, entities: list, signal: float) -> str:
        """Generate a human-readable explanation of why this topic is high-signal."""
        parts = [f"{len(entities)} entities clustered"]
        avg_conf = sum(e.confidence for e in entities) / len(entities)
        if avg_conf > 0.7:
            parts.append(f"high confidence ({avg_conf:.2f})")
        desc_avg = sum(len(e.description or "") for e in entities) / len(entities)
        if desc_avg > 150:
            parts.append("rich descriptions")
        return f"Signal {signal:.2f}: {', '.join(parts)}"

    def _get_current_depth(self, entities: list) -> int:
        """Determine the current research depth for these entities."""
        max_depth = 0
        for entity in entities:
            attrs = entity.attributes_json if hasattr(entity, "attributes_json") else {}
            if isinstance(attrs, dict):
                depth = attrs.get("deepening_depth", 0)
                if isinstance(depth, (int, float)):
                    max_depth = max(max_depth, int(depth))
        return max_depth

    def _recently_deepened(self, topic: str, repo: Any, all_entities: list) -> bool:
        """Check if this topic was deepened within the cooldown window."""
        cutoff = datetime.now(UTC) - timedelta(hours=self.COOLDOWN_HOURS)
        topic_lower = topic.lower()

        for entity in all_entities:
            attrs = entity.attributes_json if hasattr(entity, "attributes_json") else {}
            if not isinstance(attrs, dict):
                continue
            if attrs.get("deepening_topic", "").lower() == topic_lower:
                if entity.updated_at and entity.updated_at >= cutoff:
                    return True
        return False

    def _mark_deepened(self, topic: str, depth: int) -> None:
        """Mark a topic as deepened in memory for cooldown tracking."""
        try:
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)
            mm.store(
                content=f"Deepened topic '{topic}' to depth {depth}",
                memory_type="episodic",
                title=f"Deepening: {topic[:60]} (d={depth})",
                category="iterative_deepening",
                importance=0.4,
                tags=["deepening_marker"],
                source_type="system",
                metadata={"topic": topic, "depth": depth},
            )
        except Exception as e:
            logger.warning("Failed to mark deepening: %s", e)

    def _generate_medium_queries(self, candidate: DeepeningCandidate) -> list[str]:
        """Generate search queries for medium-depth research."""
        topic = candidate.topic
        # Remove clustering prefix if present
        if topic.startswith("topic:"):
            topic = topic[6:]

        queries = [
            f"{topic} latest developments 2026",
            f"{topic} research breakthrough",
            f"{topic} key companies organizations",
        ]

        # Add entity-specific queries for the top entities
        for name in candidate.entity_names[:2]:
            queries.append(f"{name} {topic}")

        return queries[:5]

    def _generate_deep_queries(self, candidate: DeepeningCandidate) -> list[str]:
        """Generate search queries for deep research."""
        topic = candidate.topic
        if topic.startswith("topic:"):
            topic = topic[6:]

        queries = [
            f"{topic} comprehensive analysis 2026",
            f"{topic} technical architecture details",
            f"{topic} comparison alternatives",
            f"{topic} challenges limitations",
            f"{topic} future roadmap predictions",
            f"{topic} site:arxiv.org OR site:openreview.net",
        ]

        # Cross-reference top entities
        if len(candidate.entity_names) >= 2:
            queries.append(f"{candidate.entity_names[0]} vs {candidate.entity_names[1]}")

        return queries[:7]


# Common English stop words to filter from keyword clustering
_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "been", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "can", "not", "but", "also", "than",
    "then", "when", "what", "which", "where", "who", "how", "its",
    "it's", "they", "their", "them", "there", "here", "more", "most",
    "very", "just", "about", "into", "over", "such", "only", "other",
    "some", "each", "does", "did", "being", "using", "used", "based",
    "like", "well", "between", "through", "after", "before",
})
