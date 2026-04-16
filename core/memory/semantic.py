"""Semantic memory (Tier 1) — stores factual knowledge, domain concepts, definitions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    content: str
    source: str = ""
    confidence: float = 0.8
    tags: list[str] = field(default_factory=list)


@dataclass
class Concept:
    name: str
    definition: str
    relations: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    domain: str = ""


@dataclass
class Rule:
    rule: str
    conditions: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)


@dataclass
class Contradiction:
    fact_a: str
    fact_b: str
    explanation: str = ""
    resolution: str = ""


@dataclass
class KnowledgeMap:
    domain: str = ""
    facts: list[dict] = field(default_factory=list)
    concepts: list[dict] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)
    total_entries: int = 0


class SemanticMemory:
    """Semantic/factual memory — stores domain knowledge, concepts, rules.

    Facts gain or lose confidence based on access patterns and
    validation against new information.
    """

    def __init__(self, memory_manager: MemoryManager, lieutenant_id: str = ""):
        self.mm = memory_manager
        self.lieutenant_id = lieutenant_id

    def store_fact(
        self,
        fact: str,
        source: str = "",
        confidence: float = 0.8,
        tags: list[str] | None = None,
    ) -> dict:
        """Store a factual claim."""
        return self.mm.store(
            content=fact,
            memory_type="semantic",
            lieutenant_id=self.lieutenant_id,
            title=f"Fact: {fact[:80]}",
            category="fact",
            importance=0.6,
            confidence=confidence,
            tags=(tags or []) + ["fact"],
            metadata={"source": source},
            source_type="extraction",
        )

    def store_concept(self, concept: Concept) -> dict:
        """Store a domain concept."""
        content = f"Concept: {concept.name}\nDefinition: {concept.definition}"
        if concept.relations:
            content += f"\nRelations: {', '.join(concept.relations)}"
        if concept.examples:
            content += f"\nExamples: {', '.join(concept.examples)}"

        return self.mm.store(
            content=content,
            memory_type="semantic",
            lieutenant_id=self.lieutenant_id,
            title=f"Concept: {concept.name}",
            category="concept",
            importance=0.7,
            tags=["concept", concept.domain] if concept.domain else ["concept"],
            metadata={"name": concept.name, "domain": concept.domain},
        )

    def store_rule(self, rule: Rule) -> dict:
        """Store a domain rule."""
        content = f"Rule: {rule.rule}"
        if rule.conditions:
            content += f"\nConditions: {'; '.join(rule.conditions)}"
        if rule.exceptions:
            content += f"\nExceptions: {'; '.join(rule.exceptions)}"

        return self.mm.store(
            content=content,
            memory_type="semantic",
            lieutenant_id=self.lieutenant_id,
            title=f"Rule: {rule.rule[:80]}",
            category="rule",
            importance=0.75,
            tags=["rule"],
        )

    def query_facts(self, topic: str, min_confidence: float = 0.5, limit: int = 10) -> list[dict]:
        """Query facts about a topic."""
        results = self.mm.recall(
            query=topic,
            memory_types=["semantic"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )
        return [r for r in results if r.get("importance", 0) >= min_confidence * 0.5]

    def query_concepts(self, domain: str = "", limit: int = 20) -> list[dict]:
        """Query stored concepts, optionally filtered by domain."""
        return self.mm.recall(
            query=domain or "concept",
            memory_types=["semantic"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def find_contradictions(self, new_fact: str) -> list[Contradiction]:
        """Check if a new fact contradicts existing knowledge."""
        existing = self.query_facts(new_fact, limit=5)
        contradictions = []
        # Simple heuristic: check for negation keywords
        negation_words = {"not", "never", "no", "none", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't"}
        new_words = set(new_fact.lower().split())

        for existing_mem in existing:
            existing_words = set(existing_mem.get("content", "").lower().split())
            # If one has negation and the other doesn't, might be contradictory
            new_has_neg = bool(new_words & negation_words)
            existing_has_neg = bool(existing_words & negation_words)
            # Check for significant word overlap with different polarity
            overlap = new_words & existing_words - negation_words
            if len(overlap) > 3 and new_has_neg != existing_has_neg:
                contradictions.append(Contradiction(
                    fact_a=new_fact,
                    fact_b=existing_mem.get("content", ""),
                    explanation="Potential contradiction detected (opposite polarity with shared concepts)",
                ))

        return contradictions

    def merge_facts(self, fact_ids: list[str]) -> dict | None:
        """Merge multiple similar facts into one consolidated fact."""
        facts = []
        for fid in fact_ids:
            results = self.mm.recall(query=fid, limit=1)
            if results:
                facts.append(results[0])

        if len(facts) < 2:
            return None

        # Create merged content
        merged_content = "Consolidated fact:\n" + "\n".join(
            f"- {f.get('content', '')[:200]}" for f in facts
        )

        return self.mm.store(
            content=merged_content,
            memory_type="semantic",
            lieutenant_id=self.lieutenant_id,
            title="Merged fact",
            category="fact",
            importance=max(f.get("importance", 0.5) for f in facts),
            tags=["fact", "merged"],
            metadata={"source_ids": fact_ids},
        )

    def get_knowledge_map(self, domain: str = "") -> KnowledgeMap:
        """Get a map of all knowledge for a domain."""
        facts = self.mm.recall(
            query=f"fact {domain}".strip(),
            memory_types=["semantic"],
            lieutenant_id=self.lieutenant_id,
            limit=50,
        )
        concepts = self.query_concepts(domain, limit=30)

        return KnowledgeMap(
            domain=domain,
            facts=[f for f in facts if "fact" in (f.get("tags") or [])],
            concepts=[c for c in concepts if "concept" in (c.get("tags") or [])],
            total_entries=len(facts) + len(concepts),
        )
