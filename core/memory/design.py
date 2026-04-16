"""Design/pattern memory (Tier 3) — stores patterns, decisions, templates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    name: str
    description: str
    template: str = ""
    use_cases: list[str] = field(default_factory=list)
    effectiveness: float = 0.7
    domain: str = ""


@dataclass
class Decision:
    decision: str
    rationale: str
    alternatives: list[str] = field(default_factory=list)
    outcome: str = ""
    context: str = ""


@dataclass
class Template:
    name: str
    template: str
    variables: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    domain: str = ""


@dataclass
class AntiPattern:
    name: str
    description: str
    why_bad: str
    alternative: str


@dataclass
class PatternSuggestion:
    pattern: dict
    relevance: float = 0.7
    reasoning: str = ""


@dataclass
class DesignLibrary:
    domain: str = ""
    patterns: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    templates: list[dict] = field(default_factory=list)
    antipatterns: list[dict] = field(default_factory=list)
    total: int = 0


class DesignMemory:
    """Design/pattern memory — stores architectural patterns, design decisions, templates.

    Tracks how patterns evolve over time and shares effective patterns
    across lieutenants.
    """

    def __init__(self, memory_manager: MemoryManager, lieutenant_id: str = ""):
        self.mm = memory_manager
        self.lieutenant_id = lieutenant_id

    def store_pattern(self, pattern: Pattern) -> dict:
        """Store a design pattern."""
        content = f"Pattern: {pattern.name}\n{pattern.description}"
        if pattern.template:
            content += f"\n\nTemplate:\n{pattern.template}"
        if pattern.use_cases:
            content += f"\n\nUse cases: {', '.join(pattern.use_cases)}"

        return self.mm.store(
            content=content,
            memory_type="design",
            lieutenant_id=self.lieutenant_id,
            title=f"Pattern: {pattern.name}",
            category="pattern",
            importance=0.7,
            tags=["pattern", pattern.domain] if pattern.domain else ["pattern"],
            metadata={
                "name": pattern.name,
                "effectiveness": pattern.effectiveness,
                "use_cases": pattern.use_cases,
                "domain": pattern.domain,
            },
        )

    def store_decision(self, decision: Decision) -> dict:
        """Store a design decision with rationale."""
        content = f"Decision: {decision.decision}\nRationale: {decision.rationale}"
        if decision.alternatives:
            content += f"\nAlternatives considered: {', '.join(decision.alternatives)}"
        if decision.outcome:
            content += f"\nOutcome: {decision.outcome}"

        return self.mm.store(
            content=content,
            memory_type="design",
            lieutenant_id=self.lieutenant_id,
            title=f"Decision: {decision.decision[:80]}",
            category="decision",
            importance=0.65,
            tags=["decision"],
            metadata={
                "alternatives": decision.alternatives,
                "outcome": decision.outcome,
                "context": decision.context,
            },
        )

    def store_template(self, template: Template) -> dict:
        """Store a reusable template."""
        content = f"Template: {template.name}\n\n{template.template}"
        if template.variables:
            content += f"\n\nVariables: {', '.join(template.variables)}"

        return self.mm.store(
            content=content,
            memory_type="design",
            lieutenant_id=self.lieutenant_id,
            title=f"Template: {template.name}",
            category="template",
            importance=0.6,
            tags=["template", template.domain] if template.domain else ["template"],
            metadata={"variables": template.variables, "domain": template.domain},
        )

    def store_antipattern(self, antipattern: AntiPattern) -> dict:
        """Store an anti-pattern (what NOT to do)."""
        content = (
            f"Anti-pattern: {antipattern.name}\n"
            f"Description: {antipattern.description}\n"
            f"Why bad: {antipattern.why_bad}\n"
            f"Better alternative: {antipattern.alternative}"
        )

        return self.mm.store(
            content=content,
            memory_type="design",
            lieutenant_id=self.lieutenant_id,
            title=f"Anti-pattern: {antipattern.name}",
            category="antipattern",
            importance=0.75,
            tags=["antipattern", "warning"],
        )

    def query_patterns(self, context: str = "", limit: int = 10) -> list[dict]:
        """Query stored patterns relevant to a context."""
        return self.mm.recall(
            query=f"pattern {context}".strip(),
            memory_types=["design"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def query_decisions(self, topic: str = "", limit: int = 10) -> list[dict]:
        """Query past design decisions."""
        return self.mm.recall(
            query=f"decision {topic}".strip(),
            memory_types=["design"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def suggest_pattern(self, task_description: str) -> PatternSuggestion | None:
        """Suggest a pattern for a given task.

        Args:
            task_description: Description of the task.

        Returns:
            Pattern suggestion or None.
        """
        patterns = self.query_patterns(task_description, limit=3)
        if patterns:
            best = patterns[0]
            return PatternSuggestion(
                pattern=best,
                relevance=best.get("importance", 0.5),
                reasoning=f"Pattern '{best.get('title', '')}' matches task context",
            )
        return None

    def evolve_pattern(self, pattern_title: str, improvement: str, evidence: str = "") -> dict:
        """Record an evolution/improvement of a pattern.

        Args:
            pattern_title: Title of the pattern being improved.
            improvement: Description of the improvement.
            evidence: Evidence supporting the improvement.

        Returns:
            Updated memory entry.
        """
        return self.mm.store(
            content=f"Pattern evolution for '{pattern_title}':\n{improvement}\n\nEvidence: {evidence}",
            memory_type="design",
            lieutenant_id=self.lieutenant_id,
            title=f"Evolution: {pattern_title}",
            category="pattern_evolution",
            importance=0.7,
            tags=["pattern", "evolution"],
            metadata={"original_pattern": pattern_title, "evidence": evidence},
        )

    def get_design_library(self, domain: str = "") -> DesignLibrary:
        """Get the complete design library for a domain."""
        all_design = self.mm.recall(
            query=domain or "design pattern template decision",
            memory_types=["design"],
            lieutenant_id=self.lieutenant_id,
            limit=100,
        )

        library = DesignLibrary(domain=domain)
        for entry in all_design:
            tags = entry.get("tags", [])
            if "pattern" in tags:
                library.patterns.append(entry)
            elif "decision" in tags:
                library.decisions.append(entry)
            elif "template" in tags:
                library.templates.append(entry)
            elif "antipattern" in tags:
                library.antipatterns.append(entry)

        library.total = len(all_design)
        return library
