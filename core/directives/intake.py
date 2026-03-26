"""Directive intake — validates, enriches, and prepares directives for execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IntakeResult:
    """Result of directive intake processing."""
    valid: bool = True
    directive_id: str = ""
    title: str = ""
    description: str = ""
    enriched_description: str = ""
    complexity: str = "moderate"
    estimated_tokens: int = 5000
    estimated_cost: float = 0.0
    estimated_waves: int = 1
    estimated_tasks: int = 1
    recommended_lieutenants: list[str] = field(default_factory=list)
    recommended_models: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    """Result of directive enrichment."""
    enriched_description: str = ""
    suggested_subtasks: list[dict] = field(default_factory=list)
    domain_context: str = ""
    relevant_knowledge: list[str] = field(default_factory=list)
    relevant_memories: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


class DirectiveIntake:
    """Validates, enriches, and prepares directives for execution.

    The intake stage:
    1. Validates the directive has sufficient information
    2. Estimates complexity and resource requirements
    3. Enriches the description with domain knowledge
    4. Recommends lieutenants and models
    5. Identifies risks and potential issues
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def process(self, directive_data: dict) -> IntakeResult:
        """Process a directive through intake.

        Args:
            directive_data: Dict with title, description, priority, etc.

        Returns:
            IntakeResult with validation and enrichment data.
        """
        result = IntakeResult(
            directive_id=directive_data.get("id", ""),
            title=directive_data.get("title", ""),
            description=directive_data.get("description", ""),
        )

        # 1. Validate
        issues = self._validate(directive_data)
        result.validation_issues = issues
        if issues:
            # Still process but flag issues
            logger.warning("Directive intake issues: %s", issues)

        # 2. Estimate complexity
        complexity = self._estimate_complexity(result.title, result.description)
        result.complexity = complexity["level"]
        result.estimated_tokens = complexity["tokens"]
        result.estimated_waves = complexity["waves"]
        result.estimated_tasks = complexity["tasks"]

        # 3. Estimate cost
        result.estimated_cost = self._estimate_cost(result.complexity, result.estimated_tokens)

        # 4. Recommend lieutenants
        result.recommended_lieutenants = self._recommend_lieutenants(result.title, result.description)

        # 5. Recommend models
        result.recommended_models = self._recommend_models(result.complexity)

        # 6. Identify risks
        result.risks = self._identify_risks(result)

        # 7. Enrich
        enrichment = self._enrich(result.title, result.description)
        result.enriched_description = enrichment.enriched_description
        result.tags = self._extract_tags(result.title, result.description)

        result.valid = len([i for i in issues if "CRITICAL" in i]) == 0
        return result

    def _validate(self, data: dict) -> list[str]:
        """Validate directive data."""
        issues = []

        title = data.get("title", "")
        description = data.get("description", "")

        if not title:
            issues.append("CRITICAL: Title is required")
        elif len(title) < 5:
            issues.append("WARNING: Title is very short")
        elif len(title) > 256:
            issues.append("WARNING: Title is very long")

        if not description:
            issues.append("CRITICAL: Description is required")
        elif len(description) < 20:
            issues.append("WARNING: Description is very short — may not provide enough context")

        priority = data.get("priority", 5)
        if not isinstance(priority, int) or priority < 1 or priority > 10:
            issues.append("WARNING: Priority should be between 1 and 10")

        return issues

    def _estimate_complexity(self, title: str, description: str) -> dict:
        """Estimate directive complexity."""
        text = f"{title} {description}".lower()
        word_count = len(text.split())

        complex_keywords = ["analyze", "comprehensive", "research", "design", "evaluate", "compare", "strategy"]
        expert_keywords = ["novel", "breakthrough", "cutting-edge", "mathematical", "proof"]
        simple_keywords = ["list", "summarize", "extract", "format", "check"]

        complex_score = sum(1 for k in complex_keywords if k in text)
        expert_score = sum(1 for k in expert_keywords if k in text)
        simple_score = sum(1 for k in simple_keywords if k in text)

        if expert_score >= 2 or word_count > 500:
            level = "expert"
            tokens = 20000
            waves = 4
            tasks = 10
        elif complex_score >= 2 or word_count > 200:
            level = "complex"
            tokens = 12000
            waves = 3
            tasks = 6
        elif simple_score >= 2 or word_count < 30:
            level = "simple"
            tokens = 3000
            waves = 1
            tasks = 2
        else:
            level = "moderate"
            tokens = 7000
            waves = 2
            tasks = 4

        return {"level": level, "tokens": tokens, "waves": waves, "tasks": tasks}

    def _estimate_cost(self, complexity: str, tokens: int) -> float:
        """Estimate cost based on complexity."""
        from core.routing.pricing import PricingEngine
        engine = PricingEngine()
        estimate = engine.estimate_task_cost(complexity=complexity, input_text_length=tokens * 4)
        return estimate.estimated_cost_usd * 1.5  # Buffer

    def _recommend_lieutenants(self, title: str, description: str) -> list[str]:
        """Recommend lieutenants based on directive content."""
        try:
            from core.lieutenant.workload import WorkloadBalancer
            balancer = WorkloadBalancer(self.empire_id)
            assignment = balancer.assign_task(f"{title} {description}")
            if assignment.lieutenant_id:
                return [assignment.lieutenant_id]
        except Exception:
            pass
        return []

    def _recommend_models(self, complexity: str) -> list[str]:
        """Recommend models based on complexity."""
        model_map = {
            "simple": ["gpt-4o-mini", "claude-haiku-4.5"],
            "moderate": ["claude-sonnet-4", "gpt-4o"],
            "complex": ["claude-sonnet-4", "claude-opus-4"],
            "expert": ["claude-opus-4"],
        }
        return model_map.get(complexity, ["claude-sonnet-4"])

    def _identify_risks(self, result: IntakeResult) -> list[str]:
        """Identify potential risks for the directive."""
        risks = []

        if result.complexity == "expert":
            risks.append("High complexity — may require multiple iterations")

        if result.estimated_cost > 10:
            risks.append(f"High estimated cost (${result.estimated_cost:.2f})")

        if result.estimated_tasks > 8:
            risks.append("Many tasks — coordination overhead may be significant")

        if not result.description or len(result.description) < 50:
            risks.append("Short description — lieutenants may need to make assumptions")

        # Check budget
        try:
            from core.routing.budget import BudgetManager
            bm = BudgetManager(self.empire_id)
            remaining = bm.get_remaining_daily()
            if result.estimated_cost > remaining:
                risks.append(f"Estimated cost exceeds remaining daily budget (${remaining:.2f})")
        except Exception:
            pass

        return risks

    def _enrich(self, title: str, description: str) -> EnrichmentResult:
        """Enrich directive with domain knowledge."""
        result = EnrichmentResult(enriched_description=description)

        try:
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)

            # Find relevant memories
            relevant = mm.recall(query=f"{title} {description[:200]}", limit=5)
            result.relevant_memories = [m.get("content", "")[:200] for m in relevant]

            # Find relevant knowledge
            from core.knowledge.graph import KnowledgeGraph
            graph = KnowledgeGraph(self.empire_id)
            entities = graph.find_entities(query=title, limit=5)
            result.relevant_knowledge = [f"{e.name}: {e.description[:150]}" for e in entities]

            # Build enriched description
            if result.relevant_knowledge or result.relevant_memories:
                enrichment = f"\n\n## Context from Empire Knowledge\n"
                if result.relevant_knowledge:
                    enrichment += "Relevant knowledge:\n" + "\n".join(f"- {k}" for k in result.relevant_knowledge[:3])
                if result.relevant_memories:
                    enrichment += "\nRelevant experience:\n" + "\n".join(f"- {m}" for m in result.relevant_memories[:3])
                result.enriched_description = description + enrichment

        except Exception as e:
            logger.debug("Enrichment failed: %s", e)

        return result

    def _extract_tags(self, title: str, description: str) -> list[str]:
        """Extract relevant tags from the directive."""
        text = f"{title} {description}".lower()
        tags = []

        tag_keywords = {
            "research": ["research", "investigate", "study", "survey"],
            "analysis": ["analyze", "analysis", "evaluate", "assess"],
            "strategy": ["strategy", "strategic", "plan", "roadmap"],
            "technical": ["code", "implementation", "architecture", "system"],
            "financial": ["financial", "budget", "cost", "revenue", "market"],
            "content": ["write", "content", "article", "document"],
            "security": ["security", "vulnerability", "threat", "audit"],
        }

        for tag, keywords in tag_keywords.items():
            if any(k in text for k in keywords):
                tags.append(tag)

        return tags
