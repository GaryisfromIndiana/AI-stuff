"""Upgrade proposer — lieutenants propose system improvements from domain knowledge."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProposalType(str, Enum):
    OPTIMIZATION = "optimization"
    BUG_FIX = "bug_fix"
    NEW_CAPABILITY = "new_capability"
    REFACTOR = "refactor"
    KNOWLEDGE_UPDATE = "knowledge_update"
    PROCESS_IMPROVEMENT = "process_improvement"


@dataclass
class EvolutionProposal:
    """A system improvement proposal from a lieutenant."""
    title: str = ""
    description: str = ""
    proposal_type: str = "optimization"
    rationale: str = ""
    changes: list[dict] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)
    estimated_impact: str = "medium"
    risk_level: str = "low"
    implementation_steps: list[str] = field(default_factory=list)
    lieutenant_id: str = ""
    confidence: float = 0.5
    source: str = ""  # knowledge, failure, pattern, retrospective


@dataclass
class ImpactEstimate:
    """Estimated impact of a proposal."""
    affected_components: list[str] = field(default_factory=list)
    risk: str = "low"
    effort: str = "medium"
    expected_improvement: str = ""
    confidence: float = 0.5


@dataclass
class ValidationResult:
    """Result of proposal validation."""
    valid: bool = True
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    quality_score: float = 0.5


class UpgradeProposer:
    """Generates system improvement proposals from lieutenant knowledge.

    Lieutenants use their domain expertise, past failures, observed patterns,
    and retrospective insights to propose concrete improvements.
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id
        self._router = None

    def _get_router(self):
        if self._router is None:
            from llm.router import ModelRouter
            self._router = ModelRouter()
        return self._router

    def generate_proposals(
        self,
        lieutenant_id: str,
        lieutenant_name: str = "",
        domain: str = "",
        context: str = "",
        max_proposals: int = 3,
    ) -> list[EvolutionProposal]:
        """Generate improvement proposals from a lieutenant.

        Args:
            lieutenant_id: Lieutenant ID.
            lieutenant_name: Lieutenant name.
            domain: Lieutenant's domain.
            context: System context.
            max_proposals: Max proposals to generate.

        Returns:
            List of proposals.
        """
        proposals = []

        # Generate from different sources
        from_knowledge = self.propose_from_knowledge(lieutenant_id, domain)
        if from_knowledge:
            proposals.extend(from_knowledge[:1])

        from_failures = self.propose_from_failures(lieutenant_id)
        if from_failures:
            proposals.extend(from_failures[:1])

        from_patterns = self.propose_from_patterns(lieutenant_id, domain)
        if from_patterns:
            proposals.extend(from_patterns[:1])

        # Deduplicate and prioritize
        proposals = self.deduplicate_proposals(proposals)
        proposals = self.prioritize_proposals(proposals)

        return proposals[:max_proposals]

    def propose_from_knowledge(self, lieutenant_id: str, domain: str = "") -> list[EvolutionProposal]:
        """Generate proposals from domain knowledge insights."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        # Get relevant knowledge
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        memories = mm.recall(query=f"improvement opportunity {domain}", memory_types=["semantic", "experiential"], lieutenant_id=lieutenant_id, limit=5)

        if not memories:
            return []

        mem_text = "\n".join(f"- {m.get('content', '')[:200]}" for m in memories[:5])

        prompt = f"""Based on this domain knowledge, propose a specific system improvement.

Domain: {domain}
Relevant knowledge:
{mem_text}

Propose ONE concrete improvement:
1. What to change
2. Why it would help
3. Implementation steps
4. Expected impact
5. Risk level

Respond as JSON:
{{
    "title": "...",
    "description": "...",
    "proposal_type": "optimization|new_capability|process_improvement",
    "rationale": "...",
    "implementation_steps": ["..."],
    "estimated_impact": "low|medium|high",
    "risk_level": "low|medium|high",
    "confidence": 0.7
}}
"""
        try:
            router = self._get_router()
            request = LLMRequest(messages=[LLMMessage.user(prompt)], temperature=0.5, max_tokens=1500)
            response = router.execute(request, TaskMetadata(task_type="analysis"))

            try:
                data = json.loads(response.content)
            except json.JSONDecodeError:
                from llm.schemas import _find_json_object
                json_str = _find_json_object(response.content)
                data = json.loads(json_str) if json_str else {}

            if data.get("title"):
                return [EvolutionProposal(
                    title=data.get("title", ""),
                    description=data.get("description", ""),
                    proposal_type=data.get("proposal_type", "optimization"),
                    rationale=data.get("rationale", ""),
                    implementation_steps=data.get("implementation_steps", []),
                    estimated_impact=data.get("estimated_impact", "medium"),
                    risk_level=data.get("risk_level", "low"),
                    lieutenant_id=lieutenant_id,
                    confidence=float(data.get("confidence", 0.5)),
                    source="knowledge",
                )]

        except Exception as e:
            logger.warning("Knowledge-based proposal failed: %s", e)

        return []

    def propose_from_failures(self, lieutenant_id: str) -> list[EvolutionProposal]:
        """Generate proposals from past failures."""
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        failures = mm.recall(query="failure error", memory_types=["experiential"], lieutenant_id=lieutenant_id, limit=5)

        if not failures:
            return []

        proposals = []
        for failure in failures[:2]:
            content = failure.get("content", "")
            if "Error" in content or "failure" in content.lower():
                proposals.append(EvolutionProposal(
                    title=f"Fix: {content[:80]}",
                    description=f"Address recurring failure: {content[:500]}",
                    proposal_type="bug_fix",
                    rationale="This failure has occurred previously and should be prevented",
                    lieutenant_id=lieutenant_id,
                    confidence=0.6,
                    source="failure",
                ))

        return proposals

    def propose_from_patterns(self, lieutenant_id: str, domain: str = "") -> list[EvolutionProposal]:
        """Generate proposals from observed patterns."""
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        patterns = mm.recall(query="pattern success improvement", memory_types=["design", "experiential"], lieutenant_id=lieutenant_id, limit=5)

        if not patterns:
            return []

        proposals = []
        for pattern in patterns[:2]:
            content = pattern.get("content", "")
            proposals.append(EvolutionProposal(
                title=f"Optimize: {content[:80]}",
                description=f"Apply observed pattern: {content[:500]}",
                proposal_type="optimization",
                rationale="Pattern identified through repeated observation",
                lieutenant_id=lieutenant_id,
                confidence=0.5,
                source="pattern",
            ))

        return proposals

    def propose_from_retrospectives(self, lieutenant_id: str) -> list[EvolutionProposal]:
        """Generate proposals from retrospective action items."""
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        retro_items = mm.recall(query="retrospective improvement action", memory_types=["experiential", "design"], lieutenant_id=lieutenant_id, limit=5)

        proposals = []
        for item in retro_items[:2]:
            content = item.get("content", "")
            if "improvement" in content.lower() or "retrospective" in content.lower():
                proposals.append(EvolutionProposal(
                    title=f"Retrospective: {content[:80]}",
                    description=content[:500],
                    proposal_type="process_improvement",
                    rationale="Identified during retrospective",
                    lieutenant_id=lieutenant_id,
                    confidence=0.65,
                    source="retrospective",
                ))

        return proposals

    def validate_proposal(self, proposal: EvolutionProposal) -> ValidationResult:
        """Validate a proposal for quality and completeness."""
        issues = []
        suggestions = []

        if not proposal.title or len(proposal.title) < 5:
            issues.append("Title is too short or missing")
        if not proposal.description or len(proposal.description) < 20:
            issues.append("Description is too short")
        if not proposal.rationale:
            suggestions.append("Add a rationale explaining why this change is needed")
        if not proposal.implementation_steps:
            suggestions.append("Add implementation steps")
        if proposal.confidence < 0.3:
            issues.append(f"Very low confidence ({proposal.confidence:.2f})")

        quality_score = 1.0
        quality_score -= len(issues) * 0.2
        quality_score -= len(suggestions) * 0.1
        quality_score = max(0.0, min(1.0, quality_score))

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            quality_score=quality_score,
        )

    def deduplicate_proposals(self, proposals: list[EvolutionProposal]) -> list[EvolutionProposal]:
        """Remove duplicate proposals."""
        seen_titles = set()
        unique = []
        for p in proposals:
            title_key = p.title.lower().strip()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(p)
        return unique

    def prioritize_proposals(self, proposals: list[EvolutionProposal]) -> list[EvolutionProposal]:
        """Prioritize proposals by impact and confidence."""
        impact_scores = {"high": 3, "medium": 2, "low": 1}
        risk_scores = {"low": 3, "medium": 2, "high": 1}

        def score(p: EvolutionProposal) -> float:
            return (
                impact_scores.get(p.estimated_impact, 2) * 0.4 +
                risk_scores.get(p.risk_level, 2) * 0.2 +
                p.confidence * 3 * 0.4
            )

        proposals.sort(key=score, reverse=True)
        return proposals

    def estimate_impact(self, proposal: EvolutionProposal) -> ImpactEstimate:
        """Estimate the impact of a proposal."""
        return ImpactEstimate(
            affected_components=proposal.affected_components or ["general"],
            risk=proposal.risk_level,
            effort="medium",
            expected_improvement=proposal.estimated_impact,
            confidence=proposal.confidence,
        )
