"""Expert proposal reviewer — uses high-tier models to evaluate proposals."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Risk assessment for a proposal."""
    risk_level: str = "medium"  # low, medium, high, critical
    risk_factors: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    acceptable: bool = True
    confidence: float = 0.7


@dataclass
class QualityAssessment:
    """Quality assessment for a proposal."""
    clarity: float = 0.5
    completeness: float = 0.5
    correctness: float = 0.5
    novelty: float = 0.5
    overall: float = 0.5

    def passes(self, threshold: float = 0.5) -> bool:
        return self.overall >= threshold


@dataclass
class FeasibilityAssessment:
    """Feasibility assessment for a proposal."""
    implementable: bool = True
    effort_estimate: str = "medium"  # low, medium, high
    dependencies: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    confidence: float = 0.7


class ReviewAction:
    """Possible review actions."""
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    DEFER = "defer"


@dataclass
class ProposalReview:
    """Complete review of a proposal."""
    proposal_id: str = ""
    proposal_title: str = ""
    recommendation: str = "reject"  # approve, reject, revise, defer
    quality: QualityAssessment = field(default_factory=QualityAssessment)
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    feasibility: FeasibilityAssessment = field(default_factory=FeasibilityAssessment)
    notes: str = ""
    reviewer_model: str = ""
    confidence: float = 0.5
    cost_usd: float = 0.0
    required_changes: list[str] = field(default_factory=list)


@dataclass
class ReviewStats:
    """Statistics about reviews."""
    total_reviewed: int = 0
    approved: int = 0
    rejected: int = 0
    revised: int = 0
    deferred: int = 0
    approval_rate: float = 0.0
    avg_quality: float = 0.0
    avg_confidence: float = 0.0


class ProposalReviewer:
    """Expert LLM reviews each proposal for quality, risk, and feasibility.

    Uses a higher-tier model (e.g., Opus) for review accuracy.
    Applies auto-rejection rules for clearly poor proposals and
    tracks review consistency over time.
    """

    def __init__(self, empire_id: str = "", review_model: str = ""):
        self.empire_id = empire_id
        self._review_model = review_model or "claude-opus-4"
        self._router = None
        self._stats = ReviewStats()

        # Auto-rejection thresholds
        self._auto_reject_confidence_below = 0.3
        self._min_description_length = 20

    def _get_router(self):
        if self._router is None:
            from llm.router import ModelRouter
            self._router = ModelRouter()
        return self._router

    def review_proposal(self, proposal: dict) -> ProposalReview:
        """Review a single proposal.

        Args:
            proposal: Proposal dict with title, description, rationale, etc.

        Returns:
            ProposalReview with recommendation.
        """
        # Auto-rejection checks
        auto_reject = self._check_auto_reject(proposal)
        if auto_reject:
            self._stats.total_reviewed += 1
            self._stats.rejected += 1
            return auto_reject

        # Full LLM review
        review = self._llm_review(proposal)

        # Update stats
        self._stats.total_reviewed += 1
        if review.recommendation == ReviewAction.APPROVE:
            self._stats.approved += 1
        elif review.recommendation == ReviewAction.REJECT:
            self._stats.rejected += 1
        elif review.recommendation == ReviewAction.REVISE:
            self._stats.revised += 1
        else:
            self._stats.deferred += 1

        if self._stats.total_reviewed > 0:
            self._stats.approval_rate = self._stats.approved / self._stats.total_reviewed

        return review

    def batch_review(self, proposals: list[dict]) -> list[ProposalReview]:
        """Review multiple proposals.

        Args:
            proposals: List of proposal dicts.

        Returns:
            List of reviews.
        """
        return [self.review_proposal(p) for p in proposals]

    def _check_auto_reject(self, proposal: dict) -> ProposalReview | None:
        """Check if proposal should be auto-rejected."""
        title = proposal.get("title", "")
        description = proposal.get("description", "")
        confidence = proposal.get("confidence", 0.5)

        reasons = []

        if not title or len(title.strip()) < 5:
            reasons.append("Title too short or missing")

        if not description or len(description.strip()) < self._min_description_length:
            reasons.append(f"Description too short (min {self._min_description_length} chars)")

        if confidence < self._auto_reject_confidence_below:
            reasons.append(f"Confidence too low ({confidence:.2f} < {self._auto_reject_confidence_below})")

        if reasons:
            return ProposalReview(
                proposal_title=title,
                recommendation=ReviewAction.REJECT,
                notes=f"Auto-rejected: {'; '.join(reasons)}",
                confidence=0.95,
                quality=QualityAssessment(overall=0.1),
            )

        return None

    def _llm_review(self, proposal: dict) -> ProposalReview:
        """Perform full LLM review of a proposal."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        prompt = f"""You are an expert system reviewer. Evaluate this improvement proposal rigorously.

## Proposal
Title: {proposal.get('title', '')}
Type: {proposal.get('proposal_type', 'optimization')}
Description: {proposal.get('description', '')[:3000]}
Rationale: {proposal.get('rationale', '')[:1000]}
Risk Level: {proposal.get('risk_level', 'unknown')}
Estimated Impact: {proposal.get('estimated_impact', 'unknown')}
Confidence: {proposal.get('confidence', 'unknown')}
Implementation Steps: {proposal.get('implementation_steps', [])}

## Evaluation Criteria
1. **Quality** — Is the proposal clear, complete, correct, and novel? (score 0-1 each)
2. **Risk** — What are the risk factors? Are mitigations adequate? (low/medium/high/critical)
3. **Feasibility** — Can this actually be implemented? What effort is required?
4. **Recommendation** — approve, reject, revise, or defer?

Be rigorous but fair. Approve only proposals that are clearly beneficial with manageable risk.

Respond as JSON:
{{
    "recommendation": "approve|reject|revise|defer",
    "quality": {{
        "clarity": 0.0-1.0,
        "completeness": 0.0-1.0,
        "correctness": 0.0-1.0,
        "novelty": 0.0-1.0,
        "overall": 0.0-1.0
    }},
    "risk": {{
        "risk_level": "low|medium|high|critical",
        "risk_factors": ["..."],
        "mitigations": ["..."],
        "acceptable": true/false
    }},
    "feasibility": {{
        "implementable": true/false,
        "effort_estimate": "low|medium|high",
        "dependencies": ["..."],
        "blockers": ["..."]
    }},
    "notes": "Overall assessment...",
    "required_changes": ["changes needed if revise"],
    "confidence": 0.0-1.0
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                system_prompt="You are a senior technical reviewer. Be thorough and honest.",
                model=self._review_model,
                temperature=0.2,
                max_tokens=2000,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis", complexity="complex"))

            try:
                data = json.loads(response.content)
            except json.JSONDecodeError:
                from llm.schemas import _find_json_object
                json_str = _find_json_object(response.content)
                data = json.loads(json_str) if json_str else {}

            quality_data = data.get("quality", {})
            risk_data = data.get("risk", {})
            feasibility_data = data.get("feasibility", {})

            return ProposalReview(
                proposal_title=proposal.get("title", ""),
                recommendation=data.get("recommendation", "reject"),
                quality=QualityAssessment(
                    clarity=float(quality_data.get("clarity", 0.5)),
                    completeness=float(quality_data.get("completeness", 0.5)),
                    correctness=float(quality_data.get("correctness", 0.5)),
                    novelty=float(quality_data.get("novelty", 0.5)),
                    overall=float(quality_data.get("overall", 0.5)),
                ),
                risk=RiskAssessment(
                    risk_level=risk_data.get("risk_level", "medium"),
                    risk_factors=risk_data.get("risk_factors", []),
                    mitigations=risk_data.get("mitigations", []),
                    acceptable=risk_data.get("acceptable", True),
                ),
                feasibility=FeasibilityAssessment(
                    implementable=feasibility_data.get("implementable", True),
                    effort_estimate=feasibility_data.get("effort_estimate", "medium"),
                    dependencies=feasibility_data.get("dependencies", []),
                    blockers=feasibility_data.get("blockers", []),
                ),
                notes=data.get("notes", ""),
                reviewer_model=response.model,
                confidence=float(data.get("confidence", 0.5)),
                cost_usd=response.cost_usd,
                required_changes=data.get("required_changes", []),
            )

        except Exception as e:
            logger.error("LLM review failed: %s", e)
            return ProposalReview(
                proposal_title=proposal.get("title", ""),
                recommendation=ReviewAction.DEFER,
                notes=f"Review failed: {e}",
                confidence=0.3,
            )

    def assess_risk(self, proposal: dict) -> RiskAssessment:
        """Quick risk assessment without full review."""
        review = self.review_proposal(proposal)
        return review.risk

    def assess_quality(self, proposal: dict) -> QualityAssessment:
        """Quick quality assessment."""
        review = self.review_proposal(proposal)
        return review.quality

    def check_conflicts(
        self,
        proposal: dict,
        existing_proposals: list[dict],
    ) -> list[dict]:
        """Check if a proposal conflicts with existing ones."""
        conflicts = []
        proposal_title = proposal.get("title", "").lower()
        proposal_components = set(proposal.get("affected_components", []))

        for existing in existing_proposals:
            existing_title = existing.get("title", "").lower()
            existing_components = set(existing.get("affected_components", []))

            # Check component overlap
            overlap = proposal_components & existing_components
            if overlap:
                conflicts.append({
                    "existing_title": existing.get("title", ""),
                    "overlap": list(overlap),
                    "severity": "high" if len(overlap) > 1 else "medium",
                })

            # Check title similarity
            title_words = set(proposal_title.split())
            existing_words = set(existing_title.split())
            if len(title_words & existing_words) > 3:
                conflicts.append({
                    "existing_title": existing.get("title", ""),
                    "reason": "Similar title — may be duplicate",
                    "severity": "low",
                })

        return conflicts

    def get_review_stats(self) -> ReviewStats:
        """Get review statistics."""
        return self._stats

    def recommend_action(self, review: ProposalReview) -> str:
        """Generate a clear recommendation based on review."""
        if review.recommendation == ReviewAction.APPROVE:
            return f"APPROVE: {review.notes[:200]}"
        elif review.recommendation == ReviewAction.REJECT:
            return f"REJECT: {review.notes[:200]}"
        elif review.recommendation == ReviewAction.REVISE:
            changes = "; ".join(review.required_changes[:3])
            return f"REVISE: {changes}"
        else:
            return f"DEFER: {review.notes[:200]}"
