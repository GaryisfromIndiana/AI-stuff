"""Critic agent — the quality gatekeeper of the ACE pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm.base import LLMMessage, LLMRequest
from llm.router import ModelRouter, TaskMetadata
from llm.schemas import safe_json_loads

logger = logging.getLogger(__name__)


@dataclass
class CriticEvaluation:
    """Complete evaluation from the critic."""
    scores: dict = field(default_factory=dict)
    overall_score: float = 0.0
    approved: bool = False
    issues: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    summary: str = ""
    retry_recommended: bool = False
    retry_hints: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.approved or self.overall_score >= 0.6


@dataclass
class HallucinationCheck:
    """Result of hallucination checking."""
    hallucination_score: float = 0.0  # 0 = no hallucination, 1 = fully hallucinated
    unsupported_claims: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)
    confidence: float = 0.7


@dataclass
class CompletenessCheck:
    """Result of completeness checking."""
    completeness_score: float = 0.0
    requirements_met: list[str] = field(default_factory=list)
    requirements_missing: list[str] = field(default_factory=list)
    partial_requirements: list[str] = field(default_factory=list)


@dataclass
class RetryDecision:
    """Whether to retry a task after quality evaluation."""
    should_retry: bool = False
    reason: str = ""
    hints: list[str] = field(default_factory=list)
    escalate_model: bool = False
    suggested_changes: list[str] = field(default_factory=list)


class Critic:
    """The quality gatekeeper — evaluates output quality and decides retries.

    Scores outputs on confidence, completeness, coherence, and accuracy.
    Provides detailed feedback for retry iterations.
    """

    def __init__(
        self,
        router: ModelRouter | None = None,
        default_model: str = "",
        min_quality: float = 0.6,
    ):
        self.router = router or ModelRouter()
        self._default_model = default_model or "claude-haiku-4.5"
        self._min_quality = min_quality

    def evaluate(
        self,
        task_title: str,
        task_description: str,
        result_content: str,
        requirements: list[str] | None = None,
        context: str = "",
    ) -> CriticEvaluation:
        """Evaluate the quality of a task result.

        Args:
            task_title: Original task title.
            task_description: Original task description.
            result_content: The output to evaluate.
            requirements: List of requirements to check against.
            context: Additional context.

        Returns:
            CriticEvaluation with scores, issues, and suggestions.
        """
        if not result_content or not result_content.strip():
            return CriticEvaluation(
                overall_score=0.0,
                approved=False,
                issues=[{"severity": "critical", "description": "No content produced"}],
                suggestions=["The execution produced no output. Retry with clearer instructions."],
                retry_recommended=True,
            )

        reqs_text = "\n".join(f"- {r}" for r in (requirements or []))
        prompt = f"""Evaluate this task output for quality.

## Original Task
Title: {task_title}
Description: {task_description}
{f"Requirements:{chr(10)}{reqs_text}" if reqs_text else ""}

## Output to Evaluate
{result_content[:8000]}

## Evaluation Criteria
Score each dimension from 0.0 to 1.0:
1. **Confidence** — Are claims well-supported? Is uncertainty acknowledged?
2. **Completeness** — Are all requirements addressed? Is anything missing?
3. **Coherence** — Is the output logically consistent and well-structured?
4. **Accuracy** — Are facts and reasoning sound?

Also:
- List any issues found (with severity: low/medium/high/critical)
- Provide specific, actionable suggestions for improvement
- State whether you approve the output
- If not approved, explain what needs to change

Respond as JSON:
{{
    "confidence": 0.0-1.0,
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "approved": true/false,
    "issues": [{{"severity": "high", "description": "..."}}],
    "suggestions": ["..."],
    "summary": "Brief evaluation"
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._default_model,
                system_prompt="You are an expert quality evaluator. Be fair but rigorous. Focus on substance.",
                temperature=0.2,
                max_tokens=1500,
            )
            response = self.router.execute(request, TaskMetadata(task_type="analysis", complexity="moderate"))

            # Parse response
            data = safe_json_loads(response.content)

            overall = float(data.get("overall_score", 0.5))
            approved = data.get("approved", overall >= self._min_quality)

            return CriticEvaluation(
                scores={
                    "confidence": float(data.get("confidence", 0.5)),
                    "completeness": float(data.get("completeness", 0.5)),
                    "coherence": float(data.get("coherence", 0.5)),
                    "accuracy": float(data.get("accuracy", 0.5)),
                },
                overall_score=overall,
                approved=approved,
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                summary=data.get("summary", ""),
                retry_recommended=not approved,
                retry_hints=data.get("suggestions", []),
            )

        except Exception as e:
            logger.warning("Critic evaluation failed: %s", e)
            return CriticEvaluation(
                overall_score=0.5,
                approved=False,
                summary=f"Evaluation error: {e}",
            )

    def score_quality(self, content: str, context: str = "") -> dict:
        """Quick quality scoring without full evaluation."""
        eval_result = self.evaluate(
            task_title="Quality Check",
            task_description=context or "Evaluate this content",
            result_content=content,
        )
        return eval_result.scores

    def check_hallucination(
        self,
        content: str,
        sources: list[str] | None = None,
        context: str = "",
    ) -> HallucinationCheck:
        """Check content for hallucinations/unsupported claims.

        Args:
            content: Content to check.
            sources: Source material to check against.
            context: Additional context.

        Returns:
            HallucinationCheck result.
        """
        sources_text = "\n".join(f"Source {i+1}: {s[:2000]}" for i, s in enumerate(sources or []))

        prompt = f"""Check this content for hallucinations (unsupported or fabricated claims).

## Content to Check
{content[:6000]}

{f"## Source Material{chr(10)}{sources_text}" if sources_text else "## Note: No source material provided. Check for obviously false or fabricated claims."}

Identify:
1. Claims that are NOT supported by the source material or general knowledge
2. Claims that ARE well-supported
3. Overall hallucination score (0.0 = no hallucination, 1.0 = fully fabricated)

Respond as JSON:
{{
    "hallucination_score": 0.0-1.0,
    "unsupported_claims": ["..."],
    "supported_claims": ["..."],
    "confidence": 0.0-1.0
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._default_model,
                temperature=0.1,
                max_tokens=1500,
            )
            response = self.router.execute(request, TaskMetadata(task_type="analysis"))

            data = safe_json_loads(response.content)

            return HallucinationCheck(
                hallucination_score=float(data.get("hallucination_score", 0.3)),
                unsupported_claims=data.get("unsupported_claims", []),
                supported_claims=data.get("supported_claims", []),
                confidence=float(data.get("confidence", 0.7)),
            )

        except Exception as e:
            logger.warning("Hallucination check failed: %s", e)
            return HallucinationCheck()

    def check_completeness(
        self,
        content: str,
        requirements: list[str],
    ) -> CompletenessCheck:
        """Check if content meets all specified requirements.

        Args:
            content: Content to check.
            requirements: List of requirements.

        Returns:
            CompletenessCheck result.
        """
        if not requirements:
            return CompletenessCheck(completeness_score=1.0)

        reqs_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(requirements))
        prompt = f"""Check if this content meets all requirements.

## Content
{content[:6000]}

## Requirements
{reqs_text}

For each requirement, classify as:
- MET: Fully addressed
- PARTIAL: Partially addressed
- MISSING: Not addressed at all

Respond as JSON:
{{
    "completeness_score": 0.0-1.0,
    "requirements_met": ["..."],
    "requirements_missing": ["..."],
    "partial_requirements": ["..."]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._default_model,
                temperature=0.1,
                max_tokens=1000,
            )
            response = self.router.execute(request, TaskMetadata(task_type="analysis"))

            data = safe_json_loads(response.content)

            return CompletenessCheck(
                completeness_score=float(data.get("completeness_score", 0.5)),
                requirements_met=data.get("requirements_met", []),
                requirements_missing=data.get("requirements_missing", []),
                partial_requirements=data.get("partial_requirements", []),
            )

        except Exception as e:
            logger.warning("Completeness check failed: %s", e)
            return CompletenessCheck()

    def should_retry(self, evaluation: CriticEvaluation, attempt: int = 0, max_attempts: int = 5) -> RetryDecision:
        """Decide whether a task should be retried.

        Args:
            evaluation: The critic evaluation.
            attempt: Current attempt number.
            max_attempts: Maximum attempts allowed.

        Returns:
            RetryDecision.
        """
        if evaluation.approved:
            return RetryDecision(should_retry=False, reason="Approved by critic")

        if attempt >= max_attempts:
            return RetryDecision(should_retry=False, reason=f"Max attempts ({max_attempts}) reached")

        if evaluation.overall_score == 0.0:
            return RetryDecision(
                should_retry=True,
                reason="No output produced",
                escalate_model=True,
                hints=["Retry with clearer instructions and stronger model"],
            )

        if evaluation.overall_score < 0.3:
            return RetryDecision(
                should_retry=True,
                reason=f"Very low quality ({evaluation.overall_score:.2f})",
                escalate_model=attempt >= 1,
                hints=evaluation.retry_hints,
                suggested_changes=evaluation.suggestions,
            )

        if evaluation.overall_score < self._min_quality:
            return RetryDecision(
                should_retry=True,
                reason=f"Quality ({evaluation.overall_score:.2f}) below threshold ({self._min_quality:.2f})",
                escalate_model=attempt >= 2,
                hints=evaluation.retry_hints,
                suggested_changes=evaluation.suggestions,
            )

        return RetryDecision(should_retry=False, reason="Quality acceptable")

    def suggest_improvements(self, content: str, context: str = "") -> list[str]:
        """Get improvement suggestions for content without full evaluation.

        Args:
            content: Content to improve.
            context: Context about the content.

        Returns:
            List of improvement suggestions.
        """
        eval_result = self.evaluate(
            task_title="Improvement Check",
            task_description=context or "Suggest improvements for this content",
            result_content=content,
        )
        return eval_result.suggestions
