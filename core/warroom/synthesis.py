"""Chief of Staff synthesis — combines lieutenant outputs into unified decisions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Synthesis:
    """Result of synthesizing multiple inputs."""
    summary: str = ""
    key_decisions: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)
    confidence: float = 0.7
    cost_usd: float = 0.0


@dataclass
class UnifiedPlan:
    """A unified plan synthesized from multiple lieutenant plans."""
    summary: str = ""
    waves: list[dict] = field(default_factory=list)
    task_assignments: list[dict] = field(default_factory=list)
    dependencies: list[dict] = field(default_factory=list)
    milestones: list[str] = field(default_factory=list)
    budget_estimate: float = 0.0
    total_tasks: int = 0
    risks: list[str] = field(default_factory=list)


@dataclass
class ExecutiveSummary:
    """Executive summary of a directive result."""
    headline: str = ""
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    cost: float = 0.0


@dataclass
class ActionPlan:
    """Actionable plan from synthesis."""
    phases: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    owners: list[dict] = field(default_factory=list)
    deadlines: list[dict] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)


@dataclass
class Theme:
    """A theme identified across multiple inputs."""
    name: str = ""
    description: str = ""
    supporting_inputs: list[str] = field(default_factory=list)
    strength: float = 0.5


@dataclass
class PrioritizedRec:
    """A prioritized recommendation."""
    recommendation: str = ""
    priority: str = "medium"
    impact: str = "medium"
    effort: str = "medium"
    supporting_evidence: list[str] = field(default_factory=list)


class Synthesizer:
    """The Chief of Staff — synthesizes multiple lieutenant outputs.

    Combines debate results, plans, task results, and retrospectives
    into unified, actionable outputs. Resolves conflicts, identifies
    themes, and generates executive summaries.
    """

    def __init__(self, model: str = ""):
        self._model = model or "claude-sonnet-4"
        self._router = None

    def _get_router(self):
        if self._router is None:
            from llm.router import ModelRouter
            self._router = ModelRouter()
        return self._router

    def synthesize_debate(self, debate_data: dict) -> Synthesis:
        """Synthesize debate results into decisions.

        Args:
            debate_data: Debate data with contributions.

        Returns:
            Synthesis with decisions and action items.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        contributions = debate_data.get("contributions", [])
        contrib_text = "\n\n".join(
            f"**{c.get('name', 'Unknown')}** ({c.get('domain', '')}):\n"
            f"Position: {c.get('position', '')}\n"
            f"Reasoning: {str(c.get('arguments', ''))[:300]}\n"
            f"Confidence: {c.get('confidence', 0)}"
            for c in contributions
        )

        prompt = f"""As Chief of Staff, synthesize this debate into clear decisions.

Topic: {debate_data.get('topic', '')}

Contributions:
{contrib_text}

Synthesize into:
1. Summary — what was decided
2. Key decisions — specific, actionable decisions
3. Action items — concrete next steps
4. Risks — potential issues
5. Dissenting views — important minority opinions

Respond as JSON:
{{
    "summary": "...",
    "key_decisions": ["..."],
    "action_items": [{{"description": "...", "assigned_to": "", "priority": "high"}}],
    "risks": ["..."],
    "dissenting_views": ["..."],
    "confidence": 0.7
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                system_prompt="You are the Chief of Staff. Synthesize fairly, capturing all perspectives. Be decisive but balanced.",
                model=self._model,
                temperature=0.3,
                max_tokens=2000,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis", complexity="complex"))

            from llm.schemas import safe_json_loads
            data = safe_json_loads(response.content)

            return Synthesis(
                summary=data.get("summary", ""),
                key_decisions=data.get("key_decisions", []),
                action_items=data.get("action_items", []),
                risks=data.get("risks", []),
                dissenting_views=data.get("dissenting_views", []),
                confidence=float(data.get("confidence", 0.7)),
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Debate synthesis failed: %s", e)
            return Synthesis(summary=f"Synthesis failed: {e}")

    def synthesize_plans(self, plans: list[dict], directive_title: str = "") -> UnifiedPlan:
        """Synthesize multiple plans into a unified execution plan.

        Args:
            plans: Individual plans from lieutenants.
            directive_title: Title of the directive.

        Returns:
            UnifiedPlan.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        plans_text = "\n\n---\n\n".join(
            f"**{p.get('name', 'Unknown')}** ({p.get('domain', '')}):\n{p.get('plan', '')[:2000]}"
            for p in plans
        )

        prompt = f"""Synthesize these individual plans into one unified execution plan.

Directive: {directive_title}

Individual Plans:
{plans_text}

Create a unified plan with:
1. Summary of the approach
2. Waves of execution (tasks grouped by dependency)
3. Task assignments to specific lieutenants
4. Milestones and checkpoints
5. Risks and mitigations

Respond as JSON:
{{
    "summary": "...",
    "waves": [
        {{
            "wave_number": 1,
            "description": "...",
            "tasks": [{{"title": "...", "assigned_to": "...", "description": "...", "estimated_tokens": 2000}}]
        }}
    ],
    "milestones": ["..."],
    "risks": ["..."],
    "budget_estimate_tasks": 0
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._model,
                temperature=0.3,
                max_tokens=3000,
            )
            response = router.execute(request, TaskMetadata(task_type="planning", complexity="complex"))

            from llm.schemas import safe_json_loads
            data = safe_json_loads(response.content)

            total_tasks = sum(len(w.get("tasks", [])) for w in data.get("waves", []))

            return UnifiedPlan(
                summary=data.get("summary", ""),
                waves=data.get("waves", []),
                milestones=data.get("milestones", []),
                risks=data.get("risks", []),
                total_tasks=total_tasks,
            )

        except Exception as e:
            logger.error("Plan synthesis failed: %s", e)
            return UnifiedPlan(summary=f"Synthesis failed: {e}")

    def synthesize_results(self, task_results: list[dict], directive_title: str = "") -> str:
        """Synthesize task results into a directive output.

        Args:
            task_results: Individual task results.
            directive_title: Directive title.

        Returns:
            Synthesized output text.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        results_text = "\n\n---\n\n".join(
            f"**Task {i+1}** ({r.get('task_type', 'general')}):\n{r.get('content', '')[:1500]}"
            for i, r in enumerate(task_results[:10])
        )

        prompt = f"""Synthesize these task results into a comprehensive output.

Directive: {directive_title}

Task Results:
{results_text}

Create a unified output that:
1. Combines all findings coherently
2. Resolves any contradictions
3. Highlights key insights
4. Provides actionable conclusions
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._model,
                temperature=0.4,
                max_tokens=4000,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis", complexity="complex"))
            return response.content

        except Exception as e:
            logger.error("Result synthesis failed: %s", e)
            return f"Synthesis failed: {e}"

    def create_executive_summary(
        self,
        directive_title: str,
        results: dict,
    ) -> ExecutiveSummary:
        """Create an executive summary of directive results.

        Args:
            directive_title: Directive title.
            results: Directive execution results.

        Returns:
            ExecutiveSummary.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        results_json = json.dumps(results, indent=2, default=str)[:5000]

        prompt = f"""Create an executive summary for this completed directive.

Directive: {directive_title}
Results:
{results_json}

Provide:
1. Headline — one sentence summary
2. Key findings — 3-5 bullet points
3. Recommendations — what to do next
4. Risks — what to watch out for
5. Next steps — immediate actions

Respond as JSON:
{{
    "headline": "...",
    "key_findings": ["..."],
    "recommendations": ["..."],
    "risks": ["..."],
    "next_steps": ["..."]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._model,
                temperature=0.3,
                max_tokens=1500,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis"))

            from llm.schemas import safe_json_loads
            data = safe_json_loads(response.content)

            return ExecutiveSummary(
                headline=data.get("headline", ""),
                key_findings=data.get("key_findings", []),
                recommendations=data.get("recommendations", []),
                risks=data.get("risks", []),
                next_steps=data.get("next_steps", []),
                cost=response.cost_usd,
            )

        except Exception as e:
            return ExecutiveSummary(headline=f"Summary generation failed: {e}")

    def identify_themes(self, contributions: list[str]) -> list[Theme]:
        """Identify common themes across multiple inputs.

        Args:
            contributions: List of text contributions.

        Returns:
            List of identified themes.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        text = "\n\n---\n\n".join(
            f"Input {i+1}: {c[:500]}" for i, c in enumerate(contributions[:10])
        )

        prompt = f"""Identify common themes across these inputs.

{text}

Respond as JSON:
{{
    "themes": [
        {{
            "name": "...",
            "description": "...",
            "supporting_inputs": [1, 3, 5],
            "strength": 0.8
        }}
    ]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                temperature=0.3,
                max_tokens=1500,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis"))

            try:
                data = json.loads(response.content)
            except json.JSONDecodeError:
                return []

            return [
                Theme(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    supporting_inputs=[str(s) for s in t.get("supporting_inputs", [])],
                    strength=float(t.get("strength", 0.5)),
                )
                for t in data.get("themes", [])
            ]

        except Exception as e:
            logger.error("Theme identification failed: %s", e)
            return []

    def prioritize_recommendations(
        self,
        recommendations: list[str],
    ) -> list[PrioritizedRec]:
        """Prioritize a list of recommendations by impact and effort.

        Args:
            recommendations: List of recommendation strings.

        Returns:
            Prioritized recommendations.
        """
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        recs_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations[:15]))

        prompt = f"""Prioritize these recommendations by impact and effort.

Recommendations:
{recs_text}

For each, assess:
- Priority: critical/high/medium/low
- Impact: high/medium/low
- Effort: high/medium/low

Respond as JSON:
{{
    "prioritized": [
        {{
            "recommendation": "...",
            "priority": "high",
            "impact": "high",
            "effort": "low"
        }}
    ]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                temperature=0.2,
                max_tokens=1500,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis"))

            try:
                data = json.loads(response.content)
            except json.JSONDecodeError:
                return [PrioritizedRec(recommendation=r) for r in recommendations]

            return [
                PrioritizedRec(
                    recommendation=p.get("recommendation", ""),
                    priority=p.get("priority", "medium"),
                    impact=p.get("impact", "medium"),
                    effort=p.get("effort", "medium"),
                )
                for p in data.get("prioritized", [])
            ]

        except Exception as e:
            return [PrioritizedRec(recommendation=r) for r in recommendations]

    def resolve_conflicts(self, conflicting_outputs: list[dict]) -> dict:
        """Resolve conflicts between different lieutenant outputs.

        Uses quality-weighted synthesis — higher quality inputs get more weight.

        Args:
            conflicting_outputs: List of {content, quality_score, lieutenant_name} dicts.

        Returns:
            Resolution dict with resolved content.
        """
        if not conflicting_outputs:
            return {"resolved": "", "method": "none"}

        # Sort by quality
        sorted_outputs = sorted(conflicting_outputs, key=lambda x: x.get("quality_score", 0), reverse=True)

        if len(sorted_outputs) == 1:
            return {"resolved": sorted_outputs[0].get("content", ""), "method": "single_input"}

        # Quality-weighted synthesis
        total_quality = sum(o.get("quality_score", 0.5) for o in sorted_outputs)
        weights = {
            o.get("lieutenant_name", f"lt_{i}"): o.get("quality_score", 0.5) / max(total_quality, 0.001)
            for i, o in enumerate(sorted_outputs)
        }

        # Use highest quality output as base, enrich with others
        best = sorted_outputs[0]
        others_text = "\n".join(
            f"[{o.get('lieutenant_name', '')} (quality: {o.get('quality_score', 0):.2f})]: {o.get('content', '')[:500]}"
            for o in sorted_outputs[1:3]
        )

        resolved = (
            f"Primary (from {best.get('lieutenant_name', 'unknown')}, quality: {best.get('quality_score', 0):.2f}):\n"
            f"{best.get('content', '')}\n\n"
            f"Additional perspectives:\n{others_text}"
        )

        return {
            "resolved": resolved,
            "method": "quality_weighted",
            "weights": weights,
            "primary_source": best.get("lieutenant_name", ""),
        }
