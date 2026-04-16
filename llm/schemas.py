"""Pydantic models for structured LLM outputs and schema conversion."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

# Re-export the JSON parsing helper for backward compatibility.
# The canonical location is now utils.text — please prefer importing from there.
from utils.text import _extract_json_block, _find_json_object, safe_json_loads

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Planning schemas
# ═══════════════════════════════════════════════════════════════════════════

class PlanStep(BaseModel):
    """A single step in a plan."""
    step_number: int
    title: str
    description: str
    estimated_tokens: int = 1000
    model_recommendation: str = ""
    dependencies: list[int] = Field(default_factory=list)
    task_type: str = "general"


class PlanningOutput(BaseModel):
    """Output from the planning agent."""
    goal: str
    approach: str
    steps: list[PlanStep]
    estimated_complexity: str = "moderate"  # simple, moderate, complex, expert
    estimated_total_tokens: int = 5000
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    wave_structure: list[list[int]] = Field(default_factory=list)  # Groups of step numbers per wave


class TaskPlan(BaseModel):
    """Plan for a single task within a wave."""
    title: str
    description: str
    task_type: str = "general"
    assigned_lieutenant: str = ""
    estimated_tokens: int = 2000
    estimated_cost: float = 0.0
    model_recommendation: str = ""
    dependencies: list[str] = Field(default_factory=list)  # Task titles this depends on


class WavePlan(BaseModel):
    """Plan for a single execution wave."""
    wave_number: int
    description: str
    tasks: list[TaskPlan] = Field(default_factory=list)
    dependencies: list[int] = Field(default_factory=list)  # Wave numbers this depends on


class DirectivePlan(BaseModel):
    """Plan for a full directive with wave structure."""
    directive_id: str = ""
    summary: str
    waves: list[WavePlan] = Field(default_factory=list)
    total_estimated_cost: float = 0.0
    total_estimated_tokens: int = 0
    assigned_lieutenants: list[str] = Field(default_factory=list)
    dependencies: list[dict] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Analysis schemas
# ═══════════════════════════════════════════════════════════════════════════

class Finding(BaseModel):
    """A single finding from analysis."""
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    evidence: list[str] = Field(default_factory=list)
    impact: str = "medium"  # low, medium, high, critical
    category: str = ""


class AnalysisOutput(BaseModel):
    """Output from analysis tasks."""
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    methodology: str = ""
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Critic/Quality schemas
# ═══════════════════════════════════════════════════════════════════════════

class QualityScore(BaseModel):
    """Quality scores from the critic."""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    completeness: float = Field(ge=0.0, le=1.0, default=0.5)
    coherence: float = Field(ge=0.0, le=1.0, default=0.5)
    accuracy: float = Field(ge=0.0, le=1.0, default=0.5)
    overall: float = Field(ge=0.0, le=1.0, default=0.5)

    def passes_threshold(self, min_score: float = 0.6) -> bool:
        return self.overall >= min_score


class Issue(BaseModel):
    """An issue found by the critic."""
    severity: str = "medium"  # low, medium, high, critical
    description: str
    location: str = ""
    suggestion: str = ""


class CriticOutput(BaseModel):
    """Output from the critic agent."""
    scores: QualityScore
    approved: bool = False
    issues: list[Issue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    summary: str = ""
    retry_recommended: bool = False
    retry_hints: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Entity extraction schemas
# ═══════════════════════════════════════════════════════════════════════════

class ExtractedEntity(BaseModel):
    """An entity extracted from text."""
    name: str
    entity_type: str  # person, organization, concept, technology, process, metric, event, location, product
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    source_text: str = ""


class ExtractedRelation(BaseModel):
    """A relation between extracted entities."""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    evidence: str = ""


class EntityExtractionOutput(BaseModel):
    """Output from entity extraction."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    total_entities: int = 0
    total_relations: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Debate & Synthesis schemas
# ═══════════════════════════════════════════════════════════════════════════

class Argument(BaseModel):
    """An argument in a debate."""
    position: str
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    counterpoints: list[str] = Field(default_factory=list)


class DebateOutput(BaseModel):
    """Output from a debate participant."""
    lieutenant_id: str = ""
    position: str
    arguments: list[Argument] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    areas_of_agreement: list[str] = Field(default_factory=list)
    areas_of_disagreement: list[str] = Field(default_factory=list)
    recommended_action: str = ""


class SynthesisOutput(BaseModel):
    """Output from synthesis (Chief of Staff)."""
    summary: str
    key_decisions: list[str] = Field(default_factory=list)
    action_items: list[dict] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    dissenting_views: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    next_steps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Evolution schemas
# ═══════════════════════════════════════════════════════════════════════════

class ProposalOutput(BaseModel):
    """Output from upgrade proposal generation."""
    title: str
    description: str
    proposal_type: str = "optimization"
    rationale: str = ""
    changes: list[dict] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    estimated_impact: str = "medium"
    risk_level: str = "low"
    implementation_steps: list[str] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    """Output from proposal review."""
    recommendation: str = "reject"  # approve, reject, revise, defer
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.5)
    risk_assessment: str = ""
    feasibility: str = ""
    notes: str = ""
    required_changes: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Research schemas
# ═══════════════════════════════════════════════════════════════════════════

class ResearchOutput(BaseModel):
    """Output from research tasks."""
    topic: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    knowledge_gaps: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    entities_discovered: list[ExtractedEntity] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Retrospective schemas
# ═══════════════════════════════════════════════════════════════════════════

class RetrospectiveOutput(BaseModel):
    """Output from retrospective analysis."""
    what_went_well: list[str] = Field(default_factory=list)
    what_went_wrong: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    action_items: list[dict] = Field(default_factory=list)
    effectiveness_score: float = Field(ge=0.0, le=1.0, default=0.5)
    plan_accuracy: float = Field(ge=0.0, le=1.0, default=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Schema utilities
# ═══════════════════════════════════════════════════════════════════════════

def pydantic_to_tool_schema(model_class: type[BaseModel], description: str = "") -> dict:
    """Convert a Pydantic model to an LLM tool/function schema.

    Args:
        model_class: Pydantic model class.
        description: Tool description.

    Returns:
        JSON schema suitable for tool definitions.
    """
    schema = model_class.model_json_schema()

    # Remove Pydantic-specific fields
    schema.pop("title", None)

    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


def parse_llm_output(content: str, schema_class: type[BaseModel]) -> BaseModel | None:
    """Parse LLM text output into a structured Pydantic model.

    Handles JSON extraction from mixed text/JSON responses.

    Args:
        content: Raw LLM output text.
        schema_class: Target Pydantic model class.

    Returns:
        Parsed model instance, or None if parsing fails.
    """
    # Try direct JSON parse
    try:
        data = json.loads(content)
        return schema_class.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Try to extract JSON from markdown code blocks
    json_str = _extract_json_block(content)
    if json_str:
        try:
            data = json.loads(json_str)
            return schema_class.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    # Try to find JSON object in text
    json_str = _find_json_object(content)
    if json_str:
        try:
            data = json.loads(json_str)
            return schema_class.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    logger.warning("Failed to parse LLM output as %s", schema_class.__name__)
    return None


# Schema registry for easy lookup
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "planning": PlanningOutput,
    "directive_plan": DirectivePlan,
    "analysis": AnalysisOutput,
    "critic": CriticOutput,
    "entity_extraction": EntityExtractionOutput,
    "debate": DebateOutput,
    "synthesis": SynthesisOutput,
    "proposal": ProposalOutput,
    "review": ReviewOutput,
    "research": ResearchOutput,
    "retrospective": RetrospectiveOutput,
    "quality_score": QualityScore,
}


def get_schema(name: str) -> type[BaseModel] | None:
    """Get a schema class by name."""
    return SCHEMA_REGISTRY.get(name)
