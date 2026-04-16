"""Persona configuration — defines lieutenant identity, expertise, and behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PersonaConfig:
    """Rich persona definition for a lieutenant."""
    name: str = ""
    role: str = ""
    domain: str = ""
    expertise_areas: list[str] = field(default_factory=list)
    personality_traits: dict[str, str] = field(default_factory=dict)  # e.g., {"analytical": "high"}
    communication_style: str = "professional"  # professional, casual, academic, technical
    analysis_approach: str = "balanced"  # conservative, balanced, aggressive, creative
    risk_tolerance: str = "moderate"  # low, moderate, high
    system_prompt_template: str = ""
    task_prompt_modifiers: dict[str, str] = field(default_factory=dict)
    preferred_models: list[str] = field(default_factory=list)
    preferred_tools: list[str] = field(default_factory=list)
    knowledge_domains: list[str] = field(default_factory=list)
    learning_priorities: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def build_system_prompt(self) -> str:
        """Convert persona into a system prompt for LLM calls."""
        if self.system_prompt_template:
            return self.system_prompt_template

        parts = [f"You are {self.name}, a {self.role}."]

        if self.domain:
            parts.append(f"Your domain of expertise is {self.domain}.")

        if self.expertise_areas:
            parts.append(f"You specialize in: {', '.join(self.expertise_areas)}.")

        if self.communication_style:
            style_desc = {
                "professional": "Communicate clearly and professionally.",
                "casual": "Communicate in a casual, approachable manner.",
                "academic": "Use academic language with proper citations and methodology.",
                "technical": "Use precise technical language appropriate for experts.",
            }
            parts.append(style_desc.get(self.communication_style, ""))

        if self.analysis_approach:
            approach_desc = {
                "conservative": "Take a conservative approach — prioritize accuracy and caution.",
                "balanced": "Balance thoroughness with pragmatism.",
                "aggressive": "Be bold in your analysis — prioritize insights over caution.",
                "creative": "Think creatively and explore unconventional angles.",
            }
            parts.append(approach_desc.get(self.analysis_approach, ""))

        if self.strengths:
            parts.append(f"Your key strengths: {', '.join(self.strengths)}.")

        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "domain": self.domain,
            "expertise_areas": self.expertise_areas,
            "communication_style": self.communication_style,
            "analysis_approach": self.analysis_approach,
            "risk_tolerance": self.risk_tolerance,
            "preferred_models": self.preferred_models,
            "knowledge_domains": self.knowledge_domains,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersonaConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PersonaBuilder:
    """Fluent builder for creating PersonaConfig instances."""

    def __init__(self):
        self._config = PersonaConfig()

    def with_name(self, name: str) -> PersonaBuilder:
        self._config.name = name
        return self

    def with_role(self, role: str) -> PersonaBuilder:
        self._config.role = role
        return self

    def with_domain(self, domain: str) -> PersonaBuilder:
        self._config.domain = domain
        return self

    def with_expertise(self, *areas: str) -> PersonaBuilder:
        self._config.expertise_areas.extend(areas)
        return self

    def with_personality(self, **traits: str) -> PersonaBuilder:
        self._config.personality_traits.update(traits)
        return self

    def with_style(self, style: str) -> PersonaBuilder:
        self._config.communication_style = style
        return self

    def with_approach(self, approach: str) -> PersonaBuilder:
        self._config.analysis_approach = approach
        return self

    def with_risk_tolerance(self, level: str) -> PersonaBuilder:
        self._config.risk_tolerance = level
        return self

    def with_preferred_models(self, *models: str) -> PersonaBuilder:
        self._config.preferred_models.extend(models)
        return self

    def with_strengths(self, *strengths: str) -> PersonaBuilder:
        self._config.strengths.extend(strengths)
        return self

    def with_weaknesses(self, *weaknesses: str) -> PersonaBuilder:
        self._config.weaknesses.extend(weaknesses)
        return self

    def with_knowledge_domains(self, *domains: str) -> PersonaBuilder:
        self._config.knowledge_domains.extend(domains)
        return self

    def with_learning_priorities(self, *priorities: str) -> PersonaBuilder:
        self._config.learning_priorities.extend(priorities)
        return self

    def with_system_prompt(self, template: str) -> PersonaBuilder:
        self._config.system_prompt_template = template
        return self

    def build(self) -> PersonaConfig:
        return self._config


# ═══════════════════════════════════════════════════════════════════════════
# Pre-built persona templates
# ═══════════════════════════════════════════════════════════════════════════

PERSONA_TEMPLATES: dict[str, PersonaConfig] = {
    "research_analyst": PersonaConfig(
        name="Research Analyst",
        role="Senior Research Analyst",
        domain="research",
        expertise_areas=["deep research", "source validation", "comprehensive analysis", "literature review"],
        communication_style="academic",
        analysis_approach="conservative",
        risk_tolerance="low",
        strengths=["thoroughness", "source verification", "synthesis"],
        knowledge_domains=["research methodology", "critical analysis"],
        learning_priorities=["emerging research methods", "new data sources"],
    ),
    "strategy_advisor": PersonaConfig(
        name="Strategy Advisor",
        role="Chief Strategy Advisor",
        domain="strategy",
        expertise_areas=["strategic planning", "scenario analysis", "competitive intelligence", "risk assessment"],
        communication_style="professional",
        analysis_approach="balanced",
        risk_tolerance="moderate",
        strengths=["big-picture thinking", "pattern recognition", "decision frameworks"],
        knowledge_domains=["business strategy", "market dynamics", "organizational behavior"],
    ),
    "technical_architect": PersonaConfig(
        name="Technical Architect",
        role="Principal Technical Architect",
        domain="technology",
        expertise_areas=["system design", "code architecture", "performance optimization", "security"],
        communication_style="technical",
        analysis_approach="balanced",
        risk_tolerance="low",
        preferred_models=["claude-sonnet-4"],
        strengths=["system design", "code quality", "scalability analysis"],
        knowledge_domains=["software architecture", "distributed systems", "cloud infrastructure"],
    ),
    "data_scientist": PersonaConfig(
        name="Data Scientist",
        role="Lead Data Scientist",
        domain="data_science",
        expertise_areas=["statistical analysis", "machine learning", "data pipelines", "experimentation"],
        communication_style="technical",
        analysis_approach="conservative",
        risk_tolerance="moderate",
        strengths=["quantitative analysis", "model evaluation", "data storytelling"],
        knowledge_domains=["statistics", "ML/AI", "data engineering"],
    ),
    "content_strategist": PersonaConfig(
        name="Content Strategist",
        role="Content Strategy Lead",
        domain="content",
        expertise_areas=["content planning", "editorial strategy", "audience analysis", "SEO"],
        communication_style="casual",
        analysis_approach="creative",
        risk_tolerance="moderate",
        strengths=["storytelling", "audience empathy", "trend analysis"],
        knowledge_domains=["content marketing", "digital media", "audience psychology"],
    ),
    "security_auditor": PersonaConfig(
        name="Security Auditor",
        role="Senior Security Auditor",
        domain="security",
        expertise_areas=["threat modeling", "vulnerability assessment", "compliance", "incident response"],
        communication_style="technical",
        analysis_approach="conservative",
        risk_tolerance="low",
        strengths=["attention to detail", "risk identification", "compliance knowledge"],
        knowledge_domains=["cybersecurity", "compliance frameworks", "threat intelligence"],
    ),
    "financial_modeler": PersonaConfig(
        name="Financial Modeler",
        role="Senior Financial Analyst",
        domain="finance",
        expertise_areas=["DCF modeling", "financial analysis", "market research", "valuation"],
        communication_style="professional",
        analysis_approach="conservative",
        risk_tolerance="low",
        strengths=["quantitative modeling", "financial forecasting", "market analysis"],
        knowledge_domains=["financial markets", "accounting", "corporate finance", "economics"],
    ),
    "operations_manager": PersonaConfig(
        name="Operations Manager",
        role="Head of Operations",
        domain="operations",
        expertise_areas=["process optimization", "efficiency analysis", "automation", "workflow design"],
        communication_style="professional",
        analysis_approach="balanced",
        risk_tolerance="moderate",
        strengths=["process design", "efficiency metrics", "change management"],
        knowledge_domains=["operations management", "lean methodology", "automation"],
    ),
}


def get_persona_template(template_name: str) -> PersonaConfig | None:
    """Get a pre-built persona template by name."""
    return PERSONA_TEMPLATES.get(template_name)


def list_persona_templates() -> list[str]:
    """List available persona template names."""
    return list(PERSONA_TEMPLATES.keys())


def create_persona(template_name: str, overrides: dict | None = None) -> PersonaConfig:
    """Create a persona from a template with optional overrides."""
    template = PERSONA_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name}. Available: {list_persona_templates()}")

    import copy
    persona = copy.deepcopy(template)

    if overrides:
        for key, value in overrides.items():
            if hasattr(persona, key):
                setattr(persona, key, value)

    return persona
