"""Report templates — structured formats for different output types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReportTemplate:
    """Template for generating a specific type of report."""
    name: str
    description: str
    sections: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    output_format: str = "markdown"  # markdown, html, json


REPORT_TEMPLATES: dict[str, ReportTemplate] = {
    "research_briefing": ReportTemplate(
        name="Research Briefing",
        description="Concise briefing on a research topic — executive summary, key findings, implications",
        sections=[
            {"name": "Executive Summary", "prompt": "Write a 3-5 sentence executive summary of the research."},
            {"name": "Key Findings", "prompt": "List the most important findings as bullet points with evidence."},
            {"name": "Analysis", "prompt": "Provide detailed analysis of the findings and what they mean."},
            {"name": "Implications", "prompt": "What are the implications for the AI landscape?"},
            {"name": "Open Questions", "prompt": "What questions remain unanswered? What needs further research?"},
        ],
        system_prompt="You are an AI research analyst writing a briefing for a technical audience. Be specific, cite sources, and distinguish between confirmed facts and speculation.",
    ),
    "weekly_digest": ReportTemplate(
        name="Weekly AI Digest",
        description="Weekly summary of AI developments — what happened, what matters, what's next",
        sections=[
            {"name": "This Week in AI", "prompt": "Summarize the most important AI developments from this week."},
            {"name": "Model Releases & Updates", "prompt": "Any new model releases, updates, or benchmark results?"},
            {"name": "Industry Moves", "prompt": "Company news, funding, partnerships, acquisitions."},
            {"name": "Research Highlights", "prompt": "Notable papers, techniques, or breakthroughs."},
            {"name": "What to Watch", "prompt": "What's coming next? Trends to monitor."},
        ],
        system_prompt="You are writing a weekly AI digest for practitioners who need to stay current. Be concise, prioritize what matters, skip the hype.",
    ),
    "deep_dive": ReportTemplate(
        name="Deep Dive Report",
        description="In-depth analysis of a single topic with full context",
        sections=[
            {"name": "Overview", "prompt": "Provide comprehensive context and background."},
            {"name": "Current State", "prompt": "What is the current state of this topic?"},
            {"name": "Key Players", "prompt": "Who are the main organizations and people involved?"},
            {"name": "Technical Details", "prompt": "Explain the technical aspects in depth."},
            {"name": "Challenges & Limitations", "prompt": "What are the unsolved problems and limitations?"},
            {"name": "Future Outlook", "prompt": "Where is this headed? Predictions with reasoning."},
            {"name": "Recommendations", "prompt": "What should practitioners do based on this analysis?"},
        ],
        system_prompt="You are writing an in-depth technical report for AI professionals. Be thorough, precise, and analytical. Support claims with evidence.",
    ),
    "competitive_analysis": ReportTemplate(
        name="Competitive Analysis",
        description="Compare and analyze competing products, models, or approaches",
        sections=[
            {"name": "Market Overview", "prompt": "Overview of the competitive landscape."},
            {"name": "Player Comparison", "prompt": "Compare the key players on capabilities, pricing, and strategy."},
            {"name": "Strengths & Weaknesses", "prompt": "Analyze each player's strengths and weaknesses."},
            {"name": "Market Dynamics", "prompt": "What forces are shaping competition?"},
            {"name": "Predictions", "prompt": "Who is likely to win and why?"},
        ],
        system_prompt="You are a competitive intelligence analyst. Be objective, use data where available, and clearly state assumptions.",
    ),
    "status_report": ReportTemplate(
        name="Empire Status Report",
        description="Internal report on Empire's knowledge state and activity",
        sections=[
            {"name": "Knowledge Overview", "prompt": "Summarize what Empire currently knows — key entities, facts, trends."},
            {"name": "Recent Activity", "prompt": "What research has Empire conducted recently?"},
            {"name": "Knowledge Gaps", "prompt": "What areas need more research?"},
            {"name": "System Health", "prompt": "How is the system performing? Quality scores, costs, reliability."},
            {"name": "Recommendations", "prompt": "What should Empire focus on next?"},
        ],
        system_prompt="You are generating an internal status report for Empire AI. Use the actual data provided — don't make up statistics.",
    ),
}


def get_template(name: str) -> ReportTemplate | None:
    """Get a report template by name."""
    return REPORT_TEMPLATES.get(name)


def list_templates() -> list[dict]:
    """List all available templates."""
    return [
        {"key": k, "name": t.name, "description": t.description, "sections": len(t.sections)}
        for k, t in REPORT_TEMPLATES.items()
    ]
