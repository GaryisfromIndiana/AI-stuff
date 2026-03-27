"""Research question generation — turns knowledge gaps into actionable queries.

Takes KnowledgeGap objects from the maintenance system and uses an LLM to
produce targeted, diverse research questions with search queries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResearchQuestion:
    """A research question with search queries and metadata."""

    question_id: str = ""
    question: str = ""
    search_queries: list[str] = field(default_factory=list)
    domain: str = ""
    lieutenant_id: str = ""
    gap_topic: str = ""
    importance: float = 0.5
    strategy: str = "general"  # general, deep_dive, trend_watch, verification, gap_fill
    expected_entity_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ResearchQuestionGenerator:
    """Generates research questions from knowledge gaps using an LLM."""

    STRATEGY_MIX = {
        "gap_fill": 0.35,
        "deep_dive": 0.25,
        "trend_watch": 0.20,
        "verification": 0.10,
        "general": 0.10,
    }

    def __init__(self, empire_id: str):
        self.empire_id = empire_id

    def generate_from_gaps(
        self,
        gaps_by_domain: dict[str, list],
        max_per_domain: int = 2,
        max_total: int = 6,
    ) -> list[ResearchQuestion]:
        """Generate research questions from detected knowledge gaps.

        Args:
            gaps_by_domain: {domain: [KnowledgeGap, ...]} from gap detection.
            max_per_domain: Max questions per domain.
            max_total: Hard cap on total questions.

        Returns:
            List of ResearchQuestion, balanced across domains and strategies.
        """
        from utils.crypto import generate_id

        lt_map = self._get_lieutenant_map()
        all_questions: list[ResearchQuestion] = []

        for domain, gaps in gaps_by_domain.items():
            if len(all_questions) >= max_total:
                break

            lt_id = lt_map.get(domain, "")

            ranked_gaps = sorted(gaps, key=lambda g: g.importance, reverse=True)

            domain_questions: list[ResearchQuestion] = []
            for gap in ranked_gaps[:max_per_domain]:
                try:
                    questions = self._generate_for_gap(gap, domain, lt_id)
                    domain_questions.extend(questions)
                except Exception as exc:
                    logger.warning("Question generation failed for gap '%s': %s", gap.topic, exc)

            for q in domain_questions[:max_per_domain]:
                q.question_id = generate_id("rq")
                all_questions.append(q)

        # If LLM generation failed, fall back to gap-derived questions
        if not all_questions:
            all_questions = self._fallback_questions(gaps_by_domain, lt_map, max_total)

        return all_questions[:max_total]

    def _generate_for_gap(
        self,
        gap: Any,
        domain: str,
        lieutenant_id: str,
    ) -> list[ResearchQuestion]:
        """Use LLM to generate research questions for a specific gap."""
        from llm.router import ModelRouter, TaskMetadata
        from llm.base import LLMRequest, LLMMessage
        from core.research.strategy import StrategyTracker

        tracker = StrategyTracker(self.empire_id)
        strategy = tracker.pick_strategy(domain)

        prompt = f"""Generate 1-2 targeted research questions for this knowledge gap.

## Gap
- Topic: {gap.topic}
- Importance: {gap.importance}
- Suggested queries from system: {gap.suggested_queries}
- Related entities: {gap.related_entities}
- Domain: {domain}
- Strategy: {strategy} (focus on this approach)

## Strategy definitions
- gap_fill: Direct research to fill the specific gap
- deep_dive: In-depth investigation of the topic area
- trend_watch: Look for the latest developments and trends
- verification: Verify uncertain or low-confidence information
- general: Broad exploration of the domain

Return valid JSON only:
[
  {{
    "question": "<specific research question>",
    "search_queries": ["<web search query 1>", "<web search query 2>", "<web search query 3>"],
    "importance": <0.0-1.0>,
    "expected_entity_types": ["<type1>", "<type2>"]
  }}
]

Rules:
1. Questions must be SPECIFIC and answerable through web research.
2. Search queries should be realistic web search strings (not academic).
3. Include the current year in at least one query for freshness.
4. Each question needs 2-3 diverse search queries.
"""

        router = ModelRouter(self.empire_id)
        metadata = TaskMetadata(
            task_type="planning",
            complexity="simple",
            required_capabilities=["reasoning"],
            estimated_tokens=800,
            priority=3,
        )

        request = LLMRequest(
            messages=[LLMMessage.user(prompt)],
            system_prompt=(
                "You are a research strategist. Generate specific, actionable "
                "research questions with web search queries. Respond with valid JSON only."
            ),
            temperature=0.5,
            max_tokens=800,
        )

        response = router.execute(request, metadata)
        questions = self._parse_questions(response.content, domain, lieutenant_id, gap.topic, strategy)

        return questions

    def _parse_questions(
        self,
        raw: str,
        domain: str,
        lieutenant_id: str,
        gap_topic: str,
        strategy: str,
    ) -> list[ResearchQuestion]:
        """Parse LLM response into ResearchQuestion objects."""
        from utils.text import extract_json_block

        text = extract_json_block(raw) if "```" in raw else raw

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        questions: list[ResearchQuestion] = []
        for item in data[:2]:
            if not isinstance(item, dict) or "question" not in item:
                continue
            questions.append(ResearchQuestion(
                question=item["question"],
                search_queries=item.get("search_queries", [])[:3],
                domain=domain,
                lieutenant_id=lieutenant_id,
                gap_topic=gap_topic,
                importance=float(item.get("importance", 0.5)),
                strategy=strategy,
                expected_entity_types=item.get("expected_entity_types", []),
            ))

        return questions

    def _fallback_questions(
        self,
        gaps_by_domain: dict[str, list],
        lt_map: dict[str, str],
        max_total: int,
    ) -> list[ResearchQuestion]:
        """Generate questions directly from gaps without LLM (fallback)."""
        from utils.crypto import generate_id

        questions: list[ResearchQuestion] = []
        now_year = datetime.now(timezone.utc).year

        for domain, gaps in gaps_by_domain.items():
            lt_id = lt_map.get(domain, "")
            for gap in gaps[:2]:
                queries = gap.suggested_queries[:2] if gap.suggested_queries else [
                    f"{gap.topic} {now_year}",
                    f"latest {gap.topic} developments",
                ]
                questions.append(ResearchQuestion(
                    question_id=generate_id("rq"),
                    question=f"What are the latest developments in {gap.topic}?",
                    search_queries=queries,
                    domain=domain,
                    lieutenant_id=lt_id,
                    gap_topic=gap.topic,
                    importance=gap.importance,
                    strategy="gap_fill",
                ))
                if len(questions) >= max_total:
                    return questions

        return questions

    def _get_lieutenant_map(self) -> dict[str, str]:
        """Return {domain: lieutenant_id} for active lieutenants."""
        from db.engine import get_engine, read_session
        from db.repositories.lieutenant import LieutenantRepository

        with read_session(get_engine()) as session:
            repo = LieutenantRepository(session)
            active = repo.get_active(self.empire_id)
            return {lt.domain: lt.id for lt in active if lt.domain}
