"""Prompt evolution — lieutenants improve their own system prompts over time.

After every N tasks, the system analyzes what worked and what didn't,
then proposes improvements to the lieutenant's persona and prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptEvolution:
    """A proposed change to a lieutenant's system prompt."""
    lieutenant_id: str = ""
    lieutenant_name: str = ""
    current_prompt: str = ""
    proposed_prompt: str = ""
    reasoning: str = ""
    confidence: float = 0.5
    based_on_tasks: int = 0
    avg_quality_before: float = 0.0


@dataclass
class StrategyLearning:
    """A learned strategy from task performance patterns."""
    strategy: str = ""
    evidence: str = ""
    effectiveness: float = 0.5
    applicable_to: list[str] = field(default_factory=list)
    source_tasks: int = 0


class PromptEvolver:
    """Evolves lieutenant system prompts based on performance data.

    Analyzes recent task outcomes, identifies what prompting strategies
    led to higher quality outputs, and proposes prompt improvements.
    """

    def __init__(self, empire_id: str = "", min_tasks_for_evolution: int = 5):
        self.empire_id = empire_id
        self.min_tasks = min_tasks_for_evolution

    def should_evolve(self, lieutenant_id: str) -> bool:
        """Check if a lieutenant has enough task history to evolve."""
        try:
            from db.engine import get_session
            from db.repositories.task import TaskRepository
            session = get_session()
            repo = TaskRepository(session)
            tasks = repo.get_by_lieutenant(lieutenant_id, limit=self.min_tasks)
            return len(tasks) >= self.min_tasks
        except Exception:
            return False

    def evolve_prompt(self, lieutenant_id: str) -> PromptEvolution | None:
        """Analyze performance and propose prompt improvements.

        Args:
            lieutenant_id: Lieutenant to evolve.

        Returns:
            PromptEvolution with proposed changes, or None.
        """
        try:
            from db.engine import get_session
            from db.repositories.lieutenant import LieutenantRepository
            from db.repositories.task import TaskRepository

            session = get_session()
            lt_repo = LieutenantRepository(session)
            task_repo = TaskRepository(session)

            lt = lt_repo.get(lieutenant_id)
            if not lt:
                return None

            # Get recent task performance
            tasks = task_repo.get_by_lieutenant(lieutenant_id, limit=20)
            if len(tasks) < self.min_tasks:
                return None

            # Analyze performance
            completed = [t for t in tasks if t.status == "completed"]
            failed = [t for t in tasks if t.status == "failed"]
            qualities = [t.quality_score for t in completed if t.quality_score]
            avg_quality = sum(qualities) / len(qualities) if qualities else 0

            # Get current persona
            persona = lt.persona_json or {}
            current_prompt = persona.get("system_prompt_template", "")

            # Get learnings from memory
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)
            lessons = mm.recall(
                query="lesson improvement quality",
                memory_types=["experiential"],
                lieutenant_id=lieutenant_id,
                limit=5,
            )
            lesson_text = "\n".join(f"- {l.get('content', '')[:200]}" for l in lessons)

            # Get failure patterns
            failure_text = ""
            if failed:
                failure_text = "\n".join(
                    f"- Failed: {t.title[:80]} (error: {t.last_error[:100] if t.last_error else 'quality too low'})"
                    for t in failed[:5]
                )

            # Ask LLM to propose prompt improvements
            from llm.base import LLMRequest, LLMMessage
            from llm.router import ModelRouter, TaskMetadata

            router = ModelRouter()
            prompt = f"""Analyze this AI agent's performance and propose improvements to its system prompt.

## Current System Prompt
{current_prompt[:2000] if current_prompt else 'No custom prompt — using default persona.'}

## Performance Data
- Tasks completed: {len(completed)}
- Tasks failed: {len(failed)}
- Average quality score: {avg_quality:.2f}
- Domain: {lt.domain}
- Specializations: {lt.specializations_json}

## Recent Lessons Learned
{lesson_text if lesson_text else 'No lessons recorded yet.'}

## Recent Failures
{failure_text if failure_text else 'No failures recorded.'}

## Instructions
Propose an improved system prompt that:
1. Keeps the core identity and domain expertise
2. Addresses failure patterns
3. Incorporates lessons learned
4. Adds specific techniques that lead to higher quality output
5. Is concise but thorough

Respond as JSON:
{{
    "proposed_prompt": "The improved system prompt...",
    "reasoning": "Why these changes will improve performance...",
    "confidence": 0.0-1.0
}}
"""
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                system_prompt="You are an AI prompt engineer. Analyze performance data and propose concrete improvements.",
                temperature=0.4,
                max_tokens=2000,
            )
            response = router.execute(request, TaskMetadata(task_type="analysis", complexity="complex"))

            # Parse response
            from llm.schemas import safe_json_loads
            data = safe_json_loads(response.content)

            if not data.get("proposed_prompt"):
                return None

            evolution = PromptEvolution(
                lieutenant_id=lieutenant_id,
                lieutenant_name=lt.name,
                current_prompt=current_prompt,
                proposed_prompt=data["proposed_prompt"],
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
                based_on_tasks=len(tasks),
                avg_quality_before=avg_quality,
            )

            logger.info(
                "Prompt evolution proposed for %s (confidence=%.2f, based on %d tasks)",
                lt.name, evolution.confidence, len(tasks),
            )

            return evolution

        except Exception as e:
            logger.error("Prompt evolution failed for %s: %s", lieutenant_id, e)
            return None

    def apply_evolution(self, evolution: PromptEvolution) -> bool:
        """Apply a prompt evolution to a lieutenant.

        Updates the lieutenant's persona with the new system prompt.

        Args:
            evolution: The proposed evolution.

        Returns:
            True if applied successfully.
        """
        if evolution.confidence < 0.4:
            logger.warning("Skipping low-confidence evolution (%.2f)", evolution.confidence)
            return False

        try:
            from db.engine import session_scope
            from db.models import Lieutenant

            with session_scope() as session:
                lt = session.get(Lieutenant, evolution.lieutenant_id)
                if not lt:
                    return False

                # Update persona
                persona = dict(lt.persona_json or {})
                persona["system_prompt_template"] = evolution.proposed_prompt
                persona["_prompt_evolution_history"] = persona.get("_prompt_evolution_history", [])
                persona["_prompt_evolution_history"].append({
                    "previous_prompt": evolution.current_prompt[:500],
                    "reasoning": evolution.reasoning[:300],
                    "confidence": evolution.confidence,
                    "avg_quality_before": evolution.avg_quality_before,
                    "based_on_tasks": evolution.based_on_tasks,
                })
                lt.persona_json = persona

            # Store in memory
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)
            mm.store(
                content=(
                    f"[Prompt Evolution] {evolution.lieutenant_name}\n"
                    f"Reasoning: {evolution.reasoning[:500]}\n"
                    f"Confidence: {evolution.confidence:.2f}\n"
                    f"Based on {evolution.based_on_tasks} tasks (avg quality: {evolution.avg_quality_before:.2f})"
                ),
                memory_type="design",
                lieutenant_id=evolution.lieutenant_id,
                title=f"Prompt evolved: {evolution.lieutenant_name}",
                category="prompt_evolution",
                importance=0.8,
                tags=["evolution", "prompt", "self_improvement"],
                source_type="evolution",
            )

            logger.info("Applied prompt evolution to %s", evolution.lieutenant_name)
            return True

        except Exception as e:
            logger.error("Failed to apply evolution: %s", e)
            return False

    def evolve_all(self) -> list[PromptEvolution]:
        """Evolve prompts for all eligible lieutenants.

        Returns:
            List of applied evolutions.
        """
        try:
            from db.engine import get_session
            from db.repositories.lieutenant import LieutenantRepository
            session = get_session()
            repo = LieutenantRepository(session)
            lieutenants = repo.get_by_empire(self.empire_id, status="active")
        except Exception:
            return []

        applied = []
        for lt in lieutenants:
            if self.should_evolve(lt.id):
                evolution = self.evolve_prompt(lt.id)
                if evolution and evolution.confidence >= 0.5:
                    if self.apply_evolution(evolution):
                        applied.append(evolution)

        return applied

    def extract_strategies(self, lieutenant_id: str) -> list[StrategyLearning]:
        """Extract learned strategies from task performance.

        Identifies patterns in what makes tasks succeed or fail.

        Returns:
            List of strategy learnings.
        """
        strategies = []

        try:
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)

            # Get experiential memories about outcomes
            outcomes = mm.recall(
                query="success failure quality outcome",
                memory_types=["experiential"],
                lieutenant_id=lieutenant_id,
                limit=10,
            )

            # Cluster by success/failure
            successes = [o for o in outcomes if "SUCCESS" in o.get("content", "")]
            failures = [o for o in outcomes if "FAIL" in o.get("content", "")]

            if successes:
                strategies.append(StrategyLearning(
                    strategy="Approaches that led to successful outcomes",
                    evidence="\n".join(o.get("content", "")[:200] for o in successes[:3]),
                    effectiveness=0.7,
                    source_tasks=len(successes),
                ))

            if failures:
                strategies.append(StrategyLearning(
                    strategy="Patterns in failed tasks to avoid",
                    evidence="\n".join(o.get("content", "")[:200] for o in failures[:3]),
                    effectiveness=0.3,
                    source_tasks=len(failures),
                ))

        except Exception as e:
            logger.debug("Strategy extraction failed: %s", e)

        return strategies
