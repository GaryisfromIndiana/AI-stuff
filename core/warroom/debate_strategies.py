"""Debate strategy implementations — different modes of structured discussion."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.warroom.debate import Argument, Rebuttal, DebateRound, DebateStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a debate strategy."""
    name: str = ""
    max_rounds: int = 3
    require_evidence: bool = True
    allow_position_change: bool = False
    require_rebuttals: bool = True
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "logic": 0.35, "evidence": 0.25, "novelty": 0.15, "persuasion": 0.25,
    })
    consensus_threshold: float = 0.6
    moderator_model: str = "claude-sonnet-4"


class BaseStrategy(ABC):
    """Abstract base for debate strategies."""

    strategy_type: DebateStrategy
    config: StrategyConfig

    @abstractmethod
    def prepare_round_prompt(
        self,
        participant: dict,
        round_number: int,
        topic: str,
        previous_arguments: list[Argument],
        previous_rebuttals: list[Rebuttal],
    ) -> str:
        """Build the prompt for a participant in a round."""
        ...

    @abstractmethod
    def should_continue(self, rounds: list[DebateRound]) -> bool:
        """Check if debate should continue to next round."""
        ...


class AdversarialStrategy(BaseStrategy):
    """Adversarial debate — participants take opposing sides.

    Promotes thorough analysis by forcing different perspectives.
    Each participant is assigned a position and must argue for it,
    even if they personally disagree.
    """

    strategy_type = DebateStrategy.ADVERSARIAL

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(name="adversarial", max_rounds=3)

    def prepare_round_prompt(
        self,
        participant: dict,
        round_number: int,
        topic: str,
        previous_arguments: list[Argument],
        previous_rebuttals: list[Rebuttal],
    ) -> str:
        position_idx = participant.get("position_index", 0)
        position = "FOR" if position_idx % 2 == 0 else "AGAINST"

        prev_text = ""
        if previous_arguments:
            prev_text = "\nPrevious arguments:\n" + "\n".join(
                f"[{a.lieutenant_name} - {a.position}]: {a.reasoning[:200]}"
                for a in previous_arguments[-5:]
            )

        return f"""ADVERSARIAL DEBATE — Round {round_number}

Topic: {topic}
Your assigned position: **{position}**

You MUST argue {position} this topic, regardless of your personal views.
This ensures we explore all angles thoroughly.

{prev_text}

Provide:
1. Your strongest argument {position} the topic
2. Evidence supporting your position
3. Preemptive rebuttals to likely counterarguments
4. Confidence in your position (0-1)

Be persuasive and thorough. Treat this as if you genuinely hold this position.
"""

    def should_continue(self, rounds: list[DebateRound]) -> bool:
        if len(rounds) >= self.config.max_rounds:
            return False
        # In adversarial mode, always run max rounds
        return True


class CollaborativeStrategy(BaseStrategy):
    """Collaborative debate — build on each other's ideas.

    Focuses on constructing the best possible answer together.
    Participants contribute their unique expertise and build on
    what others have said.
    """

    strategy_type = DebateStrategy.COLLABORATIVE

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(name="collaborative", max_rounds=3, require_rebuttals=False)

    def prepare_round_prompt(
        self,
        participant: dict,
        round_number: int,
        topic: str,
        previous_arguments: list[Argument],
        previous_rebuttals: list[Rebuttal],
    ) -> str:
        domain = participant.get("domain", "general")

        prev_text = ""
        if previous_arguments:
            prev_text = "\nOther participants have contributed:\n" + "\n".join(
                f"[{a.lieutenant_name}]: {a.reasoning[:200]}"
                for a in previous_arguments[-5:]
            )

        return f"""COLLABORATIVE DISCUSSION — Round {round_number}

Topic: {topic}
Your expertise: {domain}

{prev_text}

Build on what others have said. Contribute your unique perspective from {domain}.

Provide:
1. Your contribution from your domain expertise
2. How it connects to or builds on previous contributions
3. Any additional considerations from your perspective
4. Key insights others may have missed

Focus on adding value, not contradicting. We're building the best answer together.
"""

    def should_continue(self, rounds: list[DebateRound]) -> bool:
        if len(rounds) >= self.config.max_rounds:
            return False
        # Stop early if ideas are converging (high overlap)
        if len(rounds) >= 2:
            recent = rounds[-1]
            if recent.scored_arguments:
                avg_score = sum(s.overall_score for s in recent.scored_arguments) / len(recent.scored_arguments)
                if avg_score > 0.8:
                    return False  # Good enough
        return True


class DevilsAdvocateStrategy(BaseStrategy):
    """Devil's Advocate — one participant challenges the group.

    One lieutenant is assigned as the devil's advocate and must
    find flaws in every argument the others make.
    """

    strategy_type = DebateStrategy.DEVILS_ADVOCATE

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(name="devils_advocate", max_rounds=3)

    def prepare_round_prompt(
        self,
        participant: dict,
        round_number: int,
        topic: str,
        previous_arguments: list[Argument],
        previous_rebuttals: list[Rebuttal],
    ) -> str:
        is_advocate = participant.get("is_devils_advocate", False)
        domain = participant.get("domain", "general")

        prev_text = ""
        if previous_arguments:
            prev_text = "\nPrevious contributions:\n" + "\n".join(
                f"[{a.lieutenant_name}]: {a.reasoning[:200]}"
                for a in previous_arguments[-5:]
            )

        if is_advocate:
            return f"""DEVIL'S ADVOCATE MODE — Round {round_number}

Topic: {topic}
Your role: **DEVIL'S ADVOCATE**

{prev_text}

Your job is to find flaws, challenge assumptions, and stress-test every argument.
You are NOT trying to be contrarian for its own sake — you're trying to make
the group's thinking more rigorous.

For each argument made:
1. Identify the weakest assumption
2. Present a credible counterargument
3. Suggest what evidence would be needed to strengthen the claim
4. Rate how vulnerable each argument is (0-1)

Be thorough but constructive. Your goal is to make the final output stronger.
"""
        else:
            return f"""COLLABORATIVE DISCUSSION (with Devil's Advocate) — Round {round_number}

Topic: {topic}
Your expertise: {domain}
Note: A Devil's Advocate is challenging arguments. Address their concerns.

{prev_text}

Provide:
1. Your position and reasoning
2. Responses to any Devil's Advocate challenges
3. Strengthened arguments with better evidence
4. Your confidence level
"""

    def should_continue(self, rounds: list[DebateRound]) -> bool:
        return len(rounds) < self.config.max_rounds


class RoundRobinStrategy(BaseStrategy):
    """Round Robin — structured turn-taking with moderation.

    Each participant speaks in turn, with a moderator summarizing
    after each round.
    """

    strategy_type = DebateStrategy.ROUND_ROBIN

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig(name="round_robin", max_rounds=3, allow_position_change=True)

    def prepare_round_prompt(
        self,
        participant: dict,
        round_number: int,
        topic: str,
        previous_arguments: list[Argument],
        previous_rebuttals: list[Rebuttal],
    ) -> str:
        domain = participant.get("domain", "general")
        turn = participant.get("turn_number", 0)

        prev_text = ""
        if previous_arguments:
            prev_text = "\nPrevious speakers:\n" + "\n".join(
                f"[{a.lieutenant_name}]: {a.reasoning[:200]}"
                for a in previous_arguments[-5:]
            )

        return f"""ROUND ROBIN DISCUSSION — Round {round_number}, Turn {turn + 1}

Topic: {topic}
Your expertise: {domain}

{prev_text}

It's your turn. You've heard what others have said.

Provide:
1. Your position (you may update it based on what you've heard)
2. What new insight you bring
3. What you agree with from previous speakers
4. What you disagree with and why
5. Your confidence level

Keep it focused and build on the discussion.
"""

    def should_continue(self, rounds: list[DebateRound]) -> bool:
        return len(rounds) < self.config.max_rounds


# Strategy registry
STRATEGY_REGISTRY: dict[DebateStrategy, type[BaseStrategy]] = {
    DebateStrategy.ADVERSARIAL: AdversarialStrategy,
    DebateStrategy.COLLABORATIVE: CollaborativeStrategy,
    DebateStrategy.DEVILS_ADVOCATE: DevilsAdvocateStrategy,
    DebateStrategy.ROUND_ROBIN: RoundRobinStrategy,
}


def get_strategy(
    strategy_type: DebateStrategy,
    config: StrategyConfig | None = None,
) -> BaseStrategy:
    """Get a strategy implementation by type.

    Args:
        strategy_type: The strategy type.
        config: Optional strategy config.

    Returns:
        Strategy instance.
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_type, CollaborativeStrategy)
    return strategy_class(config)


def list_strategies() -> list[dict]:
    """List available debate strategies."""
    return [
        {"key": s.value, "name": s.value.replace("_", " ").title()}
        for s in DebateStrategy
    ]
