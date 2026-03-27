"""Multi-lieutenant debate engine — facilitates structured debate between lieutenants."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DebateStrategy(str, Enum):
    """Strategy for running a debate."""
    ADVERSARIAL = "adversarial"       # Lieutenants take opposing sides
    COLLABORATIVE = "collaborative"   # Build on each other's ideas
    DEVILS_ADVOCATE = "devils_advocate"  # One plays devil's advocate
    ROUND_ROBIN = "round_robin"       # Structured turn-taking


class DebatePhase(str, Enum):
    """Phase of the debate."""
    OPENING = "opening"
    ARGUMENTS = "arguments"
    REBUTTALS = "rebuttals"
    CLOSING = "closing"
    SCORING = "scoring"
    CONSENSUS = "consensus"


@dataclass
class Argument:
    """A single argument in a debate."""
    lieutenant_id: str
    lieutenant_name: str = ""
    position: str = ""
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.7
    counterpoints: list[str] = field(default_factory=list)
    round_number: int = 1
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Rebuttal:
    """A rebuttal to an argument."""
    lieutenant_id: str
    lieutenant_name: str = ""
    target_argument_index: int = 0
    counter_reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    strength: float = 0.5
    concessions: list[str] = field(default_factory=list)
    round_number: int = 1


@dataclass
class ScoredArgument:
    """An argument with quality scores."""
    argument: Argument
    logic_score: float = 0.5
    evidence_score: float = 0.5
    novelty_score: float = 0.5
    persuasion_score: float = 0.5
    overall_score: float = 0.5

    @property
    def weighted_score(self) -> float:
        return (
            self.logic_score * 0.35 +
            self.evidence_score * 0.25 +
            self.novelty_score * 0.15 +
            self.persuasion_score * 0.25
        )


@dataclass
class ConsensusCheck:
    """Result of checking for consensus."""
    reached: bool = False
    agreement_level: float = 0.0
    consensus_position: str = ""
    holdouts: list[str] = field(default_factory=list)
    compromise_possible: bool = False
    compromise_suggestion: str = ""


@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    arguments: list[Argument] = field(default_factory=list)
    rebuttals: list[Rebuttal] = field(default_factory=list)
    scored_arguments: list[ScoredArgument] = field(default_factory=list)
    summary: str = ""
    phase: str = "arguments"
    cost_usd: float = 0.0


@dataclass
class Debate:
    """Full debate record."""
    topic: str
    strategy: DebateStrategy = DebateStrategy.COLLABORATIVE
    participants: list[dict] = field(default_factory=list)
    rounds: list[DebateRound] = field(default_factory=list)
    status: str = "active"  # active, completed, consensus_reached
    consensus: Optional[ConsensusCheck] = None
    final_summary: str = ""
    total_cost: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class DebateSummary:
    """Summary of a completed debate."""
    topic: str = ""
    rounds_count: int = 0
    participant_count: int = 0
    consensus_reached: bool = False
    winning_position: str = ""
    key_arguments: list[str] = field(default_factory=list)
    areas_of_agreement: list[str] = field(default_factory=list)
    areas_of_disagreement: list[str] = field(default_factory=list)
    total_cost: float = 0.0


class DebateEngine:
    """Facilitates structured debate between lieutenants.

    Manages multi-round debates with arguments, rebuttals, scoring,
    and consensus detection. Supports multiple debate strategies.
    """

    def __init__(self, empire_id: str = "", max_rounds: int = 3):
        self.empire_id = empire_id
        self.max_rounds = max_rounds
        self._router = None

    def _get_router(self):
        if self._router is None:
            from llm.router import ModelRouter
            self._router = ModelRouter()
        return self._router

    def initiate_debate(
        self,
        topic: str,
        participants: list[dict],
        strategy: DebateStrategy = DebateStrategy.COLLABORATIVE,
        context: str = "",
    ) -> Debate:
        """Start a new debate.

        Args:
            topic: Debate topic.
            participants: List of {id, name, domain} dicts.
            strategy: Debate strategy.
            context: Additional context.

        Returns:
            Debate instance.
        """
        debate = Debate(
            topic=topic,
            strategy=strategy,
            participants=participants,
        )

        start_time = time.time()

        # Run debate rounds
        for round_num in range(1, self.max_rounds + 1):
            round_result = self.run_round(debate, round_num, context)
            debate.rounds.append(round_result)
            debate.total_cost += round_result.cost_usd

            # Check for consensus after each round
            if round_num >= 2:
                consensus = self.check_consensus(debate)
                if consensus.reached:
                    debate.consensus = consensus
                    debate.status = "consensus_reached"
                    break

        if debate.status != "consensus_reached":
            debate.status = "completed"
            debate.consensus = self.check_consensus(debate)

        debate.duration_seconds = time.time() - start_time
        debate.final_summary = self._generate_debate_summary(debate)

        return debate

    def run_round(
        self,
        debate: Debate,
        round_number: int,
        context: str = "",
    ) -> DebateRound:
        """Run a single debate round.

        Args:
            debate: The debate.
            round_number: Round number.
            context: Additional context.

        Returns:
            DebateRound.
        """
        debate_round = DebateRound(round_number=round_number)

        # Phase 1: Collect arguments
        arguments = self._collect_arguments(debate, round_number, context)
        debate_round.arguments = arguments

        # Phase 2: Collect rebuttals (from round 2 onwards)
        if round_number >= 2:
            rebuttals = self._collect_rebuttals(debate, arguments, round_number)
            debate_round.rebuttals = rebuttals

        # Phase 3: Score arguments
        scored = self._score_arguments(arguments)
        debate_round.scored_arguments = scored

        # Generate round summary
        debate_round.summary = self._summarize_round(debate_round)

        return debate_round

    def _collect_arguments(
        self,
        debate: Debate,
        round_number: int,
        context: str,
    ) -> list[Argument]:
        """Collect arguments from each participant."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()
        arguments = []

        # Build previous rounds context
        prev_context = ""
        if debate.rounds:
            prev_args = []
            for r in debate.rounds:
                for arg in r.arguments:
                    prev_args.append(f"[{arg.lieutenant_name}] {arg.position}: {arg.reasoning[:200]}")
            prev_context = "\n".join(prev_args[-10:])

        for participant in debate.participants:
            prompt = f"""You are participating in a structured debate.

Topic: {debate.topic}
Strategy: {debate.strategy.value}
Round: {round_number}/{self.max_rounds}
Your Role: {participant.get('name', 'Participant')} ({participant.get('domain', 'general')})

{f"Context: {context}" if context else ""}
{f"Previous arguments:{chr(10)}{prev_context}" if prev_context else ""}

Provide your argument:
1. State your position clearly
2. Give detailed reasoning
3. Provide evidence or examples
4. Anticipate counterarguments
5. Rate your confidence (0.0-1.0)

Respond as JSON:
{{
    "position": "Your clear position",
    "reasoning": "Detailed reasoning...",
    "evidence": ["evidence point 1", "evidence point 2"],
    "counterpoints": ["anticipated counter 1"],
    "confidence": 0.7
}}
"""
            try:
                request = LLMRequest(
                    messages=[LLMMessage.user(prompt)],
                    system_prompt=f"You are {participant.get('name', 'an expert')} specializing in {participant.get('domain', 'general analysis')}.",
                    temperature=0.6,
                    max_tokens=1500,
                )
                response = router.execute(request, TaskMetadata(task_type="analysis", complexity="moderate"))

                try:
                    data = json.loads(response.content)
                except json.JSONDecodeError:
                    from llm.schemas import _find_json_object
                    json_str = _find_json_object(response.content)
                    data = json.loads(json_str) if json_str else {}

                arguments.append(Argument(
                    lieutenant_id=participant.get("id", ""),
                    lieutenant_name=participant.get("name", ""),
                    position=data.get("position", ""),
                    reasoning=data.get("reasoning", response.content[:500]),
                    evidence=data.get("evidence", []),
                    confidence=float(data.get("confidence", 0.7)),
                    counterpoints=data.get("counterpoints", []),
                    round_number=round_number,
                ))

            except Exception as e:
                logger.warning("Failed to collect argument from %s: %s", participant.get("name"), e)
                arguments.append(Argument(
                    lieutenant_id=participant.get("id", ""),
                    lieutenant_name=participant.get("name", ""),
                    position=f"Error: {e}",
                    round_number=round_number,
                ))

        return arguments

    def _collect_rebuttals(
        self,
        debate: Debate,
        current_arguments: list[Argument],
        round_number: int,
    ) -> list[Rebuttal]:
        """Collect rebuttals from participants to other arguments."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()
        rebuttals = []

        for i, participant in enumerate(debate.participants):
            # Get arguments from others to rebut
            other_args = [a for a in current_arguments if a.lieutenant_id != participant.get("id")]
            if not other_args:
                continue

            args_text = "\n\n".join(
                f"**{a.lieutenant_name}**: {a.position}\nReasoning: {a.reasoning[:300]}"
                for a in other_args
            )

            prompt = f"""Review these arguments and provide rebuttals.

Topic: {debate.topic}

Arguments to address:
{args_text}

For each argument:
1. Identify the weakest point
2. Provide a counter-argument
3. Note any concessions (where they're right)

Respond as JSON:
{{
    "rebuttals": [
        {{
            "target_index": 0,
            "counter_reasoning": "...",
            "evidence": ["..."],
            "strength": 0.7,
            "concessions": ["what they got right"]
        }}
    ]
}}
"""
            try:
                request = LLMRequest(
                    messages=[LLMMessage.user(prompt)],
                    system_prompt=f"You are {participant.get('name', 'an expert')}. Be fair but rigorous.",
                    temperature=0.5,
                    max_tokens=1500,
                )
                response = router.execute(request, TaskMetadata(task_type="analysis"))

                try:
                    data = json.loads(response.content)
                except json.JSONDecodeError:
                    from llm.schemas import _find_json_object
                    json_str = _find_json_object(response.content)
                    data = json.loads(json_str) if json_str else {}

                for reb_data in data.get("rebuttals", []):
                    rebuttals.append(Rebuttal(
                        lieutenant_id=participant.get("id", ""),
                        lieutenant_name=participant.get("name", ""),
                        target_argument_index=reb_data.get("target_index", 0),
                        counter_reasoning=reb_data.get("counter_reasoning", ""),
                        evidence=reb_data.get("evidence", []),
                        strength=float(reb_data.get("strength", 0.5)),
                        concessions=reb_data.get("concessions", []),
                        round_number=round_number,
                    ))

            except Exception as e:
                logger.warning("Failed to collect rebuttals from %s: %s", participant.get("name"), e)

        return rebuttals

    def _score_arguments(self, arguments: list[Argument]) -> list[ScoredArgument]:
        """Score arguments — tries LLM scoring, falls back to heuristic."""
        try:
            return self._score_arguments_llm(arguments)
        except Exception as e:
            logger.debug("LLM argument scoring failed, using heuristic fallback: %s", e)
            return self._score_arguments_heuristic(arguments)

    def _score_arguments_llm(self, arguments: list[Argument]) -> list[ScoredArgument]:
        """Score arguments using an LLM for nuanced evaluation."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        args_text = "\n\n".join(
            f"### Argument {i+1} by {a.lieutenant_name}\n"
            f"Position: {a.position}\n"
            f"Reasoning: {a.reasoning[:400]}\n"
            f"Evidence: {a.evidence}\n"
            f"Counterpoints anticipated: {a.counterpoints}"
            for i, a in enumerate(arguments)
        )

        prompt = f"""Score each argument on 4 dimensions (0.0-1.0 each):
- logic_score: How sound is the reasoning?
- evidence_score: How well-supported with evidence?
- novelty_score: How original or insightful?
- persuasion_score: How convincing overall?

Arguments:
{args_text}

Return valid JSON only:
[
  {{"argument_index": 0, "logic_score": 0.0, "evidence_score": 0.0, "novelty_score": 0.0, "persuasion_score": 0.0}}
]
"""
        request = LLMRequest(
            messages=[LLMMessage.user(prompt)],
            system_prompt="You are a debate judge. Score arguments fairly and precisely. Return valid JSON only.",
            temperature=0.2,
            max_tokens=800,
        )
        response = router.execute(request, TaskMetadata(task_type="analysis", complexity="moderate"))

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            from llm.schemas import _find_json_object
            # Try finding JSON array
            text = response.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                raise ValueError("Could not parse scoring response")

        scored = []
        for i, arg in enumerate(arguments):
            score_data = next((d for d in data if d.get("argument_index") == i), {})
            logic = float(score_data.get("logic_score", 0.5))
            evidence = float(score_data.get("evidence_score", 0.5))
            novelty = float(score_data.get("novelty_score", 0.5))
            persuasion = float(score_data.get("persuasion_score", 0.5))

            scored.append(ScoredArgument(
                argument=arg,
                logic_score=logic,
                evidence_score=evidence,
                novelty_score=novelty,
                persuasion_score=persuasion,
                overall_score=(logic + evidence + novelty + persuasion) / 4,
            ))

        return scored

    def _score_arguments_heuristic(self, arguments: list[Argument]) -> list[ScoredArgument]:
        """Score arguments using heuristics (fallback)."""
        scored = []
        for arg in arguments:
            reasoning_len = len(arg.reasoning)
            evidence_count = len(arg.evidence)
            counterpoint_count = len(arg.counterpoints)

            logic_score = min(1.0, reasoning_len / 500)
            evidence_score = min(1.0, evidence_count / 3)
            novelty_score = 0.5
            persuasion_score = arg.confidence * 0.7 + min(1.0, counterpoint_count / 2) * 0.3

            scored_arg = ScoredArgument(
                argument=arg,
                logic_score=logic_score,
                evidence_score=evidence_score,
                novelty_score=novelty_score,
                persuasion_score=persuasion_score,
                overall_score=(logic_score + evidence_score + novelty_score + persuasion_score) / 4,
            )
            scored.append(scored_arg)

        return scored

    def check_consensus(self, debate: Debate) -> ConsensusCheck:
        """Check if consensus has been reached — tries LLM, falls back to heuristic."""
        if not debate.rounds:
            return ConsensusCheck()

        latest_round = debate.rounds[-1]
        positions = [a.position for a in latest_round.arguments if a.position]

        if not positions:
            return ConsensusCheck()

        try:
            return self._check_consensus_llm(latest_round, positions)
        except Exception as e:
            logger.debug("LLM consensus check failed, using heuristic: %s", e)
            return self._check_consensus_heuristic(latest_round, positions)

    def _check_consensus_llm(self, latest_round: DebateRound, positions: list[str]) -> ConsensusCheck:
        """Use LLM to evaluate whether positions agree on substance."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        router = self._get_router()

        positions_text = "\n\n".join(
            f"**{a.lieutenant_name}**: {a.position}\nReasoning: {a.reasoning[:200]}"
            for a in latest_round.arguments if a.position
        )

        prompt = f"""Analyze these debate positions for consensus.

{positions_text}

Evaluate:
1. Do the positions agree on the core substance (even if wording differs)?
2. What is the level of agreement (0.0-1.0)?
3. Who (if anyone) is a holdout with a fundamentally different view?
4. Is a compromise possible?

Return valid JSON:
{{
  "reached": true/false,
  "agreement_level": 0.0-1.0,
  "consensus_position": "summary of agreed position (if reached)",
  "holdouts": ["name1"],
  "compromise_possible": true/false,
  "compromise_suggestion": "how to bridge remaining gaps"
}}
"""
        request = LLMRequest(
            messages=[LLMMessage.user(prompt)],
            system_prompt="You are a neutral debate moderator assessing consensus. Return valid JSON only.",
            temperature=0.2,
            max_tokens=600,
        )
        response = router.execute(request, TaskMetadata(task_type="analysis", complexity="simple"))

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                raise ValueError("Could not parse consensus response")

        return ConsensusCheck(
            reached=bool(data.get("reached", False)),
            agreement_level=float(data.get("agreement_level", 0.0)),
            consensus_position=data.get("consensus_position", ""),
            holdouts=data.get("holdouts", []),
            compromise_possible=bool(data.get("compromise_possible", False)),
            compromise_suggestion=data.get("compromise_suggestion", ""),
        )

    def _check_consensus_heuristic(self, latest_round: DebateRound, positions: list[str]) -> ConsensusCheck:
        """Check consensus using word-overlap heuristic (fallback)."""
        position_words = [set(p.lower().split()) for p in positions]

        total_overlap = 0
        comparisons = 0
        for i in range(len(position_words)):
            for j in range(i + 1, len(position_words)):
                overlap = len(position_words[i] & position_words[j])
                union = len(position_words[i] | position_words[j])
                if union > 0:
                    total_overlap += overlap / union
                comparisons += 1

        agreement_level = total_overlap / max(comparisons, 1)
        avg_confidence = sum(a.confidence for a in latest_round.arguments) / max(len(latest_round.arguments), 1)

        reached = agreement_level > 0.4 and avg_confidence > 0.6

        holdouts = []
        for arg in latest_round.arguments:
            if arg.confidence < 0.5:
                holdouts.append(arg.lieutenant_name)

        return ConsensusCheck(
            reached=reached,
            agreement_level=agreement_level,
            consensus_position=positions[0] if reached else "",
            holdouts=holdouts,
            compromise_possible=agreement_level > 0.3,
            compromise_suggestion="Consider combining the strongest elements of each position" if not reached else "",
        )

    def force_resolution(self, debate: Debate) -> str:
        """Force a resolution when consensus can't be reached.

        Uses the highest-scored argument as the resolution.
        """
        if not debate.rounds:
            return "No arguments to resolve"

        all_scored = []
        for r in debate.rounds:
            all_scored.extend(r.scored_arguments)

        if not all_scored:
            return "No scored arguments"

        best = max(all_scored, key=lambda s: s.weighted_score)
        return (
            f"Resolution (by highest scoring argument): "
            f"{best.argument.lieutenant_name}'s position: {best.argument.position} "
            f"(score: {best.weighted_score:.2f})"
        )

    def _summarize_round(self, debate_round: DebateRound) -> str:
        """Generate a summary of a debate round."""
        parts = [f"Round {debate_round.round_number}:"]

        for arg in debate_round.arguments:
            parts.append(f"  [{arg.lieutenant_name}] {arg.position[:100]}")

        if debate_round.rebuttals:
            parts.append(f"  {len(debate_round.rebuttals)} rebuttals exchanged")

        if debate_round.scored_arguments:
            best = max(debate_round.scored_arguments, key=lambda s: s.overall_score)
            parts.append(f"  Strongest argument: {best.argument.lieutenant_name} ({best.overall_score:.2f})")

        return "\n".join(parts)

    def _generate_debate_summary(self, debate: Debate) -> str:
        """Generate a final summary of the entire debate."""
        parts = [f"Debate: {debate.topic}", f"Rounds: {len(debate.rounds)}", f"Participants: {len(debate.participants)}"]

        if debate.consensus:
            if debate.consensus.reached:
                parts.append(f"Consensus reached: {debate.consensus.consensus_position[:200]}")
            else:
                parts.append(f"No consensus (agreement level: {debate.consensus.agreement_level:.2f})")
                if debate.consensus.holdouts:
                    parts.append(f"Holdouts: {', '.join(debate.consensus.holdouts)}")

        parts.append(f"Total cost: ${debate.total_cost:.4f}")
        return "\n".join(parts)

    def get_debate_summary(self, debate: Debate) -> DebateSummary:
        """Get a structured summary of a debate."""
        key_args = []
        for r in debate.rounds:
            for scored in r.scored_arguments:
                if scored.overall_score > 0.6:
                    key_args.append(f"[{scored.argument.lieutenant_name}] {scored.argument.position[:150]}")

        return DebateSummary(
            topic=debate.topic,
            rounds_count=len(debate.rounds),
            participant_count=len(debate.participants),
            consensus_reached=debate.consensus.reached if debate.consensus else False,
            winning_position=debate.consensus.consensus_position if debate.consensus else "",
            key_arguments=key_args[:5],
            total_cost=debate.total_cost,
        )
