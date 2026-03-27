"""Research strategy tracking and selection.

Tracks which research strategies (gap_fill, deep_dive, trend_watch, etc.)
produce the best results per domain, and uses this history to pick the
best strategy for future research questions.

This is the "self-improving" component — the system learns what works.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

STRATEGIES = ["gap_fill", "deep_dive", "trend_watch", "verification", "general"]

DEFAULT_PRIORS: dict[str, float] = {
    "gap_fill": 0.6,
    "deep_dive": 0.5,
    "trend_watch": 0.55,
    "verification": 0.4,
    "general": 0.3,
}


@dataclass
class StrategyRecord:
    """Historical record of a strategy's performance."""

    strategy: str = ""
    domain: str = ""
    efficiency: float = 0.0
    findings: int = 0
    cost_usd: float = 0.0
    recorded_at: str = ""


@dataclass
class ResearchStrategy:
    """A research strategy with its historical performance."""

    name: str = ""
    description: str = ""
    avg_efficiency: float = 0.0
    total_findings: int = 0
    total_runs: int = 0
    total_cost_usd: float = 0.0

    @property
    def cost_per_finding(self) -> float:
        if self.total_findings == 0:
            return float("inf")
        return self.total_cost_usd / self.total_findings


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class StrategyTracker:
    """Tracks strategy effectiveness and picks the best one for each domain.

    Uses a simple Thompson-sampling-inspired approach:
    - Maintains a running average efficiency per (strategy, domain).
    - Picks strategies proportionally to their historical efficiency.
    - Adds exploration noise to avoid getting stuck.
    """

    EXPLORATION_RATE = 0.20

    def __init__(self, empire_id: str):
        self.empire_id = empire_id

    def pick_strategy(self, domain: str) -> str:
        """Pick the best research strategy for a domain."""
        if random.random() < self.EXPLORATION_RATE:
            return random.choice(STRATEGIES)

        scores = self._get_strategy_scores(domain)

        if not scores:
            return self._weighted_choice(DEFAULT_PRIORS)

        return self._weighted_choice(scores)

    def record_outcome(
        self,
        strategy: str,
        domain: str,
        efficiency: float,
        findings: int,
        cost_usd: float,
    ) -> None:
        """Record the outcome of a research step for future strategy selection."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.empire_id)

        mm.store(
            content=(
                f"Research strategy outcome: {strategy} in {domain}\n"
                f"Efficiency: {efficiency:.3f} (novel findings per source)\n"
                f"Findings: {findings}, Cost: ${cost_usd:.4f}"
            ),
            memory_type="experiential",
            title=f"Strategy: {strategy}/{domain} — eff={efficiency:.2f}",
            category="research_strategy",
            importance=0.4,
            tags=["research_strategy", strategy, domain],
            source_type="autonomous",
        )

    def get_strategy_stats(self) -> dict[str, ResearchStrategy]:
        """Get aggregated stats for all strategies."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.empire_id)
        memories = mm.recall(
            query="Research strategy outcome",
            memory_types=["experiential"],
            limit=100,
        )

        stats: dict[str, ResearchStrategy] = {}
        for name in STRATEGIES:
            stats[name] = ResearchStrategy(name=name)

        for mem in memories:
            content = mem.get("content", "")
            for name in STRATEGIES:
                if f"Research strategy outcome: {name}" in content:
                    s = stats[name]
                    s.total_runs += 1
                    try:
                        eff_line = [line for line in content.split("\n") if "Efficiency:" in line]
                        if eff_line:
                            eff_str = eff_line[0].split("Efficiency:")[1].split("(")[0].strip()
                            s.avg_efficiency = (
                                (s.avg_efficiency * (s.total_runs - 1) + float(eff_str))
                                / s.total_runs
                            )
                        find_line = [line for line in content.split("\n") if "Findings:" in line]
                        if find_line:
                            parts = find_line[0].split("Findings:")[1].split(",")
                            s.total_findings += int(parts[0].strip())
                            if len(parts) > 1 and "Cost:" in parts[1]:
                                cost_str = parts[1].split("$")[1].strip()
                                s.total_cost_usd += float(cost_str)
                    except (ValueError, IndexError):
                        pass
                    break

        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_strategy_scores(self, domain: str) -> dict[str, float]:
        """Get efficiency scores for each strategy in this domain."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.empire_id)
        memories = mm.recall(
            query=f"Research strategy outcome {domain}",
            memory_types=["experiential"],
            limit=30,
        )

        scores: dict[str, list[float]] = {s: [] for s in STRATEGIES}

        for mem in memories:
            content = mem.get("content", "")
            for name in STRATEGIES:
                if f"Research strategy outcome: {name}" in content and domain in content:
                    try:
                        eff_line = [line for line in content.split("\n") if "Efficiency:" in line]
                        if eff_line:
                            eff_str = eff_line[0].split("Efficiency:")[1].split("(")[0].strip()
                            scores[name].append(float(eff_str))
                    except (ValueError, IndexError):
                        pass
                    break

        result: dict[str, float] = {}
        for name in STRATEGIES:
            if scores[name]:
                result[name] = sum(scores[name]) / len(scores[name])
            else:
                result[name] = DEFAULT_PRIORS.get(name, 0.3)

        return result

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        """Pick a strategy proportional to its weight."""
        if not weights:
            return random.choice(STRATEGIES)

        items = list(weights.items())
        total = sum(w for _, w in items)
        if total <= 0:
            return random.choice(STRATEGIES)

        r = random.random() * total
        cumulative = 0.0
        for name, weight in items:
            cumulative += weight
            if r <= cumulative:
                return name

        return items[-1][0]
