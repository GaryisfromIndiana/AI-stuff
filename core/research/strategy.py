"""Research strategy tracking and selection.

Tracks which research strategies (gap_fill, deep_dive, trend_watch, etc.)
produce the best results per domain, and uses this history to pick the
best strategy for future research questions.

This is the "self-improving" component — the system learns what works.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

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
        """Record the outcome of a research step for future strategy selection.

        Stores the numeric data in metadata_json so downstream reads don't
        have to string-parse content. Content remains human-readable for logs.
        """
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
            metadata={
                "strategy": strategy,
                "domain": domain,
                "efficiency": float(efficiency),
                "findings": int(findings),
                "cost_usd": float(cost_usd),
            },
        )

    def get_strategy_stats(self) -> dict[str, ResearchStrategy]:
        """Get aggregated stats for all strategies."""
        stats: dict[str, ResearchStrategy] = {name: ResearchStrategy(name=name) for name in STRATEGIES}

        for meta in self._load_outcomes(limit=200):
            name = meta.get("strategy", "")
            if name not in stats:
                continue
            s = stats[name]
            s.total_runs += 1
            eff = float(meta.get("efficiency", 0.0))
            s.avg_efficiency = (
                (s.avg_efficiency * (s.total_runs - 1) + eff) / s.total_runs
            )
            s.total_findings += int(meta.get("findings", 0))
            s.total_cost_usd += float(meta.get("cost_usd", 0.0))

        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_outcomes(self, limit: int = 100, domain: str | None = None) -> list[dict]:
        """Load strategy outcome metadata, filtered by domain if given."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.empire_id)
        query = f"Research strategy outcome {domain}" if domain else "Research strategy outcome"
        memories = mm.recall(query=query, memory_types=["experiential"], limit=limit)

        out: list[dict] = []
        for mem in memories:
            meta = mem.get("metadata") or {}
            if not isinstance(meta, dict):
                continue
            if "strategy" not in meta or "efficiency" not in meta:
                continue  # skip legacy string-only entries
            if domain and meta.get("domain") != domain:
                continue
            out.append(meta)
        return out

    def _get_strategy_scores(self, domain: str) -> dict[str, float]:
        """Get efficiency scores for each strategy in this domain."""
        scores: dict[str, list[float]] = {s: [] for s in STRATEGIES}

        for meta in self._load_outcomes(limit=50, domain=domain):
            name = meta.get("strategy", "")
            if name in scores:
                scores[name].append(float(meta.get("efficiency", 0.0)))

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
