"""Ground-truth validation — measures whether Empire actually knows things.

The system claims to track AI developments. This module checks by querying
for known facts and measuring coverage (did we catch it?) and accuracy
(is what we stored correct?).

Usage:
    validator = GroundTruthValidator(empire_id)
    report = validator.run_validation()
    print(f"Coverage: {report.coverage:.0%}")
    print(f"Accuracy: {report.accuracy:.0%}")

Ground-truth facts are defined in GROUND_TRUTH below. They should be
updated periodically with known-true facts that the system should have
discovered through its research pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthFact:
    """A fact the system should know."""
    topic: str          # What to search for
    expected: str       # What a correct answer contains (substring match)
    domain: str         # Which domain this belongs to
    date: str = ""      # When this became true (for temporal validation)
    importance: str = "high"  # high, medium, low


@dataclass
class ValidationResult:
    """Result of checking one ground-truth fact."""
    fact: GroundTruthFact
    found: bool = False           # Did recall find anything related?
    accurate: bool = False        # Was the found content correct?
    memory_id: str = ""           # ID of the matching memory
    matched_content: str = ""     # What the system actually stored
    confidence: float = 0.0       # System's confidence in the match


@dataclass
class ValidationReport:
    """Overall validation report."""
    timestamp: str = ""
    total_facts: int = 0
    found: int = 0
    accurate: int = 0
    coverage: float = 0.0     # found / total
    accuracy: float = 0.0     # accurate / found (of what we found, how much is right)
    f1_score: float = 0.0     # harmonic mean of coverage and accuracy
    results: list[ValidationResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)     # Topics not found at all
    inaccurate: list[str] = field(default_factory=list)  # Topics found but wrong

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_facts": self.total_facts,
            "found": self.found,
            "accurate": self.accurate,
            "coverage": round(self.coverage, 3),
            "accuracy": round(self.accuracy, 3),
            "f1_score": round(self.f1_score, 3),
            "missing": self.missing,
            "inaccurate": self.inaccurate,
        }


# ── Ground Truth Facts ────────────────────────────────────────────────────
# These are known-true facts about AI that the system should have discovered.
# Update this list periodically with facts from the relevant time period.
#
# Format: GroundTruthFact(topic, expected_substring, domain)
# The validator searches for `topic` and checks if `expected` appears
# in the retrieved content.

GROUND_TRUTH: list[GroundTruthFact] = [
    # Models
    GroundTruthFact("Claude Sonnet 4", "anthropic", "models", "2025-05"),
    GroundTruthFact("Claude Opus 4", "anthropic", "models", "2025-05"),
    GroundTruthFact("GPT-4o", "openai", "models", "2024-05"),
    GroundTruthFact("Gemini 2.5 Pro", "google", "models", "2025-03"),
    GroundTruthFact("Llama 4", "meta", "models", "2025-04"),

    # Research
    GroundTruthFact("reasoning models", "chain of thought", "research"),
    GroundTruthFact("RLHF", "reinforcement learning from human feedback", "research"),

    # Agents
    GroundTruthFact("MCP protocol", "model context protocol", "agents", "2024-11"),
    GroundTruthFact("Claude Code", "cli", "agents", "2025-02"),

    # Industry
    GroundTruthFact("OpenAI", "sam altman", "industry"),
    GroundTruthFact("Anthropic", "dario amodei", "industry"),
]


class GroundTruthValidator:
    """Validates Empire's knowledge against known ground-truth facts."""

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def run_validation(self, facts: list[GroundTruthFact] | None = None) -> ValidationReport:
        """Run validation against ground-truth facts.

        Args:
            facts: Custom facts to validate against. Defaults to GROUND_TRUTH.

        Returns:
            ValidationReport with coverage, accuracy, and per-fact results.
        """
        facts = facts or GROUND_TRUTH
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.empire_id)
        report = ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_facts=len(facts),
        )

        for fact in facts:
            result = self._check_fact(mm, fact)
            report.results.append(result)

            if result.found:
                report.found += 1
                if result.accurate:
                    report.accurate += 1
                else:
                    report.inaccurate.append(fact.topic)
            else:
                report.missing.append(fact.topic)

        report.coverage = report.found / report.total_facts if report.total_facts > 0 else 0
        report.accuracy = report.accurate / report.found if report.found > 0 else 0

        # F1 = harmonic mean of coverage and accuracy
        if report.coverage + report.accuracy > 0:
            report.f1_score = 2 * (report.coverage * report.accuracy) / (report.coverage + report.accuracy)

        logger.info(
            "Ground-truth validation: coverage=%.0f%% (%d/%d), accuracy=%.0f%% (%d/%d), F1=%.2f",
            report.coverage * 100, report.found, report.total_facts,
            report.accuracy * 100, report.accurate, report.found,
            report.f1_score,
        )

        return report

    def _check_fact(self, mm, fact: GroundTruthFact) -> ValidationResult:
        """Check if Empire knows a single ground-truth fact."""
        result = ValidationResult(fact=fact)

        # Search memory for the topic
        memories = mm.recall(query=fact.topic, memory_types=["semantic"], limit=5)

        if not memories:
            return result

        # Check each result for the expected content
        for mem in memories:
            content = (mem.get("content", "") + " " + mem.get("title", "")).lower()
            topic_words = set(fact.topic.lower().split())
            content_words = set(content.split())

            # Must have meaningful overlap with the topic (not just any random match)
            overlap = len(topic_words & content_words) / len(topic_words) if topic_words else 0
            if overlap < 0.4:
                continue

            result.found = True
            result.memory_id = mem.get("id", "")
            result.matched_content = mem.get("content", "")[:300]
            result.confidence = mem.get("importance", 0)

            # Check accuracy — does the content contain the expected substring?
            if fact.expected.lower() in content:
                result.accurate = True
                break

        return result

    def get_coverage_by_domain(self, report: ValidationReport | None = None) -> dict[str, float]:
        """Get coverage breakdown by domain."""
        report = report or self.run_validation()
        domains: dict[str, list[bool]] = {}

        for result in report.results:
            domain = result.fact.domain
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(result.found)

        return {
            domain: sum(found) / len(found) if found else 0
            for domain, found in domains.items()
        }
