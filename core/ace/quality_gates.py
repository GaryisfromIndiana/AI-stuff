"""Quality gate definitions — configurable checks that output must pass."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float = 0.0
    threshold: float = 0.0
    details: str = ""
    metadata: dict = field(default_factory=dict)


class QualityGate(ABC):
    """Abstract base class for quality gates."""

    gate_name: str = "base"

    @abstractmethod
    def check(self, content: str, context: dict | None = None) -> GateResult:
        """Run this quality gate check.

        Args:
            content: Content to check.
            context: Additional context (task requirements, scores, etc.).

        Returns:
            GateResult.
        """
        ...


class ConfidenceGate(QualityGate):
    """Checks that output meets minimum confidence score."""

    gate_name = "confidence"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def check(self, content: str, context: dict | None = None) -> GateResult:
        context = context or {}
        score = float(context.get("confidence", context.get("scores", {}).get("confidence", 0.0)))

        return GateResult(
            gate_name=self.gate_name,
            passed=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            details=f"Confidence {score:.2f} {'meets' if score >= self.threshold else 'below'} threshold {self.threshold:.2f}",
        )


class CompletenessGate(QualityGate):
    """Checks that task requirements are covered."""

    gate_name = "completeness"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check(self, content: str, context: dict | None = None) -> GateResult:
        context = context or {}
        score = float(context.get("completeness", context.get("scores", {}).get("completeness", 0.0)))

        requirements = context.get("requirements", [])
        missing = context.get("requirements_missing", [])

        details = f"Completeness {score:.2f}"
        if missing:
            details += f" — missing: {', '.join(missing[:3])}"

        return GateResult(
            gate_name=self.gate_name,
            passed=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            details=details,
            metadata={"missing_requirements": missing},
        )


class CoherenceGate(QualityGate):
    """Checks logical consistency and structure."""

    gate_name = "coherence"

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def check(self, content: str, context: dict | None = None) -> GateResult:
        context = context or {}
        score = float(context.get("coherence", context.get("scores", {}).get("coherence", 0.0)))

        return GateResult(
            gate_name=self.gate_name,
            passed=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            details=f"Coherence {score:.2f}",
        )


class SourceCitationGate(QualityGate):
    """Checks for proper source citations when required."""

    gate_name = "source_citation"

    def __init__(self, required: bool = True):
        self.required = required

    def check(self, content: str, context: dict | None = None) -> GateResult:
        if not self.required:
            return GateResult(gate_name=self.gate_name, passed=True, score=1.0, details="Citations not required")

        # Heuristic: check for citation markers
        citation_markers = ["source:", "reference:", "according to", "cited", "[1]", "[2]", "http", "doi:"]
        found = sum(1 for m in citation_markers if m.lower() in content.lower())
        score = min(1.0, found / 2)  # At least 2 citation markers for full score

        return GateResult(
            gate_name=self.gate_name,
            passed=score >= 0.5,
            score=score,
            threshold=0.5,
            details=f"Found {found} citation indicators",
            metadata={"citation_count": found},
        )


class HallucinationGate(QualityGate):
    """Checks for unsupported or fabricated claims."""

    gate_name = "hallucination"

    def __init__(self, max_score: float = 0.3):
        self.max_score = max_score

    def check(self, content: str, context: dict | None = None) -> GateResult:
        context = context or {}
        hallucination_score = float(context.get("hallucination_score", 0.0))
        inverse = 1.0 - hallucination_score  # Higher is better

        return GateResult(
            gate_name=self.gate_name,
            passed=hallucination_score <= self.max_score,
            score=inverse,
            threshold=1.0 - self.max_score,
            details=f"Hallucination score {hallucination_score:.2f} (max allowed: {self.max_score:.2f})",
            metadata={
                "unsupported_claims": context.get("unsupported_claims", []),
            },
        )


class ContentLengthGate(QualityGate):
    """Checks that content meets minimum length."""

    gate_name = "content_length"

    def __init__(self, min_chars: int = 100, min_words: int = 20):
        self.min_chars = min_chars
        self.min_words = min_words

    def check(self, content: str, context: dict | None = None) -> GateResult:
        char_count = len(content)
        word_count = len(content.split())

        char_ok = char_count >= self.min_chars
        word_ok = word_count >= self.min_words
        passed = char_ok and word_ok

        return GateResult(
            gate_name=self.gate_name,
            passed=passed,
            score=1.0 if passed else min(char_count / self.min_chars, word_count / self.min_words),
            details=f"{word_count} words, {char_count} chars (min: {self.min_words} words, {self.min_chars} chars)",
        )


class OverallScoreGate(QualityGate):
    """Checks the overall quality score."""

    gate_name = "overall_score"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check(self, content: str, context: dict | None = None) -> GateResult:
        context = context or {}
        score = float(context.get("overall_score", 0.0))

        return GateResult(
            gate_name=self.gate_name,
            passed=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            details=f"Overall score {score:.2f}",
        )


class QualityGateChain:
    """Runs multiple quality gates in sequence.

    Can operate in strict mode (all gates must pass) or
    lenient mode (overall score determines pass/fail).
    """

    def __init__(
        self,
        gates: list[QualityGate] | None = None,
        strict: bool = False,
        min_pass_rate: float = 0.7,
    ):
        self.gates = gates or []
        self.strict = strict
        self.min_pass_rate = min_pass_rate

    def add_gate(self, gate: QualityGate) -> None:
        self.gates.append(gate)

    def run(self, content: str, context: dict | None = None) -> QualityGateChainResult:
        """Run all quality gates.

        Args:
            content: Content to evaluate.
            context: Context with scores and metadata.

        Returns:
            Chain result with individual gate results.
        """
        results = []
        for gate in self.gates:
            try:
                result = gate.check(content, context)
                results.append(result)
            except Exception as e:
                logger.warning("Gate %s failed: %s", gate.gate_name, e)
                results.append(GateResult(
                    gate_name=gate.gate_name,
                    passed=False,
                    details=f"Gate error: {e}",
                ))

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed_count / total if total > 0 else 0.0

        if self.strict:
            overall_passed = all(r.passed for r in results)
        else:
            overall_passed = pass_rate >= self.min_pass_rate

        return QualityGateChainResult(
            passed=overall_passed,
            gate_results=results,
            passed_count=passed_count,
            total_gates=total,
            pass_rate=pass_rate,
        )

    @classmethod
    def create_default(cls, strict: bool = False) -> QualityGateChain:
        """Create a default gate chain with standard gates.

        Args:
            strict: Whether all gates must pass.

        Returns:
            Configured QualityGateChain.
        """
        chain = cls(strict=strict)
        chain.add_gate(ContentLengthGate())
        chain.add_gate(OverallScoreGate())
        chain.add_gate(ConfidenceGate())
        chain.add_gate(CompletenessGate())
        chain.add_gate(CoherenceGate())
        return chain

    @classmethod
    def from_settings(cls) -> QualityGateChain:
        """Create a gate chain from application settings."""
        try:
            from config.settings import get_settings
            s = get_settings().quality
            chain = cls()
            chain.add_gate(ConfidenceGate(threshold=s.min_confidence_score))
            chain.add_gate(CompletenessGate(threshold=s.min_completeness_score))
            chain.add_gate(CoherenceGate(threshold=s.min_coherence_score))
            if s.require_source_citations:
                chain.add_gate(SourceCitationGate(required=True))
            chain.add_gate(HallucinationGate(max_score=s.max_hallucination_score))
            chain.add_gate(OverallScoreGate(threshold=s.auto_reject_below))
            chain.add_gate(ContentLengthGate())
            return chain
        except Exception:
            return cls.create_default()


@dataclass
class QualityGateChainResult:
    """Result from running a chain of quality gates."""
    passed: bool = False
    gate_results: list[GateResult] = field(default_factory=list)
    passed_count: int = 0
    total_gates: int = 0
    pass_rate: float = 0.0

    @property
    def failed_gates(self) -> list[GateResult]:
        return [r for r in self.gate_results if not r.passed]

    @property
    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"{status}: {self.passed_count}/{self.total_gates} gates passed ({self.pass_rate:.0%})"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "gates": [
                {
                    "name": r.gate_name,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "details": r.details,
                }
                for r in self.gate_results
            ],
        }
