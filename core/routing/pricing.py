"""Pricing engine — calculates costs and optimizes model selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import MODEL_CATALOG, LLMModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing details for a model."""
    model_key: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    total_cost_per_1k: float
    tier: int
    capabilities: list[str] = field(default_factory=list)


@dataclass
class CostEstimate:
    """Estimated cost for a task."""
    estimated_tokens_input: int = 0
    estimated_tokens_output: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    confidence: float = 0.8


@dataclass
class ModelComparison:
    """Comparison of models for a task."""
    model: str
    provider: str
    estimated_cost: float
    quality_tier: int
    capabilities: list[str] = field(default_factory=list)
    recommendation: str = ""  # best_value, cheapest, highest_quality


@dataclass
class BatchCostEstimate:
    """Cost estimate for a batch of tasks."""
    total_cost: float = 0.0
    per_task_costs: list[dict] = field(default_factory=list)
    model_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizedBatch:
    """Optimized model assignments for a batch."""
    task_assignments: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    expected_quality: float = 0.0
    savings_vs_uniform: float = 0.0


class PricingEngine:
    """Calculates costs for LLM operations and optimizes model selection.

    Provides cost estimation, model comparison, batch optimization,
    and token estimation utilities.
    """

    # Token estimation heuristics
    CHARS_PER_TOKEN = 4
    TASK_TYPE_TOKEN_MULTIPLIERS = {
        "research": 1.5,
        "analysis": 1.3,
        "code": 1.4,
        "creative": 1.2,
        "extraction": 0.8,
        "classification": 0.6,
        "planning": 1.0,
        "general": 1.0,
    }

    COMPLEXITY_TOKEN_MULTIPLIERS = {
        "simple": 0.5,
        "moderate": 1.0,
        "complex": 2.0,
        "expert": 3.0,
    }

    def calculate_cost(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
    ) -> float:
        """Calculate exact cost for a completed request.

        Args:
            model: Model key or ID.
            tokens_input: Input tokens used.
            tokens_output: Output tokens used.

        Returns:
            Cost in USD.
        """
        config = self._get_model_config(model)
        if not config:
            return 0.0

        return (
            tokens_input * config.cost_per_1k_input / 1000 +
            tokens_output * config.cost_per_1k_output / 1000
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_task_cost(
        self,
        task_type: str = "general",
        complexity: str = "moderate",
        model: str = "claude-sonnet-4",
        input_text_length: int = 0,
        max_output_tokens: int = 4096,
    ) -> CostEstimate:
        """Estimate cost for a task before execution.

        Args:
            task_type: Type of task.
            complexity: Task complexity.
            model: Model to use.
            input_text_length: Length of input text in characters.
            max_output_tokens: Maximum output tokens.

        Returns:
            CostEstimate.
        """
        config = self._get_model_config(model)
        if not config:
            return CostEstimate(model=model)

        # Estimate input tokens
        if input_text_length > 0:
            base_input = input_text_length // self.CHARS_PER_TOKEN
        else:
            base_input = 1000  # Default estimate

        # Apply multipliers
        type_mult = self.TASK_TYPE_TOKEN_MULTIPLIERS.get(task_type, 1.0)
        complexity_mult = self.COMPLEXITY_TOKEN_MULTIPLIERS.get(complexity, 1.0)

        est_input = int(base_input * type_mult)
        est_output = int(max_output_tokens * 0.6 * complexity_mult)  # Assume 60% of max

        cost = (
            est_input * config.cost_per_1k_input / 1000 +
            est_output * config.cost_per_1k_output / 1000
        )

        return CostEstimate(
            estimated_tokens_input=est_input,
            estimated_tokens_output=est_output,
            estimated_cost_usd=cost,
            model=model,
            confidence=0.7,
        )

    def get_model_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing details for a model.

        Args:
            model: Model key.

        Returns:
            ModelPricing or None.
        """
        config = self._get_model_config(model)
        if not config:
            return None

        return ModelPricing(
            model_key=model,
            provider=config.provider,
            input_cost_per_1k=config.cost_per_1k_input,
            output_cost_per_1k=config.cost_per_1k_output,
            total_cost_per_1k=config.cost_per_1k_total,
            tier=config.tier,
            capabilities=config.capabilities,
        )

    def compare_models(
        self,
        task_type: str = "general",
        complexity: str = "moderate",
        input_length: int = 2000,
    ) -> list[ModelComparison]:
        """Compare all models for a given task.

        Args:
            task_type: Type of task.
            complexity: Task complexity.
            input_length: Estimated input length.

        Returns:
            List of model comparisons, sorted by value.
        """
        comparisons = []

        for key, config in MODEL_CATALOG.items():
            estimate = self.estimate_task_cost(
                task_type=task_type,
                complexity=complexity,
                model=key,
                input_text_length=input_length,
            )

            comparisons.append(ModelComparison(
                model=key,
                provider=config.provider,
                estimated_cost=estimate.estimated_cost_usd,
                quality_tier=config.tier,
                capabilities=config.capabilities,
            ))

        # Sort by cost
        comparisons.sort(key=lambda c: c.estimated_cost)

        # Annotate recommendations
        if comparisons:
            comparisons[0].recommendation = "cheapest"
            best_quality = min(comparisons, key=lambda c: c.quality_tier)
            best_quality.recommendation = "highest_quality"

            # Best value = good quality at reasonable cost
            for c in comparisons:
                if c.quality_tier <= 2 and c.estimated_cost < comparisons[-1].estimated_cost * 0.5:
                    c.recommendation = "best_value"
                    break

        return comparisons

    def calculate_batch_cost(
        self,
        tasks: list[dict],
        model: str = "claude-sonnet-4",
    ) -> BatchCostEstimate:
        """Calculate cost for a batch of tasks using one model.

        Args:
            tasks: List of task dicts with task_type, complexity, input_length.
            model: Model to use for all tasks.

        Returns:
            BatchCostEstimate.
        """
        per_task = []
        total = 0.0

        for task in tasks:
            estimate = self.estimate_task_cost(
                task_type=task.get("task_type", "general"),
                complexity=task.get("complexity", "moderate"),
                model=model,
                input_text_length=task.get("input_length", 2000),
            )
            per_task.append({
                "title": task.get("title", ""),
                "cost": estimate.estimated_cost_usd,
                "tokens": estimate.estimated_tokens_input + estimate.estimated_tokens_output,
            })
            total += estimate.estimated_cost_usd

        return BatchCostEstimate(
            total_cost=total,
            per_task_costs=per_task,
            model_breakdown={model: total},
        )

    def optimize_batch(
        self,
        tasks: list[dict],
        budget: float = 50.0,
    ) -> OptimizedBatch:
        """Assign models to minimize cost while maintaining quality.

        Simple tasks get cheap models, complex tasks get premium models.

        Args:
            tasks: List of task dicts.
            budget: Total budget for the batch.

        Returns:
            OptimizedBatch with assignments.
        """
        assignments = []
        total_cost = 0.0
        total_quality = 0.0

        # Model selection by complexity
        complexity_model_map = {
            "simple": "gpt-4o-mini",
            "moderate": "claude-haiku-4.5",
            "complex": "claude-sonnet-4",
            "expert": "claude-opus-4",
        }

        for task in tasks:
            complexity = task.get("complexity", "moderate")
            model = complexity_model_map.get(complexity, "claude-haiku-4.5")

            estimate = self.estimate_task_cost(
                task_type=task.get("task_type", "general"),
                complexity=complexity,
                model=model,
                input_text_length=task.get("input_length", 2000),
            )

            # Check budget
            if total_cost + estimate.estimated_cost_usd > budget:
                # Downgrade to cheaper model
                model = "gpt-4o-mini"
                estimate = self.estimate_task_cost(
                    task_type=task.get("task_type", "general"),
                    complexity=complexity,
                    model=model,
                )

            config = self._get_model_config(model)
            quality = (5 - (config.tier if config else 3)) / 4  # Tier 1 = 1.0, Tier 4 = 0.25

            assignments.append({
                "title": task.get("title", ""),
                "model": model,
                "estimated_cost": estimate.estimated_cost_usd,
                "quality_tier": config.tier if config else 3,
            })
            total_cost += estimate.estimated_cost_usd
            total_quality += quality

        # Calculate savings vs using premium model for all
        uniform_cost = self.calculate_batch_cost(tasks, "claude-sonnet-4").total_cost

        return OptimizedBatch(
            task_assignments=assignments,
            total_cost=total_cost,
            expected_quality=total_quality / max(len(tasks), 1),
            savings_vs_uniform=max(0, uniform_cost - total_cost),
        )

    def _get_model_config(self, model: str) -> LLMModelConfig | None:
        """Get model config by key or ID."""
        if model in MODEL_CATALOG:
            return MODEL_CATALOG[model]
        # Try matching by model_id
        for key, config in MODEL_CATALOG.items():
            if config.model_id == model:
                return config
        return None
