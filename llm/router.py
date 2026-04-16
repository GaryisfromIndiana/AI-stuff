"""Smart model router — picks optimal model per task based on complexity, budget, and capabilities."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from config.settings import MODEL_CATALOG, LLMModelConfig, get_settings
from llm.base import LLMClient, LLMRequest, LLMResponse
from llm.cache import cache_llm_response, get_cache
from utils.circuit_breaker import CircuitOpenError, get_circuit

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Spend recording — inversion of control so llm/ doesn't import core/.
# Edges (web/, cli/) register a factory at bootstrap. See ARCHITECTURE.md.
# ─────────────────────────────────────────────────────────────────────
@runtime_checkable
class SpendRecorder(Protocol):
    """Anything that can persist an LLM spend event (e.g. core.routing.budget.BudgetManager)."""

    def record_spend(
        self,
        cost_usd: float,
        model: str,
        provider: str,
        tokens_input: int,
        tokens_output: int,
        purpose: str = "llm_call",
    ) -> None: ...


SpendRecorderFactory = Callable[[str], SpendRecorder]
_spend_recorder_factory: SpendRecorderFactory | None = None


def register_spend_recorder_factory(factory: SpendRecorderFactory | None) -> None:
    """Register a factory that builds a SpendRecorder for a given empire_id.

    Call once at app startup. Passing ``None`` clears the factory (useful in tests).
    If no factory is registered, the router silently skips cost recording.
    """
    global _spend_recorder_factory
    _spend_recorder_factory = factory


@dataclass
class TaskMetadata:
    """Metadata describing a task for routing decisions."""
    task_type: str = "general"  # research, analysis, code, creative, extraction, classification
    complexity: str = "moderate"  # simple, moderate, complex, expert
    required_capabilities: list[str] = field(default_factory=list)
    estimated_tokens: int = 2000
    budget_remaining_usd: float = 100.0
    priority: int = 5  # 1 = highest
    preferred_provider: str | None = None
    preferred_model: str | None = None
    require_tool_use: bool = False
    require_vision: bool = False
    require_json_output: bool = False


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    model_key: str
    model_config: LLMModelConfig
    provider: str
    estimated_cost_usd: float
    reasoning: str
    fallback_model: str | None = None
    confidence: float = 0.9


@dataclass
class CostEstimate:
    """Estimated cost for a task."""
    estimated_tokens_input: int
    estimated_tokens_output: int
    estimated_cost_usd: float
    model: str
    confidence: float = 0.8


@dataclass
class ModelHealth:
    """Health status of a model endpoint."""
    model_key: str
    available: bool = True
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    last_success: float | None = None
    last_error: float | None = None
    consecutive_errors: int = 0
    total_requests: int = 0
    total_errors: int = 0


# ── Task Routing Table ────────────────────────────────────────────────────
# Single source of truth for model selection.
# Each task type maps to: model, tier, capabilities needed, and reason.
#   - Haiku 4.5 (tier 3): cheap tasks (classification, extraction, tagging, routing)
#   - Sonnet 4 (tier 2):  agent work (research, ACE pipeline, lieutenant tasks)
#   - Opus 4 (tier 1):    heavy infra (synthesis, evolution, audits, planning)

@dataclass
class TaskRouting:
    """Routing config for a task type."""
    model: str
    tier: int
    capabilities: list[str]
    reason: str

TASK_ROUTING: dict[str, TaskRouting] = {
    # ── Haiku tier (cheap, fast) ─────────────────────────────
    "classification": TaskRouting("claude-haiku-4.5", 3, ["analysis"], "fast classification"),
    "extraction":     TaskRouting("claude-haiku-4.5", 3, ["analysis"], "entity/attribute extraction"),
    "tagging":        TaskRouting("claude-haiku-4.5", 3, ["analysis"], "simple tagging"),
    "routing":        TaskRouting("claude-haiku-4.5", 3, ["analysis"], "command routing"),
    "summarization":  TaskRouting("claude-haiku-4.5", 3, ["analysis"], "quick summarization"),
    "formatting":     TaskRouting("claude-haiku-4.5", 3, ["analysis"], "text formatting"),
    # ── Sonnet tier (agent work) ─────────────────────────────
    "research":       TaskRouting("claude-sonnet-4", 2, ["reasoning", "analysis"], "research analysis"),
    "analysis":       TaskRouting("claude-sonnet-4", 2, ["reasoning", "analysis"], "moderate analysis"),
    "code":           TaskRouting("claude-sonnet-4", 2, ["code", "reasoning"], "code generation"),
    "general":        TaskRouting("claude-sonnet-4", 2, ["reasoning", "analysis"], "general agent work"),
    "creative":       TaskRouting("claude-sonnet-4", 2, ["creative", "reasoning"], "creative generation"),
    "debate":         TaskRouting("claude-sonnet-4", 2, ["reasoning", "creative"], "war room debate"),
    "math":           TaskRouting("claude-sonnet-4", 2, ["math", "reasoning"], "mathematical reasoning"),
    "vision":         TaskRouting("claude-sonnet-4", 2, ["vision"], "vision analysis"),
    # ── Opus tier (heavy infra) ──────────────────────────────
    "synthesis":      TaskRouting("claude-opus-4", 1, ["reasoning", "analysis"], "deep synthesis"),
    "evolution":      TaskRouting("claude-opus-4", 1, ["reasoning", "code"], "self-improvement proposals"),
    "planning":       TaskRouting("claude-opus-4", 1, ["reasoning", "analysis"], "strategic planning"),
    "audit":          TaskRouting("claude-opus-4", 1, ["reasoning", "analysis"], "deep knowledge audit"),
    "expert":         TaskRouting("claude-opus-4", 1, ["reasoning", "analysis"], "expert-level reasoning"),
}

# Complexity → minimum tier (used in fallback scoring when task type isn't in TASK_ROUTING)
COMPLEXITY_TIERS = {"simple": 4, "moderate": 3, "complex": 2, "expert": 1}


class ModelRouter:
    """Smart model router that selects the optimal model for each task.

    Considers task complexity, required capabilities, budget constraints,
    current model health, and historical performance data.
    """

    _global_provider_outage_until: dict[str, float] = {}
    _global_outage_lock = threading.Lock()

    def __init__(self, empire_id: str = ""):
        if not empire_id:
            try:
                from config.settings import get_settings
                empire_id = get_settings().empire_id
            except Exception:
                pass
        self._empire_id = empire_id
        self._health: dict[str, ModelHealth] = {}
        self._clients: dict[str, LLMClient] = {}
        self._initialized = False

    def _init_clients(self) -> None:
        """Lazily initialize LLM clients."""
        if self._initialized:
            return
        settings = get_settings()

        if settings.anthropic_api_key:
            from llm.anthropic import AnthropicClient
            self._clients["anthropic"] = AnthropicClient(settings.anthropic_api_key)

        if settings.openai_api_key:
            # Only register OpenAI if key looks valid (not a placeholder)
            key = settings.openai_api_key.strip()
            if key and not key.startswith("sk-placeholder") and len(key) > 20:
                from llm.openai import OpenAIClient
                self._clients["openai"] = OpenAIClient(key)
            else:
                logger.info("OpenAI key looks invalid — skipping OpenAI provider")

        # Initialize health for all models
        for key in MODEL_CATALOG:
            self._health[key] = ModelHealth(model_key=key)

        self._initialized = True

    def get_client(self, provider: str) -> LLMClient:
        """Get the LLM client for a provider.

        Args:
            provider: Provider name (anthropic, openai).

        Returns:
            LLM client instance.
        """
        self._init_clients()
        if provider not in self._clients:
            raise ValueError(f"No client available for provider: {provider}")
        return self._clients[provider]

    def route(self, metadata: TaskMetadata) -> RoutingDecision:
        """Select the optimal model for a task.

        Args:
            metadata: Task metadata describing the work.

        Returns:
            RoutingDecision with selected model and reasoning.
        """
        self._init_clients()

        # If user specified a preferred model, use it if available
        if metadata.preferred_model and metadata.preferred_model in MODEL_CATALOG:
            config = MODEL_CATALOG[metadata.preferred_model]
            return RoutingDecision(
                model_key=metadata.preferred_model,
                model_config=config,
                provider=config.provider,
                estimated_cost_usd=self._estimate_cost(config, metadata.estimated_tokens),
                reasoning=f"User-preferred model: {metadata.preferred_model}",
                fallback_model=self._find_fallback(metadata.preferred_model),
            )

        # Smart tiering: check if task type has a pre-assigned model
        task_routing = TASK_ROUTING.get(metadata.task_type)
        if task_routing:
            model_key = task_routing.model
            if model_key in MODEL_CATALOG:
                config = MODEL_CATALOG[model_key]
                if config.provider in self._clients:
                    health = self._health.get(model_key)
                    if not health or health.consecutive_errors < 3:
                        return RoutingDecision(
                            model_key=model_key,
                            model_config=config,
                            provider=config.provider,
                            estimated_cost_usd=self._estimate_cost(config, metadata.estimated_tokens),
                            reasoning=f"Tiered routing: {task_routing.reason} → {model_key}",
                            fallback_model=self._find_fallback(model_key),
                        )

        # Get candidates based on capabilities
        candidates = self._filter_candidates(metadata)

        if not candidates:
            # Fallback to any available model
            candidates = list(MODEL_CATALOG.items())

        # Score candidates
        scored = []
        for key, config in candidates:
            score = self._score_candidate(key, config, metadata)
            scored.append((key, config, score))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[2], reverse=True)

        best_key, best_config, best_score = scored[0]

        # Prefer fallback from the same provider (avoids cross-provider key issues)
        fallback_key = None
        for key, config, score in scored[1:]:
            if config.provider == best_config.provider:
                fallback_key = key
                break
        # If no same-provider fallback, use best alternative
        if fallback_key is None and len(scored) > 1:
            fallback_key = scored[1][0]

        estimated_cost = self._estimate_cost(best_config, metadata.estimated_tokens)

        return RoutingDecision(
            model_key=best_key,
            model_config=best_config,
            provider=best_config.provider,
            estimated_cost_usd=estimated_cost,
            reasoning=self._explain_decision(best_key, best_config, metadata, best_score),
            fallback_model=fallback_key,
            confidence=min(1.0, best_score),
        )

    def route_batch(self, metadatas: list[TaskMetadata]) -> list[RoutingDecision]:
        """Route multiple tasks, optimizing for total cost.

        Args:
            metadatas: List of task metadata.

        Returns:
            List of routing decisions.
        """
        return [self.route(m) for m in metadatas]

    def execute(
        self,
        request: LLMRequest,
        metadata: TaskMetadata | None = None,
    ) -> LLMResponse:
        """Route and execute an LLM request.

        Args:
            request: The LLM request.
            metadata: Optional task metadata for routing. If None, uses defaults.

        Returns:
            LLM response.
        """
        if metadata is None:
            metadata = TaskMetadata()

        decision = self.route(metadata)

        # Check cache first
        cache = get_cache()
        prompt_text = request.messages[-1].content if request.messages else ""
        cached = cache.get(
            model=decision.model_key,
            prompt=prompt_text,
            system_prompt=request.system_prompt,
        )
        if cached:
            logger.debug("Cache HIT for %s", decision.model_key)
            return LLMResponse(
                content=cached.content,
                model=decision.model_config.model_id,
                provider=decision.provider,
                tokens_input=cached.tokens_input,
                tokens_output=cached.tokens_output,
                cost_usd=0.0,
                latency_ms=0.0,
            )

        # Override model in request
        request.model = decision.model_config.model_id

        client = self.get_client(decision.provider)
        circuit = get_circuit(f"llm:{decision.provider}")

        try:
            with self._global_outage_lock:
                outage_until = self._global_provider_outage_until.get(decision.provider, 0.0)
            if outage_until > time.time():
                raise ConnectionError(f"Provider {decision.provider} temporarily unavailable")

            if not circuit.allow_request():
                raise CircuitOpenError(
                    f"Circuit for {decision.provider} is OPEN — skipping to fallback"
                )

            start = time.time()
            response = client.complete(request)
            latency = (time.time() - start) * 1000

            circuit.record_success()
            self._update_health(decision.model_key, success=True, latency_ms=latency)
            self._record_cost(response, decision.model_key, decision.provider)

            # Cache the response
            cache_llm_response(
                model=decision.model_key,
                prompt=prompt_text,
                content=response.content,
                system_prompt=request.system_prompt,
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
                cost_usd=response.cost_usd,
            )

            return response

        except (Exception, CircuitOpenError) as e:
            circuit.record_failure(e if not isinstance(e, CircuitOpenError) else None)
            self._update_health(decision.model_key, success=False)
            logger.warning("Primary model %s failed: %s", decision.model_key, e)

            # Fail fast for network connectivity errors on same-provider fallbacks.
            # Runtime evidence shows repeated Anthropic->Anthropic retries produce only APIConnectionError.
            is_connection_error = type(e).__name__ in {"APIConnectionError", "ConnectionError"}
            same_provider_fallback = (
                decision.fallback_model is not None
                and MODEL_CATALOG.get(decision.fallback_model) is not None
                and MODEL_CATALOG[decision.fallback_model].provider == decision.provider
            )
            if is_connection_error:
                with self._global_outage_lock:
                    self._global_provider_outage_until[decision.provider] = time.time() + 30.0
            if is_connection_error and same_provider_fallback:
                raise

            # Try fallback
            if decision.fallback_model:
                fallback_config = MODEL_CATALOG[decision.fallback_model]
                fallback_client = self.get_client(fallback_config.provider)
                request.model = fallback_config.model_id

                try:
                    response = fallback_client.complete(request)
                    logger.info("Fallback to %s succeeded", decision.fallback_model)
                    self._record_cost(response, decision.fallback_model, fallback_config.provider)
                    return response
                except Exception as e2:
                    self._update_health(decision.fallback_model, success=False)
                    logger.warning("Fallback model %s also failed: %s", decision.fallback_model, e2)

            # Last resort: try any available Anthropic model
            for key, config in MODEL_CATALOG.items():
                if config.provider != "anthropic" or key == decision.model_key:
                    continue
                if key == decision.fallback_model:
                    continue
                if config.provider not in self._clients:
                    continue
                try:
                    request.model = config.model_id
                    response = self._clients["anthropic"].complete(request)
                    logger.info("Last-resort fallback to %s succeeded", key)
                    self._record_cost(response, key, "anthropic")
                    return response
                except Exception:
                    continue

            raise

    def _filter_candidates(
        self,
        metadata: TaskMetadata,
    ) -> list[tuple[str, LLMModelConfig]]:
        """Filter models by required capabilities and constraints."""
        max_tier = COMPLEXITY_TIERS.get(metadata.complexity, 3)
        required_caps = set(metadata.required_capabilities)

        # Add capabilities from task routing
        task_routing = TASK_ROUTING.get(metadata.task_type)
        if task_routing:
            required_caps.update(task_routing.capabilities)

        if metadata.require_vision:
            required_caps.add("vision")

        candidates = []
        for key, config in MODEL_CATALOG.items():
            # Check provider preference
            if metadata.preferred_provider and config.provider != metadata.preferred_provider:
                continue

            # Check tier (allow models at or below the required tier)
            if config.tier > max_tier + 1:  # Allow one tier above for cost savings
                continue

            # Check capabilities
            if required_caps and not required_caps.issubset(set(config.capabilities)):
                continue

            # Check health
            health = self._health.get(key)
            if health and health.consecutive_errors >= 3:
                continue

            # Check budget
            estimated_cost = self._estimate_cost(config, metadata.estimated_tokens)
            if estimated_cost > metadata.budget_remaining_usd:
                continue

            # Check provider availability
            if config.provider not in self._clients:
                continue

            candidates.append((key, config))

        return candidates

    def _score_candidate(
        self,
        key: str,
        config: LLMModelConfig,
        metadata: TaskMetadata,
    ) -> float:
        """Score a model candidate for a task. Higher is better."""
        score = 0.0

        # Tier match (prefer the tier that matches complexity)
        target_tier = COMPLEXITY_TIERS.get(metadata.complexity, 3)
        tier_diff = abs(config.tier - target_tier)
        score += max(0, 1.0 - tier_diff * 0.3)  # Penalize tier mismatch

        # Cost efficiency (cheaper is better, scaled by priority)
        cost = self._estimate_cost(config, metadata.estimated_tokens)
        max_cost = metadata.budget_remaining_usd
        if max_cost > 0:
            cost_ratio = 1.0 - (cost / max_cost)
            if metadata.priority <= 3:
                # High priority: quality over cost
                score += cost_ratio * 0.2
            else:
                # Normal priority: balance cost
                score += cost_ratio * 0.4

        # Health score
        health = self._health.get(key)
        if health and health.total_requests > 0:
            reliability = 1.0 - health.error_rate
            score += reliability * 0.3
        else:
            score += 0.25  # Unknown health, assume decent

        # Capability coverage
        required = set(metadata.required_capabilities)
        task_routing = TASK_ROUTING.get(metadata.task_type)
        if task_routing:
            required.update(task_routing.capabilities)
        if required:
            coverage = len(required.intersection(set(config.capabilities))) / len(required)
            score += coverage * 0.3

        # Provider preference bonus
        if metadata.preferred_provider and config.provider == metadata.preferred_provider:
            score += 0.1

        return score

    def _estimate_cost(self, config: LLMModelConfig, estimated_tokens: int) -> float:
        """Estimate cost for a model given estimated token count."""
        # Assume roughly 60% input, 40% output split
        tokens_in = int(estimated_tokens * 0.6)
        tokens_out = int(estimated_tokens * 0.4)
        return (
            tokens_in * config.cost_per_1k_input / 1000
            + tokens_out * config.cost_per_1k_output / 1000
        )

    def _find_fallback(self, model_key: str) -> str | None:
        """Find a fallback model — prefers same provider, then any available."""
        config = MODEL_CATALOG.get(model_key)
        if not config:
            return None

        # Prefer same-provider fallback
        same_provider = [
            (k, c) for k, c in MODEL_CATALOG.items()
            if k != model_key and c.provider == config.provider and c.provider in self._clients
        ]
        if same_provider:
            # Pick closest tier (prefer stepping down, not up)
            same_provider.sort(key=lambda x: abs(x[1].tier - config.tier))
            return same_provider[0][0]

        # Any provider fallback
        candidates = [
            (k, c) for k, c in MODEL_CATALOG.items()
            if k != model_key and c.provider in self._clients
        ]
        if candidates:
            candidates.sort(key=lambda x: x[1].tier)
            return candidates[0][0]
        return None

    def get_tier_map(self) -> dict:
        """Return the current model tiering map for debugging/display."""
        self._init_clients()
        result = {}
        for task_type, routing in TASK_ROUTING.items():
            available = routing.model in MODEL_CATALOG and MODEL_CATALOG[routing.model].provider in self._clients
            result[task_type] = {
                "model": routing.model,
                "tier": routing.tier,
                "reason": routing.reason,
                "capabilities": routing.capabilities,
                "available": available,
            }
        return result

    def _explain_decision(
        self,
        key: str,
        config: LLMModelConfig,
        metadata: TaskMetadata,
        score: float,
    ) -> str:
        """Generate a human-readable explanation of the routing decision."""
        cost = self._estimate_cost(config, metadata.estimated_tokens)
        return (
            f"Selected {key} (tier {config.tier}, {config.provider}) "
            f"for {metadata.complexity} {metadata.task_type} task. "
            f"Estimated cost: ${cost:.4f}. Score: {score:.2f}."
        )

    def _record_cost(self, response: LLMResponse, model_key: str, provider: str) -> None:
        """Record LLM cost via the registered spend recorder (if any)."""
        if response.cost_usd <= 0:
            return
        if _spend_recorder_factory is None:
            # No recorder registered — fine for tests and for entry points that
            # don't care about budget tracking. Logged at debug to avoid noise.
            logger.debug("No SpendRecorder factory registered; skipping cost recording.")
            return
        try:
            recorder = _spend_recorder_factory(self._empire_id)
            recorder.record_spend(
                cost_usd=response.cost_usd,
                model=model_key,
                provider=provider,
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
                purpose="llm_call",
            )
        except Exception as e:
            logger.warning("Could not record cost: %s", e)

    def _update_health(
        self,
        model_key: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Update model health after a request."""
        if model_key not in self._health:
            self._health[model_key] = ModelHealth(model_key=model_key)

        health = self._health[model_key]
        health.total_requests += 1

        if success:
            health.consecutive_errors = 0
            health.last_success = time.time()
            if latency_ms > 0:
                if health.avg_latency_ms == 0:
                    health.avg_latency_ms = latency_ms
                else:
                    health.avg_latency_ms = health.avg_latency_ms * 0.9 + latency_ms * 0.1
        else:
            health.total_errors += 1
            health.consecutive_errors += 1
            health.last_error = time.time()

        health.error_rate = health.total_errors / health.total_requests
        health.available = health.consecutive_errors < 5

    def estimate_cost(self, metadata: TaskMetadata) -> CostEstimate:
        """Estimate cost without routing.

        Args:
            metadata: Task metadata.

        Returns:
            Cost estimate.
        """
        decision = self.route(metadata)
        tokens_in = int(metadata.estimated_tokens * 0.6)
        tokens_out = int(metadata.estimated_tokens * 0.4)

        return CostEstimate(
            estimated_tokens_input=tokens_in,
            estimated_tokens_output=tokens_out,
            estimated_cost_usd=decision.estimated_cost_usd,
            model=decision.model_key,
        )

    def get_model_health(self, model_key: str) -> ModelHealth | None:
        """Get health status for a model."""
        return self._health.get(model_key)

    def get_all_health(self) -> dict[str, ModelHealth]:
        """Get health status for all models."""
        self._init_clients()
        return dict(self._health)

    def get_routing_stats(self) -> dict:
        """Get routing statistics."""
        self._init_clients()
        stats = {
            "available_providers": list(self._clients.keys()),
            "available_models": len(MODEL_CATALOG),
            "model_health": {},
        }
        for key, health in self._health.items():
            stats["model_health"][key] = {
                "available": health.available,
                "error_rate": health.error_rate,
                "avg_latency_ms": health.avg_latency_ms,
                "total_requests": health.total_requests,
            }
        return stats

    def get_available_models(
        self,
        capabilities: list[str] | None = None,
    ) -> list[dict]:
        """Get available models with optional capability filter."""
        self._init_clients()
        models = []
        for key, config in MODEL_CATALOG.items():
            if config.provider not in self._clients:
                continue
            if capabilities:
                if not all(c in config.capabilities for c in capabilities):
                    continue
            models.append({
                "key": key,
                "model_id": config.model_id,
                "provider": config.provider,
                "tier": config.tier,
                "capabilities": config.capabilities,
                "cost_per_1k": config.cost_per_1k_total,
            })
        return models

    def compare_models(
        self,
        metadata: TaskMetadata,
    ) -> list[dict]:
        """Compare all eligible models for a task."""
        candidates = self._filter_candidates(metadata)
        results = []
        for key, config in candidates:
            score = self._score_candidate(key, config, metadata)
            cost = self._estimate_cost(config, metadata.estimated_tokens)
            results.append({
                "model": key,
                "provider": config.provider,
                "tier": config.tier,
                "estimated_cost": cost,
                "score": score,
                "capabilities": config.capabilities,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
