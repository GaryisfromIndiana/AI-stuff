"""Ralph Wiggum retry loop — named after 'I'm in danger'.

Injects previous errors as context, escalates to stronger models,
and uses successful sibling outputs to help failed tasks succeed.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from config.settings import MODEL_CATALOG

logger = logging.getLogger(__name__)


class ErrorClass(str, Enum):
    """Classification of errors for retry strategy."""
    TRANSIENT = "transient"        # Temporary API errors
    MODEL_LIMIT = "model_limit"    # Token limit exceeded
    QUALITY_FAILURE = "quality"    # Output quality too low
    TIMEOUT = "timeout"            # Execution timeout
    PERMANENT = "permanent"        # Logic/input errors (don't retry)
    RATE_LIMIT = "rate_limit"      # API rate limiting


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""
    attempt_number: int
    model_used: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    quality_score: float = 0.0
    success: bool = False


@dataclass
class RetryResult:
    """Final result after all retry attempts."""
    success: bool = False
    final_content: str = ""
    final_quality: float = 0.0
    attempts: list[RetryAttempt] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0
    final_model: str = ""
    error_classes: list[str] = field(default_factory=list)


@dataclass
class RetryStats:
    """Statistics about retry operations."""
    total_retries: int = 0
    success_after_retry: int = 0
    permanent_failures: int = 0
    avg_attempts_to_success: float = 0.0
    total_retry_cost: float = 0.0
    model_escalation_count: int = 0


# Model escalation ladder
MODEL_ESCALATION = [
    "claude-haiku-4.5",    # Tier 4 — cheapest
    "gpt-4o-mini",         # Tier 4 — alternative
    "claude-sonnet-4",     # Tier 2 — mid-range
    "gpt-4o",              # Tier 2 — alternative
    "claude-opus-4",       # Tier 1 — strongest
]


class RalphWiggumRetry:
    """The Ralph Wiggum retry loop.

    'I'm in danger' — but instead of giving up, this retry loop:
    1. Injects previous error context into the next attempt
    2. Escalates to a more powerful model after N failures
    3. Uses successful sibling outputs to help failing tasks
    4. Applies exponential backoff with jitter
    5. Classifies errors to determine retry strategy
    """

    def __init__(
        self,
        max_retries: int = 5,
        escalate_after: int = 2,
        backoff_base: float = 2.0,
        backoff_multiplier: float = 1.5,
        backoff_max: float = 60.0,
        inject_errors: bool = True,
        inject_siblings: bool = True,
    ):
        self.max_retries = max_retries
        self.escalate_after = escalate_after
        self.backoff_base = backoff_base
        self.backoff_multiplier = backoff_multiplier
        self.backoff_max = backoff_max
        self.inject_errors = inject_errors
        self.inject_siblings = inject_siblings

        # Stats
        self._total_retries = 0
        self._success_after_retry = 0
        self._permanent_failures = 0
        self._escalation_count = 0

    def execute_with_retry(
        self,
        task_fn: Callable[[str, str], Any],
        initial_model: str = "",
        task_context: str = "",
        min_quality: float = 0.6,
    ) -> RetryResult:
        """Execute a task with retry logic.

        Args:
            task_fn: Function that takes (model, context) and returns result dict.
                     Result must have: success, content, quality_score, cost_usd, error
            initial_model: Starting model.
            task_context: Original task context.
            min_quality: Minimum quality threshold.

        Returns:
            RetryResult.
        """
        result = RetryResult()
        current_model = initial_model or MODEL_ESCALATION[0]
        errors: list[str] = []
        start_time = time.time()

        for attempt in range(self.max_retries):
            # Build context with error injection
            context = task_context
            if self.inject_errors and errors:
                context = self._inject_error_context(task_context, errors)

            # Escalate model if needed
            if attempt >= self.escalate_after:
                new_model = self._escalate_model(current_model, attempt)
                if new_model != current_model:
                    current_model = new_model
                    self._escalation_count += 1
                    logger.info("Escalated to model: %s (attempt %d)", current_model, attempt + 1)

            # Execute
            try:
                attempt_start = time.time()
                task_result = task_fn(current_model, context)
                attempt_duration = time.time() - attempt_start

                attempt_record = RetryAttempt(
                    attempt_number=attempt + 1,
                    model_used=current_model,
                    duration_seconds=attempt_duration,
                    cost_usd=task_result.get("cost_usd", 0.0),
                    quality_score=task_result.get("quality_score", 0.0),
                    success=task_result.get("success", False),
                )

                result.attempts.append(attempt_record)
                result.total_cost += attempt_record.cost_usd
                if current_model not in result.models_used:
                    result.models_used.append(current_model)

                # Check success
                if task_result.get("success") and task_result.get("quality_score", 0) >= min_quality:
                    result.success = True
                    result.final_content = task_result.get("content", "")
                    result.final_quality = task_result.get("quality_score", 0)
                    result.final_model = current_model
                    self._success_after_retry += 1 if attempt > 0 else 0
                    break

                # Record error for next attempt
                error = task_result.get("error", "")
                if not error and not task_result.get("success"):
                    error = f"Quality below threshold: {task_result.get('quality_score', 0):.2f} < {min_quality}"

                errors.append(f"Attempt {attempt + 1} ({current_model}): {error}")
                attempt_record.error = error

                # Classify error
                error_class = self._classify_error(error)
                result.error_classes.append(error_class.value)

                # Don't retry permanent errors
                if error_class == ErrorClass.PERMANENT:
                    logger.info("Permanent error, not retrying: %s", error)
                    break

                # Backoff
                backoff = self._calculate_backoff(attempt)
                if attempt < self.max_retries - 1:
                    logger.info("Retry in %.1fs (attempt %d/%d)", backoff, attempt + 1, self.max_retries)
                    time.sleep(backoff)

            except Exception as e:
                errors.append(f"Attempt {attempt + 1}: Exception: {e}")
                result.attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    model_used=current_model,
                    error=str(e),
                ))

                error_class = self._classify_error(str(e))
                if error_class == ErrorClass.PERMANENT:
                    break

                backoff = self._calculate_backoff(attempt)
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)

        if not result.success:
            self._permanent_failures += 1

        result.total_duration = time.time() - start_time
        self._total_retries += len(result.attempts) - 1

        return result

    def retry_failed_batch(
        self,
        failed_tasks: list[dict],
        successful_tasks: list[dict],
        task_fn: Callable[[str, str], Any],
    ) -> list[RetryResult]:
        """Retry failed tasks using successful sibling context.

        Args:
            failed_tasks: Tasks that failed.
            successful_tasks: Tasks that succeeded (for context).
            task_fn: Execution function.

        Returns:
            List of retry results.
        """
        results = []

        # Build sibling context from successful tasks
        sibling_context = ""
        if self.inject_siblings and successful_tasks:
            sibling_context = self._build_sibling_context(successful_tasks)

        for task in failed_tasks:
            context = task.get("context", "")
            if sibling_context:
                context = f"{context}\n\n## Context from Successful Sibling Tasks\n{sibling_context}"

            result = self.execute_with_retry(
                task_fn=task_fn,
                initial_model=task.get("model", ""),
                task_context=context,
                min_quality=task.get("min_quality", 0.6),
            )
            results.append(result)

        return results

    def _escalate_model(self, current_model: str, attempt: int) -> str:
        """Pick a stronger model for the next attempt."""
        try:
            current_idx = MODEL_ESCALATION.index(current_model)
        except ValueError:
            current_idx = 0

        # Move up the escalation ladder
        next_idx = min(current_idx + 1, len(MODEL_ESCALATION) - 1)

        # For later attempts, jump further up
        if attempt >= 4:
            next_idx = len(MODEL_ESCALATION) - 1  # Go straight to strongest

        return MODEL_ESCALATION[next_idx]

    def _inject_error_context(self, original_context: str, errors: list[str]) -> str:
        """Inject previous errors into the context for the next attempt."""
        error_section = "\n".join(f"- {e}" for e in errors[-3:])  # Last 3 errors
        return (
            f"{original_context}\n\n"
            f"## Previous Attempts Failed\n"
            f"The following errors occurred in previous attempts. "
            f"Please avoid these issues:\n{error_section}\n\n"
            f"Be extra careful to produce high-quality, complete output."
        )

    def _build_sibling_context(self, successful_tasks: list[dict]) -> str:
        """Build context from successful sibling tasks."""
        parts = []
        for task in successful_tasks[:5]:
            content = task.get("content", "")[:500]
            title = task.get("title", "Sibling task")
            parts.append(f"**{title}:**\n{content}")

        return "\n\n".join(parts)

    def _classify_error(self, error: str) -> ErrorClass:
        """Classify an error to determine retry strategy."""
        error_lower = error.lower()

        if any(w in error_lower for w in ["rate limit", "429", "too many requests"]):
            return ErrorClass.RATE_LIMIT

        if any(w in error_lower for w in ["timeout", "timed out", "deadline"]):
            return ErrorClass.TIMEOUT

        if any(w in error_lower for w in ["server error", "500", "503", "overloaded", "service unavailable"]):
            return ErrorClass.TRANSIENT

        if any(w in error_lower for w in ["token limit", "max tokens", "context length", "too long"]):
            return ErrorClass.MODEL_LIMIT

        if any(w in error_lower for w in ["quality", "threshold", "below minimum"]):
            return ErrorClass.QUALITY_FAILURE

        if any(w in error_lower for w in ["invalid", "malformed", "unsupported", "not found", "permission"]):
            return ErrorClass.PERMANENT

        return ErrorClass.TRANSIENT  # Default: assume transient

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay with jitter."""
        delay = self.backoff_base * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.backoff_max)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)

    def get_stats(self) -> RetryStats:
        """Get retry statistics."""
        return RetryStats(
            total_retries=self._total_retries,
            success_after_retry=self._success_after_retry,
            permanent_failures=self._permanent_failures,
            avg_attempts_to_success=(
                self._total_retries / self._success_after_retry
                if self._success_after_retry > 0 else 0.0
            ),
            model_escalation_count=self._escalation_count,
        )
