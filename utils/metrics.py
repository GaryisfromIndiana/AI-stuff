"""Metrics collection and reporting for Empire."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    avg_value: float = 0.0
    last_value: float = 0.0
    rate_per_minute: float = 0.0


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str):
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()

    def increment(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> float:
        with self._lock:
            val = self._value
            self._value = 0.0
            return val


class Gauge:
    """Thread-safe gauge metric (can go up or down)."""

    def __init__(self, name: str, initial: float = 0.0):
        self.name = name
        self._value = initial
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def increment(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Tracks distribution of values."""

    def __init__(self, name: str, max_samples: int = 1000):
        self.name = name
        self._samples: list[float] = []
        self._max_samples = max_samples
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._samples.append(value)
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples:]

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def total(self) -> float:
        return sum(self._samples)

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def min(self) -> float:
        return min(self._samples) if self._samples else 0.0

    @property
    def max(self) -> float:
        return max(self._samples) if self._samples else 0.0

    def percentile(self, p: float) -> float:
        """Get the p-th percentile (0-100)."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def summary(self) -> dict:
        return {
            "count": self.count,
            "total": self.total,
            "avg": self.avg,
            "min": self.min,
            "max": self.max,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p99": self.percentile(99),
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram):
        self._histogram = histogram
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        duration = time.time() - self._start
        self._histogram.observe(duration)

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


class MetricsCollector:
    """Central metrics collector for Empire.

    Provides counters, gauges, histograms, and timers for
    tracking system performance and behavior.
    """

    _instance: MetricsCollector | None = None

    def __init__(self):
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._points: list[MetricPoint] = []
        self._max_points = 10000
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def counter(self, name: str) -> Counter:
        """Get or create a counter."""
        if name not in self._counters:
            self._counters[name] = Counter(name)
        return self._counters[name]

    def gauge(self, name: str, initial: float = 0.0) -> Gauge:
        """Get or create a gauge."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, initial)
        return self._gauges[name]

    def histogram(self, name: str) -> Histogram:
        """Get or create a histogram."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name)
        return self._histograms[name]

    def timer(self, name: str) -> Timer:
        """Get a timer context manager for a histogram."""
        return Timer(self.histogram(name))

    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a metric point.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Optional tags.
        """
        point = MetricPoint(name=name, value=value, tags=tags or {})
        with self._lock:
            self._points.append(point)
            if len(self._points) > self._max_points:
                self._points = self._points[-self._max_points:]

    def get_summary(self, name: str) -> MetricSummary:
        """Get summary for a specific metric."""
        # Check histograms first
        if name in self._histograms:
            h = self._histograms[name]
            return MetricSummary(
                name=name,
                count=h.count,
                total=h.total,
                min_value=h.min,
                max_value=h.max,
                avg_value=h.avg,
                last_value=h._samples[-1] if h._samples else 0,
            )

        # Check counters
        if name in self._counters:
            c = self._counters[name]
            return MetricSummary(
                name=name,
                count=1,
                total=c.value,
                last_value=c.value,
                avg_value=c.value,
            )

        # Check gauges
        if name in self._gauges:
            g = self._gauges[name]
            return MetricSummary(
                name=name,
                count=1,
                total=g.value,
                last_value=g.value,
                avg_value=g.value,
            )

        return MetricSummary(name=name)

    def get_all_summaries(self) -> dict[str, MetricSummary]:
        """Get summaries for all metrics."""
        summaries = {}
        for name in self._counters:
            summaries[name] = self.get_summary(name)
        for name in self._gauges:
            summaries[name] = self.get_summary(name)
        for name in self._histograms:
            summaries[name] = self.get_summary(name)
        return summaries

    def get_recent_points(self, name: str = "", limit: int = 100) -> list[MetricPoint]:
        """Get recent metric points."""
        points = self._points
        if name:
            points = [p for p in points if p.name == name]
        return points[-limit:]

    def get_stats(self) -> dict:
        """Get metrics collector statistics."""
        return {
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
            "total_points": len(self._points),
        }

    def reset_all(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        for h in self._histograms.values():
            h.reset()
        self._points.clear()

    def export(self) -> dict:
        """Export all metrics as a dict."""
        return {
            "counters": {name: c.value for name, c in self._counters.items()},
            "gauges": {name: g.value for name, g in self._gauges.items()},
            "histograms": {name: h.summary() for name, h in self._histograms.items()},
            "timestamp": datetime.now(UTC).isoformat(),
        }


# ── Pre-defined metric names ──────────────────────────────────────────

class MetricNames:
    """Standard metric names used throughout Empire."""

    # Task metrics
    TASKS_TOTAL = "tasks.total"
    TASKS_COMPLETED = "tasks.completed"
    TASKS_FAILED = "tasks.failed"
    TASK_DURATION = "tasks.duration_seconds"
    TASK_QUALITY = "tasks.quality_score"
    TASK_COST = "tasks.cost_usd"

    # LLM metrics
    LLM_REQUESTS = "llm.requests"
    LLM_TOKENS_INPUT = "llm.tokens.input"
    LLM_TOKENS_OUTPUT = "llm.tokens.output"
    LLM_COST = "llm.cost_usd"
    LLM_LATENCY = "llm.latency_ms"
    LLM_ERRORS = "llm.errors"

    # Memory metrics
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_DECAYED = "memory.decayed"
    MEMORY_PROMOTED = "memory.promoted"

    # Knowledge metrics
    KNOWLEDGE_ENTITIES = "knowledge.entities"
    KNOWLEDGE_RELATIONS = "knowledge.relations"
    KNOWLEDGE_EXTRACTIONS = "knowledge.extractions"

    # Evolution metrics
    EVOLUTION_PROPOSALS = "evolution.proposals"
    EVOLUTION_APPROVED = "evolution.approved"
    EVOLUTION_APPLIED = "evolution.applied"

    # Budget metrics
    BUDGET_DAILY_SPEND = "budget.daily_spend"
    BUDGET_MONTHLY_SPEND = "budget.monthly_spend"

    # Scheduler metrics
    SCHEDULER_TICKS = "scheduler.ticks"
    SCHEDULER_JOBS_RUN = "scheduler.jobs_run"
    SCHEDULER_ERRORS = "scheduler.errors"


# Module-level convenience
def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return MetricsCollector.get_instance()
