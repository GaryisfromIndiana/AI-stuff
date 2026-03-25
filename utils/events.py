"""Event bus for Empire — decoupled communication between subsystems."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A system event."""
    event_type: str
    source: str = ""
    data: dict = field(default_factory=dict)
    timestamp: str = ""
    empire_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# Event type constants
class EventTypes:
    # Task events
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_RETRYING = "task.retrying"

    # Directive events
    DIRECTIVE_CREATED = "directive.created"
    DIRECTIVE_STARTED = "directive.started"
    DIRECTIVE_COMPLETED = "directive.completed"
    DIRECTIVE_FAILED = "directive.failed"
    DIRECTIVE_CANCELLED = "directive.cancelled"

    # Lieutenant events
    LIEUTENANT_CREATED = "lieutenant.created"
    LIEUTENANT_ACTIVATED = "lieutenant.activated"
    LIEUTENANT_DEACTIVATED = "lieutenant.deactivated"
    LIEUTENANT_LEARNING = "lieutenant.learning"

    # Evolution events
    EVOLUTION_CYCLE_STARTED = "evolution.cycle_started"
    EVOLUTION_CYCLE_COMPLETED = "evolution.cycle_completed"
    EVOLUTION_PROPOSAL_CREATED = "evolution.proposal_created"
    EVOLUTION_PROPOSAL_APPROVED = "evolution.proposal_approved"
    EVOLUTION_PROPOSAL_REJECTED = "evolution.proposal_rejected"
    EVOLUTION_PROPOSAL_APPLIED = "evolution.proposal_applied"

    # Knowledge events
    KNOWLEDGE_ENTITY_CREATED = "knowledge.entity_created"
    KNOWLEDGE_RELATION_CREATED = "knowledge.relation_created"
    KNOWLEDGE_MAINTENANCE = "knowledge.maintenance"

    # Memory events
    MEMORY_STORED = "memory.stored"
    MEMORY_DECAYED = "memory.decayed"
    MEMORY_PROMOTED = "memory.promoted"
    MEMORY_CLEANED = "memory.cleaned"

    # War Room events
    WARROOM_CREATED = "warroom.created"
    WARROOM_DEBATE_STARTED = "warroom.debate_started"
    WARROOM_CONSENSUS_REACHED = "warroom.consensus_reached"
    WARROOM_CLOSED = "warroom.closed"

    # Budget events
    BUDGET_SPEND = "budget.spend"
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Health events
    HEALTH_CHECK = "health.check"
    HEALTH_DEGRADED = "health.degraded"
    HEALTH_UNHEALTHY = "health.unhealthy"

    # Scheduler events
    SCHEDULER_STARTED = "scheduler.started"
    SCHEDULER_STOPPED = "scheduler.stopped"
    SCHEDULER_JOB_COMPLETED = "scheduler.job_completed"
    SCHEDULER_JOB_FAILED = "scheduler.job_failed"

    # Empire events
    EMPIRE_CREATED = "empire.created"
    EMPIRE_SYNC = "empire.sync"


EventHandler = Callable[[Event], None]


class EventBus:
    """Simple in-process event bus for decoupled communication.

    Subsystems publish events without knowing who listens.
    Subscribers register handlers for specific event types.
    Supports wildcard subscriptions (e.g., 'task.*').
    """

    _instance: EventBus | None = None

    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._event_log: list[Event] = []
        self._max_log_size = 1000
        self._total_events = 0
        self._total_handlers_called = 0

    @classmethod
    def get_instance(cls) -> EventBus:
        """Get the singleton EventBus instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Event type to listen for. Use '*' suffix for wildcards (e.g., 'task.*').
            handler: Function to call when event occurs.
        """
        if event_type.endswith(".*"):
            prefix = event_type[:-2]
            self._wildcard_handlers[prefix].append(handler)
        else:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type.endswith(".*"):
            prefix = event_type[:-2]
            if prefix in self._wildcard_handlers:
                self._wildcard_handlers[prefix] = [
                    h for h in self._wildcard_handlers[prefix] if h != handler
                ]
        else:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]

    def publish(self, event: Event) -> int:
        """Publish an event to all subscribers.

        Args:
            event: The event to publish.

        Returns:
            Number of handlers called.
        """
        self._total_events += 1
        self._log_event(event)

        handlers_called = 0

        # Exact match handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
                handlers_called += 1
            except Exception as e:
                logger.error("Event handler error for %s: %s", event.event_type, e)

        # Wildcard handlers
        prefix = event.event_type.rsplit(".", 1)[0] if "." in event.event_type else ""
        if prefix:
            for handler in self._wildcard_handlers.get(prefix, []):
                try:
                    handler(event)
                    handlers_called += 1
                except Exception as e:
                    logger.error("Wildcard handler error for %s: %s", event.event_type, e)

        self._total_handlers_called += handlers_called
        return handlers_called

    def emit(self, event_type: str, source: str = "", data: dict | None = None, empire_id: str = "") -> int:
        """Convenience method to create and publish an event.

        Args:
            event_type: Event type.
            source: Event source.
            data: Event data.
            empire_id: Empire ID.

        Returns:
            Number of handlers called.
        """
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {},
            empire_id=empire_id,
        )
        return self.publish(event)

    def get_recent_events(self, limit: int = 50, event_type: str = "") -> list[Event]:
        """Get recent events from the log.

        Args:
            limit: Maximum events to return.
            event_type: Optional filter by event type.

        Returns:
            List of recent events.
        """
        events = self._event_log
        if event_type:
            events = [e for e in events if e.event_type == event_type or e.event_type.startswith(event_type)]
        return events[-limit:]

    def get_stats(self) -> dict:
        """Get event bus statistics."""
        return {
            "total_events": self._total_events,
            "total_handlers_called": self._total_handlers_called,
            "registered_handlers": sum(len(h) for h in self._handlers.values()),
            "registered_wildcards": sum(len(h) for h in self._wildcard_handlers.values()),
            "log_size": len(self._event_log),
        }

    def clear_handlers(self) -> None:
        """Remove all handlers (useful for testing)."""
        self._handlers.clear()
        self._wildcard_handlers.clear()

    def clear_log(self) -> None:
        """Clear the event log."""
        self._event_log.clear()

    def _log_event(self, event: Event) -> None:
        """Add event to the log, trimming if needed."""
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]


# Module-level convenience functions
def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return EventBus.get_instance()


def subscribe(event_type: str, handler: EventHandler) -> None:
    """Subscribe to an event type on the global bus."""
    get_event_bus().subscribe(event_type, handler)


def emit(event_type: str, source: str = "", data: dict | None = None, empire_id: str = "") -> int:
    """Emit an event on the global bus."""
    return get_event_bus().emit(event_type, source, data, empire_id)
