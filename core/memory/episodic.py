"""Episodic memory (Tier 4) — stores raw task records and conversation context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    task_id: str = ""
    context: str = ""
    actions: list[str] = field(default_factory=list)
    result: str = ""
    duration_seconds: float = 0.0
    success: bool = False
    quality_score: float = 0.0
    model_used: str = ""
    cost_usd: float = 0.0


@dataclass
class EpisodeSummary:
    total_episodes: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_quality: float = 0.0
    avg_duration: float = 0.0
    total_cost: float = 0.0
    key_themes: list[str] = field(default_factory=list)
    summary_text: str = ""


@dataclass
class Timeline:
    events: list[dict] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_events: int = 0


class EpisodicMemory:
    """Episodic memory — stores raw task execution records.

    Short-lived memories that capture task context, actions, and outcomes.
    High-value episodes get promoted to experiential memory.
    Old episodes are summarized and archived.
    """

    def __init__(self, memory_manager: MemoryManager, lieutenant_id: str = ""):
        self.mm = memory_manager
        self.lieutenant_id = lieutenant_id

    def store_episode(self, episode: Episode) -> dict:
        """Store a task execution episode."""
        content = f"Task: {episode.task_id}\n"
        if episode.context:
            content += f"Context: {episode.context[:500]}\n"
        if episode.actions:
            content += "Actions:\n" + "\n".join(f"- {a}" for a in episode.actions[:10]) + "\n"
        content += f"Result: {'SUCCESS' if episode.success else 'FAILURE'}\n"
        if episode.result:
            content += f"Output: {episode.result[:1000]}\n"
        content += f"Duration: {episode.duration_seconds:.1f}s, Cost: ${episode.cost_usd:.4f}"

        importance = 0.5
        if not episode.success:
            importance = 0.65  # Failures are more valuable to remember
        if episode.quality_score > 0.8:
            importance = 0.6  # High quality results worth remembering

        return self.mm.store(
            content=content,
            memory_type="episodic",
            lieutenant_id=self.lieutenant_id,
            title=f"Episode: {episode.task_id}",
            category="episode",
            importance=importance,
            tags=[
                "episode",
                "success" if episode.success else "failure",
                episode.model_used,
            ],
            source_task_id=episode.task_id,
            metadata={
                "success": episode.success,
                "quality_score": episode.quality_score,
                "duration": episode.duration_seconds,
                "cost": episode.cost_usd,
                "model": episode.model_used,
            },
            expires_hours=720,  # 30 days default TTL
        )

    def store_conversation(self, messages: list[dict], metadata: dict | None = None) -> dict:
        """Store a conversation record."""
        content = "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')[:200]}"
            for m in messages[:20]
        )

        return self.mm.store(
            content=content,
            memory_type="episodic",
            lieutenant_id=self.lieutenant_id,
            title="Conversation record",
            category="conversation",
            importance=0.3,
            tags=["conversation"],
            metadata=metadata or {},
            expires_hours=168,  # 7 days
        )

    def store_context_snapshot(self, context: dict) -> dict:
        """Store a snapshot of the current working context."""
        import json
        content = json.dumps(context, indent=2, default=str)[:3000]

        return self.mm.store(
            content=content,
            memory_type="episodic",
            lieutenant_id=self.lieutenant_id,
            title="Context snapshot",
            category="snapshot",
            importance=0.25,
            tags=["snapshot", "context"],
            expires_hours=48,  # 2 days
        )

    def get_recent_episodes(self, limit: int = 10) -> list[dict]:
        """Get recent episodes."""
        return self.mm.recall(
            query="episode",
            memory_types=["episodic"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def get_similar_episodes(self, context: str, limit: int = 5) -> list[dict]:
        """Find episodes similar to a given context."""
        return self.mm.recall(
            query=context,
            memory_types=["episodic"],
            lieutenant_id=self.lieutenant_id,
            limit=limit,
        )

    def get_recent_context(self, n: int = 5) -> list[dict]:
        """Get the n most recent episodic memories for context."""
        return self.mm.recall(
            memory_types=["episodic"],
            lieutenant_id=self.lieutenant_id,
            limit=n,
        )

    def summarize_episodes(self, episodes: list[dict] | None = None) -> EpisodeSummary:
        """Summarize a set of episodes."""
        if episodes is None:
            episodes = self.get_recent_episodes(limit=50)

        summary = EpisodeSummary(total_episodes=len(episodes))

        for ep in episodes:
            meta = ep.get("metadata", {}) if isinstance(ep.get("metadata"), dict) else {}
            if meta.get("success"):
                summary.success_count += 1
            else:
                summary.failure_count += 1

            if meta.get("quality_score"):
                summary.avg_quality += float(meta["quality_score"])
            if meta.get("duration"):
                summary.avg_duration += float(meta["duration"])
            if meta.get("cost"):
                summary.total_cost += float(meta["cost"])

        if summary.total_episodes > 0:
            summary.avg_quality /= summary.total_episodes
            summary.avg_duration /= summary.total_episodes

        summary.summary_text = (
            f"{summary.total_episodes} episodes: "
            f"{summary.success_count} successes, {summary.failure_count} failures. "
            f"Avg quality: {summary.avg_quality:.2f}, Total cost: ${summary.total_cost:.4f}"
        )

        return summary

    def promote_to_experiential(self, episode_id: str) -> dict | None:
        """Promote a high-value episode to experiential memory.

        Args:
            episode_id: ID of the episode to promote.

        Returns:
            Created experiential memory, or None.
        """
        episodes = self.mm.recall(query=episode_id, memory_types=["episodic"], limit=1)
        if not episodes:
            return None

        episode = episodes[0]
        return self.mm.store(
            content=f"[Promoted from episode] {episode.get('content', '')}",
            memory_type="experiential",
            lieutenant_id=self.lieutenant_id,
            title=f"Promoted: {episode.get('title', '')}",
            category="promoted",
            importance=episode.get("importance", 0.5) * 1.1,
            tags=["promoted", "from_episodic"],
            source_type="promotion",
            metadata={"promoted_from": episode_id},
        )

    def get_execution_timeline(self, task_ids: list[str] | None = None, hours: int = 24) -> Timeline:
        """Get an execution timeline of recent episodes.

        Args:
            task_ids: Optional filter by task IDs.
            hours: Hours to look back.

        Returns:
            Timeline of events.
        """
        episodes = self.get_recent_episodes(limit=50)

        events = []
        for ep in episodes:
            events.append({
                "time": ep.get("created_at", ""),
                "title": ep.get("title", ""),
                "type": "episode",
                "importance": ep.get("importance", 0),
            })

        events.sort(key=lambda e: e.get("time", ""))

        return Timeline(
            events=events,
            start_time=events[0]["time"] if events else "",
            end_time=events[-1]["time"] if events else "",
            total_events=len(events),
        )
