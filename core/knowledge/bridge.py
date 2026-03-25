"""Cross-empire knowledge bridge — syncs knowledge between empires."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a knowledge sync operation."""
    source_empire_id: str = ""
    target_empire_id: str = ""
    entities_synced: int = 0
    relations_synced: int = 0
    conflicts_found: int = 0
    conflicts_resolved: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class SyncStatus:
    """Status of sync between two empires."""
    source_empire_id: str = ""
    target_empire_id: str = ""
    last_sync: str = ""
    pending_changes: int = 0
    conflicts: int = 0
    healthy: bool = True


@dataclass
class SyncPlan:
    """Plan for a sync operation."""
    source_empire_id: str = ""
    target_empire_id: str = ""
    entities_to_sync: int = 0
    relations_to_sync: int = 0
    estimated_conflicts: int = 0
    filters: dict = field(default_factory=dict)


@dataclass
class Resolution:
    """Resolution of a sync conflict."""
    entity_name: str = ""
    strategy: str = "newest_wins"  # newest_wins, highest_confidence, manual
    source_version: dict = field(default_factory=dict)
    target_version: dict = field(default_factory=dict)
    resolved_version: dict = field(default_factory=dict)


class KnowledgeBridge:
    """Syncs knowledge between empires in the network.

    Supports incremental sync, conflict resolution, and
    filtered sync by entity type or confidence threshold.
    """

    def __init__(self):
        self._graph_cache: dict[str, Any] = {}

    def _get_graph(self, empire_id: str):
        """Get or create a KnowledgeGraph for an empire."""
        if empire_id not in self._graph_cache:
            from core.knowledge.graph import KnowledgeGraph
            self._graph_cache[empire_id] = KnowledgeGraph(empire_id)
        return self._graph_cache[empire_id]

    def sync_to(
        self,
        source_empire_id: str,
        target_empire_id: str,
        entity_types: list[str] | None = None,
        min_confidence: float = 0.6,
    ) -> SyncResult:
        """Sync knowledge from source to target empire.

        Args:
            source_empire_id: Source empire.
            target_empire_id: Target empire.
            entity_types: Optional entity type filter.
            min_confidence: Minimum confidence to sync.

        Returns:
            SyncResult.
        """
        import time
        start = time.time()

        source_graph = self._get_graph(source_empire_id)
        target_graph = self._get_graph(target_empire_id)

        try:
            # Export from source
            export = source_graph.export_graph()
            nodes = export.get("nodes", [])
            edges = export.get("edges", [])

            # Filter
            if entity_types:
                nodes = [n for n in nodes if n.get("type") in entity_types]
            nodes = [n for n in nodes if n.get("confidence", 0) >= min_confidence]

            # Import to target
            result = target_graph.import_graph({"nodes": nodes, "edges": edges})

            # Record sync
            self._record_sync(source_empire_id, target_empire_id, result)

            return SyncResult(
                source_empire_id=source_empire_id,
                target_empire_id=target_empire_id,
                entities_synced=result.get("nodes_imported", 0),
                relations_synced=result.get("edges_imported", 0),
                duration_seconds=time.time() - start,
            )

        except Exception as e:
            logger.error("Sync failed: %s", e)
            return SyncResult(
                source_empire_id=source_empire_id,
                target_empire_id=target_empire_id,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def sync_from(
        self,
        target_empire_id: str,
        source_empire_id: str,
        entity_types: list[str] | None = None,
    ) -> SyncResult:
        """Pull knowledge from another empire."""
        return self.sync_to(source_empire_id, target_empire_id, entity_types)

    def full_sync(self, empire_ids: list[str]) -> list[SyncResult]:
        """Full bidirectional sync between all empires.

        Args:
            empire_ids: List of empire IDs to sync.

        Returns:
            List of sync results.
        """
        results = []
        for i, source_id in enumerate(empire_ids):
            for target_id in empire_ids[i + 1:]:
                # Sync both directions
                results.append(self.sync_to(source_id, target_id))
                results.append(self.sync_to(target_id, source_id))
        return results

    def get_sync_status(self, empire_id: str) -> list[SyncStatus]:
        """Get sync status for an empire with all other empires."""
        try:
            from db.engine import get_session
            from db.repositories.empire import EmpireRepository
            session = get_session()
            repo = EmpireRepository(session)
            syncs = repo.get_sync_status(empire_id)

            return [
                SyncStatus(
                    source_empire_id=s.get("source", ""),
                    target_empire_id=s.get("target", ""),
                    last_sync=s.get("created_at", ""),
                    conflicts=s.get("conflicts", 0),
                    healthy=s.get("status") == "completed",
                )
                for s in syncs
            ]
        except Exception as e:
            logger.error("Failed to get sync status: %s", e)
            return []

    def resolve_conflicts(
        self,
        conflicts: list[dict],
        strategy: str = "highest_confidence",
    ) -> list[Resolution]:
        """Resolve sync conflicts.

        Args:
            conflicts: List of conflict dicts.
            strategy: Resolution strategy.

        Returns:
            List of resolutions.
        """
        resolutions = []

        for conflict in conflicts:
            source = conflict.get("source", {})
            target = conflict.get("target", {})

            if strategy == "newest_wins":
                resolved = source if source.get("updated_at", "") > target.get("updated_at", "") else target
            elif strategy == "highest_confidence":
                resolved = source if source.get("confidence", 0) > target.get("confidence", 0) else target
            else:
                resolved = source  # Default to source

            resolutions.append(Resolution(
                entity_name=source.get("name", ""),
                strategy=strategy,
                source_version=source,
                target_version=target,
                resolved_version=resolved,
            ))

        return resolutions

    def create_sync_plan(
        self,
        source_id: str,
        target_id: str,
        entity_types: list[str] | None = None,
        min_confidence: float = 0.6,
    ) -> SyncPlan:
        """Create a plan for sync without executing.

        Args:
            source_id: Source empire.
            target_id: Target empire.
            entity_types: Entity type filter.
            min_confidence: Confidence filter.

        Returns:
            SyncPlan.
        """
        source_graph = self._get_graph(source_id)
        stats = source_graph.get_stats()

        entity_count = stats.entity_count
        relation_count = stats.relation_count

        return SyncPlan(
            source_empire_id=source_id,
            target_empire_id=target_id,
            entities_to_sync=entity_count,
            relations_to_sync=relation_count,
            filters={
                "entity_types": entity_types,
                "min_confidence": min_confidence,
            },
        )

    def get_shared_entities(self, empire_ids: list[str]) -> list[dict]:
        """Find entities that exist across multiple empires."""
        entity_maps: dict[str, list[str]] = {}

        for empire_id in empire_ids:
            graph = self._get_graph(empire_id)
            entities = graph.find_entities(limit=1000)
            for entity in entities:
                name = entity.name.lower()
                if name not in entity_maps:
                    entity_maps[name] = []
                entity_maps[name].append(empire_id)

        shared = []
        for name, empires in entity_maps.items():
            if len(empires) > 1:
                shared.append({"name": name, "empires": empires, "count": len(empires)})

        shared.sort(key=lambda x: x["count"], reverse=True)
        return shared

    def _record_sync(self, source_id: str, target_id: str, result: dict) -> None:
        """Record a sync operation in the database."""
        try:
            from db.engine import session_scope
            from db.models import CrossEmpireSync

            with session_scope() as session:
                sync = CrossEmpireSync(
                    source_empire_id=source_id,
                    target_empire_id=target_id,
                    sync_type="knowledge",
                    status="completed",
                    entities_synced=result.get("nodes_imported", 0),
                    relations_synced=result.get("edges_imported", 0),
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                session.add(sync)
        except Exception as e:
            logger.warning("Failed to record sync: %s", e)
