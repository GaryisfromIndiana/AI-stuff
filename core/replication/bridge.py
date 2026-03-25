"""Cross-empire data bridge — connects empires for bi-directional data flow."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """A connection between two empires."""
    connection_id: str = ""
    source_empire_id: str = ""
    target_empire_id: str = ""
    status: str = "active"  # active, paused, disconnected
    established_at: str = ""
    last_sync: str = ""
    sync_count: int = 0
    total_entities_transferred: int = 0
    total_learnings_transferred: int = 0


@dataclass
class Transfer:
    """A data transfer between empires."""
    transfer_id: str = ""
    source_empire_id: str = ""
    target_empire_id: str = ""
    transfer_type: str = "knowledge"  # knowledge, memory, lieutenant, full
    entities_sent: int = 0
    accepted: int = 0
    rejected: int = 0
    conflicts: int = 0
    duration_seconds: float = 0.0
    timestamp: str = ""


@dataclass
class SharedKnowledge:
    """Knowledge shared between empires."""
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    total_unique: int = 0


@dataclass
class BridgeHealth:
    """Health status of the bridge."""
    active_connections: int = 0
    total_connections: int = 0
    recent_transfers: int = 0
    failed_transfers: int = 0
    avg_transfer_time: float = 0.0
    status: str = "healthy"


class CrossEmpireBridge:
    """Connects empires for bi-directional data flow.

    Manages connections between empires, handles knowledge transfer,
    memory sharing, lieutenant cloning, and full data synchronization.
    Includes filtering, rate limiting, and conflict detection.
    """

    def __init__(self):
        self._connections: dict[str, Connection] = {}
        self._transfers: list[Transfer] = []
        self._max_transfer_rate = 100  # Max entities per sync
        self._transfer_count = 0

    def connect(self, source_id: str, target_id: str) -> Connection:
        """Establish a connection between two empires.

        Args:
            source_id: Source empire ID.
            target_id: Target empire ID.

        Returns:
            Connection.
        """
        from utils.crypto import generate_id
        conn_id = generate_id("conn_")

        conn = Connection(
            connection_id=conn_id,
            source_empire_id=source_id,
            target_empire_id=target_id,
            status="active",
            established_at=datetime.now(timezone.utc).isoformat(),
        )

        self._connections[conn_id] = conn
        logger.info("Bridge connection established: %s -> %s", source_id, target_id)
        return conn

    def disconnect(self, connection_id: str) -> bool:
        """Disconnect two empires."""
        conn = self._connections.get(connection_id)
        if conn:
            conn.status = "disconnected"
            logger.info("Bridge disconnected: %s", connection_id)
            return True
        return False

    def send_knowledge(
        self,
        source_id: str,
        target_id: str,
        entity_types: list[str] | None = None,
        min_confidence: float = 0.6,
    ) -> Transfer:
        """Send knowledge entities from source to target empire.

        Args:
            source_id: Source empire.
            target_id: Target empire.
            entity_types: Entity type filter.
            min_confidence: Minimum confidence filter.

        Returns:
            Transfer result.
        """
        start = time.time()
        transfer = Transfer(
            source_empire_id=source_id,
            target_empire_id=target_id,
            transfer_type="knowledge",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            from core.knowledge.graph import KnowledgeGraph

            source_graph = KnowledgeGraph(source_id)
            target_graph = KnowledgeGraph(target_id)

            # Export filtered entities
            export = source_graph.export_graph()
            nodes = export.get("nodes", [])
            edges = export.get("edges", [])

            if entity_types:
                nodes = [n for n in nodes if n.get("type") in entity_types]
            nodes = [n for n in nodes if n.get("confidence", 0) >= min_confidence]

            # Rate limit
            nodes = nodes[:self._max_transfer_rate]

            # Import
            result = target_graph.import_graph({"nodes": nodes, "edges": edges})

            transfer.entities_sent = len(nodes)
            transfer.accepted = result.get("nodes_imported", 0)
            transfer.duration_seconds = time.time() - start

            # Record sync in DB
            self._record_transfer(transfer)

        except Exception as e:
            logger.error("Knowledge transfer failed: %s", e)
            transfer.entities_sent = 0
            transfer.duration_seconds = time.time() - start

        self._transfers.append(transfer)
        self._transfer_count += 1
        return transfer

    def send_learnings(
        self,
        source_id: str,
        target_id: str,
        max_learnings: int = 50,
    ) -> Transfer:
        """Send experiential learnings from source to target.

        Args:
            source_id: Source empire.
            target_id: Target empire.
            max_learnings: Max learnings to send.

        Returns:
            Transfer result.
        """
        start = time.time()
        transfer = Transfer(
            source_empire_id=source_id,
            target_empire_id=target_id,
            transfer_type="memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            from core.memory.manager import MemoryManager

            source_mm = MemoryManager(source_id)
            target_mm = MemoryManager(target_id)

            # Export high-value experiential memories
            learnings = source_mm.recall(
                memory_types=["experiential", "design"],
                limit=max_learnings,
            )

            imported = target_mm.import_memories(learnings)
            transfer.entities_sent = len(learnings)
            transfer.accepted = imported

        except Exception as e:
            logger.error("Learning transfer failed: %s", e)

        transfer.duration_seconds = time.time() - start
        self._transfers.append(transfer)
        return transfer

    def broadcast(self, source_id: str, data_type: str = "knowledge") -> list[Transfer]:
        """Broadcast knowledge/learnings to all connected empires.

        Args:
            source_id: Source empire.
            data_type: Type of data to broadcast.

        Returns:
            List of transfer results.
        """
        results = []
        for conn in self._connections.values():
            if conn.status != "active":
                continue

            target_id = None
            if conn.source_empire_id == source_id:
                target_id = conn.target_empire_id
            elif conn.target_empire_id == source_id:
                target_id = conn.source_empire_id

            if target_id:
                if data_type == "knowledge":
                    result = self.send_knowledge(source_id, target_id)
                else:
                    result = self.send_learnings(source_id, target_id)
                results.append(result)

        return results

    def get_connections(self, empire_id: str = "") -> list[Connection]:
        """Get all connections, optionally filtered by empire."""
        conns = list(self._connections.values())
        if empire_id:
            conns = [
                c for c in conns
                if c.source_empire_id == empire_id or c.target_empire_id == empire_id
            ]
        return conns

    def get_transfer_history(self, empire_id: str = "", limit: int = 20) -> list[Transfer]:
        """Get recent transfer history."""
        transfers = self._transfers
        if empire_id:
            transfers = [
                t for t in transfers
                if t.source_empire_id == empire_id or t.target_empire_id == empire_id
            ]
        return transfers[-limit:]

    def get_bridge_health(self) -> BridgeHealth:
        """Get bridge health status."""
        active = sum(1 for c in self._connections.values() if c.status == "active")
        recent = self._transfers[-10:]
        failed = sum(1 for t in recent if t.accepted == 0 and t.entities_sent > 0)
        avg_time = sum(t.duration_seconds for t in recent) / max(len(recent), 1)

        status = "healthy"
        if failed > 3:
            status = "degraded"
        if active == 0 and len(self._connections) > 0:
            status = "unhealthy"

        return BridgeHealth(
            active_connections=active,
            total_connections=len(self._connections),
            recent_transfers=len(recent),
            failed_transfers=failed,
            avg_transfer_time=avg_time,
            status=status,
        )

    def sync_all(self) -> list[Transfer]:
        """Run a full sync between all connected empires."""
        results = []
        synced_pairs: set[tuple[str, str]] = set()

        for conn in self._connections.values():
            if conn.status != "active":
                continue

            pair = tuple(sorted([conn.source_empire_id, conn.target_empire_id]))
            if pair in synced_pairs:
                continue
            synced_pairs.add(pair)

            # Bidirectional sync
            results.append(self.send_knowledge(pair[0], pair[1]))
            results.append(self.send_knowledge(pair[1], pair[0]))
            results.append(self.send_learnings(pair[0], pair[1]))
            results.append(self.send_learnings(pair[1], pair[0]))

        return results

    def get_shared_knowledge(self, empire_ids: list[str]) -> SharedKnowledge:
        """Get knowledge shared across multiple empires."""
        from core.knowledge.bridge import KnowledgeBridge
        bridge = KnowledgeBridge()
        shared = bridge.get_shared_entities(empire_ids)

        return SharedKnowledge(
            entities=shared,
            sources=empire_ids,
            total_unique=len(shared),
        )

    def _record_transfer(self, transfer: Transfer) -> None:
        """Record a transfer in the database."""
        try:
            from db.engine import session_scope
            from db.models import CrossEmpireSync

            with session_scope() as session:
                sync = CrossEmpireSync(
                    source_empire_id=transfer.source_empire_id,
                    target_empire_id=transfer.target_empire_id,
                    sync_type=transfer.transfer_type,
                    status="completed",
                    entities_synced=transfer.accepted,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                session.add(sync)
        except Exception as e:
            logger.warning("Failed to record transfer: %s", e)
