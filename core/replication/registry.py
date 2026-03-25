"""Cross-empire registry — tracks all empires and their capabilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmpireRegistryEntry:
    """An empire entry in the network registry."""
    empire_id: str
    name: str
    domain: str
    status: str = "active"
    lieutenant_count: int = 0
    knowledge_count: int = 0
    capabilities: list[str] = field(default_factory=list)
    last_active: str = ""


@dataclass
class EmpireStatus:
    """Detailed status of an empire."""
    empire_id: str = ""
    name: str = ""
    running: bool = True
    health: str = "unknown"
    active_directives: int = 0
    budget_remaining: float = 0.0
    lieutenant_count: int = 0


@dataclass
class NetworkStats:
    """Statistics about the empire network."""
    total_empires: int = 0
    active_empires: int = 0
    total_lieutenants: int = 0
    total_knowledge: int = 0
    total_tasks_completed: int = 0
    total_cost: float = 0.0


@dataclass
class CapabilityMap:
    """Map of domains and capabilities across empires."""
    domains: list[str] = field(default_factory=list)
    empire_capabilities: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of routing a directive to the best empire."""
    empire_id: str = ""
    empire_name: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    alternatives: list[dict] = field(default_factory=list)


class EmpireRegistry:
    """Tracks all empires and their capabilities in the network.

    Provides capability-based routing, cross-empire discovery,
    and network-wide statistics.
    """

    def __init__(self):
        self._entries: dict[str, EmpireRegistryEntry] = {}

    def register_empire(
        self,
        empire_id: str,
        name: str,
        domain: str,
        capabilities: list[str] | None = None,
    ) -> EmpireRegistryEntry:
        """Register an empire in the network."""
        entry = EmpireRegistryEntry(
            empire_id=empire_id,
            name=name,
            domain=domain,
            capabilities=capabilities or [],
        )
        self._entries[empire_id] = entry
        logger.info("Registered empire: %s (%s)", name, domain)
        return entry

    def unregister_empire(self, empire_id: str) -> bool:
        """Remove an empire from the registry."""
        if empire_id in self._entries:
            del self._entries[empire_id]
            return True
        return False

    def get_empire(self, empire_id: str) -> EmpireRegistryEntry | None:
        """Get an empire by ID."""
        return self._entries.get(empire_id)

    def list_empires(self, status: str | None = None) -> list[EmpireRegistryEntry]:
        """List all registered empires."""
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        return entries

    def get_empire_status(self, empire_id: str) -> EmpireStatus:
        """Get detailed status of an empire."""
        entry = self._entries.get(empire_id)
        if not entry:
            return EmpireStatus(empire_id=empire_id)

        try:
            from db.engine import get_session
            from db.repositories.empire import EmpireRepository
            session = get_session()
            repo = EmpireRepository(session)
            health = repo.get_health_overview(empire_id)

            return EmpireStatus(
                empire_id=empire_id,
                name=entry.name,
                running=entry.status == "active",
                health=health.get("health", {}).get("status", "unknown"),
                active_directives=health.get("directives", {}).get("active", 0),
                lieutenant_count=health.get("lieutenants", {}).get("active", 0),
            )
        except Exception:
            return EmpireStatus(empire_id=empire_id, name=entry.name)

    def get_network_stats(self) -> NetworkStats:
        """Get network-wide statistics."""
        entries = list(self._entries.values())
        active = [e for e in entries if e.status == "active"]

        return NetworkStats(
            total_empires=len(entries),
            active_empires=len(active),
            total_lieutenants=sum(e.lieutenant_count for e in entries),
            total_knowledge=sum(e.knowledge_count for e in entries),
        )

    def find_empire_by_domain(self, domain: str) -> list[EmpireRegistryEntry]:
        """Find empires in a specific domain."""
        return [e for e in self._entries.values() if e.domain == domain and e.status == "active"]

    def get_capability_map(self) -> CapabilityMap:
        """Get a map of capabilities across all empires."""
        domains = set()
        empire_caps: dict[str, dict[str, float]] = {}

        for entry in self._entries.values():
            domains.add(entry.domain)
            empire_caps[entry.empire_id] = {
                "domain": entry.domain,
                "capabilities": {c: 1.0 for c in entry.capabilities},
                "name": entry.name,
            }

        return CapabilityMap(
            domains=sorted(domains),
            empire_capabilities=empire_caps,
        )

    def route_directive(self, directive_description: str, directive_domain: str = "") -> RoutingResult:
        """Find the best empire for a directive.

        Args:
            directive_description: Description of the directive.
            directive_domain: Preferred domain.

        Returns:
            RoutingResult with recommended empire.
        """
        desc_lower = directive_description.lower()
        candidates = []

        for entry in self._entries.values():
            if entry.status != "active":
                continue

            score = 0.0
            reasons = []

            # Domain match
            if directive_domain and entry.domain == directive_domain:
                score += 0.5
                reasons.append(f"domain match ({entry.domain})")

            # Domain keyword match
            if entry.domain.lower() in desc_lower:
                score += 0.3
                reasons.append(f"domain keyword match")

            # Capability match
            for cap in entry.capabilities:
                if cap.lower() in desc_lower:
                    score += 0.1
                    reasons.append(f"capability: {cap}")

            # Lieutenant availability
            if entry.lieutenant_count > 0:
                score += 0.1

            if score > 0:
                candidates.append({
                    "empire_id": entry.empire_id,
                    "name": entry.name,
                    "score": score,
                    "reasons": reasons,
                })

        candidates.sort(key=lambda c: c["score"], reverse=True)

        if not candidates:
            return RoutingResult(reasoning="No suitable empire found")

        best = candidates[0]
        alternatives = candidates[1:3]

        return RoutingResult(
            empire_id=best["empire_id"],
            empire_name=best["name"],
            confidence=min(1.0, best["score"]),
            reasoning="; ".join(best["reasons"]),
            alternatives=[{"id": a["empire_id"], "name": a["name"], "score": a["score"]} for a in alternatives],
        )

    def get_network_health(self) -> dict:
        """Get health status of the entire network."""
        statuses = []
        for entry in self._entries.values():
            status = self.get_empire_status(entry.empire_id)
            statuses.append({
                "empire_id": entry.empire_id,
                "name": entry.name,
                "health": status.health,
                "running": status.running,
            })

        healthy = sum(1 for s in statuses if s["health"] == "healthy")
        total = len(statuses)

        return {
            "overall": "healthy" if healthy == total else "degraded" if healthy > 0 else "unhealthy",
            "healthy_count": healthy,
            "total_count": total,
            "empires": statuses,
        }

    def sync_from_db(self) -> int:
        """Sync registry from database."""
        try:
            from db.engine import get_session
            from db.repositories.empire import EmpireRepository
            session = get_session()
            repo = EmpireRepository(session)
            empires = repo.get_active()

            count = 0
            for empire in empires:
                self.register_empire(
                    empire_id=empire.id,
                    name=empire.name,
                    domain=empire.domain,
                )
                count += 1
            return count
        except Exception as e:
            logger.error("Failed to sync registry from DB: %s", e)
            return 0

    def export_registry(self) -> list[dict]:
        """Export registry for sharing."""
        return [
            {
                "empire_id": e.empire_id,
                "name": e.name,
                "domain": e.domain,
                "status": e.status,
                "capabilities": e.capabilities,
            }
            for e in self._entries.values()
        ]

    def import_registry(self, data: list[dict]) -> int:
        """Import registry entries."""
        count = 0
        for entry in data:
            self.register_empire(
                empire_id=entry.get("empire_id", ""),
                name=entry.get("name", ""),
                domain=entry.get("domain", ""),
                capabilities=entry.get("capabilities", []),
            )
            count += 1
        return count
