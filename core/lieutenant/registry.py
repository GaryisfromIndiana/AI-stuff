"""Cross-empire lieutenant registry — tracks all lieutenants across empires."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """A lieutenant entry in the cross-empire registry."""
    lieutenant_id: str
    empire_id: str
    name: str
    domain: str
    capabilities: list[str] = field(default_factory=list)
    performance_score: float = 0.5
    status: str = "active"
    last_active: str = ""


@dataclass
class CapabilityMatrix:
    """Matrix of domains × lieutenants with proficiency scores."""
    domains: list[str] = field(default_factory=list)
    entries: dict[str, dict[str, float]] = field(default_factory=dict)  # domain → {lt_id: score}


@dataclass
class CollaborationCandidate:
    """A lieutenant that could collaborate on a task."""
    lieutenant_id: str
    empire_id: str
    name: str
    domain: str
    relevance_score: float = 0.0
    reasoning: str = ""


@dataclass
class RegistryStats:
    """Statistics about the registry."""
    total_lieutenants: int = 0
    total_empires: int = 0
    by_domain: dict[str, int] = field(default_factory=dict)
    by_empire: dict[str, int] = field(default_factory=dict)
    avg_performance: float = 0.0


class LieutenantRegistry:
    """Cross-empire lieutenant registry.

    Tracks all lieutenants across all empires for capability-based
    routing, cross-empire collaboration, and network-wide insights.
    """

    def __init__(self):
        self._entries: dict[str, RegistryEntry] = {}

    def register(self, lieutenant_id: str, empire_id: str, name: str, domain: str,
                 capabilities: list[str] | None = None, performance: float = 0.5) -> RegistryEntry:
        """Register a lieutenant in the registry."""
        entry = RegistryEntry(
            lieutenant_id=lieutenant_id,
            empire_id=empire_id,
            name=name,
            domain=domain,
            capabilities=capabilities or [],
            performance_score=performance,
        )
        self._entries[lieutenant_id] = entry
        logger.debug("Registered lieutenant: %s (%s)", name, domain)
        return entry

    def unregister(self, lieutenant_id: str) -> bool:
        """Remove a lieutenant from the registry."""
        if lieutenant_id in self._entries:
            del self._entries[lieutenant_id]
            return True
        return False

    def find_by_capability(self, capability: str, empire_ids: list[str] | None = None) -> list[RegistryEntry]:
        """Find lieutenants with a specific capability."""
        results = []
        for entry in self._entries.values():
            if empire_ids and entry.empire_id not in empire_ids:
                continue
            if capability.lower() in [c.lower() for c in entry.capabilities]:
                results.append(entry)
        results.sort(key=lambda e: e.performance_score, reverse=True)
        return results

    def find_by_domain(self, domain: str, empire_ids: list[str] | None = None) -> list[RegistryEntry]:
        """Find lieutenants in a specific domain."""
        results = []
        for entry in self._entries.values():
            if empire_ids and entry.empire_id not in empire_ids:
                continue
            if entry.domain.lower() == domain.lower():
                results.append(entry)
        results.sort(key=lambda e: e.performance_score, reverse=True)
        return results

    def get_all(self, empire_id: str | None = None) -> list[RegistryEntry]:
        """Get all registered lieutenants."""
        entries = list(self._entries.values())
        if empire_id:
            entries = [e for e in entries if e.empire_id == empire_id]
        return entries

    def get_cross_empire_experts(self, domain: str) -> list[RegistryEntry]:
        """Find domain experts across all empires."""
        experts = self.find_by_domain(domain)
        # Group by empire to get diversity
        seen_empires: set[str] = set()
        diverse_experts = []
        for expert in experts:
            if expert.empire_id not in seen_empires:
                diverse_experts.append(expert)
                seen_empires.add(expert.empire_id)
        return diverse_experts

    def get_registry_stats(self) -> RegistryStats:
        """Get registry statistics."""
        entries = list(self._entries.values())

        by_domain: dict[str, int] = {}
        by_empire: dict[str, int] = {}
        total_perf = 0.0

        for entry in entries:
            by_domain[entry.domain] = by_domain.get(entry.domain, 0) + 1
            by_empire[entry.empire_id] = by_empire.get(entry.empire_id, 0) + 1
            total_perf += entry.performance_score

        return RegistryStats(
            total_lieutenants=len(entries),
            total_empires=len(by_empire),
            by_domain=by_domain,
            by_empire=by_empire,
            avg_performance=total_perf / len(entries) if entries else 0.0,
        )

    def sync_from_db(self, empire_id: str) -> int:
        """Sync registry entries from database.

        Args:
            empire_id: Empire to sync from.

        Returns:
            Number of entries synced.
        """
        try:
            from db.engine import get_session
            from db.repositories.lieutenant import LieutenantRepository
            session = get_session()
            repo = LieutenantRepository(session)

            db_lts = repo.get_by_empire(empire_id, status="active")
            count = 0
            for lt in db_lts:
                self.register(
                    lieutenant_id=lt.id,
                    empire_id=empire_id,
                    name=lt.name,
                    domain=lt.domain,
                    capabilities=lt.specializations_json or [],
                    performance=lt.performance_score,
                )
                count += 1
            return count
        except Exception as e:
            logger.error("Failed to sync registry: %s", e)
            return 0

    def get_capability_matrix(self) -> CapabilityMatrix:
        """Get a matrix of domains and lieutenant capabilities."""
        domains = set()
        for entry in self._entries.values():
            domains.add(entry.domain)

        matrix = CapabilityMatrix(domains=sorted(domains))
        for domain in domains:
            matrix.entries[domain] = {}
            for entry in self._entries.values():
                if entry.domain == domain:
                    matrix.entries[domain][entry.lieutenant_id] = entry.performance_score

        return matrix

    def find_collaboration_candidates(
        self,
        task_description: str,
        max_candidates: int = 5,
    ) -> list[CollaborationCandidate]:
        """Find lieutenants that could collaborate on a task.

        Args:
            task_description: Description of the task.
            max_candidates: Max candidates to return.

        Returns:
            List of collaboration candidates.
        """
        desc_lower = task_description.lower()
        candidates = []

        for entry in self._entries.values():
            if entry.status != "active":
                continue

            score = 0.0
            reasoning = []

            # Domain relevance
            if entry.domain.lower() in desc_lower:
                score += 0.4
                reasoning.append(f"domain match ({entry.domain})")

            # Capability relevance
            for cap in entry.capabilities:
                if cap.lower() in desc_lower:
                    score += 0.2
                    reasoning.append(f"capability match ({cap})")

            # Performance bonus
            score += entry.performance_score * 0.2

            if score > 0.2:
                candidates.append(CollaborationCandidate(
                    lieutenant_id=entry.lieutenant_id,
                    empire_id=entry.empire_id,
                    name=entry.name,
                    domain=entry.domain,
                    relevance_score=score,
                    reasoning="; ".join(reasoning),
                ))

        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:max_candidates]

    def export_registry(self) -> list[dict]:
        """Export registry for sharing."""
        return [
            {
                "lieutenant_id": e.lieutenant_id,
                "empire_id": e.empire_id,
                "name": e.name,
                "domain": e.domain,
                "capabilities": e.capabilities,
                "performance_score": e.performance_score,
            }
            for e in self._entries.values()
        ]

    def import_registry(self, data: list[dict]) -> int:
        """Import registry entries."""
        count = 0
        for entry_data in data:
            self.register(
                lieutenant_id=entry_data.get("lieutenant_id", ""),
                empire_id=entry_data.get("empire_id", ""),
                name=entry_data.get("name", ""),
                domain=entry_data.get("domain", ""),
                capabilities=entry_data.get("capabilities", []),
                performance=entry_data.get("performance_score", 0.5),
            )
            count += 1
        return count
