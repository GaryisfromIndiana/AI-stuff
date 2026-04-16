"""Repository for atomic knowledge facts with smart deduplication."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from sqlalchemy import and_, desc, func, select

from db.models import KnowledgeFact, SourceReliability
from db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

# EMA smoothing factor — higher = more weight on recent observations
_EMA_ALPHA = 0.15


def _normalize_claim(claim: str) -> str:
    """Normalize a claim for dedup comparison."""
    return " ".join(claim.lower().split())


def _claim_hash(claim: str) -> str:
    """MD5 hash of normalized claim for fast dedup lookup."""
    return hashlib.md5(_normalize_claim(claim).encode()).hexdigest()[:16]


def _word_overlap(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(_normalize_claim(a).split())
    words_b = set(_normalize_claim(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class FactsRepository(BaseRepository[KnowledgeFact]):
    """Repository for atomic knowledge facts."""

    model_class = KnowledgeFact

    # ── Store with dedup ──────────────────────────────────────────────

    def store_fact(
        self,
        empire_id: str,
        claim: str,
        evidence: str = "",
        category: str = "general",
        source_url: str = "",
        source_tool: str = "",
        source_name: str = "",
        entity_id: str | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        source_task_id: str | None = None,
        lieutenant_id: str | None = None,
        verification_status: str = "unverified",
        verification_source: str = "",
        verification_detail: str = "",
    ) -> tuple[KnowledgeFact, bool]:
        """Store a fact with smart deduplication.

        If a fact with >75% word overlap exists for the same entity,
        update it instead of creating a duplicate.

        Returns:
            (fact, is_new) — the fact record and whether it was newly created.
        """
        c_hash = _claim_hash(claim)

        # Fast path: check for exact hash match
        existing = self._find_duplicate(empire_id, claim, c_hash, entity_id)

        if existing:
            # Update the existing fact with fresher data
            if confidence > existing.confidence:
                existing.confidence = confidence
            if evidence and len(evidence) > len(existing.evidence):
                existing.evidence = evidence
            if source_url:
                existing.source_url = source_url
            if verification_status != "unverified":
                existing.verification_status = verification_status
                existing.verification_source = verification_source
                existing.verification_detail = verification_detail
            existing.access_count += 1
            existing.updated_at = datetime.now(UTC)
            self.session.flush()
            return existing, False

        # New fact
        fact = KnowledgeFact(
            empire_id=empire_id,
            entity_id=entity_id,
            claim=claim,
            evidence=evidence,
            category=category,
            source_url=source_url,
            source_tool=source_tool,
            source_name=source_name,
            confidence=confidence,
            importance=importance,
            valid_from=valid_from,
            valid_until=valid_until,
            claim_hash=c_hash,
            source_task_id=source_task_id,
            lieutenant_id=lieutenant_id,
            verification_status=verification_status,
            verification_source=verification_source,
            verification_detail=verification_detail,
        )
        self.session.add(fact)
        self.session.flush()
        return fact, True

    def _find_duplicate(
        self,
        empire_id: str,
        claim: str,
        c_hash: str,
        entity_id: str | None,
    ) -> KnowledgeFact | None:
        """Find a duplicate fact by hash then fuzzy match."""
        # First: exact hash match
        conditions = [
            KnowledgeFact.empire_id == empire_id,
            KnowledgeFact.claim_hash == c_hash,
        ]
        if entity_id:
            conditions.append(KnowledgeFact.entity_id == entity_id)

        stmt = select(KnowledgeFact).where(and_(*conditions)).limit(5)
        candidates = list(self.session.execute(stmt).scalars().all())

        for candidate in candidates:
            if _word_overlap(claim, candidate.claim) >= 0.75:
                return candidate

        # Second: broader fuzzy search within same entity
        if entity_id:
            stmt2 = (
                select(KnowledgeFact)
                .where(and_(
                    KnowledgeFact.empire_id == empire_id,
                    KnowledgeFact.entity_id == entity_id,
                ))
                .order_by(desc(KnowledgeFact.confidence))
                .limit(20)
            )
            candidates2 = list(self.session.execute(stmt2).scalars().all())
            for candidate in candidates2:
                if _word_overlap(claim, candidate.claim) >= 0.75:
                    return candidate

        return None

    # ── Query ─────────────────────────────────────────────────────────

    def get_entity_facts(
        self,
        entity_id: str,
        empire_id: str,
        verified_only: bool = False,
        limit: int = 50,
    ) -> list[KnowledgeFact]:
        """Get all facts for an entity."""
        conditions = [
            KnowledgeFact.empire_id == empire_id,
            KnowledgeFact.entity_id == entity_id,
        ]
        if verified_only:
            conditions.append(KnowledgeFact.verification_status == "supported")

        stmt = (
            select(KnowledgeFact)
            .where(and_(*conditions))
            .order_by(desc(KnowledgeFact.confidence))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def search_facts(
        self,
        query: str,
        empire_id: str,
        limit: int = 20,
    ) -> list[KnowledgeFact]:
        """Search facts by claim text (ILIKE)."""
        pattern = f"%{query}%"
        stmt = (
            select(KnowledgeFact)
            .where(and_(
                KnowledgeFact.empire_id == empire_id,
                KnowledgeFact.claim.ilike(pattern),
            ))
            .order_by(desc(KnowledgeFact.confidence))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def find_contradictions(
        self,
        entity_id: str,
        empire_id: str,
    ) -> list[tuple[KnowledgeFact, KnowledgeFact]]:
        """Find facts for the same entity where one is supported and another contradicted."""
        supported = self.get_entity_facts(entity_id, empire_id, verified_only=True)
        stmt = (
            select(KnowledgeFact)
            .where(and_(
                KnowledgeFact.empire_id == empire_id,
                KnowledgeFact.entity_id == entity_id,
                KnowledgeFact.verification_status == "contradicted",
            ))
        )
        contradicted = list(self.session.execute(stmt).scalars().all())

        pairs = []
        for c in contradicted:
            for s in supported:
                if _word_overlap(c.claim, s.claim) >= 0.3:
                    pairs.append((s, c))
        return pairs

    def get_stats(self, empire_id: str) -> dict:
        """Get fact statistics for an empire."""
        total = self.session.execute(
            select(func.count(KnowledgeFact.id)).where(KnowledgeFact.empire_id == empire_id)
        ).scalar() or 0

        by_status = {}
        for status in ("unverified", "supported", "contradicted", "unverifiable"):
            count = self.session.execute(
                select(func.count(KnowledgeFact.id)).where(and_(
                    KnowledgeFact.empire_id == empire_id,
                    KnowledgeFact.verification_status == status,
                ))
            ).scalar() or 0
            by_status[status] = count

        return {"total": total, "by_status": by_status}


class SourceReliabilityRepository(BaseRepository[SourceReliability]):
    """Repository for source reliability EMA tracking."""

    model_class = SourceReliability

    def get_or_create(self, empire_id: str, source_name: str) -> SourceReliability:
        """Get existing source record or create with default score."""
        existing = self.session.execute(
            select(SourceReliability).where(and_(
                SourceReliability.empire_id == empire_id,
                SourceReliability.source_name == source_name,
            ))
        ).scalar_one_or_none()

        if existing:
            return existing

        record = SourceReliability(
            empire_id=empire_id,
            source_name=source_name,
            reliability_score=0.7,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def record_verification(
        self,
        empire_id: str,
        source_name: str,
        status: str,  # "supported", "contradicted", "unverifiable"
    ) -> SourceReliability:
        """Update source reliability EMA based on a verification outcome.

        EMA formula: score = alpha * new_value + (1 - alpha) * old_score
        - supported → new_value = 1.0
        - contradicted → new_value = 0.0
        - unverifiable → no EMA update, just count
        """
        record = self.get_or_create(empire_id, source_name)
        record.total_checks += 1
        record.last_checked_at = datetime.now(UTC)

        if status == "supported":
            record.supported_count += 1
            record.reliability_score = (
                _EMA_ALPHA * 1.0 + (1 - _EMA_ALPHA) * record.reliability_score
            )
        elif status == "contradicted":
            record.contradicted_count += 1
            record.reliability_score = (
                _EMA_ALPHA * 0.0 + (1 - _EMA_ALPHA) * record.reliability_score
            )
        elif status == "unverifiable":
            record.unverifiable_count += 1
            # No score update for unverifiable — we can't learn from it

        # Clamp
        record.reliability_score = max(0.05, min(1.0, record.reliability_score))
        self.session.flush()
        return record

    def get_all(self, empire_id: str) -> list[SourceReliability]:
        """Get all source reliability records for an empire."""
        stmt = (
            select(SourceReliability)
            .where(SourceReliability.empire_id == empire_id)
            .order_by(desc(SourceReliability.reliability_score))
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_score(self, empire_id: str, source_name: str) -> float:
        """Get reliability score for a source, defaulting to 0.7 if unknown."""
        record = self.session.execute(
            select(SourceReliability).where(and_(
                SourceReliability.empire_id == empire_id,
                SourceReliability.source_name == source_name,
            ))
        ).scalar_one_or_none()
        return record.reliability_score if record else 0.7
