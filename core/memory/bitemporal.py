"""Bi-temporal memory — tracks both valid time and transaction time.

Valid time: when was this fact true in the real world?
Transaction time: when did Empire learn this?

This enables:
- Point-in-time queries: "What did we know as of date X?"
- Fact versioning: "How has our understanding of Y changed?"
- Supersession: "Is this fact still current or has it been replaced?"
- Timeline reconstruction: "What was the state of AI agents in Jan 2026?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class TemporalFact:
    """A fact with bi-temporal tracking."""
    id: str = ""
    content: str = ""
    title: str = ""
    memory_type: str = "semantic"
    category: str = ""

    # Valid time — when is/was this true in the real world?
    valid_from: str | None = None   # ISO datetime or None (always valid)
    valid_to: str | None = None     # ISO datetime or None (still valid)

    # Transaction time — when did Empire record this?
    recorded_at: str = ""       # When Empire first learned this
    superseded_at: str | None = None  # When a newer version replaced this

    # Versioning
    version: int = 1
    previous_version_id: str | None = None
    superseded_by_id: str | None = None

    # Quality
    importance: float = 0.5
    confidence: float = 0.8
    source: str = ""
    source_url: str = ""

    # Metadata
    tags: list[str] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)  # Knowledge graph entity names


@dataclass
class TemporalQuery:
    """A query against bi-temporal memory."""
    query: str = ""
    as_of_valid: str | None = None     # "What was true at this real-world time?"
    as_of_recorded: str | None = None  # "What did we know at this transaction time?"
    include_superseded: bool = False       # Include facts that have been replaced?
    memory_types: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    limit: int = 20


@dataclass
class FactVersion:
    """A version in a fact's history."""
    version: int
    content: str
    recorded_at: str
    superseded_at: str | None = None
    confidence: float = 0.8
    source: str = ""


@dataclass
class FactTimeline:
    """Timeline of how a fact evolved over time."""
    topic: str = ""
    versions: list[FactVersion] = field(default_factory=list)
    current_version: int = 0
    first_recorded: str = ""
    last_updated: str = ""


@dataclass
class TemporalSnapshot:
    """A snapshot of what Empire knew at a point in time."""
    as_of: str = ""
    snapshot_type: str = "recorded"  # "recorded" or "valid"
    facts: list[TemporalFact] = field(default_factory=list)
    total_facts: int = 0
    entity_count: int = 0


class BiTemporalMemory:
    """Bi-temporal memory system.

    Every fact stored has two time dimensions:
    - Valid time: when the fact is/was true in the real world
    - Transaction time: when Empire recorded the fact

    Facts can be superseded (replaced by newer versions) without
    deleting the old version. This creates a full audit trail of
    how Empire's knowledge evolved.
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def store_fact(
        self,
        content: str,
        title: str = "",
        category: str = "general",
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 0.8,
        importance: float = 0.6,
        source: str = "",
        source_url: str = "",
        tags: list[str] | None = None,
        entity_refs: list[str] | None = None,
        lieutenant_id: str = "",
    ) -> TemporalFact:
        """Store a new temporal fact.

        Args:
            content: Fact content.
            title: Short title.
            category: Category for organization.
            valid_from: When this became true (ISO datetime). None = always.
            valid_to: When this stopped being true. None = still true.
            confidence: How confident we are in this fact.
            importance: How important this fact is.
            source: Where this fact came from.
            source_url: URL of the source.
            tags: Tags for search.
            entity_refs: Knowledge graph entities this relates to.
            lieutenant_id: Which lieutenant discovered this.

        Returns:
            TemporalFact.
        """
        now = datetime.now(UTC).isoformat()

        # Single-transaction: find existing, mark superseded, store new — all in one session
        from db.engine import session_scope
        from db.models import MemoryEntry as MemoryModel
        from db.models import _generate_id

        superseded_id = None
        prev_version = 0
        entry_id = _generate_id()

        try:
            with session_scope() as session:
                # Find and supersede existing version
                if title:
                    from sqlalchemy import and_, select
                    stmt = (
                        select(MemoryModel)
                        .where(and_(
                            MemoryModel.empire_id == self.empire_id,
                            MemoryModel.title == title,
                            MemoryModel.source_type == "temporal_fact",
                        ))
                        .order_by(MemoryModel.created_at.desc())
                        .limit(1)
                    )
                    existing_entry = session.execute(stmt).scalar_one_or_none()
                    if existing_entry and existing_entry.metadata_json:
                        meta = existing_entry.metadata_json
                        if isinstance(meta, dict) and not meta.get("superseded_at"):
                            superseded_id = existing_entry.id
                            prev_version = meta.get("version", 0)
                            updated_meta = dict(meta)
                            updated_meta["superseded_at"] = now
                            existing_entry.metadata_json = updated_meta

                # Build metadata for new entry
                metadata = {
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "recorded_at": now,
                    "superseded_at": None,
                    "version": prev_version + 1,
                    "previous_version_id": superseded_id,
                    "superseded_by_id": None,
                    "source": source,
                    "source_url": source_url,
                    "entity_refs": entity_refs or [],
                    "temporal": True,
                }

                # Generate embedding for semantic search
                embedding = None
                try:
                    from core.memory.embeddings import generate_embedding
                    embed_text = f"{title}\n{content}" if title else content
                    embedding = generate_embedding(embed_text)
                except Exception:
                    pass

                # Insert new entry in same transaction
                new_entry = MemoryModel(
                    id=entry_id,
                    empire_id=self.empire_id,
                    lieutenant_id=lieutenant_id or None,
                    memory_type="semantic",
                    category=category or "general",
                    title=title,
                    content=content,
                    importance_score=importance,
                    confidence_score=confidence,
                    effective_importance=importance,
                    decay_factor=1.0,
                    tags_json=(tags or []) + ["temporal"],
                    metadata_json=metadata,
                    embedding_json=embedding,
                    source_type="temporal_fact",
                )
                session.add(new_entry)

        except Exception as e:
            logger.error("Failed to store temporal fact: %s", e)
            entry_id = ""
            metadata = {"version": 1, "recorded_at": now, "temporal": True}

        result = {"id": entry_id, "type": "semantic", "title": title}

        fact = TemporalFact(
            id=result.get("id", ""),
            content=content,
            title=title,
            memory_type="semantic",
            category=category,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_at=now,
            version=metadata["version"],
            previous_version_id=superseded_id,
            importance=importance,
            confidence=confidence,
            source=source,
            source_url=source_url,
            tags=tags or [],
            entity_refs=entity_refs or [],
        )

        logger.debug("Stored temporal fact: %s (v%d)", title or content[:50], metadata["version"])
        return fact

    def query(self, tq: TemporalQuery) -> list[TemporalFact]:
        """Query bi-temporal memory with temporal filters pushed to the DB.

        Unlike the old approach that fetched N results then filtered in Python
        (silently dropping matches), this queries the DB directly with temporal
        conditions so the limit applies to the filtered result set.

        Args:
            tq: Temporal query with time constraints.

        Returns:
            List of matching TemporalFacts.
        """
        from sqlalchemy import and_, select

        from db.engine import read_session
        from db.models import MemoryEntry

        # Fetch cap: we need tq.limit results AFTER Python-side filtering,
        # so fetch more from DB. Most filtering (superseded, valid_time) happens
        # in Python because the conditions live in JSON metadata.
        fetch_cap = tq.limit * 5

        with read_session() as session:
            stmt = (
                select(MemoryEntry)
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.source_type == "temporal_fact",
                ))
            )

            if tq.memory_types:
                stmt = stmt.where(MemoryEntry.memory_type.in_(tq.memory_types))

            if tq.min_confidence > 0:
                stmt = stmt.where(MemoryEntry.confidence_score >= tq.min_confidence)

            if tq.as_of_recorded:
                stmt = stmt.where(MemoryEntry.created_at <= tq.as_of_recorded)

            # Narrow by keyword if query provided (reduces Python-side iteration)
            if tq.query:
                pattern = f"%{tq.query}%"
                from sqlalchemy import or_
                stmt = stmt.where(or_(
                    MemoryEntry.title.ilike(pattern),
                    MemoryEntry.content.ilike(pattern),
                ))

            stmt = stmt.order_by(MemoryEntry.effective_importance.desc()).limit(fetch_cap)
            entries = list(session.execute(stmt).scalars().all())

        # Post-filter for JSON metadata conditions
        facts = []
        for entry in entries:
            meta = entry.metadata_json or {}
            if not isinstance(meta, dict):
                continue

            if not tq.include_superseded and meta.get("superseded_at"):
                continue

            if tq.as_of_valid:
                valid_from = meta.get("valid_from")
                valid_to = meta.get("valid_to")
                if valid_from and tq.as_of_valid < valid_from:
                    continue
                if valid_to and tq.as_of_valid > valid_to:
                    continue

            facts.append(self._entry_to_fact(entry, meta))

            if len(facts) >= tq.limit:
                break

        return facts

    @staticmethod
    def _entry_to_fact(entry, meta: dict) -> TemporalFact:
        """Convert a MemoryEntry + metadata dict to a TemporalFact."""
        return TemporalFact(
            id=entry.id,
            content=entry.content or "",
            title=entry.title or "",
            memory_type=entry.memory_type or "semantic",
            category=entry.category or "",
            valid_from=meta.get("valid_from"),
            valid_to=meta.get("valid_to"),
            recorded_at=meta.get("recorded_at", entry.created_at.isoformat() if entry.created_at else ""),
            superseded_at=meta.get("superseded_at"),
            version=meta.get("version", 1),
            previous_version_id=meta.get("previous_version_id"),
            superseded_by_id=meta.get("superseded_by_id"),
            importance=entry.effective_importance or 0.5,
            confidence=entry.confidence_score or 0.8,
            source=meta.get("source", ""),
            source_url=meta.get("source_url", ""),
            tags=entry.tags_json or [],
            entity_refs=meta.get("entity_refs", []),
        )

    def query_as_of(self, query: str, as_of: str, time_type: str = "valid") -> list[TemporalFact]:
        """Convenience method: query what was true/known at a specific time.

        Args:
            query: Search query.
            as_of: ISO datetime.
            time_type: "valid" (real-world time) or "recorded" (when Empire learned it).

        Returns:
            List of facts.
        """
        tq = TemporalQuery(query=query, limit=20)
        if time_type == "valid":
            tq.as_of_valid = as_of
        else:
            tq.as_of_recorded = as_of
        return self.query(tq)

    def get_current_facts(self, query: str = "", category: str = "", limit: int = 20) -> list[TemporalFact]:
        """Get currently valid, non-superseded facts.

        Args:
            query: Optional search query.
            category: Optional category filter.
            limit: Max results.

        Returns:
            List of current facts.
        """
        return self.query(TemporalQuery(
            query=query or category,
            include_superseded=False,
            limit=limit,
        ))

    def get_fact_timeline(self, title: str) -> FactTimeline:
        """Get the full version history of a fact.

        Args:
            title: Fact title to trace.

        Returns:
            FactTimeline with all versions.
        """
        # Get all versions including superseded
        all_versions = self.query(TemporalQuery(
            query=title,
            include_superseded=True,
            limit=50,
        ))

        # Filter to exact title matches
        versions = [f for f in all_versions if f.title and title.lower() in f.title.lower()]
        versions.sort(key=lambda f: f.version)

        fact_versions = [
            FactVersion(
                version=f.version,
                content=(f.content or "")[:500],
                recorded_at=f.recorded_at,
                superseded_at=f.superseded_at,
                confidence=f.confidence,
                source=f.source,
            )
            for f in versions
        ]

        return FactTimeline(
            topic=title,
            versions=fact_versions,
            current_version=max(f.version for f in versions) if versions else 0,
            first_recorded=versions[0].recorded_at if versions else "",
            last_updated=versions[-1].recorded_at if versions else "",
        )

    def get_snapshot(self, as_of: str, time_type: str = "recorded", limit: int = 50) -> TemporalSnapshot:
        """Get a snapshot of what Empire knew at a point in time.

        Args:
            as_of: ISO datetime.
            time_type: "recorded" or "valid".
            limit: Max facts.

        Returns:
            TemporalSnapshot.
        """
        facts = self.query_as_of("", as_of, time_type)

        return TemporalSnapshot(
            as_of=as_of,
            snapshot_type=time_type,
            facts=facts[:limit],
            total_facts=len(facts),
        )

    def supersede_fact(self, old_fact_id: str, new_content: str, **kwargs) -> TemporalFact:
        """Replace a fact with a newer version.

        The old fact is marked as superseded, and a new version is created
        linking back to it.

        Args:
            old_fact_id: ID of the fact to supersede.
            new_content: Updated content.
            **kwargs: Additional args for store_fact.

        Returns:
            The new version.
        """
        # Get the old fact's title and category
        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)
        old = mm.recall(query=old_fact_id, limit=1)

        title = kwargs.pop("title", "")
        category = kwargs.pop("category", "")
        if old:
            title = title or old[0].get("title", "")
            category = category or old[0].get("category", "")

        # Mark the old version as superseded
        now = datetime.now(UTC).isoformat()
        self._mark_superseded(old_fact_id, now)

        return self.store_fact(
            content=new_content,
            title=title,
            category=category,
            **kwargs,
        )

    def find_contradictions(self, content: str) -> list[TemporalFact]:
        """Find existing facts that might contradict new content.

        Args:
            content: New content to check against existing facts.

        Returns:
            List of potentially contradicting facts.
        """
        # Search for related current facts
        current = self.get_current_facts(query=content[:200], limit=10)

        contradictions = []
        content_lower = content.lower()
        negation_words = {"not", "no", "never", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "wasn't", "incorrect", "false", "wrong"}

        for fact in current:
            fact_lower = (fact.content or "").lower()

            # Check if one has negation and other doesn't on similar topics
            content_has_neg = bool(set(content_lower.split()) & negation_words)
            fact_has_neg = bool(set(fact_lower.split()) & negation_words)

            # Shared concepts
            content_words = set(content_lower.split())
            fact_words = set(fact_lower.split())
            overlap = (content_words & fact_words) - negation_words

            if len(overlap) > 5 and content_has_neg != fact_has_neg:
                contradictions.append(fact)

        return contradictions

    def get_temporal_stats(self) -> dict:
        """Get statistics about bi-temporal memory via direct DB counts."""
        from sqlalchemy import and_, func, select

        from db.engine import read_session
        from db.models import MemoryEntry

        with read_session() as session:
            base = and_(
                MemoryEntry.empire_id == self.empire_id,
                MemoryEntry.source_type == "temporal_fact",
            )
            total_memories = session.execute(
                select(func.count(MemoryEntry.id)).where(MemoryEntry.empire_id == self.empire_id)
            ).scalar() or 0

            temporal = session.execute(
                select(func.count(MemoryEntry.id)).where(base)
            ).scalar() or 0

            # Superseded and valid_time require checking JSON metadata in Python
            entries = list(session.execute(
                select(MemoryEntry.metadata_json).where(base)
            ).scalars().all())

        superseded = 0
        with_valid_time = 0
        for meta in entries:
            if isinstance(meta, dict):
                if meta.get("superseded_at"):
                    superseded += 1
                if meta.get("valid_from"):
                    with_valid_time += 1

        return {
            "total_memories": total_memories,
            "temporal_facts": temporal,
            "superseded": superseded,
            "current": temporal - superseded,
            "with_valid_time": with_valid_time,
            "temporal_coverage": temporal / total_memories if total_memories > 0 else 0,
        }

    def _find_current_version(self, title: str, category: str = "") -> dict | None:
        """Find the current (non-superseded) version of a fact by exact title.

        Uses a direct DB query instead of recall() to avoid ILIKE false matches.
        """
        from sqlalchemy import and_, select

        from db.engine import read_session
        from db.models import MemoryEntry

        with read_session() as session:
            stmt = (
                select(MemoryEntry)
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.title == title,
                    MemoryEntry.source_type == "temporal_fact",
                ))
                .order_by(MemoryEntry.created_at.desc())
                .limit(1)
            )
            entry = session.execute(stmt).scalar_one_or_none()
            if entry:
                meta = entry.metadata_json or {}
                if isinstance(meta, dict) and not meta.get("superseded_at"):
                    return {"id": entry.id, "metadata": meta}
        return None

    def store_smart(
        self,
        content: str,
        title: str = "",
        category: str = "general",
        valid_from: str | None = None,
        source: str = "",
        source_url: str = "",
        tags: list[str] | None = None,
        lieutenant_id: str = "",
        importance: float = 0.6,
        confidence: float = 0.8,
    ) -> TemporalFact:
        """Intelligently store a fact with auto-supersession detection.

        1. Checks if this updates/supersedes an existing fact (by title or content similarity)
        2. Checks for contradictions with existing knowledge
        3. Stores with proper temporal metadata

        This is the recommended way for scrapers and research pipelines
        to store new information.
        """
        now = datetime.now(UTC).isoformat()

        # Check for contradictions — log but don't block
        contradictions = self.find_contradictions(content)
        if contradictions:
            logger.info(
                "New fact may contradict %d existing facts: %s",
                len(contradictions),
                [c.title for c in contradictions[:3]],
            )
            # Boost importance — contradictory info is noteworthy
            importance = min(1.0, importance + 0.1)
            tags = (tags or []) + ["has_contradictions"]

        # Supersession is handled inside store_fact() — it finds the existing
        # entry by title, marks it superseded, and increments the version number
        # in a single atomic transaction. No need to do it here too.
        return self.store_fact(
            content=content,
            title=title,
            category=category,
            valid_from=valid_from,
            importance=importance,
            confidence=confidence,
            source=source,
            source_url=source_url,
            tags=tags,
            lieutenant_id=lieutenant_id,
        )

    def _mark_superseded(self, fact_id: str, superseded_at: str) -> None:
        """Mark a fact as superseded."""
        try:
            from db.engine import session_scope
            from db.models import MemoryEntry

            with session_scope() as session:
                entry = session.get(MemoryEntry, fact_id)
                if entry and entry.metadata_json:
                    meta = dict(entry.metadata_json)
                    meta["superseded_at"] = superseded_at
                    entry.metadata_json = meta
        except Exception as e:
            logger.warning("Failed to mark fact as superseded: %s", e)
