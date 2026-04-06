"""Scrape cache — avoids re-fetching URLs and deduplicates research."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedPage:
    """A cached scraped page."""
    url: str = ""
    url_hash: str = ""
    title: str = ""
    content: str = ""
    domain: str = ""
    word_count: int = 0
    scraped_at: str = ""
    expires_at: str = ""
    hit_count: int = 0


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    expired: int = 0
    total_words_cached: int = 0


class ScrapeCache:
    """Caches scraped web pages to avoid redundant fetches.

    Stores scraped content in the memory system with a TTL.
    Deduplicates URLs and tracks access patterns.
    """

    CACHE_MEMORY_TYPE = "semantic"
    CACHE_CATEGORY = "scrape_cache"
    DEFAULT_TTL_HOURS = 24  # Cache pages for 24 hours

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id
        self._hits = 0
        self._misses = 0

    def get(self, url: str) -> CachedPage | None:
        """Get a cached page by URL.

        Args:
            url: URL to look up.

        Returns:
            CachedPage if found and not expired, None otherwise.
        """
        url_hash = self._hash_url(url)

        try:
            from db.engine import get_session
            from db.models import MemoryEntry
            from sqlalchemy import select, and_

            session = get_session()
            stmt = (
                select(MemoryEntry)
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.category == self.CACHE_CATEGORY,
                    MemoryEntry.metadata_json.contains(url_hash),
                ))
                .limit(1)
            )
            entry = session.execute(stmt).scalar_one_or_none()

            if entry is None:
                self._misses += 1
                return None

            # Check expiry
            if entry.expires_at and entry.expires_at < datetime.now(timezone.utc):
                self._misses += 1
                return None

            # Cache hit
            self._hits += 1
            entry.access_count += 1
            entry.last_accessed_at = datetime.now(timezone.utc)
            session.commit()

            meta = entry.metadata_json or {}
            return CachedPage(
                url=url,
                url_hash=url_hash,
                title=meta.get("title", entry.title or ""),
                content=entry.content,
                domain=meta.get("domain", ""),
                word_count=meta.get("word_count", 0),
                scraped_at=meta.get("scraped_at", ""),
                hit_count=entry.access_count,
            )

        except Exception as e:
            logger.debug("Cache lookup failed: %s", e)
            self._misses += 1
            return None

    def put(
        self,
        url: str,
        title: str,
        content: str,
        domain: str = "",
        word_count: int = 0,
        ttl_hours: int | None = None,
    ) -> None:
        """Cache a scraped page, replacing any existing entry for this URL.

        Previously this blindly inserted on every call — same URL scraped
        10 times = 10 rows. Production had 9,689 scrape_cache entries,
        most duplicates. Now does upsert: delete old entry first, then insert.
        """
        url_hash = self._hash_url(url)
        ttl = ttl_hours or self.DEFAULT_TTL_HOURS

        try:
            # Remove existing cache entry for this URL before inserting
            self.invalidate(url)

            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)
            mm.store(
                content=content[:10000],
                memory_type=self.CACHE_MEMORY_TYPE,
                title=f"Cache: {title[:80]}" if title else f"Cache: {domain}",
                category=self.CACHE_CATEGORY,
                importance=0.3,
                tags=["scrape_cache", domain],
                source_type="scrape_cache",
                expires_hours=ttl,
                metadata={
                    "url": url,
                    "url_hash": url_hash,
                    "title": title,
                    "domain": domain,
                    "word_count": word_count,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug("Cached: %s (%d words, TTL=%dh)", domain, word_count, ttl)

        except Exception as e:
            logger.warning("Failed to cache page: %s", e)

    def has(self, url: str) -> bool:
        """Check if a URL is cached (without fetching content)."""
        return self.get(url) is not None

    def invalidate(self, url: str) -> bool:
        """Remove a URL from cache."""
        url_hash = self._hash_url(url)
        try:
            from db.engine import get_session
            from db.models import MemoryEntry
            from sqlalchemy import select, and_, delete

            session = get_session()
            stmt = (
                delete(MemoryEntry)
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.category == self.CACHE_CATEGORY,
                    MemoryEntry.metadata_json.contains(url_hash),
                ))
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount > 0

        except Exception:
            return False

    def clear_expired(self) -> int:
        """Remove all expired cache entries."""
        try:
            from db.engine import get_session
            from db.models import MemoryEntry
            from sqlalchemy import delete, and_

            session = get_session()
            now = datetime.now(timezone.utc)
            stmt = (
                delete(MemoryEntry)
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.category == self.CACHE_CATEGORY,
                    MemoryEntry.expires_at.is_not(None),
                    MemoryEntry.expires_at < now,
                ))
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount

        except Exception:
            return 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total = self._hits + self._misses
        try:
            from db.engine import get_session
            from db.models import MemoryEntry
            from sqlalchemy import select, func, and_

            session = get_session()
            count = session.execute(
                select(func.count(MemoryEntry.id))
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.category == self.CACHE_CATEGORY,
                ))
            ).scalar() or 0

            return CacheStats(
                total_entries=count,
                hits=self._hits,
                misses=self._misses,
                hit_rate=self._hits / total if total > 0 else 0.0,
            )

        except Exception:
            return CacheStats(hits=self._hits, misses=self._misses)

    @staticmethod
    def _hash_url(url: str) -> str:
        """Create a hash of a URL for lookup."""
        return hashlib.sha256(url.strip().lower().encode()).hexdigest()[:16]


class ResearchDeduplicator:
    """Prevents duplicate research on the same topic.

    Tracks which topics have been researched recently and skips
    duplicates within a configurable window.
    """

    def __init__(self, empire_id: str = "", window_hours: int = 12):
        self.empire_id = empire_id
        self.window_hours = window_hours

    def was_recently_researched(self, topic: str) -> bool:
        """Check if a topic was researched recently.

        Args:
            topic: Research topic.

        Returns:
            True if topic was researched within the window.
        """
        try:
            from db.engine import get_session
            from db.models import MemoryEntry
            from sqlalchemy import select, and_

            session = get_session()
            threshold = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)
            topic_lower = topic.lower().strip()

            stmt = (
                select(func.count(MemoryEntry.id))
                .where(and_(
                    MemoryEntry.empire_id == self.empire_id,
                    MemoryEntry.category.in_(["research", "auto_research"]),
                    MemoryEntry.created_at >= threshold,
                    MemoryEntry.title.ilike(f"%{topic_lower[:50]}%"),
                ))
            )

            from sqlalchemy import func
            count = session.execute(stmt).scalar() or 0
            return count > 0

        except Exception:
            return False

    def mark_researched(self, topic: str) -> None:
        """Mark a topic as researched (done automatically by MemoryManager.store)."""
        pass  # The research pipeline already stores memories — dedup reads those
