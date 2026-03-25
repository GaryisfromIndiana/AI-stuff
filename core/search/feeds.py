"""RSS feed reader — pulls from curated AI research and news feeds."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import feedparser

logger = logging.getLogger(__name__)


@dataclass
class FeedEntry:
    """A single entry from an RSS feed."""
    title: str = ""
    url: str = ""
    summary: str = ""
    author: str = ""
    published: str = ""
    source_feed: str = ""
    source_domain: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class FeedResult:
    """Result from fetching a feed."""
    feed_url: str = ""
    feed_title: str = ""
    entries: list[FeedEntry] = field(default_factory=list)
    total_entries: int = 0
    fetch_time_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class FeedConfig:
    """Configuration for an RSS feed."""
    url: str
    name: str
    domain: str = ""
    category: str = "general"  # research, official, press, blog
    credibility: float = 0.8
    enabled: bool = True


# ── Curated AI feeds ──────────────────────────────────────────────────

AI_FEEDS: list[FeedConfig] = [
    # Official AI lab blogs
    FeedConfig(
        url="https://www.anthropic.com/research/rss",
        name="Anthropic Research",
        domain="anthropic.com",
        category="official",
        credibility=0.95,
    ),
    FeedConfig(
        url="https://openai.com/blog/rss.xml",
        name="OpenAI Blog",
        domain="openai.com",
        category="official",
        credibility=0.95,
    ),
    FeedConfig(
        url="https://blog.google/technology/ai/rss/",
        name="Google AI Blog",
        domain="blog.google",
        category="official",
        credibility=0.93,
    ),
    FeedConfig(
        url="https://ai.meta.com/blog/rss/",
        name="Meta AI Blog",
        domain="ai.meta.com",
        category="official",
        credibility=0.93,
    ),
    FeedConfig(
        url="https://huggingface.co/blog/feed.xml",
        name="Hugging Face Blog",
        domain="huggingface.co",
        category="official",
        credibility=0.88,
    ),

    # Research / Academic
    FeedConfig(
        url="https://rss.arxiv.org/rss/cs.AI",
        name="arXiv cs.AI",
        domain="arxiv.org",
        category="research",
        credibility=0.92,
    ),
    FeedConfig(
        url="https://rss.arxiv.org/rss/cs.CL",
        name="arXiv cs.CL (NLP)",
        domain="arxiv.org",
        category="research",
        credibility=0.92,
    ),
    FeedConfig(
        url="https://rss.arxiv.org/rss/cs.LG",
        name="arXiv cs.LG (ML)",
        domain="arxiv.org",
        category="research",
        credibility=0.92,
    ),

    # Quality blogs
    FeedConfig(
        url="https://simonwillison.net/atom/everything/",
        name="Simon Willison",
        domain="simonwillison.net",
        category="blog",
        credibility=0.85,
    ),
    FeedConfig(
        url="https://lilianweng.github.io/index.xml",
        name="Lilian Weng",
        domain="lilianweng.github.io",
        category="blog",
        credibility=0.88,
    ),

    # Tech press
    FeedConfig(
        url="https://techcrunch.com/category/artificial-intelligence/feed/",
        name="TechCrunch AI",
        domain="techcrunch.com",
        category="press",
        credibility=0.78,
    ),
    FeedConfig(
        url="https://arstechnica.com/tag/artificial-intelligence/feed/",
        name="Ars Technica AI",
        domain="arstechnica.com",
        category="press",
        credibility=0.80,
    ),
    FeedConfig(
        url="https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        name="The Verge AI",
        domain="theverge.com",
        category="press",
        credibility=0.76,
    ),
]


class FeedReader:
    """Reads curated RSS feeds for AI news and research.

    More reliable than web scraping — feeds are designed to be read
    by machines. Entries come with titles, summaries, dates, and URLs.
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id
        self._feeds = list(AI_FEEDS)

    def fetch_feed(self, feed_config: FeedConfig, max_entries: int = 10) -> FeedResult:
        """Fetch a single RSS feed.

        Args:
            feed_config: Feed configuration.
            max_entries: Maximum entries to return.

        Returns:
            FeedResult.
        """
        start = time.time()
        result = FeedResult(feed_url=feed_config.url)

        try:
            parsed = feedparser.parse(feed_config.url)

            if parsed.bozo and not parsed.entries:
                result.success = False
                result.error = str(parsed.bozo_exception) if parsed.bozo_exception else "Feed parse error"
                result.fetch_time_ms = (time.time() - start) * 1000
                return result

            result.feed_title = parsed.feed.get("title", feed_config.name)

            for entry in parsed.entries[:max_entries]:
                # Extract tags
                tags = []
                for tag in entry.get("tags", []):
                    if isinstance(tag, dict):
                        tags.append(tag.get("term", ""))
                    elif isinstance(tag, str):
                        tags.append(tag)

                # Get published date
                published = ""
                if entry.get("published"):
                    published = entry["published"]
                elif entry.get("updated"):
                    published = entry["updated"]

                # Get summary
                summary = entry.get("summary", "")
                if not summary and entry.get("content"):
                    content_list = entry["content"]
                    if content_list and isinstance(content_list, list):
                        summary = content_list[0].get("value", "")

                # Strip HTML tags from summary
                import re
                summary = re.sub(r"<[^>]+>", "", summary).strip()
                summary = summary[:1000]

                result.entries.append(FeedEntry(
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    summary=summary,
                    author=entry.get("author", ""),
                    published=published,
                    source_feed=feed_config.name,
                    source_domain=feed_config.domain,
                    tags=tags[:5],
                ))

            result.total_entries = len(result.entries)
            result.fetch_time_ms = (time.time() - start) * 1000

            logger.info("Feed %s: %d entries (%.0fms)", feed_config.name, result.total_entries, result.fetch_time_ms)

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.fetch_time_ms = (time.time() - start) * 1000
            logger.error("Feed fetch failed for %s: %s", feed_config.name, e)

        return result

    def fetch_all(self, max_entries_per_feed: int = 5) -> list[FeedResult]:
        """Fetch all enabled feeds.

        Args:
            max_entries_per_feed: Max entries per feed.

        Returns:
            List of FeedResults.
        """
        results = []
        for feed_config in self._feeds:
            if feed_config.enabled:
                result = self.fetch_feed(feed_config, max_entries_per_feed)
                results.append(result)
        return results

    def fetch_latest(
        self,
        categories: list[str] | None = None,
        max_total: int = 20,
        max_per_feed: int = 5,
    ) -> list[FeedEntry]:
        """Fetch latest entries across all feeds, sorted by recency.

        Args:
            categories: Filter by category (official, research, press, blog).
            max_total: Maximum total entries.
            max_per_feed: Maximum per feed.

        Returns:
            List of FeedEntry sorted newest first.
        """
        all_entries: list[FeedEntry] = []

        for feed_config in self._feeds:
            if not feed_config.enabled:
                continue
            if categories and feed_config.category not in categories:
                continue

            result = self.fetch_feed(feed_config, max_per_feed)
            if result.success:
                all_entries.extend(result.entries)

        # Sort by published date (newest first)
        all_entries.sort(key=lambda e: e.published or "", reverse=True)

        return all_entries[:max_total]

    def fetch_and_store(self, max_per_feed: int = 3) -> dict:
        """Fetch all feeds and store new entries in memory + knowledge.

        Args:
            max_per_feed: Max entries per feed.

        Returns:
            Stats about what was stored.
        """
        from core.search.cache import ScrapeCache
        cache = ScrapeCache(self.empire_id)

        results = self.fetch_all(max_entries_per_feed=max_per_feed)
        total_entries = 0
        new_entries = 0
        stored_memories = 0

        from core.memory.manager import MemoryManager
        mm = MemoryManager(self.empire_id)

        for result in results:
            if not result.success:
                continue

            for entry in result.entries:
                total_entries += 1

                # Skip if we've already seen this URL
                if cache.has(entry.url):
                    continue

                new_entries += 1

                # Store in memory
                content = f"{entry.title}\n\nSource: {entry.source_feed} ({entry.source_domain})\nPublished: {entry.published}\n\n{entry.summary}"

                mm.store(
                    content=content,
                    memory_type="semantic",
                    title=f"Feed: {entry.title[:80]}",
                    category="rss_feed",
                    importance=0.55,
                    tags=["rss", entry.source_domain] + entry.tags[:3],
                    source_type="rss_feed",
                    metadata={
                        "url": entry.url,
                        "source_feed": entry.source_feed,
                        "domain": entry.source_domain,
                        "author": entry.author,
                        "published": entry.published,
                    },
                )
                stored_memories += 1

                # Mark as seen in cache (with long TTL)
                cache.put(
                    url=entry.url,
                    title=entry.title,
                    content=entry.summary,
                    domain=entry.source_domain,
                    word_count=len(entry.summary.split()),
                    ttl_hours=168,  # 7 days
                )

        logger.info("Feed sync: %d total, %d new, %d stored", total_entries, new_entries, stored_memories)

        return {
            "feeds_checked": len(results),
            "feeds_successful": sum(1 for r in results if r.success),
            "total_entries": total_entries,
            "new_entries": new_entries,
            "stored_memories": stored_memories,
        }

    def add_feed(self, url: str, name: str, category: str = "blog", credibility: float = 0.6) -> None:
        """Add a custom feed."""
        self._feeds.append(FeedConfig(
            url=url,
            name=name,
            category=category,
            credibility=credibility,
        ))

    def list_feeds(self) -> list[dict]:
        """List all configured feeds."""
        return [
            {
                "url": f.url,
                "name": f.name,
                "domain": f.domain,
                "category": f.category,
                "credibility": f.credibility,
                "enabled": f.enabled,
            }
            for f in self._feeds
        ]
