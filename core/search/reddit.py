"""Reddit search — gives lieutenants access to Reddit posts and discussions."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)

_USER_AGENT = "Empire-AI-Research/1.0 (research bot)"


class RedditSearcher:
    """Search Reddit posts and comments.

    Uses Reddit's public JSON API (no OAuth required).
    Appends .json to URLs to get structured data.
    """

    def __init__(self, empire_id: str = ""):
        self.empire_id = empire_id

    def _api_get(self, url: str) -> Any:
        """Make a GET request to Reddit's JSON API."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("Reddit API request failed: %s", e)
            return {}

    def search(
        self,
        query: str,
        subreddit: str = "",
        max_results: int = 5,
        sort: str = "relevance",
        time_filter: str = "year",
    ) -> dict:
        """Search Reddit posts.

        Args:
            query: Search query.
            subreddit: Limit to a specific subreddit (e.g. 'MachineLearning').
            max_results: Maximum results (capped at 10).
            sort: Sort order — 'relevance', 'hot', 'top', 'new', 'comments'.
            time_filter: Time filter — 'hour', 'day', 'week', 'month', 'year', 'all'.

        Returns:
            Dict with found, summary, result_count, stored_entities.
        """
        max_results = min(max_results, 10)

        params = urllib.parse.urlencode({
            "q": query,
            "sort": sort,
            "t": time_filter,
            "limit": max_results,
            "type": "link",
            "restrict_sr": "on" if subreddit else "off",
        })

        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json?{params}"
        else:
            url = f"https://www.reddit.com/search.json?{params}"

        start = time.time()
        data = self._api_get(url)
        elapsed = (time.time() - start) * 1000

        posts = data.get("data", {}).get("children", [])
        if not posts:
            return {"found": False, "query": query, "summary": "No Reddit posts found."}

        output_parts = []
        items = []
        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "")
            sub = p.get("subreddit", "")
            score = p.get("score", 0)
            num_comments = p.get("num_comments", 0)
            selftext = (p.get("selftext") or "")[:300]
            permalink = p.get("permalink", "")
            created = p.get("created_utc", 0)
            url_link = p.get("url", "")

            # Format timestamp
            if created:
                import datetime
                dt = datetime.datetime.fromtimestamp(created, tz=datetime.timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = ""

            part = (
                f"**{title}**\n"
                f"  r/{sub} | {score:,} upvotes | {num_comments:,} comments | {date_str}"
            )
            if selftext:
                # Trim to first meaningful chunk
                preview = selftext.replace("\n", " ").strip()
                if preview:
                    part += f"\n  {preview}"
            part += f"\n  URL: https://www.reddit.com{permalink}"

            output_parts.append(part)
            items.append({
                "title": title,
                "subreddit": sub,
                "score": score,
                "num_comments": num_comments,
                "permalink": permalink,
                "selftext": selftext,
                "url": url_link,
            })

        summary = "\n\n".join(output_parts)
        logger.info("Reddit search: '%s' -> %d results (%.0fms)", query, len(items), elapsed)

        stored = self._store_results(query, summary, items)

        return {
            "found": True,
            "query": query,
            "result_count": len(items),
            "summary": summary,
            "stored_entities": stored,
            "search_time_ms": elapsed,
        }

    def get_post_comments(self, permalink: str, max_comments: int = 10) -> str:
        """Fetch top comments from a Reddit post.

        Args:
            permalink: Post permalink (e.g. '/r/MachineLearning/comments/abc123/...')
            max_comments: Maximum top-level comments to return.

        Returns:
            Formatted string of top comments.
        """
        url = f"https://www.reddit.com{permalink}.json?limit={max_comments}&sort=top"
        data = self._api_get(url)

        if not isinstance(data, list) or len(data) < 2:
            return "Could not fetch comments."

        comments_data = data[1].get("data", {}).get("children", [])
        parts = []
        for c in comments_data[:max_comments]:
            cd = c.get("data", {})
            if cd.get("body"):
                score = cd.get("score", 0)
                body = cd["body"][:500]
                parts.append(f"[{score:+,} pts] {body}")

        return "\n\n---\n\n".join(parts) if parts else "No comments found."

    def search_subreddit(
        self,
        subreddit: str,
        sort: str = "hot",
        max_results: int = 5,
    ) -> dict:
        """Browse a subreddit's top/hot/new posts.

        Args:
            subreddit: Subreddit name (without r/).
            sort: 'hot', 'new', 'top', 'rising'.
            max_results: Maximum results.

        Returns:
            Dict with found, summary, result_count.
        """
        max_results = min(max_results, 10)
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={max_results}"

        data = self._api_get(url)
        posts = data.get("data", {}).get("children", [])

        if not posts:
            return {"found": False, "query": subreddit, "summary": f"No posts found in r/{subreddit}."}

        output_parts = []
        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "")
            score = p.get("score", 0)
            num_comments = p.get("num_comments", 0)
            permalink = p.get("permalink", "")

            output_parts.append(
                f"**{title}** ({score:,} pts, {num_comments:,} comments)\n"
                f"  https://www.reddit.com{permalink}"
            )

        return {
            "found": True,
            "query": subreddit,
            "result_count": len(posts),
            "summary": "\n\n".join(output_parts),
            "stored_entities": 0,
        }

    def _store_results(self, query: str, summary: str, items: list) -> int:
        """Store search results in memory and knowledge graph."""
        stored = 0
        try:
            from core.memory.manager import MemoryManager
            mm = MemoryManager(self.empire_id)
            mm.store(
                content=f"Reddit search: {query}\n\n{summary[:3000]}",
                memory_type="semantic",
                title=f"Reddit: {query[:80]}",
                category="reddit_research",
                importance=0.6,
                tags=["reddit", "research", "discussion"],
                source_type="reddit_search",
                metadata={"query": query, "result_count": len(items)},
            )

            # Store high-signal posts as KG entities
            from core.knowledge.graph import KnowledgeGraph
            graph = KnowledgeGraph(self.empire_id)
            for item in items[:3]:
                if item.get("score", 0) >= 50:
                    title = item.get("title", "")[:200]
                    sub = item.get("subreddit", "")
                    score = item.get("score", 0)
                    graph.add_entity(
                        name=title,
                        entity_type="discussion",
                        description=f"Reddit r/{sub} post ({score:,} upvotes). {item.get('selftext', '')[:200]}",
                        confidence=min(0.9, 0.5 + (score / 1000)),
                        tags=["reddit", sub],
                    )
                    stored += 1

        except Exception as e:
            logger.warning("Failed to store Reddit results: %s", e)

        return stored
