"""Search module — web search, scraping, RSS, credibility, caching, intelligence sweep."""

from core.search.cache import ResearchDeduplicator, ScrapeCache
from core.search.credibility import CredibilityScore, CredibilityScorer
from core.search.feeds import FeedEntry, FeedReader, FeedResult
from core.search.scraper import ScrapedPage, WebScraper
from core.search.sweep import Discovery, IntelligenceSweep, SweepResult
from core.search.web import SearchResult, WebSearcher, WebSearchResponse

__all__ = [
    "CredibilityScore",
    "CredibilityScorer",
    "Discovery",
    "FeedEntry",
    "FeedReader",
    "FeedResult",
    "IntelligenceSweep",
    "ResearchDeduplicator",
    "ScrapeCache",
    "ScrapedPage",
    "SearchResult",
    "SweepResult",
    "WebScraper",
    "WebSearchResponse",
    "WebSearcher",
]
