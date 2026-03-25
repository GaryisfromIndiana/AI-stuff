"""Search module — web search, scraping, RSS feeds, credibility scoring, caching."""

from core.search.web import WebSearcher, SearchResult, WebSearchResponse
from core.search.scraper import WebScraper, ScrapedPage
from core.search.credibility import CredibilityScorer, CredibilityScore
from core.search.feeds import FeedReader, FeedEntry, FeedResult
from core.search.cache import ScrapeCache, ResearchDeduplicator

__all__ = [
    "WebSearcher", "SearchResult", "WebSearchResponse",
    "WebScraper", "ScrapedPage",
    "CredibilityScorer", "CredibilityScore",
    "FeedReader", "FeedEntry", "FeedResult",
    "ScrapeCache", "ResearchDeduplicator",
]
