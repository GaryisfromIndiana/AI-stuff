"""Source credibility scoring — ranks sources by trustworthiness and relevance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class CredibilityScore:
    """Credibility assessment for a source."""
    domain: str = ""
    score: float = 0.5  # 0-1, higher = more credible
    tier: str = "unknown"  # primary, authoritative, major, standard, low
    category: str = "unknown"  # research, official, press, blog, social, unknown
    reasoning: str = ""
    bias_notes: str = ""


# ── Source tiers ───────────────────────────────────────────────────────
# Tier 1: Primary sources — the labs/orgs themselves
# Tier 2: Authoritative — peer-reviewed, academic
# Tier 3: Major press — established tech journalism
# Tier 4: Standard — general news, blogs with track record
# Tier 5: Low — unknown, user-generated, aggregators

SOURCE_REGISTRY: dict[str, dict] = {
    # ── Tier 1: Primary / Official (0.95) ──────────────────────────────
    "anthropic.com":        {"score": 0.95, "tier": "primary", "category": "official", "note": "Anthropic official"},
    "openai.com":           {"score": 0.95, "tier": "primary", "category": "official", "note": "OpenAI official"},
    "deepmind.google":      {"score": 0.95, "tier": "primary", "category": "official", "note": "Google DeepMind"},
    "blog.google":          {"score": 0.93, "tier": "primary", "category": "official", "note": "Google AI blog"},
    "ai.meta.com":          {"score": 0.95, "tier": "primary", "category": "official", "note": "Meta AI official"},
    "mistral.ai":           {"score": 0.93, "tier": "primary", "category": "official", "note": "Mistral official"},
    "x.ai":                 {"score": 0.90, "tier": "primary", "category": "official", "note": "xAI official"},
    "together.ai":          {"score": 0.88, "tier": "primary", "category": "official", "note": "Together AI"},
    "huggingface.co":       {"score": 0.90, "tier": "primary", "category": "official", "note": "Hugging Face"},
    "stability.ai":         {"score": 0.88, "tier": "primary", "category": "official", "note": "Stability AI"},
    "cohere.com":           {"score": 0.88, "tier": "primary", "category": "official", "note": "Cohere official"},

    # ── Tier 2: Authoritative / Academic (0.90) ───────────────────────
    "arxiv.org":            {"score": 0.92, "tier": "authoritative", "category": "research", "note": "Preprint server — not peer-reviewed but standard for ML"},
    "openreview.net":       {"score": 0.93, "tier": "authoritative", "category": "research", "note": "Peer review platform"},
    "proceedings.neurips.cc": {"score": 0.95, "tier": "authoritative", "category": "research", "note": "NeurIPS proceedings"},
    "proceedings.mlr.press": {"score": 0.95, "tier": "authoritative", "category": "research", "note": "ICML/AISTATS proceedings"},
    "aclanthology.org":     {"score": 0.93, "tier": "authoritative", "category": "research", "note": "ACL anthology"},
    "semanticscholar.org":  {"score": 0.85, "tier": "authoritative", "category": "research", "note": "Semantic Scholar"},
    "github.com":           {"score": 0.80, "tier": "authoritative", "category": "code", "note": "Code repos — verify claims independently"},

    # ── Tier 3: Major tech press (0.75) ────────────────────────────────
    "techcrunch.com":       {"score": 0.78, "tier": "major", "category": "press", "note": "Major tech press"},
    "theverge.com":         {"score": 0.76, "tier": "major", "category": "press", "note": "Major tech press"},
    "arstechnica.com":      {"score": 0.80, "tier": "major", "category": "press", "note": "Technical journalism"},
    "wired.com":            {"score": 0.76, "tier": "major", "category": "press", "note": "Tech journalism"},
    "reuters.com":          {"score": 0.85, "tier": "major", "category": "press", "note": "Wire service — factual"},
    "apnews.com":           {"score": 0.85, "tier": "major", "category": "press", "note": "Wire service — factual"},
    "bbc.com":              {"score": 0.82, "tier": "major", "category": "press", "note": "International news"},
    "nytimes.com":          {"score": 0.80, "tier": "major", "category": "press", "note": "Major newspaper"},
    "wsj.com":              {"score": 0.80, "tier": "major", "category": "press", "note": "Business press"},
    "bloomberg.com":        {"score": 0.80, "tier": "major", "category": "press", "note": "Financial press"},
    "cnbc.com":             {"score": 0.75, "tier": "major", "category": "press", "note": "Financial news"},
    "theregister.com":      {"score": 0.75, "tier": "major", "category": "press", "note": "IT journalism"},
    "venturebeat.com":      {"score": 0.73, "tier": "major", "category": "press", "note": "AI/tech journalism"},
    "thenextweb.com":       {"score": 0.70, "tier": "major", "category": "press", "note": "Tech news"},

    # ── Tier 4: Specialist / Blogs (0.65) ──────────────────────────────
    "simonwillison.net":    {"score": 0.82, "tier": "standard", "category": "blog", "note": "Simon Willison — high quality AI commentary"},
    "lilianweng.github.io": {"score": 0.88, "tier": "standard", "category": "blog", "note": "Lilian Weng — excellent ML explainers"},
    "jalammar.github.io":   {"score": 0.85, "tier": "standard", "category": "blog", "note": "Jay Alammar — visual ML explainers"},
    "karpathy.ai":          {"score": 0.88, "tier": "standard", "category": "blog", "note": "Andrej Karpathy"},
    "unite.ai":             {"score": 0.65, "tier": "standard", "category": "press", "note": "AI news aggregator"},
    "analyticsindiamag.com": {"score": 0.55, "tier": "standard", "category": "press", "note": "AI news — verify claims"},
    "towardsdatascience.com": {"score": 0.55, "tier": "standard", "category": "blog", "note": "Medium — quality varies"},
    "medium.com":           {"score": 0.45, "tier": "standard", "category": "blog", "note": "User-generated — quality varies widely"},
    "substack.com":         {"score": 0.50, "tier": "standard", "category": "blog", "note": "Newsletters — quality varies"},
    "macrumors.com":        {"score": 0.68, "tier": "standard", "category": "press", "note": "Apple-focused tech news"},

    # ── Tier 5: Low credibility (0.30) ─────────────────────────────────
    "reddit.com":           {"score": 0.35, "tier": "low", "category": "social", "note": "User-generated — treat as opinion"},
    "twitter.com":          {"score": 0.30, "tier": "low", "category": "social", "note": "Social media — verify independently"},
    "x.com":                {"score": 0.30, "tier": "low", "category": "social", "note": "Social media — verify independently"},
    "youtube.com":          {"score": 0.35, "tier": "low", "category": "social", "note": "Video — can't extract content well"},
    "facebook.com":         {"score": 0.20, "tier": "low", "category": "social", "note": "Social media"},
    "quora.com":            {"score": 0.30, "tier": "low", "category": "social", "note": "Q&A — quality varies"},
    "msn.com":              {"score": 0.40, "tier": "low", "category": "aggregator", "note": "Aggregator — check original source"},
    "news.yahoo.com":       {"score": 0.40, "tier": "low", "category": "aggregator", "note": "Aggregator"},
}


class CredibilityScorer:
    """Scores sources by credibility for AI research.

    Uses a curated registry of known sources with manual scores,
    plus heuristics for unknown domains.
    """

    def score(self, url: str) -> CredibilityScore:
        """Score a URL's source credibility.

        Args:
            url: URL to score.

        Returns:
            CredibilityScore.
        """
        domain = self._extract_domain(url)

        # Check registry
        registry_entry = SOURCE_REGISTRY.get(domain)
        if registry_entry:
            return CredibilityScore(
                domain=domain,
                score=registry_entry["score"],
                tier=registry_entry["tier"],
                category=registry_entry["category"],
                reasoning=registry_entry.get("note", ""),
            )

        # Check partial matches (subdomains)
        for known_domain, entry in SOURCE_REGISTRY.items():
            if domain.endswith(f".{known_domain}") or known_domain.endswith(f".{domain}"):
                return CredibilityScore(
                    domain=domain,
                    score=entry["score"] * 0.9,  # Slight discount for subdomain
                    tier=entry["tier"],
                    category=entry["category"],
                    reasoning=f"Subdomain of {known_domain}",
                )

        # Heuristic scoring for unknown domains
        return self._heuristic_score(domain, url)

    def score_batch(self, urls: list[str]) -> list[CredibilityScore]:
        """Score multiple URLs."""
        return [self.score(url) for url in urls]

    def rank_urls(self, urls: list[str]) -> list[tuple[str, CredibilityScore]]:
        """Rank URLs by credibility, highest first.

        Args:
            urls: List of URLs to rank.

        Returns:
            List of (url, score) tuples, sorted by score descending.
        """
        scored = [(url, self.score(url)) for url in urls]
        scored.sort(key=lambda x: x[1].score, reverse=True)
        return scored

    def is_trustworthy(self, url: str, min_score: float = 0.5) -> bool:
        """Check if a source meets minimum credibility threshold."""
        return self.score(url).score >= min_score

    def get_weight(self, url: str) -> float:
        """Get a weight factor for synthesis (higher credibility = more weight).

        Returns a value between 0.3 and 1.0.
        """
        score = self.score(url).score
        return max(0.3, min(1.0, score))

    def format_for_prompt(self, url: str) -> str:
        """Format credibility info for LLM prompt injection."""
        cs = self.score(url)
        reliability = "highly reliable" if cs.score >= 0.85 else "reliable" if cs.score >= 0.7 else "moderately reliable" if cs.score >= 0.5 else "low reliability — verify claims"
        return f"[{cs.tier.upper()} source, {reliability}]"

    def _heuristic_score(self, domain: str, url: str) -> CredibilityScore:
        """Score unknown domains using heuristics."""
        score = 0.45  # Default for unknown
        category = "unknown"
        reasoning_parts = ["Unknown domain"]

        # .edu and .gov are generally reliable
        if domain.endswith(".edu"):
            score = 0.82
            category = "academic"
            reasoning_parts = ["Academic institution"]
        elif domain.endswith(".gov"):
            score = 0.85
            category = "government"
            reasoning_parts = ["Government source"]
        elif domain.endswith(".org"):
            score = 0.55
            category = "organization"
            reasoning_parts = ["Non-profit/organization"]

        # Known AI/ML keywords in domain boost score
        ai_keywords = ["ai", "ml", "deep", "neural", "llm", "gpt", "model", "research"]
        if any(kw in domain.lower() for kw in ai_keywords):
            score = min(score + 0.1, 0.7)
            reasoning_parts.append("AI-related domain")

        # Determine tier
        if score >= 0.8:
            tier = "authoritative"
        elif score >= 0.65 or score >= 0.45:
            tier = "standard"
        else:
            tier = "low"

        return CredibilityScore(
            domain=domain,
            score=score,
            tier=tier,
            category=category,
            reasoning="; ".join(reasoning_parts),
        )

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.replace("www.", "").lower()
        except Exception:
            return ""


def get_source_tiers() -> dict[str, list[str]]:
    """Get all sources organized by tier."""
    tiers: dict[str, list[str]] = {"primary": [], "authoritative": [], "major": [], "standard": [], "low": []}
    for domain, info in SOURCE_REGISTRY.items():
        tier = info["tier"]
        if tier in tiers:
            tiers[tier].append(domain)
    return tiers
