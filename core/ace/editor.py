"""Editor stage — extracts atomic facts and verifies them against independent sources.

Sits between Executor and Critic in the pipeline. Takes the executor's output
+ raw tool results and:
  1. Extracts atomic claims via LLM
  2. Cross-checks top claims using a DIFFERENT tool than the executor used
  3. Scores each claim: SUPPORTED / CONTRADICTED / UNVERIFIABLE
  4. Deduplicates against existing facts in the knowledge graph
  5. Returns verification summary for the critic + facts for KG storage
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerifiedClaim:
    """A single claim extracted and verified by the Editor."""
    claim: str
    entity_name: str = ""
    category: str = "general"  # metric, release, capability, pricing, architecture, benchmark
    source_tool: str = ""  # tool the executor used to find this
    verification_status: str = "unverified"  # unverified, supported, contradicted, unverifiable
    verification_source: str = ""  # tool/source used for verification
    verification_detail: str = ""
    confidence: float = 0.5
    importance: float = 0.5
    evidence: str = ""


@dataclass
class EditorResult:
    """Result from the Editor stage."""
    claims: list[VerifiedClaim] = field(default_factory=list)
    total_claims: int = 0
    verified_count: int = 0
    supported_count: int = 0
    contradicted_count: int = 0
    unverifiable_count: int = 0
    cost_usd: float = 0.0
    summary: str = ""

    @property
    def verification_rate(self) -> float:
        return self.verified_count / max(1, self.total_claims)

    @property
    def support_rate(self) -> float:
        return self.supported_count / max(1, self.verified_count)

    def to_critic_summary(self) -> str:
        """Format verification results for the critic prompt."""
        if not self.claims:
            return "No claims were extracted for verification."

        lines = [
            f"## Fact Verification Summary",
            f"- **{self.total_claims}** claims extracted from output",
            f"- **{self.supported_count}** supported by independent verification",
            f"- **{self.contradicted_count}** contradicted by independent sources",
            f"- **{self.unverifiable_count}** could not be independently verified",
        ]

        if self.contradicted_count > 0:
            lines.append("\n### Contradicted Claims:")
            for c in self.claims:
                if c.verification_status == "contradicted":
                    lines.append(f"- {c.claim}")
                    if c.verification_detail:
                        lines.append(f"  Reason: {c.verification_detail}")

        return "\n".join(lines)


# ── Source mapping for cross-verification ────────────────────────────

# Map executor tools to DIFFERENT verification tools
_CROSS_CHECK_MAP = {
    # If executor used HuggingFace, verify via GitHub or web search
    "mcp_huggingface_hub_repo_search": "mcp_github_search_repositories",
    "mcp_huggingface_hub_repo_details": "tavily_ai_search",
    "mcp_huggingface_paper_search": "tavily_ai_search",
    "search_huggingface": "mcp_github_search_repositories",
    # If executor used GitHub, verify via HuggingFace or web
    "mcp_github_search_repositories": "search_huggingface",
    "mcp_github_get_file_contents": "tavily_ai_search",
    "mcp_github_search_code": "tavily_ai_search",
    # If executor used web search, verify via HuggingFace or GitHub
    "tavily_search": "mcp_huggingface_hub_repo_search",
    "tavily_ai_search": "mcp_github_search_repositories",
    "tavily_news": "tavily_ai_search",
    "web_search": "tavily_ai_search",
    "search_news": "tavily_ai_search",
    # If executor used URL fetch, verify via search
    "read_url": "tavily_ai_search",
    "mcp_fetch_fetch": "tavily_ai_search",
    # Papers → web or HuggingFace
    "search_ai_papers": "mcp_huggingface_paper_search",
    "search_papers": "tavily_ai_search",
    "search_papers_with_code": "mcp_github_search_repositories",
}

# Fallback verification tool
_DEFAULT_VERIFY_TOOL = "tavily_ai_search"


def _get_verify_tool(executor_tool: str) -> str:
    """Get a different tool to cross-check a claim from the executor."""
    return _CROSS_CHECK_MAP.get(executor_tool, _DEFAULT_VERIFY_TOOL)


def _source_name_from_tool(tool_name: str) -> str:
    """Extract a human-readable source name from a tool name."""
    if "huggingface" in tool_name.lower():
        return "HuggingFace"
    if "github" in tool_name.lower():
        return "GitHub"
    if "tavily" in tool_name.lower():
        return "Tavily"
    if "fetch" in tool_name.lower():
        return "Web"
    if "arxiv" in tool_name.lower() or "paper" in tool_name.lower():
        return "Academic"
    if "reddit" in tool_name.lower():
        return "Reddit"
    if "hackernews" in tool_name.lower():
        return "HackerNews"
    return tool_name


class Editor:
    """Extracts and verifies atomic facts from executor output.

    Uses one LLM call to extract claims, then targeted tool calls
    to cross-verify the most important ones.
    """

    def __init__(
        self,
        router: Any,
        tool_registry: Any = None,
        empire_id: str = "",
        max_verify_claims: int = 7,
        model: str = "",
    ):
        self.router = router
        self.tool_registry = tool_registry
        self.empire_id = empire_id
        self.max_verify_claims = max_verify_claims
        self._model = model

    def run(
        self,
        content: str,
        tool_log: list[dict],
        task_title: str = "",
    ) -> EditorResult:
        """Run the Editor: extract claims, then verify top ones.

        Args:
            content: The executor's output text.
            tool_log: List of {tool, args, result, chars} from the executor's tool calls.
            task_title: Task title for context.

        Returns:
            EditorResult with verified claims and summary.
        """
        result = EditorResult()

        if not content or len(content) < 50:
            result.summary = "Output too short for fact extraction."
            return result

        # Step 1: Extract claims
        claims = self._extract_claims(content, tool_log, task_title)
        result.claims = claims
        result.total_claims = len(claims)

        if not claims:
            result.summary = "No verifiable claims extracted."
            return result

        # Step 2: Verify top claims (by importance, up to max_verify_claims)
        claims_to_verify = sorted(claims, key=lambda c: c.importance, reverse=True)
        claims_to_verify = claims_to_verify[:self.max_verify_claims]

        for claim in claims_to_verify:
            self._verify_claim(claim, tool_log)
            result.cost_usd += 0.0  # tool calls are cheap, tracked elsewhere

            if claim.verification_status == "supported":
                result.supported_count += 1
            elif claim.verification_status == "contradicted":
                result.contradicted_count += 1
            elif claim.verification_status == "unverifiable":
                result.unverifiable_count += 1

        result.verified_count = (
            result.supported_count + result.contradicted_count + result.unverifiable_count
        )
        result.summary = result.to_critic_summary()

        logger.info(
            "Editor: %d claims extracted, %d verified (%d supported, %d contradicted, %d unverifiable)",
            result.total_claims, result.verified_count,
            result.supported_count, result.contradicted_count, result.unverifiable_count,
        )

        return result

    def _extract_claims(
        self,
        content: str,
        tool_log: list[dict],
        task_title: str,
    ) -> list[VerifiedClaim]:
        """Use one LLM call to extract atomic claims from the output."""
        from llm.base import LLMRequest, LLMMessage
        from llm.router import TaskMetadata

        # Build tool source summary so the LLM knows which tools produced what
        tool_sources = []
        for entry in tool_log[:10]:
            tool_name = entry.get("tool", "unknown")
            source = _source_name_from_tool(tool_name)
            chars = entry.get("chars", 0)
            tool_sources.append(f"- {source} ({tool_name}): {chars} chars returned")

        sources_text = "\n".join(tool_sources) if tool_sources else "No tool calls recorded."

        prompt = f"""Extract atomic, verifiable factual claims from this research output.

## Task: {task_title}

## Sources Used
{sources_text}

## Output to Analyze
{content[:5000]}

## Instructions
Extract 5-15 key factual claims. Each claim should be:
- A single verifiable statement (not an opinion or prediction)
- Specific enough to cross-check (include numbers, names, dates)
- Tagged with the source tool that likely produced it

Respond as a JSON array:
[
  {{
    "claim": "exact factual statement",
    "entity_name": "main entity this fact is about",
    "category": "metric|release|capability|pricing|architecture|benchmark|license|adoption",
    "source_tool": "most likely tool name from the sources list above",
    "importance": 0.0-1.0,
    "evidence": "brief quote or data point supporting this"
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._model or "",
                system_prompt="You extract atomic factual claims from research output. Return valid JSON only.",
                temperature=0.1,
                max_tokens=2000,
            )
            metadata = TaskMetadata(
                task_type="extraction",
                complexity="simple",
                estimated_tokens=1500,
            )
            response = self.router.execute(request, metadata)
            self._cost = response.cost_usd

            # Parse JSON response
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            from llm.schemas import _find_json_object
            # Try direct parse first
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Try to find array
                start = raw.find("[")
                end = raw.rfind("]")
                if start >= 0 and end > start:
                    data = json.loads(raw[start:end + 1])
                else:
                    logger.warning("Editor: could not parse extraction response")
                    return []

            if not isinstance(data, list):
                return []

            claims = []
            for item in data[:15]:
                if not isinstance(item, dict):
                    continue
                claim_text = item.get("claim", "").strip()
                if not claim_text or len(claim_text) < 10:
                    continue
                claims.append(VerifiedClaim(
                    claim=claim_text,
                    entity_name=item.get("entity_name", ""),
                    category=item.get("category", "general"),
                    source_tool=item.get("source_tool", ""),
                    importance=min(1.0, max(0.0, float(item.get("importance", 0.5)))),
                    evidence=item.get("evidence", ""),
                ))

            logger.info("Editor: extracted %d claims from %d chars", len(claims), len(content))
            return claims

        except Exception as e:
            logger.error("Editor claim extraction failed: %s", e)
            return []

    def _verify_claim(self, claim: VerifiedClaim, tool_log: list[dict]) -> None:
        """Verify a single claim by querying a different tool than the executor used."""
        if not self.tool_registry:
            claim.verification_status = "unverifiable"
            claim.verification_detail = "No tool registry available"
            return

        verify_tool = _get_verify_tool(claim.source_tool)

        # Build a targeted search query from the claim
        query = claim.claim
        if claim.entity_name:
            query = f"{claim.entity_name} {claim.claim[:80]}"

        # Determine appropriate arguments based on tool type
        args = self._build_verify_args(verify_tool, query, claim)

        try:
            result_str = self.tool_registry.execute_tool_call(verify_tool, args)

            if result_str.startswith("Error:") or len(result_str) < 20:
                # Tool failed or returned nothing useful — try fallback
                if verify_tool != _DEFAULT_VERIFY_TOOL:
                    verify_tool = _DEFAULT_VERIFY_TOOL
                    args = self._build_verify_args(verify_tool, query, claim)
                    result_str = self.tool_registry.execute_tool_call(verify_tool, args)

                if result_str.startswith("Error:") or len(result_str) < 20:
                    claim.verification_status = "unverifiable"
                    claim.verification_detail = "Verification tool returned no useful data"
                    claim.verification_source = verify_tool
                    return

            # Score the verification by checking if the result supports or contradicts
            claim.verification_source = verify_tool
            self._score_verification(claim, result_str)

        except Exception as e:
            logger.warning("Editor: verification failed for claim '%s': %s", claim.claim[:50], e)
            claim.verification_status = "unverifiable"
            claim.verification_detail = str(e)

    def _build_verify_args(self, tool: str, query: str, claim: VerifiedClaim) -> dict:
        """Build appropriate arguments for a verification tool call."""
        short_query = query[:120]

        if "search_repositories" in tool:
            return {"query": short_query, "perPage": 3}
        elif "search_huggingface" in tool or "hub_repo_search" in tool:
            return {"query": claim.entity_name or short_query[:60], "type": "model", "limit": 3}
        elif "hub_repo_details" in tool:
            return {"repo_id": claim.entity_name} if claim.entity_name else {"query": short_query}
        elif "paper_search" in tool:
            return {"query": short_query, "limit": 3}
        elif "tavily" in tool:
            return {"query": short_query, "max_results": 3}
        elif "web_search" in tool:
            return {"query": short_query, "max_results": 3}
        else:
            return {"query": short_query}

    def _score_verification(self, claim: VerifiedClaim, verification_result: str) -> None:
        """Determine if verification result supports, contradicts, or is inconclusive."""
        result_lower = verification_result.lower()
        claim_lower = claim.claim.lower()

        # Extract key terms from the claim for matching
        import re
        numbers = re.findall(r'\b\d[\d,.]*[BMKbmk]?\b', claim.claim)
        key_names = [w for w in claim.claim.split() if w[0:1].isupper() and len(w) > 2]

        # Check if key terms appear in verification result
        name_matches = sum(1 for name in key_names if name.lower() in result_lower)
        number_matches = sum(1 for num in numbers if num.lower() in result_lower)

        total_key_terms = len(key_names) + len(numbers)
        if total_key_terms == 0:
            claim.verification_status = "unverifiable"
            claim.verification_detail = "No specific terms to verify"
            return

        match_rate = (name_matches + number_matches) / total_key_terms

        if match_rate >= 0.5:
            claim.verification_status = "supported"
            claim.confidence = min(1.0, claim.confidence + 0.2)
            claim.verification_detail = (
                f"Key terms confirmed in independent source ({name_matches} names, "
                f"{number_matches} numbers matched)"
            )
        elif match_rate >= 0.2:
            # Partial match — check for contradictory numbers
            contradicted = False
            for num in numbers:
                # Look for the same metric with a different value
                entity = claim.entity_name.lower() if claim.entity_name else ""
                if entity and entity in result_lower and num.lower() not in result_lower:
                    # Entity found but number doesn't match — possible contradiction
                    contradicted = True
                    break

            if contradicted:
                claim.verification_status = "contradicted"
                claim.confidence = max(0.1, claim.confidence - 0.3)
                claim.verification_detail = (
                    f"Entity found in verification source but key metrics differ"
                )
            else:
                claim.verification_status = "unverifiable"
                claim.verification_detail = "Partial match — insufficient evidence to confirm or deny"
        else:
            claim.verification_status = "unverifiable"
            claim.verification_detail = "Claim not found in verification source"
