"""Text processing utilities."""

from __future__ import annotations

import json
import re
import hashlib
from typing import Any


def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from text length."""
    return max(1, len(text) // chars_per_token)


def extract_json_block(text: str) -> str | None:
    """Extract JSON from a markdown code block or mixed LLM output.

    Delegates to llm.schemas for robust extraction with brace matching.
    """
    from llm.schemas import safe_json_loads
    result = safe_json_loads(text)
    return json.dumps(result) if result else None


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def content_hash(text: str) -> str:
    """Generate a short hash of text content."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def extract_list_items(text: str) -> list[str]:
    """Extract list items from text (bulleted or numbered)."""
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            items.append(line[2:].strip())
        elif re.match(r"^\d+[\.\)]\s", line):
            items.append(re.sub(r"^\d+[\.\)]\s", "", line).strip())
    return items


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the chunk boundary
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def format_cost(cost_usd: float) -> str:
    """Format a USD cost value."""
    if cost_usd < 0.001:
        return f"${cost_usd:.6f}"
    elif cost_usd < 1.0:
        return f"${cost_usd:.4f}"
    else:
        return f"${cost_usd:.2f}"


def format_tokens(tokens: int) -> str:
    """Format a token count with K/M suffixes."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def sanitize_for_prompt(text: str, max_length: int = 10000) -> str:
    """Sanitize text for use in LLM prompts.

    Removes potentially problematic characters and truncates.
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize unicode
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # Truncate
    return truncate(text, max_length)
