"""Unit tests for safe_json_loads — the consolidated JSON extraction utility."""

from __future__ import annotations

from llm.schemas import safe_json_loads


def test_parses_clean_json():
    """Direct JSON string parses cleanly."""
    result = safe_json_loads('{"key": "value", "count": 3}')
    assert result == {"key": "value", "count": 3}


def test_extracts_from_markdown_block():
    """JSON inside a markdown code block is extracted."""
    text = 'Here is the result:\n```json\n{"action": "RESEARCH", "topic": "AI"}\n```\nDone.'
    result = safe_json_loads(text)
    assert result["action"] == "RESEARCH"
    assert result["topic"] == "AI"


def test_extracts_from_mixed_text():
    """JSON embedded in LLM prose is found via brace matching."""
    text = 'I think the answer is:\n\n{"score": 0.85, "approved": true}\n\nHope that helps!'
    result = safe_json_loads(text)
    assert result["score"] == 0.85
    assert result["approved"] is True


def test_returns_default_on_invalid_json():
    """Completely invalid text returns the default."""
    result = safe_json_loads("This is not JSON at all", default={"fallback": True})
    assert result == {"fallback": True}


def test_returns_empty_dict_when_no_default():
    """No default specified returns empty dict."""
    result = safe_json_loads("not json")
    assert result == {}


def test_handles_empty_string():
    """Empty string returns default."""
    result = safe_json_loads("")
    assert result == {}


def test_handles_nested_braces():
    """Nested JSON objects are parsed correctly."""
    text = '{"outer": {"inner": "value"}, "list": [1, 2]}'
    result = safe_json_loads(text)
    assert result["outer"]["inner"] == "value"
    assert result["list"] == [1, 2]


def test_handles_json_with_trailing_text():
    """JSON followed by non-JSON text is extracted."""
    text = '{"action": "SWEEP"}\n\nLet me know if you need anything else.'
    result = safe_json_loads(text)
    assert result["action"] == "SWEEP"
