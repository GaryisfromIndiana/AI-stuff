"""Unit tests for the LLM base class template method (complete with retries)."""

from __future__ import annotations

from typing import Any, Generator, Optional
from unittest.mock import MagicMock

import pytest

from llm.base import (
    LLMClient, LLMRequest, LLMResponse, LLMMessage,
    RateLimiter, StreamChunk,
)


class FakeClient(LLMClient):
    """Minimal LLMClient subclass for testing the template method."""

    provider_name = "fake"
    _default_model = "fake-model-1"
    _max_retries = 3

    def __init__(self):
        super().__init__()
        self._rate_limiter = RateLimiter(requests_per_minute=1000, tokens_per_minute=1_000_000)
        self.call_count = 0
        self.raw_response = {"text": "hello"}
        self.error_sequence: list[Exception | None] = []

    def _call_provider(self, request: LLMRequest, model: str) -> Any:
        self.call_count += 1
        if self.error_sequence:
            err = self.error_sequence.pop(0)
            if err is not None:
                raise err
        return self.raw_response

    def _parse_response(self, raw: Any, model: str, latency_ms: float) -> LLMResponse:
        return LLMResponse(
            content=raw["text"],
            model=model,
            provider="fake",
            tokens_input=10,
            tokens_output=5,
            cost_usd=0.001,
            latency_ms=latency_ms,
        )

    def _classify_error(self, error: Exception, attempt: int) -> Optional[float]:
        if isinstance(error, ConnectionError):
            return 0.0  # Retry immediately
        return None  # Fatal

    def stream(self, request: LLMRequest) -> Generator[StreamChunk, None, None]:
        yield StreamChunk(content="hello", is_final=True)


def _simple_request() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage.user("test")], max_tokens=100)


def test_complete_succeeds_on_first_try():
    """Normal case — no retries needed."""
    client = FakeClient()
    resp = client.complete(_simple_request())
    assert resp.content == "hello"
    assert resp.model == "fake-model-1"
    assert client.call_count == 1
    assert client._errors == 0


def test_complete_retries_on_retryable_error():
    """Retryable errors trigger retries, then succeed."""
    client = FakeClient()
    client.error_sequence = [ConnectionError("timeout"), None]

    resp = client.complete(_simple_request())
    assert resp.content == "hello"
    assert client.call_count == 2
    assert client._errors == 0


def test_complete_raises_on_fatal_error():
    """Fatal errors propagate immediately without retry."""
    client = FakeClient()
    client.error_sequence = [TypeError("bad type")]

    with pytest.raises(TypeError, match="bad type"):
        client.complete(_simple_request())

    assert client.call_count == 1
    assert client._errors == 1


def test_complete_exhausts_retries():
    """All retries exhausted raises the last error."""
    client = FakeClient()
    client.error_sequence = [
        ConnectionError("fail 1"),
        ConnectionError("fail 2"),
        ConnectionError("fail 3"),
    ]

    with pytest.raises(ConnectionError, match="fail 3"):
        client.complete(_simple_request())

    assert client.call_count == 3
    assert client._errors == 1


def test_complete_records_usage():
    """Successful calls record usage stats."""
    client = FakeClient()
    client.complete(_simple_request())

    assert client._total_requests == 1
    assert client._total_tokens == 15  # 10 in + 5 out
    assert client._total_cost == 0.001


def test_complete_uses_request_model_override():
    """Request.model overrides the default model."""
    client = FakeClient()
    req = LLMRequest(messages=[LLMMessage.user("test")], model="custom-model", max_tokens=100)
    resp = client.complete(req)
    assert resp.model == "custom-model"


def test_max_retries_configurable():
    """_max_retries controls how many attempts are made."""
    client = FakeClient()
    client._max_retries = 5
    client.error_sequence = [ConnectionError()] * 4 + [None]

    resp = client.complete(_simple_request())
    assert resp.content == "hello"
    assert client.call_count == 5
