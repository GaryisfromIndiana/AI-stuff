"""Anthropic/Claude LLM client implementation."""

from __future__ import annotations

import logging
import random
from collections.abc import Generator
from typing import Any

import anthropic

from llm.base import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Client for the Anthropic Claude API."""

    provider_name = "anthropic"
    _max_retries = 5

    def __init__(self, api_key: str | None = None):
        super().__init__()
        if api_key is None:
            from config.settings import get_settings
            api_key = get_settings().anthropic_api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self._default_model = "claude-sonnet-4-20250514"
        from llm.base import RateLimiter
        self._rate_limiter = RateLimiter(requests_per_minute=30, tokens_per_minute=80_000)

    def _call_provider(self, request: LLMRequest, model: str) -> Any:
        """Build kwargs and call the Anthropic messages API."""
        messages = self._format_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        if request.has_tools:
            kwargs["tools"] = [self._format_tool(t) for t in request.tools]
            if request.tool_choice:
                if request.tool_choice == "auto":
                    kwargs["tool_choice"] = {"type": "auto"}
                elif request.tool_choice == "required":
                    kwargs["tool_choice"] = {"type": "any"}
                elif request.tool_choice == "none":
                    pass
                else:
                    kwargs["tool_choice"] = {"type": "tool", "name": request.tool_choice}

        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        return self.client.messages.create(**kwargs)

    def _parse_response(self, raw: Any, model: str, latency_ms: float) -> LLMResponse:
        """Parse an Anthropic response into LLMResponse."""
        content = ""
        tool_calls = []

        for block in raw.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        finish_reason = "stop"
        if raw.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif raw.stop_reason == "max_tokens":
            finish_reason = "length"

        tokens_in = raw.usage.input_tokens if raw.usage else 0
        tokens_out = raw.usage.output_tokens if raw.usage else 0

        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=self._calculate_cost(model, tokens_in, tokens_out),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=raw,
        )

    def _classify_error(self, error: Exception, attempt: int) -> float | None:
        """Classify Anthropic errors for retry decisions."""
        if isinstance(error, anthropic.RateLimitError):
            base_wait = min(2 ** attempt * 5, 60)
            jitter = random.uniform(0, base_wait * 0.5)
            logger.warning("Rate limited by Anthropic, waiting %.1fs (attempt %d)", base_wait + jitter, attempt + 1)
            return base_wait + jitter
        if isinstance(error, anthropic.InternalServerError):
            wait = min(2 ** attempt * 3, 60)
            logger.warning("Anthropic server error, waiting %ds (attempt %d)", wait, attempt + 1)
            return float(wait)
        if isinstance(error, anthropic.APIStatusError):
            logger.error("Anthropic API error: %s", error)
            return None
        logger.error("Unexpected error calling Anthropic: %s", error)
        return None

    def stream(self, request: LLMRequest) -> Generator[StreamChunk, None, None]:
        """Stream a completion response from Claude."""
        model = request.model or self._default_model
        messages = self._format_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        if request.has_tools:
            kwargs["tools"] = [self._format_tool(t) for t in request.tools]

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield StreamChunk(content=event.delta.text)
                            elif hasattr(event.delta, "partial_json"):
                                yield StreamChunk(
                                    tool_call_delta={"partial_json": event.delta.partial_json}
                                )
                        elif event.type == "message_stop":
                            yield StreamChunk(is_final=True, finish_reason="stop")

                final = stream.get_final_message()
                if final and final.usage:
                    tokens_in = final.usage.input_tokens
                    tokens_out = final.usage.output_tokens
                    self._record_usage(LLMResponse(
                        content="",
                        model=model,
                        provider="anthropic",
                        tokens_input=tokens_in,
                        tokens_output=tokens_out,
                        cost_usd=self._calculate_cost(model, tokens_in, tokens_out),
                    ))

        except Exception as e:
            self._errors += 1
            logger.error("Streaming error: %s", e)
            yield StreamChunk(is_final=True, finish_reason="error")

    def _format_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Format messages for the Anthropic API.

        Anthropic uses a different format: system prompt is separate,
        and tool results have a specific structure. Consecutive tool
        results are merged into a single user message as required by
        the API.
        """
        formatted = []

        for msg in messages:
            if msg.role == "system":
                continue

            if msg.role == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                if formatted and formatted[-1]["role"] == "user" and isinstance(formatted[-1]["content"], list):
                    last_content = formatted[-1]["content"]
                    if last_content and last_content[-1].get("type") == "tool_result":
                        last_content.append(tool_result)
                        continue
                formatted.append({
                    "role": "user",
                    "content": [tool_result],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content_blocks: list[dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                formatted.append({"role": "assistant", "content": content_blocks})
            else:
                formatted.append({"role": msg.role, "content": msg.content})

        return formatted

    def _format_tool(self, tool: ToolDefinition) -> dict:
        """Format a tool definition for the Anthropic API."""
        return tool.to_anthropic_schema()

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Estimate token count for text."""
        return estimate_tokens(text)

    def count_message_tokens(
        self,
        messages: list[LLMMessage],
        system_prompt: str = "",
        model: str | None = None,
    ) -> int:
        """Estimate total tokens for a message sequence."""
        total = estimate_tokens(system_prompt) if system_prompt else 0
        for msg in messages:
            total += estimate_tokens(msg.content) + 4
        return total
