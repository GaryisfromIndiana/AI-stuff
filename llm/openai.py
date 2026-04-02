"""OpenAI LLM client implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Generator, Optional

import openai

from llm.base import (
    LLMClient, LLMRequest, LLMResponse, LLMMessage,
    StreamChunk, ToolCall, ToolDefinition, estimate_tokens,
)

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Client for the OpenAI API."""

    provider_name = "openai"
    _max_retries = 3

    def __init__(self, api_key: str | None = None):
        super().__init__()
        if api_key is None:
            from config.settings import get_settings
            api_key = get_settings().openai_api_key
        self.client = openai.OpenAI(api_key=api_key)
        self._default_model = "gpt-4o"

    def _call_provider(self, request: LLMRequest, model: str) -> Any:
        """Build kwargs and call the OpenAI chat completions API."""
        messages = self._format_messages(request.messages, request.system_prompt)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        if request.has_tools:
            kwargs["tools"] = [self._format_tool(t) for t in request.tools]
            if request.tool_choice:
                if request.tool_choice in ("auto", "none", "required"):
                    kwargs["tool_choice"] = request.tool_choice
                else:
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice},
                    }

        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        return self.client.chat.completions.create(**kwargs)

    def _parse_response(self, raw: Any, model: str, latency_ms: float) -> LLMResponse:
        """Parse an OpenAI response into LLMResponse."""
        choice = raw.choices[0]
        content = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        finish_reason = choice.finish_reason or "stop"

        tokens_in = raw.usage.prompt_tokens if raw.usage else 0
        tokens_out = raw.usage.completion_tokens if raw.usage else 0

        return LLMResponse(
            content=content,
            model=model,
            provider="openai",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=self._calculate_cost(model, tokens_in, tokens_out),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=raw,
        )

    def _classify_error(self, error: Exception, attempt: int) -> Optional[float]:
        """Classify OpenAI errors for retry decisions."""
        if isinstance(error, openai.RateLimitError):
            wait = min(2 ** attempt * 2, 30)
            logger.warning("Rate limited by OpenAI, waiting %ds (attempt %d)", wait, attempt + 1)
            return float(wait)
        if isinstance(error, openai.InternalServerError):
            wait = min(2 ** attempt * 3, 60)
            logger.warning("OpenAI server error, waiting %ds (attempt %d)", wait, attempt + 1)
            return float(wait)
        if isinstance(error, openai.APIStatusError):
            logger.error("OpenAI API error: %s", error)
            return None
        logger.error("Unexpected error calling OpenAI: %s", error)
        return None

    def stream(self, request: LLMRequest) -> Generator[StreamChunk, None, None]:
        """Stream a completion response from OpenAI."""
        model = request.model or self._default_model
        messages = self._format_messages(request.messages, request.system_prompt)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if request.has_tools:
            kwargs["tools"] = [self._format_tool(t) for t in request.tools]

        try:
            stream = self.client.chat.completions.create(**kwargs)
            total_content = ""

            for chunk in stream:
                if not chunk.choices:
                    if chunk.usage:
                        tokens_in = chunk.usage.prompt_tokens
                        tokens_out = chunk.usage.completion_tokens
                        self._record_usage(LLMResponse(
                            content=total_content,
                            model=model,
                            provider="openai",
                            tokens_input=tokens_in,
                            tokens_output=tokens_out,
                            cost_usd=self._calculate_cost(model, tokens_in, tokens_out),
                        ))
                    continue

                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason

                if delta and delta.content:
                    total_content += delta.content
                    yield StreamChunk(content=delta.content)

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        yield StreamChunk(tool_call_delta={
                            "index": tc_delta.index,
                            "id": tc_delta.id,
                            "name": tc_delta.function.name if tc_delta.function else None,
                            "arguments": tc_delta.function.arguments if tc_delta.function else None,
                        })

                if finish:
                    yield StreamChunk(is_final=True, finish_reason=finish)

        except Exception as e:
            self._errors += 1
            logger.error("OpenAI streaming error: %s", e)
            yield StreamChunk(is_final=True, finish_reason="error")

    def _format_messages(
        self,
        messages: list[LLMMessage],
        system_prompt: str = "",
    ) -> list[dict]:
        """Format messages for the OpenAI API."""
        formatted = []

        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "system":
                formatted.append({"role": "system", "content": msg.content})
            elif msg.role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                tool_calls_formatted = []
                for tc in msg.tool_calls:
                    tool_calls_formatted.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    })
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls_formatted,
                }
                if msg.content:
                    entry["content"] = msg.content
                formatted.append(entry)
            else:
                formatted.append({"role": msg.role, "content": msg.content})

        return formatted

    def _format_tool(self, tool: ToolDefinition) -> dict:
        """Format a tool definition for the OpenAI API."""
        return tool.to_openai_schema()

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Estimate token count using character-based heuristic."""
        return estimate_tokens(text)

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        """Create an embedding vector for text."""
        try:
            response = self.client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error("Embedding error: %s", e)
            raise

    def create_embeddings_batch(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """Create embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as e:
            logger.error("Batch embedding error: %s", e)
            raise
