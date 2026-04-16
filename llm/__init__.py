"""LLM client layer — providers, routing, and structured output schemas."""

from llm.anthropic import AnthropicClient
from llm.base import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    RateLimiter,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)
from llm.cache import LLMCache, cache_llm_response, get_cache, get_cached_response
from llm.openai import OpenAIClient
from llm.router import (
    ModelRouter,
    RoutingDecision,
    SpendRecorder,
    SpendRecorderFactory,
    TaskMetadata,
    register_spend_recorder_factory,
)
from llm.schemas import (
    AnalysisOutput,
    CriticOutput,
    DebateOutput,
    EntityExtractionOutput,
    PlanningOutput,
    ProposalOutput,
    ResearchOutput,
    ReviewOutput,
    SynthesisOutput,
    parse_llm_output,
    pydantic_to_tool_schema,
)

__all__ = [
    "AnalysisOutput",
    "AnthropicClient",
    "CriticOutput",
    "DebateOutput",
    "EntityExtractionOutput",
    "LLMCache",
    "LLMClient",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "ModelRouter",
    "OpenAIClient",
    "PlanningOutput",
    "ProposalOutput",
    "RateLimiter",
    "ResearchOutput",
    "ReviewOutput",
    "RoutingDecision",
    "SpendRecorder",
    "SpendRecorderFactory",
    "StreamChunk",
    "SynthesisOutput",
    "TaskMetadata",
    "ToolCall",
    "ToolDefinition",
    "cache_llm_response",
    "get_cache",
    "get_cached_response",
    "parse_llm_output",
    "pydantic_to_tool_schema",
    "register_spend_recorder_factory",
]
