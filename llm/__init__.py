"""LLM client layer — providers, routing, and structured output schemas."""

from llm.base import (
    LLMClient, LLMRequest, LLMResponse, LLMMessage,
    StreamChunk, ToolCall, ToolDefinition, RateLimiter,
)
from llm.anthropic import AnthropicClient
from llm.openai import OpenAIClient
from llm.router import ModelRouter, TaskMetadata, RoutingDecision
from llm.schemas import (
    PlanningOutput, AnalysisOutput, CriticOutput,
    EntityExtractionOutput, DebateOutput, SynthesisOutput,
    ProposalOutput, ReviewOutput, ResearchOutput,
    parse_llm_output, pydantic_to_tool_schema,
)

__all__ = [
    "LLMClient", "LLMRequest", "LLMResponse", "LLMMessage",
    "StreamChunk", "ToolCall", "ToolDefinition", "RateLimiter",
    "AnthropicClient", "OpenAIClient", "ModelRouter",
    "TaskMetadata", "RoutingDecision",
    "PlanningOutput", "AnalysisOutput", "CriticOutput",
    "EntityExtractionOutput", "DebateOutput", "SynthesisOutput",
    "ProposalOutput", "ReviewOutput", "ResearchOutput",
    "parse_llm_output", "pydantic_to_tool_schema",
]
