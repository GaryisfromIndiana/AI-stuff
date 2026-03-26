"""Execution agent — the workhorse of the ACE pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.base import LLMRequest, LLMResponse, LLMMessage
from llm.router import ModelRouter, TaskMetadata

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single execution."""
    content: str = ""
    structured_output: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    model_used: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "content": self.content[:1000],
            "model": self.model_used,
            "tokens": self.tokens_input + self.tokens_output,
            "cost": self.cost_usd,
            "success": self.success,
        }


@dataclass
class ResearchResult:
    """Result of a research execution."""
    topic: str = ""
    findings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.7
    knowledge_gaps: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of an analysis execution."""
    summary: str = ""
    findings: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.7
    methodology: str = ""


@dataclass
class SynthesisResult:
    """Result of a synthesis (combining multiple inputs)."""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)


class Executor:
    """The execution agent — produces task outputs.

    Handles different task types with specialized prompts and
    post-processing. Supports research, analysis, synthesis,
    code generation, and general execution.
    """

    def __init__(self, router: ModelRouter | None = None, default_model: str = ""):
        self.router = router or ModelRouter()
        self._default_model = default_model or "claude-sonnet-4"

    def execute(
        self,
        title: str,
        description: str,
        plan: dict | None = None,
        context: str = "",
        task_type: str = "general",
        input_data: dict | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: str = "",
        previous_output: str = "",
        feedback: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute a task and produce output.

        Args:
            title: Task title.
            description: Task description.
            plan: Plan from the planning agent.
            context: System prompt / persona context.
            task_type: Type of task.
            input_data: Additional input data.
            max_tokens: Max output tokens.
            temperature: LLM temperature.
            model: Model override.
            previous_output: Previous attempt output (for retry).
            feedback: Critic feedback from previous attempt.

        Returns:
            ExecutionResult.
        """
        start = time.time()
        prompt = self._build_prompt(
            title, description, plan, task_type,
            input_data, previous_output, feedback,
        )

        model = model or self._default_model
        metadata = TaskMetadata(
            task_type=task_type,
            complexity="moderate",
            estimated_tokens=max_tokens,
        )

        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=model,
                system_prompt=context,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = self.router.execute(request, metadata)

            return ExecutionResult(
                content=response.content,
                model_used=response.model,
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
                cost_usd=response.cost_usd,
                duration_seconds=time.time() - start,
            )

        except Exception as e:
            logger.error("Execution failed: %s", e)
            return ExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def execute_research(
        self,
        topic: str,
        depth: str = "standard",
        context: str = "",
        focus_areas: list[str] | None = None,
    ) -> ResearchResult:
        """Execute a research task.

        Args:
            topic: Research topic.
            depth: Research depth (shallow, standard, deep).
            context: Additional context.
            focus_areas: Specific areas to focus on.

        Returns:
            ResearchResult.
        """
        focus = "\n".join(f"- {a}" for a in (focus_areas or []))
        prompt = f"""Conduct {depth} research on the following topic:

Topic: {topic}
{f"Focus areas:{chr(10)}{focus}" if focus else ""}

Provide:
1. Key findings (specific, factual claims)
2. Sources or basis for claims
3. A comprehensive summary
4. Knowledge gaps — what we still don't know
5. Follow-up questions for deeper research

Be thorough, specific, and cite your reasoning.
"""

        result = self.execute(
            title=f"Research: {topic}",
            description=prompt,
            context=context,
            task_type="research",
            max_tokens=6000 if depth == "deep" else 4000,
        )

        return ResearchResult(
            topic=topic,
            summary=result.content,
            findings=self._extract_list(result.content, "finding"),
            confidence=0.7 if result.success else 0.3,
        )

    def execute_analysis(
        self,
        data: str,
        framework: str = "",
        question: str = "",
        context: str = "",
    ) -> AnalysisResult:
        """Execute an analysis task.

        Args:
            data: Data to analyze.
            framework: Analysis framework to use.
            question: Specific question to answer.
            context: Additional context.

        Returns:
            AnalysisResult.
        """
        prompt = f"""Analyze the following data:

{data[:8000]}

{f"Framework: {framework}" if framework else ""}
{f"Question: {question}" if question else ""}

Provide:
1. Summary of key findings
2. Detailed analysis with evidence
3. Actionable recommendations
4. Confidence level in your analysis
5. Methodology used
"""

        result = self.execute(
            title="Analysis",
            description=prompt,
            context=context,
            task_type="analysis",
            max_tokens=5000,
        )

        return AnalysisResult(
            summary=result.content,
            confidence=0.7 if result.success else 0.3,
        )

    def execute_synthesis(
        self,
        inputs: list[str],
        goal: str = "",
        context: str = "",
    ) -> SynthesisResult:
        """Synthesize multiple inputs into a unified output.

        Args:
            inputs: List of input texts to synthesize.
            goal: Goal of the synthesis.
            context: Additional context.

        Returns:
            SynthesisResult.
        """
        formatted_inputs = "\n\n---\n\n".join(
            f"**Input {i+1}:**\n{inp[:3000]}" for i, inp in enumerate(inputs)
        )

        prompt = f"""Synthesize these inputs into a unified, coherent output:

{formatted_inputs}

{f"Goal: {goal}" if goal else ""}

Provide:
1. Unified summary combining all inputs
2. Key themes across inputs
3. Points of agreement
4. Points of disagreement or tension
5. Action items or next steps
"""

        result = self.execute(
            title="Synthesis",
            description=prompt,
            context=context,
            task_type="analysis",
            max_tokens=4000,
        )

        return SynthesisResult(summary=result.content)

    def execute_code_generation(
        self,
        spec: str,
        language: str = "python",
        context: str = "",
    ) -> ExecutionResult:
        """Generate code from a specification.

        Args:
            spec: Code specification.
            language: Programming language.
            context: Additional context.

        Returns:
            ExecutionResult with generated code.
        """
        prompt = f"""Generate {language} code based on this specification:

{spec}

Requirements:
- Production-quality code
- Include type hints and docstrings
- Handle edge cases
- Follow {language} best practices
"""

        return self.execute(
            title="Code Generation",
            description=prompt,
            context=context,
            task_type="code",
            max_tokens=8000,
            temperature=0.3,
        )

    def _build_prompt(
        self,
        title: str,
        description: str,
        plan: dict | None,
        task_type: str,
        input_data: dict | None,
        previous_output: str,
        feedback: list[str] | None,
    ) -> str:
        """Build the execution prompt."""
        parts = [f"## Task: {title}\n\n{description}"]

        if plan:
            plan_text = plan.get("plan", "")
            if plan_text:
                parts.append(f"\n## Execution Plan\n{plan_text}")

        if input_data:
            formatted = "\n".join(f"- {k}: {v}" for k, v in input_data.items())
            parts.append(f"\n## Input Data\n{formatted}")

        if previous_output and feedback:
            parts.append(f"\n## Previous Attempt\n{previous_output[:2000]}")
            fb = "\n".join(f"- {f}" for f in feedback)
            parts.append(f"\n## Feedback to Address\n{fb}")
            parts.append("\nPlease produce an improved version addressing the feedback above.")

        parts.append("\n## Instructions\nExecute this task thoroughly. Be specific, accurate, and comprehensive.")

        return "\n".join(parts)

    def _extract_list(self, text: str, item_type: str = "item") -> list[str]:
        """Extract a list of items from text."""
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "• ")):
                items.append(line[2:].strip())
            elif line and line[0].isdigit() and ". " in line:
                items.append(line.split(". ", 1)[1].strip())
        return items
