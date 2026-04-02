"""Planning agent — the strategic brain of the ACE pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.base import LLMRequest, LLMMessage
from llm.router import ModelRouter, TaskMetadata
from llm.schemas import PlanningOutput, parse_llm_output, safe_json_loads

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A decomposed sub-task."""
    title: str
    description: str
    task_type: str = "general"
    estimated_tokens: int = 1000
    dependencies: list[str] = field(default_factory=list)
    priority: int = 5


@dataclass
class ComplexityEstimate:
    """Assessment of task difficulty."""
    level: str = "moderate"  # simple, moderate, complex, expert
    score: float = 0.5  # 0-1 scale
    reasoning: str = ""
    estimated_tokens: int = 2000
    recommended_model_tier: int = 2
    estimated_cost_usd: float = 0.01


@dataclass
class Plan:
    """Execution plan for a task."""
    goal: str = ""
    approach: str = ""
    steps: list[dict] = field(default_factory=list)
    dependencies: list[dict] = field(default_factory=list)
    complexity: ComplexityEstimate = field(default_factory=ComplexityEstimate)
    estimated_tokens: int = 5000
    model_recommendation: str = ""
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class DependencyGraph:
    """Map of task dependencies."""
    tasks: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (from, to) = from depends on to
    levels: dict[str, int] = field(default_factory=dict)  # task → depth level

    def get_execution_order(self) -> list[list[str]]:
        """Get tasks grouped by execution wave."""
        if not self.levels:
            return [self.tasks]
        max_level = max(self.levels.values()) if self.levels else 0
        waves = []
        for level in range(max_level + 1):
            wave = [t for t, l in self.levels.items() if l == level]
            if wave:
                waves.append(wave)
        return waves


@dataclass
class Wave:
    """A group of tasks that can execute in parallel."""
    number: int
    tasks: list[SubTask] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)  # Wave numbers this depends on


class Planner:
    """The planning agent — strategizes, decomposes, and estimates complexity.

    Uses LLM to analyze tasks, create execution plans, break complex tasks
    into subtasks, and suggest optimal wave structures.
    """

    def __init__(self, router: ModelRouter | None = None, default_model: str = ""):
        self.router = router or ModelRouter()
        self._default_model = default_model or "claude-sonnet-4"

    def plan_task(self, title: str, description: str, context: str = "") -> Plan:
        """Create an execution plan for a task.

        Args:
            title: Task title.
            description: Task description.
            context: Additional context (persona, domain, etc.).

        Returns:
            Execution plan.
        """
        prompt = f"""Analyze this task and create a detailed execution plan.

Task: {title}
Description: {description}
{f"Context: {context}" if context else ""}

Create a plan with:
1. Goal — what we're trying to achieve
2. Approach — high-level strategy
3. Steps — ordered execution steps (with estimated tokens each)
4. Complexity — simple/moderate/complex/expert
5. Risks — potential issues
6. Assumptions — things we're assuming

Respond as JSON:
{{
    "goal": "...",
    "approach": "...",
    "steps": [{{"step": 1, "title": "...", "description": "...", "estimated_tokens": 1000}}],
    "estimated_complexity": "moderate",
    "risks": ["..."],
    "assumptions": ["..."]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._default_model,
                system_prompt="You are a strategic planning expert. Create clear, actionable plans.",
                temperature=0.3,
                max_tokens=2000,
            )
            response = self.router.execute(request, TaskMetadata(task_type="planning", complexity="moderate"))

            parsed = parse_llm_output(response.content, PlanningOutput)
            if parsed:
                return Plan(
                    goal=parsed.goal,
                    approach=parsed.approach,
                    steps=[{"step": i + 1, "title": s.title, "description": s.description} for i, s in enumerate(parsed.steps)],
                    complexity=ComplexityEstimate(level=parsed.estimated_complexity),
                    estimated_tokens=parsed.estimated_total_tokens,
                    risks=parsed.risks,
                    assumptions=parsed.assumptions,
                    raw_output=response.content,
                )

            return Plan(goal=title, approach=response.content, raw_output=response.content)

        except Exception as e:
            logger.error("Planning failed: %s", e)
            return Plan(goal=title, approach="Direct execution (planning failed)")

    def decompose_task(self, title: str, description: str, max_subtasks: int = 10) -> list[SubTask]:
        """Break a complex task into subtasks.

        Args:
            title: Task title.
            description: Task description.
            max_subtasks: Maximum subtasks to generate.

        Returns:
            List of subtasks.
        """
        prompt = f"""Break this task into smaller, manageable subtasks.

Task: {title}
Description: {description}

Rules:
- Each subtask should be independently executable
- Identify dependencies between subtasks
- Keep subtasks focused and specific
- Maximum {max_subtasks} subtasks

Respond as JSON:
{{
    "subtasks": [
        {{
            "title": "...",
            "description": "...",
            "task_type": "research|analysis|code|creative|extraction",
            "estimated_tokens": 1000,
            "dependencies": ["title of dependency subtask"],
            "priority": 1-10
        }}
    ]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._default_model,
                temperature=0.3,
                max_tokens=2000,
            )
            response = self.router.execute(request, TaskMetadata(task_type="planning"))

            data = safe_json_loads(response.content, default={"subtasks": []})

            return [
                SubTask(
                    title=st.get("title", ""),
                    description=st.get("description", ""),
                    task_type=st.get("task_type", "general"),
                    estimated_tokens=st.get("estimated_tokens", 1000),
                    dependencies=st.get("dependencies", []),
                    priority=st.get("priority", 5),
                )
                for st in data.get("subtasks", [])[:max_subtasks]
            ]

        except Exception as e:
            logger.error("Task decomposition failed: %s", e)
            return [SubTask(title=title, description=description)]

    def estimate_complexity(self, title: str, description: str) -> ComplexityEstimate:
        """Assess task difficulty and resource requirements.

        Args:
            title: Task title.
            description: Task description.

        Returns:
            Complexity estimate.
        """
        # Heuristic-based estimation (no LLM call for speed)
        text = f"{title} {description}".lower()
        word_count = len(text.split())

        # Keyword analysis
        complex_keywords = ["analyze", "design", "architect", "optimize", "strategy", "comprehensive", "evaluate"]
        expert_keywords = ["novel", "cutting-edge", "breakthrough", "research paper", "mathematical proof"]
        simple_keywords = ["list", "summarize", "extract", "classify", "format", "convert"]

        complex_score = sum(1 for k in complex_keywords if k in text)
        expert_score = sum(1 for k in expert_keywords if k in text)
        simple_score = sum(1 for k in simple_keywords if k in text)

        if expert_score >= 2:
            level = "expert"
            score = 0.9
            tier = 1
        elif complex_score >= 2 or word_count > 200:
            level = "complex"
            score = 0.7
            tier = 2
        elif simple_score >= 2 or word_count < 30:
            level = "simple"
            score = 0.2
            tier = 4
        else:
            level = "moderate"
            score = 0.5
            tier = 3

        token_multiplier = {"simple": 1000, "moderate": 2500, "complex": 5000, "expert": 10000}

        return ComplexityEstimate(
            level=level,
            score=score,
            reasoning=f"Based on task analysis: {word_count} words, complexity indicators present",
            estimated_tokens=token_multiplier.get(level, 2500),
            recommended_model_tier=tier,
        )

    def identify_dependencies(self, subtasks: list[SubTask]) -> DependencyGraph:
        """Map dependencies between subtasks.

        Args:
            subtasks: List of subtasks.

        Returns:
            Dependency graph.
        """
        task_titles = [st.title for st in subtasks]
        edges = []
        levels: dict[str, int] = {}

        # Build edges from declared dependencies
        for st in subtasks:
            for dep in st.dependencies:
                if dep in task_titles:
                    edges.append((st.title, dep))

        # Topological sort to assign levels
        in_degree = {t: 0 for t in task_titles}
        adj: dict[str, list[str]] = {t: [] for t in task_titles}
        for from_task, to_task in edges:
            adj[to_task].append(from_task)
            in_degree[from_task] += 1

        queue = [t for t in task_titles if in_degree[t] == 0]
        level = 0
        while queue:
            for t in queue:
                levels[t] = level
            next_queue = []
            for t in queue:
                for neighbor in adj[t]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
            level += 1

        # Assign level 0 to any tasks not in the graph
        for t in task_titles:
            if t not in levels:
                levels[t] = 0

        return DependencyGraph(tasks=task_titles, edges=edges, levels=levels)

    def suggest_wave_structure(self, subtasks: list[SubTask]) -> list[Wave]:
        """Group subtasks into execution waves.

        Args:
            subtasks: List of subtasks.

        Returns:
            List of waves.
        """
        graph = self.identify_dependencies(subtasks)
        execution_order = graph.get_execution_order()
        task_map = {st.title: st for st in subtasks}

        waves = []
        for i, wave_titles in enumerate(execution_order):
            wave = Wave(
                number=i + 1,
                tasks=[task_map[t] for t in wave_titles if t in task_map],
                dependencies=list(range(1, i + 1)) if i > 0 else [],
            )
            waves.append(wave)

        return waves
