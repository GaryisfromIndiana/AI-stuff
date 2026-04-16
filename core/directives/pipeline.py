"""Directive execution pipeline — orchestrates full directive lifecycle."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Stages of the directive pipeline."""
    INTAKE = "intake"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    RETROSPECTIVE = "retrospective"
    DELIVERY = "delivery"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool = True
    output: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str = ""
    skipped: bool = False


@dataclass
class PipelineStatus:
    """Current status of the pipeline."""
    directive_id: str = ""
    current_stage: str = ""
    stages_completed: list[str] = field(default_factory=list)
    stages_remaining: list[str] = field(default_factory=list)
    progress_percent: float = 0.0
    total_cost: float = 0.0
    total_duration: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class DirectiveResult:
    """Result of the full directive pipeline."""
    directive_id: str = ""
    success: bool = False
    stages: list[StageResult] = field(default_factory=list)
    final_output: dict = field(default_factory=dict)
    quality_score: float = 0.0
    total_cost: float = 0.0
    total_duration: float = 0.0
    task_count: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    wave_count: int = 0
    retrospective: dict = field(default_factory=dict)
    executive_summary: str = ""


@dataclass
class PipelineConfig:
    """Configuration for the directive pipeline."""
    enable_planning: bool = True
    enable_review: bool = True
    enable_retrospective: bool = True
    planning_model: str = ""
    review_model: str = ""
    max_waves: int = 10
    max_tasks_per_wave: int = 10
    quality_threshold: float = 0.6
    budget_limit: float = 25.0
    skip_stages: list[str] = field(default_factory=list)

    # Hooks
    before_stage: dict[str, list[Callable]] = field(default_factory=dict)
    after_stage: dict[str, list[Callable]] = field(default_factory=dict)
    on_error: list[Callable] = field(default_factory=list)


class DirectivePipeline:
    """Orchestrates the full directive lifecycle through stages.

    Stages:
    1. INTAKE — Validate and enrich the directive
    2. PLANNING — War Room planning session, wave structure
    3. EXECUTION — Wave-by-wave task execution
    4. REVIEW — Quality review of results
    5. RETROSPECTIVE — Lessons learned
    6. DELIVERY — Final output compilation
    """

    def __init__(self, empire_id: str = "", config: PipelineConfig | None = None):
        self.empire_id = empire_id
        self.config = config or PipelineConfig()
        self._stages = [
            PipelineStage.INTAKE,
            PipelineStage.PLANNING,
            PipelineStage.EXECUTION,
            PipelineStage.REVIEW,
            PipelineStage.RETROSPECTIVE,
            PipelineStage.DELIVERY,
        ]

    def run(self, directive_id: str) -> DirectiveResult:
        """Run the full pipeline for a directive.

        Args:
            directive_id: The directive to execute.

        Returns:
            DirectiveResult.
        """
        start_time = time.time()
        result = DirectiveResult(directive_id=directive_id)

        # Load directive
        from db.engine import get_session
        from db.repositories.directive import DirectiveRepository
        session = get_session()
        repo = DirectiveRepository(session)
        directive = repo.get(directive_id)

        if not directive:
            result.stages.append(StageResult(
                stage=PipelineStage.INTAKE,
                success=False,
                error="Directive not found",
            ))
            return result

        context = {
            "directive_id": directive_id,
            "title": directive.title,
            "description": directive.description,
            "priority": directive.priority,
            "assigned_lieutenants": directive.assigned_lieutenants_json or [],
        }

        # Run each stage
        for stage in self._stages:
            if stage.value in self.config.skip_stages:
                result.stages.append(StageResult(stage=stage, skipped=True))
                continue

            # Before hooks
            for hook in self.config.before_stage.get(stage.value, []):
                try:
                    hook(context)
                except Exception as e:
                    logger.warning("Before hook error: %s", e)

            # Execute stage
            stage_start = time.time()
            try:
                stage_result = self._execute_stage(stage, context)
                stage_result.duration_seconds = time.time() - stage_start
                result.stages.append(stage_result)
                result.total_cost += stage_result.cost_usd

                # Update context with stage output
                context[stage.value] = stage_result.output

                # Update directive in DB
                repo.update(directive_id, pipeline_stage=stage.value)
                repo.commit()

                if not stage_result.success:
                    logger.error("Pipeline stage %s failed: %s", stage.value, stage_result.error)
                    break

            except Exception as e:
                stage_result = StageResult(
                    stage=stage,
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - stage_start,
                )
                result.stages.append(stage_result)

                for hook in self.config.on_error:
                    try:
                        hook(stage, e, context)
                    except Exception:
                        pass
                break

            # After hooks
            for hook in self.config.after_stage.get(stage.value, []):
                try:
                    hook(context, stage_result)
                except Exception as e:
                    logger.warning("After hook error: %s", e)

        # Finalize result
        result.total_duration = time.time() - start_time
        result.success = all(s.success for s in result.stages if not s.skipped)

        # Extract task stats from execution stage
        exec_output = context.get("execution", {})
        result.task_count = exec_output.get("total_tasks", 0)
        result.tasks_succeeded = exec_output.get("tasks_succeeded", 0)
        result.tasks_failed = exec_output.get("tasks_failed", 0)
        result.wave_count = exec_output.get("waves_completed", 0)

        # Quality from review stage
        review_output = context.get("review", {})
        result.quality_score = review_output.get("quality_score", 0)

        # Retrospective
        result.retrospective = context.get("retrospective", {})

        # Executive summary from delivery
        result.executive_summary = context.get("delivery", {}).get("summary", "")
        result.final_output = context.get("delivery", {})

        return result

    def _execute_stage(self, stage: PipelineStage, context: dict) -> StageResult:
        """Execute a single pipeline stage."""
        handlers = {
            PipelineStage.INTAKE: self._stage_intake,
            PipelineStage.PLANNING: self._stage_planning,
            PipelineStage.EXECUTION: self._stage_execution,
            PipelineStage.REVIEW: self._stage_review,
            PipelineStage.RETROSPECTIVE: self._stage_retrospective,
            PipelineStage.DELIVERY: self._stage_delivery,
        }

        handler = handlers.get(stage)
        if handler:
            return handler(context)

        return StageResult(stage=stage, success=False, error=f"Unknown stage: {stage}")

    def _stage_intake(self, context: dict) -> StageResult:
        """Intake stage — validate and enrich the directive."""
        title = context.get("title", "")
        description = context.get("description", "")

        if not title:
            return StageResult(stage=PipelineStage.INTAKE, success=False, error="Directive has no title")

        if not description:
            return StageResult(stage=PipelineStage.INTAKE, success=False, error="Directive has no description")

        # Estimate complexity
        from core.ace.planner import Planner
        planner = Planner()
        complexity = planner.estimate_complexity(title, description)

        output = {
            "title": title,
            "description": description,
            "complexity": complexity.level,
            "estimated_tokens": complexity.estimated_tokens,
            "recommended_tier": complexity.recommended_model_tier,
        }

        return StageResult(stage=PipelineStage.INTAKE, success=True, output=output)

    def _stage_planning(self, context: dict) -> StageResult:
        """Planning stage — War Room planning session."""
        if not self.config.enable_planning:
            return StageResult(stage=PipelineStage.PLANNING, skipped=True)

        try:
            from core.lieutenant.manager import LieutenantManager
            from core.warroom.session import WarRoomSession

            lt_manager = LieutenantManager(self.empire_id)
            assigned = context.get("assigned_lieutenants", [])

            if not assigned:
                available = lt_manager.list_lieutenants(status="active")
                assigned = [lt["id"] for lt in available[:5]]

            session = WarRoomSession(
                empire_id=self.empire_id,
                directive_id=context["directive_id"],
                session_type="planning",
            )

            for lt_id in assigned:
                lt_info = next((lt for lt in lt_manager.list_lieutenants() if lt["id"] == lt_id), None)
                if lt_info:
                    session.add_participant(lt_id, lt_info.get("name", ""), lt_info.get("domain", ""))

            plan_result = session.run_planning_phase(context["title"], context["description"])

            return StageResult(
                stage=PipelineStage.PLANNING,
                success=True,
                output=plan_result,
                cost_usd=session._total_cost,
            )

        except Exception as e:
            return StageResult(stage=PipelineStage.PLANNING, success=False, error=str(e))

    def _stage_execution(self, context: dict) -> StageResult:
        """Execution stage — wave-by-wave task execution."""
        try:
            from core.directives.waves import WaveExecutor

            plan = context.get("planning", {})
            unified_plan = plan.get("unified_plan", {})
            waves = unified_plan.get("waves", [])

            if not waves:
                # No waves from planning — create a single wave
                waves = [{"wave_number": 1, "tasks": [{"title": context["title"], "description": context["description"]}]}]

            executor = WaveExecutor(self.empire_id)
            results = executor.execute_all_waves(waves, context)

            total_tasks = sum(len(w.get("task_results", [])) for w in results)
            succeeded = sum(
                sum(1 for t in w.get("task_results", []) if t.get("success"))
                for w in results
            )
            total_cost = sum(w.get("cost", 0) for w in results)

            return StageResult(
                stage=PipelineStage.EXECUTION,
                success=True,
                output={
                    "wave_results": results,
                    "total_tasks": total_tasks,
                    "tasks_succeeded": succeeded,
                    "tasks_failed": total_tasks - succeeded,
                    "waves_completed": len(results),
                    "total_cost": total_cost,
                },
                cost_usd=total_cost,
            )

        except Exception as e:
            return StageResult(stage=PipelineStage.EXECUTION, success=False, error=str(e))

    def _stage_review(self, context: dict) -> StageResult:
        """Review stage — quality review of results."""
        if not self.config.enable_review:
            return StageResult(stage=PipelineStage.REVIEW, skipped=True)

        try:
            exec_output = context.get("execution", {})
            wave_results = exec_output.get("wave_results", [])

            # Aggregate quality scores
            all_quality = []
            for wave in wave_results:
                for task in wave.get("task_results", []):
                    if task.get("quality_score"):
                        all_quality.append(task["quality_score"])

            avg_quality = sum(all_quality) / len(all_quality) if all_quality else 0.5
            success_rate = exec_output.get("tasks_succeeded", 0) / max(exec_output.get("total_tasks", 1), 1)

            return StageResult(
                stage=PipelineStage.REVIEW,
                success=True,
                output={
                    "quality_score": avg_quality,
                    "success_rate": success_rate,
                    "meets_threshold": avg_quality >= self.config.quality_threshold,
                },
            )

        except Exception as e:
            return StageResult(stage=PipelineStage.REVIEW, success=False, error=str(e))

    def _stage_retrospective(self, context: dict) -> StageResult:
        """Retrospective stage — extract lessons learned."""
        if not self.config.enable_retrospective:
            return StageResult(stage=PipelineStage.RETROSPECTIVE, skipped=True)

        try:
            from core.warroom.retrospective import RetrospectiveEngine

            engine = RetrospectiveEngine(self.empire_id)
            exec_output = context.get("execution", {})

            retro = engine.run_retrospective(context["title"], exec_output)

            # Feed back to memory
            assigned = context.get("assigned_lieutenants", [])
            if assigned:
                engine.feed_back_to_memory(retro, assigned)

            return StageResult(
                stage=PipelineStage.RETROSPECTIVE,
                success=True,
                output={
                    "what_went_well": retro.what_went_well,
                    "what_went_wrong": retro.what_went_wrong,
                    "lessons": retro.lessons_learned,
                    "improvements": retro.improvements,
                    "action_items": retro.action_items,
                    "effectiveness": retro.effectiveness_score,
                },
                cost_usd=retro.cost_usd,
            )

        except Exception as e:
            return StageResult(stage=PipelineStage.RETROSPECTIVE, success=False, error=str(e))

    def _stage_delivery(self, context: dict) -> StageResult:
        """Delivery stage — compile and format final output."""
        try:
            from core.warroom.synthesis import Synthesizer

            synth = Synthesizer()
            exec_output = context.get("execution", {})

            exec_summary = synth.create_executive_summary(context["title"], exec_output)

            return StageResult(
                stage=PipelineStage.DELIVERY,
                success=True,
                output={
                    "summary": exec_summary.headline,
                    "key_findings": exec_summary.key_findings,
                    "recommendations": exec_summary.recommendations,
                    "next_steps": exec_summary.next_steps,
                },
                cost_usd=exec_summary.cost,
            )

        except Exception as e:
            return StageResult(stage=PipelineStage.DELIVERY, success=False, error=str(e))

    def get_status(self, directive_id: str) -> PipelineStatus:
        """Get current pipeline status for a directive."""
        try:
            from db.engine import get_session
            from db.repositories.directive import DirectiveRepository
            session = get_session()
            repo = DirectiveRepository(session)
            directive = repo.get(directive_id)

            if not directive:
                return PipelineStatus(directive_id=directive_id)

            current = directive.pipeline_stage
            stage_names = [s.value for s in self._stages]
            current_idx = stage_names.index(current) if current in stage_names else 0

            return PipelineStatus(
                directive_id=directive_id,
                current_stage=current,
                stages_completed=stage_names[:current_idx],
                stages_remaining=stage_names[current_idx + 1:],
                progress_percent=(current_idx + 1) / len(stage_names) * 100,
                total_cost=directive.total_cost_usd,
            )

        except Exception:
            return PipelineStatus(directive_id=directive_id)
