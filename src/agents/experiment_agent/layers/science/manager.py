"""
Experiment Manager Agent - Centralized Coordination

Based on the paper "Towards a Science of Scaling Agent Systems":
- Implements DAG-based scheduling for experiment execution
- Parallelizes independent experiments
- Tracks experiment status and handles failures
- Reports progress and results
"""

import logging
import os
import re
from typing import List, Dict, Optional, Set

from src.agents.experiment_agent.layers.base.manager import BaseManager, TaskWrapper
from src.agents.experiment_agent.layers.base.state import GlobalPhase
from src.agents.experiment_agent.layers.science.schemas.experiment import (
    ExperimentPlan,
    ExperimentTask,
    ExperimentResult,
)
from src.agents.experiment_agent.layers.science.worker import ExpWorkerAgent
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.shared.utils.dag import TaskStatus
from src.agents.experiment_agent.shared.utils.config import (
    SCIENCE_MANAGER_MODEL,
    SCIENCE_WORKER_MODEL,
)
from src.agents.experiment_agent.shared.tools.core import SecurityContext


logger = logging.getLogger(__name__)


class ExpManagerAgent(BaseManager[ExperimentTask, ExperimentResult]):
    """
    Experiment Manager / Orchestrator Agent.

    Implements the "Centralized Coordination" pattern:
    - DAG-based scheduling respecting task dependencies
    - Parallel execution of independent experiments
    - Status tracking and failure recovery
    """

    def __init__(
        self,
        model: str = SCIENCE_MANAGER_MODEL,
        worker_model: str = SCIENCE_WORKER_MODEL,
        max_parallel_workers: int = 3,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="ExpManager",
            model=model,
            worker_model=worker_model,
            max_parallel_workers=max_parallel_workers,
            verbose=verbose,
        )

        self.project_root = None
        # state_manager is inherited from BaseManager

    def _get_task_id(self, task: ExperimentTask) -> str:
        return task.task_id

    def _get_task_dependencies(self, task: ExperimentTask) -> List[str]:
        return task.dependencies

    def _create_worker(self) -> ExpWorkerAgent:
        """Create a new worker agent."""
        return ExpWorkerAgent(
            model=self.worker_model,
            verbose=self.verbose,
        )

    def _build_system_prompt(self, **kwargs) -> str:
        return ""

    def _build_user_prompt(self, **kwargs) -> str:
        return ""

    def _compile_plan(
        self,
        plan: ExperimentPlan,
        plan_id: str,
        step_index: int,
    ) -> ExperimentPlan:
        """
        Compile a high-level experiment plan into executable DAG tasks.

        Responsibilities centralized here:
        - Normalize/clean dependencies (drop unknown deps)
        - Ensure each task has a concrete result_dir aligned with the current step
        - Cache the compiled plan for robust resume / integrator inspection
        """
        step4 = f"{step_index:04d}"
        all_task_ids = {t.task_id for t in plan.tasks}

        def _render_placeholders(text: str, values: Dict[str, str]) -> str:
            """
            Render only the supported placeholders without using str.format().

            Why: experiment commands often embed Python dict literals / braces (heredocs),
            and str.format() would treat those as format fields and break or silently skip.

            Supported placeholders (both `{k}` and `{{k}}` forms):
            - step, step4, task_id, plan_id, result_dir
            """
            if not isinstance(text, str) or not text:
                return "" if text is None else str(text)

            # Replace double-brace tokens first, then single-brace tokens.
            for k, v in values.items():
                text = text.replace(f"{{{{{k}}}}}", v)
            for k, v in values.items():
                text = text.replace(f"{{{k}}}", v)
            return text

        _placeholder_pat = re.compile(
            r"\{\{?(step4|step|task_id|plan_id|result_dir)\}?\}"
        )

        compiled_tasks: List[ExperimentTask] = []
        for t in plan.tasks:
            # Hard guardrails: obvious invalid paths should fail fast (avoid wasting retries).
            if "/tmp/" in t.command:
                raise ValueError(
                    f"Invalid experiment command for task '{t.task_id}': references /tmp. "
                    f"Commands/config/data must live under result_dir for robustness."
                )

            # Clean dependencies: keep only deps that exist within this plan
            deps = [
                d
                for d in (t.dependencies or [])
                if d in all_task_ids and d != t.task_id
            ]

            # Ensure result_dir exists and render placeholders.
            # If the blueprint provided a concrete result_dir not tied to this step, override it.
            provided = t.result_dir or ""
            has_step_placeholder = ("{step" in provided) or ("{step4" in provided)
            if (not provided) or (not has_step_placeholder):
                result_dir = "_science_runs/step_{step4}/{task_id}"
            else:
                result_dir = provided
            # Safe render (no str.format()).
            values = {
                "step": str(step_index),
                "step4": str(step4),
                "task_id": str(t.task_id),
                "plan_id": str(plan_id),
                "result_dir": "",  # filled after enforcing prefix
            }
            result_dir = _render_placeholders(str(result_dir), values)

            expected_prefix = f"_science_runs/step_{step4}/"
            expected_root = f"_science_runs/step_{step4}"
            rd_str = str(result_dir).rstrip("/")
            if not (
                rd_str == expected_root or str(result_dir).startswith(expected_prefix)
            ):
                result_dir = f"{expected_prefix}{t.task_id}"

            # Render allowed placeholders inside command safely (without str.format()).
            # This supports patterns like `_science_runs/step_{step4}/...` even inside heredocs,
            # without being confused by unrelated `{}` braces in Python dict literals.
            values["result_dir"] = str(result_dir)
            cmd = _render_placeholders(str(t.command).strip(), values)

            # Fail fast if placeholders remain: commands must be executable as-is.
            if _placeholder_pat.search(cmd):
                raise ValueError(
                    f"Unrendered placeholder(s) remain in command for task '{t.task_id}'. "
                    "Use only supported placeholders {step},{step4},{task_id},{plan_id},{result_dir} "
                    "or avoid placeholders inside command by using SCIENCE_RESULT_DIR."
                )

            # Fix common "prepare_config -> train/eval" handoff:
            # Many plans write config.yaml in the prepare task's result_dir, but execution tasks
            # often read from "$SCIENCE_RESULT_DIR/config.yaml" (their own result_dir).
            # Bridge this by copying the prepared config into the execution task result dir.
            if "$SCIENCE_RESULT_DIR/config.yaml" in cmd and deps:
                prepare_deps = [d for d in deps if str(d).startswith("prepare_config")]
                if prepare_deps:
                    src = f"_science_runs/step_{step4}/{prepare_deps[0]}/config.yaml"
                    # Use bash -lc to support env var expansion and fail-fast.
                    cmd = (
                        "bash -lc 'set -e; "
                        f'cp -f "{src}" "$SCIENCE_RESULT_DIR/config.yaml"; '
                        + cmd.replace("'", "'\"'\"'")
                        + "'"
                    )

            # Render placeholders in expected_output_files and metric_specs file paths.
            # These are used by ExpWorker to verify artifacts and extract metrics; leaving placeholders
            # unrendered will cause false failures like `_science_runs/step_{{step4}}/...` not found.
            expected_output_files: List[str] = []
            for p in t.expected_output_files or []:
                rendered = _render_placeholders(str(p).strip(), values)
                # If path is relative (e.g., "logs/metrics.jsonl"), make it relative to result_dir.
                if rendered and not rendered.startswith("_science_runs/"):
                    rd = str(result_dir).rstrip("/")
                    if rd and not rendered.startswith(rd + "/") and rendered != rd:
                        rendered = f"{rd}/{rendered.lstrip('/')}"
                if _placeholder_pat.search(rendered):
                    raise ValueError(
                        f"Unrendered placeholder(s) remain in expected_output_files for task '{t.task_id}': {rendered}"
                    )
                if rendered:
                    expected_output_files.append(rendered)

            metric_specs = []
            for ms in getattr(t, "metric_specs", None) or []:
                try:
                    ms_data = ms.model_dump() if hasattr(ms, "model_dump") else dict(ms)
                except Exception:
                    ms_data = dict(ms)
                fp = str(ms_data.get("file_path", "") or "").strip()
                fp = _render_placeholders(fp, values)
                if fp and not fp.startswith("_science_runs/"):
                    rd = str(result_dir).rstrip("/")
                    if rd and not fp.startswith(rd + "/") and fp != rd:
                        fp = f"{rd}/{fp.lstrip('/')}"
                if _placeholder_pat.search(fp):
                    raise ValueError(
                        f"Unrendered placeholder(s) remain in metric_specs.file_path for task '{t.task_id}': {fp}"
                    )
                ms_data["file_path"] = fp
                try:
                    from src.agents.experiment_agent.layers.science.schemas.experiment import (
                        MetricSpec,
                    )

                    metric_specs.append(MetricSpec(**ms_data))
                except Exception:
                    metric_specs.append(ms_data)

            compiled_tasks.append(
                ExperimentTask(
                    task_id=t.task_id,
                    command=cmd,
                    description=t.description,
                    result_dir=result_dir,
                    dependencies=deps,
                    config_overrides=t.config_overrides or {},
                    expected_output_files=expected_output_files,
                    metric_specs=metric_specs,
                )
            )

            if self.verbose and not (getattr(t, "metric_specs", None) or []):
                print(
                    f"    ⚠️ ExpManager: Task '{t.task_id}' has empty metric_specs; metrics will be empty unless artifacts are analyzed later."
                )

        compiled_plan = ExperimentPlan(
            tasks=compiled_tasks, analysis_goal=plan.analysis_goal
        )

        # Cache compiled plan under plan_id (blueprint_id)
        cached_bp = Cache.get_blueprint(plan_id)
        if (
            not cached_bp
            or "blueprint" not in cached_bp
            or cached_bp.get("blueprint") != compiled_plan.model_dump()
        ):
            Cache.set_blueprint(plan_id, compiled_plan.model_dump())

        return compiled_plan

    async def execute_plan(
        self,
        plan: ExperimentPlan,
        project_root: str,
        resume: bool = False,
        blueprint_id: Optional[str] = None,
    ) -> List[ExperimentResult]:
        """
        Execute the experiment plan using DAG-based scheduling with state persistence.

        Args:
            plan: The experiment plan
            project_root: Root directory of the project
            resume: If True, try to resume from last state
            blueprint_id: Optional ID for the plan (used for cache + state resume alignment)
        """
        self.project_root = project_root
        # Constrain tool-based file operations to this project root.
        try:
            SecurityContext.set_roots(
                project_root=os.path.abspath(project_root),
                workspace_root=os.path.dirname(os.path.abspath(project_root)),
            )
        except Exception:
            pass

        # Initialize StateManager (use base class method)
        workspace_root = os.path.dirname(self.project_root)
        self.init_state_manager(workspace_root, namespace="science")

        # Always try to load existing step to align with Science entry-created state.
        # (In Code Layer, entry creates the step, and Manager loads it.)
        self.try_resume_state()

        # 1. If resume requested and we have cached plan data, restore it
        if resume and self.state_manager and self.state_manager.current_state:
            if self.state_manager.current_data:
                try:
                    plan = ExperimentPlan(**self.state_manager.current_data)
                    print(
                        f"ExpManager: Resumed execution from Step {self.state_manager.step} "
                        f"({self.state_manager.phase})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore plan from state: {e}")

        # Resolve effective blueprint_id (prefer argument, then loaded state, then fallback)
        effective_blueprint_id = blueprint_id
        if (
            effective_blueprint_id is None
            and self.state_manager
            and self.state_manager.current_state
            and self.state_manager.current_state.blueprint_id
        ):
            effective_blueprint_id = self.state_manager.current_state.blueprint_id

        # 2. Initialize state if needed (use base class method)
        task_ids = [t.task_id for t in plan.tasks]
        # Fallback only (should normally come from Science entry.py)
        if effective_blueprint_id is None:
            effective_blueprint_id = f"exp_plan_{len(task_ids)}"

        # Ensure the plan is cached under the same blueprint_id that StateManager will reference
        cached_bp = Cache.get_blueprint(effective_blueprint_id)
        if not cached_bp or "blueprint" not in cached_bp:
            Cache.set_blueprint(effective_blueprint_id, plan.model_dump())

        self.init_state_if_needed(
            task_ids=task_ids,
            initial_data=plan.model_dump(),
            blueprint_id=effective_blueprint_id,
        )

        # Compile plan into executable DAG tasks (centralized compilation logic)
        step_index = self.state_manager.step if self.state_manager else 0
        plan = self._compile_plan(
            plan=plan,
            plan_id=effective_blueprint_id,
            step_index=step_index,
        )

        print(f"\n{'='*60}")
        print(f"ExpManager: Starting execution for {len(plan.tasks)} tasks...")
        print(f"  - Project root: {project_root}")
        print(f"  - Analysis goal: {plan.analysis_goal}")
        print(f"  - Max parallel workers: {self.max_parallel_workers}")
        print(f"{'='*60}")

        self.plan = plan  # Store for saving snapshots

        # Execute using base class DAG scheduler
        completed_ids: Optional[Set[str]] = None
        if resume:
            completed_set: Set[str] = self.get_completed_task_ids_from_state()
            if completed_set:
                completed_ids = completed_set
                self._log_info(
                    f"Science resume: skipping {len(completed_ids)} completed tasks from execution state"
                )

        await self.execute_tasks(
            tasks=plan.tasks,
            resume=resume,
            completed_ids=completed_ids,
            project_root=project_root,
            plan=plan,
        )

        # NOTE: Do NOT mark phase as COMPLETED here.
        # COMPLETED must mean "goal achieved / verification passed" and is set by the
        # Science orchestrator (layers/science/entry.py) after analysis.success.

        # Collect results
        results = []
        for task_id, wrapper in self.tasks.items():
            if wrapper.result:
                results.append(wrapper.result)
            else:
                # Task never ran
                results.append(
                    ExperimentResult(
                        task_id=task_id,
                        success=False,
                        error="Task was not executed",
                    )
                )

        return results

    async def _process_task(
        self, wrapper: TaskWrapper[ExperimentTask], **kwargs
    ) -> bool:
        """Process a single experiment task."""
        task = wrapper.task
        project_root = kwargs.get("project_root", ".")

        print(f"\n  [Task: {task.task_id}]")
        print(f"    Description: {task.description}")
        print(f"    Command: {task.command}")
        if getattr(task, "result_dir", ""):
            print(f"    Result Dir: {task.result_dir}")

        # Create worker
        worker = self._create_worker()

        # Allow a small number of in-task retries with feedback, similar to CodeManager.
        # This lets the LLM worker self-correct (e.g. create/fix configs/logs under result_dir)
        # without splitting "prepare config" and "run experiment" into separate tasks.
        wrapper.max_attempts = max(int(getattr(wrapper, "max_attempts", 1) or 1), 2)

        while wrapper.attempts < wrapper.max_attempts:
            wrapper.attempts += 1
            print(f"    [Attempt {wrapper.attempts}/{wrapper.max_attempts}]")

            feedback = ""
            if wrapper.last_error:
                feedback = f"Previous attempt failed: {wrapper.last_error}"

            try:
                result = await worker.run_task(
                    task=task,
                    project_root=project_root,
                    feedback=feedback,
                )
                wrapper.result = result
            except Exception as e:
                wrapper.last_error = f"Worker error: {str(e)}"
                print(f"    ❌ Worker error: {e}")
                continue

            if result.success:
                print("    ✅ Task completed successfully")
                if result.metrics:
                    metrics_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in result.metrics.items()
                    )
                    print(f"    Metrics: {metrics_str}")
                self.mark_task_completed(task.task_id, wrapper.attempts)
                return True

            wrapper.last_error = result.error or "Unknown error"
            print(f"    ⚠️ Task failed: {wrapper.last_error}")

        # Failed after max attempts
        self.mark_task_failed(task.task_id, wrapper.attempts, wrapper.last_error)
        return False
