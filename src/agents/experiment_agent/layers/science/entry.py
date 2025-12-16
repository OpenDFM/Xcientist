"""
Science Layer Entry Point

This module provides the main entry point for the Science Agent cycle.
It orchestrates:
1. Experiment Design (Architect)
2. Experiment Execution (Manager + Workers)
3. Result Analysis (Integrator)
4. Optimization feedback loop to Code Layer
"""

import os
import json
import sys
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.shared.schemas.action_plan import (
    build_action_plan_from_science,
    action_task_id,
    ActionSource,
)
from src.agents.experiment_agent.layers.science.architect import ExpArchitectAgent
from src.agents.experiment_agent.layers.science.manager import ExpManagerAgent
from src.agents.experiment_agent.layers.science.integrator import ExpIntegratorAgent
from src.agents.experiment_agent.layers.science.schemas.experiment import (
    ExperimentPlan,
    ExperimentResult,
    ScienceAnalysis,
)

from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.utils.config import (
    ProjectContext,
    setup_openai_api,
    ensure_experiment_dirs,
)
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.layers.base.state import StateManager, GlobalPhase, StepStatus
from src.agents.experiment_agent.layers.code.schemas.manifest import CodeManifest
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint

from src.agents.experiment_agent.shared.logger import print_phase
from src.agents.experiment_agent.layers.base.state import TaskStatus as StateTaskStatus


async def run_science_cycle(
    code_manifest: CodeManifest,
    proposal: Proposal,
    experiment_id: Optional[str] = None,
    max_iterations: int = 3,
    resume: bool = False,
    quick_analysis: bool = False,
    stop_after_science_architect: bool = False,
) -> Optional[List[Dict]]:
    """
    Run the complete Science Agent cycle.

    Returns:
        List of OptimizationTickets if improvements needed,
        None if science goal achieved or failed completely.
    """
    print_phase("Science Experimentation", phase_num=1)

    project_root = code_manifest.project_root
    workspace_root = os.path.dirname(project_root)

    # Setup caching + state root (MUST be the same root for Cache and StateManager)
    if experiment_id:
        paths = ensure_experiment_dirs(experiment_id)
        workspace_root = paths["workspace_dir"]
        cache_root = paths["cache_dir"]  # workspace_root/cached
        dataset_dir = paths.get("dataset_dir")
    else:
        cache_root = os.path.join(workspace_root, "cached")
        dataset_dir = None
    Cache.initialize(cache_root, enabled=True)

    code_blueprint: Optional[Blueprint] = None
    try:
        proposal_hash = Cache.hash_proposal(proposal)
        cached_bp = Cache.get_blueprint(proposal_hash)
        if cached_bp and "blueprint" in cached_bp:
            try:
                code_blueprint = Blueprint(**cached_bp["blueprint"])
            except Exception:
                code_blueprint = None
    except Exception:
        code_blueprint = None

    # Initialize StateManager.
    # Steps are stored in a unified directory with global step_id; namespace is stored in step JSON
    # to support deterministic per-layer resume without separate folders.
    state_manager = StateManager(workspace_root, namespace="science")

    # Setup API
    if not setup_openai_api(verbose=False):
        print("✗ Failed to setup OpenAI API")
        return None

    # Initialize agents
    architect = ExpArchitectAgent()
    manager = ExpManagerAgent()
    integrator = ExpIntegratorAgent()

    # Track results
    previous_results: List[ExperimentResult] = []
    all_tickets: List[Dict] = []
    current_iteration = 0
    plan_override: Optional[ExperimentPlan] = None
    resume_incomplete_step = False
    resume_plan: Optional[ExperimentPlan] = None
    resume_plan_id: Optional[str] = None
    resume_iteration: int = 0

    # Resume from state (do NOT create an empty step; step should correspond to a real plan)
    if resume and state_manager.load() and state_manager.current_state:
        current_iteration = state_manager.current_state.meta.get("iteration", 0)
        all_tickets = state_manager.current_state.meta.get("all_tickets", [])
        phase = state_manager.current_state.phase
        print(f"Science: Resuming from iteration {current_iteration} (phase={phase})")

        # If last iteration was not completed, re-run it (resume execution should skip completed tasks)
        if (
            state_manager.current_state.status != StepStatus.COMPLETED
            and current_iteration > 0
        ):
            # We'll resume the existing step rather than creating a new step for this iteration
            resume_incomplete_step = True
            resume_iteration = current_iteration
            resume_plan_id = state_manager.current_state.blueprint_id
            if state_manager.current_data:
                try:
                    resume_plan = ExperimentPlan(**state_manager.current_data)
                except Exception:
                    resume_plan = None
            current_iteration -= 1

        cached_results = _load_cached_results(project_root)
        if cached_results:
            previous_results = cached_results
            print(f"  Loaded {len(previous_results)} previous results")

    while current_iteration < max_iterations:
        current_iteration += 1
        print_phase(f"Iteration {current_iteration}/{max_iterations}", phase_num=2)

        # Phase 1: Design Experiments
        print("\n[1/3] Designing experiments...")

        # If resuming an incomplete step, reuse the saved plan and plan_id, do not create a new step.
        if (
            resume_incomplete_step
            and current_iteration == resume_iteration
            and resume_plan_id
            and resume_plan
        ):
            plan = resume_plan
            plan_id = resume_plan_id
            print(f"✓ Resuming existing plan: {plan_id}")
        else:
            try:
                if plan_override is not None:
                    plan = plan_override
                    plan_override = None
                    print("✓ Using corrected ExperimentPlan proposed by Integrator")
                else:
                    plan = await architect.design_experiments(
                        manifest=code_manifest,
                        proposal=proposal,
                        code_blueprint=code_blueprint,
                        project_root=project_root,
                        previous_results=previous_results if previous_results else None,
                        experiment_id=experiment_id,
                        dataset_dir=dataset_dir,
                    )
            except Exception as e:
                exit_on_rate_limit(e)
                print(f"✗ Experiment design failed: {e}")
                return None

        if not plan.tasks:
            print("⚠ No experiments designed. Ending cycle.")
            break

        print(f"✓ Plan created with {len(plan.tasks)} experiments")
        for task in plan.tasks:
            deps_str = (
                f" (depends on: {', '.join(task.dependencies)})"
                if task.dependencies
                else ""
            )
            print(f"  - {task.task_id}: {task.description}{deps_str}")

        _save_plan(plan, project_root, current_iteration)

        # Only create a new step when NOT resuming an incomplete step for this iteration.
        if not (resume_incomplete_step and current_iteration == resume_iteration):
            next_step_index = state_manager.get_next_step_index()
            step4 = state_manager.format_step4(next_step_index)
            plan_id = f"exp_step_{step4}"
            task_ids = [t.task_id for t in plan.tasks]
            state_manager.init_state(
                initial_data=plan.model_dump(),
                blueprint_id=plan_id,
                task_ids=task_ids,
                step_index=next_step_index,
            )

            # Persist the raw plan to cache. Manager will compile it into executable DAG tasks
            # (e.g., render result_dir with step markers) and re-cache the compiled plan.
            Cache.set_blueprint(plan_id, plan.model_dump())

            # Update phase
            state_manager.set_phase(
                GlobalPhase.PLANNING,
                meta={"iteration": current_iteration, "plan_id": plan_id},
            )

        # Debug breakpoint: stop right after Science Architect is done and plan is persisted.
        if stop_after_science_architect:
            print(
                "\n🛑 stop-after-science-architect: exiting after Science Architect (plan persisted)."
            )
            print(
                "    Inspect cached plan JSON under: workspaces/<experiment>/cached/blueprints/exp_step_XXXX.json"
            )
            sys.exit(0)

        # Phase 2: Execute Experiments
        print("\n[2/3] Executing experiments...")

        # Phase semantics: after planning, we are executing tasks (even if they may fail).
        if state_manager.current_state:
            state_manager.set_phase(
                GlobalPhase.EXECUTION,
                meta={
                    "iteration": current_iteration,
                    "plan_id": plan_id,
                    "stage": "EXECUTION",
                },
            )

        try:
            results = await manager.execute_plan(
                plan=plan,
                project_root=project_root,
                resume=resume
                and (
                    (resume_incomplete_step and current_iteration == resume_iteration)
                    or (current_iteration == 1)
                ),
                blueprint_id=plan_id,
            )
        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ Experiment execution failed: {e}")
            return None

        previous_results.extend(results)
        _save_results(results, project_root, current_iteration)

        success_count = sum(1 for r in results if r.success)
        print(f"\n✓ Experiments complete: {success_count}/{len(results)} succeeded")

        # Phase 3: Analyze Results
        print("\n[3/3] Analyzing results...")

        try:
            if quick_analysis:
                analysis = await integrator.quick_analyze(results, plan.analysis_goal)
            else:
                analysis = await integrator.analyze_results(
                    results=results,
                    goal=plan.analysis_goal,
                    project_root=project_root,
                    proposal=proposal,
                    code_blueprint=code_blueprint,
                    plan=plan,
                )
        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ Result analysis failed: {e}")
            return None

        _save_analysis(analysis, project_root, current_iteration)

        # Persist a unified ActionPlan into the current step for audit + resume-friendly dispatch.
        # This does NOT change control-flow here; it only makes the "what to do next" explicit and durable.
        try:
            if state_manager.current_state:
                src = ActionSource(
                    namespace="science",
                    step_index=int(state_manager.current_state.step_index),
                    blueprint_id=str(state_manager.current_state.blueprint_id or ""),
                    iteration=int(current_iteration),
                )
                action_plan = build_action_plan_from_science(
                    analysis=analysis,
                    results=results,
                    source=src,
                    policy_snapshot={
                        "version": "v1",
                        "generator": "build_action_plan_from_science",
                    },
                )

                meta = dict(state_manager.current_state.meta or {})
                meta["analysis"] = analysis.model_dump()
                meta["action_plan"] = action_plan.model_dump()
                state_manager.set_phase(state_manager.current_state.phase, meta=meta)

                # Track each action as a resumable task under this step.
                for a in action_plan.actions:
                    state_manager.update_task(action_task_id(a.action_id))
        except Exception:
            pass

        # Print summary
        print(f"\n{'─'*40}")
        print("Science Analysis Summary:")
        print(f"  Success: {analysis.success}")
        print(f"  Summary: {analysis.summary}")
        if analysis.key_findings:
            print("  Key Findings:")
            for finding in analysis.key_findings:
                print(f"    - {finding}")
        print(f"{'─'*40}")

        # Check if goal achieved
        if analysis.success:
            print("\n✓ Science goal achieved!")
            state_manager.set_status(
                StepStatus.COMPLETED,
                meta={
                    "iteration": current_iteration,
                    "analysis": analysis.model_dump(),
                },
            )
            return None

        # After successfully finishing the resumed step once, clear the flag.
        if resume_incomplete_step and current_iteration == resume_iteration:
            resume_incomplete_step = False

        # Collect tickets
        if analysis.optimization_tickets:
            all_tickets.extend(analysis.optimization_tickets)
            print(
                f"\n⚠ Generated {len(analysis.optimization_tickets)} optimization tickets"
            )

        # Update phase with iteration state
        state_manager.set_phase(
            GlobalPhase.VERIFICATION,
            meta={
                "iteration": current_iteration,
                "all_tickets": all_tickets,
                "analysis": analysis.model_dump(),
            },
        )

        # Check for more experiments
        if analysis.next_experiments:
            print(
                "\n→ Integrator proposed a corrected experiment plan; continuing with that plan..."
            )
            plan_override = analysis.next_experiments
            # Mark the SCIENCE_CHANGE action as completed (idempotent resume signal).
            try:
                if state_manager.current_state and isinstance(
                    state_manager.current_state.meta, dict
                ):
                    ap = state_manager.current_state.meta.get("action_plan")
                    if isinstance(ap, dict):
                        actions = ap.get("actions") or []
                        for a in actions:
                            if (
                                isinstance(a, dict)
                                and a.get("kind") == "SCIENCE_CHANGE"
                                and a.get("action_id")
                            ):
                                state_manager.update_task(
                                    action_task_id(str(a.get("action_id"))),
                                    status=StateTaskStatus.COMPLETED,
                                    attempts=1,
                                    last_error=None,
                                )
                                break
            except Exception:
                pass
            continue

        if all_tickets:
            print(
                f"\n→ Returning {len(all_tickets)} optimization tickets to Code Layer"
            )
            break

        print("\n⚠ No more experiments or tickets. Ending cycle.")
        break

    # Final summary
    print_phase("Science Cycle Complete", phase_num=3)
    print(f"  Total iterations: {current_iteration}")
    print(f"  Total experiments run: {len(previous_results)}")
    print(f"  Optimization tickets: {len(all_tickets)}")

    if all_tickets:
        return all_tickets
    return None


def _save_plan(plan: ExperimentPlan, project_root: str, iteration: int):
    """Save experiment plan to disk."""
    output_dir = os.path.join(project_root, "_science_logs")
    os.makedirs(output_dir, exist_ok=True)

    plan_path = os.path.join(output_dir, f"plan_iter{iteration}.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))


def _save_results(results: List[ExperimentResult], project_root: str, iteration: int):
    """Save experiment results to disk."""
    output_dir = os.path.join(project_root, "_science_logs")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"results_iter{iteration}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # Save a compact summary only (stdout/stderr live in each task's result_dir)
        data = [
            {
                "task_id": r.task_id,
                "success": r.success,
                "metrics": r.metrics,
                "artifacts": r.artifacts,
                "result_dir": r.result_dir,
                "error": r.error,
            }
            for r in results
        ]
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_analysis(analysis: ScienceAnalysis, project_root: str, iteration: int):
    """Save analysis to disk."""
    output_dir = os.path.join(project_root, "_science_logs")
    os.makedirs(output_dir, exist_ok=True)

    analysis_path = os.path.join(output_dir, f"analysis_iter{iteration}.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis.model_dump_json(indent=2))


def _load_cached_results(project_root: str) -> List[ExperimentResult]:
    """Load cached results from previous runs."""
    output_dir = os.path.join(project_root, "_science_logs")
    results = []

    if not os.path.exists(output_dir):
        return results

    result_files = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if f.startswith("results_iter") and f.endswith(".json")
        ]
    )

    for result_file in result_files:
        try:
            with open(
                os.path.join(output_dir, result_file), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
                for item in data:
                    results.append(ExperimentResult(**item))
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")

    return results


async def run_single_experiment(
    task_command: str,
    project_root: str,
    task_id: str = "manual_exp",
    expected_outputs: Optional[List[str]] = None,
) -> ExperimentResult:
    """Run a single experiment directly."""
    from layers.science.schemas.experiment import ExperimentTask
    from layers.science.worker import ExpWorkerAgent

    task = ExperimentTask(
        task_id=task_id,
        command=task_command,
        description="Manual experiment",
        expected_output_files=expected_outputs or [],
    )

    worker = ExpWorkerAgent(verbose=True)
    return await worker.run_task(task=task, project_root=project_root, feedback="")


async def analyze_existing_results(
    project_root: str,
    goal: str,
) -> ScienceAnalysis:
    """Analyze existing experiment results from disk."""
    results = _load_cached_results(project_root)

    if not results:
        return ScienceAnalysis(
            success=False,
            summary="No cached results found",
            key_findings=[],
        )

    integrator = ExpIntegratorAgent()
    return await integrator.analyze_results(
        results=results,
        goal=goal,
        project_root=project_root,
    )
