"""
Science Layer Entry Point (simplified, non-versioned).

The iteration loop is:
1) Architect writes plan.md (background + completed + next plan) and tasks.md (history + current tasks)
2) Worker executes the full iteration from tasks.md (Markdown, no parser)
3) Integrator analyzes results, appends to report.md, and overwrites feedback.md

All source-of-truth docs stored under:
- <workspace_root>/cached/science/

All results stored under:
- <project_root>/result/science/
"""

import os
import sys
from typing import Dict, List, Optional

from src.agents.experiment_agent.layers.base.state import (
    GlobalPhase,
    StateManager,
    StepStatus,
    TaskStatus,
)
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.layers.code.schemas.manifest import CodeManifest
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.layers.science.architect import ExpArchitectAgent
from src.agents.experiment_agent.layers.science.docs import (
    ScienceDocPaths,
    snapshot_idea_to_cache,
    sync_science_docs_to_specs,
)
from src.agents.experiment_agent.layers.science.integrator import ExpIntegratorAgent
from src.agents.experiment_agent.layers.science.schemas.experiment import (
    ScienceAnalysis,
)
from src.agents.experiment_agent.layers.science.worker import ExpWorkerAgent
from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.logger import print_phase
from src.agents.experiment_agent.shared.tools.core import SecurityContext
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.shared.utils.config import (
    ensure_experiment_dirs,
    setup_openai_api,
)


async def run_science_cycle(
    code_manifest: CodeManifest,
    proposal: Proposal,
    experiment_id: Optional[str] = None,
    max_iterations: int = 5,
    resume: bool = False,
) -> Optional[List[Dict]]:
    print_phase("Science Experimentation (simplified)", phase_num=1)

    project_root = code_manifest.project_root
    workspace_root = os.path.dirname(project_root)

    if experiment_id:
        paths = ensure_experiment_dirs(experiment_id)
        workspace_root = paths["workspace_dir"]
        cache_root = paths["cache_dir"]
        dataset_dir = paths.get("dataset_dir")
        idea_input_path = paths.get("idea_input")
    else:
        cache_root = os.path.join(workspace_root, "cached")
        dataset_dir = None
        idea_input_path = None

    Cache.initialize(cache_root, enabled=True)
    try:
        SecurityContext.set_roots(
            project_root=os.path.abspath(project_root),
            workspace_root=os.path.abspath(workspace_root),
        )
    except Exception:
        pass

    if not setup_openai_api(verbose=False):
        print("✗ Failed to setup OpenAI API")
        return None

    doc_paths = ScienceDocPaths(workspace_root=workspace_root, cache_root=cache_root)
    doc_paths.ensure_dirs()

    proposal_text = ""
    try:
        proposal_text = (
            "# Idea\n\n"
            + "## Title\n"
            + str(proposal.idea.title or "")
            + "\n\n## Description\n"
            + str(proposal.idea.description or "")
            + "\n"
        )
    except Exception:
        proposal_text = ""
    snapshot_idea_to_cache(
        idea_input_path=idea_input_path,
        proposal_text=proposal_text,
        doc_paths=doc_paths,
    )

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

    state_manager = StateManager(workspace_root, namespace="science")
    architect = ExpArchitectAgent()
    worker = ExpWorkerAgent(verbose=True)
    integrator = ExpIntegratorAgent()

    all_tickets: List[Dict] = []
    current_iteration = 0

    if resume and state_manager.load() and state_manager.current_state:
        try:
            current_iteration = int(
                state_manager.current_state.meta.get("iteration", 0) or 0
            )
        except Exception:
            current_iteration = 0

    while current_iteration < max_iterations:
        current_iteration += 1
        print_phase(f"Iteration {current_iteration}/{max_iterations}", phase_num=2)

        result_dir = os.path.join(project_root, "result", "science")
        os.makedirs(result_dir, exist_ok=True)

        plan_path = doc_paths.plan_md()
        tasks_path = doc_paths.tasks_md()
        report_path = doc_paths.report_md()
        feedback_path = doc_paths.feedback_md()

        prev_plan_exists = os.path.exists(plan_path) and current_iteration > 1
        prev_tasks_exists = os.path.exists(tasks_path) and current_iteration > 1
        prev_report_exists = os.path.exists(report_path) and current_iteration > 1

        if not (resume and state_manager.current_state and state_manager.current_state.status == TaskStatus.COMPLETED):
            next_step_index = state_manager.get_next_step_index()
            step4 = state_manager.format_step4(next_step_index)
            step_id = f"science_iter_{current_iteration}_step_{step4}"
            state_manager.init_state(
                initial_data={"iteration": current_iteration},
                blueprint_id=step_id,
                task_ids=["ARCHITECT", "WORKER", "INTEGRATOR"],
                step_index=next_step_index,
            )
            state_manager.set_phase(
                GlobalPhase.PLANNING,
                meta={
                    "iteration": current_iteration,
                    "plan_id": step_id,
                    "all_tickets": all_tickets,
                },
            )

        plan_id = (
            state_manager.current_state.blueprint_id
            if state_manager.current_state
            else f"science_iter_{current_iteration}"
        )

        print("\n[1/3] Writing iteration docs (Architect)...")
        try:
            t = state_manager.get_task("ARCHITECT")
            if t and t.status == TaskStatus.COMPLETED:
                print("✓ Architect already completed (resume)")
            else:
                state_manager.update_task(
                    "ARCHITECT",
                    status=TaskStatus.IN_PROGRESS,
                    attempts=0,
                    last_error=None,
                )
                architect_doc_paths = {
                    "constitution_path": os.path.join(cache_root, "constitution.md"),
                    "idea_path": doc_paths.idea_md(),
                    "spec_path": doc_paths.spec_md(),
                    "plan_path": plan_path,
                    "tasks_path": tasks_path,
                    "prev_plan_path": plan_path if prev_plan_exists else "",
                    "prev_tasks_path": tasks_path if prev_tasks_exists else "",
                    "prev_report_path": report_path if prev_report_exists else "",
                    "prev_feedback_path": feedback_path if os.path.exists(feedback_path) else "",
                    "spec_out_path": doc_paths.spec_md(),
                    "plan_out_path": plan_path,
                    "tasks_out_path": tasks_path,
                }
                await architect.write_iteration_docs(
                    manifest=code_manifest,
                    proposal=proposal,
                    code_blueprint=code_blueprint,
                    project_root=project_root,
                    previous_results=None,
                    experiment_id=experiment_id,
                    dataset_dir=dataset_dir,
                    doc_paths=architect_doc_paths,
                    iteration=current_iteration,
                )
                sync_science_docs_to_specs(doc_paths=doc_paths)
                state_manager.update_task(
                    "ARCHITECT",
                    status=TaskStatus.COMPLETED,
                    attempts=1,
                    last_error=None,
                )
        except Exception as e:
            exit_on_rate_limit(e)
            state_manager.update_task(
                "ARCHITECT",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=str(e),
            )
            raise

        print("\n[2/3] Executing iteration (Worker)...")
        if state_manager.current_state:
            state_manager.set_phase(
                GlobalPhase.EXECUTION,
                meta={"iteration": current_iteration, "plan_id": plan_id, "stage": "EXECUTION"},
            )
        try:
            t = state_manager.get_task("WORKER")
            if t and t.status == TaskStatus.COMPLETED:
                print("✓ Worker already completed (resume)")
            else:
                state_manager.update_task(
                    "WORKER",
                    status=TaskStatus.IN_PROGRESS,
                    attempts=0,
                    last_error=None,
                )
                await worker.run_iteration_from_tasks_md(
                    project_root=project_root,
                    tasks_path=tasks_path,
                    iteration_result_dir=result_dir,
                    idea_path=doc_paths.idea_md(),
                    spec_path=doc_paths.spec_md(),
                    plan_path=plan_path,
                )
                state_manager.update_task(
                    "WORKER",
                    status=TaskStatus.COMPLETED,
                    attempts=1,
                    last_error=None,
                )
        except Exception as e:
            exit_on_rate_limit(e)
            state_manager.update_task(
                "WORKER",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=str(e),
            )
            raise

        print("\n[3/3] Analyzing results (Integrator)...")
        try:
            t = state_manager.get_task("INTEGRATOR")
            if t and t.status == TaskStatus.COMPLETED:
                print("✓ Integrator already completed (resume)")
                analysis = ScienceAnalysis(
                    success=False,
                    summary="Resumed: integrator already completed",
                    key_findings=[],
                )
            else:
                state_manager.update_task(
                    "INTEGRATOR",
                    status=TaskStatus.IN_PROGRESS,
                    attempts=0,
                    last_error=None,
                )
                analysis_goal = f"Validate idea: {proposal.idea.title}"
                analysis = await integrator.analyze_iteration_dir(
                    goal=analysis_goal,
                    project_root=project_root,
                    iteration_result_dir=result_dir,
                    proposal=proposal,
                    doc_paths={
                        "idea_path": doc_paths.idea_md(),
                        "spec_path": doc_paths.spec_md(),
                        "plan_path": plan_path,
                        "tasks_path": tasks_path,
                        "prev_report_path": report_path if prev_report_exists else "",
                        "prev_feedback_path": feedback_path if os.path.exists(feedback_path) else "",
                    },
                    iteration=current_iteration,
                )
                state_manager.update_task(
                    "INTEGRATOR",
                    status=TaskStatus.COMPLETED,
                    attempts=1,
                    last_error=None,
                )
        except Exception as e:
            exit_on_rate_limit(e)
            state_manager.update_task(
                "INTEGRATOR",
                status=TaskStatus.FAILED,
                attempts=1,
                last_error=str(e),
            )
            raise

        _persist_analysis(
            doc_paths=doc_paths,
            analysis=analysis,
            iteration=current_iteration,
        )

        print(f"\n{'─'*40}")
        print("Science Analysis Summary:")
        print(f"  Success: {analysis.success}")
        print(f"  Summary: {analysis.summary}")
        print(f"{'─'*40}")

        if analysis.success:
            state_manager.set_status(
                StepStatus.COMPLETED,
                meta={"iteration": current_iteration, "analysis": analysis.model_dump()},
            )
            print("\n✓ Science goal achieved!")
            return []

        state_manager.set_phase(
            GlobalPhase.VERIFICATION,
            meta={
                "iteration": current_iteration,
                "analysis": analysis.model_dump(),
                "all_tickets": all_tickets,
            },
        )

        if current_iteration < max_iterations:
            print(f"\n→ Continuing to iteration {current_iteration + 1}/{max_iterations}...")
        else:
            print(f"\n⚠️  Max iterations limit reached.")

    print_phase("Science Cycle Complete", phase_num=3)
    print(f"\n⚠️  Max iterations ({max_iterations}) reached without achieving success.")
    return []


def _persist_analysis(
    doc_paths: ScienceDocPaths,
    analysis: ScienceAnalysis,
    iteration: int,
) -> None:
    report_path = doc_paths.report_md()
    feedback_path = doc_paths.feedback_md()

    if getattr(analysis, "report_md", ""):
        prev_report = ""
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                prev_report = f.read()
        with open(report_path, "w", encoding="utf-8") as f:
            if prev_report:
                f.write(prev_report + "\n\n---\n\n")
            f.write(f"## Iteration {iteration}\n\n")
            f.write(str(getattr(analysis, "report_md", "") or "").rstrip() + "\n")

    if getattr(analysis, "feedback_md", ""):
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(str(getattr(analysis, "feedback_md", "") or "").rstrip() + "\n")

    sync_science_docs_to_specs(doc_paths=doc_paths)
