"""
Science Layer Entry Point (MD-driven, single worker, resumable).

This layer is intentionally NOT JSON/DAG-driven.
The iteration loop is:
1) Architect writes Markdown docs (spec/plan/tasks) under cached/science/ (versioned)
2) Worker executes the full iteration from tasks.md (Markdown, no parser)
3) Integrator reads iteration results and writes report/feedback (Markdown)

All source-of-truth docs are stored under:
- <workspace_root>/cached/science/
and mirrored to:
- <workspace_root>/specs/

All iteration results are stored under:
- <project_root>/result/science/iter_v###/
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
    print_phase("Science Experimentation (MD-driven)", phase_num=1)

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

    # Snapshot idea into cache/science/idea.md for robust resume + prompts
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

    # Optional: load code blueprint from cache (for architect's context only)
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
    resume_incomplete_step = False
    resume_iteration = 0

    if resume and state_manager.load() and state_manager.current_state:
        try:
            current_iteration = int(
                state_manager.current_state.meta.get("iteration", 0) or 0
            )
        except Exception:
            current_iteration = 0
        if (
            state_manager.current_state.status != StepStatus.COMPLETED
            and current_iteration > 0
        ):
            resume_incomplete_step = True
            resume_iteration = current_iteration
            current_iteration -= 1

    while current_iteration < max_iterations:
        current_iteration += 1
        version = int(current_iteration)
        print_phase(f"Iteration {current_iteration}/{max_iterations}", phase_num=2)

        iteration_result_dir = os.path.join(
            project_root, "result", "science", f"iter_v{version:03d}"
        )
        os.makedirs(iteration_result_dir, exist_ok=True)

        spec_out_path = doc_paths.spec_md()
        plan_out_path = doc_paths.plan_md(version)
        tasks_out_path = doc_paths.tasks_md(version)
        prev_plan_path = doc_paths.plan_md(version - 1) if version > 1 else ""
        prev_tasks_path = doc_paths.tasks_md(version - 1) if version > 1 else ""
        prev_report_path = doc_paths.report_md(version - 1) if version > 1 else ""
        prev_feedback_path = doc_paths.feedback_md(version - 1) if version > 1 else ""

        # Create a new state step per iteration (unless resuming the same incomplete step)
        if not (resume_incomplete_step and current_iteration == resume_iteration):
            next_step_index = state_manager.get_next_step_index()
            step4 = state_manager.format_step4(next_step_index)
            step_id = f"science_iter_v{version:03d}_step_{step4}"
            state_manager.init_state(
                initial_data={"iteration": version},
                blueprint_id=step_id,
                task_ids=["ARCHITECT", "WORKER", "INTEGRATOR"],
                step_index=next_step_index,
            )
            state_manager.set_phase(
                GlobalPhase.PLANNING,
                meta={
                    "iteration": version,
                    "plan_id": step_id,
                    "all_tickets": all_tickets,
                },
            )

        plan_id = (
            state_manager.current_state.blueprint_id
            if state_manager.current_state
            else f"science_iter_v{version:03d}"
        )

        # [1/3] Architect
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
                    "spec_path": spec_out_path,
                    "plan_path": plan_out_path,
                    "tasks_path": tasks_out_path,
                    "prev_plan_path": (
                        prev_plan_path if os.path.exists(prev_plan_path) else ""
                    ),
                    "prev_tasks_path": (
                        prev_tasks_path if os.path.exists(prev_tasks_path) else ""
                    ),
                    "prev_report_path": (
                        prev_report_path if os.path.exists(prev_report_path) else ""
                    ),
                    "prev_feedback_path": (
                        prev_feedback_path if os.path.exists(prev_feedback_path) else ""
                    ),
                    "spec_out_path": spec_out_path,
                    "plan_out_path": plan_out_path,
                    "tasks_out_path": tasks_out_path,
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
                )
                sync_science_docs_to_specs(doc_paths=doc_paths, version=version)
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

        # [2/3] Worker
        print("\n[2/3] Executing iteration (Worker; MD-driven)...")
        if state_manager.current_state:
            state_manager.set_phase(
                GlobalPhase.EXECUTION,
                meta={"iteration": version, "plan_id": plan_id, "stage": "EXECUTION"},
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
                    tasks_path=tasks_out_path,
                    iteration_result_dir=iteration_result_dir,
                    idea_path=doc_paths.idea_md(),
                    spec_path=doc_paths.spec_md(),
                    plan_path=plan_out_path,
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

        # [3/3] Integrator
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
                    iteration_result_dir=iteration_result_dir,
                    proposal=proposal,
                    doc_paths={
                        "idea_path": doc_paths.idea_md(),
                        "spec_path": doc_paths.spec_md(),
                        "plan_path": plan_out_path,
                        "tasks_path": tasks_out_path,
                        "prev_report_path": (
                            prev_report_path if os.path.exists(prev_report_path) else ""
                        ),
                        "prev_feedback_path": (
                            prev_feedback_path
                            if os.path.exists(prev_feedback_path)
                            else ""
                        ),
                    },
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

        _persist_iteration_docs(
            doc_paths=doc_paths,
            version=version,
            analysis=analysis,
            iteration_result_dir=iteration_result_dir,
        )

        print(f"\n{'─'*40}")
        print("Science Analysis Summary:")
        print(f"  Success: {analysis.success}")
        print(f"  Summary: {analysis.summary}")
        print(f"{'─'*40}")

        if analysis.success:
            state_manager.set_status(
                StepStatus.COMPLETED,
                meta={"iteration": version, "analysis": analysis.model_dump()},
            )
            print("\n✓ Science goal achieved!")
            return []

        if resume_incomplete_step and current_iteration == resume_iteration:
            resume_incomplete_step = False

        state_manager.set_phase(
            GlobalPhase.VERIFICATION,
            meta={
                "iteration": version,
                "analysis": analysis.model_dump(),
                "all_tickets": all_tickets,
            },
        )

        # Check if more iterations are available
        if current_iteration < max_iterations:
            print(f"\n→ Integrator requested next iteration via feedback.md")
            print(
                f"   Continuing to iteration {current_iteration + 1}/{max_iterations}..."
            )
        else:
            print(f"\n⚠️  Goal not achieved but max iterations limit reached.")
            print(f"   Will not proceed to iteration {current_iteration + 1}.")

    print_phase("Science Cycle Complete", phase_num=3)
    print(f"\n⚠️  Max iterations ({max_iterations}) reached without achieving success.")
    print(
        "    Consider increasing --max-iterations or reviewing the experimental design."
    )
    # Normal completion without emitting optimization tickets.
    return []


def _persist_iteration_docs(
    doc_paths: ScienceDocPaths,
    version: int,
    analysis: ScienceAnalysis,
    iteration_result_dir: str,
) -> None:
    """
    Persist report/feedback into cache/science (versioned), mirror into specs/, and copy into result/.
    """
    report_path = doc_paths.report_md(int(version))
    feedback_path = doc_paths.feedback_md(int(version))

    if getattr(analysis, "report_md", ""):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(str(getattr(analysis, "report_md", "") or "").rstrip() + "\n")
    if getattr(analysis, "feedback_md", ""):
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(str(getattr(analysis, "feedback_md", "") or "").rstrip() + "\n")

    sync_science_docs_to_specs(doc_paths=doc_paths, version=int(version))

    os.makedirs(iteration_result_dir, exist_ok=True)
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as fsrc:
            with open(
                os.path.join(iteration_result_dir, "report.md"), "w", encoding="utf-8"
            ) as fdst:
                fdst.write(fsrc.read())
    if os.path.exists(feedback_path):
        with open(feedback_path, "r", encoding="utf-8") as fsrc:
            with open(
                os.path.join(iteration_result_dir, "feedback.md"), "w", encoding="utf-8"
            ) as fdst:
                fdst.write(fsrc.read())
