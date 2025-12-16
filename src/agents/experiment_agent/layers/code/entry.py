"""
Code Layer Entry Point

This module provides the main entry point for the Code Agent cycle.
It orchestrates:
1. Proposal loading
2. Blueprint generation (Architect)
3. Code implementation (Manager + Workers)
4. Integration verification (Integrator)
5. Fix loops if needed
"""

import os
import json
from typing import List, Optional, Dict

from src.agents.experiment_agent.layers.code.architect import CodeArchitectAgent
from src.agents.experiment_agent.layers.code.manager import CodeManagerAgent
from src.agents.experiment_agent.layers.code.integrator import CodeIntegratorAgent
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.layers.code.schemas.idea_parser import load_idea_file
from src.agents.experiment_agent.layers.code.schemas.manifest import CodeManifest
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.layers.code.schemas.fix_blueprint import FixBlueprint

from src.agents.experiment_agent.shared.tools.core import SecurityContext
from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.logger import print_phase
from src.agents.experiment_agent.shared.utils.config import (
    ProjectContext,
    setup_openai_api,
    ensure_experiment_dirs,
    get_reference_repos,
)
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.layers.base.state import StateManager, GlobalPhase, StepStatus


async def run_code_generation_loop(
    experiment_id: str,
    proposal_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    extra_reference_repos: Optional[List[str]] = None,
    skip_integration: bool = False,
    resume: bool = False,
    fresh: bool = False,
) -> Optional[CodeManifest]:
    """
    Run the Code Agent engineering cycle.

    Returns CodeManifest if successful, None otherwise.
    """
    reference_repos = []

    if experiment_id:
        print(f"\n[CodeAgent: {experiment_id}]")
        paths = ensure_experiment_dirs(experiment_id)

        if proposal_path is None:
            proposal_path = paths["idea_input"]
        if output_dir is None:
            output_dir = paths["project_dir"]

        workspace_root = paths["workspace_dir"]
        cache_dir = paths["cache_dir"]
        dataset_dir = paths.get("dataset_dir")

        reference_repos = get_reference_repos(experiment_id)
    else:
        if not output_dir or not proposal_path:
            raise ValueError(
                "output_dir and proposal_path required if no experiment_id"
            )

        output_dir = os.path.abspath(output_dir)
        workspace_root = os.path.dirname(output_dir)
        cache_dir = os.path.join(output_dir, ".cache")
        dataset_dir = None
        os.makedirs(output_dir, exist_ok=True)

    # Initialize cache
    Cache.initialize(cache_dir, enabled=True)

    # Handle resume/fresh logic
    if fresh:
        print("\n[Fresh start requested]")
        Cache.clear()
        Cache.initialize(cache_dir, enabled=True)
        if os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

    # Add extra repos
    if extra_reference_repos:
        for repo in extra_reference_repos:
            if repo not in reference_repos:
                reference_repos.append(repo)

    # Initialize context
    ProjectContext.initialize(
        project_root=output_dir,
        workspace_root=workspace_root,
        project_id=experiment_id,
        reference_repos=reference_repos,
    )
    SecurityContext.set_roots(project_root=output_dir, workspace_root=workspace_root)

    state_manager = StateManager(workspace_root, namespace="code")

    # Setup API
    if not setup_openai_api(verbose=True):
        return None

    # Load Proposal
    print_phase("Loading Proposal", phase_num=1)
    try:
        proposal = load_idea_file(proposal_path)
    except Exception as e:
        exit_on_rate_limit(e)
        print(f"✗ Failed to load proposal: {e}")
        return None

    # Architect Phase
    print_phase("System Architecture", phase_num=2)

    proposal_hash = Cache.hash_proposal(proposal)
    cached_blueprint = Cache.get_blueprint(proposal_hash)

    blueprint = None
    if cached_blueprint:
        print("✓ Using cached blueprint")
        if dataset_dir and os.path.isdir(dataset_dir):
            try:
                has_files = any(
                    os.path.isfile(os.path.join(dataset_dir, name))
                    for name in os.listdir(dataset_dir)
                )
            except Exception:
                has_files = False
            if has_files:
                print(
                    f"  ⚠ Dataset directory appears non-empty ({dataset_dir}). "
                    "If the dataset has changed and you want the Architect to re-design the blueprint based on it, rerun with --fresh."
                )
        try:
            blueprint = Blueprint(**cached_blueprint["blueprint"])
            # Validate cached blueprint; if invalid, drop cache and regenerate via Architect.
            blueprint.validate_dag()
        except Exception as e:
            print(f"  ⚠ Cached blueprint invalid; will regenerate. Reason: {e}")
            cached_blueprint = None
            blueprint = None

    if not blueprint:
        architect = CodeArchitectAgent()
        try:
            blueprint = await architect.create_blueprint(
                proposal,
                reference_repos=reference_repos,
                experiment_id=experiment_id,
                dataset_dir=dataset_dir,
            )
            blueprint.validate_dag()

            Cache.set_blueprint(proposal_hash, blueprint.model_dump())

            # Initial state snapshot with blueprint
            task_ids = [f.file_path for f in blueprint.files]
            state_manager.init_state(
                blueprint.model_dump(),
                blueprint_id=proposal_hash,
                task_ids=task_ids,
                )

        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ Architect failed: {e}")
            return None

    # Check if we should skip implementation phase (already in REFINEMENT)
    skip_implementation = False
    start_fix_iteration = 0
    resume_fix_blueprint = None  # FixBlueprint to resume if in REFINEMENT

    if resume and state_manager.load() and state_manager.current_state:
        current_phase = state_manager.current_state.phase
        if current_phase == GlobalPhase.REFINEMENT:
            # We're in fix mode, skip implementation
            skip_implementation = True
            fix_iter = state_manager.current_state.meta.get("fix_iteration", 1)
            start_fix_iteration = (
                fix_iter - 1
            )  # -1 because loop will start from this value
            print(f"\n[Resume] Detected REFINEMENT phase (fix iteration {fix_iter})")
            print("[Resume] Skipping implementation phase, continuing fix loop...")

            # Try to load the fix_blueprint from cache or file
            fix_blueprint_id = state_manager.current_state.blueprint_id
            if fix_blueprint_id:
                cached_fix = Cache.get_blueprint(fix_blueprint_id)
                if cached_fix and "blueprint" in cached_fix:
                    try:
                        resume_fix_blueprint = FixBlueprint(**cached_fix["blueprint"])
                        print(
                            f"[Resume] Loaded fix_blueprint from cache: {fix_blueprint_id}"
                        )
                    except Exception as e:
                        print(f"[Resume] Failed to parse fix_blueprint: {e}")

    # Implementation Phase
    manager = CodeManagerAgent(
        project_root=output_dir,
        idea_md_path=proposal_path or "",
        reference_repos=reference_repos,
    )

    if not skip_implementation:
        print_phase("Code Implementation", phase_num=3)

        try:
            await manager.execute_blueprint(
                blueprint, blueprint_id=proposal_hash, resume=resume
            )
        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ Manager failed: {e}")
            return None
    else:
        print_phase("Code Implementation (Skipped - Resuming Fix)", phase_num=3)

    # Integration Phase
    success = True
    if not skip_integration:
        print_phase("Integration & Verification", phase_num=4)

        MAX_FIX_ITERATIONS = 5

        integrator = CodeIntegratorAgent(project_root=output_dir)
        try:
            # If resuming with a fix_blueprint, continue from there
            if resume_fix_blueprint:
                print(
                    f"\n[Resume Fix Loop {start_fix_iteration + 1}/{MAX_FIX_ITERATIONS}]"
                )
                print(
                    f"  Resuming with {len(resume_fix_blueprint.tasks)} pending fix tasks..."
                )
                await manager.fix_blueprint(resume_fix_blueprint, blueprint)
                # After resume, increment to next iteration
                start_fix_iteration += 1

            success = await integrator.verify_project(blueprint.entry_point)

            if not success:
                for i in range(start_fix_iteration, MAX_FIX_ITERATIONS):
                    fix_blueprint = await integrator.generate_fix_blueprint(
                        blueprint=blueprint,
                        entry_point=blueprint.entry_point,
                    )
                    if not fix_blueprint.tasks:
                        break

                    print(f"\n[Fix Loop {i+1}/{MAX_FIX_ITERATIONS}]")

                    # Generate fix blueprint ID
                    fix_blueprint_id = f"fix_{i+1}_{proposal_hash[:8]}"

                    # Also save to cached/blueprints/ for consistency with original blueprint
                    Cache.set_blueprint(fix_blueprint_id, fix_blueprint.model_dump())

                    # Create a new step for this fix iteration
                    fix_task_ids = [t.task_id for t in fix_blueprint.tasks]
                    state_manager.init_state(
                        initial_data=fix_blueprint.model_dump(),
                        blueprint_id=fix_blueprint_id,
                        task_ids=fix_task_ids,
                        )

                    # Update phase with meta info (set_phase saves to disk)
                    state_manager.set_phase(
                        GlobalPhase.REFINEMENT,
                        meta={
                            "fix_iteration": i + 1,
                            "trigger": "integration_verify",
                            "original_blueprint_id": proposal_hash,
                        },
                    )

                    await manager.fix_blueprint(fix_blueprint, blueprint)

                    success = await integrator.verify_project(blueprint.entry_point)
                    if success:
                        break
        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ Integration failed: {e}")
            success = False

    if success:
        state_manager.set_status(StepStatus.COMPLETED)
        print("\n✓ Code Generation Complete")

        manifest = _create_code_manifest(
            output_dir=output_dir,
            blueprint=blueprint,
            proposal=proposal,
        )
        return manifest
    else:
        print("\n⚠ Code Generation Completed with Issues")
        return None


async def run_optimization(
    tickets: List[dict],
    experiment_id: str,
) -> bool:
    """Run an optimization loop based on feedback tickets."""
    print_phase("Code Optimization (Feedback Loop)", phase_num=5)

    paths = ensure_experiment_dirs(experiment_id)
    output_dir = paths["project_dir"]
    Cache.initialize(paths["cache_dir"], enabled=True)

    # Load blueprint
    blueprint = None
    # Do NOT rely on run_state.json. Recover Blueprint ID from execution step files.
    candidate_ids = []
    try:
        sm = StateManager(paths["workspace_dir"], namespace="code")
        if sm.load() and sm.current_state:
            # If we are in a fix step, try to recover the original blueprint_id first.
            original_id = sm.current_state.meta.get("original_blueprint_id")
            if original_id:
                candidate_ids.append(str(original_id))
            state_bid = sm.current_state.blueprint_id
            if state_bid:
                candidate_ids.append(str(state_bid))
    except Exception:
        pass

    # De-duplicate while preserving order
    seen = set()
    candidate_ids = [
        x for x in candidate_ids if x and (x not in seen and not seen.add(x))
    ]

    for cid in candidate_ids:
        cached = Cache.get_blueprint(cid)
        if cached and "blueprint" in cached:
            try:
                blueprint = Blueprint(**cached["blueprint"])
                break
            except Exception:
                blueprint = None

    # Backward-compatible fallback (older runs wrote _blueprint.json into project_dir).
    if blueprint is None:
        blueprint_path = os.path.join(output_dir, "_blueprint.json")
        if os.path.exists(blueprint_path):
            with open(blueprint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            blueprint = Blueprint(**data)

    if blueprint is None:
        print("✗ Blueprint not found for optimization (cache/state)")
        return False

    manager = CodeManagerAgent(project_root=output_dir)

    try:
        await manager.fix_files(tickets, blueprint)
        return True
    except Exception as e:
        exit_on_rate_limit(e)
        print(f"✗ Optimization failed: {e}")
        return False


def _create_code_manifest(
    output_dir: str,
    blueprint: Blueprint,
    proposal: Proposal,
) -> CodeManifest:
    """Create a comprehensive CodeManifest for the Science Layer."""
    from layers.code.schemas.manifest import (
        ConfigurationSpec,
        MetricsSpec,
    )

    def _py_cmd(rel_path: str) -> str:
        return f"python {rel_path}"

    # Entry points: do not guess; rely on blueprint.handover when available.
    entry_point = blueprint.entry_point
    entry_points: Dict[str, str] = {}

    if blueprint.handover and blueprint.handover.entry_points:
        entry_points.update(dict(blueprint.handover.entry_points))
    if "run" not in entry_points and entry_point:
        entry_points["run"] = _py_cmd(entry_point)

    configuration = None
    config_file = None
    if blueprint.handover and blueprint.handover.config_file:
        config_file = blueprint.handover.config_file
        config_format = blueprint.handover.config_format or ""
        configuration = ConfigurationSpec(
            config_file=config_file,
            config_format=config_format if config_format else "yaml",
            hyperparameters={},
        )

    # Metrics: do not guess; rely on blueprint.handover when available.
    metrics = None
    if blueprint.handover and blueprint.handover.metrics_log_file:
        metrics_file = blueprint.handover.metrics_log_file
        metrics_format = blueprint.handover.metrics_log_format or "json"
        metrics = MetricsSpec(
            log_file=metrics_file,
            log_format=metrics_format,
            keys=list(blueprint.handover.metrics_keys or []),
            primary_metric=blueprint.handover.primary_metric,
            higher_is_better=(
                bool(blueprint.handover.higher_is_better)
                if blueprint.handover.higher_is_better is not None
                else True
            ),
        )

    return CodeManifest(
        project_root=output_dir,
        entry_point=entry_point,
        description=proposal.idea.description,
        entry_points=entry_points,
        configuration=configuration,
        metrics=metrics,
        source_files=blueprint.file_tree,
        config_file=config_file,
    )
