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
import shutil
from typing import List, Optional, Dict

from src.agents.experiment_agent.layers.code.architect import CodeArchitectAgent
from src.agents.experiment_agent.layers.code.manager import CodeManagerAgent
from src.agents.experiment_agent.layers.code.integrator import CodeIntegratorAgent
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.layers.code.schemas.idea_parser import load_idea_file
from src.agents.experiment_agent.layers.code.schemas.manifest import CodeManifest
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.shared.tools.core import SecurityContext
from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.logger import print_phase
from src.agents.experiment_agent.shared.utils.config import (
    ProjectContext,
    setup_openai_api,
    ensure_experiment_dirs,
    get_reference_repos,
    MAX_FIX_ITERATIONS,
)
from src.agents.experiment_agent.layers.code.docs import (
    build_code_doc_paths,
    snapshot_idea_to_cache,
    sync_code_docs_to_specs,
)
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.layers.base.state import (
    StateManager,
    GlobalPhase,
    StepStatus,
)

DEFAULT_CONSTITUTION_MD = """# Constitution (per-experiment) - v1

This file is the source of truth for this experiment.

## Non-negotiables

1. No secrets in code. Use environment variables for API keys/tokens.
2. Spec-driven workflow: generate and follow `specs/spec.md` and `specs/plan.md` before generating tasks (Blueprint).
3. Task-driven implementation: workers implement strictly according to Blueprint tasks derived from plan.md.
4. Worker single-file discipline: modify only the assigned file.
5. Tool-based reading (spec-kit intent): before writing code, read:
   - `cached/constitution.md`
   - `specs/plan.md`
6. Templates/prompts are read-only unless explicitly requested:
   - Do not modify prompt/template files under `src/agents/experiment_agent/layers/**/prompts/` during normal runs.
5. UTF-8 everywhere: all file I/O uses encoding=\"utf-8\".
"""


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _sync_templates_into_workspace(
    workspace_root: str,
    templates_dir: str,
    fresh: bool = False,
) -> None:
    """
    Copy spec-kit templates into the experiment workspace templates/ directory.

    - Source: local spec-kit checkout under agent_workspace/spec-kit/templates
    - Destination: workspaces/<experiment_id>/templates
    - If fresh=True: overwrite destination by deleting and re-copying.
    - If fresh=False: copy missing files only, do not overwrite user-modified templates.
    """
    if not workspace_root or not templates_dir:
        return

    # Source templates directory (preferred): vendored copy under code layer.
    # This is created by copying spec-kit/templates into:
    #   src/agents/experiment_agent/layers/code/templates/
    # and then synchronized into each experiment workspace.
    vendored_templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    # Optional override: user can point to another templates directory.
    spec_kit_templates_dir = os.environ.get(
        "SPEC_KIT_TEMPLATES_DIR", vendored_templates_dir
    )

    if not os.path.isdir(spec_kit_templates_dir):
        return

    if fresh and os.path.isdir(templates_dir):
        shutil.rmtree(templates_dir)
        os.makedirs(templates_dir, exist_ok=True)

    for root, dirs, files in os.walk(spec_kit_templates_dir):
        rel = os.path.relpath(root, spec_kit_templates_dir)
        dest_root = templates_dir if rel == "." else os.path.join(templates_dir, rel)
        os.makedirs(dest_root, exist_ok=True)

        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(dest_root, name)
            if (not fresh) and os.path.exists(dst_path):
                continue
            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                continue


async def run_code_generation_loop(
    experiment_id: str,
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

        proposal_path = paths["idea_input"]
        output_dir = paths["project_dir"]

        workspace_root = paths["workspace_dir"]
        cache_dir = paths["cache_dir"]
        dataset_dir = paths.get("dataset_dir")
        doc_paths = build_code_doc_paths(
            workspace_root=workspace_root, cache_root=cache_dir, ensure=True
        )
        templates_dir = doc_paths.templates_dir()

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
        doc_paths = build_code_doc_paths(
            workspace_root=workspace_root, cache_root=cache_dir, ensure=True
        )
        templates_dir = doc_paths.templates_dir()

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

    # Spec/Plan (spec-kit-aligned source-of-truth)
    constitution_path = doc_paths.constitution_path() if experiment_id else ""
    if constitution_path and (not os.path.exists(constitution_path)):
        _write_text(constitution_path, DEFAULT_CONSTITUTION_MD)

    # Sync spec-kit templates into this workspace and pass absolute templates dir to prompts.
    _sync_templates_into_workspace(
        workspace_root=workspace_root,
        templates_dir=templates_dir,
        fresh=bool(fresh),
    )

    # Code layer spec-coding source-of-truth lives in cache/code/
    spec_path = doc_paths.spec_md()
    plan_path = doc_paths.plan_md()

    constitution_md = _read_text(constitution_path) if constitution_path else ""

    architect = CodeArchitectAgent()
    # Snapshot idea.md into cache/code/idea.md for resumability and consistent prompts.
    try:
        proposal_text = f"# Idea\n\n## Title\n{proposal.idea.title}\n\n## Description\n{proposal.idea.description}\n"
    except Exception:
        proposal_text = ""
    snapshot_idea_to_cache(
        idea_input_path=proposal_path or "",
        proposal_text=proposal_text,
        doc_paths=doc_paths,
    )

    # One-shot only (single-run guarantee):
    # - If cached blueprint is valid AND spec/plan already exist in cache, reuse cache and DO NOT call the architect.
    # - Otherwise, run one-shot exactly once to (re)write spec/plan and output blueprint JSON.
    spec_md = _read_text(spec_path)
    plan_md = _read_text(plan_path)
    proposal_hash = Cache.hash_proposal(proposal)
    plan_hash = Cache.hash_text(plan_md)

    blueprint = None
    cached_blueprint = Cache.get_blueprint(plan_hash)
    if cached_blueprint and "blueprint" in cached_blueprint:
        try:
            blueprint = Blueprint(**cached_blueprint["blueprint"])
            blueprint.validate_dag()
            print("✓ Using cached blueprint")
        except Exception as e:
            print(f"  ⚠ Cached blueprint invalid; will run one-shot. Reason: {e}")
            blueprint = None

    need_oneshot = False
    if fresh:
        need_oneshot = True
    if (not spec_md.strip()) or (not plan_md.strip()):
        need_oneshot = True
    if blueprint is None:
        need_oneshot = True

    if need_oneshot:
        print_phase("Spec + Plan + Blueprint (one-shot)", phase_num=2)
        try:
            blueprint = await architect.create_spec_plan_and_blueprint(
                proposal_path=proposal_path or "",
                constitution_md=constitution_md,
                templates_dir=os.path.abspath(templates_dir),
                spec_output_path=os.path.abspath(spec_path),
                plan_output_path=os.path.abspath(plan_path),
                experiment_id=experiment_id or "",
                dataset_dir=dataset_dir or "",
                reference_repos=reference_repos,
            )
            blueprint.validate_dag()
        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ One-shot spec/plan/blueprint failed: {e}")
            return None

        # Refresh texts + hashes after one-shot (plan/spec may have changed)
        spec_md = _read_text(spec_path)
        plan_md = _read_text(plan_path)
        plan_hash = Cache.hash_text(plan_md)
        sync_code_docs_to_specs(doc_paths)

    # Ensure blueprint is cached under plan_hash and proposal_hash (resume-friendly)
    try:
        cached = Cache.get_blueprint(plan_hash)
        if (not cached) or ("blueprint" not in cached):
            Cache.set_blueprint(plan_hash, blueprint.model_dump())
        Cache.set_blueprint(proposal_hash, blueprint.model_dump())
    except Exception:
        pass

    # Ensure a state step exists (even when using cached/one-shot blueprint) to keep resume robust.
    try:
        has_state = state_manager.load() and state_manager.current_state is not None
    except Exception:
        has_state = False
    if not has_state:
        try:
            task_ids = [f.file_path for f in blueprint.files]
            state_manager.init_state(
                blueprint.model_dump(),
                blueprint_id=plan_hash,
                task_ids=task_ids,
            )
            if state_manager.current_state:
                state_manager.set_phase(
                    state_manager.current_state.phase,
                    meta={
                        "proposal_hash": proposal_hash,
                        "plan_hash": plan_hash,
                    },
                )
        except Exception:
            pass

    # Check if we should skip implementation phase (already in REFINEMENT)
    skip_implementation = False
    start_fix_iteration = 0

    if resume and state_manager.load() and state_manager.current_state:
        current_phase = state_manager.current_state.phase
        # Only treat REFINEMENT as "resume fix" when the step is not completed yet.
        if (
            current_phase == GlobalPhase.REFINEMENT
            and state_manager.current_state.status != StepStatus.COMPLETED
        ):
            # We're in fix mode, skip implementation
            skip_implementation = True
            fix_iter = state_manager.current_state.meta.get("fix_iteration", 1)
            start_fix_iteration = (
                fix_iter - 1
            )  # -1 because loop will start from this value
            print(f"\n[Resume] Detected REFINEMENT phase (fix iteration {fix_iter})")
            print(
                "[Resume] Skipping implementation phase, continuing integrator fix loop..."
            )

    # Implementation Phase
    manager = CodeManagerAgent(
        project_root=output_dir,
        idea_md_path=doc_paths.idea_md(),
        reference_repos=reference_repos,
    )

    if not skip_implementation:
        print_phase("Code Implementation", phase_num=5)

        try:
            await manager.execute_blueprint(
                blueprint, blueprint_id=plan_hash, resume=resume
            )
        except Exception as e:
            exit_on_rate_limit(e)
            print(f"✗ Manager failed: {e}")
            return None
    else:
        print_phase("Code Implementation (Skipped - Resuming Fix)", phase_num=3)

    # Integration Phase
    success = True
    print_phase("Integration & Verification", phase_num=6)

    integrator = CodeIntegratorAgent(project_root=output_dir)
    try:
        try:
            if state_manager.load() and state_manager.current_state:
                meta = dict(state_manager.current_state.meta or {})
                meta["stage"] = "INTEGRATION_FIX"
                state_manager.set_phase(GlobalPhase.REFINEMENT, meta=meta)
                state_manager.set_status(StepStatus.RUNNING, meta=meta)
        except Exception:
            pass

        # Fix is handled by the Integrator itself (single-agent loop inside the LLM agent).
        success = await integrator.fix_until_tests_pass(
            entry_point=blueprint.entry_point
        )
    except Exception as e:
        exit_on_rate_limit(e)
        print(f"✗ Integration failed: {e}")
        success = False

    if success:
        try:
            state_manager.load()
        except Exception:
            pass
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

    try:
        integrator = CodeIntegratorAgent(project_root=output_dir, verbose=True)
        return await integrator.fix_until_tests_pass(
            entry_point=blueprint.entry_point, tickets=tickets
        )
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
    # IMPORTANT: Use the same import path as CodeManifest to avoid duplicate class instances
    # (pydantic will reject objects from a different module path even if they have the same name).
    from src.agents.experiment_agent.layers.code.schemas.manifest import (
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
