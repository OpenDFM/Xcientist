#!/usr/bin/env python3
"""
SuperAgent - Dual-Layer AI System for Automated Scientific Discovery

This is the main orchestrator that coordinates the interaction between:
1. Code Layer (Engineering): Generates and maintains the codebase
2. Science Layer (Experimentation): Runs experiments and analyzes results

Communication Protocols:
- CHP (Code Handover Protocol): Code Layer -> Science Layer via CodeManifest
- ORP (Optimization Request Protocol): Science Layer -> Code Layer via OptimizationTicket
"""

import asyncio
import argparse
import sys
import os
import json
import glob

# Add current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.experiment_agent.layers.code.entry import (
    run_code_generation_loop,
    run_optimization,
)
from src.agents.experiment_agent.layers.science.entry import run_science_cycle
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.layers.code.schemas.manifest import CodeManifest
from src.agents.experiment_agent.layers.code.schemas.idea_parser import load_idea_file
from src.agents.experiment_agent.shared.schemas.protocols import (
    OptimizationBatch,
    tickets_from_science_analysis,
)
from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.logger import print_phase
from src.agents.experiment_agent.shared.utils.config import (
    print_config,
    ensure_experiment_dirs,
    get_idea_input_path,
)
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.layers.base.state import (
    StateManager,
    GlobalPhase,
    StepStatus,
    TaskStatus as StateTaskStatus,
)
from src.agents.experiment_agent.shared.schemas.action_plan import action_task_id


def get_args():
    parser = argparse.ArgumentParser(
        description="SuperAgent - Dual-Layer AI for Scientific Discovery"
    )
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment ID (unique identifier)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous checkpoint"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Start a fresh run (clears previous state)"
    )
    parser.add_argument(
        "--stop-after-science-architect",
        action="store_true",
        help="Exit right after Science Architect finishes (for debugging plan extraction)",
    )
    parser.add_argument(
        "--skip-science",
        action="store_true",
        help="Skip science layer (only run code generation)",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=3,
        help="Maximum science-optimization loops (default: 3)",
    )
    parser.add_argument(
        "--quick-analysis",
        action="store_true",
        help="Use quick heuristic analysis instead of LLM",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    return parser.parse_args()


async def run_engineering_layer(
    experiment_id: str,
    resume: bool = False,
    fresh: bool = False,
) -> CodeManifest:
    """
    Run the Engineering Layer (Code Generation).

    This is Phase 1: Build the scientific tool.
    """
    print_phase(
        "ENGINEERING LAYER",
        "Building the scientific tool from research proposal...",
        width=65,
    )

    code_manifest = await run_code_generation_loop(
        experiment_id=experiment_id,
        resume=resume,
        fresh=fresh,
    )

    if not code_manifest:
        raise RuntimeError("Engineering Layer failed to generate code")

    print(f"\n✓ Code Generation Complete")
    print(f"  Project Root: {code_manifest.project_root}")
    print(f"  Entry Point: {code_manifest.entry_point}")
    if code_manifest.entry_points:
        print(f"  Available Scripts:")
        for name, cmd in code_manifest.entry_points.items():
            print(f"    - {name}: {cmd}")

    return code_manifest


async def run_science_layer(
    code_manifest: CodeManifest,
    proposal,
    experiment_id: str,
    max_iterations: int = 3,
    quick_analysis: bool = False,
    resume: bool = False,
    stop_after_science_architect: bool = False,
) -> OptimizationBatch:
    """
    Run the Science Layer (Experimentation).

    This is Phase 2: Use the tool to conduct experiments.
    """
    print_phase(
        "SCIENCE LAYER",
        "Conducting experiments to validate research claims...",
        width=65,
    )

    optimization_tickets = await run_science_cycle(
        code_manifest=code_manifest,
        proposal=proposal,
        experiment_id=experiment_id,
        max_iterations=max_iterations,
        resume=resume,
        quick_analysis=quick_analysis,
        stop_after_science_architect=stop_after_science_architect,
    )

    # None means Science Layer crashed/failed; do NOT treat as goal achieved.
    if optimization_tickets is None:
        raise RuntimeError("Science Layer failed (see logs above for traceback)")

    if optimization_tickets:
        tickets = tickets_from_science_analysis(
            optimization_tickets,
            experiment_id=experiment_id,
        )
        batch = OptimizationBatch(
            tickets=tickets,
            experiment_id=experiment_id,
            summary=f"Science Layer requested {len(tickets)} optimizations",
        )

        print(f"\n⚠ Science Layer Generated Optimization Requests")
        print(f"  Total Tickets: {len(batch.tickets)}")
        print(f"  Critical: {batch.critical_count}")
        print(f"  High Priority: {batch.high_count}")

        by_file = batch.group_by_file()
        if by_file:
            print(f"  Files Affected:")
            for file_path, file_tickets in by_file.items():
                print(f"    - {file_path}: {len(file_tickets)} issues")

        return batch

    print("\n✓ Science Goal Achieved (no optimization needed)")
    return OptimizationBatch(
        tickets=[],
        experiment_id=experiment_id,
        summary="Science goal achieved",
    )


async def run_optimization_loop(
    batch: OptimizationBatch,
    experiment_id: str,
) -> bool:
    """
    Run the Optimization Loop (ORP Interface).

    This sends optimization requests back to the Code Layer.
    """
    if not batch.tickets:
        return True

    print_phase(
        "OPTIMIZATION LOOP",
        "Applying Science Layer feedback to improve code...",
        width=65,
    )

    fix_tickets = batch.to_fix_tickets()
    print(f"  Sending {len(fix_tickets)} tickets to Code Layer...")

    success = await run_optimization(fix_tickets, experiment_id)

    if success:
        print("\n✓ Optimization Applied Successfully")
        # Mark CODE_CHANGE actions in the latest Science step as completed for resume consistency.
        try:
            paths = ensure_experiment_dirs(experiment_id)
            sm_science = StateManager(paths["workspace_dir"], namespace="science")
            if sm_science.load() and sm_science.current_state:
                meta = sm_science.current_state.meta or {}
                ap = meta.get("action_plan") if isinstance(meta, dict) else None
                if isinstance(ap, dict):
                    actions = ap.get("actions") or []
                    for a in actions:
                        if (
                            isinstance(a, dict)
                            and a.get("kind") == "CODE_CHANGE"
                            and a.get("action_id")
                        ):
                            sm_science.update_task(
                                action_task_id(str(a.get("action_id"))),
                                status=StateTaskStatus.COMPLETED,
                                attempts=1,
                                last_error=None,
                            )
        except Exception:
            pass
    else:
        print("\n✗ Optimization Failed")

    return success


async def main():
    """Main entry point for SuperAgent."""
    args = get_args()

    print_config()

    experiment_id = args.experiment

    # Ensure experiment directories exist
    paths = ensure_experiment_dirs(experiment_id)
    # Initialize cache early so --resume can read cached blueprints / main_loop
    Cache.initialize(paths["cache_dir"], enabled=True)
    print(f"\nExperiment: {experiment_id}")
    print(f"Workspace: {paths['workspace_dir']}")

    # Resume policy: do NOT rely on run_state.json.
    # Derive resume intent purely from execution step files (StateManager).
    resume_science = False
    if args.resume and (not args.fresh):
        try:
            sm_science = StateManager(paths["workspace_dir"], namespace="science")
            if sm_science.load() and sm_science.current_state:
                # If science has an incomplete step, we can skip engineering and resume science.
                resume_science = sm_science.current_state.status != StepStatus.COMPLETED
        except Exception:
            resume_science = False

    if args.resume and resume_science and (not args.fresh):
        project_root = paths["project_dir"]
        proposal = load_idea_file(get_idea_input_path(experiment_id))

        blueprint = None
        candidate_ids = []

        # If the last code step was an intermediate fix step (id like "fix_*"), try to recover the original
        # Blueprint ID from the code execution state meta.
        try:
            sm = StateManager(paths["workspace_dir"], namespace="code")
            if sm.load() and sm.current_state:
                original_id = sm.current_state.meta.get("original_blueprint_id")
                if original_id:
                    candidate_ids.append(str(original_id))
                # Also consider the state blueprint_id if it looks like a real Blueprint hash.
                state_bid = sm.current_state.blueprint_id
                if (
                    state_bid
                    and isinstance(state_bid, str)
                    and not state_bid.startswith("fix_")
                ):
                    candidate_ids.append(state_bid)
        except Exception:
            pass

        # Final fallback: recompute proposal hash (the normal Blueprint cache key).
        try:
            candidate_ids.append(Cache.hash_proposal(proposal))
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
            blueprint_path = os.path.join(project_root, "_blueprint.json")
            if os.path.exists(blueprint_path):
                with open(blueprint_path, "r", encoding="utf-8") as f:
                    blueprint_data = json.load(f)
                blueprint = Blueprint(**blueprint_data)

        if blueprint is not None:
            code_manifest = CodeManifest.from_blueprint(
                blueprint=blueprint,
                project_root=project_root,
                description=proposal.idea.description,
            )
            # Ensure at least a runnable entry point is present
            if blueprint.entry_point and "run" not in code_manifest.entry_points:
                code_manifest.entry_points["run"] = f"python {blueprint.entry_point}"

            print(
                "\n[Resume] Detected SCIENCE stage in cache; skipping Engineering/Integration."
            )
        else:
            # Fall back to Engineering if blueprint is missing
            resume_science = False

    if not (args.resume and resume_science and (not args.fresh)):
        # Phase 1: Engineering Layer
        try:
            code_manifest = await run_engineering_layer(
                experiment_id=experiment_id,
                resume=args.resume,
                fresh=args.fresh,
            )
        except RuntimeError as e:
            print(f"\n✗ {e}")
            print("Aborting SuperAgent mission.")
            sys.exit(1)

    # Skip science layer if requested
    if args.skip_science:
        print("\n>>> Skipping Science Layer (--skip-science) <<<")
        print("\n>>> SUPERAGENT MISSION COMPLETE <<<")
        return

    # Load proposal for Science Layer
    proposal = load_idea_file(get_idea_input_path(experiment_id))

    # Determine starting loop number (for resume)
    start_loop = 1
    if args.resume:
        cached_loop = Cache.get_main_loop_num()
        if cached_loop > 1:
            start_loop = cached_loop
            print(f"Resuming from main loop {start_loop}")

    # Phase 2 & 3: Science Layer + Optimization Loop
    loop_num = start_loop
    for loop_num in range(start_loop, args.max_loops + 1):
        Cache.set_main_loop_num(loop_num)

        print(f"\n{'─'*65}")
        print(f"LOOP {loop_num}/{args.max_loops}")
        print("─" * 65)

        # Run Science Layer
        optimization_batch = await run_science_layer(
            code_manifest=code_manifest,
            proposal=proposal,
            experiment_id=experiment_id,
            max_iterations=3,
            quick_analysis=args.quick_analysis,
            resume=args.resume,
            stop_after_science_architect=args.stop_after_science_architect,
        )

        # Check if science goal achieved
        if not optimization_batch.tickets:
            print("\n✓ Science Goal Achieved!")
            break

        # Run Optimization Loop
        success = await run_optimization_loop(
            batch=optimization_batch,
            experiment_id=experiment_id,
        )

        if not success:
            print("\n⚠ Optimization failed. Stopping loop.")
            break

        optimization_batch.iteration = loop_num + 1

    # Final Summary
    print_phase("SUPERAGENT MISSION COMPLETE", width=65)
    print(f"  Experiment: {experiment_id}")
    print(f"  Project: {code_manifest.project_root}")
    print(f"  Loops Completed: {loop_num}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(130)
    except Exception as e:
        exit_on_rate_limit(e)
        print(f"\n✗ Unhandled error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
