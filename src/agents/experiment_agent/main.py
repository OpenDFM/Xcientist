#!/usr/bin/env python3
"""Unified CLI entrypoint for experiment-agent."""

import asyncio
import argparse
import sys
import os
import logging

# Configure logging to suppress TextContent length warnings from OpenHands SDK
logging.getLogger("openhands.message").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.experiment_agent.agents.master import run_master
from src.agents.experiment_agent.agents.prepare import run_prepare
from src.agents.experiment_agent.agents.integration import run_iteration_reporter
from src.agents.experiment_agent.config import print_config
from src.agents.experiment_agent.config import (
    copy_prepared_data_to_workspace,
    ensure_experiment_dirs,
    get_idea_input_path,
    get_science_max_iterations,
    write_workspace_env_file,
)
from src.agents.experiment_agent.runtime.cache import Cache
from src.agents.experiment_agent.runtime.manifests import artifact_paths, load_json_file
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.telemetry import print_phase


def get_args():
    default_max_iterations = get_science_max_iterations()
    parser = argparse.ArgumentParser(
        description="Experiment Agent - unified prepare + orchestration entrypoint"
    )
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment ID (unique identifier)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run prepare phase and exit",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Deprecated: prepare is now a required startup prerequisite and cannot be skipped",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite prepare_idea.md and re-download/clone"
    )
    parser.add_argument(
        "--clone-depth", type=int, default=1, help="git clone depth (default: 1)"
    )
    parser.add_argument("--skip-repos", action="store_true", help="Skip cloning repos")
    parser.add_argument(
        "--skip-datasets", action="store_true", help="Skip downloading datasets"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=default_max_iterations,
        help=f"Maximum iterations (default: {default_max_iterations})",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from prior conversation state"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    return parser.parse_args()


def _prepare_ready_for_master(workspace_root: str, project_root: str) -> bool:
    paths = artifact_paths(workspace_root, project_root)
    payload = load_json_file(paths["prepare_validator"])
    phase_report = normalize_phase_report(payload)
    if phase_report["status"] == "PASS":
        return True
    if phase_report["status"] == "PARTIAL" and phase_report["ready_for_next_phase"]:
        return True
    return False


async def main_async(args) -> int:
    print_config()
    print_phase(
        "EXPERIMENT AGENT",
        "Unified OpenHands orchestration pipeline",
        width=65,
    )

    experiment_id = args.experiment
    paths = ensure_experiment_dirs(experiment_id)
    copy_prepared_data_to_workspace(paths["workspace_dir"])
    write_workspace_env_file(experiment_id)
    Cache.initialize(paths["cache_dir"], enabled=True)
    workspace_root = paths["workspace_dir"]
    project_root = paths["project_dir"]

    print(f"\nExperiment: {experiment_id}")
    print(f"Workspace: {workspace_root}")
    print(f"Project: {project_root}")
    print(f"Results: {paths['results_dir']}")
    print(f"Model Candidate: {paths['model_dir']}")
    print(f"Agent Reports: {paths['reports_dir']}")

    if args.skip_prepare:
        raise ValueError("--skip-prepare is no longer supported; prepare is a required startup prerequisite")

    should_run_prepare = bool(args.force) or not _prepare_ready_for_master(workspace_root, project_root)

    if should_run_prepare:
        prepare_report = await run_prepare(
            experiment_id=experiment_id,
            force=bool(args.force),
            clone_depth=int(args.clone_depth),
            skip_repos=bool(args.skip_repos),
            skip_datasets=bool(args.skip_datasets),
            verbose=bool(args.verbose),
        )
        print_phase("PREPARE COMPLETE", width=65)
        print(f"  Prepare Idea: {prepare_report.idea_md_path}")
        print(f"  Project dir: {prepare_report.project_dir}")
        print(f"  Repos dir: {prepare_report.repos_dir}")
        print(f"  Dataset dir: {prepare_report.dataset_dir}")
        print(f"  Model dir: {prepare_report.model_dir}")
        print(f"  Results dir: {prepare_report.results_dir}")
        print(f"  Agent reports dir: {prepare_report.reports_dir}")
        if args.prepare_only:
            return 0
    else:
        print_phase("PREPARE REUSED", width=65)
        print("  Existing validator-backed prepare handoff is ready; skipping prepare rerun.")
        if args.prepare_only:
            return 0

    idea_path = get_idea_input_path(experiment_id)
    print(f"Idea: {idea_path}")

    result = await run_master(
        experiment_id=experiment_id,
        idea_path=idea_path,
        workspace_root=workspace_root,
        project_root=project_root,
        max_iterations=args.max_iterations,
        verbose=bool(args.verbose),
        resume=bool(args.resume),
    )

    # After master loop, run iteration integration to summarize status
    iteration_result = await run_iteration_reporter(
        workspace_root=workspace_root,
        project_root=project_root,
        verbose=bool(args.verbose),
        resume=bool(args.resume),
    )
    if iteration_result.get("valid"):
        result["iteration_summary_path"] = iteration_result["iteration_summary_path"]

    if result.get("stopped_due_to_iteration_limit"):
        print_phase("ITERATION LIMIT HIT", width=65)
    else:
        print_phase("MISSION COMPLETE", width=65)
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final Report: {result.get('final_path', result.get('ablation_results_path'))}")
    return 0


def main() -> int:
    return int(asyncio.run(main_async(get_args())))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unhandled error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
