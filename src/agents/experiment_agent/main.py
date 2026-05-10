#!/usr/bin/env python3
"""Unified library and CLI entrypoints for experiment-agent."""

import argparse
import asyncio
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.experiment_agent.agents.master import run_master
from src.agents.experiment_agent.agents.prepare import run_prepare
from src.agents.experiment_agent.agents.reporting import run_ablation_report_integrator
from src.agents.experiment_agent.config import print_config
from src.agents.experiment_agent.config import (
    copy_prepared_data_to_workspace,
    ensure_experiment_dirs,
    get_idea_input_path,
    get_science_max_iterations,
    write_workspace_env_file,
)
from src.agents.experiment_agent.runtime.cache import Cache
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    ensure_canonical_workspace_artifacts,
    load_json_file,
)
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


@contextmanager
def _temporary_environ(overrides: Dict[str, Optional[str]]) -> Iterator[None]:
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    result = await run_experiment_once(
        experiment_id=args.experiment,
        workspace_root=os.environ.get("EXPERIMENT_AGENT_WORKSPACE_DIR") or None,
        max_iterations=args.max_iterations,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
        prepare_only=bool(args.prepare_only),
        force_prepare=bool(args.force),
        clone_depth=int(args.clone_depth),
        skip_repos=bool(args.skip_repos),
        skip_datasets=bool(args.skip_datasets),
    )
    return 0 if result.get("ok", True) else 1


async def run_experiment_once(
    *,
    experiment_id: str,
    workspace_root: str | None = None,
    max_iterations: int | None = None,
    resume: bool = False,
    verbose: bool = False,
    prepare_only: bool = False,
    force_prepare: bool = False,
    clone_depth: int = 1,
    skip_repos: bool = False,
    skip_datasets: bool = False,
    config_path: str | None = None,
) -> Dict[str, Any]:
    env_overrides: Dict[str, Optional[str]] = {}
    if workspace_root:
        env_overrides["EXPERIMENT_AGENT_WORKSPACE_DIR"] = workspace_root
    if config_path:
        env_overrides["EXPERIMENT_AGENT_CONFIG_PATH"] = config_path

    with _temporary_environ(env_overrides):
        if config_path:
            from src.config import reload_config

            reload_config(config_path)

        print_config()
        print_phase(
            "EXPERIMENT AGENT",
            "Unified Claude Code orchestration pipeline",
            width=65,
        )

        paths = ensure_experiment_dirs(experiment_id)
        copy_prepared_data_to_workspace(paths["workspace_dir"])
        write_workspace_env_file(experiment_id)
        Cache.initialize(paths["cache_dir"], enabled=True)
        resolved_workspace_root = paths["workspace_dir"]
        project_root = paths["project_dir"]
        ensure_canonical_workspace_artifacts(resolved_workspace_root, project_root)

        print(f"\nExperiment: {experiment_id}")
        print(f"Workspace: {resolved_workspace_root}")
        print(f"Project: {project_root}")
        print(f"Results: {paths['results_dir']}")
        print(f"Model Candidate: {paths['model_dir']}")
        print(f"Agent Reports: {paths['reports_dir']}")

        should_run_prepare = bool(force_prepare) or not _prepare_ready_for_master(
            resolved_workspace_root,
            project_root,
        )

        if should_run_prepare:
            prepare_report = await run_prepare(
                experiment_id=experiment_id,
                force=bool(force_prepare),
                clone_depth=int(clone_depth),
                skip_repos=bool(skip_repos),
                skip_datasets=bool(skip_datasets),
                verbose=bool(verbose),
            )
            print_phase("PREPARE COMPLETE", width=65)
            print(f"  Prepare Idea: {prepare_report.idea_md_path}")
            print(f"  Project dir: {prepare_report.project_dir}")
            print(f"  Repos dir: {prepare_report.repos_dir}")
            print(f"  Dataset dir: {prepare_report.dataset_dir}")
            print(f"  Model dir: {prepare_report.model_dir}")
            print(f"  Results dir: {prepare_report.results_dir}")
            print(f"  Agent reports dir: {prepare_report.reports_dir}")
            ensure_canonical_workspace_artifacts(resolved_workspace_root, project_root)
            if prepare_only:
                return {
                    "ok": True,
                    "iterations": 0,
                    "converged": False,
                    "final_path": "",
                    "ablation_results_path": "",
                    "workspace_root": resolved_workspace_root,
                    "project_root": project_root,
                    "prepare_only": True,
                }
        else:
            print_phase("PREPARE REUSED", width=65)
            print("  Existing validator-backed prepare handoff is ready; skipping prepare rerun.")
            if prepare_only:
                return {
                    "ok": True,
                    "iterations": 0,
                    "converged": False,
                    "final_path": "",
                    "ablation_results_path": "",
                    "workspace_root": resolved_workspace_root,
                    "project_root": project_root,
                    "prepare_only": True,
                }

        idea_path = get_idea_input_path(experiment_id)
        print(f"Idea: {idea_path}")

        resolved_max_iterations = (
            int(max_iterations)
            if max_iterations is not None
            else get_science_max_iterations()
        )
        result = await run_master(
            experiment_id=experiment_id,
            idea_path=idea_path,
            workspace_root=resolved_workspace_root,
            project_root=project_root,
            max_iterations=resolved_max_iterations,
            verbose=bool(verbose),
            resume=bool(resume),
        )

        if result.get("converged"):
            reporting_result = await run_ablation_report_integrator(
                workspace_root=resolved_workspace_root,
                project_root=project_root,
                verbose=bool(verbose),
                resume=bool(resume),
            )
            if not reporting_result.get("valid"):
                raise RuntimeError(
                    "Final ablation report agent did not produce a valid ablation_results.json. "
                    f"See {reporting_result.get('integrator_report_path')} for details."
                )
            result["final_path"] = reporting_result["ablation_results_path"]
            result["ablation_results_path"] = reporting_result["ablation_results_path"]
            result["integrator_report_path"] = reporting_result["integrator_report_path"]

        if result.get("stopped_due_to_iteration_limit"):
            print_phase("ITERATION LIMIT HIT", width=65)
        else:
            print_phase("MISSION COMPLETE", width=65)
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final Report: {result.get('final_path', result.get('ablation_results_path'))}")

        return {
            "ok": True,
            "iterations": int(result.get("iterations") or 0),
            "converged": bool(result.get("converged")),
            "stopped_due_to_iteration_limit": bool(result.get("stopped_due_to_iteration_limit")),
            "final_path": str(result.get("final_path") or ""),
            "ablation_results_path": str(result.get("ablation_results_path") or ""),
            "integrator_report_path": str(result.get("integrator_report_path") or ""),
            "workspace_root": resolved_workspace_root,
            "project_root": project_root,
            "idea_path": idea_path,
        }


def run_experiment_once_sync(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(run_experiment_once(**kwargs))


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
