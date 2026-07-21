#!/usr/bin/env python3
"""Unified library and CLI entrypoints for experiment-agent."""

import argparse
import asyncio
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from src.agents.experiment_agent.agents.master import run_master
from src.agents.experiment_agent.agents.prepare import run_prepare
from src.agents.experiment_agent.agents.finalization import run_finalization_agent
from src.agents.experiment_agent.config import print_config
from src.agents.experiment_agent.config import (
    copy_prepared_data_to_workspace,
    ensure_experiment_dirs,
    get_idea_input_path,
    write_workspace_env_file,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    ensure_canonical_workspace_artifacts,
    load_json_file,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.telemetry import print_kv_table, print_phase


def get_args():
    parser = argparse.ArgumentParser(
        description="Experiment Agent - unified prepare + orchestration entrypoint"
    )
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment ID (unique identifier)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to unified project config YAML. Falls back to XCIENTIST_CONFIG, then src/config/default.yaml.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run prepare phase and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate managed prepare artifacts and re-download/clone as needed",
    )
    parser.add_argument(
        "--clone-depth", type=int, default=1, help="git clone depth (default: 1)"
    )
    parser.add_argument("--skip-repos", action="store_true", help="Skip cloning repos")
    parser.add_argument(
        "--skip-datasets", action="store_true", help="Skip downloading datasets"
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
    payload = load_json_file(paths["prepare_reviewer"])
    phase_report = normalize_phase_report(payload)
    return (
        phase_report["status"] == "PASS"
        and phase_report["phase_completion_status"] == "complete"
        and phase_report["ready_for_next_phase"] is True
    )


async def main_async(args) -> int:
    result = await run_experiment_once(
        experiment_id=args.experiment,
        workspace_root=os.environ.get("EXPERIMENT_AGENT_WORKSPACE_DIR") or None,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
        prepare_only=bool(args.prepare_only),
        force_prepare=bool(args.force),
        clone_depth=int(args.clone_depth),
        skip_repos=bool(args.skip_repos),
        skip_datasets=bool(args.skip_datasets),
        config_path=args.config,
    )
    return 0 if result.get("ok", True) else 1


async def run_experiment_once(
    *,
    experiment_id: str,
    workspace_root: str | None = None,
    resume: bool = False,
    verbose: bool = False,
    prepare_only: bool = False,
    force_prepare: bool = False,
    clone_depth: int = 1,
    skip_repos: bool = False,
    skip_datasets: bool = False,
    config_path: str | None = None,
    symbolic_memory_path: str | None = None,
) -> Dict[str, Any]:
    env_overrides: Dict[str, Optional[str]] = {}
    if workspace_root:
        env_overrides["EXPERIMENT_AGENT_WORKSPACE_DIR"] = workspace_root
    env_overrides["XCIENTIST_CONFIG"] = config_path if config_path else os.environ.get("XCIENTIST_CONFIG")
    if symbolic_memory_path:
        env_overrides["XCIENTIST_SYMBOLIC_MEMORY_PATH"] = symbolic_memory_path

    with _temporary_environ(env_overrides):
        from src.config import reload_config, resolve_runtime_config_path

        resolved_config_path = resolve_runtime_config_path(config_path)
        os.environ["XCIENTIST_CONFIG"] = str(resolved_config_path)
        runtime_config = reload_config(str(resolved_config_path))

        print_config()
        print_phase(
            "EXPERIMENT AGENT",
            "OpenHarness orchestration with prefinish review gates",
            width=76,
        )

        paths = ensure_experiment_dirs(experiment_id)
        copy_prepared_data_to_workspace(paths["workspace_dir"])
        write_workspace_env_file(experiment_id)
        resolved_workspace_root = paths["workspace_dir"]
        project_root = paths["project_dir"]
        ensure_canonical_workspace_artifacts(resolved_workspace_root, project_root)

        print_kv_table(
            "Run Context",
            {
                "experiment": experiment_id,
                "config": resolved_config_path,
                "workspace": resolved_workspace_root,
                "project": project_root,
                "results": paths["results_dir"],
                "model_candidate": paths["model_dir"],
                "agent_reports": paths["reports_dir"],
            },
            width=88,
            mask_sensitive=False,
        )

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
            ensure_canonical_workspace_artifacts(resolved_workspace_root, project_root)
            prepare_payload = load_json_file(artifact_paths(resolved_workspace_root, project_root)["prepare_reviewer"])
            prepare_phase_report = normalize_phase_report(prepare_payload)
            if not _prepare_ready_for_master(resolved_workspace_root, project_root):
                print_kv_table(
                    "Prepare Blocked",
                    {
                        "status": prepare_phase_report["status"] or "UNKNOWN",
                        "phase_completion_status": prepare_phase_report["phase_completion_status"],
                        "ready_for_next_phase": prepare_phase_report["ready_for_next_phase"],
                        "reviewer_report": artifact_paths(resolved_workspace_root, project_root)["prepare_reviewer"],
                        "blocking_issues": "; ".join(prepare_phase_report["blocking_issues"]) or "(none)",
                        "required_followup": "; ".join(prepare_phase_report["required_followup"]) or "(none)",
                    },
                    width=88,
                    mask_sensitive=False,
                )
                return {
                    "ok": False,
                    "iterations": 0,
                    "converged": False,
                    "final_path": "",
                    "ablation_results_path": "",
                    "workspace_root": resolved_workspace_root,
                    "project_root": project_root,
                    "prepare_only": bool(prepare_only),
                    "prepare_status": prepare_phase_report["status"],
                    "prepare_reviewer_path": artifact_paths(resolved_workspace_root, project_root)["prepare_reviewer"],
                }
            print_kv_table(
                "Prepare Complete",
                {
                    "prepare_idea": prepare_report.idea_md_path,
                    "project_dir": prepare_report.project_dir,
                    "repos_dir": prepare_report.repos_dir,
                    "dataset_dir": prepare_report.dataset_dir,
                    "model_dir": prepare_report.model_dir,
                    "results_dir": prepare_report.results_dir,
                    "agent_reports_dir": prepare_report.reports_dir,
                },
                width=88,
                mask_sensitive=False,
            )
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
            print_phase(
                "PREPARE REUSED",
                "Existing reviewer-approved prepare handoff is ready.",
                width=76,
            )
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
        print_kv_table("Master Input", {"idea": idea_path}, width=88, mask_sensitive=False)

        result = await run_master(
            experiment_id=experiment_id,
            idea_path=idea_path,
            workspace_root=resolved_workspace_root,
            project_root=project_root,
            verbose=bool(verbose),
            resume=bool(resume),
        )

        if result.get("converged"):
            receipt = await run_finalization_agent(
                experiment_id=experiment_id,
                workspace_root=resolved_workspace_root,
                project_root=project_root,
                config=runtime_config,
                verbose=bool(verbose),
            )
            result["symbolic_memory_receipt_path"] = artifact_paths(resolved_workspace_root, project_root)["symbolic_memory_receipt"]
            if receipt.get("status") != "PASS":
                print_kv_table(
                    "Finalization Blocked",
                    {
                        "status": receipt.get("status", "FAIL"),
                        "receipt": result["symbolic_memory_receipt_path"],
                        "blocker": receipt.get("blocker", ""),
                    },
                    width=88,
                    mask_sensitive=False,
                )
                return {
                    "ok": False,
                    "iterations": int(result.get("iterations") or 0),
                    "converged": False,
                    "stopped_due_to_iteration_limit": False,
                    "final_path": "",
                    "ablation_results_path": str(receipt.get("ablation_results_path") or ""),
                    "symbolic_memory_receipt_path": result["symbolic_memory_receipt_path"],
                    "workspace_root": resolved_workspace_root,
                    "project_root": project_root,
                    "idea_path": idea_path,
                    "finalization_status": str(receipt.get("status") or "FAIL"),
                    "finalization_blocker": str(receipt.get("blocker") or ""),
                }
            result["final_path"] = receipt["ablation_results_path"]
            result["ablation_results_path"] = receipt["ablation_results_path"]

        if not result.get("converged"):
            print_phase("EXPERIMENT BLOCKED", width=76)
            print_kv_table(
                "Blocking State",
                {
                    "iterations": int(result.get("iterations") or 0),
                    "decision": str(result.get("decision") or ""),
                    "blocking_issues": "; ".join(str(item) for item in result.get("blocking_issues") or []) or "(none)",
                },
                width=88,
                mask_sensitive=False,
            )
            return {
                "ok": False,
                "iterations": int(result.get("iterations") or 0),
                "converged": False,
                "stopped_due_to_iteration_limit": bool(result.get("stopped_due_to_iteration_limit")),
                "final_path": str(result.get("final_path") or ""),
                "ablation_results_path": str(result.get("ablation_results_path") or ""),
                "symbolic_memory_receipt_path": str(result.get("symbolic_memory_receipt_path") or ""),
                "workspace_root": resolved_workspace_root,
                "project_root": project_root,
                "idea_path": idea_path,
                "decision": str(result.get("decision") or ""),
                "blocking_issues": list(result.get("blocking_issues") or []),
            }

        print_phase("MISSION COMPLETE", width=76)
        print_kv_table(
            "Final Artifacts",
            {
                "iterations": result["iterations"],
                "final_report": result.get("final_path", result.get("ablation_results_path")),
                "symbolic_memory_receipt": result.get("symbolic_memory_receipt_path", ""),
            },
            width=88,
            mask_sensitive=False,
        )

        return {
            "ok": True,
            "iterations": int(result.get("iterations") or 0),
            "converged": bool(result.get("converged")),
            "stopped_due_to_iteration_limit": bool(result.get("stopped_due_to_iteration_limit")),
            "final_path": str(result.get("final_path") or ""),
            "ablation_results_path": str(result.get("ablation_results_path") or ""),
            "symbolic_memory_receipt_path": str(result.get("symbolic_memory_receipt_path") or ""),
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
