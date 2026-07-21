"""Workspace artifact paths and small JSON helpers for experiment-agent.

Runtime-owned contracts and hooks enforce artifact locations, schema shape, and
phase boundaries elsewhere. This module is the central path map used by those
checks and by phase agents.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.runtime.report_layout import ReportLayout


def workspace_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
    workspace_dir = os.path.realpath(workspace_root)
    project_dir = os.path.realpath(project_root or os.path.join(workspace_dir, "project"))
    repos_dir = os.path.join(workspace_dir, "repos")
    dataset_dir = os.path.join(workspace_dir, "dataset_candidate")
    model_dir = os.path.join(workspace_dir, "model_candidate")
    model_share_dir = os.path.join(model_dir, "model_share")
    results_dir = os.path.join(workspace_dir, "results")
    agent_reports_dir = os.path.join(workspace_dir, "agent_reports")
    return {
        "workspace_dir": workspace_dir,
        "project_dir": project_dir,
        "repos_dir": repos_dir,
        "dataset_dir": dataset_dir,
        "model_dir": model_dir,
        "model_share_dir": model_share_dir,
        "results_dir": results_dir,
        "science_results_dir": os.path.join(results_dir, "science"),
        "agent_reports_dir": agent_reports_dir,
        "env_file": os.path.join(workspace_dir, ".env"),
    }


def artifact_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
    contract = workspace_paths(workspace_root, project_root)
    layout = ReportLayout(contract["workspace_dir"])
    return {
        **contract,
        "idea": layout.phase_file("prepare", "artifacts", "idea.md"),
        "idea_json": os.path.join(contract["workspace_dir"], "idea.json"),
        "prepare_target_inventory": layout.phase_file("prepare", "artifacts", "target_inventory.json"),
        "prepare_discovery": layout.phase_file("prepare", "artifacts", "discovery.json"),
        "prepare_repos": layout.phase_file("prepare", "artifacts", "repos.json"),
        "prepare_dataset": layout.phase_file("prepare", "artifacts", "dataset.json"),
        "prepare_model": layout.phase_file("prepare", "artifacts", "model.json"),
        "prepare_env": layout.phase_file("prepare", "artifacts", "env.json"),
        "artifact_ledger": layout.artifact_ledger,
        "artifact_registry": layout.artifact_registry,
        "run_timeline": layout.run_timeline,
        "stray_outputs": layout.stray_outputs,
        "mcp_status": layout.phase_file("runtime", "mcp_status.json"),
        "prepare_planner_report": layout.planner_file("prepare", "planner_report.json"),
        "prepare_blueprint": layout.planner_file("prepare", "blueprint.md"),
        "prepare_plan": layout.planner_file("prepare", "latest.json"),
        "prepare_executable_plan": layout.planner_file("prepare", "executable.json"),
        "prepare_repo_worker": layout.phase_file("prepare", "worker", "repos", "latest.json"),
        "prepare_repo_reviewer": layout.phase_file("prepare", "review", "repos", "latest.json"),
        "prepare_env_worker": layout.phase_file("prepare", "worker", "env", "latest.json"),
        "prepare_env_reviewer": layout.phase_file("prepare", "review", "env", "latest.json"),
        "prepare_dataset_worker": layout.phase_file("prepare", "worker", "dataset", "latest.json"),
        "prepare_dataset_reviewer": layout.phase_file("prepare", "review", "dataset", "latest.json"),
        "prepare_model_worker": layout.phase_file("prepare", "worker", "model", "latest.json"),
        "prepare_model_reviewer": layout.phase_file("prepare", "review", "model", "latest.json"),
        "prepare_handoff_worker": layout.phase_file("prepare", "worker", "synthesis", "latest.json"),
        "prepare_reviewer": layout.phase_file("prepare", "phase.json"),
        "code_plan": layout.planner_file("code", "latest.json"),
        "code_executable_plan": layout.planner_file("code", "executable.json"),
        "code_blueprint": layout.planner_file("code", "blueprint.md"),
        "code_planner_report": layout.planner_file("code", "planner_report.json"),
        "code_summary": layout.phase_file("code", "summary.md"),
        "code_usage": layout.phase_file("code", "usage.md"),
        "code_integration_readiness": layout.phase_file("code", "artifacts", "integration_readiness.json"),
        "code_worker": layout.phase_file("code", "worker", "phase", "latest.json"),
        "code_reviewer": layout.phase_file("code", "phase.json"),
        "science_plan": layout.planner_file("science", "latest.json"),
        "science_executable_plan": layout.planner_file("science", "executable.json"),
        "science_blueprint": layout.planner_file("science", "blueprint.md"),
        "science_planner_report": layout.planner_file("science", "planner_report.json"),
        "science_summary": layout.phase_file("science", "summary.md"),
        "science_reviewer": layout.phase_file("science", "phase.json"),
        "master_report": layout.phase_file("runtime", "master_report.md"),
        "results_summary": layout.phase_file("runtime", "master_summary.md"),
        "ablation_results": layout.phase_file("ablation", "final", "ablation_results.json"),
        "ablation_results_manifest": layout.phase_file("ablation", "final", "ablation_results_manifest.json"),
        "final_artifact_contract": layout.phase_file("ablation", "final", "ablation_results_manifest.json"),
        "ablation_materialization_report": layout.phase_file("ablation", "final", "materialization_report.json"),
        "symbolic_memory_receipt": layout.phase_file("ablation", "final", "symbolic_memory_receipt.json"),
        "master_decision": layout.phase_file("runtime", "master_decision.json"),
        "runtime_phase_state": layout.phase_file("runtime", "phase_state.json"),
        "self_contained_report": layout.phase_file("runtime", "self_contained.json"),
        "execution_budget": layout.phase_file("runtime", "execution_budget.json"),
        "blocker_state": layout.phase_file("runtime", "blocker_state.json"),
        "step_attempt_state": layout.phase_file("runtime", "step_attempt_state.json"),
        "iteration_summary": layout.phase_file("runtime", "iteration_summary.md"),
        "iteration_status": layout.phase_file("runtime", "iteration_status.json"),
    }


def resolve_prepare_idea_path(workspace_root: str, project_root: Optional[str] = None) -> str:
    paths = artifact_paths(workspace_root, project_root)
    return paths["idea"]


def ensure_canonical_workspace_artifacts(
    workspace_root: str,
    project_root: Optional[str] = None,
) -> Dict[str, str]:
    paths = artifact_paths(workspace_root, project_root)
    updates: Dict[str, str] = {}
    layout = ReportLayout(paths["workspace_dir"])
    for directory in (
        layout.reports_dir,
        layout.runtime_dir,
        layout.phase_dir("prepare"),
        layout.phase_dir("code"),
        layout.phase_dir("science"),
        layout.phase_file("prepare", "plan"),
        layout.phase_file("prepare", "artifacts"),
        layout.phase_file("code", "plan"),
        layout.phase_file("code", "artifacts"),
        layout.phase_file("science", "plan"),
        layout.phase_file("science", "evidence"),
        layout.phase_file("ablation"),
        layout.phase_file("ablation", "final"),
    ):
        os.makedirs(directory, exist_ok=True)
    return updates


def load_json_payload(path: str) -> Optional[Any]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_plan_steps(payload: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(payload, list):
        return payload if all(isinstance(item, dict) for item in payload) else None
    if isinstance(payload, dict):
        for key in ("stages", "steps"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return candidate if all(isinstance(item, dict) for item in candidate) else None
    return None

def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    payload = load_json_payload(path)
    return payload if isinstance(payload, dict) else None


def write_json_file(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_env_file(path: str, env_vars: Dict[str, str]) -> str:
    """Write environment variables to a .env file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"{k}={v}" for k, v in env_vars.items()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def load_workspace_state(workspace_root: str) -> Dict[str, Optional[Dict[str, Any]]]:
    paths = artifact_paths(workspace_root)
    return {name: load_json_file(path) for name, path in paths.items() if path.endswith(".json")}


__all__ = [
    "artifact_paths",
    "ensure_canonical_workspace_artifacts",
    "extract_plan_steps",
    "load_json_file",
    "load_json_payload",
    "load_workspace_state",
    "resolve_prepare_idea_path",
    "workspace_paths",
    "write_json_file",
]
