"""
Lightweight workspace artifact helpers for experiment-agent.

This module intentionally avoids semantic validation of experiment outputs.
Planner/worker/validator agents own correctness decisions. The runtime only
provides small helpers for loading and writing workspace artifacts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def workspace_contract_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
    workspace_dir = os.path.realpath(workspace_root)
    project_dir = os.path.realpath(project_root or os.path.join(workspace_dir, "project"))
    repos_dir = os.path.join(workspace_dir, "repos")
    dataset_dir = os.path.join(workspace_dir, "dataset_candidate")
    results_dir = os.path.join(workspace_dir, "results")
    agent_reports_dir = os.path.join(workspace_dir, "agent_reports")
    return {
        "workspace_dir": workspace_dir,
        "project_dir": project_dir,
        "repos_dir": repos_dir,
        "dataset_dir": dataset_dir,
        "results_dir": results_dir,
        "standard_results_dir": os.path.join(results_dir, "standard"),
        "ablation_results_dir": os.path.join(results_dir, "ablation"),
        "agent_reports_dir": agent_reports_dir,
        "env_file": os.path.join(workspace_dir, ".env"),
    }


def artifact_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
    contract = workspace_contract_paths(workspace_root, project_root)
    reports_dir = contract["agent_reports_dir"]
    return {
        **contract,
        "idea": os.path.join(reports_dir, "prepare_idea.md"),
        "idea_json": os.path.join(contract["workspace_dir"], "idea.json"),
        "prepare_runtime_debug": os.path.join(reports_dir, "prepare_runtime_debug.json"),
        "prepare_target_inventory": os.path.join(reports_dir, "prepare_target_inventory.json"),
        "prepare_planner_report": os.path.join(reports_dir, "prepare_planner_report.json"),
        "code_plan": os.path.join(reports_dir, "code_plan.json"),
        "code_planner_report": os.path.join(reports_dir, "code_planner_report.json"),
        "code_summary": os.path.join(reports_dir, "code_summary.md"),
        "code_usage": os.path.join(reports_dir, "code_usage.md"),
        "code_integration_readiness": os.path.join(reports_dir, "code_integration_readiness.json"),
        "standard_science_plan": os.path.join(reports_dir, "standard_science_plan.json"),
        "standard_science_planner_report": os.path.join(reports_dir, "standard_science_planner_report.json"),
        "standard_summary": os.path.join(reports_dir, "standard_science_summary.md"),
        "ablation_science_plan": os.path.join(reports_dir, "ablation_science_plan.json"),
        "ablation_science_planner_report": os.path.join(reports_dir, "ablation_science_planner_report.json"),
        "ablation_summary": os.path.join(reports_dir, "ablation_science_summary.md"),
        "master_report": os.path.join(reports_dir, "master_report.md"),
        "results_summary": os.path.join(reports_dir, "master_summary.md"),
        "ablation_results": os.path.join(contract["workspace_dir"], "ablation_results.json"),
        "ablation_report_integrator_report": os.path.join(
            reports_dir, "ablation_report_integrator_report.json"
        ),
        "prepare_plan": os.path.join(reports_dir, "prepare_plan.json"),
        "prepare_repo_worker": os.path.join(reports_dir, "prepare_repo_worker_report.json"),
        "prepare_repo_validator": os.path.join(reports_dir, "prepare_repo_validator_report.json"),
        "prepare_env_worker": os.path.join(reports_dir, "prepare_env_worker_report.json"),
        "prepare_env_validator": os.path.join(reports_dir, "prepare_env_validator_report.json"),
        "prepare_dataset_worker": os.path.join(reports_dir, "prepare_dataset_worker_report.json"),
        "prepare_dataset_validator": os.path.join(reports_dir, "prepare_dataset_validator_report.json"),
        "prepare_handoff_worker": os.path.join(reports_dir, "prepare_handoff_worker_report.json"),
        "prepare_validator": os.path.join(reports_dir, "prepare_validator_report.json"),
        "prepare_phase_report": os.path.join(reports_dir, "prepare_validator_report.json"),
        "code_worker": os.path.join(reports_dir, "code_worker_report.json"),
        "code_validator": os.path.join(reports_dir, "code_validator_report.json"),
        "code_phase_report": os.path.join(reports_dir, "code_validator_report.json"),
        "standard_science_validator": os.path.join(
            reports_dir, "standard_science_validator_report.json"
        ),
        "standard_science_phase_report": os.path.join(
            reports_dir, "standard_science_validator_report.json"
        ),
        "ablation_science_validator": os.path.join(
            reports_dir, "ablation_science_validator_report.json"
        ),
        "ablation_science_phase_report": os.path.join(
            reports_dir, "ablation_science_validator_report.json"
        ),
    }

def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
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
    "load_json_file",
    "load_workspace_state",
    "workspace_contract_paths",
    "write_json_file",
]
