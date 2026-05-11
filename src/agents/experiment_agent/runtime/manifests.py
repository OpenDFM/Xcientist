"""
Lightweight workspace artifact helpers for experiment-agent.

This module intentionally avoids semantic validation of experiment outputs.
Planner/worker/validator agents own correctness decisions. The runtime only
provides small helpers for loading and writing workspace artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional


def workspace_contract_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
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
        "standard_results_dir": os.path.join(results_dir, "standard"),
        "ablation_results_dir": os.path.join(results_dir, "ablation"),
        "agent_reports_dir": agent_reports_dir,
        "env_file": os.path.join(workspace_dir, ".env"),
    }


def artifact_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
    contract = workspace_contract_paths(workspace_root, project_root)
    reports_dir = contract["agent_reports_dir"]
    claude_trace_dir = os.path.join(contract["workspace_dir"], "logs", "claude_code")
    return {
        **contract,
        "idea": os.path.join(reports_dir, "prepare_idea.md"),
        "idea_json": os.path.join(contract["workspace_dir"], "idea.json"),
        "prepare_target_inventory": os.path.join(reports_dir, "prepare_target_inventory.json"),
        "project_code_provenance": os.path.join(reports_dir, "project_code_provenance.json"),
        "prepare_planner_report": os.path.join(reports_dir, "prepare_planner_report.json"),
        "prepare_blueprint": os.path.join(reports_dir, "prepare_blueprint.md"),
        "prepare_plan": os.path.join(reports_dir, "prepare_plan.json"),
        "prepare_repo_worker": os.path.join(reports_dir, "prepare_repo_worker_report.json"),
        "prepare_repo_validator": os.path.join(reports_dir, "prepare_repo_validator_report.json"),
        "prepare_env_worker": os.path.join(reports_dir, "prepare_env_worker_report.json"),
        "prepare_env_validator": os.path.join(reports_dir, "prepare_env_validator_report.json"),
        "prepare_dataset_worker": os.path.join(reports_dir, "prepare_dataset_worker_report.json"),
        "prepare_dataset_validator": os.path.join(reports_dir, "prepare_dataset_validator_report.json"),
        "prepare_model_worker": os.path.join(reports_dir, "prepare_model_worker_report.json"),
        "prepare_model_validator": os.path.join(reports_dir, "prepare_model_validator_report.json"),
        "prepare_handoff_worker": os.path.join(reports_dir, "prepare_handoff_worker_report.json"),
        "prepare_validator": os.path.join(reports_dir, "prepare_validator_report.json"),
        "code_plan": os.path.join(reports_dir, "code_plan.json"),
        "code_blueprint": os.path.join(reports_dir, "code_blueprint.md"),
        "code_planner_report": os.path.join(reports_dir, "code_planner_report.json"),
        "code_summary": os.path.join(reports_dir, "code_summary.md"),
        "code_usage": os.path.join(reports_dir, "code_usage.md"),
        "code_integration_readiness": os.path.join(reports_dir, "code_integration_readiness.json"),
        "standard_science_plan": os.path.join(reports_dir, "standard_science_plan.json"),
        "standard_science_blueprint": os.path.join(reports_dir, "standard_science_blueprint.md"),
        "standard_science_planner_report": os.path.join(reports_dir, "standard_science_planner_report.json"),
        "standard_summary": os.path.join(reports_dir, "standard_science_summary.md"),
        "ablation_science_plan": os.path.join(reports_dir, "ablation_science_plan.json"),
        "ablation_science_blueprint": os.path.join(reports_dir, "ablation_science_blueprint.md"),
        "ablation_science_planner_report": os.path.join(reports_dir, "ablation_science_planner_report.json"),
        "ablation_summary": os.path.join(reports_dir, "ablation_science_summary.md"),
        "master_report": os.path.join(reports_dir, "master_report.md"),
        "results_summary": os.path.join(reports_dir, "master_summary.md"),
        "ablation_results": os.path.join(contract["workspace_dir"], "ablation_results.json"),
        "final_artifact_contract": os.path.join(reports_dir, "final_artifact_contract.json"),
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
        "prepare_model_worker": os.path.join(reports_dir, "prepare_model_worker_report.json"),
        "prepare_model_validator": os.path.join(reports_dir, "prepare_model_validator_report.json"),
        "prepare_handoff_worker": os.path.join(reports_dir, "prepare_handoff_worker_report.json"),
        "prepare_validator": os.path.join(reports_dir, "prepare_validator_report.json"),
        "code_worker": os.path.join(reports_dir, "code_worker_report.json"),
        "code_validator": os.path.join(reports_dir, "code_validator_report.json"),
        "standard_science_validator": os.path.join(
            reports_dir, "standard_science_validator_report.json"
        ),
        "ablation_science_validator": os.path.join(
            reports_dir, "ablation_science_validator_report.json"
        ),
        "master_decision": os.path.join(reports_dir, "master_decision.json"),
        "runtime_phase_state": os.path.join(reports_dir, "runtime_phase_state.json"),
        "self_contained_report": os.path.join(reports_dir, "self_contained_report.json"),
        "execution_budget": os.path.join(reports_dir, "execution_budget.json"),
        "blocker_state": os.path.join(reports_dir, "blocker_state.json"),
        "step_attempt_state": os.path.join(reports_dir, "step_attempt_state.json"),
        "iteration_summary": os.path.join(reports_dir, "iteration_summary.md"),
        "iteration_status": os.path.join(reports_dir, "iteration_status.json"),
        "claude_trace_dir": claude_trace_dir,
        "claude_trace_latest": os.path.join(claude_trace_dir, "latest.json"),
        "claude_trace_index": os.path.join(claude_trace_dir, "index.jsonl"),
        "claude_trace_errors": os.path.join(claude_trace_dir, "trace_errors.log"),
        "claude_trace_index_report": os.path.join(reports_dir, "claude_trace_index.json"),
    }


def resolve_prepare_idea_path(workspace_root: str, project_root: Optional[str] = None) -> str:
    paths = artifact_paths(workspace_root, project_root)
    candidates = [
        paths["idea"],
        os.path.join(paths["workspace_dir"], "prepare_idea.md"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return paths["idea"]


def resolve_provenance_manifest_path(
    workspace_root: str,
    project_root: Optional[str] = None,
) -> str:
    paths = artifact_paths(workspace_root, project_root)
    canonical = paths["project_code_provenance"]
    legacy = os.path.join(paths["agent_reports_dir"], "provenance_manifest.json")
    if os.path.exists(canonical):
        return canonical
    if os.path.exists(legacy):
        return legacy
    return canonical


def provenance_manifest_exists(
    workspace_root: str,
    project_root: Optional[str] = None,
) -> bool:
    return os.path.exists(resolve_provenance_manifest_path(workspace_root, project_root))


def ensure_canonical_workspace_artifacts(
    workspace_root: str,
    project_root: Optional[str] = None,
) -> Dict[str, str]:
    paths = artifact_paths(workspace_root, project_root)
    updates: Dict[str, str] = {}

    legacy_prepare = os.path.join(paths["workspace_dir"], "prepare_idea.md")
    if os.path.exists(legacy_prepare) and not os.path.exists(paths["idea"]):
        os.makedirs(os.path.dirname(paths["idea"]), exist_ok=True)
        shutil.copy2(legacy_prepare, paths["idea"])
        updates["idea"] = paths["idea"]

    legacy_provenance = os.path.join(paths["agent_reports_dir"], "provenance_manifest.json")
    if os.path.exists(legacy_provenance) and not os.path.exists(paths["project_code_provenance"]):
        os.makedirs(os.path.dirname(paths["project_code_provenance"]), exist_ok=True)
        shutil.copy2(legacy_provenance, paths["project_code_provenance"])
        updates["project_code_provenance"] = paths["project_code_provenance"]

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


def _planner_report_path_for_plan(plan_path: str) -> str:
    directory = os.path.dirname(plan_path)
    name = os.path.basename(plan_path)
    if name == "prepare_plan.json":
        return os.path.join(directory, "prepare_planner_report.json")
    if name == "code_plan.json":
        return os.path.join(directory, "code_planner_report.json")
    if name == "standard_science_plan.json":
        return os.path.join(directory, "standard_science_planner_report.json")
    if name == "ablation_science_plan.json":
        return os.path.join(directory, "ablation_science_planner_report.json")
    return ""


def _plan_metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines: List[str] = []
        for item in value:
            if item is None:
                continue
            text = _plan_metadata_text(item)
            if text:
                lines.append(text)
        return "\n".join(lines).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2).strip()
    return str(value).strip()


def _fill_plan_metadata(payload: Dict[str, Any], plan_path: str) -> Dict[str, Any]:
    normalized = dict(payload)
    report_path = _planner_report_path_for_plan(plan_path)
    report_payload = load_json_file(report_path) if report_path else None
    if not normalized.get("summary"):
        normalized["summary"] = (
            report_payload.get("summary")
            if isinstance(report_payload, dict)
            else ""
        )
    if not normalized.get("usage_notes"):
        normalized["usage_notes"] = (
            report_payload.get("usage_notes")
            if isinstance(report_payload, dict)
            else ""
        )
    normalized["summary"] = _plan_metadata_text(normalized.get("summary"))
    normalized["usage_notes"] = _plan_metadata_text(normalized.get("usage_notes"))
    return normalized


def coerce_plan_payload(payload: Any, plan_path: str, *, scope: str) -> Dict[str, Any]:
    """Return a planner payload, falling back to the planner-written artifact."""
    if isinstance(payload, dict) and isinstance(payload.get("stages"), list):
        return _fill_plan_metadata(payload, plan_path)

    artifact_payload = load_json_file(plan_path)
    if isinstance(artifact_payload, dict) and isinstance(artifact_payload.get("stages"), list):
        artifact_payload = _fill_plan_metadata(artifact_payload, plan_path)
        print(
            f"[claude-plan] using {scope} plan artifact because Claude stdout did not contain top-level stages: {plan_path}",
            flush=True,
        )
        return artifact_payload

    payload_keys = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    artifact_keys = (
        sorted(artifact_payload.keys()) if isinstance(artifact_payload, dict) else artifact_payload
    )
    raise ValueError(
        f"{scope} planner did not return top-level stages and no valid plan artifact was found. "
        f"stdout_payload={payload_keys}; plan_path={plan_path}; artifact_payload={artifact_keys}"
    )


def write_json_file(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def ensure_claude_trace_paths(workspace_root: str, project_root: Optional[str] = None) -> Dict[str, str]:
    paths = artifact_paths(workspace_root, project_root)
    os.makedirs(paths["claude_trace_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(paths["claude_trace_index_report"]), exist_ok=True)
    return {
        "trace_dir": paths["claude_trace_dir"],
        "latest_path": paths["claude_trace_latest"],
        "index_path": paths["claude_trace_index"],
        "errors_path": paths["claude_trace_errors"],
        "report_path": paths["claude_trace_index_report"],
    }


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
    "ensure_claude_trace_paths",
    "ensure_canonical_workspace_artifacts",
    "extract_plan_steps",
    "load_json_file",
    "load_json_payload",
    "load_workspace_state",
    "provenance_manifest_exists",
    "resolve_prepare_idea_path",
    "resolve_provenance_manifest_path",
    "workspace_contract_paths",
    "write_json_file",
]
