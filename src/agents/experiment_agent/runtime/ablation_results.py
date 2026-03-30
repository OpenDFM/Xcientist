"""
Deterministic ablation-results materialization helpers.

This module keeps the final ``ablation_results.json`` contract under runtime
control. LLM-based integrators may still assist with evidence discovery, but
the final artifact shape is validated and written here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.agents.experiment_agent.runtime.idea_components import (
    find_idea_json_path,
    load_idea_json,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    write_json_file,
)


REQUIRED_COMPONENT_FIELDS = (
    "result",
    "metric",
    "value",
    "confidence",
    "analysis",
    "method_context",
)
REQUIRED_SUMMARY_FIELDS = ("feasible", "confidence", "key_findings")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _canonical_component_records(
    workspace_root: str,
    idea_json_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    payload = load_idea_json(workspace_root, idea_json_path=idea_json_path)
    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("idea.json.components must be a non-empty list.")

    records: List[Dict[str, str]] = []
    seen = set()
    for index, item in enumerate(raw_components, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"idea.json.components[{index - 1}] must be an object.")
        name = str(item.get("component") or "").strip()
        if not name:
            raise ValueError(f"idea.json.components[{index - 1}] is missing `component`.")
        if name in seen:
            raise ValueError(f"idea.json.components contains duplicate component `{name}`.")
        seen.add(name)
        method_context = str(
            item.get("description")
            or item.get("explanation")
            or item.get("summary")
            or ""
        ).strip()
        records.append(
            {
                "component": name,
                "method_context": method_context,
                "index": str(index),
            }
        )
    return records


def _collect_component_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    for key in ("ablation_components", "components", "component_results"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {
                str(name): data
                for name, data in value.items()
                if isinstance(name, str) and isinstance(data, dict)
            }
    return {}


def _collect_step_component_map(agent_reports_dir: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    component_map: Dict[str, Dict[str, Any]] = {}
    evidence_files: List[str] = []
    if not os.path.isdir(agent_reports_dir):
        return component_map, evidence_files

    for name in sorted(os.listdir(agent_reports_dir)):
        if not name.endswith(".json"):
            continue
        lowered = name.lower()
        if "ablation" not in lowered or "validator" not in lowered:
            continue
        path = os.path.join(agent_reports_dir, name)
        payload = load_json_file(path)
        if not isinstance(payload, dict):
            continue
        component_name = str(
            payload.get("component_or_condition")
            or payload.get("component")
            or payload.get("component_name")
            or ""
        ).strip()
        if not component_name:
            continue
        if all(field in payload for field in ("result", "metric", "value", "confidence", "analysis")):
            component_map.setdefault(component_name, payload)
            evidence_files.append(path)
    return component_map, evidence_files


def _normalize_component_entry(
    *,
    name: str,
    source_payload: Dict[str, Any],
    method_context: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    missing = [field for field in ("result", "metric", "value", "confidence", "analysis") if field not in source_payload]
    if missing:
        return None, f"Component `{name}` is missing fields: {', '.join(missing)}."

    confidence = _coerce_float(source_payload.get("confidence"))
    if confidence is None:
        return None, f"Component `{name}` has non-numeric confidence."
    result = str(source_payload.get("result") or "").strip().lower()
    if result not in {"positive", "negative", "inconclusive"}:
        return None, f"Component `{name}` has invalid result `{result}`."
    if bool(source_payload.get("follow_up_required")):
        return None, f"Component `{name}` still requires follow-up."
    if result == "inconclusive":
        return None, f"Component `{name}` is still inconclusive."
    if confidence < 0.6:
        return None, f"Component `{name}` confidence {confidence:.2f} is below 0.6."
    if not method_context:
        return None, f"Component `{name}` is missing canonical method_context."

    entry = {
        "result": result,
        "metric": str(source_payload.get("metric") or "").strip(),
        "value": str(source_payload.get("value") or "").strip(),
        "confidence": confidence,
        "analysis": str(source_payload.get("analysis") or "").strip(),
        "method_context": method_context,
    }
    if any(entry[field] in ("", None) for field in REQUIRED_COMPONENT_FIELDS):
        return None, f"Component `{name}` has empty required fields."
    return entry, None


def _normalize_summary(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_summary = payload.get("summary") or payload.get("experiment_summary")
    if not isinstance(raw_summary, dict):
        return None, "Ablation validator report is missing summary."

    feasible = _coerce_bool(raw_summary.get("feasible"))
    confidence = _coerce_float(raw_summary.get("confidence"))
    key_findings = raw_summary.get("key_findings")
    if feasible is None:
        return None, "Summary is missing boolean `feasible`."
    if confidence is None:
        return None, "Summary is missing numeric `confidence`."
    if not isinstance(key_findings, list):
        return None, "Summary is missing list `key_findings`."

    normalized_findings = [str(item).strip() for item in key_findings if str(item).strip()]
    return {
        "feasible": feasible,
        "confidence": confidence,
        "key_findings": normalized_findings,
    }, None


def validate_ablation_results_payload(
    payload: Dict[str, Any],
    *,
    canonical_component_names: List[str],
) -> Tuple[bool, Optional[str]]:
    if not isinstance(payload, dict):
        return False, "Payload must be an object."
    if set(payload.keys()) != {"components", "summary"}:
        return False, "Payload must contain exactly `components` and `summary`."

    components = payload.get("components")
    summary = payload.get("summary")
    if not isinstance(components, dict) or not isinstance(summary, dict):
        return False, "Payload components/summary must both be objects."
    if list(components.keys()) != list(canonical_component_names):
        return False, "Component keys must match canonical component order exactly."

    required_component_fields = set(REQUIRED_COMPONENT_FIELDS)
    for name in canonical_component_names:
        entry = components.get(name)
        if not isinstance(entry, dict):
            return False, f"Component `{name}` entry must be an object."
        if set(entry.keys()) != required_component_fields:
            return False, f"Component `{name}` must contain exactly {sorted(required_component_fields)}."
        if any(entry.get(field) in (None, "") for field in REQUIRED_COMPONENT_FIELDS):
            return False, f"Component `{name}` has empty required fields."

    required_summary_fields = set(REQUIRED_SUMMARY_FIELDS)
    if set(summary.keys()) != required_summary_fields:
        return False, f"Summary must contain exactly {sorted(required_summary_fields)}."
    if summary.get("feasible") is None or summary.get("confidence") is None:
        return False, "Summary requires feasible/confidence."
    if not isinstance(summary.get("key_findings"), list):
        return False, "Summary key_findings must be a list."
    return True, None


def build_ablation_results_artifacts(
    workspace_root: str,
    project_root: Optional[str] = None,
    *,
    generated_by: str = "runtime",
) -> Dict[str, Any]:
    paths = artifact_paths(workspace_root, project_root)
    idea_json_path = find_idea_json_path(workspace_root) or paths["idea_json"]
    try:
        canonical_records = _canonical_component_records(
            workspace_root,
            idea_json_path=idea_json_path,
        )
    except Exception as exc:
        return {
            "valid": False,
            "blocker": f"Failed to load canonical components: {exc}",
            "source_evidence_files": [idea_json_path],
        }
    canonical_names = [record["component"] for record in canonical_records]
    canonical_context = {record["component"]: record["method_context"] for record in canonical_records}

    phase_payload = load_json_file(paths["ablation_science_validator"])
    if not isinstance(phase_payload, dict):
        return {
            "valid": False,
            "blocker": f"Missing ablation validator report at {paths['ablation_science_validator']}.",
            "source_evidence_files": [idea_json_path],
        }

    status = str(phase_payload.get("status") or "").strip().upper()
    if status != "PASS":
        return {
            "valid": False,
            "blocker": f"Ablation validator status is `{status or 'UNKNOWN'}`, not PASS.",
            "source_evidence_files": [idea_json_path, paths["ablation_science_validator"]],
        }

    source_evidence_files = [idea_json_path, paths["ablation_science_validator"]]
    phase_components = _collect_component_map(phase_payload)
    step_components, step_files = _collect_step_component_map(paths["agent_reports_dir"])
    source_evidence_files.extend(step_files)

    normalized_components: Dict[str, Dict[str, Any]] = {}
    for name in canonical_names:
        component_payload = phase_components.get(name) or step_components.get(name)
        if not isinstance(component_payload, dict):
            return {
                "valid": False,
                "blocker": f"Missing ablation component evidence for `{name}`.",
                "source_evidence_files": source_evidence_files,
            }
        normalized_entry, error = _normalize_component_entry(
            name=name,
            source_payload=component_payload,
            method_context=canonical_context.get(name, ""),
        )
        if error:
            return {
                "valid": False,
                "blocker": error,
                "source_evidence_files": source_evidence_files,
            }
        normalized_components[name] = normalized_entry or {}

    normalized_summary, summary_error = _normalize_summary(phase_payload)
    if summary_error:
        return {
            "valid": False,
            "blocker": summary_error,
            "source_evidence_files": source_evidence_files,
        }

    payload = {
        "components": normalized_components,
        "summary": normalized_summary,
    }
    valid, error = validate_ablation_results_payload(
        payload,
        canonical_component_names=canonical_names,
    )
    report = {
        "valid": bool(valid),
        "generated_by": generated_by,
        "mode": "deterministic",
        "source_evidence_files": source_evidence_files,
        "canonical_components": canonical_names,
        "blocker": None if valid else error,
    }
    return {
        "valid": bool(valid),
        "payload": payload if valid else None,
        "report": report,
        "blocker": None if valid else error,
        "source_evidence_files": source_evidence_files,
    }


def write_ablation_results_artifacts(
    workspace_root: str,
    project_root: Optional[str] = None,
    *,
    generated_by: str = "runtime",
    ablation_results_path: Optional[str] = None,
    integrator_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    paths = artifact_paths(workspace_root, project_root)
    result = build_ablation_results_artifacts(
        workspace_root,
        project_root,
        generated_by=generated_by,
    )
    if not result.get("valid"):
        return result

    target_results_path = ablation_results_path or paths["ablation_results"]
    target_report_path = integrator_report_path or paths["ablation_report_integrator_report"]
    write_json_file(target_results_path, result["payload"])
    write_json_file(target_report_path, result["report"])
    result["ablation_results_path"] = target_results_path
    result["integrator_report_path"] = target_report_path
    return result


__all__ = [
    "REQUIRED_COMPONENT_FIELDS",
    "REQUIRED_SUMMARY_FIELDS",
    "build_ablation_results_artifacts",
    "validate_ablation_results_payload",
    "write_ablation_results_artifacts",
]
