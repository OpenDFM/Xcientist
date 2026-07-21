"""
Deterministic ablation-results materialization helpers.

This module keeps the final ``ablation_results.json`` contract under runtime
control. It materializes only reviewer-approved science evidence, then validates
and writes the final artifact shape here.
"""

from __future__ import annotations

import json
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
from src.agents.experiment_agent.runtime.artifacts import record_runtime_artifact
from src.agents.experiment_agent.runtime.phase_contracts import (
    ARTIFACT_ROLE_FINAL_RESULT,
    ARTIFACT_ROLE_PHASE_RESULT,
    normalize_phase_report,
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
FINAL_COMPONENT_RESULTS = ("positive", "negative", "neutral", "inconclusive")


def build_ablation_results_manifest(
    workspace_root: str,
    idea_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    canonical_records = _canonical_component_records(
        workspace_root,
        idea_json_path=idea_json_path,
    )
    return {
        "artifact_name": "ablation_results",
        "artifact_role": ARTIFACT_ROLE_FINAL_RESULT,
        "top_level_keys": ["components", "summary"],
        "component_order_source": "idea.json.components",
        "component_required_fields": list(REQUIRED_COMPONENT_FIELDS),
        "summary_required_fields": list(REQUIRED_SUMMARY_FIELDS),
        "canonical_components": [record["component"] for record in canonical_records],
    }


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
    for key in ("science_component_results", "components"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {
                str(name): data
                for name, data in value.items()
                if isinstance(name, str) and isinstance(data, dict)
            }
    return {}


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
    # `result` vocabulary:
    #   - `positive`    : removing this component clearly hurts the metric (component helps).
    #   - `negative`    : removing this component clearly helps the metric (component hurts).
    #   - `neutral`     : the ablation ran end-to-end and the metric difference is
    #                     within noise or small enough to treat as no practical effect.
    #   - `inconclusive`: the ablation ran end-to-end but the evidence does not support
    #                     a directional conclusion. It is still useful symbolic memory
    #                     as long as `follow_up_required` is false.
    result = str(source_payload.get("result") or "").strip().lower()
    if result not in {"positive", "negative", "neutral", "inconclusive"}:
        return None, f"Component `{name}` has invalid result `{result}`."
    if bool(source_payload.get("follow_up_required")):
        return None, f"Component `{name}` still requires follow-up."
    if confidence < 0.0 or confidence > 1.0:
        return None, f"Component `{name}` confidence must be in [0.0, 1.0]."
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
        return None, "Ablation reviewer report is missing summary."

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
        result = entry.get("result")
        if result not in FINAL_COMPONENT_RESULTS:
            return False, (
                f"Component `{name}` result must be one of {list(FINAL_COMPONENT_RESULTS)}, "
                f"got {result!r}."
            )
        confidence = _coerce_float(entry.get("confidence"))
        if confidence is None or confidence < 0.0 or confidence > 1.0:
            return False, f"Component `{name}` confidence must be numeric in [0.0, 1.0]."
        for field in ("metric", "value", "analysis", "method_context"):
            if not isinstance(entry.get(field), str) or not entry.get(field).strip():
                return False, f"Component `{name}` field `{field}` must be a non-empty string."

    required_summary_fields = set(REQUIRED_SUMMARY_FIELDS)
    if set(summary.keys()) != required_summary_fields:
        return False, f"Summary must contain exactly {sorted(required_summary_fields)}."
    if not isinstance(summary.get("feasible"), bool):
        return False, "Summary `feasible` must be a boolean."
    summary_confidence = _coerce_float(summary.get("confidence"))
    if summary_confidence is None or summary_confidence < 0.0 or summary_confidence > 1.0:
        return False, "Summary `confidence` must be numeric in [0.0, 1.0]."
    key_findings = summary.get("key_findings")
    if (
        not isinstance(key_findings, list)
        or not key_findings
        or not all(isinstance(item, str) and item.strip() for item in key_findings)
    ):
        return False, "Summary `key_findings` must be a non-empty-string list."
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

    phase_report_path = paths["science_reviewer"]
    phase_payload = load_json_file(phase_report_path)
    if not isinstance(phase_payload, dict):
        return {
            "valid": False,
            "blocker": f"Missing science reviewer report at {phase_report_path}.",
            "source_evidence_files": [idea_json_path],
        }

    normalized_phase = normalize_phase_report(phase_payload)
    if normalized_phase["status"] != "PASS":
        return {
            "valid": False,
            "blocker": (
                f"Science reviewer status is "
                f"`{normalized_phase['status'] or 'UNKNOWN'}`, not PASS."
            ),
            "source_evidence_files": [idea_json_path, phase_report_path],
        }
    if normalized_phase["artifact_role"] != ARTIFACT_ROLE_PHASE_RESULT:
        return {
            "valid": False,
            "blocker": (
                "Science reviewer artifact_role must be `phase_result` before "
                "final materialization."
            ),
            "source_evidence_files": [idea_json_path, phase_report_path],
        }
    if normalized_phase["phase_completion_status"] != "complete":
        blockers = normalized_phase["blocking_issues"]
        blocker_text = blockers[0] if blockers else "Science phase is not marked complete."
        return {
            "valid": False,
            "blocker": blocker_text,
            "source_evidence_files": [idea_json_path, phase_report_path],
        }

    source_evidence_files = [idea_json_path, phase_report_path]
    phase_components = _collect_component_map(phase_payload)

    normalized_components: Dict[str, Dict[str, Any]] = {}
    for name in canonical_names:
        component_payload = phase_components.get(name)
        if not isinstance(component_payload, dict):
            return {
                "valid": False,
                "blocker": (
                    f"Missing ablation component evidence for `{name}` in science phase report "
                    f"`{phase_report_path}`. Final ablation materialization only trusts "
                    "`agent_reports/science/phase.json`; repair the science phase aggregation "
                    "so `science_component_results` contains every idea component."
                ),
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
        "artifact_role": ARTIFACT_ROLE_FINAL_RESULT,
        "source_evidence_files": source_evidence_files,
        "canonical_components": canonical_names,
        "blocker": None if valid else error,
        "ablation_results_manifest_path": paths["ablation_results_manifest"],
        "final_artifact_contract_path": paths["ablation_results_manifest"],
    }
    return {
        "valid": bool(valid),
        "payload": payload if valid else None,
        "ablation_results_manifest": build_ablation_results_manifest(
            workspace_root,
            idea_json_path=idea_json_path,
        ),
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
    materialization_report_path: Optional[str] = None,
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
    target_report_path = materialization_report_path or paths["ablation_materialization_report"]
    target_manifest_path = paths["ablation_results_manifest"]
    write_json_file(target_results_path, result["payload"])
    record_runtime_artifact(
        workspace_root=workspace_root,
        artifact_id="final.ablation_results",
        path=target_results_path,
        stage="finalization",
        step_id="ablation_results",
        extra={"generated_by": generated_by},
    )
    write_json_file(target_report_path, result["report"])
    record_runtime_artifact(
        workspace_root=workspace_root,
        artifact_id="runtime.ablation_materialization_report",
        path=target_report_path,
        stage="finalization",
        step_id="ablation_results",
        extra={"generated_by": generated_by},
    )
    manifest = result["ablation_results_manifest"]
    write_json_file(target_manifest_path, manifest)
    record_runtime_artifact(
        workspace_root=workspace_root,
        artifact_id="runtime.ablation_results_manifest",
        path=target_manifest_path,
        stage="finalization",
        step_id="ablation_results",
        extra={"generated_by": generated_by},
    )
    result["ablation_results_path"] = target_results_path
    result["materialization_report_path"] = target_report_path
    result["ablation_results_manifest_path"] = target_manifest_path
    result["final_artifact_contract_path"] = target_manifest_path
    result["final_artifact_contract"] = manifest
    return result


__all__ = [
    "REQUIRED_COMPONENT_FIELDS",
    "REQUIRED_SUMMARY_FIELDS",
    "FINAL_COMPONENT_RESULTS",
    "build_ablation_results_manifest",
    "build_ablation_results_artifacts",
    "validate_ablation_results_payload",
    "write_ablation_results_artifacts",
]
