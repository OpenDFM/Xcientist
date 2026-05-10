"""
Generic phase-report normalization helpers.

These helpers keep runtime routing logic independent from task-specific metric
names or workspace-specific artifact names. They normalize validator-backed
phase reports into a small generic contract that master/reporting code can
consume safely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


ARTIFACT_ROLE_SMOKE_CHECK = "smoke_check"
ARTIFACT_ROLE_PHASE_RESULT = "phase_result"
ARTIFACT_ROLE_FINAL_RESULT = "final_result"

RUN_LEVEL_SMOKE = "smoke"
RUN_LEVEL_FULL = "full"
RUN_LEVEL_MIXED = "mixed"

VALID_PHASE_COMPLETION_STATES = {"not_started", "partial", "complete", "completed", "blocked"}


def _normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if isinstance(item, dict):
                summary = ", ".join(
                    str(v).strip()
                    for v in item.values()
                    if str(v).strip()
                )
                text = f"{key}: {summary}".strip(": ")
            else:
                text = f"{key}: {str(item).strip()}".strip(": ")
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def phase_report_status(payload: Dict[str, Any]) -> str:
    raw = payload.get("status")
    if raw in (None, ""):
        raw = payload.get("verdict")
    if raw in (None, ""):
        raw = payload.get("validation_status")
    if raw in (None, ""):
        raw = payload.get("validation_result")
    if raw in (None, ""):
        validation_summary = payload.get("validation_summary")
        if isinstance(validation_summary, dict):
            raw = validation_summary.get("stage_status")
    status = str(raw or "").strip().upper()
    if status in {"PASS", "PARTIAL", "FAIL"}:
        return status
    if status in {"COMPLETED", "COMPLETE", "SUCCESS"}:
        return "PASS"
    return "UNKNOWN"


def phase_blocking_issues(payload: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for key in (
        "blocking_issues",
        "required_followup",
        "required_fixes",
        "required_fixes_for_complete_evaluation",
        "blocked_metrics",
    ):
        issues.extend(_normalize_text_list(payload.get(key)))
    return issues


def infer_phase_completion_status(payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("phase_completion_status") or "").strip().lower()
    if explicit in VALID_PHASE_COMPLETION_STATES:
        return explicit

    status = phase_report_status(payload)
    blockers = phase_blocking_issues(payload)
    terminal_blocker = bool(payload.get("terminal_blocker"))
    ready_to_proceed = payload.get("ready_to_proceed")
    ready_for_next = payload.get("ready_for_next_phase")

    if status == "FAIL":
        return "blocked" if terminal_blocker else "partial"
    if status == "PARTIAL":
        return "partial"
    if status == "PASS":
        if blockers:
            return "partial"
        if ready_for_next is False:
            return "partial"
        if ready_to_proceed is False:
            return "partial"
        return "complete"
    return "not_started"


def phase_ready_for_next(payload: Dict[str, Any]) -> bool:
    explicit = payload.get("ready_for_next_phase")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(payload.get("ready_to_proceed"), bool):
        return bool(payload.get("ready_to_proceed"))
    completion_status = infer_phase_completion_status(payload)
    return completion_status == "complete"


def phase_artifact_role(
    payload: Dict[str, Any],
    *,
    default_role: str = ARTIFACT_ROLE_PHASE_RESULT,
) -> str:
    role = str(payload.get("artifact_role") or "").strip().lower()
    if role in {
        ARTIFACT_ROLE_SMOKE_CHECK,
        ARTIFACT_ROLE_PHASE_RESULT,
        ARTIFACT_ROLE_FINAL_RESULT,
    }:
        return role
    return default_role


def phase_run_level(
    payload: Dict[str, Any],
    *,
    default_level: str = RUN_LEVEL_FULL,
) -> str:
    level = str(payload.get("run_level") or "").strip().lower()
    if level in {RUN_LEVEL_SMOKE, RUN_LEVEL_FULL, RUN_LEVEL_MIXED}:
        return level
    return default_level


def normalize_phase_report(
    payload: Optional[Dict[str, Any]],
    *,
    default_artifact_role: str = ARTIFACT_ROLE_PHASE_RESULT,
    default_run_level: str = RUN_LEVEL_FULL,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "UNKNOWN",
            "phase_completion_status": "not_started",
            "ready_for_next_phase": False,
            "blocking_issues": [],
            "required_followup": [],
            "artifact_role": default_artifact_role,
            "run_level": default_run_level,
            "terminal_blocker": False,
            "self_contained_project": None,
            "self_contained_violations": [],
            "provenance_manifest_present": None,
            "provenance_manifest_path": "",
        }

    blocking_issues = phase_blocking_issues(payload)
    required_followup = _normalize_text_list(payload.get("required_followup"))
    self_contained_value = payload.get("self_contained_project")
    self_contained_project = (
        bool(self_contained_value)
        if isinstance(self_contained_value, bool)
        else None
    )
    self_contained_violations = _normalize_text_list(payload.get("self_contained_violations"))
    provenance_manifest_present = payload.get("provenance_manifest_present")
    if not isinstance(provenance_manifest_present, bool):
        provenance_manifest_present = None
    provenance_manifest_path = str(payload.get("provenance_manifest_path") or "").strip()
    return {
        "status": phase_report_status(payload),
        "phase_completion_status": infer_phase_completion_status(payload),
        "ready_for_next_phase": phase_ready_for_next(payload),
        "blocking_issues": blocking_issues,
        "required_followup": required_followup,
        "artifact_role": phase_artifact_role(
            payload,
            default_role=default_artifact_role,
        ),
        "run_level": phase_run_level(
            payload,
            default_level=default_run_level,
        ),
        "terminal_blocker": bool(payload.get("terminal_blocker")),
        "self_contained_project": self_contained_project,
        "self_contained_violations": self_contained_violations,
        "provenance_manifest_present": provenance_manifest_present,
        "provenance_manifest_path": provenance_manifest_path,
    }


__all__ = [
    "ARTIFACT_ROLE_FINAL_RESULT",
    "ARTIFACT_ROLE_PHASE_RESULT",
    "ARTIFACT_ROLE_SMOKE_CHECK",
    "RUN_LEVEL_FULL",
    "RUN_LEVEL_MIXED",
    "RUN_LEVEL_SMOKE",
    "infer_phase_completion_status",
    "normalize_phase_report",
    "phase_artifact_role",
    "phase_blocking_issues",
    "phase_ready_for_next",
    "phase_report_status",
    "phase_run_level",
]
