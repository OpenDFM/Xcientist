from __future__ import annotations

import json
import os
import time
import asyncio
import copy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.agents.experiment_agent.runtime.openharness_vendor import ensure_vendored_openharness_path


ensure_vendored_openharness_path()

from openharness.hooks.types import HookResult

from src.agents.experiment_agent.runtime.manifests import write_json_file
from src.agents.experiment_agent.runtime.openharness_runner import (
    extract_json_object,
    validate_json_schema_fragment,
)
from src.agents.experiment_agent.runtime.artifacts import (
    ArtifactLedger,
    ArtifactRegistry,
    artifact_schema_repair_guide,
    build_step_artifact_registry,
    ensure_runtime_report_paths,
    scan_workspace_hygiene,
    validate_artifact_contract,
    write_artifact_registry_snapshot,
)
from src.agents.experiment_agent.runtime.code_review_context import (
    audit_code_scientific_invariants,
    build_code_review_context,
    format_code_invariant_feedback,
)
from src.agents.experiment_agent.runtime.phase_contracts import ARTIFACT_ROLE_PHASE_RESULT
from src.agents.experiment_agent.runtime.project_integrity import (
    audit_code_project_integrity,
    format_project_integrity_feedback,
)
from src.agents.experiment_agent.runtime.prepare_contracts import (
    PREPARE_BLOCKED,
    PREPARE_READY,
    validate_prepare_stage_artifacts,
)
from src.agents.experiment_agent.runtime.contracts import (
    SCIENCE_COMPONENT_RESULT_VALUES,
    validate_science_condition_step_fields,
    validate_science_evidence_payload,
)
from src.agents.experiment_agent.runtime.report_layout import ReportLayout, artifact_rel
from src.agents.experiment_agent.telemetry import (
    Colors,
    format_seconds,
    print_phase,
    print_status,
)


SCIENCE_COMPONENT_RESULT_REVIEWER = "statistical_interpretation"
FINALIZABLE_SCIENCE_COMPONENT_RESULTS = SCIENCE_COMPONENT_RESULT_VALUES
MIN_FINAL_COMPONENT_CONFIDENCE = 0.0
_SOURCE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_GENERIC_CHECKED_ARTIFACTS = {
    "agent_reports/evidence.json",
    "evidence.json",
    "artifact",
    "artifacts",
    "logs",
    "project",
    "results",
}


def worker_output_schema(
    *,
    include_outcome: bool = False,
    require_outcome: bool = False,
) -> Dict[str, Any]:
    """Minimal worker output: what was done, what was produced, what's still blocking.
    The orchestrator synthesises executor/audit metadata from this + step context.
    """
    properties: Dict[str, Any] = {
        "summary": {"type": "string"},
        "artifact_ids_touched": {"type": "array", "items": {"type": "string"}},
        "remaining_blockers": {"type": "array", "items": {"type": "string"}},
    }
    required = [
        "summary",
        "artifact_ids_touched",
        "remaining_blockers",
    ]
    if include_outcome:
        properties["outcome"] = {"type": "string", "enum": [PREPARE_READY, PREPARE_BLOCKED]}
    if require_outcome and "outcome" not in required:
        required.append("outcome")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def worker_output_schema_for_scope(scope: str) -> Dict[str, Any]:
    is_prepare = str(scope or "").strip() == "prepare"
    return worker_output_schema(
        include_outcome=is_prepare,
        require_outcome=is_prepare,
    )


def review_output_schema(*, include_science_condition_fields: bool = False) -> Dict[str, Any]:
    """Unified prefinish review report output."""
    properties: Dict[str, Any] = {
        "reviewer_id": {"type": "string"},
        "reviewer_kind": {"type": "string", "enum": ["deterministic", "agent"]},
        "status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "blocking": {
            "type": "boolean",
            "description": "Must be true. Prefinish reviewers are gating reviewers.",
        },
        "summary": {"type": "string"},
        "checked_artifacts": {"type": "array", "items": {"type": "string"}},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "required_fix": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code", "message", "required_fix", "evidence"],
            },
        },
        "structured_findings": {"type": "object"},
    }
    required = [
        "reviewer_id",
        "reviewer_kind",
        "status",
        "blocking",
        "summary",
        "checked_artifacts",
        "issues",
        "structured_findings",
    ]
    if include_science_condition_fields:
        properties["structured_findings"] = {
            "type": "object",
            "properties": {
                "condition_id": {"type": "string"},
                "enabled_components": {"type": "array", "items": {"type": "string"}},
                "disabled_components": {"type": "array", "items": {"type": "string"}},
                "reference_condition_id": {"type": ["string", "null"]},
                "component_result": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral", "inconclusive"],
                        },
                        "metric": {"type": "string"},
                        "value": {"type": "string"},
                        "confidence": {"type": "number"},
                        "analysis": {"type": "string"},
                        "method_context": {"type": "string"},
                        "follow_up_required": {"type": "boolean"},
                    },
                },
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def planner_output_schema(*, step_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "stages": {"type": "array", "items": step_schema},
            "summary": {"type": "string"},
            "usage_notes": {"type": "string"},
        },
        "required": ["stages", "summary", "usage_notes"],
    }


def planner_artifact_prefinish_gate(
    *,
    output_schema: Dict[str, Any],
    registry: ArtifactRegistry,
    plan_artifact_id: str,
) -> Callable[[Dict[str, Any]], Any]:
    """Build a planner STOP hook that enforces structured output and managed plan artifact use."""
    schema_text = json.dumps(output_schema, ensure_ascii=False, indent=2)

    async def _gate(stop_payload: Dict[str, Any]) -> HookResult:
        text = str(stop_payload.get("assistant_text") or "").strip()
        try:
            payload = extract_json_object(text)
        except Exception as exc:
            return HookResult(
                hook_type="xcientist_planner_prefinish_gate",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist planner prefinish hook blocked completion because the final "
                    "response is not exactly one JSON object.\n\n"
                    f"Error: {exc}\n\n"
                    "Expected planner final response schema:\n"
                    "```json\n"
                    f"{schema_text}\n"
                    "```"
                ),
            )

        schema_issues = validate_json_schema_fragment(payload, output_schema)
        if schema_issues:
            return HookResult(
                hook_type="xcientist_planner_prefinish_gate",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist planner prefinish hook blocked completion because the final "
                    "JSON does not satisfy the required schema.\n\n"
                    "Schema issues:\n"
                    + "\n".join(f"- {issue}" for issue in schema_issues)
                    + "\n\nExpected planner final response schema:\n"
                    "```json\n"
                    f"{schema_text}\n"
                    "```"
                ),
                metadata={"schema_issues": schema_issues},
            )

        spec = registry.get(plan_artifact_id)
        contract = validate_artifact_contract(registry=registry, review_status="PASS")
        contract_issues = [str(issue) for issue in contract.get("issues") or [] if str(issue).strip()]
        artifact_payload: Dict[str, Any] | None = None
        if spec is not None:
            path = spec.resolved_path(registry.workspace_root)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    artifact_payload = loaded
                else:
                    contract_issues.append(f"`{plan_artifact_id}` must be a JSON object at `{spec.path}`.")
            except Exception as exc:
                contract_issues.append(f"Could not read `{plan_artifact_id}` at `{spec.path}`: {exc}")
        else:
            contract_issues.append(f"Unknown planner artifact id `{plan_artifact_id}`.")

        if artifact_payload is not None:
            for field in ("stages", "summary", "usage_notes"):
                if artifact_payload.get(field) != payload.get(field):
                    contract_issues.append(
                        f"Final planner response `{field}` must exactly match the managed plan artifact "
                        f"`{plan_artifact_id}`. Repair by returning the same `{field}` written with "
                        "`write_artifact`, or update the managed artifact with artifact tools before finishing."
                    )

        if contract_issues:
            schema_guide = ""
            if spec is not None:
                schema_guide = artifact_schema_repair_guide(
                    spec.schema_name,
                    artifact_id=spec.artifact_id,
                    path=spec.path,
                )
            return HookResult(
                hook_type="xcientist_planner_prefinish_gate",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist planner prefinish hook blocked completion because the managed "
                    "planner artifact contract is not satisfied. Fix this in the same planner "
                    "session, using artifact tools only, then finish again.\n\n"
                    f"Required planner artifact: `{plan_artifact_id}`\n"
                    f"Artifact ledger: `{contract.get('ledger_path') or ''}`\n\n"
                    "Issues:\n"
                    + "\n".join(f"- {issue}" for issue in contract_issues)
                    + (
                        "\n\nExpected managed artifact schema / repair template:\n"
                        + schema_guide
                        if schema_guide
                        else ""
                    )
                ),
                metadata={"contract": contract, "contract_issues": contract_issues},
            )
        return HookResult(
            hook_type="xcientist_planner_prefinish_gate",
            success=True,
            blocked=False,
            metadata={"contract": contract},
        )

    return _gate


def with_phase_defaults(payload: Dict[str, Any], *, scope: str) -> Dict[str, Any]:
    merged = dict(payload)
    merged.setdefault("scope", scope)
    merged.setdefault("checked_artifacts", [])
    merged.setdefault("findings", [])
    merged.setdefault("evidence_summary", "")
    merged.setdefault("phase_completion_status", "partial")
    merged.setdefault("ready_for_next_phase", False)
    merged.setdefault("blocking_issues", [])
    merged.setdefault("required_followup", [])
    merged.setdefault("artifact_role", ARTIFACT_ROLE_PHASE_RESULT)
    merged.setdefault("run_level", "full")
    merged.setdefault("self_contained_project", True)
    merged.setdefault("self_contained_violations", [])
    merged.setdefault("artifact_ledger_present", False)
    merged.setdefault("artifact_ledger_path", "")
    merged.setdefault("terminal_blocker", False)
    merged.setdefault("next_worker_input", "")
    merged.setdefault("review_scope", [])
    return merged


def _append_path_value(target: List[str], value: Any) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            target.append(text)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_path_value(target, item)


def _collect_checked_artifacts_from_report(target: List[str], report: Any) -> None:
    if not isinstance(report, dict):
        return
    _append_path_value(target, report.get("checked_artifacts"))
    for key in (
        "worker_report_path",
        "worker_attempt_report_path",
        "review_report_path",
        "review_attempt_report_path",
        "review_aggregate_report_path",
        "hook_report_path",
        "hook_attempt_report_path",
    ):
        _append_path_value(target, report.get(key))
    review_matrix = report.get("review_matrix")
    if isinstance(review_matrix, dict):
        _append_path_value(target, review_matrix.get("checked_artifacts"))
        for child in review_matrix.get("reports") or []:
            _collect_checked_artifacts_from_report(target, child)


def phase_checked_artifacts(step_result: Dict[str, Any], *extra_paths: str) -> List[str]:
    """Collect meaningful phase-level evidence from step reports and gate outputs."""
    checked: List[str] = []
    _append_path_value(checked, list(extra_paths))
    if isinstance(step_result, dict):
        for report in step_result.get("step_reports") or []:
            _collect_checked_artifacts_from_report(checked, report)
        _collect_checked_artifacts_from_report(checked, step_result.get("failed_review_report"))
        _collect_checked_artifacts_from_report(checked, step_result.get("blocked_review_report"))
    return sorted(dict.fromkeys(item for item in checked if item))


def _step_id_from_report(report: Any) -> str:
    if not isinstance(report, dict):
        return ""
    for key in ("step_id", "condition_id", "stage_id"):
        value = str(report.get(key) or "").strip()
        if value:
            return value
    matrix = report.get("review_matrix")
    if isinstance(matrix, dict):
        findings = matrix.get("structured_findings")
        if isinstance(findings, dict):
            value = str(findings.get("step_id") or "").strip()
            if value:
                return value
    return ""


def _load_json_object(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _completed_step_report_from_workspace(
    *,
    layout: ReportLayout,
    scope: str,
    step_id: str,
) -> Dict[str, Any] | None:
    hook_latest, _ = layout.latest_and_attempt(scope, step_id, "hook", 1)
    hook_payload = _load_json_object(hook_latest)
    if not isinstance(hook_payload, dict):
        return None
    contract_payload = hook_payload.get("prefinish_contract")
    if not isinstance(contract_payload, dict):
        return None
    if (
        hook_payload.get("status") != "PASS"
        or hook_payload.get("review_status") != "PASS"
        or contract_payload.get("status") != "PASS"
        or hook_payload.get("returned_to_worker") is True
    ):
        return None

    review_path = str(hook_payload.get("review_report_path") or "").strip()
    if not review_path:
        review_path, _ = layout.latest_and_attempt(scope, step_id, "review", 1)
    if not os.path.isabs(review_path):
        review_path = os.path.join(layout.workspace_root, review_path)
    review_payload = _load_json_object(review_path)
    if not isinstance(review_payload, dict) or review_payload.get("status") != "PASS":
        return None

    review_payload = copy.deepcopy(review_payload)
    review_payload.setdefault("scope", scope)
    review_payload.setdefault("step_id", step_id)
    review_payload.setdefault("review_report_path", review_path)
    review_payload.setdefault("review_attempt_report_path", hook_payload.get("review_attempt_report_path", ""))
    review_payload.setdefault("review_aggregate_report_path", hook_payload.get("review_aggregate_report_path", ""))
    review_payload.setdefault("worker_report_path", hook_payload.get("worker_report_path", ""))
    review_payload.setdefault("worker_attempt_report_path", hook_payload.get("worker_attempt_report_path", ""))
    review_payload["prefinish_contract"] = contract_payload
    review_payload["hook_report_path"] = hook_latest
    review_payload["hook_attempt_report_path"] = hook_payload.get("hook_attempt_report_path", "")
    if "review_matrix" not in review_payload and isinstance(hook_payload.get("review_matrix"), dict):
        review_payload["review_matrix"] = hook_payload["review_matrix"]
    return review_payload


def phase_step_ids(step_result: Dict[str, Any]) -> List[str]:
    """Return completed step ids from phase step reports without fabricating names."""
    if not isinstance(step_result, dict):
        return []
    ids = [_step_id_from_report(report) for report in step_result.get("step_reports") or []]
    return [item for item in ids if item]


def _review_issue(
    *,
    code: str,
    message: str,
    required_fix: str = "",
    evidence: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return {
        "code": str(code or "review_issue"),
        "message": str(message or "").strip(),
        "required_fix": str(required_fix or message or "").strip(),
        "evidence": [str(item) for item in (evidence or []) if str(item).strip()],
    }


def _review_report(
    *,
    reviewer_id: str,
    reviewer_kind: str,
    status: str,
    blocking: bool,
    summary: str,
    checked_artifacts: Sequence[str] | None = None,
    issues: Sequence[Dict[str, Any]] | None = None,
    structured_findings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_issues = []
    for issue in issues or []:
        if not isinstance(issue, dict):
            normalized_issues.append(
                _review_issue(
                    code="review_issue",
                    message=str(issue),
                    required_fix=str(issue),
                )
            )
            continue
        normalized_issues.append(
            _review_issue(
                code=str(issue.get("code") or "review_issue"),
                message=str(issue.get("message") or ""),
                required_fix=str(issue.get("required_fix") or issue.get("message") or ""),
                evidence=list(issue.get("evidence") or []),
            )
        )
    verdict = "PASS" if str(status).upper() == "PASS" and not normalized_issues else "FAIL"
    return {
        "reviewer_id": str(reviewer_id or "reviewer"),
        "reviewer_kind": "agent" if reviewer_kind == "agent" else "deterministic",
        "status": verdict,
        "blocking": bool(blocking),
        "summary": str(summary or "").strip(),
        "checked_artifacts": [str(item) for item in (checked_artifacts or []) if str(item).strip()],
        "issues": normalized_issues,
        "structured_findings": structured_findings or {},
    }


def _structured_hook_issues_to_review_issues(
    payload: Dict[str, Any],
    *,
    default_code: str,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for item in payload.get("issues") or []:
        if isinstance(item, dict):
            evidence = []
            if item.get("path"):
                evidence.append(str(item.get("path")))
            if item.get("line") is not None:
                evidence.append(f"line {item.get('line')}")
            issues.append(
                _review_issue(
                    code=str(item.get("rule") or item.get("code") or default_code),
                    message=str(item.get("message") or item),
                    required_fix=str(item.get("fix") or item.get("required_fix") or item.get("message") or item),
                    evidence=evidence,
                )
            )
            continue
        issues.append(
            _review_issue(
                code=default_code,
                message=str(item),
                required_fix=str(item),
            )
        )
    return issues


def _review_schema_repair_text(*, scope: str) -> str:
    schema = review_output_schema(include_science_condition_fields=(scope == "science"))
    return (
        "Return exactly the unified prefinish review JSON schema below. "
        "All repair instructions must be expressed inside `issues[].required_fix`; "
        "`status` must be either `PASS` or `FAIL`.\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _worker_schema_repair_text(*, scope: str) -> str:
    schema = worker_output_schema_for_scope(scope)
    return (
        "Return exactly one worker final JSON object and no markdown fences or prose. "
        "The object must satisfy this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _schema_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _validate_worker_payload_schema(payload: Dict[str, Any], *, scope: str) -> List[str]:
    schema = worker_output_schema_for_scope(scope)
    required = set(schema.get("required") or [])
    allowed = set((schema.get("properties") or {}).keys())
    issues: List[str] = []
    for field in sorted(required):
        if field not in payload:
            issues.append(f"Missing required worker field `{field}`.")
    extras = sorted(str(field) for field in payload if field not in allowed)
    if extras:
        issues.append("Worker final JSON contains unsupported fields: " + ", ".join(extras))
    if "summary" in payload and not isinstance(payload.get("summary"), str):
        issues.append(f"`summary` must be a string, got {_schema_type_name(payload.get('summary'))}.")
    if "artifact_ids_touched" in payload:
        value = payload.get("artifact_ids_touched")
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            issues.append("`artifact_ids_touched` must be an array of strings.")
    if "remaining_blockers" in payload:
        value = payload.get("remaining_blockers")
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            issues.append("`remaining_blockers` must be an array of strings.")
    if "outcome" in allowed:
        outcome = payload.get("outcome")
        if outcome not in {PREPARE_READY, PREPARE_BLOCKED}:
            issues.append(f"`outcome` must be `{PREPARE_READY}` or `{PREPARE_BLOCKED}`.")
    return issues


def _validate_agent_review_schema(payload: Dict[str, Any], *, scope: str) -> List[str]:
    schema = review_output_schema(include_science_condition_fields=(scope == "science"))
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    allowed = set(properties.keys())
    issues: List[str] = []
    for field in sorted(required):
        if field not in payload:
            issues.append(f"Missing required review field `{field}`.")
    extras = sorted(str(field) for field in payload if field not in allowed)
    if extras:
        issues.append("Review JSON contains unsupported fields: " + ", ".join(extras))
    if "reviewer_id" in payload and not isinstance(payload.get("reviewer_id"), str):
        issues.append(f"`reviewer_id` must be a string, got {_schema_type_name(payload.get('reviewer_id'))}.")
    if payload.get("reviewer_kind") != "agent":
        issues.append("Agent reviewer JSON must set `reviewer_kind` to `agent`.")
    if payload.get("status") not in {"PASS", "FAIL"}:
        issues.append("Review `status` must be `PASS` or `FAIL`.")
    if "blocking" in payload:
        if not isinstance(payload.get("blocking"), bool):
            issues.append(f"`blocking` must be a boolean, got {_schema_type_name(payload.get('blocking'))}.")
        elif payload.get("blocking") is not True:
            issues.append("`blocking` must be true; prefinish agent reviewers cannot mark findings non-blocking.")
    if "summary" in payload and not isinstance(payload.get("summary"), str):
        issues.append(f"`summary` must be a string, got {_schema_type_name(payload.get('summary'))}.")
    checked_artifacts = payload.get("checked_artifacts")
    if "checked_artifacts" in payload and (
        not isinstance(checked_artifacts, list)
        or not all(isinstance(item, str) for item in checked_artifacts)
    ):
        issues.append("`checked_artifacts` must be an array of strings.")
    review_issues = payload.get("issues")
    if "issues" in payload:
        if not isinstance(review_issues, list):
            issues.append("`issues` must be an array of issue objects.")
        else:
            issue_allowed = {"code", "message", "required_fix", "evidence"}
            for index, issue in enumerate(review_issues):
                if not isinstance(issue, dict):
                    issues.append(f"`issues[{index}]` must be an object.")
                    continue
                missing = sorted(issue_allowed - set(issue))
                if missing:
                    issues.append(f"`issues[{index}]` is missing fields: {', '.join(missing)}.")
                extra = sorted(str(field) for field in issue if field not in issue_allowed)
                if extra:
                    issues.append(f"`issues[{index}]` contains unsupported fields: {', '.join(extra)}.")
                for field in ("code", "message", "required_fix"):
                    if field in issue and not isinstance(issue.get(field), str):
                        issues.append(f"`issues[{index}].{field}` must be a string.")
                evidence = issue.get("evidence")
                if "evidence" in issue and (
                    not isinstance(evidence, list)
                    or not all(isinstance(item, str) for item in evidence)
                ):
                    issues.append(f"`issues[{index}].evidence` must be an array of strings.")
    if "structured_findings" in payload and not isinstance(payload.get("structured_findings"), dict):
        issues.append("`structured_findings` must be an object.")
    return issues


def _normalize_agent_review_report(
    payload: Dict[str, Any],
    *,
    reviewer_id: str,
    scope: str,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        repair = _review_schema_repair_text(scope=scope)
        return _review_report(
            reviewer_id=reviewer_id,
            reviewer_kind="agent",
            status="FAIL",
            blocking=True,
            summary="Reviewer returned a non-object payload.",
            issues=[
                _review_issue(
                    code="review_schema",
                    message="Reviewer output must be a JSON object.",
                    required_fix=repair,
                )
            ],
        )
    schema_issues = _validate_agent_review_schema(payload, scope=scope)
    if schema_issues:
        repair = _review_schema_repair_text(scope=scope)
        return _review_report(
            reviewer_id=reviewer_id,
            reviewer_kind="agent",
            status="FAIL",
            blocking=True,
            summary="Reviewer did not return the unified review report schema.",
            checked_artifacts=list(payload.get("checked_artifacts") or []) if isinstance(payload.get("checked_artifacts"), list) else [],
            issues=[
                _review_issue(
                    code="review_schema",
                    message="Reviewer output does not satisfy the unified schema: " + "; ".join(schema_issues),
                    required_fix=repair,
                    evidence=schema_issues,
                )
            ],
            structured_findings={},
        )
    return _review_report(
        reviewer_id=str(payload.get("reviewer_id") or reviewer_id),
        reviewer_kind="agent",
        status=str(payload.get("status") or "FAIL"),
        blocking=True,
        summary=str(payload.get("summary") or ""),
        checked_artifacts=list(payload.get("checked_artifacts") or []),
        issues=list(payload.get("issues") or []),
        structured_findings=dict(payload.get("structured_findings") or {}),
    )


def _is_external_checked_artifact(value: str) -> bool:
    return bool(
        re.match(r"^https?://[^\s]+$", value)
        or re.match(r"^(doi|arxiv):[^\s]+$", value, re.IGNORECASE)
        or re.match(r"^git\+https://[^\s]+$", value, re.IGNORECASE)
    )


def _checked_artifact_path_issue(
    *,
    workspace_root: str,
    reviewer_id: str,
    index: int,
    value: str,
) -> str:
    text = str(value or "").strip()
    if not text:
        return f"Reviewer `{reviewer_id}` checked_artifacts[{index}] is empty."
    if text in _GENERIC_CHECKED_ARTIFACTS:
        return (
            f"Reviewer `{reviewer_id}` checked_artifacts[{index}] is too generic: `{text}`. "
            "List concrete existing files/directories or explicit external URLs actually inspected."
        )
    if _is_external_checked_artifact(text):
        return ""
    candidate = text if os.path.isabs(text) else os.path.join(workspace_root, text)
    candidate = os.path.realpath(candidate)
    if not (_path_under(workspace_root, candidate) or _path_under(_SOURCE_ROOT, candidate)):
        return (
            f"Reviewer `{reviewer_id}` checked_artifacts[{index}] points outside the active workspace/source tree: `{text}`."
        )
    if not os.path.exists(candidate):
        return (
            f"Reviewer `{reviewer_id}` checked_artifacts[{index}] does not exist: `{text}`. "
            "Repair the review by inspecting and listing concrete existing evidence paths."
        )
    return ""


def _run_reviewer_checked_artifacts_hook(
    *,
    scope: str,
    step_id: str,
    workspace_root: str,
    reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any] | None:
    agent_reports = [
        report
        for report in reports
        if isinstance(report, dict) and report.get("reviewer_kind") == "agent"
    ]
    if not agent_reports:
        return None
    issues: List[str] = []
    checked: List[str] = []
    for report in agent_reports:
        reviewer_id = str(report.get("reviewer_id") or "reviewer")
        artifacts = report.get("checked_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            issues.append(
                f"Reviewer `{reviewer_id}` must declare non-empty `checked_artifacts` with concrete files/directories or external URLs."
            )
            continue
        for index, raw in enumerate(artifacts):
            text = str(raw or "").strip()
            issue = _checked_artifact_path_issue(
                workspace_root=workspace_root,
                reviewer_id=reviewer_id,
                index=index,
                value=text,
            )
            if issue:
                issues.append(issue)
            else:
                checked.append(text)
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "reviewer_checked_artifacts",
        "scope": scope,
        "step_id": step_id,
        "issues": issues,
        "checked_artifacts": sorted(dict.fromkeys(checked)),
    }


def _aggregate_review_reports(
    *,
    scope: str,
    step_id: str,
    attempt: int,
    reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    failed_reports = [report for report in reports if report.get("status") != "PASS"]
    issues = []
    for report in reports:
        for issue in report.get("issues") or []:
            if isinstance(issue, dict):
                issues.append(
                    {
                        **issue,
                        "reviewer_id": report.get("reviewer_id"),
                    }
                )
    status = "PASS" if not failed_reports else "FAIL"
    checked_artifacts: List[str] = []
    for report in reports:
        checked_artifacts.extend(str(item) for item in report.get("checked_artifacts") or [])
    return {
        "reviewer_id": "aggregate",
        "reviewer_kind": "deterministic",
        "status": status,
        "blocking": True,
        "summary": (
            f"{len(reports)} reviewer(s) passed."
            if status == "PASS"
            else f"{len(failed_reports)} reviewer(s) failed."
        ),
        "checked_artifacts": sorted(set(item for item in checked_artifacts if item)),
        "issues": issues,
        "structured_findings": {
            "scope": scope,
            "step_id": step_id,
            "attempt": int(attempt),
            "reviewer_statuses": {
                str(report.get("reviewer_id") or "reviewer"): str(report.get("status") or "FAIL")
                for report in reports
            },
        },
        "reports": list(reports),
    }


def _unified_to_phase_review_payload(
    *,
    aggregate: Dict[str, Any],
    scope: str,
    science_findings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    issues = [issue for issue in aggregate.get("issues") or [] if isinstance(issue, dict)]
    repair_instructions = [
        str(issue.get("required_fix") or issue.get("message") or "")
        for issue in issues
        if str(issue.get("required_fix") or issue.get("message") or "").strip()
    ]
    payload = with_phase_defaults(
        {
            "status": aggregate.get("status"),
            "evidence_summary": str(aggregate.get("summary") or ""),
            "terminal_blocker": False,
            "next_worker_input": "\n".join(repair_instructions),
            "checked_artifacts": list(aggregate.get("checked_artifacts") or []),
            "review_scope": [
                str(item)
                for item in (aggregate.get("structured_findings") or {}).get("reviewer_statuses", {})
            ],
            "review_matrix": aggregate,
            "blocking_issues": [
                str(issue.get("message") or "")
                for issue in issues
                if str(issue.get("message") or "").strip()
            ],
        },
        scope=scope,
    )
    if science_findings:
        component_result = dict(science_findings.get("component_result") or {})
        for field in (
            "result",
            "metric",
            "value",
            "confidence",
            "analysis",
            "method_context",
            "follow_up_required",
        ):
            component_result.setdefault(field, "")
        payload.update(
            {
                "condition_id": science_findings.get("condition_id"),
                "enabled_components": science_findings.get("enabled_components") or [],
                "disabled_components": science_findings.get("disabled_components") or [],
                "reference_condition_id": science_findings.get("reference_condition_id"),
                **component_result,
            }
        )
    return payload


def _path_under(root: str, candidate: str) -> bool:
    if not candidate:
        return False
    root_real = os.path.realpath(root)
    candidate_real = os.path.realpath(candidate)
    return candidate_real == root_real or candidate_real.startswith(root_real + os.sep)


def _resolve_workspace_path(workspace_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(workspace_root, path)


def _prepare_worker_reports_blocked(
    *,
    scope: str,
    worker_payload: Dict[str, Any],
) -> bool:
    return (
        scope == "prepare"
        and str(worker_payload.get("outcome") or "").strip().upper() == PREPARE_BLOCKED
    )


def _prepare_blockers_are_allowed(
    *,
    scope: str,
    worker_payload: Dict[str, Any],
) -> bool:
    blockers = worker_payload.get("remaining_blockers")
    return (
        _prepare_worker_reports_blocked(scope=scope, worker_payload=worker_payload)
        and isinstance(blockers, list)
        and any(str(item).strip() for item in blockers)
    )


def _worker_blocker_texts(worker_payload: Dict[str, Any]) -> List[str]:
    blockers = worker_payload.get("remaining_blockers")
    if not isinstance(blockers, list):
        return []
    return [str(item).strip() for item in blockers if str(item).strip()]


def _required_output_paths(step: Dict[str, Any], workspace_root: str) -> List[str]:
    paths: List[str] = []
    for field in ("worker_report_path", "review_report_path", "hook_report_path"):
        value = step.get(field)
        if isinstance(value, list):
            paths.extend(str(item) for item in value if str(item).strip())
        elif value:
            paths.append(str(value))
    return [
        _resolve_workspace_path(workspace_root, path)
        for path in paths
    ]


def _run_step_contract_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
    worker_payload: Dict[str, Any],
    review_payload: Dict[str, Any],
    attempt: int = 1,
) -> Dict[str, Any]:
    """Deterministic prefinish contract for a single worker/reviewer step."""
    issues: List[str] = []
    registry = build_step_artifact_registry(
        workspace_root=workspace_root,
        scope=scope,
        step=step,
    )
    artifact_payload = validate_artifact_contract(
        registry=registry,
        review_status=str(review_payload.get("status") or ""),
    )
    if artifact_payload["status"] != "PASS":
        issues.extend(list(artifact_payload.get("issues") or []))
    hygiene_payload = scan_workspace_hygiene(workspace_root)
    if hygiene_payload["status"] != "PASS":
        for issue in hygiene_payload.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            reason = str(issue.get("reason") or "").strip()
            path = str(issue.get("path") or "").strip()
            issues.append(f"Workspace hygiene issue at `{path}`: {reason}".strip())
    for path in _required_output_paths(step, workspace_root):
        if not _path_under(workspace_root, path):
            issues.append(f"Output path escapes workspace: {path}")
            continue
        parent = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    blocker_texts = _worker_blocker_texts(worker_payload)
    if blocker_texts and not _prepare_blockers_are_allowed(scope=scope, worker_payload=worker_payload):
        if scope == "prepare":
            issues.append(
                "Worker reported `remaining_blockers` but did not use the accepted prepare BLOCKED shape. "
                "Repair the final worker JSON to set `outcome: \"BLOCKED\"`, keep `remaining_blockers` non-empty, "
                "and make the managed prepare artifact status `BLOCKED` with a complete `blocker` object. "
                "Current blockers: " + "; ".join(blocker_texts)
            )
        else:
            issues.append(
                "Worker final JSON for this scope must have `remaining_blockers: []`. "
                "Fix the work in the same session instead of finishing with blockers. "
                "Current blockers: " + "; ".join(blocker_texts)
            )
    if _prepare_worker_reports_blocked(scope=scope, worker_payload=worker_payload) and not blocker_texts:
        issues.append(
            "Prepare worker final JSON has `outcome: \"BLOCKED\"` but `remaining_blockers` is empty. "
            "Add concrete blockers matching the managed BLOCKED artifact's `blocker.missing_requirements` "
            "and `blocker.user_action_required`, or change outcome/artifact status to READY after real repair."
        )

    artifact_ids_touched = worker_payload.get("artifact_ids_touched")
    expected_artifact_ids = set(step.get("artifact_ids") or [])
    expected_text = ", ".join(sorted(expected_artifact_ids)) or "(no worker-owned artifacts expected)"
    if not isinstance(artifact_ids_touched, list):
        issues.append(
            "Worker final JSON must declare `artifact_ids_touched` as a JSON array of managed artifact ids, "
            f"not file paths. Allowed ids for this step: {expected_text}."
        )
    else:
        touched_ids = {str(item).strip() for item in artifact_ids_touched if str(item).strip()}
        unknown_ids = [
            str(item)
            for item in artifact_ids_touched
            if str(item) and str(item) not in expected_artifact_ids
        ]
        if unknown_ids:
            issues.append(
                "Worker reported unknown artifact ids: "
                + ", ".join(sorted(unknown_ids))
                + f". Allowed ids for this step: {expected_text}. "
                "Use artifact ids from the Artifact Registry, not paths or runtime report names."
            )
        missing_ids = sorted(expected_artifact_ids - touched_ids)
        if missing_ids:
            issues.append(
                "Worker final JSON `artifact_ids_touched` must include every required managed artifact id "
                "written or updated for this step. Missing: "
                + ", ".join(missing_ids)
                + f". Allowed ids for this step: {expected_text}. "
                "After repairing the artifact through artifact tools, finish again with these ids listed."
            )

    if review_payload.get("status") != "PASS":
        issues.append(f"Reviewer status is {review_payload.get('status') or 'UNKNOWN'}, not PASS.")

    review_scope = review_payload.get("review_scope")
    if not isinstance(review_scope, list) or not review_scope:
        issues.append(
            "Reviewer aggregate must expose a non-empty `review_scope` reviewer-id list. "
            "This is normally synthesized from the parallel reviewer matrix; if it is empty, "
            "ensure at least one reviewer returned the unified review JSON schema with `reviewer_id`."
        )

    checked_artifacts = review_payload.get("checked_artifacts")
    if not isinstance(checked_artifacts, list) or not checked_artifacts:
        issues.append("Reviewer must declare non-empty `checked_artifacts`.")

    missing_contract_fields = [
        field
        for field in (
            "worker_report_path",
            "review_report_path",
            "hook_report_path",
        )
        if field not in step
    ]
    if missing_contract_fields:
        issues.append("Step contract missing required runtime fields: " + ", ".join(missing_contract_fields))

    payload = {
        "status": "PASS" if not issues else "FAIL",
        "hook": "step_prefinish_contract",
        "scope": scope,
        "step_id": step.get("condition_id") or step.get("step_id") or step.get("stage_id"),
        "attempt": int(attempt),
        "issues": issues,
        "contract_path": "",
        "checked_outputs": _required_output_paths(step, workspace_root),
        "artifact_contract": artifact_payload,
        "workspace_hygiene": hygiene_payload,
    }
    ArtifactLedger(workspace_root).append(
        {
            "event": "prefinish_contract",
            "artifact_id": f"runtime.{scope}.{payload['step_id']}.prefinish_contract",
            "stage": scope,
            "step_id": payload["step_id"],
            "attempt": int(attempt),
            "status": payload["status"],
            "review_status": review_payload.get("status"),
            "issues": issues,
            "artifact_contract": artifact_payload,
            "workspace_hygiene": hygiene_payload,
        }
    )
    return payload


def _run_worker_completion_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    worker_payload: Dict[str, Any],
) -> Dict[str, Any]:
    issues: List[str] = []
    blocker_texts = _worker_blocker_texts(worker_payload)
    if blocker_texts and not _prepare_blockers_are_allowed(scope=scope, worker_payload=worker_payload):
        if scope == "prepare":
            issues.append(
                "Worker reported `remaining_blockers` but did not use the accepted prepare BLOCKED shape. "
                "Repair the final worker JSON to set `outcome: \"BLOCKED\"`, keep `remaining_blockers` non-empty, "
                "and make the managed prepare artifact status `BLOCKED` with a complete `blocker` object. "
                "Current blockers: " + "; ".join(blocker_texts)
            )
        else:
            issues.append(
                "Worker final JSON for this scope must have `remaining_blockers: []`. "
                "Fix the work in the same session instead of finishing with blockers. "
                "Current blockers: " + "; ".join(blocker_texts)
            )
    if _prepare_worker_reports_blocked(scope=scope, worker_payload=worker_payload) and not blocker_texts:
        issues.append(
            "Prepare worker final JSON has `outcome: \"BLOCKED\"` but `remaining_blockers` is empty. "
            "Add concrete blockers matching the managed BLOCKED artifact's `blocker.missing_requirements` "
            "and `blocker.user_action_required`, or change outcome/artifact status to READY after real repair."
        )
    artifact_ids_touched = worker_payload.get("artifact_ids_touched")
    expected_artifact_ids = set(step.get("artifact_ids") or [])
    if not isinstance(artifact_ids_touched, list):
        issues.append(
            "Worker final JSON must declare `artifact_ids_touched` as a JSON array of managed artifact ids, "
            f"not file paths. Allowed ids for this step: {', '.join(sorted(expected_artifact_ids)) or '(none)'}."
        )
    else:
        touched_ids = {str(item).strip() for item in artifact_ids_touched if str(item).strip()}
        unknown_ids = [
            str(item)
            for item in artifact_ids_touched
            if str(item) and str(item) not in expected_artifact_ids
        ]
        if unknown_ids:
            issues.append(
                "Worker reported unknown artifact ids: "
                + ", ".join(sorted(unknown_ids))
                + f". Allowed ids for this step: {', '.join(sorted(expected_artifact_ids)) or '(none)'}. "
                "Use artifact ids from the Artifact Registry, not paths or runtime report names."
            )
        missing_ids = sorted(expected_artifact_ids - touched_ids)
        if missing_ids:
            issues.append(
                "Worker final JSON `artifact_ids_touched` must include every required managed artifact id "
                "written or updated for this step. Missing: "
                + ", ".join(missing_ids)
                + f". Allowed ids for this step: {', '.join(sorted(expected_artifact_ids)) or '(none)'}. "
                "After repairing the artifact through artifact tools, finish again with these ids listed."
            )
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "worker_completion_contract",
        "scope": scope,
        "step_id": step.get("condition_id") or step.get("step_id") or step.get("stage_id"),
        "issues": issues,
        "expected_artifact_ids": sorted(expected_artifact_ids),
        "artifact_ids_touched": artifact_ids_touched if isinstance(artifact_ids_touched, list) else [],
    }


def _run_artifact_contract_review_hook(
    *,
    registry: ArtifactRegistry,
    worker_paths: Dict[str, str],
) -> Dict[str, Any]:
    payload = validate_artifact_contract(
        registry=registry,
        review_status="PASS",
    )
    return {
        "status": payload.get("status"),
        "hook": "artifact_contract",
        "issues": payload.get("issues") or [],
        "checked_artifacts": payload.get("checked_artifacts") or [],
        "ledger_path": payload.get("ledger_path"),
        "registry_path": payload.get("registry_path"),
        "worker_report_path": worker_paths.get("latest", ""),
    }


def _run_code_project_integrity_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
    plan_steps: Sequence[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if scope != "code":
        return None
    project_root = os.path.join(workspace_root, "project")
    if not os.path.exists(project_root):
        return None
    return audit_code_project_integrity(
        workspace_root=workspace_root,
        project_root=project_root,
        plan_steps=plan_steps,
    )


def _normalize_recorded_source(workspace_root: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" in text or text.startswith(("doi:", "arxiv:", "git@")):
        return text
    return os.path.realpath(_resolve_workspace_path(workspace_root, text))


def _run_code_source_provenance_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
) -> Dict[str, Any] | None:
    if scope != "code":
        return None
    if str(step.get("repo_copy_intent") or "").strip() != "copy_and_modify":
        return None
    expected_artifact_ids = {
        str(item).strip()
        for item in step.get("artifact_ids") or []
        if str(item).strip()
    }
    expected_sources = [
        str(item).strip()
        for item in step.get("repo_source_paths") or []
        if str(item).strip()
    ]
    issues: List[str] = []
    if not expected_artifact_ids:
        issues.append(
            "`repo_copy_intent=copy_and_modify` requires managed artifact ids so source provenance can be attached."
        )
    if not expected_sources:
        issues.append(
            "`repo_copy_intent=copy_and_modify` requires non-empty `repo_source_paths`."
        )

    expected_source_keys = {
        _normalize_recorded_source(workspace_root, source)
        for source in expected_sources
    }
    expected_source_keys.discard("")
    records = [
        record
        for record in ArtifactLedger(workspace_root).read()
        if record.get("event") == "record_sources"
        and str(record.get("artifact_id") or "") in expected_artifact_ids
    ]
    recorded_sources = {
        _normalize_recorded_source(workspace_root, source)
        for record in records
        for source in (record.get("sources") or [])
    }
    recorded_sources.discard("")
    missing_sources = sorted(expected_source_keys - recorded_sources)
    if not records:
        issues.append(
            "Copied repo code has no `record_sources` ledger entry. Call `record_sources` "
            "for the managed handoff artifact after copying, listing every copied `repo_source_paths` entry."
        )
    elif missing_sources:
        issues.append(
            "`record_sources` does not cover every copied repo source path. Missing: "
            + ", ".join(missing_sources)
        )
    if any(not str(record.get("reason") or "").strip() for record in records):
        issues.append("Each `record_sources` ledger entry must include a concrete reason for the copy.")

    ledger_path = ArtifactLedger(workspace_root).path
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "code_source_provenance",
        "scope": scope,
        "step_id": step.get("step_id"),
        "issues": issues,
        "checked_artifacts": [ledger_path, *expected_sources],
        "expected_artifact_ids": sorted(expected_artifact_ids),
        "expected_sources": expected_sources,
        "recorded_sources": sorted(recorded_sources),
    }


def _run_prepare_stage_contract_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
    worker_payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    if scope != "prepare":
        return None
    return validate_prepare_stage_artifacts(
        stage=step,
        workspace_root=workspace_root,
        worker_payload=worker_payload,
    )


def _run_science_step_contract_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
    plan_steps: Sequence[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if scope != "science":
        return None
    reference_ids: set[str] = set()
    for candidate in plan_steps:
        if candidate is step:
            break
        if not isinstance(candidate, dict):
            continue
        disabled = [
            str(item).strip()
            for item in candidate.get("disabled_components") or []
            if str(item).strip()
        ]
        condition_id = str(candidate.get("condition_id") or "").strip()
        if condition_id and not disabled:
            reference_ids.add(condition_id)
    project_root = os.path.join(workspace_root, "project")
    issues = validate_science_condition_step_fields(
        step,
        project_dir=project_root,
        workspace_root=workspace_root,
        known_reference_ids=reference_ids,
    )
    output_dir = str(step.get("output_dir") or "").strip().rstrip("/")
    if output_dir:
        output_path = _resolve_workspace_path(workspace_root, output_dir)
        if not os.path.isdir(output_path):
            issues.append(f"Declared science output_dir does not exist: `{output_dir}`.")
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "science_step_contract",
        "scope": scope,
        "condition_id": str(step.get("condition_id") or "").strip(),
        "issues": issues,
        "checked_output_dir": output_dir,
        "reference_ids_available": sorted(reference_ids),
    }


def _run_science_raw_evidence_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
) -> Dict[str, Any] | None:
    if scope != "science":
        return None
    issues: List[str] = []
    checked: List[str] = []
    condition_id = str(step.get("condition_id") or "").strip()
    evidence_rel = artifact_rel(scope, condition_id, "evidence.json")
    evidence_path = _resolve_workspace_path(workspace_root, evidence_rel)
    checked.append(evidence_path)
    if not os.path.isfile(evidence_path):
        issues.append(f"Managed science evidence manifest is missing: `{evidence_rel}`.")
    else:
        try:
            with open(evidence_path, "r", encoding="utf-8") as f:
                evidence_payload = json.load(f)
        except Exception as exc:
            evidence_payload = None
            issues.append(f"Managed science evidence manifest is not valid JSON: `{evidence_rel}` ({exc}).")
        if evidence_payload is not None:
            issues.extend(
                validate_science_evidence_payload(
                    evidence_payload,
                    workspace_root=workspace_root,
                    step=step,
                )
            )

    def _check_json_numbers(path: str, payload: Any) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not (float("-inf") < float(value) < float("inf")):
                    issues.append(f"Non-finite numeric value in `{path}` at key `{key}`.")
                elif isinstance(value, (dict, list)):
                    _check_json_numbers(path, value)
        elif isinstance(payload, list):
            for item in payload:
                _check_json_numbers(path, item)

    for rel in step.get("raw_evidence") or []:
        raw_path = _resolve_workspace_path(workspace_root, str(rel))
        checked.append(raw_path)
        if not _path_under(workspace_root, raw_path):
            issues.append(f"Raw evidence path escapes workspace: `{rel}`.")
            continue
        if not os.path.isfile(raw_path):
            issues.append(f"Declared raw evidence file is missing: `{rel}`.")
            continue
        if raw_path.endswith(".json"):
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    _check_json_numbers(str(rel), json.load(f))
            except Exception as exc:
                issues.append(f"Raw evidence JSON could not be parsed: `{rel}` ({exc}).")
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "science_raw_evidence",
        "scope": scope,
        "condition_id": condition_id,
        "issues": issues,
        "checked_artifacts": checked,
    }


def _run_science_condition_hook(
    *,
    step: Dict[str, Any],
    scope: str,
    workspace_root: str,
    plan_steps: Sequence[Dict[str, Any]],
    review_payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    if scope != "science":
        return None
    issues: List[str] = []
    _ = workspace_root, plan_steps
    condition_id = str(step.get("condition_id") or "").strip()
    if str(review_payload.get("condition_id") or "").strip() != condition_id:
        issues.append("Reviewer output `condition_id` must match the condition contract.")
    for field in ("enabled_components", "disabled_components"):
        expected = [str(item).strip() for item in step.get(field) or [] if str(item).strip()]
        actual = [str(item).strip() for item in review_payload.get(field) or [] if str(item).strip()]
        if actual != expected:
            issues.append(f"Reviewer output `{field}` must match the condition contract: expected {expected}, got {actual}.")
    expected_reference = str(step.get("reference_condition_id") or "").strip()
    actual_reference = str(review_payload.get("reference_condition_id") or "").strip()
    if actual_reference != expected_reference:
        issues.append(
            "`reference_condition_id` in reviewer output must match the condition contract: "
            f"expected `{expected_reference}`, got `{actual_reference}`."
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "science_condition_contract",
        "scope": scope,
        "condition_id": condition_id,
        "issues": issues,
        "checked_output_dir": str(step.get("output_dir") or "").strip().rstrip("/"),
    }


def _run_science_result_contract_hook(
    *,
    step: Dict[str, Any],
    review_payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not step.get("disabled_components"):
        return None
    issues: List[str] = []
    expected_component_result = {
            "structured_findings": {
                "condition_id": step.get("condition_id"),
                "enabled_components": step.get("enabled_components") or [],
                "disabled_components": step.get("disabled_components") or [],
                "reference_condition_id": step.get("reference_condition_id"),
                "component_result": {
                    "result": "|".join(FINALIZABLE_SCIENCE_COMPONENT_RESULTS),
                    "metric": "metric used for the comparison",
                    "value": "effect size or comparison string",
                    "confidence": "number in [0.0, 1.0]",
                    "analysis": "evidence-backed interpretation against the reference condition",
                    "method_context": "short method context",
                    "follow_up_required": False,
                },
            }
    }
    result = str(review_payload.get("result") or "").strip().lower()
    if result not in set(SCIENCE_COMPONENT_RESULT_VALUES):
        issues.append(
            "Science component-disabled review must provide `structured_findings.component_result.result`; "
            "`result` must be one of "
            + "|".join(SCIENCE_COMPONENT_RESULT_VALUES)
            + f", got `{result or 'EMPTY'}`. Expected shape:\n"
            + json.dumps(expected_component_result, ensure_ascii=False, indent=2)
        )
    try:
        confidence = float(review_payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
        issues.append(
            "Science component-disabled review must provide numeric "
            "`structured_findings.component_result.confidence` between 0 and 1."
        )
    if confidence < 0 or confidence > 1:
        issues.append(
            "Science component-disabled review `structured_findings.component_result.confidence` "
            f"must be between 0 and 1, got `{review_payload.get('confidence')}`."
        )
    for field in ("metric", "value", "analysis", "method_context"):
        if not str(review_payload.get(field) or "").strip():
            issues.append(
                "Science component-disabled review must provide non-empty "
                f"`structured_findings.component_result.{field}`."
            )
    if review_payload.get("follow_up_required") is not False:
        issues.append(
            "Science component-disabled review must set "
            "`structured_findings.component_result.follow_up_required` to false before formal science can complete."
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "science_result_contract",
        "condition_id": step.get("condition_id"),
        "issues": issues,
        "allowed_results": list(SCIENCE_COMPONENT_RESULT_VALUES),
        "finalizable_results": list(FINALIZABLE_SCIENCE_COMPONENT_RESULTS),
        "min_final_confidence": MIN_FINAL_COMPONENT_CONFIDENCE,
        "expected_component_result": expected_component_result,
    }


def _science_structured_findings_from_review(
    *,
    step: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    structured = dict(review.get("structured_findings") or {})
    if str(review.get("reviewer_id") or "").strip() != SCIENCE_COMPONENT_RESULT_REVIEWER:
        structured.pop("component_result", None)
    for field in ("condition_id", "enabled_components", "disabled_components", "reference_condition_id"):
        structured.setdefault(field, review.get(field, step.get(field)))
    return structured


def _latest_science_structured_findings(
    *,
    step: Dict[str, Any],
    reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    for report in reversed(list(reports)):
        if str(report.get("reviewer_id") or "").strip() != SCIENCE_COMPONENT_RESULT_REVIEWER:
            continue
        structured = dict(report.get("structured_findings") or {})
        component_result = structured.get("component_result")
        if isinstance(component_result, dict) and component_result:
            for field in ("condition_id", "enabled_components", "disabled_components", "reference_condition_id"):
                structured.setdefault(field, step.get(field))
            return structured
    return {
        "condition_id": step.get("condition_id"),
        "enabled_components": step.get("enabled_components") or [],
        "disabled_components": step.get("disabled_components") or [],
        "reference_condition_id": step.get("reference_condition_id"),
        "component_result": {},
    }


def _deterministic_report_from_payload(
    *,
    reviewer_id: str,
    payload: Dict[str, Any] | None,
    issue_code: str,
    checked_artifacts: Sequence[str] | None = None,
) -> Dict[str, Any] | None:
    if payload is None:
        return None
    issues = [
        _review_issue(
            code=issue_code,
            message=str(item),
            required_fix=str(item),
        )
        for item in payload.get("issues") or []
        if str(item).strip()
    ]
    return _review_report(
        reviewer_id=reviewer_id,
        reviewer_kind="deterministic",
        status=str(payload.get("status") or "FAIL"),
        blocking=True,
        summary=f"{reviewer_id}: {payload.get('status') or 'UNKNOWN'}",
        checked_artifacts=checked_artifacts or [],
        issues=issues,
        structured_findings={payload.get("hook") or reviewer_id: payload},
    )


def _write_reviewer_report(
    *,
    layout: ReportLayout,
    scope: str,
    step_id: str,
    reviewer_id: str,
    attempt: int,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    latest_path, attempt_path = layout.review_latest_and_attempt(scope, step_id, reviewer_id, attempt)
    enriched = {
        **payload,
        "scope": payload.get("scope", scope),
        "step_id": payload.get("step_id", step_id),
        "attempt": int(attempt),
        "reviewer_id": payload.get("reviewer_id", reviewer_id),
    }
    write_json_file(attempt_path, enriched)
    write_json_file(latest_path, enriched)
    return {"latest": latest_path, "attempt": attempt_path}


def _write_attempt_and_latest(
    *,
    layout: ReportLayout,
    scope: str,
    step_id: str,
    role: str,
    attempt: int,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    latest_path, attempt_path = layout.latest_and_attempt(scope, step_id, role, attempt)
    enriched = {
        **payload,
        "scope": payload.get("scope", scope),
        "step_id": payload.get("step_id", step_id),
        "attempt": int(attempt),
    }
    write_json_file(attempt_path, enriched)
    write_json_file(latest_path, enriched)
    return {"latest": latest_path, "attempt": attempt_path}


def _append_timeline(workspace_root: str, event: Dict[str, Any]) -> None:
    layout = ReportLayout(workspace_root)
    os.makedirs(os.path.dirname(layout.run_timeline), exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **event,
    }
    with open(layout.run_timeline, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_executable_plan_contract(
    *,
    workspace_root: str,
    scope: str,
    plan_payload: Dict[str, Any],
) -> List[str]:
    steps = plan_payload.get("stages")
    if not isinstance(steps, list) or not steps:
        return [f"{scope}_plan must contain a non-empty top-level `stages` list."]
    project_dir = os.path.join(os.path.realpath(workspace_root), "project")
    if scope == "prepare":
        from src.agents.experiment_agent.runtime.prepare_contracts import validate_prepare_plan

        return validate_prepare_plan(plan_payload)
    if scope == "code":
        from src.agents.experiment_agent.runtime.contracts import validate_code_step_contract_fields

        issues: List[str] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                issues.append(f"step {index}: step must be an object")
                continue
            issues.extend(
                f"step {index}: {message}"
                for message in validate_code_step_contract_fields(
                    step,
                    project_dir=project_dir,
                    workspace_root=workspace_root,
                )
            )
        if isinstance(steps[-1], dict) and steps[-1].get("step_id") != "final_integration_smoke":
            issues.append("final step_id must be `final_integration_smoke`")
        return issues
    if scope == "science":
        from src.agents.experiment_agent.runtime.contracts import validate_science_condition_plan

        return validate_science_condition_plan(
            plan_payload,
            project_dir=project_dir,
            workspace_root=workspace_root,
        )
    return []


def materialize_executable_plan(
    *,
    workspace_root: str,
    scope: str,
    plan_payload: Dict[str, Any],
    planner_report: Dict[str, Any],
) -> Dict[str, str]:
    """Persist latest/raw plan, executable plan, and planner report."""
    layout = ReportLayout(workspace_root)
    raw_steps = plan_payload.get("stages")
    contract_steps = raw_steps if isinstance(raw_steps, list) else []
    plan_contract_errors = _validate_executable_plan_contract(
        workspace_root=workspace_root,
        scope=scope,
        plan_payload={**plan_payload, "stages": contract_steps},
    )
    plan_contract_status = "PASS" if not plan_contract_errors else "FAIL"
    if plan_contract_errors:
        raise RuntimeError(
            f"Managed `{scope}.plan` artifact failed executable plan contract:\n"
            + "\n".join(f"- {issue}" for issue in plan_contract_errors)
        )
    paths = {
        "latest": layout.planner_file(scope, "latest.json"),
        "executable": layout.planner_file(scope, "executable.json"),
        "planner_report": layout.planner_file(scope, "planner_report.json"),
    }
    if not os.path.isfile(paths["latest"]):
        raise RuntimeError(
            f"Managed planner artifact is missing at `{paths['latest']}`. "
            "The planner must write it through artifact tools before runtime materialization."
        )
    try:
        with open(paths["latest"], "r", encoding="utf-8") as f:
            managed_plan_payload = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Managed planner artifact is not readable JSON: {paths['latest']} ({exc})") from exc
    if managed_plan_payload != plan_payload:
        raise RuntimeError(
            f"Managed planner artifact `{paths['latest']}` does not match the planner payload. "
            "Repair the managed plan artifact with artifact tools instead of relying on runtime overwrite."
        )
    executable_payload = copy.deepcopy(plan_payload)
    executable_steps = executable_payload.get("stages")
    if not isinstance(executable_steps, list):
        executable_steps = []
    for step in executable_steps:
        if isinstance(step, dict):
            ensure_runtime_report_paths(scope, step, workspace_root)
    executable = {
        **executable_payload,
        "stages": executable_steps,
        "scope": scope,
        "plan_contract_status": plan_contract_status,
        "plan_contract_errors": plan_contract_errors,
    }
    write_json_file(paths["executable"], executable)
    write_json_file(
        paths["planner_report"],
        {
            **planner_report,
            "scope": planner_report.get("scope", scope),
            "plan_path": paths["latest"],
            "executable_plan_path": paths["executable"],
            "plan_contract_status": plan_contract_status,
            "plan_contract_errors": plan_contract_errors,
        },
    )
    _append_timeline(
        workspace_root,
        {
                "event": "plan_materialized",
                "scope": scope,
                "stage_count": len(executable_steps),
                "plan_path": paths["latest"],
            "executable_plan_path": paths["executable"],
            "plan_contract_status": plan_contract_status,
        },
    )
    return paths


def _parse_worker_payload_from_stop(payload: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str]:
    text = str(payload.get("assistant_text") or "").strip()
    if not text:
        return None, "Worker returned empty final text."
    try:
        parsed = extract_json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "Worker final response must be a JSON object."
    return parsed, ""


def _artifact_schema_feedback_from_contract(contract_payload: Dict[str, Any]) -> str:
    artifact_contract = contract_payload.get("artifact_contract")
    if not isinstance(artifact_contract, dict):
        return ""
    guides: List[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in artifact_contract.get("checked_artifacts") or []:
        if not isinstance(item, dict):
            continue
        schema_name = str(item.get("schema_name") or "").strip()
        if not schema_name:
            continue
        artifact_id = str(item.get("artifact_id") or "").strip()
        path = str(item.get("path") or "").strip()
        key = (artifact_id, path, schema_name)
        if key in seen:
            continue
        seen.add(key)
        guide = artifact_schema_repair_guide(
            schema_name,
            artifact_id=artifact_id,
            path=path,
        )
        if guide:
            guides.append(guide)
    return "\n\n".join(guides)


def _format_prefinish_feedback(
    *,
    scope: str,
    step_id: str,
    review_payload: Dict[str, Any],
    contract_payload: Dict[str, Any],
) -> str:
    review_status = str(review_payload.get("status") or "UNKNOWN")
    review_fixes = [
        str(item)
        for item in (
            list(review_payload.get("blocking_issues") or [])
            + list(review_payload.get("required_followup") or [])
            + str(review_payload.get("next_worker_input") or "").splitlines()
        )
        if str(item).strip()
    ]
    contract_issues = [str(item) for item in contract_payload.get("issues") or [] if str(item).strip()]
    schema_feedback = _artifact_schema_feedback_from_contract(contract_payload)
    lines = [
        "Xcientist prefinish hook blocked worker completion.",
        "",
        f"Scope: `{scope}`",
        f"Step: `{step_id}`",
        f"Reviewer status: `{review_status}`",
        "",
        "Review issues:",
        *(f"- {item}" for item in (review_fixes or ["Reviewer did not PASS this work unit."])),
        "",
        "Contract issues:",
        *(f"- {item}" for item in (contract_issues or ["No deterministic contract issue reported."])),
        "",
        "Fix the work in this same worker session, update managed artifacts through artifact tools, then finish again.",
    ]
    if schema_feedback:
        lines.extend(["", "Expected artifact schema / repair templates:", schema_feedback])
    next_worker_input = str(review_payload.get("next_worker_input") or "").strip()
    if next_worker_input:
        lines.extend(["", "Reviewer next_worker_input:", next_worker_input])
    return "\n".join(lines)


async def execute_step_with_prefinish_review(
    *,
    steps: List[Dict[str, Any]],
    scope: str,
    workspace_root: str,
    call_worker: Callable[[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]], Any],
    call_reviewer: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Any],
) -> Dict[str, Any]:
    final_reports: List[Dict[str, Any]] = []
    total_steps = len(steps)
    for idx, step in enumerate(steps, start=1):
        step_id = step.get("condition_id") or step.get("step_id") or step.get("stage_id") or f"step-{idx}"
        step_id = str(step_id)
        print_phase(
            f"{scope} step {idx}/{total_steps}",
            str(step_id),
            width=76,
        )
        attempt = 1
        layout = ReportLayout(workspace_root)
        completed_report = _completed_step_report_from_workspace(
            layout=layout,
            scope=scope,
            step_id=step_id,
        )
        if completed_report is not None:
            print_status("step result", "SKIP", "previously passed")
            final_reports.append(completed_report)
            _append_timeline(
                workspace_root,
                {
                    "event": "step_resume_skip",
                    "scope": scope,
                    "step_id": step_id,
                    "status": "PASS",
                },
            )
            continue
        registry = build_step_artifact_registry(
            workspace_root=workspace_root,
            scope=scope,
            step=step,
        )
        artifact_context = registry.to_context(stage=scope, step_id=str(step_id), attempt=attempt)
        write_artifact_registry_snapshot(registry)
        gate_state: Dict[str, Any] = {
            "worker_payload": None,
            "review_payload": None,
            "contract_payload": None,
            "attempt": 0,
        }

        async def _prefinish_gate(stop_payload: Dict[str, Any]) -> HookResult:
            gate_state["attempt"] = int(gate_state.get("attempt") or 0) + 1
            current_attempt = int(gate_state["attempt"])
            artifact_context["attempt"] = current_attempt
            _append_timeline(
                workspace_root,
                {
                    "event": "worker_stop",
                    "scope": scope,
                    "step_id": step_id,
                    "attempt": current_attempt,
                },
            )
            worker_payload, parse_error = _parse_worker_payload_from_stop(stop_payload)
            if worker_payload is None:
                return HookResult(
                    hook_type="xcientist_prefinish_gate",
                    success=False,
                    blocked=True,
                    reason=(
                        "Xcientist prefinish hook could not parse the worker final response as JSON. "
                        "Return the required worker JSON schema and finish again.\n\n"
                        f"Error: {parse_error}\n\n"
                        "Expected worker final JSON schema:\n"
                        "```json\n"
                        f"{_worker_schema_repair_text(scope=scope)}\n"
                        "```"
                    ),
                )
            worker_schema_issues = _validate_worker_payload_schema(worker_payload, scope=scope)
            if worker_schema_issues:
                return HookResult(
                    hook_type="xcientist_prefinish_gate",
                    success=False,
                    blocked=True,
                    reason=(
                        "Xcientist prefinish hook blocked worker completion because "
                        "the final worker JSON does not satisfy the required schema.\n\n"
                        "Schema issues:\n"
                        + "\n".join(f"- {issue}" for issue in worker_schema_issues)
                        + "\n\n"
                        + _worker_schema_repair_text(scope=scope)
                    ),
                    metadata={"schema_issues": worker_schema_issues},
                )
            worker_paths = _write_attempt_and_latest(
                layout=layout,
                scope=scope,
                step_id=step_id,
                role="worker",
                attempt=current_attempt,
                payload=worker_payload,
            )

            t1 = time.monotonic()
            deterministic_reports: List[Dict[str, Any]] = []
            print_status("formal hooks", "running", "stop hook", color=Colors.WARNING)
            worker_completion_payload = _run_worker_completion_hook(
                step=step,
                scope=scope,
                worker_payload=worker_payload,
            )
            worker_completion_report = _deterministic_report_from_payload(
                reviewer_id="worker_completion",
                payload=worker_completion_payload,
                issue_code="worker_completion",
                checked_artifacts=[worker_paths["latest"]],
            )
            if worker_completion_report:
                deterministic_reports.append(worker_completion_report)

            artifact_contract_payload = _run_artifact_contract_review_hook(
                registry=registry,
                worker_paths=worker_paths,
            )
            artifact_contract_report = _deterministic_report_from_payload(
                reviewer_id="artifact_contract",
                payload=artifact_contract_payload,
                issue_code="artifact_contract",
                checked_artifacts=[worker_paths["latest"]],
            )
            if artifact_contract_report:
                deterministic_reports.append(artifact_contract_report)

            hygiene_payload = scan_workspace_hygiene(workspace_root)
            hygiene_report = _deterministic_report_from_payload(
                reviewer_id="workspace_hygiene",
                payload={
                    **hygiene_payload,
                    "issues": [
                        (
                            f"Workspace hygiene issue at `{issue.get('path')}`: "
                            f"{issue.get('reason')}"
                        )
                        for issue in hygiene_payload.get("issues") or []
                        if isinstance(issue, dict)
                    ],
                },
                issue_code="workspace_hygiene",
                checked_artifacts=[],
            )
            if hygiene_report:
                deterministic_reports.append(hygiene_report)

            code_review_context: Dict[str, Any] | None = None
            code_review_context_path = ""
            if scope == "code":
                code_review_context = build_code_review_context(
                    workspace_root=workspace_root,
                    step=step,
                    worker_payload=worker_payload,
                    registry=registry,
                    plan_steps=steps,
                )
                code_review_context_path, code_review_context_attempt_path = layout.review_latest_and_attempt(
                    scope,
                    step_id,
                    "review_context",
                    current_attempt,
                )
                write_json_file(code_review_context_path, code_review_context)
                write_json_file(code_review_context_attempt_path, code_review_context)
                artifact_context["code_review_context"] = code_review_context
                artifact_context["code_review_context_path"] = code_review_context_path
                artifact_context["code_review_context_attempt_path"] = code_review_context_attempt_path
                artifact_context["selected_code_reviewer_ids"] = list(
                    code_review_context.get("selected_code_reviewer_ids") or []
                )
                deterministic_reports.append(
                    _review_report(
                        reviewer_id="code_review_context",
                        reviewer_kind="deterministic",
                        status="PASS",
                        blocking=True,
                        summary=(
                            "code_review_context: "
                            f"{code_review_context.get('risk_level')} risk; "
                            f"reviewers={','.join(code_review_context.get('selected_code_reviewer_ids') or [])}"
                        ),
                        checked_artifacts=[worker_paths["latest"], os.path.join(workspace_root, "project")],
                        issues=[],
                        structured_findings={"code_review_context": code_review_context},
                    )
                )

            project_integrity_payload = _run_code_project_integrity_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
                plan_steps=steps,
            )
            if project_integrity_payload:
                project_issues = []
                if project_integrity_payload.get("status") != "PASS":
                    project_issues = _structured_hook_issues_to_review_issues(
                        project_integrity_payload,
                        default_code="code_project_integrity",
                    )
                    project_issues.append(
                        _review_issue(
                            code="code_project_integrity_feedback",
                            message=format_project_integrity_feedback(project_integrity_payload),
                            required_fix=format_project_integrity_feedback(project_integrity_payload),
                        )
                    )
                deterministic_reports.append(
                    _review_report(
                        reviewer_id="code_project_integrity",
                        reviewer_kind="deterministic",
                        status=str(project_integrity_payload.get("status") or "FAIL"),
                        blocking=True,
                        summary=f"code_project_integrity: {project_integrity_payload.get('status')}",
                        checked_artifacts=[worker_paths["latest"], os.path.join(workspace_root, "project")],
                        issues=project_issues,
                        structured_findings={"code_project_integrity": project_integrity_payload},
                    )
                )

            source_provenance_payload = _run_code_source_provenance_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
            )
            source_provenance_report = _deterministic_report_from_payload(
                reviewer_id="code_source_provenance",
                payload=source_provenance_payload,
                issue_code="code_source_provenance",
                checked_artifacts=list((source_provenance_payload or {}).get("checked_artifacts") or []),
            )
            if source_provenance_report:
                deterministic_reports.append(source_provenance_report)

            prepare_stage_payload = _run_prepare_stage_contract_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
                worker_payload=worker_payload,
            )
            prepare_stage_report = _deterministic_report_from_payload(
                reviewer_id="prepare_stage_contract",
                payload=prepare_stage_payload,
                issue_code="prepare_stage_contract",
                checked_artifacts=list((prepare_stage_payload or {}).get("checked_artifacts") or []),
            )
            if prepare_stage_report:
                deterministic_reports.append(prepare_stage_report)

            code_invariant_payload = None
            if scope == "code" and code_review_context:
                code_invariant_payload = audit_code_scientific_invariants(
                    workspace_root=workspace_root,
                    review_context=code_review_context,
                )
                code_invariant_issues = []
                if code_invariant_payload.get("status") != "PASS":
                    code_invariant_issues = _structured_hook_issues_to_review_issues(
                        code_invariant_payload,
                        default_code="code_scientific_invariants",
                    )
                    code_invariant_issues.append(
                        _review_issue(
                            code="code_scientific_invariants_feedback",
                            message=format_code_invariant_feedback(code_invariant_payload),
                            required_fix=format_code_invariant_feedback(code_invariant_payload),
                        )
                    )
                deterministic_reports.append(
                    _review_report(
                        reviewer_id="code_scientific_invariants",
                        reviewer_kind="deterministic",
                        status=str(code_invariant_payload.get("status") or "FAIL"),
                        blocking=True,
                        summary=f"code_scientific_invariants: {code_invariant_payload.get('status')}",
                        checked_artifacts=[
                            worker_paths["latest"],
                            os.path.join(workspace_root, "project"),
                            code_review_context_path,
                        ],
                        issues=code_invariant_issues,
                        structured_findings={"code_scientific_invariants": code_invariant_payload},
                    )
                )

            science_step_payload = _run_science_step_contract_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
                plan_steps=steps,
            )
            science_step_report = _deterministic_report_from_payload(
                reviewer_id="science_step_contract",
                payload=science_step_payload,
                issue_code="science_step_contract",
                checked_artifacts=[worker_paths["latest"]],
            )
            if science_step_report:
                deterministic_reports.append(science_step_report)

            science_evidence_payload = _run_science_raw_evidence_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
            )
            science_evidence_report = _deterministic_report_from_payload(
                reviewer_id="science_raw_evidence",
                payload=science_evidence_payload,
                issue_code="science_raw_evidence",
                checked_artifacts=list((science_evidence_payload or {}).get("checked_artifacts") or []),
            )
            if science_evidence_report:
                deterministic_reports.append(science_evidence_report)

            deterministic_failed = any(
                report.get("status") != "PASS" and bool(report.get("blocking", True))
                for report in deterministic_reports
            )
            elapsed_formal = time.monotonic() - t1
            print_status(
                "formal hooks",
                "FAIL" if deterministic_failed else "PASS",
                format_seconds(elapsed_formal),
                color=Colors.FAIL if deterministic_failed else Colors.OKGREEN,
            )

            agent_reports: List[Dict[str, Any]] = []
            if deterministic_failed:
                print_status(
                    "agent review",
                    "skipped",
                    "formal hooks failed",
                    color=Colors.WARNING,
                )
            else:
                print_status("agent review", "running", "prefinish", color=Colors.WARNING)
                t_agent = time.monotonic()
                raw_agent_result = await call_reviewer(step, worker_payload, artifact_context)
                raw_agent_reports = raw_agent_result if isinstance(raw_agent_result, list) else [raw_agent_result]
                for index, raw_report in enumerate(raw_agent_reports, start=1):
                    reviewer_id = (
                        str(raw_report.get("reviewer_id") or "") if isinstance(raw_report, dict) else ""
                    ) or ("science_interpretation" if scope == "science" else f"{scope}_semantic")
                    normalized = _normalize_agent_review_report(
                        raw_report if isinstance(raw_report, dict) else {},
                        reviewer_id=reviewer_id,
                        scope=scope,
                    )
                    if scope == "science":
                        normalized["structured_findings"] = _science_structured_findings_from_review(
                            step=step,
                            review=normalized,
                        )
                    normalized["worker_report_path"] = worker_paths["latest"]
                    normalized["worker_attempt_report_path"] = worker_paths["attempt"]
                    agent_reports.append(normalized)
                agent_failed = any(report.get("status") != "PASS" for report in agent_reports)
                print_status(
                    "agent review",
                    "FAIL" if agent_failed else "PASS",
                    format_seconds(time.monotonic() - t_agent),
                    color=Colors.FAIL if agent_failed else Colors.OKGREEN,
                )

            checked_artifacts_payload = _run_reviewer_checked_artifacts_hook(
                scope=scope,
                step_id=step_id,
                workspace_root=workspace_root,
                reports=agent_reports,
            )
            checked_artifacts_report = _deterministic_report_from_payload(
                reviewer_id="reviewer_checked_artifacts",
                payload=checked_artifacts_payload,
                issue_code="reviewer_checked_artifacts",
                checked_artifacts=list((checked_artifacts_payload or {}).get("checked_artifacts") or []),
            )
            if checked_artifacts_report:
                deterministic_reports.append(checked_artifacts_report)

            science_findings = _latest_science_structured_findings(
                step=step,
                reports=agent_reports,
            ) if scope == "science" else None
            review_payload = _unified_to_phase_review_payload(
                aggregate=_aggregate_review_reports(
                    scope=scope,
                    step_id=step_id,
                    attempt=current_attempt,
                    reports=[*deterministic_reports, *agent_reports],
                ),
                scope=scope,
                science_findings=science_findings,
            )
            review_payload["worker_report_path"] = worker_paths["latest"]
            review_payload["worker_attempt_report_path"] = worker_paths["attempt"]
            if project_integrity_payload:
                review_payload["project_integrity"] = project_integrity_payload
            if source_provenance_payload:
                review_payload["source_provenance"] = source_provenance_payload
            if prepare_stage_payload:
                review_payload["prepare_stage_contract"] = prepare_stage_payload

            science_condition_payload = _run_science_condition_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
                plan_steps=steps,
                review_payload=review_payload,
            )
            if science_condition_payload:
                science_condition_report = _deterministic_report_from_payload(
                    reviewer_id="science_condition_output",
                    payload=science_condition_payload,
                    issue_code="science_condition_output",
                    checked_artifacts=[worker_paths["latest"]],
                )
                if science_condition_report:
                    deterministic_reports.append(science_condition_report)
                    review_payload["science_condition_contract"] = science_condition_payload

            science_result_payload = _run_science_result_contract_hook(
                step=step,
                review_payload=review_payload,
            ) if scope == "science" else None
            science_result_report = _deterministic_report_from_payload(
                reviewer_id="science_result_contract",
                payload=science_result_payload,
                issue_code="science_result_contract",
                checked_artifacts=[worker_paths["latest"]],
            )
            if science_result_report:
                deterministic_reports.append(science_result_report)
                review_payload["science_result_contract"] = science_result_payload

            aggregate = _aggregate_review_reports(
                scope=scope,
                step_id=step_id,
                attempt=current_attempt,
                reports=[*deterministic_reports, *agent_reports],
            )
            review_payload = _unified_to_phase_review_payload(
                aggregate=aggregate,
                scope=scope,
                science_findings=science_findings,
            )
            review_payload["worker_report_path"] = worker_paths["latest"]
            review_payload["worker_attempt_report_path"] = worker_paths["attempt"]
            if project_integrity_payload:
                review_payload["project_integrity"] = project_integrity_payload
            if source_provenance_payload:
                review_payload["source_provenance"] = source_provenance_payload
            if prepare_stage_payload:
                review_payload["prepare_stage_contract"] = prepare_stage_payload
            if science_condition_payload:
                review_payload["science_condition_contract"] = science_condition_payload
            if science_result_payload:
                review_payload["science_result_contract"] = science_result_payload

            for report in [*deterministic_reports, *agent_reports]:
                _write_reviewer_report(
                    layout=layout,
                    scope=scope,
                    step_id=step_id,
                    reviewer_id=str(report.get("reviewer_id") or "reviewer"),
                    attempt=current_attempt,
                    payload=report,
                )
            aggregate_paths = _write_reviewer_report(
                layout=layout,
                scope=scope,
                step_id=step_id,
                reviewer_id="aggregate",
                attempt=current_attempt,
                payload=aggregate,
            )

            print_status("contract hook", "running", "stop hook", color=Colors.WARNING)
            contract_payload = _run_step_contract_hook(
                step=step,
                scope=scope,
                workspace_root=workspace_root,
                worker_payload=worker_payload,
                review_payload=review_payload,
                attempt=current_attempt,
            )
            contract_color = (
                Colors.OKGREEN if contract_payload["status"] == "PASS" else Colors.FAIL
            )
            print_status("contract hook", contract_payload["status"], color=contract_color)

            if contract_payload["status"] != "PASS" and review_payload.get("status") == "PASS":
                contract_report = _deterministic_report_from_payload(
                    reviewer_id="prefinish_contract",
                    payload=contract_payload,
                    issue_code="prefinish_contract",
                    checked_artifacts=list(contract_payload.get("checked_outputs") or []),
                )
                if contract_report:
                    deterministic_reports.append(contract_report)
                    _write_reviewer_report(
                        layout=layout,
                        scope=scope,
                        step_id=step_id,
                        reviewer_id="prefinish_contract",
                        attempt=current_attempt,
                        payload=contract_report,
                    )
                    aggregate = _aggregate_review_reports(
                        scope=scope,
                        step_id=step_id,
                        attempt=current_attempt,
                        reports=[*deterministic_reports, *agent_reports],
                    )
                    review_payload = _unified_to_phase_review_payload(
                        aggregate=aggregate,
                        scope=scope,
                        science_findings=science_findings,
                    )
                    review_payload["worker_report_path"] = worker_paths["latest"]
                    review_payload["worker_attempt_report_path"] = worker_paths["attempt"]
                    if project_integrity_payload:
                        review_payload["project_integrity"] = project_integrity_payload
                    if source_provenance_payload:
                        review_payload["source_provenance"] = source_provenance_payload
                    if prepare_stage_payload:
                        review_payload["prepare_stage_contract"] = prepare_stage_payload
                    if science_condition_payload:
                        review_payload["science_condition_contract"] = science_condition_payload
                    if science_result_payload:
                        review_payload["science_result_contract"] = science_result_payload
                    aggregate_paths = _write_reviewer_report(
                        layout=layout,
                        scope=scope,
                        step_id=step_id,
                        reviewer_id="aggregate",
                        attempt=current_attempt,
                        payload=aggregate,
                    )
            review_payload["prefinish_contract"] = contract_payload
            review_payload["review_report_path"] = aggregate_paths["latest"]
            review_payload["review_attempt_report_path"] = aggregate_paths["attempt"]
            review_payload["review_aggregate_report_path"] = aggregate_paths["latest"]

            review_paths = _write_attempt_and_latest(
                layout=layout,
                scope=scope,
                step_id=step_id,
                role="review",
                attempt=current_attempt,
                payload=review_payload,
            )
            review_payload["review_report_path"] = review_paths["latest"]
            review_payload["review_attempt_report_path"] = review_paths["attempt"]
            hook_payload = {
                "scope": scope,
                "step_id": step_id,
                "attempt": current_attempt,
                "status": contract_payload.get("status"),
                "review_status": review_payload.get("status"),
                "worker_report_path": worker_paths["latest"],
                "worker_attempt_report_path": worker_paths["attempt"],
                "review_report_path": review_paths["latest"],
                "review_attempt_report_path": review_paths["attempt"],
                "review_aggregate_report_path": aggregate_paths["latest"],
                "returned_to_worker": not (
                    review_payload.get("status") == "PASS" and contract_payload["status"] == "PASS"
                )
                ,
                "prefinish_contract": contract_payload,
                "review_matrix": aggregate,
            }
            hook_paths = _write_attempt_and_latest(
                layout=layout,
                scope=scope,
                step_id=step_id,
                role="hook",
                attempt=current_attempt,
                payload=hook_payload,
            )
            contract_payload["hook_report_path"] = hook_paths["latest"]
            contract_payload["hook_attempt_report_path"] = hook_paths["attempt"]

            gate_state["worker_payload"] = worker_payload
            gate_state["review_payload"] = review_payload
            gate_state["contract_payload"] = contract_payload
            _append_timeline(
                workspace_root,
                {
                    "event": "prefinish_gate",
                    "scope": scope,
                    "step_id": step_id,
                    "attempt": current_attempt,
                    "review_status": review_payload.get("status"),
                    "contract_status": contract_payload.get("status"),
                    "returned_to_worker": hook_payload["returned_to_worker"],
                },
            )

            if review_payload.get("status") == "PASS" and contract_payload["status"] == "PASS":
                return HookResult(
                    hook_type="xcientist_prefinish_gate",
                    success=True,
                    blocked=False,
                    metadata={
                        "review_payload": review_payload,
                        "contract_payload": contract_payload,
                    },
                )

            reason = _format_prefinish_feedback(
                scope=scope,
                step_id=str(step_id),
                review_payload=review_payload,
                contract_payload=contract_payload,
            )
            return HookResult(
                hook_type="xcientist_prefinish_gate",
                success=False,
                blocked=True,
                reason=reason,
                metadata={
                    "review_payload": review_payload,
                    "contract_payload": contract_payload,
                },
            )

        label = "run"
        t0 = time.monotonic()
        print_status("worker", "running", label, color=Colors.WARNING)
        worker_payload = await call_worker(step, None, artifact_context, _prefinish_gate)
        elapsed = time.monotonic() - t0
        print_status("worker", "done", format_seconds(elapsed))
        review_payload = gate_state.get("review_payload")
        contract_payload = gate_state.get("contract_payload")
        if not isinstance(review_payload, dict) or not isinstance(contract_payload, dict):
            print_status(
                "step result",
                "FAIL",
                "prefinish gate did not run",
                color=Colors.FAIL,
            )
            print_status(
                "contract hook",
                "FAIL",
                "worker completed without STOP hook metadata",
                color=Colors.FAIL,
            )
            return {
                "status": "FAIL",
                "failed_step": step,
                "failed_review_report": {
                    "status": "FAIL",
                    "evidence_summary": "worker completed without running prefinish gate",
                    "blocking_issues": ["prefinish gate did not run"],
                    "terminal_blocker": True,
                    "next_worker_input": "",
                },
                "step_reports": final_reports,
            }
        if review_payload.get("status") == "PASS" and contract_payload.get("status") == "PASS":
            prepare_stage_contract = review_payload.get("prepare_stage_contract")
            if (
                scope == "prepare"
                and isinstance(prepare_stage_contract, dict)
                and str(prepare_stage_contract.get("stage_status") or "").strip().upper() == PREPARE_BLOCKED
            ):
                print_status("step result", "BLOCKED", "credible prepare blocker", color=Colors.WARNING)
                final_reports.append(review_payload)
                return {
                    "status": "BLOCKED",
                    "blocked_step": step,
                    "blocked_review_report": review_payload,
                    "step_reports": final_reports,
                }
            print_status("step result", "PASS")
            final_reports.append(review_payload)
        else:
            print_status(
                "step result",
                "FAIL",
                f"terminal_blocker={review_payload.get('terminal_blocker')}",
                color=Colors.FAIL,
            )
            return {
                "status": "FAIL",
                "failed_step": step,
                "failed_review_report": review_payload,
                "step_reports": final_reports,
            }
    print_phase(f"{scope} complete", f"All {total_steps} steps completed.", width=76)
    return {"status": "PASS", "step_reports": final_reports}
