"""
Shared path-contract and verdict-contract helpers for experiment-agent prompts.

These helpers intentionally stay lightweight. They do not validate agent output
semantics; they only keep the phase prompts aligned on the same contract fields
and path labels so information can move cleanly between agents.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence, Tuple


PREPARE_STAGE_CONTRACT_FIELDS: Tuple[str, ...] = (
    "stage_id",
    "goal",
    "stage_contract_path",
    "executor_report_path",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
    "repos_policy",
    "project_must_be_self_contained",
    "provenance_manifest_path",
    "research_required",
    "acquisition_required",
    "existing_local_hints",
    "max_repair_rounds",
    "done_condition",
)

CODE_STEP_CONTRACT_FIELDS: Tuple[str, ...] = (
    "step_id",
    "goal",
    "step_contract_path",
    "executor_report_path",
    "repo_source_paths",
    "repo_copy_intent",
    "project_target_paths",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
    "repos_policy",
    "project_must_be_self_contained",
    "provenance_manifest_path",
    "write_scope",
    "verify_command",
    "max_repair_rounds",
    "done_condition",
)

SCIENCE_STANDARD_STEP_FIELDS: Tuple[str, ...] = (
    "step_id",
    "goal",
    "step_contract_path",
    "executor_report_path",
    "repo_source_paths",
    "repo_copy_intent",
    "project_target_paths",
    "target_scope",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
    "repos_policy",
    "project_must_be_self_contained",
    "provenance_manifest_path",
    "command",
    "output_dir",
    "raw_evidence",
    "max_repair_rounds",
    "pass_condition",
)

SCIENCE_ABLATION_STEP_FIELDS: Tuple[str, ...] = (
    "step_id",
    "goal",
    "step_contract_path",
    "executor_report_path",
    "repo_source_paths",
    "repo_copy_intent",
    "project_target_paths",
    "component_or_condition",
    "canonical_component_index",
    "component_explanation",
    "method_context_change",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
    "repos_policy",
    "project_must_be_self_contained",
    "provenance_manifest_path",
    "command",
    "output_dir",
    "raw_evidence",
    "max_repair_rounds",
    "pass_condition",
)

PHASE_VERDICT_FIELDS: Tuple[str, ...] = (
    "status",
    "scope",
    "checked_artifacts",
    "findings",
    "required_fixes",
    "evidence_summary",
    "phase_completion_status",
    "ready_for_next_phase",
    "blocking_issues",
    "required_followup",
    "artifact_role",
    "run_level",
    "self_contained_project",
    "self_contained_violations",
    "provenance_manifest_present",
    "provenance_manifest_path",
)

ABLATION_COMPONENT_RESULT_FIELDS: Tuple[str, ...] = (
    "result",
    "metric",
    "value",
    "confidence",
    "analysis",
    "method_context",
    "follow_up_required",
)


def format_field_bullets(fields: Sequence[str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}`{field}`" for field in fields)


def format_named_paths(paths: Mapping[str, str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}{name}: {value}" for name, value in paths.items())


def validate_repo_contract_fields(
    payload: Mapping[str, Any],
    *,
    project_dir: str,
) -> list[str]:
    errors: list[str] = []
    for field in ("repo_source_paths", "repo_copy_intent", "project_target_paths"):
        if field not in payload:
            errors.append(f"missing required field `{field}`")

    repo_source_paths = payload.get("repo_source_paths")
    if not isinstance(repo_source_paths, list) or not all(
        isinstance(item, str) for item in (repo_source_paths or [])
    ):
        errors.append("`repo_source_paths` must be a list of strings")
        repo_source_paths = []
    else:
        repo_source_paths = [item.strip() for item in repo_source_paths if item.strip()]

    repo_copy_intent = str(payload.get("repo_copy_intent") or "").strip()
    if repo_copy_intent not in {"none", "reference_only", "copy_and_modify"}:
        errors.append("`repo_copy_intent` must be one of `none|reference_only|copy_and_modify`")

    project_target_paths = payload.get("project_target_paths")
    if not isinstance(project_target_paths, list) or not all(
        isinstance(item, str) for item in (project_target_paths or [])
    ):
        errors.append("`project_target_paths` must be a list of strings")
        project_target_paths = []
    else:
        project_target_paths = [item.strip() for item in project_target_paths if item.strip()]

    normalized_project_dir = os.path.realpath(project_dir)
    if repo_copy_intent != "none" and not repo_source_paths:
        errors.append("`repo_source_paths` must be non-empty when `repo_copy_intent` is not `none`")
    if repo_copy_intent == "copy_and_modify" and not project_target_paths:
        errors.append(
            "`project_target_paths` must be non-empty when `repo_copy_intent` is `copy_and_modify`"
        )
    for target in project_target_paths:
        target_path = target if os.path.isabs(target) else os.path.join(project_dir, target)
        normalized_target = os.path.realpath(target_path)
        if normalized_target != normalized_project_dir and not normalized_target.startswith(
            normalized_project_dir + os.sep
        ):
            errors.append(f"`project_target_paths` entry must stay inside project_dir: {target}")

    return errors
