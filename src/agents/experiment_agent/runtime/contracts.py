"""
Shared path-contract and verdict-contract helpers for experiment-agent prompts.

These helpers intentionally stay lightweight. They do not validate agent output
semantics; they only keep the phase prompts aligned on the same contract fields
and path labels so information can move cleanly between agents.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, MutableMapping, Sequence, Tuple


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
    "component_scope",
    "code_artifacts",
    "interface_contract",
    "implementation_requirements",
    "ablation_hooks",
    "experiment_bindings",
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
    "train_dataset_binding",
    "evaluation_dataset_bindings",
    "metric_bindings",
    "baseline_binding",
    "full_method_binding",
    "comparison_table_schema",
    "claim_axis_binding",
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
    "train_dataset_binding",
    "evaluation_dataset_bindings",
    "metric_bindings",
    "baseline_reference",
    "ablated_condition_binding",
    "expected_effect_axis",
    "result_interpretation_rule",
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
    "evidence_summary",
    "required_fixes",
    "terminal_blocker",
    "next_worker_input",
    "checked_artifacts",
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
    # LLMs often add explanatory text after the enum value; extract the prefix.
    _VALID_INTENTS = {"none", "reference_only", "copy_and_modify"}
    if repo_copy_intent not in _VALID_INTENTS:
        # Try extracting just the first token/word in case LLM appended explanation
        first_word = repo_copy_intent.split()[0] if repo_copy_intent else ""
        if first_word in _VALID_INTENTS:
            repo_copy_intent = first_word
            payload = dict(payload)
            payload["repo_copy_intent"] = repo_copy_intent
        else:
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
    if repo_copy_intent == "reference_only" and not repo_source_paths:
        repo_copy_intent = "none"
        if isinstance(payload, MutableMapping):
            payload["repo_copy_intent"] = repo_copy_intent
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


def validate_code_step_contract_fields(
    payload: Mapping[str, Any],
    *,
    project_dir: str,
) -> list[str]:
    errors = validate_repo_contract_fields(payload, project_dir=project_dir)

    if not isinstance(payload.get("component_scope"), list) or not all(
        isinstance(item, str) and item.strip() for item in (payload.get("component_scope") or [])
    ):
        errors.append("`component_scope` must be a non-empty list of strings")

    code_artifacts = payload.get("code_artifacts")
    if not isinstance(code_artifacts, list) or not code_artifacts:
        errors.append("`code_artifacts` must be a non-empty list")
    else:
        required_artifact_keys = {
            "path",
            "artifact_type",
            "symbols",
            "responsibility",
            "dependencies",
            "config_keys",
            "entrypoint_role",
        }
        for index, item in enumerate(code_artifacts, start=1):
            if not isinstance(item, Mapping):
                errors.append(f"`code_artifacts[{index}]` must be an object")
                continue
            missing = required_artifact_keys - set(item.keys())
            if missing:
                errors.append(
                    f"`code_artifacts[{index}]` missing required keys: {', '.join(sorted(missing))}"
                )

    for field in ("interface_contract", "implementation_requirements", "experiment_bindings"):
        if not isinstance(payload.get(field), Mapping) or not payload.get(field):
            errors.append(f"`{field}` must be a non-empty object")

    ablation_hooks = payload.get("ablation_hooks")
    component_scope = payload.get("component_scope") or []
    if component_scope and (not isinstance(ablation_hooks, list) or not ablation_hooks):
        errors.append("`ablation_hooks` must be a non-empty list when `component_scope` is defined")

    return errors
