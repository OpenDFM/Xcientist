"""
Shared path-contract and verdict-contract helpers for experiment-agent prompts.

These helpers intentionally stay lightweight. They do not validate agent output
semantics; they only keep the phase prompts aligned on the same contract fields
and path labels so information can move cleanly between agents.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple


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
    "max_repair_rounds",
    "done_condition",
)

CODE_STEP_CONTRACT_FIELDS: Tuple[str, ...] = (
    "step_id",
    "goal",
    "step_contract_path",
    "executor_report_path",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
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
    "target_scope",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
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
    "component_or_condition",
    "canonical_component_index",
    "component_explanation",
    "method_context_change",
    "input_paths",
    "allowed_write_roots",
    "required_output_roots",
    "worker_report_path",
    "validator_report_path",
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
