"""
Runtime support utilities for experiment-agent.
"""

from src.agents.experiment_agent.runtime.cache import Cache
from src.agents.experiment_agent.runtime.ablation_results import (
    REQUIRED_COMPONENT_FIELDS,
    REQUIRED_SUMMARY_FIELDS,
    build_ablation_results_artifacts,
    validate_ablation_results_payload,
    write_ablation_results_artifacts,
)
from src.agents.experiment_agent.runtime.contracts import (
    ABLATION_COMPONENT_RESULT_FIELDS,
    CODE_STEP_CONTRACT_FIELDS,
    PHASE_VERDICT_FIELDS,
    PREPARE_STAGE_CONTRACT_FIELDS,
    SCIENCE_ABLATION_STEP_FIELDS,
    SCIENCE_STANDARD_STEP_FIELDS,
    format_field_bullets,
    format_named_paths,
)
from src.agents.experiment_agent.runtime.idea_components import (
    IDEA_COMPONENTS_HEADING,
    canonical_component_names,
    find_idea_json_path,
    format_canonical_components_markdown,
    load_canonical_components,
    load_idea_json,
)
from src.agents.experiment_agent.runtime.phase_contracts import (
    ARTIFACT_ROLE_FINAL_RESULT,
    ARTIFACT_ROLE_PHASE_RESULT,
    ARTIFACT_ROLE_SMOKE_CHECK,
    RUN_LEVEL_FULL,
    RUN_LEVEL_MIXED,
    RUN_LEVEL_SMOKE,
    infer_phase_completion_status,
    normalize_phase_report,
    phase_artifact_role,
    phase_blocking_issues,
    phase_ready_for_next,
    phase_report_status,
    phase_run_level,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    load_workspace_state,
    workspace_contract_paths,
    write_json_file,
)
from src.agents.experiment_agent.runtime.self_contained import (
    scan_project_self_contained,
)

__all__ = [
    "Cache",
    "ABLATION_COMPONENT_RESULT_FIELDS",
    "CODE_STEP_CONTRACT_FIELDS",
    "IDEA_COMPONENTS_HEADING",
    "PHASE_VERDICT_FIELDS",
    "PREPARE_STAGE_CONTRACT_FIELDS",
    "SCIENCE_ABLATION_STEP_FIELDS",
    "SCIENCE_STANDARD_STEP_FIELDS",
    "REQUIRED_COMPONENT_FIELDS",
    "REQUIRED_SUMMARY_FIELDS",
    "ARTIFACT_ROLE_FINAL_RESULT",
    "ARTIFACT_ROLE_PHASE_RESULT",
    "ARTIFACT_ROLE_SMOKE_CHECK",
    "RUN_LEVEL_FULL",
    "RUN_LEVEL_MIXED",
    "RUN_LEVEL_SMOKE",
    "artifact_paths",
    "canonical_component_names",
    "build_ablation_results_artifacts",
    "find_idea_json_path",
    "format_field_bullets",
    "format_canonical_components_markdown",
    "format_named_paths",
    "load_canonical_components",
    "load_idea_json",
    "load_json_file",
    "load_workspace_state",
    "infer_phase_completion_status",
    "normalize_phase_report",
    "phase_artifact_role",
    "phase_blocking_issues",
    "phase_ready_for_next",
    "phase_report_status",
    "phase_run_level",
    "validate_ablation_results_payload",
    "scan_project_self_contained",
    "workspace_contract_paths",
    "write_ablation_results_artifacts",
    "write_json_file",
]
