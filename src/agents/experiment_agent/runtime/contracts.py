"""
Shared path-contract and verdict-contract helpers for experiment-agent prompts.

These helpers intentionally stay lightweight. They do not validate agent output
semantics; they only keep the phase prompts aligned on the same contract fields
and path labels so information can move cleanly between agents.
"""

from __future__ import annotations

import os
import json
import re
from typing import Any, Mapping, Sequence, Tuple


_PLACEHOLDER_VALUES = {
    "bad",
    "demo",
    "example",
    "fake",
    "foo",
    "mock",
    "placeholder",
    "sample",
    "test",
    "test_artifact",
    "test_component",
    "test_goal",
    "test_hook",
    "test_step",
    "todo",
    "tbd",
}


PREPARE_STAGE_CONTRACT_FIELDS: Tuple[str, ...] = (
    "stage_id",
    "goal",
    "input_paths",
    "repos_policy",
    "project_must_be_self_contained",
    "research_required",
    "acquisition_required",
    "existing_local_hints",
    "done_condition",
    "artifact_ids",
)

CODE_STEP_CONTRACT_FIELDS: Tuple[str, ...] = (
    "step_id",
    "goal",
    "component_scope",
    "code_artifacts",
    "interface_contract",
    "implementation_requirements",
    "component_disable_hooks",
    "experiment_bindings",
    "repo_source_paths",
    "repo_copy_intent",
    "project_target_paths",
    "input_paths",
    "repos_policy",
    "project_must_be_self_contained",
    "write_scope",
    "verify_command",
    "done_condition",
    "artifact_ids",
)

SCIENCE_CONDITION_STEP_FIELDS: Tuple[str, ...] = (
    "condition_id",
    "goal",
    "enabled_components",
    "disabled_components",
    "reference_condition_id",
    "train_dataset_binding",
    "evaluation_dataset_bindings",
    "metric_bindings",
    "component_set_description",
    "result_interpretation_rule",
    "run_level",
    "setup_rationale",
    "source_basis",
    "runtime_probe_summary",
    "training_protocol",
    "evaluation_protocol",
    "repo_source_paths",
    "repo_copy_intent",
    "project_target_paths",
    "input_paths",
    "repos_policy",
    "project_must_be_self_contained",
    "command",
    "output_dir",
    "raw_evidence",
    "pass_condition",
    "artifact_ids",
)

PHASE_VERDICT_FIELDS: Tuple[str, ...] = (
    "status",
    "evidence_summary",
    "phase_completion_status",
    "ready_for_next_phase",
    "blocking_issues",
    "required_followup",
    "terminal_blocker",
    "next_worker_input",
    "checked_artifacts",
    "artifact_role",
    "run_level",
)

SCIENCE_COMPONENT_RESULT_FIELDS: Tuple[str, ...] = (
    "result",
    "metric",
    "value",
    "confidence",
    "analysis",
    "method_context",
    "follow_up_required",
)

SCIENCE_COMPONENT_RESULT_VALUES: Tuple[str, ...] = (
    "positive",
    "negative",
    "neutral",
    "inconclusive",
)

SCIENCE_TRAINING_PROTOCOL_FIELDS: Tuple[str, ...] = (
    "epochs",
    "max_batches",
    "batch_size",
    "device",
    "seed",
    "expected_runtime_sec",
    "full_setup_basis",
)

SCIENCE_EVALUATION_PROTOCOL_FIELDS: Tuple[str, ...] = (
    "horizons",
    "mask_rates",
    "mask_patterns",
    "metrics",
    "reference_condition_id",
    "perturbation_boundary",
    "preprocessing_boundary",
    "ablation_isolation_assumptions",
)

SCIENCE_CONDITION_REVIEW_FIELDS: Tuple[str, ...] = (
    "condition_id",
    "enabled_components",
    "disabled_components",
    "reference_condition_id",
    *SCIENCE_COMPONENT_RESULT_FIELDS,
)

SCIENCE_EVIDENCE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "condition_id",
    "enabled_components",
    "disabled_components",
    "reference_condition_id",
    "run_level",
    "command",
    "returncode",
    "output_dir",
    "raw_outputs",
    "logs",
    "metrics_files",
    "dataset_bindings",
    "model_bindings",
)

SCIENCE_EVIDENCE_PATH_LIST_FIELDS: Tuple[str, ...] = (
    "raw_outputs",
    "logs",
    "metrics_files",
)

CODE_HANDOFF_REQUIRED_FIELDS: Tuple[str, ...] = (
    "project_files",
    "verification",
    "verify_command",
    "returncode",
    "logs",
)

CODE_SMOKE_EVIDENCE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "command",
    "returncode",
    "raw_outputs",
    "logs",
    "metrics_files",
    "dataset_bindings",
    "component_toggles",
    "bounded_runtime",
)


def format_field_bullets(fields: Sequence[str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}`{field}`" for field in fields)


def format_named_paths(paths: Mapping[str, str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}{name}: {value}" for name, value in paths.items())


def validate_repo_contract_fields(
    payload: Mapping[str, Any],
    *,
    project_dir: str,
    workspace_root: str | None = None,
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
    _VALID_INTENTS = {"none", "reference_only", "copy_and_modify"}
    if repo_copy_intent not in _VALID_INTENTS:
        errors.append("`repo_copy_intent` must be exactly one of `none|reference_only|copy_and_modify`")

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
    normalized_workspace = os.path.realpath(
        workspace_root or os.path.dirname(normalized_project_dir)
    )
    normalized_repos_dir = os.path.join(normalized_workspace, "repos")

    for source in repo_source_paths:
        source_path = _resolve_workspace_or_absolute_path(source, normalized_workspace)
        normalized_source = os.path.realpath(source_path)
        if not _path_under(normalized_workspace, normalized_source):
            errors.append(f"`repo_source_paths` entry must stay inside workspace: {source}")
        elif not _path_under(normalized_repos_dir, normalized_source):
            errors.append(f"`repo_source_paths` entry must be under repos/: {source}")
        elif workspace_root and not os.path.exists(normalized_source):
            errors.append(f"`repo_source_paths` entry does not exist: {source}")

    for target in project_target_paths:
        target_path = _resolve_project_path(target, normalized_project_dir, normalized_workspace)
        normalized_target = os.path.realpath(target_path)
        if not _path_under(normalized_project_dir, normalized_target):
            errors.append(f"`project_target_paths` entry must stay inside project_dir: {target}")

    return errors


def _path_under(root: str, path: str) -> bool:
    root = os.path.realpath(root)
    path = os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


def _resolve_workspace_or_absolute_path(path: str, workspace_root: str) -> str:
    path = str(path or "").strip()
    if os.path.isabs(path):
        return path
    return os.path.join(workspace_root, path)


def _resolve_project_path(path: str, project_dir: str, workspace_root: str) -> str:
    path = str(path or "").strip()
    if os.path.isabs(path):
        return path
    if path == "project" or path.startswith("project/"):
        return os.path.join(workspace_root, path)
    return os.path.join(project_dir, path)


def _is_placeholder_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    compact = text.replace("-", "_").replace(" ", "_")
    return compact in _PLACEHOLDER_VALUES


def _contains_placeholder_only_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for key, item in value.items():
        if not _is_placeholder_text(key) or not _is_placeholder_text(item):
            return False
    return True


def _load_canonical_component_names(workspace_root: str | None) -> list[str]:
    if not workspace_root:
        return []
    idea_path = os.path.join(os.path.realpath(workspace_root), "idea.json")
    if not os.path.isfile(idea_path):
        return []
    try:
        import json

        with open(idea_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    raw_components = payload.get("components") if isinstance(payload, Mapping) else None
    if not isinstance(raw_components, list):
        return []
    names: list[str] = []
    for item in raw_components:
        if isinstance(item, Mapping):
            name = str(item.get("component") or "").strip()
            if name:
                names.append(name)
    return names


def validate_code_step_contract_fields(
    payload: Mapping[str, Any],
    *,
    project_dir: str,
    workspace_root: str | None = None,
) -> list[str]:
    normalized_project_dir = os.path.realpath(project_dir)
    normalized_workspace = os.path.realpath(
        workspace_root or os.path.dirname(normalized_project_dir)
    )
    canonical_components = _load_canonical_component_names(normalized_workspace)
    errors = validate_repo_contract_fields(
        payload,
        project_dir=normalized_project_dir,
        workspace_root=normalized_workspace,
    )

    step_id = str(payload.get("step_id") or "").strip()
    if _is_placeholder_text(step_id):
        errors.append("`step_id` must be a concrete implementation step id, not a placeholder.")

    goal = str(payload.get("goal") or "").strip()
    if _is_placeholder_text(goal):
        errors.append("`goal` must describe real experiment implementation work, not a placeholder.")

    component_scope = payload.get("component_scope")
    if not isinstance(component_scope, list) or not all(
        isinstance(item, str) and item.strip() for item in (component_scope or [])
    ):
        errors.append("`component_scope` must be a non-empty list of strings")
        component_scope = []
    else:
        component_scope = [str(item).strip() for item in component_scope if str(item).strip()]
        placeholders = [item for item in component_scope if _is_placeholder_text(item)]
        if placeholders:
            errors.append("`component_scope` contains placeholder names: " + ", ".join(placeholders))
        if canonical_components:
            unknown = [item for item in component_scope if item not in canonical_components]
            if unknown:
                errors.append(
                    "`component_scope` entries must come from idea.json.components: "
                    + ", ".join(unknown)
                )

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
            artifact_path = str(item.get("path") or "").strip()
            if _is_placeholder_text(item.get("artifact_type")):
                errors.append(f"`code_artifacts[{index}].artifact_type` must not be a placeholder")
            if _is_placeholder_text(item.get("responsibility")):
                errors.append(f"`code_artifacts[{index}].responsibility` must not be a placeholder")
            if _is_placeholder_text(item.get("entrypoint_role")):
                errors.append(f"`code_artifacts[{index}].entrypoint_role` must not be a placeholder")
            if not artifact_path:
                errors.append(f"`code_artifacts[{index}].path` must be non-empty")
            else:
                resolved_artifact_path = _resolve_project_path(
                    artifact_path,
                    normalized_project_dir,
                    normalized_workspace,
                )
                if not _path_under(normalized_project_dir, resolved_artifact_path):
                    errors.append(
                        f"`code_artifacts[{index}].path` must stay inside project_dir: {artifact_path}"
                    )
            for list_field in ("symbols", "dependencies", "config_keys"):
                if not isinstance(item.get(list_field), list):
                    errors.append(f"`code_artifacts[{index}].{list_field}` must be a list")
            symbols = item.get("symbols")
            if isinstance(symbols, list) and not any(str(symbol).strip() for symbol in symbols):
                errors.append(f"`code_artifacts[{index}].symbols` must name at least one real symbol")

    for field in ("interface_contract", "implementation_requirements", "experiment_bindings"):
        if not isinstance(payload.get(field), Mapping) or not payload.get(field):
            errors.append(f"`{field}` must be a non-empty object")
        elif _contains_placeholder_only_mapping(payload.get(field)):
            errors.append(f"`{field}` must describe real bindings, not placeholder key/value pairs")

    disable_hooks = payload.get("component_disable_hooks")
    if component_scope and (not isinstance(disable_hooks, list) or not disable_hooks):
        errors.append("`component_disable_hooks` must be a non-empty list when `component_scope` is defined")
    elif isinstance(disable_hooks, list):
        scoped_components = set(str(item).strip() for item in component_scope)
        for index, item in enumerate(disable_hooks, start=1):
            if not isinstance(item, Mapping):
                errors.append(f"`component_disable_hooks[{index}]` must be an object")
                continue
            component = str(item.get("component") or item.get("component_name") or "").strip()
            if not component:
                errors.append(f"`component_disable_hooks[{index}]` must name a `component`")
            elif scoped_components and component not in scoped_components:
                errors.append(
                    f"`component_disable_hooks[{index}].component` must be in component_scope: {component}"
                )
            if not any(
                str(item.get(key) or "").strip()
                for key in (
                    "flag",
                    "config_key",
                    "config_override",
                    "condition",
                    "command_arg",
                    "env_var",
                    "mode",
                )
            ):
                errors.append(
                    f"`component_disable_hooks[{index}]` must declare a concrete disable/toggle mechanism"
                )

    input_paths = payload.get("input_paths")
    if not isinstance(input_paths, Mapping):
        errors.append("`input_paths` must be an object")

    if payload.get("repos_policy") != "reference_or_copy":
        errors.append("`repos_policy` must be `reference_or_copy`")
    if payload.get("project_must_be_self_contained") is not True:
        errors.append("`project_must_be_self_contained` must be true")
    write_scope = str(payload.get("write_scope") or "").strip().rstrip("/")
    if write_scope != "project":
        errors.append("`write_scope` must be `project` or `project/`")

    verify_command = str(payload.get("verify_command") or "").strip()
    if not verify_command:
        errors.append("`verify_command` must be non-empty")
    else:
        normalized_command = verify_command.lower().strip()
        if normalized_command in {"true", ":", "echo test", "echo ok"} or normalized_command.startswith(
            ("echo ", "printf ")
        ):
            errors.append("`verify_command` must run real verification, not echo/printf/true placeholders")
        if not any(token in normalized_command for token in ("python", "pytest", "bash", " sh ")):
            errors.append("`verify_command` should invoke a real Python/test/shell verification command")

    done_condition = str(payload.get("done_condition") or "").strip()
    lowered_done = done_condition.lower()
    if _is_placeholder_text(done_condition):
        errors.append("`done_condition` must be concrete, not a placeholder")
    if "sys.path" not in lowered_done:
        errors.append("`done_condition` must explicitly forbid sys.path injection")
    if "editable" not in lowered_done and "pip install -e" not in lowered_done:
        errors.append("`done_condition` must explicitly forbid editable installs of repos/")
    if "outside project" not in lowered_done and "outside `project/`" not in lowered_done:
        errors.append("`done_condition` must explicitly forbid imports reaching outside project/")

    artifact_ids = payload.get("artifact_ids")
    expected_handoff = f"code.{step_id}.handoff" if step_id else ""
    if not isinstance(artifact_ids, list) or not all(isinstance(item, str) for item in artifact_ids):
        errors.append("`artifact_ids` must be a list of strings")
        artifact_ids = []
    else:
        artifact_ids = [item.strip() for item in artifact_ids if item.strip()]
        if expected_handoff and expected_handoff not in artifact_ids:
            errors.append(f"`artifact_ids` must include `{expected_handoff}`")

    if step_id == "final_integration_smoke":
        if "code.final_integration_smoke.evidence" not in artifact_ids:
            errors.append("final_integration_smoke `artifact_ids` must include `code.final_integration_smoke.evidence`")
        final_text = " ".join(
            str(payload.get(field) or "")
            for field in (
                "goal",
                "interface_contract",
                "implementation_requirements",
                "experiment_bindings",
                "verify_command",
                "done_condition",
                "artifact_ids",
            )
        ).lower()
        if "dataset_candidate" not in final_text:
            errors.append("final_integration_smoke must explicitly use real data from `dataset_candidate/`")
        if (
            "code.final_integration_smoke.evidence" not in final_text
            and "agent_reports/code/artifacts/final_integration_smoke.json" not in final_text
        ):
            errors.append(
                "final_integration_smoke must require `code.final_integration_smoke.evidence` "
                "at `agent_reports/code/artifacts/final_integration_smoke.json`"
            )
        if not any(term in final_text for term in ("metric", "evaluation", "mae", "rmse", "mape")):
            errors.append("final_integration_smoke must require real evaluation/metrics evidence")
        if any(term in final_text for term in ("synthetic data", "random data", "mock", "dry-run-only", "imports-only")):
            if not any(term in final_text for term in ("forbid", "prohibit", "not ", "no ", "never")):
                errors.append("final_integration_smoke must prohibit mocks, synthetic data, dry-run-only, and imports-only evidence")

    return errors


def _validate_workspace_path_list(
    payload: Mapping[str, Any],
    *,
    field: str,
    workspace_root: str,
    errors: list[str],
    require_files: bool = True,
    require_project: bool = False,
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"`{field}` must be a non-empty list of paths.")
        return []
    paths: list[str] = []
    normalized_workspace = os.path.realpath(workspace_root)
    project_root = os.path.join(normalized_workspace, "project")
    for index, item in enumerate(value, start=1):
        raw = str(item or "").strip()
        if not raw:
            errors.append(f"`{field}[{index}]` must be a non-empty path.")
            continue
        abs_path = _resolve_workspace_or_absolute_path(raw, normalized_workspace)
        if not _path_under(normalized_workspace, abs_path):
            errors.append(f"`{field}[{index}]` escapes workspace: {raw}")
            continue
        if require_project and not _path_under(project_root, abs_path):
            errors.append(f"`{field}[{index}]` must stay under project/: {raw}")
            continue
        if require_files and not os.path.isfile(abs_path):
            errors.append(f"`{field}[{index}]` file does not exist: {raw}")
        paths.append(raw)
    return paths


def _validate_json_metric_files(
    *,
    paths: Sequence[str],
    workspace_root: str,
    errors: list[str],
    field: str,
) -> None:
    for raw in paths:
        abs_path = _resolve_workspace_or_absolute_path(raw, workspace_root)
        if not os.path.isfile(abs_path) or not abs_path.endswith(".json"):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                _validate_science_evidence_numbers(raw, json.load(f), errors)
        except Exception as exc:
            errors.append(f"`{field}` JSON could not be parsed: `{raw}` ({exc}).")


def validate_code_handoff_payload(
    payload: Any,
    *,
    workspace_root: str,
) -> list[str]:
    """Validate a managed code handoff manifest.

    The handoff is worker-authored evidence, so failures must be returned to
    the same worker through artifact-tool or prefinish-hook feedback.
    """
    if not isinstance(payload, Mapping):
        return ["code_handoff must be a JSON object."]
    normalized_workspace = os.path.realpath(workspace_root)
    errors: list[str] = []
    for field in CODE_HANDOFF_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"code_handoff missing required `{field}`.")

    project_files = _validate_workspace_path_list(
        payload,
        field="project_files",
        workspace_root=normalized_workspace,
        errors=errors,
        require_project=True,
    )
    if not project_files:
        errors.append("code_handoff must name at least one existing project/ file touched by this step.")

    verification = str(payload.get("verification") or "").strip()
    if _is_placeholder_text(verification):
        errors.append("`verification` must summarize real verification, not empty or placeholder text.")

    verify_command = str(payload.get("verify_command") or "").strip()
    if not verify_command:
        errors.append("`verify_command` must be non-empty.")
    else:
        lowered = verify_command.lower().strip()
        if lowered in {"true", ":", "echo test", "echo ok"} or lowered.startswith(("echo ", "printf ")):
            errors.append("`verify_command` must run real verification, not echo/printf/true placeholders.")
        if re.search(r"(^|[;&|]\s*)cd\s+project\b", verify_command):
            errors.append("`verify_command` must run from the workspace root; do not `cd project`.")

    returncode = payload.get("returncode")
    if not isinstance(returncode, int):
        errors.append("`returncode` must be an integer.")
    elif returncode != 0:
        errors.append("`returncode` must be 0 for an accepted code handoff.")

    _validate_workspace_path_list(
        payload,
        field="logs",
        workspace_root=normalized_workspace,
        errors=errors,
    )
    for optional_field in ("raw_outputs", "metrics_files"):
        if optional_field in payload:
            paths = _validate_workspace_path_list(
                payload,
                field=optional_field,
                workspace_root=normalized_workspace,
                errors=errors,
            )
            if optional_field == "metrics_files":
                _validate_json_metric_files(
                    paths=paths,
                    workspace_root=normalized_workspace,
                    errors=errors,
                    field=optional_field,
                )
    return errors


def validate_code_smoke_evidence_payload(
    payload: Any,
    *,
    workspace_root: str,
) -> list[str]:
    """Validate final bounded code-smoke evidence before science can run."""
    if not isinstance(payload, Mapping):
        return ["code_smoke_evidence must be a JSON object."]
    normalized_workspace = os.path.realpath(workspace_root)
    errors: list[str] = []
    for field in CODE_SMOKE_EVIDENCE_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"code_smoke_evidence missing required `{field}`.")

    command = str(payload.get("command") or "").strip()
    if not command:
        errors.append("`command` must be non-empty.")
    elif re.search(r"(^|[;&|]\s*)cd\s+project\b", command):
        errors.append("`command` must run from the workspace root; do not `cd project`.")

    returncode = payload.get("returncode")
    if not isinstance(returncode, int):
        errors.append("`returncode` must be an integer.")
    elif returncode != 0:
        errors.append("`returncode` must be 0 for accepted final integration smoke evidence.")

    raw_outputs = _validate_workspace_path_list(
        payload,
        field="raw_outputs",
        workspace_root=normalized_workspace,
        errors=errors,
    )
    _validate_workspace_path_list(
        payload,
        field="logs",
        workspace_root=normalized_workspace,
        errors=errors,
    )
    metrics_files = _validate_workspace_path_list(
        payload,
        field="metrics_files",
        workspace_root=normalized_workspace,
        errors=errors,
    )
    _validate_json_metric_files(
        paths=metrics_files,
        workspace_root=normalized_workspace,
        errors=errors,
        field="metrics_files",
    )

    dataset_text = json.dumps(payload.get("dataset_bindings"), ensure_ascii=False, sort_keys=True)
    if not isinstance(payload.get("dataset_bindings"), (Mapping, list)) or not payload.get("dataset_bindings"):
        errors.append("`dataset_bindings` must be a non-empty object or list.")
    elif "dataset_candidate" not in dataset_text:
        errors.append("`dataset_bindings` must reference real prepared data under dataset_candidate/.")

    if not isinstance(payload.get("component_toggles"), (Mapping, list)) or not payload.get("component_toggles"):
        errors.append("`component_toggles` must be a non-empty object or list describing all-components and disabled-component paths.")

    bounded_runtime = payload.get("bounded_runtime")
    if isinstance(bounded_runtime, Mapping):
        if not bounded_runtime:
            errors.append("`bounded_runtime` must describe the smoke runtime bound.")
    elif not str(bounded_runtime or "").strip():
        errors.append("`bounded_runtime` must describe the smoke runtime bound.")

    if not raw_outputs and not metrics_files:
        errors.append("code_smoke_evidence must provide real raw_outputs or metrics_files.")
    return errors


def _is_supported_source_uri(value: str) -> bool:
    return bool(
        re.match(r"^https?://[^\s]+$", value)
        or re.match(r"^(doi|arxiv):[^\s]+$", value, re.IGNORECASE)
        or re.match(r"^git\+https://[^\s]+$", value, re.IGNORECASE)
    )


def validate_science_source_basis(
    source_basis: Any,
    *,
    workspace_root: str,
) -> list[str]:
    """Validate traceable setup-rationale sources for science planning."""
    if not isinstance(source_basis, list) or not source_basis:
        return ["`source_basis` must be a non-empty list of local findings or external sources used for setup."]
    normalized_workspace = os.path.realpath(workspace_root)
    errors: list[str] = []
    traceable_count = 0
    for index, item in enumerate(source_basis, start=1):
        if isinstance(item, Mapping):
            basis_text = str(item.get("basis") or item.get("summary") or "").strip()
            raw_path = str(item.get("path") or "").strip()
            raw_url = str(item.get("url") or item.get("source") or "").strip()
            if _is_placeholder_text(basis_text or raw_path or raw_url):
                errors.append(f"`source_basis[{index}]` must describe a concrete source/basis.")
            if raw_path:
                resolved = _resolve_workspace_or_absolute_path(raw_path, normalized_workspace)
                if not _path_under(normalized_workspace, resolved):
                    errors.append(f"`source_basis[{index}].path` escapes workspace: {raw_path}")
                elif not os.path.exists(resolved):
                    errors.append(f"`source_basis[{index}].path` does not exist: {raw_path}")
                else:
                    traceable_count += 1
            if raw_url:
                if _is_supported_source_uri(raw_url):
                    traceable_count += 1
                elif not raw_path:
                    errors.append(
                        f"`source_basis[{index}].url/source` must be http(s), doi:, arxiv:, git+https, or an existing local `path`: {raw_url}"
                    )
            if not raw_path and not raw_url and basis_text:
                traceable_count += 1
            if not raw_path and not raw_url and not basis_text:
                errors.append(f"`source_basis[{index}]` must include `basis`, `path`, or `url/source`.")
            continue
        if _is_placeholder_text(item):
            errors.append(f"`source_basis[{index}]` must be concrete, not a placeholder.")
        else:
            traceable_count += 1
    if traceable_count == 0:
        errors.append("`source_basis` must contain at least one traceable local path, external URL, or concrete local-basis statement.")
    return errors


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _protocol_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _extract_command_option(command: str, names: Sequence[str]) -> str:
    for name in names:
        match = re.search(rf"(?:^|\s)--{re.escape(name)}(?:=|\s+)([^\s]+)", command)
        if match:
            return match.group(1).strip().strip("'\"")
    return ""


def _validate_required_mapping(
    *,
    payload: Mapping[str, Any],
    field: str,
    required_keys: Sequence[str],
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, Mapping) or not value:
        return [f"`{field}` must be a non-empty object."]
    errors: list[str] = []
    for key in required_keys:
        if key not in value:
            errors.append(f"`{field}` missing required key `{key}`.")
            continue
        if field == "evaluation_protocol" and key == "reference_condition_id":
            continue
        if _is_placeholder_text(value.get(key)):
            errors.append(f"`{field}.{key}` must be concrete, not empty or placeholder.")
    return errors


def _validate_science_protocol_command_consistency(
    *,
    payload: Mapping[str, Any],
    command: str,
) -> list[str]:
    errors: list[str] = []
    training = payload.get("training_protocol")
    evaluation = payload.get("evaluation_protocol")
    if not isinstance(training, Mapping):
        training = {}
    if not isinstance(evaluation, Mapping):
        evaluation = {}

    lowered_command = f" {command.lower()} "
    pilot_terms = (
        "--smoke",
        "--dry-run",
        "--dry_run",
        "--debug",
        "--probe-only",
        "--probe_only",
        "--quick",
        " smoke ",
        " dry-run ",
    )
    if any(term in lowered_command for term in pilot_terms):
        errors.append("formal science `command` must not be a smoke, dry-run, debug, probe-only, or quick run.")

    for option_names, protocol_key, minimum_invalid in (
        (("epochs", "max-epochs", "max_epochs"), "epochs", 1),
        (("max-batches", "max_batches"), "max_batches", 2),
    ):
        option_value = _extract_command_option(command, option_names)
        if option_value:
            try:
                numeric = float(option_value)
            except ValueError:
                numeric = None
            if numeric is not None and numeric <= minimum_invalid:
                errors.append(
                    f"formal science `command` uses pilot-like `{option_names[0]}` value {option_value}."
                )

    for option_names, protocol_key in (
        (("epochs", "max-epochs", "max_epochs"), "epochs"),
        (("batch-size", "batch_size"), "batch_size"),
        (("seed",), "seed"),
        (("device",), "device"),
    ):
        command_value = _extract_command_option(command, option_names)
        protocol_text = _protocol_value(training.get(protocol_key))
        if command_value and protocol_text and command_value != protocol_text:
            errors.append(
                f"`training_protocol.{protocol_key}` ({protocol_text}) conflicts with command option `{option_names[0]}` ({command_value})."
            )

    command_reference = _extract_command_option(command, ("reference-condition-id", "reference_condition_id", "reference"))
    protocol_reference = _protocol_value(evaluation.get("reference_condition_id"))
    if command_reference and protocol_reference and command_reference != protocol_reference:
        errors.append(
            "`evaluation_protocol.reference_condition_id` conflicts with the command reference condition."
        )
    return errors


def _science_evidence_path_list(
    payload: Mapping[str, Any],
    *,
    field: str,
    workspace_root: str,
    output_dir: str,
    require_exists: bool,
    errors: list[str],
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"`{field}` must be a non-empty list of paths.")
        return []
    paths: list[str] = []
    output_abs = _resolve_workspace_or_absolute_path(output_dir, workspace_root) if output_dir else ""
    for index, item in enumerate(value, start=1):
        raw = str(item or "").strip()
        if not raw:
            errors.append(f"`{field}[{index}]` must be a non-empty path.")
            continue
        abs_path = _resolve_workspace_or_absolute_path(raw, workspace_root)
        if not _path_under(workspace_root, abs_path):
            errors.append(f"`{field}[{index}]` escapes workspace: {raw}")
            continue
        if output_abs and not _path_under(output_abs, abs_path):
            errors.append(f"`{field}[{index}]` must stay under `output_dir`: {raw}")
        if require_exists and not os.path.isfile(abs_path):
            errors.append(f"`{field}[{index}]` file does not exist: {raw}")
        paths.append(raw)
    return paths


def _science_evidence_bindings(
    payload: Mapping[str, Any],
    *,
    field: str,
    errors: list[str],
) -> None:
    value = payload.get(field)
    if not isinstance(value, (list, Mapping)) or not value:
        errors.append(f"`{field}` must be a non-empty object or list describing the prepared resource binding.")


def _science_evidence_duration(payload: Mapping[str, Any], errors: list[str]) -> None:
    if "duration_sec" in payload:
        try:
            duration = float(payload.get("duration_sec"))
        except (TypeError, ValueError):
            duration = -1.0
        if duration <= 0:
            errors.append("`duration_sec` must be a positive number when provided.")
        return
    if not str(payload.get("started_at") or "").strip() or not str(payload.get("ended_at") or "").strip():
        errors.append("Science evidence must include `duration_sec` or both `started_at` and `ended_at`.")


def _validate_science_evidence_numbers(path: str, payload: Any, errors: list[str]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not (float("-inf") < float(value) < float("inf")):
                errors.append(f"Non-finite numeric value in `{path}` at key `{key}`.")
            elif isinstance(value, (Mapping, list)):
                _validate_science_evidence_numbers(path, value, errors)
    elif isinstance(payload, list):
        for item in payload:
            _validate_science_evidence_numbers(path, item, errors)


def validate_science_evidence_payload(
    payload: Any,
    *,
    workspace_root: str,
    step: Mapping[str, Any] | None = None,
    require_exists: bool = True,
) -> list[str]:
    """Validate a managed science evidence manifest.

    This is intentionally a contract check, not an auto-normalizer. Failures are
    returned to the same worker through artifact tool or prefinish-hook feedback.
    """
    if not isinstance(payload, Mapping):
        return ["science_evidence must be a JSON object."]
    normalized_workspace = os.path.realpath(workspace_root)
    errors: list[str] = []
    for field in SCIENCE_EVIDENCE_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"science_evidence missing required `{field}`.")

    condition_id = _condition_id(payload)
    if _is_placeholder_text(condition_id):
        errors.append("`condition_id` must be concrete, not empty or placeholder.")
    if "/" in condition_id or "\\" in condition_id:
        errors.append("`condition_id` must be path-safe without slashes.")

    for field in ("enabled_components", "disabled_components"):
        if not isinstance(payload.get(field), list) or not all(isinstance(item, str) for item in payload.get(field) or []):
            errors.append(f"`{field}` must be a list of strings.")

    run_level = str(payload.get("run_level") or "").strip().lower()
    if run_level != "full":
        errors.append("science evidence must declare `run_level` exactly `full`.")

    command = str(payload.get("command") or "").strip()
    if not command:
        errors.append("`command` must be non-empty.")
    if re.search(r"(^|[;&|]\s*)cd\s+project\b", command):
        errors.append("`command` must run from the workspace root; do not `cd project`.")

    returncode = payload.get("returncode")
    if not isinstance(returncode, int):
        errors.append("`returncode` must be an integer.")
    elif returncode != 0:
        errors.append("`returncode` must be 0 for accepted formal science evidence.")

    output_dir = str(payload.get("output_dir") or "").strip().rstrip("/")
    expected_prefix = f"results/science/{condition_id}" if condition_id else "results/science/"
    if not output_dir:
        errors.append("`output_dir` must be non-empty.")
    elif not (output_dir == expected_prefix or output_dir.startswith(expected_prefix + "/")):
        errors.append(
            "`output_dir` must be under `results/science/<condition_id>/`: "
            f"expected prefix `{expected_prefix}`, got `{output_dir}`"
        )
    if output_dir:
        output_abs = _resolve_workspace_or_absolute_path(output_dir, normalized_workspace)
        if not _path_under(normalized_workspace, output_abs):
            errors.append(f"`output_dir` escapes workspace: {output_dir}")
        elif require_exists and not os.path.isdir(output_abs):
            errors.append(f"`output_dir` does not exist: {output_dir}")

    path_values: dict[str, list[str]] = {}
    for field in SCIENCE_EVIDENCE_PATH_LIST_FIELDS:
        path_values[field] = _science_evidence_path_list(
            payload,
            field=field,
            workspace_root=normalized_workspace,
            output_dir=output_dir,
            require_exists=require_exists,
            errors=errors,
        )

    for metrics_path in path_values.get("metrics_files") or []:
        abs_path = _resolve_workspace_or_absolute_path(metrics_path, normalized_workspace)
        if not os.path.isfile(abs_path):
            continue
        if abs_path.endswith(".json"):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    _validate_science_evidence_numbers(metrics_path, json.load(f), errors)
            except Exception as exc:
                errors.append(f"`metrics_files` JSON could not be parsed: `{metrics_path}` ({exc}).")

    for field in ("dataset_bindings", "model_bindings"):
        _science_evidence_bindings(payload, field=field, errors=errors)
    _science_evidence_duration(payload, errors)

    if step is not None:
        expected_condition = _condition_id(step)
        if condition_id != expected_condition:
            errors.append(
                f"`condition_id` must match the condition contract: expected `{expected_condition}`, got `{condition_id}`."
            )
        for field in ("enabled_components", "disabled_components"):
            expected = _string_list(step.get(field))
            actual = _string_list(payload.get(field))
            if actual != expected:
                errors.append(f"`{field}` must match the condition contract: expected {expected}, got {actual}.")
        expected_reference = str(step.get("reference_condition_id") or "").strip()
        actual_reference = str(payload.get("reference_condition_id") or "").strip()
        if actual_reference != expected_reference:
            errors.append(
                "`reference_condition_id` must match the condition contract: "
                f"expected `{expected_reference}`, got `{actual_reference}`."
            )
        expected_command = str(step.get("command") or "").strip()
        if expected_command and command != expected_command:
            errors.append("`command` in science evidence must exactly match the condition contract command.")
        expected_output_dir = str(step.get("output_dir") or "").strip().rstrip("/")
        if expected_output_dir and output_dir != expected_output_dir:
            errors.append(
                f"`output_dir` must match the condition contract: expected `{expected_output_dir}`, got `{output_dir}`."
            )
        declared = {str(item).strip() for item in step.get("raw_evidence") or [] if str(item).strip()}
        manifest_paths = set(path_values.get("raw_outputs") or []) | set(path_values.get("logs") or []) | set(path_values.get("metrics_files") or [])
        missing_from_manifest = sorted(declared - manifest_paths)
        if missing_from_manifest:
            errors.append(
                "Condition `raw_evidence` entries must appear in the science evidence manifest: "
                + ", ".join(missing_from_manifest)
            )
    return errors


def _condition_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("condition_id") or payload.get("step_id") or "").strip()


def validate_science_condition_step_fields(
    payload: Mapping[str, Any],
    *,
    project_dir: str,
    workspace_root: str | None = None,
    known_reference_ids: set[str] | None = None,
) -> list[str]:
    normalized_project_dir = os.path.realpath(project_dir)
    normalized_workspace = os.path.realpath(
        workspace_root or os.path.dirname(normalized_project_dir)
    )
    canonical_components = _load_canonical_component_names(normalized_workspace)
    canonical_set = set(canonical_components)
    errors = validate_repo_contract_fields(
        payload,
        project_dir=normalized_project_dir,
        workspace_root=normalized_workspace,
    )

    condition_id = _condition_id(payload)
    if _is_placeholder_text(condition_id):
        errors.append("`condition_id` must be a concrete science condition id, not a placeholder.")
    if "/" in condition_id or "\\" in condition_id:
        errors.append("`condition_id` must be a simple path-safe id without slashes.")

    goal = str(payload.get("goal") or "").strip()
    if _is_placeholder_text(goal):
        errors.append("`goal` must describe the real experiment condition, not a placeholder.")

    enabled_components = _string_list(payload.get("enabled_components"))
    disabled_components = _string_list(payload.get("disabled_components"))
    if payload.get("enabled_components") is None or not isinstance(payload.get("enabled_components"), list):
        errors.append("`enabled_components` must be a list.")
    if payload.get("disabled_components") is None or not isinstance(payload.get("disabled_components"), list):
        errors.append("`disabled_components` must be a list.")

    if canonical_components:
        all_declared = enabled_components + disabled_components
        unknown = sorted(set(all_declared) - canonical_set)
        if unknown:
            errors.append(
                "`enabled_components`/`disabled_components` must come from idea.json.components: "
                + ", ".join(unknown)
            )
        duplicate_across = sorted(set(enabled_components) & set(disabled_components))
        if duplicate_across:
            errors.append(
                "components cannot be both enabled and disabled: " + ", ".join(duplicate_across)
            )
        if set(all_declared) != canonical_set:
            errors.append(
                "`enabled_components` plus `disabled_components` must cover idea.json.components exactly: "
                f"expected {canonical_components}, got enabled={enabled_components}, disabled={disabled_components}"
            )

    reference = str(payload.get("reference_condition_id") or "").strip()
    if disabled_components:
        if len(disabled_components) != 1:
            errors.append("component-disabled science conditions must disable exactly one canonical component.")
        if not reference:
            errors.append("conditions with disabled components must set `reference_condition_id`.")
        elif known_reference_ids is not None and reference not in known_reference_ids:
            errors.append(
                "`reference_condition_id` must refer to an earlier all-components condition: "
                + reference
            )
    elif reference:
        errors.append("all-components reference conditions must leave `reference_condition_id` empty.")

    output_dir = str(payload.get("output_dir") or "").strip().rstrip("/")
    expected_prefix = f"results/science/{condition_id}" if condition_id else "results/science/"
    if not output_dir:
        errors.append("`output_dir` must be non-empty.")
    elif not (output_dir == expected_prefix or output_dir.startswith(expected_prefix + "/")):
        errors.append(
            "`output_dir` must be under `results/science/<condition_id>/`: "
            f"expected prefix `{expected_prefix}`, got `{output_dir}`"
        )
    command = str(payload.get("command") or "").strip()
    lowered_command = command.lower()
    if not command:
        errors.append("`command` must be non-empty.")
    if re.search(r"(^|[;&|]\s*)cd\s+project\b", command):
        errors.append("`command` must run from the workspace root; do not `cd project`.")
    if output_dir and output_dir not in command:
        errors.append("`command` should include the declared `output_dir` for metrics/log/checkpoint outputs.")
    if disabled_components and "--disable" not in lowered_command and "disable" not in lowered_command:
        errors.append(
            "conditions with disabled components must make the component toggle visible in `command` or config arguments."
        )

    run_level = str(payload.get("run_level") or "").strip().lower()
    if run_level != "full":
        errors.append("formal science conditions must set `run_level` exactly to `full`.")
    for field in ("setup_rationale", "runtime_probe_summary"):
        if _is_placeholder_text(payload.get(field)):
            errors.append(f"`{field}` must explain the runtime setup choice, not be empty or placeholder.")
    source_basis = payload.get("source_basis")
    errors.extend(
        validate_science_source_basis(
            source_basis,
            workspace_root=normalized_workspace,
        )
    )
    errors.extend(
        _validate_required_mapping(
            payload=payload,
            field="training_protocol",
            required_keys=SCIENCE_TRAINING_PROTOCOL_FIELDS,
        )
    )
    errors.extend(
        _validate_required_mapping(
            payload=payload,
            field="evaluation_protocol",
            required_keys=SCIENCE_EVALUATION_PROTOCOL_FIELDS,
        )
    )
    evaluation_protocol = payload.get("evaluation_protocol")
    if isinstance(evaluation_protocol, Mapping):
        protocol_reference = str(evaluation_protocol.get("reference_condition_id") or "").strip()
        expected_reference = reference
        if protocol_reference != expected_reference:
            errors.append(
                "`evaluation_protocol.reference_condition_id` must match the condition `reference_condition_id` "
                f"(expected `{expected_reference}`, got `{protocol_reference}`)."
            )
    errors.extend(
        _validate_science_protocol_command_consistency(
            payload=payload,
            command=command,
        )
    )

    for field in ("train_dataset_binding", "component_set_description", "result_interpretation_rule", "input_paths"):
        if not payload.get(field):
            errors.append(f"`{field}` must be non-empty.")
    for field in ("evaluation_dataset_bindings", "metric_bindings", "raw_evidence", "artifact_ids"):
        value = payload.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"`{field}` must be a non-empty list.")

    pass_condition = str(payload.get("pass_condition") or "").strip()
    if _is_placeholder_text(pass_condition):
        errors.append("`pass_condition` must be concrete, not a placeholder.")

    return errors


def validate_science_condition_plan(
    payload: Mapping[str, Any],
    *,
    project_dir: str,
    workspace_root: str,
) -> list[str]:
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        return ["science_plan must contain a non-empty top-level `stages` list."]
    canonical_components = _load_canonical_component_names(workspace_root)
    canonical_set = set(canonical_components)
    issues: list[str] = []
    seen_ids: set[str] = set()
    reference_ids: set[str] = set()
    reference_conditions: list[tuple[int, str]] = []
    disabled_counts: dict[str, int] = {}

    for index, step in enumerate(stages, start=1):
        if not isinstance(step, Mapping):
            issues.append(f"condition {index}: condition must be an object")
            continue
        condition_id = _condition_id(step)
        if condition_id in seen_ids:
            issues.append(f"condition {index}: duplicate condition_id `{condition_id}`")
        if condition_id:
            seen_ids.add(condition_id)
        step_issues = validate_science_condition_step_fields(
            step,
            project_dir=project_dir,
            workspace_root=workspace_root,
            known_reference_ids=reference_ids,
        )
        issues.extend(f"condition {index}: {message}" for message in step_issues)
        disabled_components = _string_list(step.get("disabled_components"))
        enabled_components = _string_list(step.get("enabled_components"))
        if not disabled_components and canonical_set and set(enabled_components) == canonical_set and condition_id:
            reference_ids.add(condition_id)
            reference_conditions.append((index, condition_id))
        for component in disabled_components:
            disabled_counts[component] = disabled_counts.get(component, 0) + 1

    if canonical_components:
        expected_stage_count = len(canonical_components) + 1
        expected_pattern = (
            "Expected science plan pattern: stage 1 is the only all-components reference "
            f"with enabled_components={canonical_components}, disabled_components=[]; "
            "then one component-disabled full condition for each canonical component, disabling exactly one of: "
            + ", ".join(canonical_components)
            + "."
        )
        if len(stages) != expected_stage_count:
            issues.append(
                "science_plan must contain exactly 1 + len(idea.json.components) conditions "
                f"({expected_stage_count} total): got {len(stages)}. {expected_pattern}"
            )
        if not reference_conditions:
            issues.append(
                "science_plan must include exactly one all-components reference condition as stage 1. "
                + expected_pattern
            )
        else:
            if reference_conditions[0][0] != 1:
                issues.append(
                    "science_plan stage 1 must be the all-components reference condition. "
                    + expected_pattern
                )
            if len(reference_conditions) != 1:
                issues.append(
                    "science_plan must include exactly one all-components reference condition, "
                    f"got {len(reference_conditions)}. {expected_pattern}"
                )
        missing = [component for component in canonical_components if disabled_counts.get(component, 0) == 0]
        repeated = [component for component in canonical_components if disabled_counts.get(component, 0) > 1]
        extra = sorted(set(disabled_counts) - canonical_set)
        if missing or repeated or extra:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if repeated:
                details.append("repeated=" + ",".join(repeated))
            if extra:
                details.append("extra=" + ",".join(extra))
            issues.append(
                "science_plan must include exactly one component-disabled condition per idea component: "
                + "; ".join(details)
                + ". "
                + expected_pattern
            )
    elif disabled_counts:
        issues.append(
            "science_plan must not include component-disabled conditions unless idea.json.components are available."
        )
    return issues
