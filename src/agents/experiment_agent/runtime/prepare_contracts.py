"""Deterministic prepare-stage artifact checks."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping

from src.agents.experiment_agent.runtime.idea_components import IDEA_COMPONENTS_HEADING
from src.agents.experiment_agent.runtime.manifests import load_json_file


PREPARE_STAGE_ORDER = ["repos", "dataset", "model", "env", "synthesis"]

PREPARE_ARTIFACT_IDS = {
    "repos": ["prepare.discovery", "prepare.repos"],
    "dataset": ["prepare.dataset"],
    "model": ["prepare.model"],
    "env": ["prepare.env"],
    "synthesis": ["prepare.idea", "prepare.target_inventory"],
}

PREPARE_REQUIRED_ARTIFACTS = {
    "repos": ["agent_reports/prepare/artifacts/discovery.json", "agent_reports/prepare/artifacts/repos.json"],
    "dataset": ["agent_reports/prepare/artifacts/dataset.json"],
    "model": ["agent_reports/prepare/artifacts/model.json"],
    "env": ["agent_reports/prepare/artifacts/env.json"],
    "synthesis": [
        "agent_reports/prepare/artifacts/idea.md",
        "agent_reports/prepare/artifacts/target_inventory.json",
    ],
}

PREPARE_READY = "READY"
PREPARE_BLOCKED = "BLOCKED"
PREPARE_STATUSES = {PREPARE_READY, PREPARE_BLOCKED}

_PATH_KEYS = {
    "evidence_path",
    "evidence_paths",
    "expected_file",
    "expected_files",
    "expected_path",
    "expected_paths",
    "file",
    "files",
    "local_checkpoint",
    "local_checkpoints",
    "local_path",
    "local_paths",
    "loader_probe_log",
    "probe_log",
    "probe_logs",
    "repo_path",
    "repo_paths",
    "schema_probe_log",
    "selected_path",
    "selected_paths",
    "smoke_log",
    "smoke_logs",
}

_BLOCKER_REQUIRED_KEYS = (
    "reason",
    "attempted_queries",
    "rejected_candidates",
    "missing_requirements",
    "user_action_required",
    "evidence_paths",
)


def _path_under(root: str, path: str) -> bool:
    root = os.path.realpath(root)
    path = os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


def _resolve_workspace_path(workspace_root: str, path: str) -> str:
    path = str(path or "").strip()
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.realpath(workspace_root), path)


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and any(str(item).strip() for item in value)


def _is_nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _has_any_key(payload: Mapping[str, Any], keys: Iterable[str]) -> bool:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _require_object(payload: Mapping[str, Any], field: str, issues: List[str], *, rel_path: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping) or not value:
        issues.append(f"`{rel_path}.{field}` must be a non-empty object.")
        return {}
    return value


def _require_list(
    payload: Mapping[str, Any],
    field: str,
    issues: List[str],
    *,
    rel_path: str,
    allow_empty: bool = False,
) -> List[Any]:
    value = payload.get(field)
    if not isinstance(value, list) or (not allow_empty and not value):
        requirement = "a list" if allow_empty else "a non-empty list"
        issues.append(f"`{rel_path}.{field}` must be {requirement}.")
        return []
    return value


def _validate_discovery_candidate_table(
    payload: Mapping[str, Any],
    *,
    rel_path: str,
    issues: List[str],
) -> None:
    table = _require_object(payload, "candidate_table", issues, rel_path=rel_path)
    if not table:
        return
    required_groups = ("repos", "datasets", "models")
    for group in required_groups:
        candidates = table.get(group)
        if not isinstance(candidates, list) or not candidates:
            issues.append(f"`{rel_path}.candidate_table.{group}` must be a non-empty candidate list.")
            continue
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                issues.append(f"`{rel_path}.candidate_table.{group}[{index}]` must be an object.")
                continue
            for key in ("candidate_id", "name", "source", "task_fit", "decision", "reason"):
                if not str(candidate.get(key) or "").strip():
                    issues.append(
                        f"`{rel_path}.candidate_table.{group}[{index}].{key}` must be non-empty."
                    )
            decision = str(candidate.get("decision") or "").strip().lower()
            if decision not in {"selected", "rejected"}:
                issues.append(
                    f"`{rel_path}.candidate_table.{group}[{index}].decision` must be `selected` or `rejected`."
                )

    selected_candidate_ids = _require_object(payload, "selected_candidate_ids", issues, rel_path=rel_path)
    if selected_candidate_ids:
        for group in required_groups:
            value = selected_candidate_ids.get(group)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"`{rel_path}.selected_candidate_ids.{group}` must name the selected candidate id.")
    rejected_candidates = _require_list(payload, "rejected_candidates", issues, rel_path=rel_path)
    for index, item in enumerate(rejected_candidates):
        if not isinstance(item, Mapping):
            issues.append(f"`{rel_path}.rejected_candidates[{index}]` must be an object.")
            continue
        for key in ("candidate_id", "resource_type", "reason"):
            if not str(item.get(key) or "").strip():
                issues.append(f"`{rel_path}.rejected_candidates[{index}].{key}` must be non-empty.")


def _status(payload: Mapping[str, Any], *, rel_path: str, issues: List[str]) -> str:
    status = str(payload.get("status") or "").strip().upper()
    if status not in PREPARE_STATUSES:
        issues.append(f"`{rel_path}` must declare status `READY` or `BLOCKED`.")
        return ""
    return status


def _iter_key_values(payload: Any, keys: Iterable[str]) -> Iterator[tuple[str, Any]]:
    wanted = set(keys)
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            if key_text in wanted:
                yield key_text, value
            yield from _iter_key_values(value, wanted)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_key_values(item, wanted)


def _path_strings_from_value(value: Any) -> List[str]:
    paths: List[str] = []
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                paths.append(item.strip())
            elif isinstance(item, Mapping):
                for _key, nested in _iter_key_values(item, _PATH_KEYS):
                    paths.extend(_path_strings_from_value(nested))
        return paths
    if isinstance(value, Mapping):
        for _key, nested in _iter_key_values(value, _PATH_KEYS):
            paths.extend(_path_strings_from_value(nested))
    return paths


def _check_declared_paths(
    payload: Mapping[str, Any],
    *,
    workspace_root: str,
    keys: Iterable[str],
    issues: List[str],
    require_exists: bool = True,
) -> None:
    for key, value in _iter_key_values(payload, keys):
        values = _path_strings_from_value(value)
        for raw_path in values:
            path = _resolve_workspace_path(workspace_root, raw_path)
            if not _path_under(workspace_root, path):
                issues.append(f"`{key}` path escapes workspace: `{raw_path}`.")
                continue
            if require_exists and not os.path.exists(path):
                issues.append(f"`{key}` declared path does not exist: `{raw_path}`.")


def _check_all_declared_paths(
    payload: Mapping[str, Any],
    *,
    workspace_root: str,
    issues: List[str],
) -> None:
    _check_declared_paths(
        payload,
        workspace_root=workspace_root,
        keys=_PATH_KEYS,
        issues=issues,
    )


def _validate_blocker_payload(
    payload: Mapping[str, Any],
    *,
    rel_path: str,
    workspace_root: str,
    issues: List[str],
) -> None:
    blocker = payload.get("blocker")
    if not isinstance(blocker, Mapping) or not blocker:
        issues.append(f"`{rel_path}` status BLOCKED requires a non-empty `blocker` object.")
        return
    for key in _BLOCKER_REQUIRED_KEYS:
        if key not in blocker:
            issues.append(f"`{rel_path}.blocker` missing required `{key}`.")
    if not str(blocker.get("reason") or "").strip():
        issues.append(f"`{rel_path}.blocker.reason` must be non-empty.")
    for key in ("attempted_queries", "rejected_candidates", "missing_requirements"):
        value = blocker.get(key)
        if not isinstance(value, list) or not value:
            issues.append(f"`{rel_path}.blocker.{key}` must be a non-empty list.")
    if not str(blocker.get("user_action_required") or "").strip():
        issues.append(f"`{rel_path}.blocker.user_action_required` must be non-empty.")
    evidence_paths = blocker.get("evidence_paths")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        issues.append(f"`{rel_path}.blocker.evidence_paths` must be a non-empty list of local evidence paths.")
    else:
        _check_declared_paths(
            blocker,
            workspace_root=workspace_root,
            keys=("evidence_paths",),
            issues=issues,
        )


def _validate_common_ready_payload(
    payload: Mapping[str, Any],
    *,
    rel_path: str,
    workspace_root: str,
    issues: List[str],
) -> None:
    if not str(payload.get("selection_rationale") or payload.get("rationale") or "").strip():
        issues.append(f"`{rel_path}` READY payload must record selection rationale.")
    _check_all_declared_paths(payload, workspace_root=workspace_root, issues=issues)


def validate_prepare_artifact_payload(
    schema_name: str,
    payload: Any,
    *,
    workspace_root: str,
) -> List[str]:
    if not isinstance(payload, Mapping):
        return [f"{schema_name} must be a JSON object."]
    issues: List[str] = []
    rel_path = f"{schema_name}.json"
    status = _status(payload, rel_path=rel_path, issues=issues)
    if status == PREPARE_BLOCKED:
        _validate_blocker_payload(
            payload,
            rel_path=rel_path,
            workspace_root=workspace_root,
            issues=issues,
        )
        return issues
    if status != PREPARE_READY:
        return issues

    if schema_name == "prepare_discovery":
        _require_object(payload, "task_signature", issues, rel_path="discovery.json")
        _require_object(payload, "resource_requirements", issues, rel_path="discovery.json")
        mcp_status = _require_object(payload, "mcp_status_snapshot", issues, rel_path="discovery.json")
        if mcp_status and "connected" not in mcp_status:
            issues.append("`discovery.json.mcp_status_snapshot.connected` must record whether Tavily/MCP was connected.")
        queries = _require_list(payload, "queries", issues, rel_path="discovery.json")
        for index, query in enumerate(queries):
            if isinstance(query, Mapping):
                if not str(query.get("query") or "").strip():
                    issues.append(f"`discovery.json.queries[{index}].query` must be non-empty.")
                if not str(query.get("purpose") or "").strip():
                    issues.append(f"`discovery.json.queries[{index}].purpose` must be non-empty.")
            elif not str(query or "").strip():
                issues.append(f"`discovery.json.queries[{index}]` must be non-empty.")
        _require_object(payload, "selection_criteria", issues, rel_path="discovery.json")
        _validate_discovery_candidate_table(payload, rel_path="discovery.json", issues=issues)
        evidence_gaps = _require_list(payload, "evidence_gaps", issues, rel_path="discovery.json", allow_empty=True)
        for index, item in enumerate(evidence_gaps):
            if not isinstance(item, (str, Mapping)):
                issues.append(f"`discovery.json.evidence_gaps[{index}]` must be a string or object.")
        if not _has_any_key(payload, ("selected_resources", "selection_rationale")):
            issues.append("`discovery.json` READY payload must record selected resources and rationale.")
        _check_all_declared_paths(payload, workspace_root=workspace_root, issues=issues)
        return issues

    if schema_name == "prepare_repos":
        if not _has_any_key(payload, ("repos", "repositories", "selected_repositories")):
            issues.append("`repos.json` READY payload must record selected repositories.")
        if not _has_any_key(payload, ("commit", "commits", "resolved_commits")):
            issues.append("`repos.json` READY payload must record resolved commit evidence.")
        if not _has_any_key(payload, ("source_url", "source_urls")):
            issues.append("`repos.json` READY payload must record source URLs.")
        if not _has_any_key(payload, ("license", "licenses", "readme", "readme_evidence")):
            issues.append("`repos.json` READY payload must record license/readme evidence.")
        if not _has_any_key(payload, ("reference_entrypoints", "entrypoints")):
            issues.append("`repos.json` READY payload must record concrete reference entrypoints.")
        _validate_common_ready_payload(payload, rel_path="repos.json", workspace_root=workspace_root, issues=issues)
        return issues

    if schema_name == "prepare_dataset":
        if not _has_any_key(payload, ("datasets", "selected_datasets", "dataset_bindings")):
            issues.append("`dataset.json` READY payload must record selected datasets or dataset bindings.")
        if not _has_any_key(payload, ("expected_files", "expected_paths", "files")):
            issues.append("`dataset.json` READY payload must record expected files or paths.")
        if not _has_any_key(payload, ("schema", "schema_probe", "split", "splits", "loader_probe", "probe_log")):
            issues.append("`dataset.json` READY payload must record schema/split details or loader/schema probe evidence.")
        if not _has_any_key(payload, ("checksums", "sizes", "schema_probe", "loader_probe", "probe_log")):
            issues.append("`dataset.json` READY payload must record checksum/size evidence or a loader/schema probe.")
        _validate_common_ready_payload(payload, rel_path="dataset.json", workspace_root=workspace_root, issues=issues)
        return issues

    if schema_name == "prepare_model":
        if not _has_any_key(payload, ("models", "model_targets", "selected_models")):
            issues.append("`model.json` READY payload must record selected models or model targets.")
        if not _has_any_key(payload, ("backend", "backends", "api_models", "local_checkpoints")):
            issues.append("`model.json` READY payload must record whether each model is API-backed or checkpoint-backed.")
        if not _has_any_key(payload, ("api_dry_run", "dry_run", "load_probe", "checkpoint_checksums", "env_vars")):
            issues.append("`model.json` READY payload must record API dry-run, load probe, checkpoint checksum, or env var evidence.")
        _validate_common_ready_payload(payload, rel_path="model.json", workspace_root=workspace_root, issues=issues)
        return issues

    if schema_name == "prepare_env":
        venv_path = str(payload.get("venv_path") or payload.get("python_env") or "project/.venv").strip()
        python_path = str(payload.get("python_path") or os.path.join(venv_path, "bin", "python")).strip()
        python_abs = _resolve_workspace_path(workspace_root, python_path)
        if not _path_under(workspace_root, python_abs):
            issues.append(f"`env.json` python_path escapes workspace: `{python_path}`.")
        elif not os.path.exists(python_abs):
            issues.append(f"`env.json` python_path does not exist: `{python_path}`.")
        if not _has_any_key(payload, ("install_commands", "installed_packages", "package_versions")):
            issues.append("`env.json` READY payload must record install commands or installed package versions.")
        if not _has_any_key(payload, ("import_smoke", "smoke_commands", "smoke_logs", "resource_binding_smoke")):
            issues.append("`env.json` READY payload must record import/resource-binding smoke evidence.")
        _validate_common_ready_payload(payload, rel_path="env.json", workspace_root=workspace_root, issues=issues)
        return issues

    if schema_name == "prepare_target_inventory":
        for key in ("components", "resources", "benchmarks", "metrics"):
            if key not in payload:
                issues.append(f"`target_inventory.json` must contain top-level `{key}`.")
        components = payload.get("components")
        if not isinstance(components, list) or not components:
            issues.append("`target_inventory.json.components` must be a non-empty ordered list.")
        else:
            for index, component in enumerate(components):
                if not isinstance(component, Mapping):
                    issues.append(f"`target_inventory.json.components[{index}]` must be an object.")
                    continue
                if not str(component.get("component") or component.get("name") or "").strip():
                    issues.append(f"`target_inventory.json.components[{index}]` must name the canonical component.")
                targets = component.get("targets")
                if not isinstance(targets, list) or not targets:
                    issues.append(f"`target_inventory.json.components[{index}].targets` must be a non-empty list.")
        resources = payload.get("resources")
        if not isinstance(resources, Mapping) or not resources:
            issues.append("`target_inventory.json.resources` must be a non-empty object.")
        else:
            for key in ("dataset", "model"):
                if key not in resources or not str(resources.get(key) or "").strip():
                    issues.append(f"`target_inventory.json.resources.{key}` must be non-empty.")
        for key in ("benchmarks", "metrics"):
            value = payload.get(key)
            if not isinstance(value, list) or not value:
                issues.append(f"`target_inventory.json.{key}` must be a non-empty list.")
        return issues

    return issues


def _load_prepare_json(workspace_root: str, rel_path: str, issues: List[str]) -> Dict[str, Any]:
    path = _resolve_workspace_path(workspace_root, rel_path)
    if not os.path.isfile(path):
        issues.append(f"Required prepare artifact is missing: `{rel_path}`.")
        return {}
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        issues.append(f"Prepare artifact must be a JSON object: `{rel_path}`.")
        return {}
    return payload


def validate_prepare_plan(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return ["prepare_plan must be a JSON object."]
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        return ["prepare_plan must contain a non-empty top-level `stages` list."]
    planned_order = [
        str(stage.get("stage_id") or "") if isinstance(stage, dict) else ""
        for stage in stages
    ]
    issues: List[str] = []
    if planned_order != PREPARE_STAGE_ORDER:
        issues.append(f"prepare stages must be ordered exactly as {PREPARE_STAGE_ORDER}, got {planned_order}")
    required_fields = {
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
    }
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            issues.append(f"stage {index}: stage must be an object")
            continue
        missing = required_fields - set(stage.keys())
        if missing:
            issues.append(f"stage {index}: missing required keys: {', '.join(sorted(missing))}")
            continue
        stage_id = str(stage.get("stage_id") or "").strip()
        expected_ids = PREPARE_ARTIFACT_IDS.get(stage_id, [])
        actual_ids = [str(item) for item in stage.get("artifact_ids") or [] if str(item).strip()]
        if actual_ids != expected_ids:
            issues.append(f"stage {index} `{stage_id}` artifact_ids must be exactly {expected_ids}, got {actual_ids}")
        if stage.get("repos_policy") != "reference_or_copy":
            issues.append(f"stage {index} `{stage_id}` repos_policy must be exactly `reference_or_copy`.")
        if stage.get("project_must_be_self_contained") is not True:
            issues.append(f"stage {index} `{stage_id}` project_must_be_self_contained must be true.")
        done_condition = str(stage.get("done_condition") or "")
        lowered_done = done_condition.lower()
        if "artifact tool" not in lowered_done and "artifact tools" not in lowered_done:
            issues.append(f"stage {index} `{stage_id}` done_condition must require using Xcientist artifact tools.")
        if "artifact ledger" not in lowered_done:
            issues.append(f"stage {index} `{stage_id}` done_condition must identify the artifact ledger as proof.")
        for rel_path in PREPARE_REQUIRED_ARTIFACTS.get(stage_id, []):
            if rel_path not in done_condition:
                issues.append(f"stage {index} `{stage_id}` done_condition must mention `{rel_path}`.")
        if stage_id == "synthesis":
            if IDEA_COMPONENTS_HEADING not in done_condition:
                issues.append(
                    f"stage {index} `synthesis` done_condition must require `{IDEA_COMPONENTS_HEADING}` in idea.md."
                )
            if "idea.json component" not in lowered_done and "canonical idea component" not in lowered_done:
                issues.append(
                    f"stage {index} `synthesis` done_condition must require target_inventory.json "
                    "to map every idea.json component to concrete implementation targets."
                )
    return issues


def validate_prepare_stage_artifacts(
    *,
    stage: Mapping[str, Any],
    workspace_root: str,
    worker_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    stage_id = str(stage.get("stage_id") or "").strip()
    issues: List[str] = []
    checked: List[str] = []
    statuses: Dict[str, str] = {}
    for rel_path in PREPARE_REQUIRED_ARTIFACTS.get(stage_id, []):
        path = _resolve_workspace_path(workspace_root, rel_path)
        checked.append(path)
        if not os.path.exists(path):
            issues.append(f"Required prepare artifact is missing: `{rel_path}`.")

    if stage_id == "repos":
        discovery = _load_prepare_json(workspace_root, "agent_reports/prepare/artifacts/discovery.json", issues)
        repos = _load_prepare_json(workspace_root, "agent_reports/prepare/artifacts/repos.json", issues)
        if discovery:
            statuses["discovery"] = str(discovery.get("status") or "").strip().upper()
            issues.extend(validate_prepare_artifact_payload("prepare_discovery", discovery, workspace_root=workspace_root))
        if repos:
            statuses["repos"] = str(repos.get("status") or "").strip().upper()
            issues.extend(validate_prepare_artifact_payload("prepare_repos", repos, workspace_root=workspace_root))

    if stage_id == "dataset":
        payload = _load_prepare_json(workspace_root, "agent_reports/prepare/artifacts/dataset.json", issues)
        if payload:
            statuses["dataset"] = str(payload.get("status") or "").strip().upper()
            issues.extend(validate_prepare_artifact_payload("prepare_dataset", payload, workspace_root=workspace_root))

    if stage_id == "model":
        payload = _load_prepare_json(workspace_root, "agent_reports/prepare/artifacts/model.json", issues)
        if payload:
            statuses["model"] = str(payload.get("status") or "").strip().upper()
            issues.extend(validate_prepare_artifact_payload("prepare_model", payload, workspace_root=workspace_root))

    if stage_id == "env":
        payload = _load_prepare_json(workspace_root, "agent_reports/prepare/artifacts/env.json", issues)
        if payload:
            statuses["env"] = str(payload.get("status") or "").strip().upper()
            issues.extend(validate_prepare_artifact_payload("prepare_env", payload, workspace_root=workspace_root))

    if stage_id == "synthesis":
        idea_path = _resolve_workspace_path(workspace_root, "agent_reports/prepare/artifacts/idea.md")
        target_path = _resolve_workspace_path(workspace_root, "agent_reports/prepare/artifacts/target_inventory.json")
        if os.path.isfile(idea_path):
            with open(idea_path, "r", encoding="utf-8") as f:
                idea_text = f.read()
            for heading in (
                "## Idea Summary",
                "## Code Implementation Guidance",
                "## Dataset Usage Guidance",
                "## Environment Variable Usage Guidance",
                "## Resource Acquisition Log",
                "## Repository-to-Dataset Mapping",
                "## Real Experiment Targets",
                IDEA_COMPONENTS_HEADING,
            ):
                if heading not in idea_text:
                    issues.append(f"`idea.md` is missing required heading `{heading}`.")
        if os.path.isfile(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    target_payload = json.load(f)
            except Exception as exc:
                target_payload = {}
                issues.append(f"`target_inventory.json` is not valid JSON: {exc}.")
            if isinstance(target_payload, dict):
                statuses["target_inventory"] = str(target_payload.get("status") or "").strip().upper()
                issues.extend(
                    validate_prepare_artifact_payload(
                        "prepare_target_inventory",
                        target_payload,
                        workspace_root=workspace_root,
                    )
                )
            else:
                issues.append("`target_inventory.json` must be a JSON object.")

    stage_status = PREPARE_BLOCKED if PREPARE_BLOCKED in set(statuses.values()) else PREPARE_READY
    worker_outcome = str((worker_payload or {}).get("outcome") or "").strip().upper()
    if worker_payload is not None:
        if worker_outcome not in PREPARE_STATUSES:
            issues.append("Prepare worker final JSON must declare `outcome` as READY or BLOCKED.")
        elif worker_outcome != stage_status:
            issues.append(
                f"Prepare worker `outcome` must match managed artifact status: expected `{stage_status}`, got `{worker_outcome}`."
            )

    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "prepare_stage_contract",
        "scope": "prepare",
        "stage_id": stage_id,
        "stage_status": stage_status,
        "artifact_statuses": statuses,
        "issues": issues,
        "checked_artifacts": checked,
    }


__all__ = [
    "PREPARE_ARTIFACT_IDS",
    "PREPARE_REQUIRED_ARTIFACTS",
    "PREPARE_STAGE_ORDER",
    "PREPARE_BLOCKED",
    "PREPARE_READY",
    "validate_prepare_artifact_payload",
    "validate_prepare_plan",
    "validate_prepare_stage_artifacts",
]
