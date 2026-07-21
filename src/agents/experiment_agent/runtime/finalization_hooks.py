"""Experiment finalization hooks."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.runtime.ablation_results import (
    FINAL_COMPONENT_RESULTS,
    write_ablation_results_artifacts,
)
from src.agents.experiment_agent.runtime.artifacts import (
    build_step_artifact_registry,
    record_runtime_artifact,
    validate_artifact_contract,
)
from src.agents.experiment_agent.runtime.contracts import validate_science_evidence_payload
from src.agents.experiment_agent.runtime.idea_components import canonical_component_names
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    write_json_file,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.report_layout import artifact_rel, step_report_abs_paths


def _config_lookup(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for part in str(path or "").split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            try:
                current = current.get(part)
            except Exception:
                current = None
        else:
            current = getattr(current, part, None)
    return default if current is None else current


def _configured_symbolic_memory_path(config: Any) -> str:
    for path in (
        "pipeline.symbolic_memory_path",
        "flow.memory.symbolic_memory_path",
        "symbolic_memory_path",
    ):
        value = str(_config_lookup(config, path, "") or "").strip()
        if value:
            return value
    return "idea_skill_priors"


def _resolve_symbolic_memory_path(config: Any, *, workspace_root: str = "") -> str:
    symbolic_memory_path = os.environ.get("XCIENTIST_SYMBOLIC_MEMORY_PATH", "").strip()
    configured_symbolic_path = symbolic_memory_path or _configured_symbolic_memory_path(config)
    configured_symbolic_path = os.path.expanduser(str(configured_symbolic_path or "idea_skill_priors"))
    if os.path.isabs(configured_symbolic_path):
        return os.path.abspath(configured_symbolic_path)
    root = str(workspace_root or os.environ.get("XCIENTIST_WORKSPACE_ROOT", "")).strip()
    if root:
        return os.path.abspath(os.path.join(root, configured_symbolic_path))
    workspace_cfg_root = str(_config_lookup(config, "workspace.root", "") or "").strip()
    if workspace_cfg_root:
        return os.path.abspath(os.path.join(workspace_cfg_root, configured_symbolic_path))
    return os.path.abspath(configured_symbolic_path)


def _finalization_repair_instructions(error: str) -> str:
    return (
        "Finalization produced valid final ablation artifacts, but symbolic-memory writeback failed. "
        "Do not delete or rewrite `agent_reports/ablation/final/ablation_results.json`. "
        "Repair the symbolic-memory configuration or converter, then rerun finalization. "
        "The converter must read pipeline-scoped config keys such as "
        "`pipeline.default_macro_roles` and `pipeline.symbolic_memory_path` from the unified config. "
        f"Observed error: {error}"
    )


def _normalize_component_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _validate_symbolic_writeback(
    *,
    ablation_path: str,
    symbolic_memory_path: str,
    records: Any,
) -> Dict[str, Any]:
    ablation_payload = load_json_file(ablation_path)
    components = ablation_payload.get("components") if isinstance(ablation_payload, dict) else {}
    component_names = [
        str(name)
        for name in (components or {}).keys()
        if str(name or "").strip()
    ] if isinstance(components, dict) else []
    record_list = records if isinstance(records, list) else []
    errors = []
    if not component_names:
        errors.append("`ablation_results.json` has no component entries to write back.")
    if not symbolic_memory_path:
        errors.append("`symbolic_memory_path` is empty.")
    if not isinstance(records, list):
        errors.append("converter must return a list of symbolic records.")
    if len(record_list) < len(component_names):
        errors.append(
            "converter returned fewer records than ablated components "
            f"({len(record_list)} < {len(component_names)})."
        )

    memory_file = os.path.join(symbolic_memory_path, "symbolic_memory.json") if symbolic_memory_path else ""
    memory_payload = load_json_file(memory_file) if memory_file else None
    if not memory_file or not os.path.exists(memory_file):
        errors.append(f"symbolic memory file was not written at `{memory_file}`.")
    elif not isinstance(memory_payload, dict):
        errors.append(f"symbolic memory file is not a JSON object: `{memory_file}`.")
    else:
        memory_records = memory_payload.get("records")
        if not isinstance(memory_records, dict):
            errors.append("symbolic memory file missing object field `records`.")
        else:
            memory_components = {
                _normalize_component_name(record.get("component"))
                for record in memory_records.values()
                if isinstance(record, dict)
            }
            missing_components = [
                name
                for name in component_names
                if _normalize_component_name(name) not in memory_components
            ]
            if missing_components:
                errors.append(
                    "symbolic memory file does not contain records for components: "
                    + ", ".join(missing_components)
                )

    return {
        "valid": not errors,
        "errors": errors,
        "component_count": len(component_names),
        "components": component_names,
        "symbolic_memory_file_path": memory_file,
    }


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _component_result_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "result": str(payload.get("result") or "").strip().lower(),
        "metric": str(payload.get("metric") or "").strip(),
        "value": str(payload.get("value") or "").strip(),
        "confidence": payload.get("confidence"),
        "analysis": str(payload.get("analysis") or "").strip(),
        "method_context": str(payload.get("method_context") or "").strip(),
        "follow_up_required": bool(payload.get("follow_up_required")),
    }


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _load_object(path: str, label: str, issues: List[str], checked: List[str]) -> Dict[str, Any]:
    checked.append(path)
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        issues.append(f"{label} is missing or is not a JSON object: `{path}`.")
        return {}
    return payload


def _condition_id(step: Dict[str, Any]) -> str:
    return str(step.get("condition_id") or step.get("step_id") or step.get("stage_id") or "").strip()


def verify_final_science_lineage(
    *,
    workspace_root: str,
    project_root: str,
) -> Dict[str, Any]:
    """Verify finalization can trace science phase output back to step hooks.

    Final materialization intentionally trusts the aggregated
    `agent_reports/science/phase.json` for final values, but this verifier
    ensures that phase report is backed by the per-condition prefinish hooks and
    managed evidence artifacts from the same workspace.
    """
    paths = artifact_paths(workspace_root, project_root)
    issues: List[str] = []
    checked: List[str] = []
    canonical: List[str] = []
    try:
        canonical = canonical_component_names(workspace_root)
    except Exception as exc:
        issues.append(f"Could not load canonical idea components from `idea.json`: {exc}.")

    plan_payload = _load_object(paths["science_executable_plan"], "Science executable plan", issues, checked)
    phase_payload = _load_object(paths["science_reviewer"], "Science phase report", issues, checked)
    phase_report = normalize_phase_report(phase_payload)
    if phase_report["status"] != "PASS":
        issues.append(f"Science phase report status is `{phase_report['status']}`, not PASS.")
    if phase_report["phase_completion_status"] != "complete":
        issues.append(
            "Science phase report `phase_completion_status` must be `complete`, "
            f"got `{phase_report['phase_completion_status']}`."
        )
    if phase_report["ready_for_next_phase"] is not True:
        issues.append("Science phase report `ready_for_next_phase` must be true.")

    phase_components = phase_payload.get("science_component_results") if isinstance(phase_payload, dict) else {}
    if not isinstance(phase_components, dict):
        phase_components = {}
        issues.append("Science phase report must contain object `science_component_results`.")
    if canonical:
        phase_keys = [str(item) for item in phase_components.keys()]
        missing = [name for name in canonical if name not in phase_components]
        extra = sorted(set(phase_keys) - set(canonical))
        if missing:
            issues.append(
                "`agent_reports/science/phase.json` is missing component conclusions for: "
                + ", ".join(missing)
            )
        if extra:
            issues.append(
                "`agent_reports/science/phase.json` contains non-canonical component conclusions: "
                + ", ".join(extra)
            )

    steps = plan_payload.get("stages") if isinstance(plan_payload, dict) else None
    if not isinstance(steps, list) or not steps:
        steps = []
        issues.append("Science executable plan must contain a non-empty top-level `stages` list.")

    reference_conditions: List[str] = []
    disabled_by_component: Dict[str, str] = {}
    seen_conditions: set[str] = set()
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            issues.append(f"Science plan stages[{index}] must be a JSON object.")
            continue
        condition_id = _condition_id(raw_step)
        if not condition_id:
            issues.append(f"Science plan stages[{index}] is missing `condition_id`.")
            continue
        if condition_id in seen_conditions:
            issues.append(f"Science plan has duplicate condition_id `{condition_id}`.")
        seen_conditions.add(condition_id)
        enabled = _string_list(raw_step.get("enabled_components"))
        disabled = _string_list(raw_step.get("disabled_components"))
        if not disabled:
            reference_conditions.append(condition_id)
            if canonical and enabled != canonical:
                issues.append(
                    f"Reference condition `{condition_id}` enabled_components must match "
                    f"`idea.json.components` exactly: expected {canonical}, got {enabled}."
                )
            continue
        if len(disabled) != 1:
            issues.append(
                f"Science condition `{condition_id}` disables {len(disabled)} components. "
                "Finalization expects one component-disabled condition per idea component."
            )
            continue
        component = disabled[0]
        if canonical:
            expected_enabled = [name for name in canonical if name != component]
            if enabled != expected_enabled:
                issues.append(
                    f"Component-disabled condition `{condition_id}` enabled_components must equal "
                    f"all canonical components except `{component}`: expected {expected_enabled}, got {enabled}."
                )
        if component in disabled_by_component:
            issues.append(
                f"Science plan has multiple disabled-component conditions for `{component}`: "
                f"`{disabled_by_component[component]}` and `{condition_id}`."
            )
        disabled_by_component[component] = condition_id

    if len(reference_conditions) != 1:
        issues.append(
            "Science plan must contain exactly one all-components reference condition; "
            f"found {len(reference_conditions)} ({', '.join(reference_conditions) or 'none'})."
        )
    if canonical:
        missing_disabled = [name for name in canonical if name not in disabled_by_component]
        extra_disabled = sorted(set(disabled_by_component) - set(canonical))
        if missing_disabled:
            issues.append(
                "Science plan is missing one component-disabled condition for: "
                + ", ".join(missing_disabled)
            )
        if extra_disabled:
            issues.append(
                "Science plan disables non-canonical components: "
                + ", ".join(extra_disabled)
            )

    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        condition_id = _condition_id(raw_step)
        if not condition_id:
            continue
        disabled = _string_list(raw_step.get("disabled_components"))
        step_paths = step_report_abs_paths(workspace_root, "science", condition_id)
        hook_payload = _load_object(
            step_paths["hook_report_path"],
            f"Science hook report for `{condition_id}`",
            issues,
            checked,
        )
        review_payload = _load_object(
            step_paths["review_report_path"],
            f"Science review report for `{condition_id}`",
            issues,
            checked,
        )
        if hook_payload:
            if hook_payload.get("status") != "PASS":
                issues.append(f"Science hook report for `{condition_id}` status is not PASS.")
            if hook_payload.get("review_status") != "PASS":
                issues.append(f"Science hook report for `{condition_id}` review_status is not PASS.")
            if hook_payload.get("returned_to_worker") is not False:
                issues.append(
                    f"Science hook report for `{condition_id}` must have `returned_to_worker: false`."
                )
            prefinish_contract = hook_payload.get("prefinish_contract")
            if not isinstance(prefinish_contract, dict) or prefinish_contract.get("status") != "PASS":
                issues.append(
                    f"Science hook report for `{condition_id}` is missing PASS `prefinish_contract`."
                )
            review_matrix = hook_payload.get("review_matrix")
            if not isinstance(review_matrix, dict) or review_matrix.get("status") != "PASS":
                issues.append(
                    f"Science hook report for `{condition_id}` is missing PASS `review_matrix`."
                )
        if review_payload:
            if review_payload.get("status") != "PASS":
                issues.append(f"Science review report for `{condition_id}` status is not PASS.")
            contract_payload = review_payload.get("prefinish_contract")
            if not isinstance(contract_payload, dict) or contract_payload.get("status") != "PASS":
                issues.append(
                    f"Science review report for `{condition_id}` is missing PASS `prefinish_contract`."
                )

        evidence_rel = artifact_rel("science", condition_id, "evidence.json")
        evidence_path = os.path.join(os.path.realpath(workspace_root), evidence_rel)
        evidence_payload = _load_object(
            evidence_path,
            f"Science evidence manifest for `{condition_id}`",
            issues,
            checked,
        )
        if evidence_payload:
            issues.extend(
                validate_science_evidence_payload(
                    evidence_payload,
                    workspace_root=workspace_root,
                    step=raw_step,
                )
            )
        registry = build_step_artifact_registry(
            workspace_root=workspace_root,
            scope="science",
            step=dict(raw_step),
        )
        artifact_contract = validate_artifact_contract(registry=registry, review_status="PASS")
        checked.extend(
            str(item.get("path") or "")
            for item in artifact_contract.get("checked_artifacts") or []
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        )
        if artifact_contract.get("status") != "PASS":
            issues.extend(str(item) for item in artifact_contract.get("issues") or [])

        for component in disabled:
            component_payload = phase_components.get(component)
            if not isinstance(component_payload, dict):
                continue
            if str(component_payload.get("condition_id") or "").strip() != condition_id:
                issues.append(
                    f"Science phase component `{component}` must point to condition `{condition_id}` "
                    f"via `condition_id`."
                )
            if str(component_payload.get("result") or "").strip().lower() not in FINAL_COMPONENT_RESULTS:
                issues.append(
                    f"Science phase component `{component}` has invalid result "
                    f"`{component_payload.get('result')}`."
                )
            if component_payload.get("follow_up_required") is not False:
                issues.append(
                    f"Science phase component `{component}` has `follow_up_required` not false."
                )
            if review_payload:
                review_fields = _component_result_fields(review_payload)
                phase_fields = _component_result_fields(component_payload)
                for field in ("result", "metric", "value", "analysis", "follow_up_required"):
                    if review_fields.get(field) != phase_fields.get(field):
                        issues.append(
                            f"Science phase component `{component}` field `{field}` does not match "
                            f"the reviewer-approved condition report `{condition_id}`."
                        )
                if not _numeric_equal(review_fields.get("confidence"), phase_fields.get("confidence")):
                    issues.append(
                        f"Science phase component `{component}` field `confidence` does not match "
                        f"the reviewer-approved condition report `{condition_id}`."
                    )

    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "final_science_lineage",
        "issues": issues,
        "checked_artifacts": sorted(set(item for item in checked if item)),
        "canonical_components": canonical,
        "reference_conditions": reference_conditions,
        "disabled_by_component": disabled_by_component,
    }


def _science_lineage_repair_instructions(lineage: Dict[str, Any]) -> str:
    issues = [str(item) for item in lineage.get("issues") or [] if str(item).strip()]
    return (
        "Finalization cannot proceed because the science phase summary is not backed by "
        "a complete per-condition hook/evidence lineage. Do not edit final ablation outputs "
        "or symbolic-memory receipts by hand. Repair upstream science and rerun the science "
        "phase so every canonical idea component has one reviewer-approved component-disabled "
        "full run, each condition has PASS hook/review reports, and each managed evidence "
        "manifest is written through artifact tools with a matching ledger sha256.\n\n"
        "Lineage issues:\n"
        + "\n".join(f"- {issue}" for issue in issues)
    )


def run_final_science_prefinish_hooks(
    *,
    workspace_root: str,
    project_root: str,
    experiment_id: str,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Materialize final science artifacts and feed results into symbolic memory.

    This hook runs only after all phase-level reviewer gates have passed. It
    uses deterministic runtime contracts for `ablation_results.json` before
    writing symbolic memory, then records a receipt consumed by the pipeline.
    """
    paths = artifact_paths(workspace_root, project_root)
    lineage = verify_final_science_lineage(
        workspace_root=workspace_root,
        project_root=project_root,
    )
    if lineage.get("status") != "PASS":
        receipt = {
            "status": "FAIL",
            "hook": "final_science_prefinish",
            "experiment_id": experiment_id,
            "ablation_results_path": "",
            "symbolic_memory_path": "",
            "records_created": 0,
            "record_ids": [],
            "blocker": "science lineage verification failed",
            "repair_instructions": _science_lineage_repair_instructions(lineage),
            "science_lineage": lineage,
        }
        write_json_file(paths["symbolic_memory_receipt"], receipt)
        record_runtime_artifact(
            workspace_root=workspace_root,
            artifact_id="runtime.symbolic_memory_receipt",
            path=paths["symbolic_memory_receipt"],
            stage="finalization",
            step_id="symbolic_memory",
            extra={"status": "FAIL", "experiment_id": experiment_id},
        )
        return receipt

    materialized = write_ablation_results_artifacts(
        workspace_root,
        project_root,
        generated_by="prefinish_hook",
    )
    if not materialized.get("valid"):
        receipt = {
            "status": "FAIL",
            "hook": "final_science_prefinish",
            "experiment_id": experiment_id,
            "ablation_results_path": "",
            "symbolic_memory_path": "",
            "records_created": 0,
            "record_ids": [],
            "blocker": materialized.get("blocker") or "ablation_results materialization failed",
            "materialization": materialized,
        }
        write_json_file(paths["symbolic_memory_receipt"], receipt)
        record_runtime_artifact(
            workspace_root=workspace_root,
            artifact_id="runtime.symbolic_memory_receipt",
            path=paths["symbolic_memory_receipt"],
            stage="finalization",
            step_id="symbolic_memory",
            extra={"status": "FAIL", "experiment_id": experiment_id},
        )
        return receipt

    ablation_path = str(materialized.get("ablation_results_path") or paths["ablation_results"])
    try:
        from src.config import load_config
        from src.pipeline.experiment_to_symbolic import convert_ablation_to_symbolic_memory

        cfg = config or load_config()
        symbolic_memory_path = _resolve_symbolic_memory_path(cfg, workspace_root=workspace_root)
        records = convert_ablation_to_symbolic_memory(
            ablation_path=ablation_path,
            experiment_id=experiment_id,
            symbolic_memory_path=symbolic_memory_path,
            config=cfg,
        )
        writeback_validation = _validate_symbolic_writeback(
            ablation_path=ablation_path,
            symbolic_memory_path=symbolic_memory_path,
            records=records,
        )
        if not writeback_validation["valid"]:
            raise RuntimeError(
                "symbolic-memory writeback validation failed: "
                + "; ".join(writeback_validation["errors"])
            )
        receipt = {
            "status": "PASS",
            "hook": "final_science_prefinish",
            "experiment_id": experiment_id,
            "ablation_results_path": ablation_path,
            "symbolic_memory_path": symbolic_memory_path,
            "symbolic_memory_file_path": writeback_validation["symbolic_memory_file_path"],
            "component_count": writeback_validation["component_count"],
            "records_created": len(records),
            "record_ids": [
                str(record.get("id") or record.get("record_id") or record.get("component") or "")
                for record in records
            ],
            "blocker": "",
            "writeback_validation": writeback_validation,
            "science_lineage": lineage,
            "materialization": {
                "materialization_report_path": materialized.get("materialization_report_path"),
                "ablation_results_manifest_path": materialized.get("ablation_results_manifest_path"),
            },
        }
        write_json_file(paths["symbolic_memory_receipt"], receipt)
        record_runtime_artifact(
            workspace_root=workspace_root,
            artifact_id="runtime.symbolic_memory_receipt",
            path=paths["symbolic_memory_receipt"],
            stage="finalization",
            step_id="symbolic_memory",
            extra={"status": "PASS", "experiment_id": experiment_id},
        )
        return receipt
    except Exception as exc:
        try:
            from src.config import load_config

            cfg = config or load_config()
            symbolic_memory_path = _resolve_symbolic_memory_path(cfg, workspace_root=workspace_root)
        except Exception:
            symbolic_memory_path = ""
        error = str(exc)
        receipt = {
            "status": "FAIL",
            "hook": "final_science_prefinish",
            "experiment_id": experiment_id,
            "ablation_results_path": ablation_path,
            "symbolic_memory_path": symbolic_memory_path,
            "symbolic_memory_file_path": (
                os.path.join(symbolic_memory_path, "symbolic_memory.json")
                if symbolic_memory_path
                else ""
            ),
            "records_created": 0,
            "record_ids": [],
            "blocker": error,
            "repair_instructions": _finalization_repair_instructions(error),
            "science_lineage": lineage,
            "materialization": materialized,
        }
        write_json_file(paths["symbolic_memory_receipt"], receipt)
        record_runtime_artifact(
            workspace_root=workspace_root,
            artifact_id="runtime.symbolic_memory_receipt",
            path=paths["symbolic_memory_receipt"],
            stage="finalization",
            step_id="symbolic_memory",
            extra={"status": "FAIL", "experiment_id": experiment_id},
        )
        return receipt


__all__ = ["run_final_science_prefinish_hooks", "verify_final_science_lineage"]
