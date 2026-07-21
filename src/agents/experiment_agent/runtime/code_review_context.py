"""Code-stage review context and deterministic invariant hooks.

The code prefinish gate uses this module before spawning the full read-only
reviewer matrix. It summarizes the actual workspace surface for reviewers and
keeps checks that can be expressed as code out of free-form reviewer prompts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from src.agents.experiment_agent.agents.code.reviewer import (
    CODE_REVIEWER_IDS,
)
from src.agents.experiment_agent.runtime.artifacts import ArtifactRegistry


HIGH_RISK_TAGS = {
    "data",
    "masking",
    "metrics",
    "model",
    "runner",
    "component_toggles",
}
PROJECT_FILE_LIMIT = 80


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _normalize_project_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/")
    if not value:
        return ""
    if value.startswith("project/"):
        return value
    if value.startswith("/"):
        return value
    if value.startswith("agent_reports/"):
        return value
    return f"project/{value.lstrip('/')}"


def _iter_project_files(project_root: Path) -> Iterable[Path]:
    ignored = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "env",
        "venv",
        ".venv",
        "node_modules",
    }
    if not project_root.exists():
        return
    for current_root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [name for name in dirnames if name not in ignored]
        current = Path(current_root)
        for filename in filenames:
            yield current / filename


def _project_file_summary(workspace_root: Path) -> List[Dict[str, Any]]:
    project_root = workspace_root / "project"
    summaries: List[Dict[str, Any]] = []
    for path in sorted(_iter_project_files(project_root), key=lambda item: item.as_posix()):
        if len(summaries) >= PROJECT_FILE_LIMIT:
            break
        rel = f"project/{_rel(path, project_root)}"
        try:
            stat = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if stat.st_size <= 2_000_000 else ""
        except OSError:
            stat = None
            digest = ""
        summaries.append(
            {
                "path": rel,
                "size": int(stat.st_size) if stat else 0,
                "sha256_16": digest,
            }
        )
    return summaries


def _collect_declared_paths(
    *,
    step: Mapping[str, Any],
    registry: ArtifactRegistry,
) -> Set[str]:
    paths: Set[str] = set()
    for field in ("project_target_paths", "repo_source_paths"):
        for value in step.get(field) or []:
            if isinstance(value, str) and value.strip():
                paths.add(_normalize_project_path(value) if "project" in field else value.strip())
    for item in step.get("code_artifacts") or []:
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            paths.add(_normalize_project_path(str(item.get("path"))))
    for spec in registry.specs.values():
        if spec.path:
            paths.add(str(spec.path))
    return {path for path in paths if path}


def _collect_keywords(step: Mapping[str, Any], worker_payload: Mapping[str, Any]) -> str:
    try:
        step_text = json.dumps(step, ensure_ascii=False, sort_keys=True)
    except TypeError:
        step_text = str(step)
    try:
        worker_text = json.dumps(worker_payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        worker_text = str(worker_payload)
    return f"{step_text}\n{worker_text}".lower()


def _surface_tags_from_text(text: str, declared_paths: Set[str]) -> Set[str]:
    tags: Set[str] = set()
    path_text = "\n".join(sorted(declared_paths)).lower()
    combined = f"{text}\n{path_text}"
    if re.search(r"\b(idea|component|ablation|coverage|innovation|topk|diffusion)\b", combined):
        tags.add("idea")
    if re.search(r"\b(model|module|layer|cell|forward|architecture|gwnet|graphwavenet)\b", combined):
        tags.add("model")
    if re.search(r"\b(data|dataset|loader|dataloader|scaler|normaliz|split|batch)\b", combined):
        tags.add("data")
    if re.search(r"\b(mask|masked|missing|nan|imputation|dropout)\b", combined):
        tags.add("masking")
    if re.search(r"\b(metric|mae|rmse|mape|loss|evaluate|evaluation|calibration)\b", combined):
        tags.add("metrics")
    if re.search(r"\b(run|runner|train|smoke|entrypoint|argparse|command|condition)\b", combined):
        tags.add("runner")
    if re.search(r"\b(disable|enabled|toggle|flag|condition|component_disable_hooks)\b", combined):
        tags.add("component_toggles")
    if re.search(r"\b(path|checkpoint|resource|adj|topk|indices|dataset_candidate|model_candidate)\b", combined):
        tags.add("resources")
    if any(path.startswith("project/") for path in declared_paths):
        tags.add("project_structure")
    return tags


def build_code_review_context(
    *,
    workspace_root: str,
    step: Mapping[str, Any],
    worker_payload: Mapping[str, Any],
    registry: ArtifactRegistry,
    plan_steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a deterministic context shared by hooks and code reviewers."""
    workspace_dir = Path(os.path.realpath(workspace_root))
    step_id = str(step.get("step_id") or step.get("stage_id") or "step")
    declared_paths = _collect_declared_paths(step=step, registry=registry)
    text = _collect_keywords(step, worker_payload)
    surface_tags = _surface_tags_from_text(text, declared_paths)
    project_files = _project_file_summary(workspace_dir)
    if not project_files:
        risk_reasons = ["project_files_unavailable"]
    else:
        risk_reasons = []
    if not declared_paths:
        risk_reasons.append("no_declared_paths")
    touched = [
        str(item)
        for item in worker_payload.get("artifact_ids_touched") or []
        if str(item).strip()
    ]
    if not touched:
        risk_reasons.append("no_worker_artifact_ids_touched")
    if any(tag in HIGH_RISK_TAGS for tag in surface_tags):
        risk_reasons.append("high_risk_surface")
    run_full_matrix = True
    risk_level = "high" if risk_reasons else "medium"
    selected = list(CODE_REVIEWER_IDS)
    return {
        "hook": "code_review_context",
        "status": "PASS",
        "step_id": step_id,
        "surface_tags": sorted(surface_tags),
        "declared_paths": sorted(declared_paths),
        "worker_artifact_ids_touched": touched,
        "project_files": project_files,
        "project_file_count_sampled": len(project_files),
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "run_full_matrix": run_full_matrix,
        "selected_code_reviewer_ids": selected,
        "review_policy": {
            "deterministic_hooks_first": True,
            "agent_reviewers_parallel": True,
            "always_run_full_code_reviewer_matrix": True,
            "failure_mode": "return_feedback_to_same_worker_session",
            "auto_fix": False,
        },
        "plan_step_count": len(plan_steps),
    }


def _names_loaded_from_call(call: ast.Call) -> Set[str]:
    loaded: Set[str] = set()
    for node in ast.walk(call):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    return loaded


def _check_mask_variable_consumed(
    *,
    evaluate_path: Path,
    project_root: Path,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        source = evaluate_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception as exc:
        return {"checked": False, "error": str(exc)}
    mask_names: Set[str] = set()
    loader_calls: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            call = node.value
            if isinstance(call, ast.Call):
                func_name = ""
                if isinstance(call.func, ast.Name):
                    func_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    func_name = call.func.attr
                if func_name in {"apply_point_mask", "apply_block_mask"}:
                    for target in node.targets:
                        if isinstance(target, ast.Tuple) and len(target.elts) >= 2:
                            candidate = target.elts[1]
                            if isinstance(candidate, ast.Name):
                                mask_names.add(candidate.id)
                        elif isinstance(target, ast.Name):
                            mask_names.add(target.id)
        if isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if "DataLoader" in func_name or func_name in {"load_dataset", "load_masked_dataset"}:
                loader_calls.append(
                    {
                        "line": getattr(node, "lineno", None),
                        "loaded_names": sorted(_names_loaded_from_call(node)),
                    }
                )
    if not mask_names:
        return {
            "checked": True,
            "mask_names": [],
            "loader_calls": loader_calls,
            "rule_applied": False,
        }
    consumed = any(mask_names.intersection(set(call["loaded_names"])) for call in loader_calls)
    if not consumed:
        rel_path = _rel(evaluate_path, project_root)
        issues.append(
            {
                "rule": "masked_evaluation_mask_not_propagated",
                "path": rel_path,
                "message": (
                    "`evaluate.py` creates masked-evaluation mask variables from "
                    "`apply_point_mask`/`apply_block_mask`, but no DataLoader or evaluation "
                    "call consumes those mask variables. This can recompute an all-one mask "
                    "after NaN filling and invalidate missingness-sensitive evaluation."
                ),
                "fix": (
                    "Pass the explicit missingness mask into the masked evaluation loader or "
                    "evaluation function, and add bounded smoke evidence proving masked batches "
                    "contain masked entries."
                ),
            }
        )
    return {
        "checked": True,
        "mask_names": sorted(mask_names),
        "loader_calls": loader_calls,
        "mask_consumed_by_loader_or_evaluation": consumed,
        "rule_applied": True,
    }


def _stress_file_candidates(project_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in _iter_project_files(project_root):
        if path.suffix != ".py":
            continue
        name = path.name.lower()
        if any(token in name for token in ("stress", "mask", "eval", "evaluate", "runner", "run")):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.as_posix())


def _read_project_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _check_stress_missingness_preprocessing_order(
    *,
    project_root: Path,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Catch stress masks that bypass the same missingness preprocessing path.

    The check is intentionally pattern-based and conservative. It only fires
    when a project has an explicit gap/imputation/fill preprocessing surface and
    a stress/masked-evaluation surface that creates new missingness from an
    already prepared loader without reusing that preprocessing surface.
    """
    project_files = list(_iter_project_files(project_root))
    preprocessing_files = [
        path
        for path in project_files
        if path.suffix == ".py"
        and re.search(r"(gap|imput|fill|missing)", path.name.lower())
    ]
    if not preprocessing_files:
        return {"checked": False, "reason": "no gap/imputation preprocessing module detected"}

    preprocessing_tokens = {
        "gap_fill",
        "gap_filler",
        "GapAdaptive",
        "impute",
        "imputer",
        "fill_missing",
        "prepare_split",
        "missing_preprocess",
    }
    checked_files: List[str] = []
    flagged_files: List[str] = []
    for path in _stress_file_candidates(project_root):
        text = _read_project_text(path)
        lowered = text.lower()
        has_stress_mask = (
            ("stress" in lowered or "mask" in lowered)
            and re.search(r"(point_mask|block_mask|mask_rate|masked_loader|stress)", lowered)
            and ("test_loader" in lowered or ".get_iterator(" in lowered or "DataLoader" in text)
        )
        if not has_stress_mask:
            continue
        checked_files.append(f"project/{_rel(path, project_root)}")
        reuses_preprocess = any(token.lower() in lowered for token in preprocessing_tokens)
        creates_loader_from_masked = bool(
            re.search(r"DataLoader\s*\(", text)
            or "masked_loader" in lowered
            or "test_loader.get_iterator" in lowered
        )
        if creates_loader_from_masked and not reuses_preprocess:
            rel_path = f"project/{_rel(path, project_root)}"
            flagged_files.append(rel_path)
            issues.append(
                {
                    "rule": "stress_missingness_bypasses_preprocessing",
                    "path": rel_path,
                    "message": (
                        "Stress/masked evaluation appears to create new missingness from an already prepared "
                        "loader without re-running the project's gap/imputation/missingness preprocessing. "
                        "This can make the stress protocol bypass the component it is meant to evaluate."
                    ),
                    "fix": (
                        "Apply stress-induced missingness at the raw/pre-fill input boundary, then run the same "
                        "missingness preprocessing used by the normal data path, or explicitly call the shared "
                        "preprocessing function on masked inputs before evaluation. Add bounded evidence showing "
                        "the stress mask passes through that preprocessing path."
                    ),
                }
            )
    return {
        "checked": bool(checked_files),
        "preprocessing_files": [f"project/{_rel(path, project_root)}" for path in preprocessing_files],
        "checked_stress_files": checked_files,
        "flagged_files": flagged_files,
    }


def audit_code_scientific_invariants(
    *,
    workspace_root: str,
    review_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run deterministic code-science invariants that are cheap and formal."""
    workspace_dir = Path(os.path.realpath(workspace_root))
    project_root = workspace_dir / "project"
    issues: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}
    surface_tags = set(review_context.get("surface_tags") or [])
    should_check_masking = bool(surface_tags.intersection({"masking", "metrics", "data"}))
    evaluate_path = project_root / "evaluate.py"
    if should_check_masking and evaluate_path.exists():
        checks["masked_evaluation_mask_propagation"] = _check_mask_variable_consumed(
            evaluate_path=evaluate_path,
            project_root=project_root,
            issues=issues,
        )
    elif should_check_masking:
        checks["masked_evaluation_mask_propagation"] = {
            "checked": False,
            "reason": "project/evaluate.py not present",
        }
    if should_check_masking:
        checks["stress_missingness_preprocessing_order"] = _check_stress_missingness_preprocessing_order(
            project_root=project_root,
            issues=issues,
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "code_scientific_invariants",
        "issues": issues,
        "checks": checks,
        "surface_tags": sorted(surface_tags),
        "project_root": str(project_root),
    }


def format_code_invariant_feedback(payload: Mapping[str, Any]) -> str:
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
    lines = [
        "Xcientist formal code-science invariant hook blocked worker completion.",
        "",
        "The non-formal reviewer was not called because deterministic checks failed first.",
        "",
        "Issues:",
    ]
    if not issues:
        lines.append("- Unknown code invariant failure.")
    for item in issues:
        if not isinstance(item, Mapping):
            lines.append(f"- {item}")
            continue
        location = str(item.get("path") or "").strip()
        prefix = f"[{item.get('rule') or 'code_scientific_invariant'}]"
        message = str(item.get("message") or "").strip()
        fix = str(item.get("fix") or "").strip()
        line = f"- {prefix} {location} {message}".strip()
        if fix:
            line += f" Fix: {fix}"
        lines.append(line)
    lines.extend(
        [
            "",
            "Fix these issues in the same worker session, update managed artifacts through artifact tools, then finish again.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "audit_code_scientific_invariants",
    "build_code_review_context",
    "format_code_invariant_feedback",
]
