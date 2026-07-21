"""Artifact registry, ledger, tools, and hooks for experiment-agent.

The registry is runtime-owned. Workers publish managed artifacts through the
artifact tools instead of writing contract paths directly with generic tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, model_validator

from src.agents.experiment_agent.runtime.openharness_vendor import ensure_vendored_openharness_path


ensure_vendored_openharness_path()

from openharness.hooks.events import HookEvent
from openharness.hooks.types import AggregatedHookResult, HookResult
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.utils.shell import create_shell_subprocess
from src.agents.experiment_agent.runtime.report_layout import (
    ReportLayout,
    artifact_rel,
    phase_rel,
    planner_rel,
    scope_dir,
    step_report_abs_paths,
    step_report_paths,
)

ARTIFACT_LEDGER_REL = "agent_reports/_runtime/artifact_ledger.jsonl"
ARTIFACT_CONTRACT_REL = "agent_reports/_runtime/artifact_registry.json"

RUNTIME_REPORT_SUFFIXES = (
    "_planner_report.json",
    "_worker_report.json",
    "_reviewer_report.json",
    "planner_report.json",
    "executable.json",
    "latest.json",
    "phase.json",
    "run_timeline.jsonl",
    "stray_outputs.json",
    "artifact_registry.json",
    "artifact_ledger.jsonl",
)

_REPORT_PHASE_DIRS = {"prepare", "code", "science", "ablation", "_runtime"}
_REPORT_RUNTIME_ROLES = {"plan", "worker", "review", "hook", "final"}
_OUTPUT_ALLOWED_ROOTS = {
    "agent_reports",
    "dataset_candidate",
    "model_candidate",
    "project",
    "repos",
    "results",
}
_OUTPUT_LIKE_FILENAMES = {
    "metrics.json",
    "result.json",
    "results.json",
    "run.log",
    "train.log",
    "training_log.txt",
    "eval.log",
    "evaluation.log",
}
_OUTPUT_LIKE_SUFFIXES = (".ckpt", ".pth", ".pt")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REPO_WORKSPACE_ROOT = _REPO_ROOT / "workspace"
_AGENT_WORKSPACE_MARKER = "/agent_workspace"
_WORKSPACE_BOUNDARY_TOOLS = {"read_file", "write_file", "edit_file", "glob", "grep"}
_ALLOWED_SOURCE_URI_SCHEMES = {"http", "https", "doi", "arxiv", "git+https", "git+ssh", "ssh"}


@dataclass
class ArtifactSpec:
    artifact_id: str
    stage: str
    path: str
    kind: str = "file"
    required: bool = True
    schema_name: str = ""
    writer: str = "worker"
    publish_mode: str = "write"
    description: str = ""

    def resolved_path(self, workspace_root: str) -> str:
        raw = str(self.path or "").strip()
        if not raw:
            return ""
        return os.path.realpath(raw if os.path.isabs(raw) else os.path.join(workspace_root, raw))


@dataclass
class ArtifactRegistry:
    workspace_root: str
    specs: Dict[str, ArtifactSpec] = field(default_factory=dict)
    scratch_roots: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace_root = os.path.realpath(self.workspace_root)
        if not self.scratch_roots:
            self.scratch_roots = [
                os.path.join(self.workspace_root, "project", ".scratch"),
                os.path.join(self.workspace_root, "results", "_scratch"),
                os.path.join(self.workspace_root, ".openharness_runtime"),
            ]

    def add(self, spec: ArtifactSpec) -> None:
        self.specs[spec.artifact_id] = spec

    def get(self, artifact_id: str) -> Optional[ArtifactSpec]:
        return self.specs.get(str(artifact_id or "").strip())

    def managed_paths(self) -> Dict[str, str]:
        return {
            artifact_id: spec.resolved_path(self.workspace_root)
            for artifact_id, spec in self.specs.items()
            if spec.path
        }

    def runtime_owned_paths(self) -> set[str]:
        return {
            spec.resolved_path(self.workspace_root)
            for spec in self.specs.values()
            if spec.writer == "runtime" and spec.path
        }

    def worker_required_specs(self) -> List[ArtifactSpec]:
        return [
            spec
            for spec in self.specs.values()
            if spec.required and spec.writer != "runtime"
        ]

    def to_context(self, *, stage: str = "", step_id: str = "", attempt: int = 0) -> Dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "stage": stage,
            "step_id": step_id,
            "attempt": attempt,
            "artifact_specs": [asdict(spec) for spec in self.specs.values()],
            "scratch_roots": list(self.scratch_roots),
            "ledger_path": os.path.join(self.workspace_root, ARTIFACT_LEDGER_REL),
        }

    @classmethod
    def from_context(cls, payload: Dict[str, Any] | None) -> "ArtifactRegistry":
        payload = payload or {}
        registry = cls(
            workspace_root=str(payload.get("workspace_root") or os.getcwd()),
            scratch_roots=[str(item) for item in payload.get("scratch_roots") or []],
        )
        for raw in payload.get("artifact_specs") or []:
            if not isinstance(raw, dict):
                continue
            try:
                registry.add(ArtifactSpec(**raw))
            except TypeError:
                continue
        return registry


class ArtifactLedger:
    def __init__(self, workspace_root: str, path: Optional[str] = None) -> None:
        self.workspace_root = os.path.realpath(workspace_root)
        self.path = os.path.realpath(path or os.path.join(self.workspace_root, ARTIFACT_LEDGER_REL))

    def append(self, event: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **event,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def read(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def latest_for(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        for record in reversed(self.read()):
            if record.get("artifact_id") == artifact_id and record.get("event") in {
                "write_artifact",
                "patch_artifact",
                "publish_artifact",
                "run_artifact_command",
                "runtime_artifact",
            }:
                return record
        return None

    def latest_worker_write_for(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        for record in reversed(self.read()):
            if record.get("artifact_id") == artifact_id and record.get("event") in {
                "write_artifact",
                "patch_artifact",
                "publish_artifact",
                "run_artifact_command",
            }:
                return record
        return None


def artifact_registry_path(workspace_root: str) -> str:
    return os.path.join(os.path.realpath(workspace_root), ARTIFACT_CONTRACT_REL)


def artifact_ledger_path(workspace_root: str) -> str:
    return os.path.join(os.path.realpath(workspace_root), ARTIFACT_LEDGER_REL)


def write_artifact_registry_snapshot(registry: ArtifactRegistry) -> str:
    path = artifact_registry_path(registry.workspace_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "workspace_root": registry.workspace_root,
                "scratch_roots": registry.scratch_roots,
                "artifact_specs": [asdict(spec) for spec in registry.specs.values()],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


def record_runtime_artifact(
    *,
    workspace_root: str,
    artifact_id: str,
    path: str,
    stage: str = "runtime",
    step_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a runtime-owned artifact in the shared artifact ledger."""
    resolved = _resolve_path(os.path.realpath(workspace_root), path)
    kind = "dir" if os.path.isdir(resolved) else ("json" if resolved.endswith(".json") else "file")
    record = {
        "event": "runtime_artifact",
        "artifact_id": artifact_id,
        "path": resolved,
        "kind": kind,
        "sha256": _hash_path(resolved) if os.path.exists(resolved) else "",
        "stage": stage,
        "step_id": step_id,
        "tool": "runtime",
    }
    if extra:
        record.update(extra)
    ArtifactLedger(workspace_root).append(record)
    return record


def _path_under(root: str, candidate: str) -> bool:
    root_real = os.path.realpath(root)
    candidate_real = os.path.realpath(candidate)
    return candidate_real == root_real or candidate_real.startswith(root_real + os.sep)


def _path_allowed_for_tool(registry: ArtifactRegistry, candidate: str) -> bool:
    candidate_real = os.path.realpath(candidate)
    if _path_under(registry.workspace_root, candidate_real):
        return True
    if not _path_under(str(_REPO_ROOT), candidate_real):
        return False
    return not _path_under(str(_REPO_WORKSPACE_ROOT), candidate_real)


def _workspace_boundary_reason(registry: ArtifactRegistry, *, tool_name: str, path: str) -> str:
    return (
        f"Xcientist workspace-boundary hook blocked `{tool_name}` for path `{path}`.\n"
        "Use only the current experiment workspace and the current Xcientist-2 source tree "
        "outside repository-level workspace history. "
        f"Current workspace: `{registry.workspace_root}`. "
        f"Current source tree: `{_REPO_ROOT}`. "
        f"Repository workspace history root: `{_REPO_WORKSPACE_ROOT}`. "
        "Do not search, read, or reuse artifacts from other historical Xcientist workspaces; "
        "if you need prepared inputs, use the copies already present in this workspace."
    )


def _looks_like_cross_workspace_bash(registry: ArtifactRegistry, command: str) -> bool:
    command_text = str(command or "")
    for raw_path in _absolute_paths_in_text(command_text):
        if _AGENT_WORKSPACE_MARKER not in raw_path and not _path_under(str(_REPO_WORKSPACE_ROOT), raw_path):
            continue
        if not _path_allowed_for_tool(registry, raw_path):
            return True
    for raw_path in _relative_parent_paths_in_text(command_text):
        if not _path_allowed_for_tool(registry, _resolve_path(registry.workspace_root, raw_path)):
            return True
    return False


def _absolute_paths_in_text(text: str) -> List[str]:
    paths: List[str] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])/(?:[^\s'\"`$;|&<>]+)", str(text or "")):
        value = match.group(0).rstrip("),]}:")
        if value:
            paths.append(value)
    return paths


def _relative_parent_paths_in_text(text: str) -> List[str]:
    try:
        tokens = shlex.split(str(text or ""), posix=True)
    except ValueError:
        tokens = re.split(r"\s+", str(text or ""))
    paths: List[str] = []
    for token in tokens:
        value = token.strip().strip("),]}:")
        if value in {".", ".."} or value.startswith(("./", "../")) or "/.." in value:
            paths.append(value)
    return paths


def _bash_boundary_reason(registry: ArtifactRegistry, command: str) -> str:
    return (
        "Xcientist workspace-boundary hook blocked this bash command because it appears "
        "to search or read across broader `agent_workspace` history instead of the current run.\n"
        f"Current workspace: `{registry.workspace_root}`.\n"
        f"Current source tree: `{_REPO_ROOT}`.\n"
        f"Repository workspace history root: `{_REPO_WORKSPACE_ROOT}`.\n"
        "Restrict commands to the current workspace, for example `find . ...` from the "
        "workspace root or explicit paths under `workspace/graph_seed`. Do not reuse old "
        "plans, reports, or project files from historical Xcientist workspaces.\n"
        f"Command: `{command}`"
    )


def _is_allowed_source_uri(value: str) -> bool:
    if re.match(r"^[^@\s]+@[^:\s]+:[^\s]+$", value):
        return True
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", value)
    return bool(match and match.group(1).lower() in _ALLOWED_SOURCE_URI_SCHEMES)


def _source_reference_issues(registry: ArtifactRegistry, sources: Iterable[str]) -> List[str]:
    issues: List[str] = []
    source_list = list(sources)
    if not source_list:
        return ["`sources` must list at least one concrete workspace/source-tree path or external URL."]
    for index, raw_source in enumerate(source_list):
        source = str(raw_source or "").strip()
        if not source:
            issues.append(f"`sources[{index}]` is empty.")
            continue
        if _is_allowed_source_uri(source):
            continue
        uri_match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", source)
        if uri_match:
            issues.append(
                f"`sources[{index}]` uses unsupported URI scheme `{uri_match.group(1)}`. "
                "Use http(s), doi, arxiv, git+https, git+ssh, ssh, or an existing path."
            )
            continue
        resolved = _resolve_path(registry.workspace_root, source)
        if not _path_allowed_for_tool(registry, resolved):
            issues.append(
                f"`sources[{index}]` points outside the current workspace/source tree: `{source}` -> `{resolved}`."
            )
            continue
        if not os.path.exists(resolved):
            issues.append(
                f"`sources[{index}]` is not an existing path and is not an allowed URL/identifier: `{source}`."
            )
    return issues


def _source_reference_reason(registry: ArtifactRegistry, issues: Iterable[str]) -> str:
    issue_lines = [f"- {issue}" for issue in issues]
    return (
        "Xcientist source-provenance hook blocked `record_sources` because every source "
        "must be concrete and confined to the active run.\n\n"
        "Allowed source references:\n"
        "- existing files or directories under the current experiment workspace\n"
        "- existing files or directories under the current Xcientist-2 source tree, excluding repository-level workspace history\n"
        "- external references with http(s), doi, arxiv, git+https, git+ssh, ssh, or git@host:path syntax\n\n"
        f"Current workspace: `{registry.workspace_root}`.\n"
        f"Current source tree: `{_REPO_ROOT}`.\n"
        f"Repository workspace history root: `{_REPO_WORKSPACE_ROOT}`.\n\n"
        "Issues:\n"
        + "\n".join(issue_lines)
    )


def _resolve_path(workspace_root: str, value: str) -> str:
    path = Path(str(value or "").strip()).expanduser()
    if not path.is_absolute():
        path = Path(workspace_root) / path
    return str(path.resolve())


def _workspace_root_from_artifact_path(path: str) -> str:
    resolved = os.path.realpath(path)
    parts = resolved.split(os.sep)
    if "agent_reports" in parts:
        index = parts.index("agent_reports")
        root_parts = parts[:index]
        return os.sep.join(root_parts) or os.sep
    return os.path.dirname(os.path.dirname(resolved))


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: str) -> str:
    if os.path.isfile(path):
        return _hash_file(path)
    if os.path.isdir(path):
        h = hashlib.sha256()
        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for name in files:
                file_path = os.path.join(root, name)
                rel = os.path.relpath(file_path, path)
                h.update(rel.encode("utf-8"))
                h.update(_hash_file(file_path).encode("ascii"))
        return h.hexdigest()
    return ""


def _validate_json_schema(path: str, schema_name: str) -> List[str]:
    if not schema_name:
        return []
    if not os.path.exists(path) or not os.path.isfile(path):
        return [f"JSON schema `{schema_name}` target is not a file: {path}"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return [f"Invalid JSON for schema `{schema_name}`: {exc}"]
    if schema_name == "ablation_results":
        try:
            from src.agents.experiment_agent.runtime.ablation_results import (
                build_ablation_results_manifest,
                validate_ablation_results_payload,
            )

            workspace_root = _workspace_root_from_artifact_path(path)
            manifest = build_ablation_results_manifest(workspace_root)
            ok, reason = validate_ablation_results_payload(
                payload,
                canonical_component_names=list(manifest.get("canonical_components") or []),
            )
            return [] if ok else [reason or "ablation_results schema validation failed"]
        except Exception as exc:
            return [f"ablation_results schema validation failed: {exc}"]
    if schema_name == "code_plan":
        try:
            from src.agents.experiment_agent.runtime.contracts import (
                validate_code_step_contract_fields,
            )

            if not isinstance(payload, dict):
                return ["code_plan must be a JSON object."]
            stages = payload.get("stages")
            if not isinstance(stages, list) or not stages:
                return ["code_plan must contain a non-empty top-level `stages` list."]
            issues: List[str] = []
            workspace_root = _workspace_root_from_artifact_path(path)
            project_dir = os.path.join(workspace_root, "project")
            for index, step in enumerate(stages, start=1):
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
            if isinstance(stages[-1], dict) and stages[-1].get("step_id") != "final_integration_smoke":
                issues.append("final step_id must be `final_integration_smoke`")
            if issues:
                issues.append(
                    "code_plan must be a real implementation plan: use canonical idea.json component "
                    "names, existing repos/ source paths, project/ target paths, real verification "
                    "commands, code.<step_id>.handoff artifact ids, and final smoke metrics/evidence "
                    "under agent_reports/code/artifacts/final_integration_smoke.json. Placeholder "
                    "test/demo values are rejected."
                )
            return issues
        except Exception as exc:
            return [f"code_plan schema validation failed: {exc}"]
    if schema_name == "code_handoff":
        try:
            from src.agents.experiment_agent.runtime.contracts import validate_code_handoff_payload

            workspace_root = _workspace_root_from_artifact_path(path)
            return validate_code_handoff_payload(
                payload,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            return [f"{schema_name} schema validation failed: {exc}"]
    if schema_name == "code_smoke_evidence":
        try:
            from src.agents.experiment_agent.runtime.contracts import validate_code_smoke_evidence_payload

            workspace_root = _workspace_root_from_artifact_path(path)
            return validate_code_smoke_evidence_payload(
                payload,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            return [f"{schema_name} schema validation failed: {exc}"]
    if schema_name == "prepare_plan":
        from src.agents.experiment_agent.runtime.prepare_contracts import validate_prepare_plan

        return validate_prepare_plan(payload)
    if schema_name.startswith("prepare_"):
        from src.agents.experiment_agent.runtime.prepare_contracts import validate_prepare_artifact_payload

        workspace_root = _workspace_root_from_artifact_path(path)
        return validate_prepare_artifact_payload(
            schema_name,
            payload,
            workspace_root=workspace_root,
        )
    if schema_name == "science_plan":
        try:
            from src.agents.experiment_agent.runtime.contracts import (
                validate_science_condition_plan,
            )

            if not isinstance(payload, dict):
                return [f"{schema_name} must be a JSON object."]
            stages = payload.get("stages")
            if not isinstance(stages, list) or not stages:
                return [f"{schema_name} must contain a non-empty top-level `stages` list."]
            workspace_root = _workspace_root_from_artifact_path(path)
            project_dir = os.path.join(workspace_root, "project")
            return validate_science_condition_plan(
                payload,
                project_dir=project_dir,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            return [f"{schema_name} schema validation failed: {exc}"]
    if schema_name == "science_evidence":
        try:
            from src.agents.experiment_agent.runtime.contracts import validate_science_evidence_payload

            workspace_root = _workspace_root_from_artifact_path(path)
            return validate_science_evidence_payload(
                payload,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            return [f"{schema_name} schema validation failed: {exc}"]
    if not isinstance(payload, (dict, list)):
        return [f"JSON artifact `{schema_name}` must be an object or array."]
    return []


def _load_idea_component_names(workspace_root: str) -> List[str]:
    path = os.path.join(os.path.realpath(workspace_root), "idea.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    raw_components = payload.get("components") if isinstance(payload, dict) else None
    if not isinstance(raw_components, list):
        return []
    names: List[str] = []
    for item in raw_components:
        if not isinstance(item, dict):
            continue
        name = str(item.get("component") or item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _validate_spec_path(registry: ArtifactRegistry, spec: ArtifactSpec) -> List[str]:
    issues: List[str] = []
    path = spec.resolved_path(registry.workspace_root)
    if not path:
        return [f"Artifact `{spec.artifact_id}` has empty path."]
    if not _path_under(registry.workspace_root, path):
        issues.append(f"Artifact `{spec.artifact_id}` escapes workspace: {path}")
    if spec.required:
        if spec.kind == "dir":
            if not os.path.isdir(path):
                issues.append(f"Required artifact directory missing: {spec.artifact_id} -> {path}")
        else:
            if not os.path.isfile(path):
                issues.append(f"Required artifact file missing: {spec.artifact_id} -> {path}")
    if spec.kind == "json" and os.path.exists(path):
        issues.extend(_validate_json_schema(path, spec.schema_name))
    return issues


def _validate_candidate_artifact(registry: ArtifactRegistry, spec: ArtifactSpec, candidate_path: str) -> List[str]:
    issues: List[str] = []
    if not _path_under(registry.workspace_root, candidate_path):
        issues.append(f"Artifact `{spec.artifact_id}` candidate escapes workspace: {candidate_path}")
    if spec.kind == "dir":
        if not os.path.isdir(candidate_path):
            issues.append(f"Artifact `{spec.artifact_id}` candidate is not a directory: {candidate_path}")
        return issues
    if not os.path.isfile(candidate_path):
        issues.append(f"Artifact `{spec.artifact_id}` candidate is not a file: {candidate_path}")
        return issues
    if spec.kind == "json":
        issues.extend(_validate_json_schema(candidate_path, spec.schema_name))
    return issues


def _json_template(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _prepare_blocked_template() -> Dict[str, Any]:
    return {
        "status": "BLOCKED",
        "blocker": {
            "reason": "why real acquisition or verification cannot proceed",
            "attempted_queries": ["searches, local inspections, dry-runs, or commands attempted"],
            "rejected_candidates": ["candidate plus concrete rejection reason"],
            "missing_requirements": ["resource, credential, license, data access, or dependency that is missing"],
            "user_action_required": "specific action needed from the user or environment",
            "evidence_paths": ["workspace-relative local evidence file, e.g. project/env/blocked_notes.txt"],
        },
    }


def _prepare_ready_template(schema_name: str) -> Dict[str, Any]:
    templates: Dict[str, Dict[str, Any]] = {
        "prepare_discovery": {
            "status": "READY",
            "task_signature": {
                "domain": "task/domain inferred from idea.json",
                "objective": "scientific claim or benchmark objective",
                "required_modalities": ["data/model/input modality requirements"],
                "evaluation_needs": ["metrics, splits, stress tests, benchmarks"],
            },
            "resource_requirements": {
                "repo": "implementation/reference-code requirements",
                "dataset": "real dataset/schema/split/access requirements",
                "model": "local checkpoint/API/backbone requirements",
                "benchmark": "required benchmark protocol or reason no external benchmark is needed",
            },
            "mcp_status_snapshot": {
                "status": "connected|unavailable|connect_failed",
                "connected": True,
                "servers": ["tavily"],
                "status_path": "agent_reports/_runtime/mcp_status.json",
            },
            "selection_criteria": {
                "repo": ["authoritative implementation", "license", "matches idea backbone"],
                "dataset": ["matches task", "real access", "schema/split supports evaluation"],
                "model": ["matches required backend", "load/dry-run feasible"],
            },
            "queries": [
                {
                    "source": "tavily|local|github|paper",
                    "query": "concrete query or local inspection command",
                    "purpose": "what resource requirement this query tested",
                    "result_summary": "brief outcome",
                }
            ],
            "candidate_table": {
                "repos": [
                    {
                        "candidate_id": "repo-1",
                        "name": "repo name",
                        "source": "https://...",
                        "task_fit": "why it fits or does not fit",
                        "evidence": ["URL or local evidence path"],
                        "decision": "selected",
                        "reason": "why selected or rejected",
                    }
                ],
                "datasets": [
                    {
                        "candidate_id": "dataset-1",
                        "name": "dataset name",
                        "source": "https://... or local path",
                        "task_fit": "schema/split/benchmark fit",
                        "evidence": ["URL or local evidence path"],
                        "decision": "selected",
                        "reason": "why selected or rejected",
                    }
                ],
                "models": [
                    {
                        "candidate_id": "model-1",
                        "name": "model/checkpoint/API",
                        "source": "https://... or local path/API id",
                        "task_fit": "backend and dimension fit",
                        "evidence": ["URL or local evidence path"],
                        "decision": "selected",
                        "reason": "why selected or rejected",
                    }
                ],
                "benchmarks": [],
            },
            "selected_candidate_ids": {
                "repo": "repo-1",
                "dataset": "dataset-1",
                "model": "model-1",
            },
            "selected_resources": {
                "repo": "selected repo",
                "dataset": "selected dataset",
                "model": "selected model",
            },
            "rejected_candidates": [
                {"candidate_id": "repo-2", "resource_type": "repo", "reason": "concrete rejection reason"}
            ],
            "evidence_gaps": [],
            "selection_rationale": "why the selected resources jointly satisfy the idea and experiment contract",
        },
        "prepare_repos": {
            "status": "READY",
            "selection_rationale": "why the repo is the correct implementation/reference",
            "selected_repositories": [
                {
                    "name": "repo",
                    "source_url": "https://...",
                    "repo_path": "repos/<repo>",
                    "commit": "resolved commit sha or immutable version",
                    "license": "license evidence",
                    "readme_evidence": "README or docs path",
                    "reference_entrypoints": ["repos/<repo>/path/to/entrypoint.py"],
                }
            ],
        },
        "prepare_dataset": {
            "status": "READY",
            "selection_rationale": "why this dataset/split matches the idea",
            "selected_datasets": [{"name": "dataset", "source": "url or local source"}],
            "expected_files": ["dataset_candidate/<file>"],
            "schema_probe": {"summary": "columns/shapes/splits verified", "probe_log": "dataset_candidate/<probe_log>.txt"},
            "checksums": {"dataset_candidate/<file>": "sha256/size evidence"},
        },
        "prepare_model": {
            "status": "READY",
            "selection_rationale": "why this model/backend matches the experiment",
            "selected_models": [{"name": "model", "backend": "api or checkpoint"}],
            "api_dry_run": {"model_id": "model id", "env_vars": ["API_KEY_ENV"], "probe_log": "model_candidate/<probe_log>.txt"},
            "local_checkpoints": ["model_candidate/<checkpoint>"],
        },
        "prepare_env": {
            "status": "READY",
            "selection_rationale": "why this environment is sufficient for code/science",
            "venv_path": "project/.venv",
            "python_path": "project/.venv/bin/python",
            "install_commands": ["project/.venv/bin/pip install ..."],
            "installed_packages": [{"name": "package", "version": "x.y.z"}],
            "smoke_logs": ["project/env/<smoke_log>.txt"],
            "resource_binding_smoke": {"command": "project/.venv/bin/python ...", "smoke_log": "project/env/<smoke_log>.txt"},
        },
        "prepare_target_inventory": {
            "status": "READY",
            "components": [{"component": "canonical idea component", "targets": ["project/module.py:function"]}],
            "resources": {"dataset": "dataset_candidate/...", "model": "model_candidate/... or API model id"},
            "benchmarks": [{"name": "benchmark", "entrypoint": "project/..."}],
            "metrics": [{"name": "metric", "source": "project/..."}],
        },
    }
    return templates.get(schema_name, {"status": "READY", "selection_rationale": "local evidence-backed rationale"})


def _prepare_plan_template() -> Dict[str, Any]:
    stage_artifacts = {
        "repos": ["prepare.discovery", "prepare.repos"],
        "dataset": ["prepare.dataset"],
        "model": ["prepare.model"],
        "env": ["prepare.env"],
        "synthesis": ["prepare.idea", "prepare.target_inventory"],
    }
    stage_paths = {
        "repos": [
            "agent_reports/prepare/artifacts/discovery.json",
            "agent_reports/prepare/artifacts/repos.json",
        ],
        "dataset": ["agent_reports/prepare/artifacts/dataset.json"],
        "model": ["agent_reports/prepare/artifacts/model.json"],
        "env": ["agent_reports/prepare/artifacts/env.json"],
        "synthesis": [
            "agent_reports/prepare/artifacts/idea.md",
            "agent_reports/prepare/artifacts/target_inventory.json",
        ],
    }
    return {
        "stages": [
            {
                "stage_id": stage_id,
                "goal": f"Acquire or verify real {stage_id} resources for this idea.",
                "input_paths": {"idea": "idea.json"},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "research_required": stage_id in {"repos", "dataset", "model"},
                "acquisition_required": True,
                "existing_local_hints": [],
                "done_condition": (
                    "Must write the assigned managed artifacts using Xcientist artifact tools: "
                    + ", ".join(stage_paths[stage_id])
                    + ". The artifact ledger is the proof."
                    + (
                        " idea.md must include `## Canonical Idea Components`, and "
                        "target_inventory.json must map every idea.json component to concrete implementation targets."
                        if stage_id == "synthesis"
                        else ""
                    )
                ),
                "artifact_ids": artifact_ids,
            }
            for stage_id, artifact_ids in stage_artifacts.items()
        ],
        "summary": "Prepare real repos, dataset, model/API, environment, and handoff artifacts.",
        "usage_notes": "Later phases must use prepared local evidence and paths recorded in prepare artifacts.",
    }


def _code_plan_template() -> Dict[str, Any]:
    return {
        "stages": [
            {
                "step_id": "implement_real_component_path",
                "goal": "Implement the idea-aligned experiment component inside project/.",
                "component_scope": ["canonical idea component name"],
                "code_artifacts": [
                    {
                        "path": "project/module.py",
                        "artifact_type": "python_module",
                        "symbols": ["ImplementedSymbol"],
                        "responsibility": "What this file owns in the experiment path.",
                        "dependencies": ["torch"],
                        "config_keys": ["component_flag"],
                        "entrypoint_role": "model_component",
                    }
                ],
                "interface_contract": {"entrypoint": "project/.venv/bin/python project/run.py --help"},
                "implementation_requirements": {"dataset": "dataset_candidate/<prepared dataset>"},
                "component_disable_hooks": [
                    {"component": "canonical idea component name", "flag": "--disable-component"}
                ],
                "experiment_bindings": {"metrics_json": "results/smoke/metrics.json"},
                "repo_source_paths": ["repos/<repo>/<source>.py"],
                "repo_copy_intent": "copy_and_modify",
                "project_target_paths": ["project/module.py"],
                "input_paths": {
                    "idea": "agent_reports/prepare/artifacts/idea.md",
                    "targets": "agent_reports/prepare/artifacts/target_inventory.json",
                },
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "write_scope": "project/",
                "verify_command": "project/.venv/bin/python project/run.py --bounded-smoke ...",
                "done_condition": "Verification passes with real dataset_candidate files; sys.path injection, editable installs of repos/, and imports reaching outside project/ are forbidden.",
                "artifact_ids": ["code.implement_real_component_path.handoff"],
            },
            {
                "step_id": "final_integration_smoke",
                "goal": "Run bounded real-data all-components and component-disabled smoke through project/.",
                "component_scope": ["canonical idea component name"],
                "code_artifacts": [
                    {
                        "path": "project/run.py",
                        "artifact_type": "entrypoint",
                        "symbols": ["main"],
                        "responsibility": "Workspace-root runnable experiment entrypoint.",
                        "dependencies": ["torch"],
                        "config_keys": ["condition_id"],
                        "entrypoint_role": "runner",
                    }
                ],
                "interface_contract": {"entrypoint": "project/.venv/bin/python project/run.py --condition full_all_components"},
                "implementation_requirements": {"no_synthetic_data": True},
                "component_disable_hooks": [
                    {"component": "canonical idea component name", "flag": "--disable-component"}
                ],
                "experiment_bindings": {"metrics_json": "results/code_smoke/metrics.json"},
                "repo_source_paths": [],
                "repo_copy_intent": "none",
                "project_target_paths": ["project/run.py"],
                "input_paths": {"dataset": "dataset_candidate/<prepared dataset>"},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "write_scope": "project/",
                "verify_command": "project/.venv/bin/python project/run.py --condition full_all_components --max-batches 2 --output results/code_smoke",
                "done_condition": "Bounded smoke uses real dataset_candidate files and writes agent_reports/code/artifacts/final_integration_smoke.json; sys.path injection, editable installs of repos/, imports outside project/, mocks, synthetic data, dry-run-only checks, and imports-only checks are forbidden.",
                "artifact_ids": ["code.final_integration_smoke.handoff", "code.final_integration_smoke.evidence"],
            },
        ],
        "summary": "Code plan with real implementation steps and final integration smoke.",
        "usage_notes": "Science commands should run from the workspace root against project/.",
    }


def _code_handoff_template() -> Dict[str, Any]:
    return {
        "project_files": ["project/module.py", "project/run.py"],
        "verification": "Bounded real-data verification passed for this step.",
        "verify_command": "project/.venv/bin/python project/run.py --bounded-smoke --output results/code_smoke",
        "returncode": 0,
        "logs": ["results/code_smoke/step.log"],
        "raw_outputs": ["results/code_smoke/output.json"],
        "metrics_files": ["results/code_smoke/metrics.json"],
        "notes": "Any concise implementation notes needed by reviewers.",
    }


def _code_smoke_evidence_template() -> Dict[str, Any]:
    return {
        "command": "project/.venv/bin/python project/run.py --condition full_all_components --max-batches 2 --output results/code_smoke",
        "returncode": 0,
        "raw_outputs": ["results/code_smoke/full_all_components/output.json"],
        "logs": ["results/code_smoke/full_all_components/run.log"],
        "metrics_files": ["results/code_smoke/full_all_components/metrics.json"],
        "dataset_bindings": {"train": "dataset_candidate/<prepared train file>", "evaluation": ["dataset_candidate/<prepared eval file>"]},
        "component_toggles": [
            {"condition_id": "full_all_components", "enabled_components": ["all canonical components"], "disabled_components": []},
            {"condition_id": "without_component_name", "enabled_components": ["all other canonical components"], "disabled_components": ["canonical component name"], "flag": "--disable-component-name"},
        ],
        "bounded_runtime": {"max_batches": 2, "max_epochs": 1, "reason": "bounded integration smoke only"},
    }


def _science_plan_template() -> Dict[str, Any]:
    reference_condition = {
        "condition_id": "full_all_components",
        "goal": "Run the full idea with all canonical components enabled.",
        "enabled_components": ["all canonical components"],
        "disabled_components": [],
        "reference_condition_id": None,
        "train_dataset_binding": {
            "name": "prepared_train_split",
            "path": "dataset_candidate/<prepared train split>",
            "split": "train",
        },
        "evaluation_dataset_bindings": [
            {
                "name": "prepared_eval_split",
                "path": "dataset_candidate/<prepared eval split>",
                "split": "eval",
            }
        ],
        "metric_bindings": [
            {"name": "MAE", "source": "project/evaluation.py"},
            {"name": "RMSE", "source": "project/evaluation.py"},
            {"name": "MAPE", "source": "project/evaluation.py"},
        ],
        "component_set_description": "All canonical idea components enabled.",
        "result_interpretation_rule": "Reference condition for component-disabled comparisons.",
        "run_level": "full",
        "setup_rationale": "Local evidence-backed full setup rationale.",
        "source_basis": [{"path": "agent_reports/code/phase.json", "basis": "code and smoke readiness"}],
        "runtime_probe_summary": "Bounded probes used only to calibrate command/runtime; not formal evidence.",
        "training_protocol": {
            "epochs": 10,
            "max_batches": 0,
            "batch_size": 64,
            "device": "cuda-or-cpu-selected-at-runtime",
            "seed": 1,
            "expected_runtime_sec": 3600,
            "full_setup_basis": "Why these values are appropriate for this task/workspace.",
        },
        "evaluation_protocol": {
            "horizons": [3, 6, 12],
            "mask_rates": [0.1, 0.2],
            "mask_patterns": ["point", "block"],
            "metrics": ["MAE", "RMSE", "MAPE"],
            "reference_condition_id": "",
            "perturbation_boundary": "Synthetic perturbations are injected before gap-fill/normalization so they traverse the same preprocessing path as natural missingness.",
            "preprocessing_boundary": "All training/evaluation conditions call the shared preprocessing pipeline before model inference and metric computation.",
            "ablation_isolation_assumptions": [
                "Only the declared component toggle differs from the all-components reference condition.",
                "Dataset split, seed, preprocessing, horizons, mask rates, and metrics stay fixed across compared conditions.",
            ],
        },
        "repo_source_paths": [],
        "repo_copy_intent": "none",
        "project_target_paths": ["project/run.py"],
        "input_paths": {"code": "agent_reports/code/phase.json"},
        "repos_policy": "reference_or_copy",
        "project_must_be_self_contained": True,
        "command": "project/.venv/bin/python project/run.py --condition full_all_components --output results/science/full_all_components",
        "output_dir": "results/science/full_all_components",
        "raw_evidence": ["results/science/full_all_components/metrics.json"],
        "pass_condition": "Formal full run completes with finite metrics and declared evidence.",
        "artifact_ids": ["science.full_all_components.evidence"],
    }
    ablation_condition = {
        **reference_condition,
        "condition_id": "without_component_name",
        "goal": "Run a full condition with one canonical component disabled.",
        "enabled_components": ["all other canonical components"],
        "disabled_components": ["canonical idea component name"],
        "reference_condition_id": "full_all_components",
        "component_set_description": "One canonical idea component disabled, all others enabled.",
        "result_interpretation_rule": "Compare against full_all_components to judge the disabled component contribution.",
        "evaluation_protocol": {
            **reference_condition["evaluation_protocol"],
            "reference_condition_id": "full_all_components",
        },
        "command": "project/.venv/bin/python project/run.py --condition without_component_name --disable-component canonical_name --reference-condition-id full_all_components --output results/science/without_component_name",
        "output_dir": "results/science/without_component_name",
        "raw_evidence": ["results/science/without_component_name/metrics.json"],
        "artifact_ids": ["science.without_component_name.evidence"],
    }
    return {
        "stages": [reference_condition, ablation_condition],
        "summary": "Science plan with an all-components reference and component-disabled full runs.",
        "usage_notes": "Final ablation_results.json is runtime-owned after reviewer-approved science evidence.",
    }


def artifact_schema_repair_guide(
    schema_name: str,
    *,
    artifact_id: str = "",
    path: str = "",
) -> str:
    """Return concise worker-facing schema repair guidance for managed artifacts."""
    schema = str(schema_name or "").strip()
    if not schema:
        return ""
    prefix = []
    if artifact_id:
        prefix.append(f"Artifact id: `{artifact_id}`")
    if path:
        prefix.append(f"Path: `{path}`")
    prefix.append(f"Schema: `{schema}`")
    header = "\n".join(prefix)

    if schema.startswith("prepare_") and schema != "prepare_plan":
        return (
            f"{header}\n"
            "Accepted prepare artifact statuses are exactly `READY` and `BLOCKED`.\n"
            "READY means real local acquisition/verification succeeded. Any declared path-like fields "
            "such as `expected_files`, `repo_path`, `probe_log`, `smoke_logs`, `local_checkpoints`, "
            "or `evidence_paths` must stay inside the current workspace and exist.\n"
            "READY template:\n"
            "```json\n"
            f"{_json_template(_prepare_ready_template(schema))}\n"
            "```\n"
            "BLOCKED template:\n"
            "```json\n"
            f"{_json_template(_prepare_blocked_template())}\n"
            "```"
        )

    if schema == "prepare_plan":
        return (
            f"{header}\n"
            "Write `prepare.plan` as a JSON object with top-level `stages` ordered exactly "
            "`repos`, `dataset`, `model`, `env`, `synthesis`. Each stage must include "
            "`stage_id`, `goal`, `input_paths`, `repos_policy`, `project_must_be_self_contained`, "
            "`research_required`, `acquisition_required`, `existing_local_hints`, "
            "`done_condition`, and exact stage `artifact_ids`. `repos_policy` must be "
            "`reference_or_copy`, `project_must_be_self_contained` must be true, and every "
            "`done_condition` must require Xcientist artifact tools and identify the artifact "
            "ledger as proof. The synthesis done_condition must also require `## Canonical Idea Components` "
            "in idea.md and target_inventory.json mappings for every idea.json component.\n"
            "Repair template:\n"
            "```json\n"
            f"{_json_template(_prepare_plan_template())}\n"
            "```"
        )

    if schema == "science_evidence":
        return (
            f"{header}\n"
            "Science evidence must describe the formal full run, not a smoke/probe/debug run. "
            "`condition_id`, component lists, `reference_condition_id`, `command`, `output_dir`, "
            "and condition `raw_evidence` entries must match the condition contract exactly. "
            "All listed raw/log/metric files must exist under `output_dir`.\n"
            "Template:\n"
            "```json\n"
            f"{_json_template({'condition_id': '<condition_id>', 'enabled_components': ['component kept enabled'], 'disabled_components': ['component disabled, or [] for reference'], 'reference_condition_id': '<reference condition id or null/empty for reference>', 'run_level': 'full', 'command': '<exact condition contract command>', 'returncode': 0, 'output_dir': 'results/science/<condition_id>', 'raw_outputs': ['results/science/<condition_id>/raw_output_or_metrics.json'], 'logs': ['results/science/<condition_id>/run.log'], 'metrics_files': ['results/science/<condition_id>/metrics.json'], 'dataset_bindings': {'train': 'dataset_candidate/...', 'evaluation': ['dataset_candidate/...']}, 'model_bindings': {'backend': 'api/checkpoint/project implementation used'}, 'duration_sec': 123.4})}\n"
            "```"
        )

    if schema == "code_handoff":
        return (
            f"{header}\n"
            "Code handoff must be a JSON object proving the step touched real `project/` files "
            "and ran a real verification command. Every listed project/log/raw/metric file must "
            "exist inside the current workspace. `returncode` must be 0.\n"
            "Template:\n"
            "```json\n"
            f"{_json_template(_code_handoff_template())}\n"
            "```"
        )

    if schema == "code_smoke_evidence":
        return (
            f"{header}\n"
            "Final integration smoke evidence must be bounded real-data evidence for the actual "
            "project runner. It must not be imports-only, dry-run-only, mock, synthetic, or a "
            "timeout. Every listed raw/log/metric file must exist inside the workspace and "
            "`dataset_bindings` must reference `dataset_candidate/`.\n"
            "Template:\n"
            "```json\n"
            f"{_json_template(_code_smoke_evidence_template())}\n"
            "```"
        )

    if schema == "science_plan":
        return (
            f"{header}\n"
            "Write a JSON object with top-level `stages`. Each science condition must include "
            "the full condition contract fields, `run_level: \"full\"`, concrete runtime setup "
            "rationale/source_basis/protocols, output under `results/science/<condition_id>/`, and "
            "each component-disabled condition must disable exactly one canonical idea component and "
            "reference an earlier all-components condition.\n"
            "Repair template:\n"
            "```json\n"
            f"{_json_template(_science_plan_template())}\n"
            "```"
        )

    if schema == "code_plan":
        return (
            f"{header}\n"
            "Write a JSON object with top-level `stages`. Each code step must use canonical idea "
            "component names, project target paths under `project/`, real verification commands, "
            "and managed handoff artifact ids such as `code.<step_id>.handoff`. The final step must "
            "be `final_integration_smoke` and write bounded smoke evidence through its managed artifact.\n"
            "Repair template:\n"
            "```json\n"
            f"{_json_template(_code_plan_template())}\n"
            "```"
        )

    if schema == "ablation_results":
        return (
            f"{header}\n"
            "`ablation_results.json` is runtime-owned final output. Do not write it from a worker; "
            "repair the upstream reviewer-approved science component results instead."
        )

    return f"{header}\nReturn a JSON object or array matching `{schema}`; repair every issue reported by the hook."


def _schema_guides_for_specs(specs: Iterable[ArtifactSpec]) -> str:
    guides: List[str] = []
    seen: set[tuple[str, str, str]] = set()
    for spec in specs:
        key = (spec.artifact_id, spec.path, spec.schema_name)
        if key in seen:
            continue
        seen.add(key)
        guide = artifact_schema_repair_guide(
            spec.schema_name,
            artifact_id=spec.artifact_id,
            path=spec.path,
        )
        if guide:
            guides.append(guide)
    return "\n\n".join(guides)


def _candidate_feedback_reason(
    *,
    registry: ArtifactRegistry,
    spec: ArtifactSpec,
    issues: Iterable[str],
) -> str:
    return _artifact_feedback_reason(
        title=(
            "Xcientist artifact tool rejected the candidate because it does not "
            "satisfy the managed artifact contract. The existing artifact was not changed."
        ),
        registry=registry,
        spec=spec,
        issues=issues,
    )


def _artifact_feedback_reason(
    *,
    title: str,
    registry: ArtifactRegistry,
    spec: ArtifactSpec,
    issues: Iterable[str],
) -> str:
    issue_lines = [str(issue) for issue in issues if str(issue).strip()]
    schema_guide = artifact_schema_repair_guide(
        spec.schema_name,
        artifact_id=spec.artifact_id,
        path=spec.path,
    )
    return (
        f"{title}\n\n"
        f"Artifact: `{spec.artifact_id}`\n"
        f"Required path: `{spec.path}`\n"
        + (f"Schema: `{spec.schema_name}`\n" if spec.schema_name else "")
        + "Fix this artifact through Xcientist artifact tools, then continue.\n\n"
        + "Issues:\n"
        + ("\n".join(f"- {issue}" for issue in issue_lines) or "- unknown artifact contract failure")
        + (f"\n\nExpected schema / repair template:\n{schema_guide}" if schema_guide else "")
        + f"\n\nRegistry: `{artifact_registry_path(registry.workspace_root)}`"
    )


def _looks_like_experiment_output(path: str) -> bool:
    name = os.path.basename(str(path or "")).lower()
    return name in _OUTPUT_LIKE_FILENAMES or name.endswith(_OUTPUT_LIKE_SUFFIXES)


def _output_root_is_allowed(name: str) -> bool:
    if not name:
        return False
    if name.startswith("."):
        return True
    return name in _OUTPUT_ALLOWED_ROOTS


def scan_workspace_hygiene(workspace_root: str) -> Dict[str, Any]:
    """Find undeclared experiment outputs that should be routed elsewhere."""
    workspace = os.path.realpath(workspace_root)
    issues: List[Dict[str, str]] = []
    for name in sorted(os.listdir(workspace)) if os.path.isdir(workspace) else []:
        path = os.path.join(workspace, name)
        if os.path.isfile(path):
            if _looks_like_experiment_output(name):
                issues.append(
                    {
                        "path": name,
                        "reason": (
                            "Root-level experiment output is not declared. Store raw outputs under "
                            "`results/science/<condition_id>/`, and publish summaries through "
                            "managed artifacts."
                        ),
                        "examples": name,
                    }
                )
            continue

        if not os.path.isdir(path) or _output_root_is_allowed(name):
            continue

        leaked = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [dirname for dirname in dirs if not dirname.startswith(".")]
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), workspace)
                if _looks_like_experiment_output(rel):
                    leaked.append(rel)
                if len(leaked) >= 10:
                    break
            if len(leaked) >= 10:
                break
        if leaked:
            root_rel = os.path.relpath(path, workspace)
            issues.append(
                {
                    "path": f"{root_rel}/",
                    "reason": (
                        "Experiment-like outputs are outside declared output roots. Store raw "
                        "run outputs under `results/science/<condition_id>/` and publish "
                        "reviewable summaries through managed artifacts in `agent_reports/`."
                    ),
                    "examples": ", ".join(sorted(leaked)[:5]),
                }
            )
    payload = {
        "status": "PASS" if not issues else "FAIL",
        "hook": "workspace_hygiene",
        "issues": issues,
    }
    layout = ReportLayout(workspace)
    os.makedirs(os.path.dirname(layout.stray_outputs), exist_ok=True)
    with open(layout.stray_outputs, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _hygiene_hook_result(workspace_root: str) -> Optional[AggregatedHookResult]:
    payload = scan_workspace_hygiene(workspace_root)
    if payload["status"] == "PASS":
        return None
    issue_lines = []
    for issue in payload["issues"]:
        line = f"- `{issue['path']}`: {issue['reason']}"
        examples = issue.get("examples")
        if examples:
            line += f" Examples: {examples}"
        issue_lines.append(line)
    return AggregatedHookResult(
        [
            HookResult(
                hook_type="xcientist_workspace_hygiene",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist workspace hygiene hook found undeclared experiment outputs. "
                    "Fix the command/output paths in this same session before finishing.\n\n"
                    + "\n".join(issue_lines)
                ),
                metadata=payload,
            )
        ]
    )


def _artifact_id_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return slug or "artifact"


def _path_kind(path: str) -> str:
    if path.endswith("/") or not os.path.splitext(path)[1]:
        return "dir"
    if path.endswith(".json"):
        return "json"
    return "file"


def _reports_artifact_path(scope: str, stage: str, suffix: str) -> str:
    return artifact_rel(scope, stage, suffix)


def _is_runtime_report_path(path: str) -> bool:
    normalized = os.path.normpath(str(path or ""))
    if any(normalized.endswith(suffix) for suffix in RUNTIME_REPORT_SUFFIXES):
        return True
    parts = normalized.split(os.sep)
    if "agent_reports" not in parts:
        return False
    index = parts.index("agent_reports")
    rel_parts = parts[index + 1 :]
    if not rel_parts:
        return False
    if rel_parts[0] == "_runtime":
        return True
    if rel_parts[0] not in _REPORT_PHASE_DIRS:
        return False
    if len(rel_parts) == 2 and rel_parts[1] in {"summary.md", "usage.md"}:
        return True
    if len(rel_parts) >= 2 and rel_parts[1] in _REPORT_RUNTIME_ROLES:
        return True
    return False


def _implicit_runtime_report_spec(registry: ArtifactRegistry, candidate: str) -> Optional[ArtifactSpec]:
    candidate = os.path.realpath(candidate)
    reports_dir = os.path.join(registry.workspace_root, "agent_reports")
    if not _path_under(reports_dir, candidate):
        return None
    rel = os.path.relpath(candidate, registry.workspace_root)
    slug = _artifact_id_slug(os.path.splitext(os.path.basename(candidate))[0])
    if not _is_runtime_report_path(candidate):
        slug = "agent_reports_" + slug
    return ArtifactSpec(
        artifact_id=f"runtime.{slug}",
        stage="runtime",
        path=rel,
        kind="json" if candidate.endswith(".json") else "file",
        required=False,
        writer="runtime",
        description="Runtime-owned or controlled agent_reports path.",
    )


def _step_id(step: Dict[str, Any]) -> str:
    return _artifact_id_slug(
        str(step.get("condition_id") or step.get("step_id") or step.get("stage_id") or "step")
    )


def _runtime_report_paths(scope: str, step: Dict[str, Any], workspace_root: str) -> Dict[str, str]:
    step_id = _step_id(step)
    latest = step_report_abs_paths(workspace_root, scope, step_id)
    return {
        "worker_report_path": latest["worker_report_path"],
        "review_report_path": latest["review_report_path"],
        "hook_report_path": latest["hook_report_path"],
    }


def ensure_runtime_report_paths(scope: str, step: Dict[str, Any], workspace_root: str) -> None:
    for key, value in _runtime_report_paths(scope, step, workspace_root).items():
        step[key] = value


def _add_unique_spec(registry: ArtifactRegistry, spec: ArtifactSpec) -> None:
    base = spec.artifact_id
    artifact_id = base
    index = 2
    while artifact_id in registry.specs:
        artifact_id = f"{base}_{index}"
        index += 1
    if artifact_id != spec.artifact_id:
        spec = ArtifactSpec(**{**asdict(spec), "artifact_id": artifact_id})
    registry.add(spec)


def build_step_artifact_registry(
    *,
    workspace_root: str,
    scope: str,
    step: Dict[str, Any],
) -> ArtifactRegistry:
    workspace_root = os.path.realpath(workspace_root)
    ensure_runtime_report_paths(scope, step, workspace_root)
    registry = ArtifactRegistry(workspace_root=workspace_root)
    stage = _step_id(step)

    if scope == "prepare":
        defaults = {
            "repos": [
                ("prepare.discovery", phase_rel("prepare", "artifacts", "discovery.json"), "json", "prepare_discovery", "Prepare resource discovery and selection manifest."),
                ("prepare.repos", phase_rel("prepare", "artifacts", "repos.json"), "json", "prepare_repos", "Prepared repos manifest."),
            ],
            "dataset": [("prepare.dataset", phase_rel("prepare", "artifacts", "dataset.json"), "json", "prepare_dataset", "Prepared dataset manifest.")],
            "model": [("prepare.model", phase_rel("prepare", "artifacts", "model.json"), "json", "prepare_model", "Prepared model/API manifest.")],
            "env": [("prepare.env", phase_rel("prepare", "artifacts", "env.json"), "json", "prepare_env", "Prepared Python environment manifest.")],
            "synthesis": [
                ("prepare.idea", phase_rel("prepare", "artifacts", "idea.md"), "file", "", "Prepared idea handoff."),
                ("prepare.target_inventory", phase_rel("prepare", "artifacts", "target_inventory.json"), "json", "prepare_target_inventory", "Prepared target inventory."),
            ],
        }
        for artifact_id, path, kind, schema_name, desc in defaults.get(stage, []):
            if artifact_id not in registry.specs:
                registry.add(
                    ArtifactSpec(
                        artifact_id=artifact_id,
                        stage=scope,
                        path=path,
                        kind=kind,
                        schema_name=schema_name,
                        description=desc,
                    )
                )

    if scope == "code":
        _add_unique_spec(
            registry,
            ArtifactSpec(
                artifact_id=f"code.{stage}.handoff",
                stage=scope,
                path=_reports_artifact_path(scope, stage, "handoff.json"),
                kind="json",
                schema_name="code_handoff",
                description="Code step handoff manifest for project changes and verification evidence.",
            ),
        )
        if stage == "final_integration_smoke":
            _add_unique_spec(
                registry,
                ArtifactSpec(
                    artifact_id="code.final_integration_smoke.evidence",
                    stage=scope,
                    path=phase_rel("code", "artifacts", "final_integration_smoke.json"),
                    kind="json",
                    schema_name="code_smoke_evidence",
                    required=True,
                    description="Required bounded smoke evidence.",
                ),
            )

    if scope == "science":
        _add_unique_spec(
            registry,
            ArtifactSpec(
                artifact_id=f"{scope}.{stage}.evidence",
                stage=scope,
                path=_reports_artifact_path(scope, stage, "evidence.json"),
                kind="json",
                schema_name="science_evidence",
                description="Science step evidence manifest pointing to raw outputs and logs.",
            ),
        )

    for key in ("worker_report_path", "review_report_path", "hook_report_path"):
        path = str(step.get(key) or "")
        if path:
            _add_unique_spec(
                registry,
                ArtifactSpec(
                    artifact_id=f"runtime.{scope}.{stage}.{key.replace('_path', '')}",
                    stage=scope,
                    path=path,
                    kind="json",
                    required=False,
                    writer="runtime",
                    description="Runtime-owned report.",
                ),
            )

    step["artifact_ids"] = [
        spec.artifact_id
        for spec in registry.specs.values()
        if spec.writer != "runtime"
    ]
    return registry


def artifact_prompt_context(registry: ArtifactRegistry) -> str:
    rows = []
    for spec in registry.specs.values():
        if spec.writer == "runtime":
            continue
        rows.append(
            {
                "artifact_id": spec.artifact_id,
                "path": spec.path,
                "kind": spec.kind,
                "required": spec.required,
                "schema_name": spec.schema_name,
                "description": spec.description,
            }
        )
    scratch = [os.path.relpath(path, registry.workspace_root) for path in registry.scratch_roots]
    schema_guides = _schema_guides_for_specs(registry.worker_required_specs())
    return (
        "## Artifact Registry\n"
        f"Tool boundary: read/search/write only inside the current experiment workspace `{registry.workspace_root}` "
        f"or the current Xcientist-2 source tree `{_REPO_ROOT}` outside `{_REPO_WORKSPACE_ROOT}`. "
        "Do not search `/agent_workspace` broadly "
        "and do not read plans, reports, project files, or results from other historical Xcientist workspaces; "
        "use the copies already present in this workspace.\n\n"
        "Formal artifacts must be written or published with Xcientist artifact tools. "
        "Do not directly write these paths with generic write_file/edit_file/bash.\n"
        "For JSON artifacts, call `write_artifact` with `json_content` as a JSON object/list; "
        "do not place serialized JSON text inside `content`.\n"
        "Use `run_artifact_command` for commands that create a managed directory, "
        "or write temporary files under scratch and publish them with `publish_artifact`.\n\n"
        "Runtime-owned planner, worker, reviewer, hook, phase, and _runtime reports must not be written with tools. "
        "When the stage is complete, stop using tools and return the required structured JSON response; "
        "the runtime will persist reports. The worker report should list touched artifact ids, "
        "not proof paths; the ledger is the proof.\n\n"
        "```json\n"
        f"{json.dumps({'artifacts': rows, 'scratch_roots': scratch}, ensure_ascii=False, indent=2)}\n"
        "```\n"
        + (f"\n## Artifact Schema Guides\n{schema_guides}\n" if schema_guides else "")
    )


def validate_artifact_contract(
    *,
    registry: ArtifactRegistry,
    review_status: str,
    require_ledger: bool = True,
) -> Dict[str, Any]:
    issues: List[str] = []
    ledger = ArtifactLedger(registry.workspace_root)
    records = ledger.read()

    if review_status != "PASS":
        issues.append(f"Reviewer status is {review_status or 'UNKNOWN'}, not PASS.")

    for spec in registry.worker_required_specs():
        issues.extend(_validate_spec_path(registry, spec))
        if require_ledger:
            record = ledger.latest_worker_write_for(spec.artifact_id)
            expected_path = spec.resolved_path(registry.workspace_root)
            if not record:
                issues.append(
                    "Required worker-owned artifact has no artifact-tool ledger write: "
                    f"{spec.artifact_id}. Use write_artifact, patch_artifact, publish_artifact, "
                    "or run_artifact_command for this artifact id, then finish again."
                )
            else:
                recorded_path = os.path.realpath(str(record.get("path") or ""))
                if recorded_path != expected_path:
                    issues.append(
                        f"Ledger path mismatch for `{spec.artifact_id}`: expected `{expected_path}`, "
                        f"latest ledger path is `{recorded_path or 'EMPTY'}`."
                    )
                current_hash = _hash_path(expected_path)
                recorded_hash = str(record.get("sha256") or "").strip()
                if current_hash:
                    if not recorded_hash:
                        issues.append(
                            f"Latest ledger write for `{spec.artifact_id}` is missing sha256; "
                            "rewrite the artifact through an artifact tool so the ledger binds "
                            "the accepted file content."
                        )
                    elif recorded_hash != current_hash:
                        issues.append(
                            f"Ledger hash mismatch for `{spec.artifact_id}` at `{expected_path}`. "
                            f"Latest ledger sha256 is `{recorded_hash}`, current sha256 is `{current_hash}`. "
                            "Repair by rewriting/publishing the managed artifact through artifact tools; "
                            "do not edit controlled artifact paths directly."
                        )
                record_schema_issues = [
                    str(item)
                    for item in record.get("schema_issues") or []
                    if str(item).strip()
                ]
                if record_schema_issues:
                    issues.append(
                        f"Latest ledger write for `{spec.artifact_id}` recorded schema issues: "
                        + "; ".join(record_schema_issues)
                    )

    for record in records:
        if record.get("event") == "policy_violation":
            issues.append(str(record.get("reason") or "Artifact policy violation"))
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "artifact_contract",
        "issues": issues,
        "ledger_path": ledger.path,
        "registry_path": write_artifact_registry_snapshot(registry),
        "checked_artifacts": [asdict(spec) for spec in registry.worker_required_specs()],
    }


class WriteArtifactInput(BaseModel):
    artifact_id: str
    content: Optional[str] = None
    json_content: Optional[Any] = None

    @model_validator(mode="after")
    def exactly_one_content(self) -> "WriteArtifactInput":
        if (self.content is None) == (self.json_content is None):
            raise ValueError("Provide exactly one of content or json_content.")
        return self


class PatchArtifactInput(BaseModel):
    artifact_id: str
    old_str: str
    new_str: str
    replace_all: bool = False


class PublishArtifactInput(BaseModel):
    artifact_id: str
    source_path: str
    mode: str = Field(default="copy", description="copy or move")


class RecordSourcesInput(BaseModel):
    artifact_id: str
    sources: List[str]
    reason: str = ""


class RunArtifactCommandInput(BaseModel):
    artifact_id: str
    command: str
    timeout_seconds: int = Field(default=7200, ge=1, le=600000)


def _registry_from_context(context: ToolExecutionContext) -> ArtifactRegistry:
    metadata = context.metadata or {}
    artifact_context = metadata.get("xcientist_artifact_context")
    return ArtifactRegistry.from_context(artifact_context if isinstance(artifact_context, dict) else {})


def _ledger_from_context(context: ToolExecutionContext, registry: ArtifactRegistry) -> ArtifactLedger:
    metadata = context.metadata or {}
    artifact_context = metadata.get("xcientist_artifact_context")
    path = None
    if isinstance(artifact_context, dict):
        path = str(artifact_context.get("ledger_path") or "")
    return ArtifactLedger(registry.workspace_root, path=path or None)


def _tool_metadata(context: ToolExecutionContext) -> Dict[str, Any]:
    artifact_context = (context.metadata or {}).get("xcientist_artifact_context")
    if not isinstance(artifact_context, dict):
        artifact_context = {}
    return {
        "stage": artifact_context.get("stage", ""),
        "step_id": artifact_context.get("step_id", ""),
        "attempt": artifact_context.get("attempt", 0),
        "agent": (context.metadata or {}).get("xcientist_agent_name", ""),
    }


def _record_artifact_event(
    *,
    context: ToolExecutionContext,
    registry: ArtifactRegistry,
    spec: ArtifactSpec,
    event: str,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ledger = _ledger_from_context(context, registry)
    record = {
        "event": event,
        "artifact_id": spec.artifact_id,
        "path": path,
        "kind": spec.kind,
        "sha256": _hash_path(path) if os.path.exists(path) else "",
        "schema_name": spec.schema_name,
        "schema_issues": _validate_json_schema(path, spec.schema_name) if os.path.exists(path) else [],
        **_tool_metadata(context),
    }
    if extra:
        record.update(extra)
    ledger.append(record)


def _format_registry_artifacts(registry: ArtifactRegistry) -> str:
    lines = []
    for spec in sorted(registry.specs.values(), key=lambda item: item.artifact_id):
        schema_suffix = f" (schema: `{spec.schema_name}`)" if spec.schema_name else ""
        writer_suffix = " [runtime-owned]" if spec.writer == "runtime" else ""
        lines.append(f"- `{spec.artifact_id}` -> `{spec.path}`{schema_suffix}{writer_suffix}")
    return "\n".join(lines) or "- (none)"


def _require_spec(registry: ArtifactRegistry, artifact_id: str) -> tuple[Optional[ArtifactSpec], Optional[str]]:
    spec = registry.get(artifact_id)
    if spec is None:
        return None, (
            f"Unknown artifact_id `{artifact_id}`. Use exactly one artifact id from the current "
            "Artifact Registry; do not use file paths or report names as artifact ids.\n\n"
            f"Known artifacts:\n{_format_registry_artifacts(registry)}\n\n"
            f"Registry: `{artifact_registry_path(registry.workspace_root)}`"
        )
    if spec.writer == "runtime":
        return None, (
            f"Artifact `{artifact_id}` is runtime-owned at `{spec.path}` and cannot be written by a worker. "
            "Return the required structured JSON response directly and let the runtime persist reports."
        )
    return spec, None


class WriteArtifactTool(BaseTool):
    name = "write_artifact"
    description = "Write a managed Xcientist artifact by artifact_id."
    input_model = WriteArtifactInput

    async def execute(self, arguments: WriteArtifactInput, context: ToolExecutionContext) -> ToolResult:
        registry = _registry_from_context(context)
        spec, error = _require_spec(registry, arguments.artifact_id)
        if error or spec is None:
            return ToolResult(output=error or "Invalid artifact.", is_error=True)
        path = spec.resolved_path(registry.workspace_root)
        if not _path_under(registry.workspace_root, path):
            return ToolResult(output=f"Artifact path escapes workspace: {path}", is_error=True)
        if spec.kind == "dir":
            return ToolResult(output="write_artifact cannot write directory artifacts; use run_artifact_command or publish_artifact.", is_error=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = f"{path}.candidate-{int(time.time() * 1000)}-{os.getpid()}"
        if arguments.json_content is not None:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(arguments.json_content, f, ensure_ascii=False, indent=2)
        else:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(arguments.content or "")
        issues = _validate_candidate_artifact(registry, spec, temp_path)
        if issues:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return ToolResult(
                output=_candidate_feedback_reason(registry=registry, spec=spec, issues=issues),
                is_error=True,
            )
        os.replace(temp_path, path)
        _record_artifact_event(context=context, registry=registry, spec=spec, event="write_artifact", path=path)
        return ToolResult(output=f"Wrote artifact {spec.artifact_id}: {path}")


class PatchArtifactTool(BaseTool):
    name = "patch_artifact"
    description = "Patch a managed Xcientist artifact by artifact_id."
    input_model = PatchArtifactInput

    async def execute(self, arguments: PatchArtifactInput, context: ToolExecutionContext) -> ToolResult:
        registry = _registry_from_context(context)
        spec, error = _require_spec(registry, arguments.artifact_id)
        if error or spec is None:
            return ToolResult(output=error or "Invalid artifact.", is_error=True)
        path = spec.resolved_path(registry.workspace_root)
        if not os.path.isfile(path):
            return ToolResult(output=f"Artifact file not found: {path}", is_error=True)
        original = Path(path).read_text(encoding="utf-8")
        if arguments.old_str not in original:
            return ToolResult(output="old_str was not found in the artifact", is_error=True)
        updated = (
            original.replace(arguments.old_str, arguments.new_str)
            if arguments.replace_all
            else original.replace(arguments.old_str, arguments.new_str, 1)
        )
        temp_path = f"{path}.candidate-{int(time.time() * 1000)}-{os.getpid()}"
        Path(temp_path).write_text(updated, encoding="utf-8")
        issues = _validate_candidate_artifact(registry, spec, temp_path)
        if issues:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return ToolResult(
                output=_candidate_feedback_reason(registry=registry, spec=spec, issues=issues),
                is_error=True,
            )
        os.replace(temp_path, path)
        _record_artifact_event(context=context, registry=registry, spec=spec, event="patch_artifact", path=path)
        return ToolResult(output=f"Patched artifact {spec.artifact_id}: {path}")


class PublishArtifactTool(BaseTool):
    name = "publish_artifact"
    description = "Publish a scratch file or directory into a managed Xcientist artifact path."
    input_model = PublishArtifactInput

    async def execute(self, arguments: PublishArtifactInput, context: ToolExecutionContext) -> ToolResult:
        registry = _registry_from_context(context)
        spec, error = _require_spec(registry, arguments.artifact_id)
        if error or spec is None:
            return ToolResult(output=error or "Invalid artifact.", is_error=True)
        source = _resolve_path(registry.workspace_root, arguments.source_path)
        dest = spec.resolved_path(registry.workspace_root)
        if not os.path.exists(source):
            return ToolResult(output=f"Source path does not exist: {source}", is_error=True)
        if not any(_path_under(root, source) for root in registry.scratch_roots):
            return ToolResult(output=f"Source must be under a scratch root before publish: {source}", is_error=True)
        issues = _validate_candidate_artifact(registry, spec, source)
        if issues:
            return ToolResult(
                output=_candidate_feedback_reason(registry=registry, spec=spec, issues=issues),
                is_error=True,
            )
        if spec.kind == "dir":
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if arguments.mode == "move":
                shutil.move(source, dest)
            else:
                shutil.copytree(source, dest)
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if arguments.mode == "move":
                shutil.move(source, dest)
            else:
                shutil.copy2(source, dest)
        _record_artifact_event(
            context=context,
            registry=registry,
            spec=spec,
            event="publish_artifact",
            path=dest,
            extra={"source_path": source, "mode": arguments.mode},
        )
        return ToolResult(output=f"Published artifact {spec.artifact_id}: {dest}")


class RecordSourcesTool(BaseTool):
    name = "record_sources"
    description = "Record source files or URLs used to produce a managed Xcientist artifact."
    input_model = RecordSourcesInput

    async def execute(self, arguments: RecordSourcesInput, context: ToolExecutionContext) -> ToolResult:
        registry = _registry_from_context(context)
        spec, error = _require_spec(registry, arguments.artifact_id)
        if error or spec is None:
            return ToolResult(output=error or "Invalid artifact.", is_error=True)
        source_issues = _source_reference_issues(registry, arguments.sources)
        if source_issues:
            return ToolResult(
                output=_source_reference_reason(registry, source_issues),
                is_error=True,
            )
        ledger = _ledger_from_context(context, registry)
        ledger.append(
            {
                "event": "record_sources",
                "artifact_id": spec.artifact_id,
                "path": spec.resolved_path(registry.workspace_root),
                "sources": list(arguments.sources),
                "reason": arguments.reason,
                **_tool_metadata(context),
            }
        )
        return ToolResult(output=f"Recorded sources for artifact {spec.artifact_id}")


class RunArtifactCommandTool(BaseTool):
    name = "run_artifact_command"
    description = "Run a shell command that creates or updates one managed Xcientist artifact."
    input_model = RunArtifactCommandInput

    async def execute(self, arguments: RunArtifactCommandInput, context: ToolExecutionContext) -> ToolResult:
        registry = _registry_from_context(context)
        spec, error = _require_spec(registry, arguments.artifact_id)
        if error or spec is None:
            return ToolResult(output=error or "Invalid artifact.", is_error=True)
        if _looks_like_cross_workspace_bash(registry, arguments.command):
            return ToolResult(
                output=_bash_boundary_reason(registry, arguments.command),
                is_error=True,
            )
        # Artifact tools can also be invoked directly by hooks and contract
        # tests, without constructing an OpenHarnessRunner first. Keep the
        # harness config/data/log state inside this experiment workspace in
        # that path as well; otherwise OpenHarness falls back to ~/.openharness.
        from src.agents.experiment_agent.runtime.openharness_runner import (
            ensure_openharness_runtime_env,
        )

        runtime_env = ensure_openharness_runtime_env(registry.workspace_root)
        dest = spec.resolved_path(registry.workspace_root)
        os.makedirs(dest if spec.kind == "dir" else os.path.dirname(dest), exist_ok=True)
        process = await create_shell_subprocess(
            arguments.command,
            cwd=Path(registry.workspace_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                **runtime_env,
                "XCIENTIST_ARTIFACT_ID": spec.artifact_id,
                "XCIENTIST_ARTIFACT_PATH": dest,
            },
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=arguments.timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(output=f"Artifact command timed out after {arguments.timeout_seconds}s", is_error=True)
        output = "\n".join(
            part
            for part in (
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            )
            if part
        )
        if process.returncode != 0:
            return ToolResult(output=output or f"Artifact command failed with exit code {process.returncode}", is_error=True)
        issues = _validate_spec_path(registry, spec)
        if issues:
            return ToolResult(
                output=_artifact_feedback_reason(
                    title="Xcientist artifact command finished, but the managed artifact does not satisfy its contract.",
                    registry=registry,
                    spec=spec,
                    issues=issues,
                ),
                is_error=True,
            )
        _record_artifact_event(
            context=context,
            registry=registry,
            spec=spec,
            event="run_artifact_command",
            path=dest,
            extra={"command": arguments.command, "returncode": process.returncode},
        )
        return ToolResult(output=output or f"Ran artifact command for {spec.artifact_id}: {dest}")


def artifact_tools() -> List[BaseTool]:
    return [
        WriteArtifactTool(),
        PatchArtifactTool(),
        PublishArtifactTool(),
        RecordSourcesTool(),
        RunArtifactCommandTool(),
    ]


class XcientistHookExecutor:
    """Runtime hook executor for artifact policy and prefinish gates."""

    def __init__(self, metadata: Dict[str, Any]) -> None:
        self._metadata = metadata
        self._last_successful_artifact_id = ""
        self._consecutive_successful_artifact_writes = 0

    async def execute(self, event: HookEvent, payload: Dict[str, Any]) -> AggregatedHookResult:
        prefinish_results: List[HookResult] = []
        prefinish_gate = self._metadata.get("xcientist_prefinish_gate")
        if event == HookEvent.STOP and callable(prefinish_gate):
            try:
                gate_result = prefinish_gate(payload)
                if hasattr(gate_result, "__await__"):
                    gate_result = await gate_result
            except Exception as exc:
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_prefinish_gate",
                            success=False,
                            blocked=True,
                            reason=(
                                f"Xcientist prefinish gate failed internally: {exc}. "
                                "Do not treat this work unit as complete; finish again with the required "
                                "structured JSON after addressing any visible hook feedback. If this repeats, "
                                "the runtime hook implementation needs inspection."
                            ),
                        )
                    ]
                )
            if isinstance(gate_result, HookResult):
                prefinish_results = [gate_result]
            elif isinstance(gate_result, AggregatedHookResult):
                prefinish_results = list(gate_result.results)
            elif isinstance(gate_result, dict):
                status = str(gate_result.get("status") or "").upper()
                reason = str(gate_result.get("reason") or gate_result.get("next_worker_input") or "").strip()
                prefinish_results = [
                    HookResult(
                        hook_type="xcientist_prefinish_gate",
                        success=status == "PASS",
                        blocked=status != "PASS",
                        reason=reason or "Xcientist prefinish gate blocked completion.",
                        metadata=gate_result,
                    )
                ]
            if AggregatedHookResult(prefinish_results).blocked:
                return AggregatedHookResult(prefinish_results)

        artifact_context = self._metadata.get("xcientist_artifact_context")
        if not isinstance(artifact_context, dict):
            return AggregatedHookResult(prefinish_results)
        registry = ArtifactRegistry.from_context(artifact_context)
        if not registry.specs:
            return AggregatedHookResult(prefinish_results)

        if event == HookEvent.USER_PROMPT_SUBMIT:
            prompt = str(payload.get("prompt") or "")
            if "Artifact Registry" not in prompt:
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_artifact_prompt",
                            success=False,
                            blocked=True,
                            reason=(
                                "Worker/reviewer prompt is missing the injected Artifact Registry section. "
                                "The runtime prompt must include `artifact_prompt_context(...)` for the current "
                                "Artifact Registry before the agent can safely write or review managed artifacts."
                            ),
                        )
                    ]
                )
            return AggregatedHookResult([HookResult(hook_type="xcientist_artifact_prompt", success=True)])

        if event == HookEvent.PRE_TOOL_USE:
            return self._pre_tool_use(registry, payload)

        if event == HookEvent.POST_TOOL_USE:
            return self._post_tool_use(registry, payload)

        if event == HookEvent.STOP:
            stop_result = self._stop(registry)
            return AggregatedHookResult([*prefinish_results, *stop_result.results])

        return AggregatedHookResult(prefinish_results)

    def _pre_tool_use(self, registry: ArtifactRegistry, payload: Dict[str, Any]) -> AggregatedHookResult:
        tool_name = str(payload.get("tool_name") or "")
        tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
        if tool_name in _WORKSPACE_BOUNDARY_TOOLS:
            raw_path = str(
                tool_input.get("path")
                or tool_input.get("file_path")
                or tool_input.get("root")
                or ""
            )
            if tool_name == "glob" and not raw_path:
                pattern = str(tool_input.get("pattern") or "")
                if pattern.startswith("/"):
                    raw_path = pattern
            if raw_path:
                resolved = _resolve_path(registry.workspace_root, raw_path)
                if not _path_allowed_for_tool(registry, resolved):
                    return AggregatedHookResult(
                        [
                            HookResult(
                                hook_type="xcientist_workspace_boundary",
                                success=False,
                                blocked=True,
                                reason=_workspace_boundary_reason(
                                    registry,
                                    tool_name=tool_name,
                                    path=raw_path,
                                ),
                            )
                        ]
                    )
        if tool_name in {"write_artifact", "patch_artifact", "publish_artifact", "record_sources", "run_artifact_command"}:
            artifact_id = str(tool_input.get("artifact_id") or "")
            spec = registry.get(artifact_id)
            if spec is None:
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_artifact_policy",
                            success=False,
                            blocked=True,
                            reason=(
                                f"Unknown artifact_id `{artifact_id}`. Use exactly one artifact id from the "
                                "current Artifact Registry; do not use file paths or report names as artifact ids.\n\n"
                                f"Known artifacts:\n{_format_registry_artifacts(registry)}\n\n"
                                f"Registry: `{artifact_registry_path(registry.workspace_root)}`"
                            ),
                        )
                    ]
                )
            if spec.writer == "runtime":
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_artifact_policy",
                            success=False,
                            blocked=True,
                            reason=(
                                f"Artifact `{artifact_id}` is runtime-owned at `{spec.path}`. "
                                "Do not write planner/worker/reviewer/hook reports with tools; "
                                "return the required structured JSON response directly and let the runtime persist it."
                            ),
                        )
                    ]
                )
            if tool_name == "record_sources":
                raw_sources = tool_input.get("sources")
                sources = raw_sources if isinstance(raw_sources, list) else []
                source_issues = _source_reference_issues(registry, [str(item) for item in sources])
                if raw_sources is not None and not isinstance(raw_sources, list):
                    source_issues.insert(0, "`sources` must be an array of strings.")
                if source_issues:
                    return AggregatedHookResult(
                        [
                            HookResult(
                                hook_type="xcientist_source_provenance",
                                success=False,
                                blocked=True,
                                reason=_source_reference_reason(registry, source_issues),
                            )
                        ]
                    )
            if tool_name == "run_artifact_command":
                command = str(tool_input.get("command") or "")
                if _looks_like_cross_workspace_bash(registry, command):
                    return AggregatedHookResult(
                        [
                            HookResult(
                                hook_type="xcientist_workspace_boundary",
                                success=False,
                                blocked=True,
                                reason=_bash_boundary_reason(registry, command),
                            )
                        ]
                    )
                mentioned = _bash_mentions_managed_write(registry, command)
                if mentioned is not None and mentioned.artifact_id != artifact_id:
                    return AggregatedHookResult(
                        [
                            HookResult(
                                hook_type="xcientist_artifact_policy",
                                success=False,
                                blocked=True,
                                reason=(
                                    f"`run_artifact_command` for `{artifact_id}` appears to write controlled "
                                    f"`agent_reports` path `{mentioned.path}`. This tool may only create or update "
                                    "the declared artifact for its own `artifact_id`. Do not write runtime reports "
                                    "or final artifacts from a worker; return structured JSON and let runtime hooks "
                                    "persist reports/final outputs."
                                ),
                            )
                        ]
                    )
            return AggregatedHookResult([HookResult(hook_type="xcientist_artifact_policy", success=True)])

        if tool_name in {"write_file", "edit_file"}:
            raw_path = str(tool_input.get("path") or tool_input.get("file_path") or "")
            if raw_path:
                resolved = _resolve_path(registry.workspace_root, raw_path)
                blocked = _managed_path_hit(registry, resolved)
                if blocked:
                    if blocked.writer == "runtime":
                        reason = (
                            f"Direct write to runtime-owned report path is blocked: {raw_path}. "
                            "All paths under `agent_reports/` are controlled: write only the current "
                            "worker-owned artifact through artifact tools, and do not create "
                            "planner/worker/reviewer/hook reports or final outputs with generic tools. "
                            "Return the required structured JSON response directly; the runtime will persist reports."
                        )
                    else:
                        schema_guide = artifact_schema_repair_guide(
                            blocked.schema_name,
                            artifact_id=blocked.artifact_id,
                            path=blocked.path,
                        )
                        reason = (
                            f"Direct write to managed artifact path is blocked: {raw_path}. "
                            f"Use write_artifact/patch_artifact/publish_artifact for `{blocked.artifact_id}` "
                            "so the candidate is schema-checked before it is accepted."
                            + (f"\n\nExpected schema / repair template:\n{schema_guide}" if schema_guide else "")
                        )
                    return AggregatedHookResult(
                        [
                            HookResult(
                                hook_type="xcientist_artifact_policy",
                                success=False,
                                blocked=True,
                                reason=reason,
                            )
                        ]
                    )

        if tool_name == "bash":
            command = str(tool_input.get("command") or "")
            if _looks_like_cross_workspace_bash(registry, command):
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_workspace_boundary",
                            success=False,
                            blocked=True,
                            reason=_bash_boundary_reason(registry, command),
                        )
                    ]
                )
            blocked = _bash_mentions_managed_write(registry, command)
            if blocked:
                if blocked.writer == "runtime":
                    reason = (
                        f"bash appears to write runtime-owned report `{blocked.artifact_id}`. "
                        "Do not create planner/worker/reviewer/hook reports with tools. "
                        "Return the required structured JSON response directly; the runtime will persist reports."
                    )
                else:
                    schema_guide = artifact_schema_repair_guide(
                        blocked.schema_name,
                        artifact_id=blocked.artifact_id,
                        path=blocked.path,
                    )
                    reason = (
                        f"bash appears to write managed artifact `{blocked.artifact_id}`. "
                        "Use run_artifact_command for commands that create managed artifacts, "
                        "or write under scratch then publish_artifact. "
                        "Do not redirect shell output directly into the managed path."
                        + (f"\n\nExpected schema / repair template:\n{schema_guide}" if schema_guide else "")
                    )
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_artifact_policy",
                            success=False,
                            blocked=True,
                            reason=reason,
                        )
                    ]
                )
        return AggregatedHookResult([HookResult(hook_type="xcientist_artifact_policy", success=True)])

    def _stop(self, registry: ArtifactRegistry) -> AggregatedHookResult:
        hygiene = _hygiene_hook_result(registry.workspace_root)
        if hygiene is not None:
            return hygiene
        contract = validate_artifact_contract(registry=registry, review_status="PASS")
        if contract["status"] == "PASS":
            return AggregatedHookResult([HookResult(hook_type="xcientist_artifact_prefinish", success=True)])
        issues = [str(issue) for issue in contract.get("issues") or [] if str(issue).strip()]
        artifact_lines = []
        for spec in registry.worker_required_specs():
            schema_suffix = f" (schema: `{spec.schema_name}`)" if spec.schema_name else ""
            artifact_lines.append(f"- `{spec.artifact_id}` -> `{spec.path}`{schema_suffix}")
        schema_guides = _schema_guides_for_specs(registry.worker_required_specs())
        reason = (
            "Xcientist prefinish hook blocked completion because managed artifacts do not "
            "satisfy their contract. Fix the artifacts now, using artifact tools only, "
            "then finish again.\n\n"
            "Managed artifacts:\n"
            + ("\n".join(artifact_lines) or "- (none)")
            + "\n\nIssues:\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + (f"\n\nExpected schema / repair templates:\n{schema_guides}" if schema_guides else "")
        )
        return AggregatedHookResult(
            [
                HookResult(
                    hook_type="xcientist_artifact_prefinish",
                    success=False,
                    blocked=True,
                    reason=reason,
                    metadata={"contract": contract},
                )
            ]
        )

    def _post_tool_use(self, registry: ArtifactRegistry, payload: Dict[str, Any]) -> AggregatedHookResult:
        tool_name = str(payload.get("tool_name") or "")
        tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
        if tool_name in {"write_artifact", "patch_artifact", "publish_artifact", "run_artifact_command"}:
            artifact_id = str(tool_input.get("artifact_id") or "")
            spec = registry.get(artifact_id)
            if spec is None or spec.writer == "runtime":
                return AggregatedHookResult([])
            issues = _validate_spec_path(registry, spec)
            if issues:
                reason = _artifact_feedback_reason(
                    title="Xcientist post-tool hook blocked the artifact write because the managed artifact does not satisfy its contract.",
                    registry=registry,
                    spec=spec,
                    issues=issues,
                )
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_artifact_post_tool",
                            success=False,
                            blocked=True,
                            reason=reason,
                            metadata={"artifact_id": artifact_id, "issues": issues},
                        )
                    ]
                )
            if artifact_id == self._last_successful_artifact_id:
                self._consecutive_successful_artifact_writes += 1
            else:
                self._last_successful_artifact_id = artifact_id
                self._consecutive_successful_artifact_writes = 1
            if self._consecutive_successful_artifact_writes >= 3:
                return AggregatedHookResult(
                    [
                        HookResult(
                            hook_type="xcientist_artifact_completion_hint",
                            success=False,
                            blocked=True,
                            reason=(
                                f"Managed artifact `{artifact_id}` has already been written successfully "
                                f"{self._consecutive_successful_artifact_writes} times in a row and satisfies "
                                "its contract. Stop calling artifact tools for this artifact now. Return the "
                                "required final JSON response directly so the STOP/prefinish hooks can run."
                            ),
                            metadata={"artifact_id": artifact_id},
                        )
                    ]
                )
            return AggregatedHookResult([HookResult(hook_type="xcientist_artifact_post_tool", success=True)])

        if tool_name not in {"bash", "write_file", "edit_file"}:
            self._last_successful_artifact_id = ""
            self._consecutive_successful_artifact_writes = 0
            return AggregatedHookResult([])
        self._last_successful_artifact_id = ""
        self._consecutive_successful_artifact_writes = 0
        if tool_name == "bash":
            hygiene = _hygiene_hook_result(registry.workspace_root)
            if hygiene is not None:
                return hygiene
        output = str(payload.get("tool_output") or "")
        if "Wrote " in output or "Updated " in output:
            for spec in registry.specs.values():
                path = spec.resolved_path(registry.workspace_root)
                if path and path in output:
                    ArtifactLedger(registry.workspace_root).append(
                        {
                            "event": "policy_violation",
                            "artifact_id": spec.artifact_id,
                            "path": path,
                            "tool": tool_name,
                            "reason": "Generic tool wrote managed artifact path.",
                        }
                    )
                    schema_guide = artifact_schema_repair_guide(
                        spec.schema_name,
                        artifact_id=spec.artifact_id,
                        path=spec.path,
                    )
                    return AggregatedHookResult(
                        [
                            HookResult(
                                hook_type="xcientist_artifact_policy",
                                success=False,
                                blocked=True,
                                reason=(
                                    f"Generic tool wrote managed artifact path for `{spec.artifact_id}`. "
                                    "This write is not accepted. Use artifact tools for worker-owned artifacts, "
                                    "or return structured JSON for runtime-owned reports."
                                    + (f"\n\nExpected schema / repair template:\n{schema_guide}" if schema_guide else "")
                                ),
                            )
                        ]
                    )
        return AggregatedHookResult([HookResult(hook_type="xcientist_artifact_policy", success=True)])


def _managed_path_hit(registry: ArtifactRegistry, candidate: str) -> Optional[ArtifactSpec]:
    for spec in registry.specs.values():
        path = spec.resolved_path(registry.workspace_root)
        if not path:
            continue
        if candidate == path or _path_under(path, candidate):
            return spec
    return _implicit_runtime_report_spec(registry, candidate)


def _bash_mentions_managed_write(registry: ArtifactRegistry, command: str) -> Optional[ArtifactSpec]:
    write_markers = (">", ">>", "tee ", "cp ", "mv ", "touch ", "mkdir ", "python -m venv", "pip install")
    if not any(marker in command for marker in write_markers):
        return None
    for spec in registry.specs.values():
        path = spec.path
        resolved = spec.resolved_path(registry.workspace_root)
        candidates = {path, resolved, os.path.relpath(resolved, registry.workspace_root)}
        if any(candidate and candidate in command for candidate in candidates):
            return spec
    report_pattern = (
        r"(?:[A-Za-z0-9_./-]*agent_reports/"
        r"(?:_runtime/[A-Za-z0-9_.-]+|"
        r"(?:prepare|code|science|ablation)/"
        r"(?:plan/[A-Za-z0-9_.-]+|"
        r"(?:worker|review|hook)/[A-Za-z0-9_.-]+/(?:latest\.json|attempts/[0-9]{3}\.json)|"
        r"final/[A-Za-z0-9_.-]+|"
        r"artifacts/[A-Za-z0-9_./-]+|"
        r"evidence/[A-Za-z0-9_./-]+|"
        r"(?:phase\.json|summary\.md|usage\.md)))"
        r")"
    )
    for match in re.findall(report_pattern, command):
        resolved = _resolve_path(registry.workspace_root, match)
        spec = _implicit_runtime_report_spec(registry, resolved)
        if spec is not None:
            return spec
    return None


def build_xcientist_hook_executor(metadata: Dict[str, Any]) -> XcientistHookExecutor:
    return XcientistHookExecutor(metadata)


__all__ = [
    "ArtifactLedger",
    "ArtifactRegistry",
    "ArtifactSpec",
    "artifact_ledger_path",
    "artifact_prompt_context",
    "artifact_registry_path",
    "artifact_schema_repair_guide",
    "artifact_tools",
    "build_step_artifact_registry",
    "build_xcientist_hook_executor",
    "ensure_runtime_report_paths",
    "record_runtime_artifact",
    "validate_artifact_contract",
    "write_artifact_registry_snapshot",
]
