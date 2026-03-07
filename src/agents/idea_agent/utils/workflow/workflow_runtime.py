"""Explicit stage-graph runtime for LigAgent workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from src.agents.idea_agent.utils.workflow.stage_contract import (
    ArtifactPatch,
    StageContext,
    StageResult,
    StageStatus,
)


StageHandler = Callable[[StageContext], StageResult]
EdgeCondition = Callable[[StageContext, StageResult], bool]


@dataclass
class WorkflowEdge:
    target: str
    when: Optional[EdgeCondition] = None


@dataclass
class StageSpec:
    name: str
    handler: StageHandler
    description: str = ""
    record_step: bool = False
    retry_limit: int = 0
    fallback_stage: Optional[str] = None


@dataclass
class WorkflowSpec:
    name: str
    entry_stage: str
    stages: Dict[str, StageSpec]
    transitions: Dict[str, List[WorkflowEdge]] = field(default_factory=dict)


@dataclass
class WorkflowRunResult:
    workflow_name: str
    status: StageStatus
    terminal_stage: Optional[str]
    trace: List[Dict[str, Any]]
    state: Dict[str, Any]


def _deep_merge_dict(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge_dict(base[key], deepcopy(value))
            continue
        base[key] = deepcopy(value)
    return base


class WorkflowExecutor:
    """Runs explicit workflow specs and records stage-level trace."""

    def __init__(self, logger, *, trace_key: str = "workflow_trace", max_steps: int = 128) -> None:
        self.logger = logger
        self.trace_key = trace_key
        self.max_steps = max(1, int(max_steps))

    def run(
        self,
        spec: WorkflowSpec,
        context: StageContext,
        *,
        initial_state: Optional[Dict[str, Any]] = None,
        raise_on_failure: bool = True,
    ) -> WorkflowRunResult:
        if spec.entry_stage not in spec.stages:
            raise ValueError(f"Workflow '{spec.name}' has unknown entry stage '{spec.entry_stage}'.")

        trace: List[Dict[str, Any]] = []
        state: Dict[str, Any] = deepcopy(initial_state or {})
        artifact = context.artifact
        artifact.setdefault(self.trace_key, [])
        artifact.setdefault("workflow_state", {})

        current_stage = spec.entry_stage
        terminal_stage: Optional[str] = None
        overall_status: StageStatus = "success"
        stage_count = 0

        while current_stage is not None:
            stage_count += 1
            if stage_count > self.max_steps:
                raise RuntimeError(
                    f"Workflow '{spec.name}' exceeded max_steps={self.max_steps}; possible graph cycle."
                )
            if current_stage not in spec.stages:
                raise ValueError(
                    f"Workflow '{spec.name}' references unknown stage '{current_stage}'."
                )

            stage_spec = spec.stages[current_stage]
            attempt = 1
            while True:
                stage_ctx = context.for_stage(current_stage, state=state, attempt=attempt)
                artifact["workflow_state"] = {
                    "workflow": spec.name,
                    "stage": current_stage,
                    "status": "running",
                    "attempt": attempt,
                }
                started_at = perf_counter()
                result = stage_spec.handler(stage_ctx)
                duration_ms = round((perf_counter() - started_at) * 1000.0, 2)

                self._apply_state_patch(state, result.state_patch)
                self._apply_artifact_patch(artifact, result.artifact_patch)

                next_stage = self._resolve_next_stage(
                    spec=spec,
                    current_stage=current_stage,
                    context=context.for_stage(current_stage, state=state, attempt=attempt),
                    result=result,
                    fallback_stage=stage_spec.fallback_stage,
                    attempt=attempt,
                    retry_limit=stage_spec.retry_limit,
                )

                if stage_spec.record_step and result.step_summary:
                    artifact.setdefault("steps", []).append(result.step_summary)

                trace_entry = {
                    "workflow": spec.name,
                    "stage": current_stage,
                    "status": result.status,
                    "attempt": attempt,
                    "duration_ms": duration_ms,
                    "next_stage": next_stage,
                    "error": result.error,
                    "metrics": deepcopy(result.metrics),
                    "state_patch_keys": sorted(result.state_patch.keys()),
                    "artifact_patch": {
                        "replace": sorted(result.artifact_patch.replace.keys()),
                        "append": {
                            key: len(value) for key, value in result.artifact_patch.append.items()
                        },
                        "merge": sorted(result.artifact_patch.merge.keys()),
                    },
                }
                if result.step_summary:
                    trace_entry["step_summary"] = result.step_summary
                trace.append(trace_entry)
                artifact[self.trace_key].append(trace_entry)

                if result.status == "retryable_failure" and attempt <= stage_spec.retry_limit:
                    attempt += 1
                    overall_status = "degraded"
                    continue

                if result.status in {"degraded", "retryable_failure"}:
                    overall_status = "degraded"

                if result.status == "terminal_failure" or (
                    result.status == "retryable_failure"
                    and next_stage is None
                    and attempt > stage_spec.retry_limit
                ):
                    terminal_stage = current_stage
                    artifact["workflow_state"] = {
                        "workflow": spec.name,
                        "stage": current_stage,
                        "status": "failed",
                        "attempt": attempt,
                        "error": result.error,
                    }
                    failure_result = WorkflowRunResult(
                        workflow_name=spec.name,
                        status="terminal_failure",
                        terminal_stage=terminal_stage,
                        trace=trace,
                        state=state,
                    )
                    if raise_on_failure:
                        raise RuntimeError(
                            f"Workflow '{spec.name}' failed at stage '{current_stage}': "
                            f"{result.error or 'no error details'}"
                        )
                    return failure_result

                terminal_stage = current_stage
                current_stage = next_stage
                break

        artifact["workflow_state"] = {
            "workflow": spec.name,
            "stage": terminal_stage,
            "status": overall_status,
        }
        return WorkflowRunResult(
            workflow_name=spec.name,
            status=overall_status,
            terminal_stage=terminal_stage,
            trace=trace,
            state=state,
        )

    def _resolve_next_stage(
        self,
        *,
        spec: WorkflowSpec,
        current_stage: str,
        context: StageContext,
        result: StageResult,
        fallback_stage: Optional[str],
        attempt: int,
        retry_limit: int,
    ) -> Optional[str]:
        if result.status == "retryable_failure" and attempt > retry_limit and fallback_stage:
            return fallback_stage
        if result.next_stage is not None:
            return result.next_stage
        for edge in spec.transitions.get(current_stage, []):
            if edge.when is None or edge.when(context, result):
                return edge.target
        return None

    def _apply_state_patch(self, state: Dict[str, Any], patch: Dict[str, Any]) -> None:
        if not patch:
            return
        _deep_merge_dict(state, patch)

    def _apply_artifact_patch(self, artifact: Dict[str, Any], patch: ArtifactPatch) -> None:
        if patch.is_empty():
            return

        for key, value in patch.replace.items():
            artifact[key] = deepcopy(value)

        for key, items in patch.append.items():
            target = artifact.setdefault(key, [])
            if not isinstance(target, list):
                raise TypeError(f"Artifact key '{key}' is not a list and cannot be appended to.")
            target.extend(deepcopy(items))

        for key, value in patch.merge.items():
            target = artifact.setdefault(key, {})
            if not isinstance(target, dict):
                raise TypeError(f"Artifact key '{key}' is not a dict and cannot be merged into.")
            _deep_merge_dict(target, value)
