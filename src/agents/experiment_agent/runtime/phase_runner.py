from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from src.agents.experiment_agent.runtime.manifests import write_json_file
from src.agents.experiment_agent.runtime.phase_contracts import ARTIFACT_ROLE_PHASE_RESULT


def worker_output_schema() -> Dict[str, Any]:
    """Minimal worker output: what was done, what was produced, what's still blocking.
    The orchestrator synthesises executor/audit metadata from this + step context.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "artifacts_produced": {"type": "array", "items": {"type": "string"}},
            "remaining_blockers": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "summary",
            "artifacts_produced",
            "remaining_blockers",
        ],
    }


def validator_output_schema(*, include_ablation_fields: bool = False) -> Dict[str, Any]:
    """Minimal validator output: verdict, reasoning, and repair feedback.
    All orchestration fields (status normalisation, phase completion, gates,
    provenance, self-containment) are synthesised by the runtime in
    ``with_phase_defaults`` so agents don't burn context on boilerplate.
    """
    properties: Dict[str, Any] = {
        "status": {"type": "string", "enum": ["PASS", "PARTIAL", "FAIL"]},
        "evidence_summary": {"type": "string"},
        "required_fixes": {"type": "array", "items": {"type": "string"}},
        "terminal_blocker": {"type": "boolean"},
        "next_worker_input": {"type": "string"},
        "checked_artifacts": {"type": "array", "items": {"type": "string"}},
    }
    required = ["status", "evidence_summary", "required_fixes"]
    if include_ablation_fields:
        properties.update(
            {
                "result": {"type": "string"},
                "metric": {"type": "string"},
                "value": {"type": "string"},
                "confidence": {"type": "number"},
                "analysis": {"type": "string"},
                "method_context": {"type": "string"},
                "follow_up_required": {"type": "boolean"},
            }
        )
        required.extend(
            ["result", "metric", "value", "confidence", "analysis", "method_context", "follow_up_required"]
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def planner_output_schema(*, step_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "stages": {"type": "array", "items": step_schema},
            "summary": {"type": "string"},
            "usage_notes": {"type": "string"},
        },
        "required": ["stages", "summary", "usage_notes"],
    }


def with_phase_defaults(payload: Dict[str, Any], *, scope: str) -> Dict[str, Any]:
    merged = dict(payload)
    merged.setdefault("scope", scope)
    merged.setdefault("checked_artifacts", [])
    merged.setdefault("findings", [])
    merged.setdefault("required_fixes", [])
    merged.setdefault("evidence_summary", "")
    merged.setdefault("phase_completion_status", "partial")
    merged.setdefault("ready_for_next_phase", False)
    merged.setdefault("blocking_issues", [])
    merged.setdefault("required_followup", [])
    merged.setdefault("artifact_role", ARTIFACT_ROLE_PHASE_RESULT)
    merged.setdefault("run_level", "full")
    merged.setdefault("self_contained_project", True)
    merged.setdefault("self_contained_violations", [])
    merged.setdefault("provenance_manifest_present", False)
    merged.setdefault("provenance_manifest_path", "")
    merged.setdefault("terminal_blocker", False)
    merged.setdefault("next_worker_input", "")
    return merged


async def execute_step_loop(
    *,
    steps: List[Dict[str, Any]],
    scope: str,
    workspace_root: str,
    call_worker: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any],
    call_validator: Callable[[Dict[str, Any], Dict[str, Any]], Any],
) -> Dict[str, Any]:
    final_reports: List[Dict[str, Any]] = []
    total_steps = len(steps)
    for idx, step in enumerate(steps, start=1):
        step_id = step.get("step_id") or step.get("stage_id") or f"step-{idx}"
        print(f"\n[{scope}] Step {idx}/{total_steps}: {step_id}", flush=True)
        max_rounds = int(step.get("max_repair_rounds") or 1)
        validator_payload: Optional[Dict[str, Any]] = None
        for attempt in range(1, max_rounds + 1):
            label = f"attempt {attempt}/{max_rounds}" if max_rounds > 1 else "run"
            t0 = time.monotonic()
            print(f"  [{scope}]   Worker ({label})... ", end="", flush=True)
            worker_payload = await call_worker(step, validator_payload)
            elapsed = time.monotonic() - t0
            print(f"done ({elapsed:.0f}s)", flush=True)
            worker_report_path = str(step.get("worker_report_path") or "")
            if worker_report_path:
                write_json_file(worker_report_path, worker_payload)
            t1 = time.monotonic()
            print(f"  [{scope}]   Validator ({label})... ", end="", flush=True)
            validator_payload = await call_validator(step, worker_payload)
            validator_payload = with_phase_defaults(validator_payload, scope=scope)
            elapsed_v = time.monotonic() - t1
            verdict = validator_payload.get("status", "UNKNOWN")
            print(f"done ({elapsed_v:.0f}s) verdict={verdict}", flush=True)
            validator_report_path = str(step.get("validator_report_path") or "")
            if validator_report_path:
                write_json_file(validator_report_path, validator_payload)
            executor_report_path = str(step.get("executor_report_path") or "")
            if executor_report_path:
                # Resolve relative paths against workspace_root
                if not os.path.isabs(executor_report_path):
                    executor_report_path = os.path.join(workspace_root, executor_report_path)
                write_json_file(
                    executor_report_path,
                    {
                        "scope": scope,
                        "step_id": step.get("step_id") or step.get("stage_id"),
                        "attempt": attempt,
                        "worker_report_path": worker_report_path,
                        "validator_report_path": validator_report_path,
                        "status": validator_payload.get("status"),
                    },
                )
            if validator_payload.get("status") == "PASS":
                print(f"  [{scope}]   => PASS", flush=True)
                final_reports.append(validator_payload)
                break
            if validator_payload.get("terminal_blocker") or attempt == max_rounds:
                print(f"  [{scope}]   => FAIL (terminal_blocker={validator_payload.get('terminal_blocker')})", flush=True)
                return {
                    "status": "FAIL",
                    "failed_step": step,
                    "failed_validator_report": validator_payload,
                    "step_reports": final_reports,
                }
            summary = validator_payload.get("evidence_summary", "")
            if summary:
                print(f"  [{scope}]   => {verdict} (repair needed): {summary[:150]}", flush=True)
            else:
                print(f"  [{scope}]   => {verdict} (repair needed)", flush=True)
        else:
            print(f"  [{scope}]   => FAIL (all {max_rounds} rounds exhausted)", flush=True)
            return {
                "status": "FAIL",
                "failed_step": step,
                "failed_validator_report": validator_payload or {},
                "step_reports": final_reports,
            }
    print(f"\n[{scope}] All {total_steps} steps completed.", flush=True)
    return {"status": "PASS", "step_reports": final_reports}
