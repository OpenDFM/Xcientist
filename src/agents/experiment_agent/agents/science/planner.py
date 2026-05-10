"""
Claude Code-backed science planners.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.agents.experiment_agent.agents.base.agent import BaseAgent
from src.agents.experiment_agent.agents.science.validator import (
    ABLATION_SCIENCE_VALIDATOR,
    STANDARD_SCIENCE_VALIDATOR,
    ablation_science_validator_prompt,
    standard_science_validator_prompt,
)
from src.agents.experiment_agent.agents.science.worker import (
    ABLATION_SCIENCE_WORKER,
    STANDARD_SCIENCE_WORKER,
    ablation_science_worker_prompt,
    standard_science_worker_prompt,
)
from src.agents.experiment_agent.config import get_agent_model
from src.agents.experiment_agent.runtime.contracts import (
    ABLATION_COMPONENT_RESULT_FIELDS,
    PHASE_VERDICT_FIELDS,
    SCIENCE_ABLATION_STEP_FIELDS,
    SCIENCE_STANDARD_STEP_FIELDS,
    format_field_bullets,
    format_named_paths,
    validate_repo_contract_fields,
)
from src.agents.experiment_agent.runtime.idea_components import (
    format_canonical_components_markdown,
    load_canonical_components,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, write_json_file, workspace_contract_paths
from src.agents.experiment_agent.runtime.phase_runner import (
    execute_step_loop,
    planner_output_schema,
    validator_output_schema,
    worker_output_schema,
    with_phase_defaults,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report


EXPERIMENT_STANDARD_SCIENCE_PLANNER = "experiment_standard_science_planner"
EXPERIMENT_ABLATION_SCIENCE_PLANNER = "experiment_ablation_science_planner"


def _science_step_schema(*, ablation: bool) -> Dict[str, Any]:
    properties = {
        "step_id": {"type": "string"},
        "goal": {"type": "string"},
        "step_contract_path": {"type": "string"},
        "executor_report_path": {"type": "string"},
        "repo_source_paths": {"type": "array", "items": {"type": "string"}},
        "repo_copy_intent": {"type": "string"},
        "project_target_paths": {"type": "array", "items": {"type": "string"}},
        "input_paths": {"type": "object"},
        "allowed_write_roots": {"type": "array", "items": {"type": "string"}},
        "required_output_roots": {"type": "array", "items": {"type": "string"}},
        "worker_report_path": {"type": "string"},
        "validator_report_path": {"type": "string"},
        "repos_policy": {"type": "string"},
        "project_must_be_self_contained": {"type": "boolean"},
        "provenance_manifest_path": {"type": "string"},
        "command": {"type": "string"},
        "output_dir": {"type": "string"},
        "raw_evidence": {"type": "array", "items": {"type": "string"}},
        "max_repair_rounds": {"type": "integer"},
        "pass_condition": {"type": "string"},
    }
    if ablation:
        properties.update(
            {
                "component_or_condition": {"type": "string"},
                "canonical_component_index": {"type": "string"},
                "component_explanation": {"type": "string"},
                "method_context_change": {"type": "string"},
            }
        )
    else:
        properties["target_scope"] = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }


def create_standard_science_planner_agent(llm):
    _ = llm
    return {"role": EXPERIMENT_STANDARD_SCIENCE_PLANNER}


def create_ablation_science_planner_agent(llm):
    _ = llm
    return {"role": EXPERIMENT_ABLATION_SCIENCE_PLANNER}


def register_science_planners() -> None:
    return None


class _BaseSciencePlanner(BaseAgent):
    planner_name = ""
    planner_role = ""
    summary_key = ""
    plan_key = ""
    validator_key = ""
    worker_role = ""
    validator_role = ""
    step_fields: tuple[str, ...] = ()
    ablation = False

    def __init__(
        self,
        experiment_id: str,
        idea_path: str,
        project_root: str,
        workspace_root: str,
        plan: str,
        code_summary: str = "",
        code_usage: str = "",
        model: str | None = None,
        verbose: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            agent_type=self.planner_name,
            model=model or get_agent_model("standard_science_agent" if not self.ablation else "ablation_science_agent", "science"),
            verbose=verbose,
            workspace_root=workspace_root,
            resume=resume,
        )
        self.experiment_id = experiment_id
        self.idea_path = idea_path
        self.project_root = project_root
        self.workspace_root = workspace_root
        self.plan = plan
        self.code_summary = code_summary
        self.code_usage = code_usage
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.report_path = self.paths[self.summary_key]
        self.plan_path = self.paths[self.plan_key]
        self.validator_report_path = self.paths[self.validator_key]
        self.canonical_components = load_canonical_components(workspace_root)

    def _input_paths(self) -> Dict[str, str]:
        return {
            "idea_path": self.idea_path,
            "prepare_validator_path": self.paths["prepare_validator"],
            "code_validator_path": self.paths["code_validator"],
            "code_summary_path": self.paths["code_summary"],
        }

    def _path_contract(self) -> Dict[str, str]:
        payload = {
            "workspace_dir": self.contract["workspace_dir"],
            "project_dir": self.contract["project_dir"],
            "dataset_dir": self.contract["dataset_dir"],
            "model_dir": self.contract["model_dir"],
            "results_dir": self.contract["results_dir"],
            "agent_reports_dir": self.contract["agent_reports_dir"],
        }
        if self.ablation:
            payload["ablation_results_dir"] = self.contract["ablation_results_dir"]
        else:
            payload["standard_results_dir"] = self.contract["standard_results_dir"]
        return payload

    def _build_user_prompt(self) -> str:
        step_fields = format_field_bullets(self.step_fields)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        blueprint_path = self.paths.get("ablation_science_blueprint" if self.ablation else "standard_science_blueprint", "")
        blueprint_instruction = ""
        if blueprint_path:
            blueprint_instruction = f"""
- After exploring the workspace, write a blueprint to `{blueprint_path}` capturing:
  - What you discovered about the project's training/evaluation scripts
  - Why you chose each step's approach
  - Key implementation decisions and trade-offs
  - What the worker should pay attention to in each step
"""
        base = [
            f"## Task: Run {'Ablation' if self.ablation else 'Standard'} Science",
            "",
            "### Master Plan",
            self.plan,
            "",
            "### Input Paths",
            format_named_paths(self._input_paths()),
            "",
            "### Path Contract",
            format_named_paths(self._path_contract()),
            "",
            "### Required Planner Output",
            "Write a JSON object with `stages`, `summary`, and `usage_notes`.",
            "Each step must include:",
            step_fields,
            "",
            blueprint_instruction,
            f"The runtime will execute each step through `{self.worker_role}` and review it through `{self.validator_role}`.",
            "The final phase-level validator report must include:",
            verdict_fields,
        ]
        if self.ablation:
            base.extend(
                [
                    "",
                    "### Canonical Idea Components",
                    format_canonical_components_markdown(self.canonical_components),
                    "",
                    "Rules:",
                    "- The number of steps must equal the number of canonical components.",
                    "- Component names and order must match `idea.json.components` exactly.",
                    "- Do not write `ablation_results.json`; preserve evidence for later deterministic materialization.",
                    "- Every step validator report must contain component result fields.",
                    format_field_bullets(ABLATION_COMPONENT_RESULT_FIELDS),
                ]
            )
        return "\n".join(base)

    async def _plan(self) -> Dict[str, Any]:
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            agent_name=self.planner_role,
            output_schema=planner_output_schema(step_schema=_science_step_schema(ablation=self.ablation)),
        )
        payload = result["output"]
        write_json_file(self.plan_path, {"stages": payload["stages"]})
        blueprint_key = "ablation_science_blueprint" if self.ablation else "standard_science_blueprint"
        blueprint_path = self.paths.get(blueprint_key, "")
        write_json_file(
            self.paths["ablation_science_planner_report" if self.ablation else "standard_science_planner_report"],
            {
                "scope": "ablation_science" if self.ablation else "standard_science",
                "summary": payload["summary"],
                "usage_notes": payload["usage_notes"],
                "plan_path": self.plan_path,
                "blueprint_path": blueprint_path,
            },
        )
        # Write a default blueprint if the planner didn't produce one directly.
        if not blueprint_path or not os.path.exists(blueprint_path):
            self._synthesize_blueprint(blueprint_path or "", payload)
        return payload

    def _synthesize_blueprint(self, blueprint_path: str, payload: Dict[str, Any]) -> None:
        """Generate a blueprint.md from the plan payload when the planner didn't produce one directly."""
        if not blueprint_path:
            return
        lines = [
            f"# {'Ablation' if self.ablation else 'Standard'} Science Blueprint",
            "",
            f"## Planner Summary",
            "",
            payload.get("summary", "(none)"),
            "",
            f"## Planned Steps ({len(payload.get('stages', []))} steps)",
            "",
        ]
        for step in payload.get("stages", []):
            lines.append(f"### `{step.get('step_id', '?')}`")
            lines.append("")
            lines.append(f"**Goal:** {step.get('goal', '')}")
            lines.append("")
            if step.get('pass_condition'):
                lines.append(f"**Pass when:** {step['pass_condition']}")
                lines.append("")
            if step.get('command'):
                lines.append(f"**Command:** `{step['command']}`")
                lines.append("")
            if step.get('project_target_paths'):
                lines.append(f"**Target paths:** {', '.join(step['project_target_paths'])}")
                lines.append("")
        lines.append(f"## Usage Notes\n\n{payload.get('usage_notes', '')}\n")
        os.makedirs(os.path.dirname(blueprint_path), exist_ok=True)
        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _validate_steps(self, steps: List[Dict[str, Any]]) -> None:
        errors: List[str] = []
        for index, step in enumerate(steps, start=1):
            errors.extend(
                f"step {index}: {message}"
                for message in validate_repo_contract_fields(step, project_dir=self.project_root)
            )
        if self.ablation:
            component_names = [item.get("component") for item in self.canonical_components]
            planned_names = [step.get("component_or_condition") for step in steps]
            if planned_names != component_names:
                errors.append("ablation steps must match canonical component order exactly")
        if errors:
            raise RuntimeError("Invalid science plan contract:\n- " + "\n- ".join(errors))

    async def _call_worker(self, step: Dict[str, Any], previous_review: Dict[str, Any] | None) -> Dict[str, Any]:
        blueprint_key = "ablation_science_blueprint" if self.ablation else "standard_science_blueprint"
        blueprint_path = self.paths.get(blueprint_key, "")
        blueprint_hint = ""
        if blueprint_path and os.path.exists(blueprint_path):
            blueprint_hint = f"""
Before executing, read the planner's blueprint to understand the full context:
`{blueprint_path}`

The blueprint contains the planner's exploration findings, design rationale, and key decisions.
Use this context to make informed implementation choices rather than blindly following the contract.
"""
        prompt = f"""Execute this science step.
{blueprint_hint}
Step contract:
```json
{json.dumps(step, ensure_ascii=False, indent=2)}
```

Previous validator feedback:
```json
{json.dumps(previous_review or {}, ensure_ascii=False, indent=2)}
```

Return structured worker JSON only. The runtime will write the worker report file.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=self.worker_role,
            system_prompt=ablation_science_worker_prompt() if self.ablation else standard_science_worker_prompt(),
            output_schema=worker_output_schema(),
        )
        return result["output"]

    async def _call_validator(self, step: Dict[str, Any], worker_payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Review this science step.

Step contract:
```json
{json.dumps(step, ensure_ascii=False, indent=2)}
```

Worker report:
```json
{json.dumps(worker_payload, ensure_ascii=False, indent=2)}
```

Return structured review JSON only. The runtime will write the validator report file.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=self.validator_role,
            system_prompt=ablation_science_validator_prompt() if self.ablation else standard_science_validator_prompt(),
            output_schema=validator_output_schema(include_ablation_fields=self.ablation),
        )
        return result["output"]

    def _phase_summary_payload(self, plan_payload: Dict[str, Any], step_result: Dict[str, Any]) -> Dict[str, Any]:
        ok = step_result["status"] == "PASS"
        failed = step_result.get("failed_validator_report") or {}
        payload = with_phase_defaults(
            {
                "status": "PASS" if ok else "FAIL",
                "scope": "ablation_science" if self.ablation else "standard_science",
                "checked_artifacts": [self.plan_path, self.report_path],
                "findings": [] if ok else list(failed.get("findings") or []),
                "required_fixes": [] if ok else list(failed.get("required_fixes") or []),
                "evidence_summary": plan_payload["summary"] if ok else str(failed.get("evidence_summary") or "science phase incomplete"),
                "phase_completion_status": "complete" if ok else "partial",
                "ready_for_next_phase": bool(ok),
                "blocking_issues": [] if ok else list(failed.get("blocking_issues") or [f"Failed step: {step_result.get('failed_step', {}).get('step_id', 'unknown')}"]),
                "required_followup": [] if ok else list(failed.get("required_followup") or []),
                "artifact_role": "phase_result",
                "run_level": "full",
                "self_contained_project": True,
                "self_contained_violations": [],
                "provenance_manifest_present": os.path.exists(self.paths["project_code_provenance"]),
                "provenance_manifest_path": self.paths["project_code_provenance"],
                "terminal_blocker": bool(failed.get("terminal_blocker")),
                "next_worker_input": str(failed.get("next_worker_input") or ""),
            },
            scope="ablation_science" if self.ablation else "standard_science",
        )
        if self.ablation and ok:
            reports = step_result.get("step_reports", [])
            component_map = {
                step["component_or_condition"]: {
                    "result": report["result"],
                    "metric": report["metric"],
                    "value": report["value"],
                    "confidence": report["confidence"],
                    "analysis": report["analysis"],
                    "method_context": report["method_context"],
                    "follow_up_required": report["follow_up_required"],
                }
                for step, report in zip(plan_payload["stages"], reports)
            }
            confidences = [entry["confidence"] for entry in component_map.values()]
            payload["ablation_components"] = component_map
            payload["summary"] = {
                "feasible": True,
                "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "key_findings": [entry["analysis"] for entry in component_map.values()],
            }
        return payload

    async def execute(self) -> Dict[str, Any]:
        plan_payload = await self._plan()
        steps = plan_payload["stages"]
        self._validate_steps(steps)
        # Contract data is already in plan JSON; execute_step_loop passes it
        # in prompts. Step contract files are not needed for execution.
        step_result = await execute_step_loop(
            steps=steps,
            scope="ablation_science" if self.ablation else "standard_science",
            workspace_root=self.workspace_root,
            call_worker=self._call_worker,
            call_validator=self._call_validator,
        )
        phase_report = self._phase_summary_payload(plan_payload, step_result)
        write_json_file(self.validator_report_path, phase_report)
        summary_lines = [
            f"# {'Ablation' if self.ablation else 'Standard'} Science Summary",
            "",
            f"- status: {phase_report['status']}",
            f"- ready_for_next_phase: {phase_report['ready_for_next_phase']}",
            f"- planner_summary: {plan_payload['summary']}",
        ]
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        return {
            "summary": self._read_text_file(self.report_path).strip(),
            "usage": plan_payload["usage_notes"],
            "summary_path": self.report_path,
            "status": "completed" if normalize_phase_report(phase_report).get("status") == "PASS" else "insufficient",
        }


class StandardScienceAgent(_BaseSciencePlanner):
    planner_name = "StandardScience"
    planner_role = EXPERIMENT_STANDARD_SCIENCE_PLANNER
    summary_key = "standard_summary"
    plan_key = "standard_science_plan"
    validator_key = "standard_science_validator"
    worker_role = STANDARD_SCIENCE_WORKER
    validator_role = STANDARD_SCIENCE_VALIDATOR
    step_fields = SCIENCE_STANDARD_STEP_FIELDS
    ablation = False


class AblationScienceAgent(_BaseSciencePlanner):
    planner_name = "AblationScience"
    planner_role = EXPERIMENT_ABLATION_SCIENCE_PLANNER
    summary_key = "ablation_summary"
    plan_key = "ablation_science_plan"
    validator_key = "ablation_science_validator"
    worker_role = ABLATION_SCIENCE_WORKER
    validator_role = ABLATION_SCIENCE_VALIDATOR
    step_fields = SCIENCE_ABLATION_STEP_FIELDS
    ablation = True


async def run_standard_science_agent(
    experiment_id: str,
    idea_path: str,
    project_root: str,
    workspace_root: str,
    plan: str,
    code_summary: str = "",
    code_usage: str = "",
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = StandardScienceAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        project_root=project_root,
        workspace_root=workspace_root,
        plan=plan,
        code_summary=code_summary,
        code_usage=code_usage,
        model=model or get_agent_model("standard_science_agent", "science"),
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()


async def run_ablation_science_agent(
    experiment_id: str,
    idea_path: str,
    project_root: str,
    workspace_root: str,
    plan: str,
    code_summary: str = "",
    code_usage: str = "",
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = AblationScienceAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        project_root=project_root,
        workspace_root=workspace_root,
        plan=plan,
        code_summary=code_summary,
        code_usage=code_usage,
        model=model or get_agent_model("ablation_science_agent", "science"),
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()


async def run_science_agent(
    experiment_id: str,
    idea_path: str,
    project_root: str,
    workspace_root: str,
    plan: str,
    code_summary: str = "",
    code_usage: str = "",
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    return await run_standard_science_agent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        project_root=project_root,
        workspace_root=workspace_root,
        plan=plan,
        code_summary=code_summary,
        code_usage=code_usage,
        model=model,
        verbose=verbose,
        resume=resume,
    )


ScienceAgent = StandardScienceAgent
