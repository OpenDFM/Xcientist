"""
Claude Code-backed code planner for experiment enablement.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.agents.experiment_agent.agents.base.agent import BaseAgent
from src.agents.experiment_agent.agents.code.validator import CODE_VALIDATOR, code_validator_prompt
from src.agents.experiment_agent.agents.code.worker import CODE_WORKER, code_worker_prompt
from src.agents.experiment_agent.config import get_code_agent_model
from src.agents.experiment_agent.runtime.contracts import (
    CODE_STEP_CONTRACT_FIELDS,
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
    format_named_paths,
    validate_repo_contract_fields,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    coerce_plan_payload,
    extract_plan_steps,
    write_json_file,
    workspace_contract_paths,
)
from src.agents.experiment_agent.runtime.phase_runner import (
    execute_step_loop,
    planner_output_schema,
    validator_output_schema,
    worker_output_schema,
    with_phase_defaults,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.self_contained import scan_project_self_contained


EXPERIMENT_CODE_PLANNER = "experiment_code_planner"


def _code_step_schema() -> Dict[str, Any]:
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
        "write_scope": {"type": "string"},
        "verify_command": {"type": "string"},
        "max_repair_rounds": {"type": "integer"},
        "done_condition": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }


def create_experiment_code_planner_agent(llm):
    _ = llm
    return {"role": EXPERIMENT_CODE_PLANNER}


def register_experiment_code_planner() -> None:
    return None


class CodeAgent(BaseAgent):
    def __init__(
        self,
        experiment_id: str,
        idea_path: str,
        project_root: str,
        workspace_root: str,
        plan: str,
        model: str | None = None,
        verbose: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            agent_type="Code",
            model=model or get_code_agent_model(),
            verbose=verbose,
            workspace_root=workspace_root,
            resume=resume,
        )
        self.experiment_id = experiment_id
        self.idea_path = idea_path
        self.project_root = project_root
        self.workspace_root = workspace_root
        self.plan = plan
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.summary_path = self.paths["code_summary"]
        self.usage_path = self.paths["code_usage"]
        self.plan_path = self.paths["code_plan"]
        self.validator_report_path = self.paths["code_validator"]

    def _get_relevant_env_var_names(self) -> List[str]:
        keywords = ("KEY", "TOKEN", "SECRET", "API", "BASE_URL", "ENDPOINT")
        names = [name for name in os.environ if any(k in name.upper() for k in keywords)]
        return sorted(names)[:50]

    def _build_user_prompt(self) -> str:
        input_paths = format_named_paths(
            {
                "idea_path": self.idea_path,
                "prepare_plan_path": self.paths["prepare_plan"],
                "prepare_phase_validator_report_path": self.paths["prepare_validator"],
            }
        )
        path_contract = format_named_paths(
            {
                "workspace_dir": self.contract["workspace_dir"],
                "project_dir": self.contract["project_dir"],
                "dataset_dir": self.contract["dataset_dir"],
                "model_dir": self.contract["model_dir"],
                "results_dir": self.contract["results_dir"],
                "agent_reports_dir": self.contract["agent_reports_dir"],
            }
        )
        step_fields = format_field_bullets(CODE_STEP_CONTRACT_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        return f"""## Task: Enable Experiment Code Paths

### Goal
Implement the full idea in `project/` to support standard science and ablation science experiments.

### Master Plan
{self.plan}

### Input Paths
{input_paths}

### Path Contract
{path_contract}

### Required Planner Output
Write a JSON object with `stages`, `summary`, and `usage_notes`.
Every step in `stages` must include:
{step_fields}

Rules:
- The plan must end with a mandatory step whose `step_id` is exactly `final_integration_smoke`.
- Every step must tie to real prepared targets discovered from prepare artifacts.
- Use unique flat report filenames under `agent_reports_dir`.
- `repos_policy` must be `reference_or_copy`.
- `project_must_be_self_contained` must be true.
- Do not ask the runtime to keep a runtime dependency on `repos/`.
- Do not materialize science-owned artifacts like `ablation_results.json`.
- If any step copies code from `repos/` into `project/` (via `repo_copy_intent=copy_and_modify`), the step's `done_condition` MUST explicitly require creating the provenance manifest at `provenance_manifest_path`. The manifest is a JSON mapping from each copied repo source file to its destination under `project/`, with a brief note on what was copied and why. The validator will FAIL the step without it.
- Every step's `done_condition` MUST explicitly forbid `sys.path` injection, editable installs of `repos/`, and local-path imports reaching outside `project/`. These are validator FAIL conditions.
- After exploring the workspace, write a blueprint to `{self.paths.get("code_blueprint", "agent_reports/code_blueprint.md")}` capturing:
  - What you discovered about the prepare artifacts (data schema, model interfaces, etc.)
  - Why you chose each step's approach
  - Key implementation decisions and trade-offs
  - What the worker should pay attention to in each step

The runtime will execute each step through `{CODE_WORKER}` and review it through `{CODE_VALIDATOR}`.
The final phase-level validator report must include:
{verdict_fields}

Candidate env vars: {", ".join(self._get_relevant_env_var_names()) if self._get_relevant_env_var_names() else "(none detected)"}.
"""

    async def _plan(self) -> Dict[str, Any]:
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            agent_name=EXPERIMENT_CODE_PLANNER,
            output_schema=planner_output_schema(step_schema=_code_step_schema()),
        )
        payload = coerce_plan_payload(result["output"], self.plan_path, scope="code")
        write_json_file(self.plan_path, {"stages": payload["stages"]})
        # Blueprint: rich markdown capturing exploration findings, context, and decision rationale.
        # Workers read this to understand intent, not just the contract JSON.
        blueprint_path = self.paths.get("code_blueprint", os.path.join(self.contract["agent_reports_dir"], "code_blueprint.md"))
        write_json_file(
            self.paths["code_planner_report"],
            {
                "scope": "code",
                "summary": payload["summary"],
                "usage_notes": payload["usage_notes"],
                "plan_path": self.plan_path,
                "blueprint_path": blueprint_path,
            },
        )
        with open(self.usage_path, "w", encoding="utf-8") as f:
            f.write(payload["usage_notes"].strip() + "\n")
        # Write a default blueprint if the planner didn't produce one via its own exploration.
        # The blueprint should be produced by the planner's exploration; if missing, synthesize from plan + summary.
        if not os.path.exists(blueprint_path):
            self._synthesize_blueprint(blueprint_path, payload)
        return payload

    def _synthesize_blueprint(self, blueprint_path: str, payload: Dict[str, Any]) -> None:
        """Generate a blueprint.md from the plan payload when the planner didn't produce one directly."""
        lines = [
            "# Code Phase Blueprint",
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
            if step.get('done_condition'):
                lines.append(f"**Done when:** {step['done_condition']}")
                lines.append("")
            if step.get('repo_source_paths'):
                lines.append(f"**Source repos:** {', '.join(step['repo_source_paths'])}")
                lines.append("")
            if step.get('project_target_paths'):
                lines.append(f"**Target paths:** {', '.join(step['project_target_paths'])}")
                lines.append("")
        lines.append(f"## Usage Notes\n\n{payload.get('usage_notes', '')}\n")
        os.makedirs(os.path.dirname(blueprint_path), exist_ok=True)
        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _validate_plan_artifact(self, steps: List[Dict[str, Any]]) -> None:
        if not steps:
            raise RuntimeError("Code plan must contain at least one step.")
        errors: List[str] = []
        for index, step in enumerate(steps, start=1):
            errors.extend(
                f"step {index}: {message}"
                for message in validate_repo_contract_fields(step, project_dir=self.project_root)
            )
        if steps[-1].get("step_id") != "final_integration_smoke":
            errors.append("final step_id must be `final_integration_smoke`")
        if errors:
            raise RuntimeError("Invalid code plan contract:\n- " + "\n- ".join(errors))

    async def _call_worker(self, step: Dict[str, Any], previous_review: Dict[str, Any] | None) -> Dict[str, Any]:
        retry_brief = ""
        if previous_review:
            retry_brief = str(previous_review.get("next_worker_input") or "").strip()
        blueprint_path = self.paths.get("code_blueprint", "")
        blueprint_hint = ""
        if blueprint_path and os.path.exists(blueprint_path):
            blueprint_hint = f"""
Before executing, read the planner's blueprint to understand the full context:
`{blueprint_path}`

The blueprint contains the planner's exploration findings, design rationale, and key decisions.
Use this context to make informed implementation choices rather than blindly following the contract.
"""
        prompt = f"""Implement this code step.
{blueprint_hint}
Step contract:
```json
{json.dumps(step, ensure_ascii=False, indent=2)}
```

Previous validator feedback:
```json
{json.dumps(previous_review or {}, ensure_ascii=False, indent=2)}
```

Return only structured JSON describing what you changed and what evidence now exists. The runtime will write the worker report file.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=CODE_WORKER,
            system_prompt=code_worker_prompt(),
            output_schema=worker_output_schema(),
        )
        payload = dict(result["output"])
        if retry_brief and retry_brief not in payload["summary"]:
            payload["summary"] = f"{payload['summary']}\nRetry brief addressed: {retry_brief}".strip()
        return payload

    async def _call_validator(self, step: Dict[str, Any], worker_payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Review this completed code step.

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
            agent_name=CODE_VALIDATOR,
            system_prompt=code_validator_prompt(),
            output_schema=validator_output_schema(),
        )
        return result["output"]

    def _write_phase_reports(
        self,
        *,
        plan_payload: Dict[str, Any],
        step_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        self_contained = scan_project_self_contained(self.project_root, self.workspace_root)
        write_json_file(self.paths["self_contained_report"], self_contained)
        ok = step_result["status"] == "PASS" and bool(self_contained.get("self_contained_project"))
        failure = step_result.get("failed_validator_report") or {}
        validator_payload = with_phase_defaults(
            {
                "status": "PASS" if ok else "FAIL",
                "scope": "code",
                "checked_artifacts": [self.plan_path, self.summary_path],
                "findings": [] if ok else list(failure.get("findings") or []),
                "required_fixes": [] if ok else list(failure.get("required_fixes") or []),
                "evidence_summary": plan_payload["summary"] if ok else str(failure.get("evidence_summary") or "code phase incomplete"),
                "phase_completion_status": "complete" if ok else "partial",
                "ready_for_next_phase": bool(ok),
                "blocking_issues": [] if ok else list(failure.get("blocking_issues") or [f"Failed step: {step_result.get('failed_step', {}).get('step_id', 'unknown')}"]),
                "required_followup": [] if ok else list(failure.get("required_followup") or []),
                "artifact_role": "phase_result",
                "run_level": "mixed",
                "self_contained_project": bool(self_contained.get("self_contained_project")),
                "self_contained_violations": list(self_contained.get("self_contained_violations") or []),
                "provenance_manifest_present": os.path.exists(self.paths["project_code_provenance"]),
                "provenance_manifest_path": self.paths["project_code_provenance"],
                "terminal_blocker": bool(failure.get("terminal_blocker")),
                "next_worker_input": str(failure.get("next_worker_input") or ""),
            },
            scope="code",
        )
        write_json_file(self.paths["code_validator"], validator_payload)
        write_json_file(
            self.paths["code_integration_readiness"],
            {
                "status": validator_payload["status"],
                "self_contained_project": validator_payload["self_contained_project"],
                "self_contained_report_path": self.paths["self_contained_report"],
                "validator_report_path": self.paths["code_validator"],
            },
        )
        worker_phase_payload = {
            "scope": "code",
            "status": validator_payload["status"],
            "summary": plan_payload["summary"],
            "executed_steps": [report.get("scope", "code") for report in step_result.get("step_reports", [])],
        }
        write_json_file(self.paths["code_worker"], worker_phase_payload)
        summary_lines = [
            "# Code Summary",
            "",
            f"- status: {validator_payload['status']}",
            f"- ready_for_next_phase: {validator_payload['ready_for_next_phase']}",
            f"- planner_summary: {plan_payload['summary']}",
        ]
        for issue in validator_payload["blocking_issues"]:
            summary_lines.append(f"- blocker: {issue}")
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        return validator_payload

    async def execute(self) -> Dict[str, Any]:
        plan_payload = await self._plan()
        steps = plan_payload["stages"]
        self._validate_plan_artifact(steps)
        # Contract data is already in plan JSON; execute_step_loop passes it
        # in prompts. Step contract files are not needed for execution.
        step_result = await execute_step_loop(
            steps=steps,
            scope="code",
            workspace_root=self.workspace_root,
            call_worker=self._call_worker,
            call_validator=self._call_validator,
        )
        validator_payload = self._write_phase_reports(plan_payload=plan_payload, step_result=step_result)
        status = "completed" if normalize_phase_report(validator_payload).get("status") == "PASS" else "insufficient"
        return {
            "summary": self._read_text_file(self.summary_path).strip(),
            "usage": self._read_text_file(self.usage_path).strip(),
            "summary_path": self.summary_path,
            "usage_path": self.usage_path,
            "status": status,
        }


async def run_code_agent(
    experiment_id: str,
    idea_path: str,
    project_root: str,
    workspace_root: str,
    plan: str,
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = CodeAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        project_root=project_root,
        workspace_root=workspace_root,
        plan=plan,
        model=model or get_code_agent_model(),
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
