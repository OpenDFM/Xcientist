"""OpenHarness-backed code planner for experiment enablement."""

from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Dict, List

from src.agents.experiment_agent.agents.base.agent import BaseAgent
from src.agents.experiment_agent.agents.code.reviewer import (
    CODE_REVIEWER,
    CODE_REVIEWER_IDS,
    code_reviewer_prompt,
)
from src.agents.experiment_agent.agents.code.worker import CODE_WORKER, code_worker_prompt
from src.agents.experiment_agent.config import get_code_agent_model
from src.agents.experiment_agent.runtime.contracts import (
    CODE_STEP_CONTRACT_FIELDS,
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
    format_named_paths,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    write_json_file,
    workspace_paths,
)
from src.agents.experiment_agent.runtime.artifacts import (
    ArtifactRegistry,
    ArtifactSpec,
    artifact_ledger_path,
    artifact_prompt_context,
    ensure_runtime_report_paths,
    write_artifact_registry_snapshot,
)
from src.agents.experiment_agent.runtime.phase_runner import (
    execute_step_with_prefinish_review,
    materialize_executable_plan,
    phase_checked_artifacts,
    planner_artifact_prefinish_gate,
    phase_step_ids,
    planner_output_schema,
    review_output_schema,
    worker_output_schema,
    with_phase_defaults,
)
from src.agents.experiment_agent.runtime.report_layout import phase_rel, planner_rel
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.self_contained import scan_project_self_contained


EXPERIMENT_CODE_PLANNER = "experiment_code_planner"


def _code_step_schema() -> Dict[str, Any]:
    properties = {
        "step_id": {"type": "string"},
        "goal": {"type": "string"},
        "component_scope": {"type": "array", "items": {"type": "string"}},
        "code_artifacts": {"type": "array", "items": {"type": "object"}},
        "interface_contract": {"type": "object"},
        "implementation_requirements": {"type": "object"},
        "component_disable_hooks": {"type": "array", "items": {"type": "object"}},
        "experiment_bindings": {"type": "object"},
        "repo_source_paths": {"type": "array", "items": {"type": "string"}},
        "repo_copy_intent": {"type": "string"},
        "project_target_paths": {"type": "array", "items": {"type": "string"}},
        "input_paths": {"type": "object"},
        "repos_policy": {"type": "string"},
        "project_must_be_self_contained": {"type": "boolean"},
        "write_scope": {"type": "string"},
        "verify_command": {"type": "string"},
        "done_condition": {"type": "string"},
        "artifact_ids": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }

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
        self.workspace_paths = workspace_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.summary_path = self.paths["code_summary"]
        self.usage_path = self.paths["code_usage"]
        self.plan_path = self.paths["code_plan"]
        self.review_report_path = self.paths["code_reviewer"]

    def _build_planner_artifact_registry(self) -> ArtifactRegistry:
        registry = ArtifactRegistry(workspace_root=self.workspace_root)
        registry.add(
            ArtifactSpec(
                artifact_id="code.plan",
                stage="code_planner",
                path=planner_rel("code", "latest.json"),
                kind="json",
                schema_name="code_plan",
                description="Validated code phase plan with stages that satisfy the code step contract.",
            )
        )
        registry.add(
            ArtifactSpec(
                artifact_id="code.blueprint",
                stage="code_planner",
                path=planner_rel("code", "blueprint.md"),
                kind="file",
                required=False,
                description="Planner exploration notes and implementation rationale for code workers.",
            )
        )
        registry.add(
            ArtifactSpec(
                artifact_id="runtime.code.planner_report",
                stage="code_planner",
                path=planner_rel("code", "planner_report.json"),
                kind="json",
                required=False,
                writer="runtime",
                description="Runtime-owned code planner report.",
            )
        )
        return registry

    def _get_relevant_env_var_names(self) -> List[str]:
        keywords = ("KEY", "TOKEN", "SECRET", "API", "BASE_URL", "ENDPOINT")
        names = [name for name in os.environ if any(k in name.upper() for k in keywords)]
        return sorted(names)[:50]

    def _build_user_prompt(self) -> str:
        input_paths = format_named_paths(
            {
                "idea_path": self.idea_path,
                "prepare_plan_path": self.paths["prepare_plan"],
                "prepare_phase_review_report_path": self.paths["prepare_reviewer"],
            }
        )
        workspace_path_text = format_named_paths(
            {
                "workspace_dir": self.workspace_paths["workspace_dir"],
                "project_dir": self.workspace_paths["project_dir"],
                "dataset_dir": self.workspace_paths["dataset_dir"],
                "model_dir": self.workspace_paths["model_dir"],
                "results_dir": self.workspace_paths["results_dir"],
                "agent_reports_dir": self.workspace_paths["agent_reports_dir"],
            }
        )
        step_fields = format_field_bullets(CODE_STEP_CONTRACT_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        planner_registry = self._build_planner_artifact_registry()
        artifact_text = artifact_prompt_context(planner_registry)
        return f"""## Task: Enable Experiment Code Paths

### Goal
Implement the full idea in `project/` to support unified science conditions.
The all-components condition is the reference case where every idea component is
enabled. Component-disabled conditions use the same runner interface with one
canonical idea component disabled per condition.

### Master Plan
{self.plan}

### Input Paths
{input_paths}

### Workspace Paths
{workspace_path_text}

### Code Project Integrity Gate
The final code step is a hard project-cleanliness gate. Before the non-formal
reviewer is called, the prefinish hook will run deterministic checks over
`project/`. If any check fails, the hook returns the issues directly to the
worker and the worker must fix them in the same session.

Plan for this gate explicitly:
- Use the workspace root as the canonical cwd for every code/science command.
  Write commands as `project/.venv/bin/python project/<entrypoint>.py ...` or
  equivalent workspace-root paths. Do not use `cd project && ...`.
- Runner defaults and verify commands must resolve from the workspace root.
- The final `project/` tree must contain one clean canonical implementation,
  not scratch files, backups, patched/repaired/vectorized variants, or old
  alternate train/model scripts.
- Required runtime resources such as datasets, adjacency files, top-k indices,
  checkpoints, and configs must fail fast when missing. Silent fallbacks to
  placeholders or degraded behavior are forbidden.
- `final_integration_smoke` must leave `project/` clean and runnable, and its
  evidence must prove all-components and component-disabled science paths with bounded real data.

### Required Planner Output
Return a JSON object with `stages`, `summary`, and `usage_notes`.
You must write the managed artifact `code.plan` with Xcientist artifact tools until it is accepted by the hook.
The plan path is fixed by the Artifact Registry: `{self.plan_path}`.
The managed `code.plan` artifact must be a JSON object with a top-level `stages` list; the hook validates that list as the executable code contract.
Do not write `code_plan.json` with generic write_file/edit_file/bash; use `write_artifact`.
Do not write `code_planner_report.json` or other planner/runtime reports with tools.
After `code.plan` and optional `code.blueprint` are accepted, stop using tools and return the required final JSON object directly.

{artifact_text}

Every step in `stages` must include:
{step_fields}

Rules:
- The plan must end with a mandatory step whose `step_id` is exactly `final_integration_smoke`.
- Do not use placeholder values such as `test`, `test_step`, `test_component`, `test_artifact`, `demo`, `example`, `mock`, or `todo` anywhere in the contract.
- `component_scope` entries must be exact canonical component names from `idea.json.components` and prepared targets.
- `repo_source_paths` must reference existing files under `repos/` unless `repo_copy_intent` is `none`.
- `project_target_paths` and every `code_artifacts[].path` must stay under `project/`.
- Each step's `artifact_ids` must include its managed handoff artifact id exactly as `code.<step_id>.handoff`.
- `final_integration_smoke.artifact_ids` must also include `code.final_integration_smoke.evidence`.
- `verify_command` must run real verification such as a bounded Python script or pytest command; `echo`, `printf`, `true`, imports-only checks, mocks, dry-run-only checks, and synthetic/random data are invalid.
- Every `code_artifacts` item must include at least:
  `path`, `artifact_type`, `symbols`, `responsibility`, `dependencies`, `config_keys`, and `entrypoint_role`.
- Every `component_disable_hooks` item must name `component` from the step's `component_scope` and a concrete toggle such as `flag`, `config_key`, `config_override`, `condition`, `command_arg`, `env_var`, or `mode`.
- Every step's `done_condition` must explicitly forbid `sys.path` injection, editable installs of `repos/`, and imports reaching outside `project/`.
- The `final_integration_smoke` step is a bounded integration smoke, not a full science run. Its contract must require:
  - Real files from `dataset_candidate/`, never synthetic or random data.
  - The actual integrated train/evaluate/component-disabled code path in `project/`.
  - A small bounded workload such as explicit data slicing, max batches, max masks, max epochs, or equivalent runtime guard.
  - Real output evidence: at least one runnable command, raw log path, checkpoint or model output path when training is exercised, and an evaluation/metrics JSON or equivalent result file.
  - A managed `code.final_integration_smoke.evidence` artifact under `{phase_rel("code", "artifacts", "final_integration_smoke.json")}`.
  - Explicit prohibition on treating timeout, imports-only, dry-run, mocks, or synthetic data as completion evidence.
  - Final cleanup of `project/` so only canonical runnable code and declared
    resources remain.
  - Validation that every runner default path and every science command path
    resolves from the workspace root.
  - Validation that required resources fail fast instead of silently falling
    back to placeholders.
- Every step must tie to real prepared targets discovered from prepare artifacts.
- Do not invent report filenames. Runtime reports are written under structured `agent_reports/code/...` directories by hooks.
- `repos_policy` must be `reference_or_copy`.
- `project_must_be_self_contained` must be true.
- Do not ask the runtime to keep a runtime dependency on `repos/`.
- Do not materialize science-owned artifacts like `ablation_results.json`.
- If any step copies code from `repos/` into `project/` (via `repo_copy_intent=copy_and_modify`), the step's `done_condition` MUST explicitly require recording copied sources with `record_sources`. Source provenance lives in the artifact ledger.
- Every step's `done_condition` MUST explicitly forbid `sys.path` injection, editable installs of `repos/`, and local-path imports reaching outside `project/`. These are reviewer FAIL conditions.
- After exploring the workspace, write the managed artifact `code.blueprint` to `{self.paths.get("code_blueprint", "agent_reports/code/plan/blueprint.md")}` capturing:
  - What you discovered about the prepare artifacts (data schema, model interfaces, etc.)
  - Why you chose each step's approach
  - Key implementation decisions and trade-offs
  - What the worker should pay attention to in each step

Minimal shape for a real step item:
```json
{{
  "step_id": "implement_diffusion_coverage_model",
  "goal": "Implement Graph WaveNet coverage residual modules inside project code.",
  "component_scope": ["diffusion_referenced_innovation_coverage_cell", "sparse_edge_attention_topk"],
  "code_artifacts": [
    {{
      "path": "project/model.py",
      "artifact_type": "python_module",
      "symbols": ["DiffusionReferencedInnovationCoverageCell", "SparseEdgeAttentionTopK"],
      "responsibility": "Model modules and forward-path integration for the coverage residual branch.",
      "dependencies": ["torch", "numpy"],
      "config_keys": ["ridge_lambda", "topk_k", "bounded_residual_scale"],
      "entrypoint_role": "model_component"
    }}
  ],
  "interface_contract": {{"entrypoint": "python project/smoke_train.py --max-epochs 1 --max-batches 2"}},
  "implementation_requirements": {{"dataset": "dataset_candidate/pems_bay/graph_wavenet", "no_synthetic_data": true}},
  "component_disable_hooks": [
    {{"component": "diffusion_referenced_innovation_coverage_cell", "flag": "--disable-coverage-cell"}}
  ],
  "experiment_bindings": {{"metrics_json": "results/smoke_metrics.json", "dataset_dir": "dataset_candidate/pems_bay/graph_wavenet"}},
  "repo_source_paths": ["repos/Graph-WaveNet/model.py", "repos/Graph-WaveNet/train.py"],
  "repo_copy_intent": "copy_and_modify",
  "project_target_paths": ["project/model.py", "project/smoke_train.py"],
  "input_paths": {{"idea": "agent_reports/prepare/artifacts/idea.md", "targets": "agent_reports/prepare/artifacts/target_inventory.json"}},
  "repos_policy": "reference_or_copy",
  "project_must_be_self_contained": true,
  "write_scope": "project/",
  "verify_command": "python project/smoke_train.py --data dataset_candidate/pems_bay/graph_wavenet --max-epochs 1 --max-batches 2",
  "done_condition": "Verification passes using dataset_candidate real files; sys.path injection, editable installs of repos/, and imports reaching outside project/ are forbidden.",
  "artifact_ids": ["code.implement_diffusion_coverage_model.handoff"]
}}
```

The runtime will execute each step through `{CODE_WORKER}` and review it through `{CODE_REVIEWER}`.
The final phase-level reviewer report must include:
{verdict_fields}

Candidate env vars: {", ".join(self._get_relevant_env_var_names()) if self._get_relevant_env_var_names() else "(none detected)"}.
"""

    async def _plan(self) -> Dict[str, Any]:
        registry = self._build_planner_artifact_registry()
        write_artifact_registry_snapshot(registry)
        self.set_artifact_context(
            registry.to_context(stage="code_planner", step_id="plan", attempt=1)
        )
        output_schema = planner_output_schema(step_schema=_code_step_schema())
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            agent_name=EXPERIMENT_CODE_PLANNER,
            output_schema=output_schema,
            extra_tool_metadata={
                "xcientist_prefinish_gate": planner_artifact_prefinish_gate(
                    output_schema=output_schema,
                    registry=registry,
                    plan_artifact_id="code.plan",
                )
            },
        )
        _ = result
        payload = self._load_required_plan_payload_from_artifact()
        self._write_plan_outputs(payload)
        return payload

    def _load_required_plan_payload_from_artifact(self) -> Dict[str, Any]:
        if not os.path.exists(self.plan_path):
            raise RuntimeError(
                f"Missing managed planner artifact `code.plan` at `{self.plan_path}`. "
                "The code planner must write it with `write_artifact` before finishing."
            )
        try:
            with open(self.plan_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Could not read managed planner artifact `code.plan`: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Managed planner artifact `code.plan` must be a JSON object.")
        if not isinstance(payload.get("stages"), list):
            raise RuntimeError("Managed planner artifact `code.plan.stages` must be a list.")
        for field in ("summary", "usage_notes"):
            if not isinstance(payload.get(field), str):
                raise RuntimeError(f"Managed planner artifact `code.plan.{field}` must be a string.")
        return payload

    def _write_plan_outputs(
        self,
        payload: Dict[str, Any],
    ) -> None:
        blueprint_path = self.paths.get(
            "code_blueprint",
            self.paths["code_blueprint"],
        )
        report_payload = {
            "scope": "code",
            "summary": payload.get("summary", ""),
            "usage_notes": payload.get("usage_notes", ""),
            "plan_path": self.plan_path,
            "blueprint_path": blueprint_path,
        }
        materialize_executable_plan(
            workspace_root=self.workspace_root,
            scope="code",
            plan_payload=payload,
            planner_report=report_payload,
        )
        with open(self.usage_path, "w", encoding="utf-8") as f:
            f.write(str(payload.get("usage_notes") or "").strip() + "\n")

    async def _call_worker(
        self,
        step: Dict[str, Any],
        previous_review: Dict[str, Any] | None,
        artifact_context: Dict[str, Any],
        prefinish_gate=None,
    ) -> Dict[str, Any]:
        self.set_artifact_context(artifact_context)
        retry_brief = ""
        if previous_review:
            retry_brief = str(previous_review.get("next_worker_input") or "").strip()
        artifact_text = artifact_prompt_context(ArtifactRegistry.from_context(artifact_context))
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

{artifact_text}

Previous reviewer feedback:
```json
{json.dumps(previous_review or {}, ensure_ascii=False, indent=2)}
```

Return only structured JSON describing what you changed and what evidence now exists. Do not write report files; the STOP/prefinish hook writes structured worker attempts and latest files under `agent_reports/code/worker/<step>/`.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=CODE_WORKER,
            system_prompt=code_worker_prompt(),
            output_schema=worker_output_schema(),
            extra_tool_metadata={"xcientist_prefinish_gate": prefinish_gate} if prefinish_gate else None,
        )
        payload = dict(result["output"])
        if retry_brief and retry_brief not in payload["summary"]:
            payload["summary"] = f"{payload['summary']}\nRetry brief addressed: {retry_brief}".strip()
        return payload

    async def _call_reviewer(self, step: Dict[str, Any], worker_payload: Dict[str, Any], artifact_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.set_artifact_context(artifact_context)
        artifact_text = artifact_prompt_context(ArtifactRegistry.from_context(artifact_context))
        selected_reviewer_ids = [
            str(item)
            for item in artifact_context.get("selected_code_reviewer_ids") or []
            if str(item).strip() in CODE_REVIEWER_IDS
        ] or list(CODE_REVIEWER_IDS)
        review_context = artifact_context.get("code_review_context") or {}
        review_context_path = str(artifact_context.get("code_review_context_path") or "")
        async def _run_one(reviewer_id: str) -> Dict[str, Any]:
            prompt = f"""Review this completed code step.

Idea file:
`{os.path.realpath(self.paths.get("idea") or self.idea_path)}`

Shared code review context:
`{review_context_path or "(not persisted)"}`
```json
{json.dumps(review_context, ensure_ascii=False, indent=2)}
```

Step contract:
```json
{json.dumps(step, ensure_ascii=False, indent=2)}
```

{artifact_text}

Worker report:
```json
{json.dumps(worker_payload, ensure_ascii=False, indent=2)}
```

Reviewer selection:
- Runtime runs the full parallel code reviewer matrix for every code step:
  `{", ".join(selected_reviewer_ids)}`.
- You are `{reviewer_id}`. Stay inside your assigned focus.

Return structured review JSON only. Do not write report files; the STOP/prefinish hook writes structured review attempts and latest files under `agent_reports/code/review/<step>/<reviewer_id>/`.
"""
            result = await self.run(
                user_prompt=prompt,
                agent_name=reviewer_id,
                system_prompt=code_reviewer_prompt(reviewer_id),
                output_schema=review_output_schema(),
                purpose="prefinish_review",
                extra_tool_metadata={"xcientist_expected_reviewer_id": reviewer_id},
            )
            payload = result["output"]
            payload.setdefault("reviewer_id", reviewer_id)
            return payload

        return await asyncio.gather(*[_run_one(reviewer_id) for reviewer_id in selected_reviewer_ids])

    def _write_phase_reports(
        self,
        *,
        plan_payload: Dict[str, Any],
        step_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        self_contained = scan_project_self_contained(self.project_root, self.workspace_root)
        write_json_file(self.paths["self_contained_report"], self_contained)
        ok = step_result["status"] == "PASS" and bool(self_contained.get("self_contained_project"))
        failure = step_result.get("failed_review_report") or {}
        review_payload = with_phase_defaults(
            {
                "status": "PASS" if ok else "FAIL",
                "scope": "code",
                "checked_artifacts": phase_checked_artifacts(
                    step_result,
                    self.plan_path,
                    self.paths["code_executable_plan"],
                    self.paths["code_planner_report"],
                    self.paths["code_integration_readiness"],
                    self.paths["self_contained_report"],
                    artifact_ledger_path(self.workspace_root),
                ),
                "findings": [] if ok else list(failure.get("findings") or []),
                "evidence_summary": plan_payload["summary"] if ok else str(failure.get("evidence_summary") or "code phase incomplete"),
                "phase_completion_status": "complete" if ok else "partial",
                "ready_for_next_phase": bool(ok),
                "blocking_issues": [] if ok else list(failure.get("blocking_issues") or [f"Failed step: {step_result.get('failed_step', {}).get('step_id', 'unknown')}"]),
                "required_followup": [] if ok else list(failure.get("required_followup") or []),
                "artifact_role": "phase_result",
                "run_level": "mixed",
                "self_contained_project": bool(self_contained.get("self_contained_project")),
                "self_contained_violations": list(self_contained.get("self_contained_violations") or []),
                "artifact_ledger_present": os.path.exists(artifact_ledger_path(self.workspace_root)),
                "artifact_ledger_path": artifact_ledger_path(self.workspace_root),
                "terminal_blocker": bool(failure.get("terminal_blocker")),
                "next_worker_input": str(failure.get("next_worker_input") or ""),
            },
            scope="code",
        )
        write_json_file(self.paths["code_reviewer"], review_payload)
        write_json_file(
            self.paths["code_integration_readiness"],
            {
                "status": review_payload["status"],
                "self_contained_project": review_payload["self_contained_project"],
                "self_contained_report_path": self.paths["self_contained_report"],
                "review_report_path": self.paths["code_reviewer"],
            },
        )
        worker_phase_payload = {
            "scope": "code",
            "status": review_payload["status"],
            "summary": plan_payload["summary"],
            "executed_steps": phase_step_ids(step_result),
        }
        write_json_file(self.paths["code_worker"], worker_phase_payload)
        summary_lines = [
            "# Code Summary",
            "",
            f"- status: {review_payload['status']}",
            f"- ready_for_next_phase: {review_payload['ready_for_next_phase']}",
            f"- planner_summary: {plan_payload['summary']}",
        ]
        for issue in review_payload["blocking_issues"]:
            summary_lines.append(f"- blocker: {issue}")
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        return review_payload

    async def execute(self) -> Dict[str, Any]:
        plan_payload = await self._plan()
        steps = plan_payload["stages"]
        for step in steps:
            ensure_runtime_report_paths("code", step, self.workspace_root)
        # Contract data is already in plan JSON; execute_step_with_prefinish_review passes it
        # in prompts. Step contract files are not needed for execution.
        step_result = await execute_step_with_prefinish_review(
            steps=steps,
            scope="code",
            workspace_root=self.workspace_root,
            call_worker=self._call_worker,
            call_reviewer=self._call_reviewer,
        )
        review_payload = self._write_phase_reports(plan_payload=plan_payload, step_result=step_result)
        status = "completed" if normalize_phase_report(review_payload).get("status") == "PASS" else "insufficient"
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
