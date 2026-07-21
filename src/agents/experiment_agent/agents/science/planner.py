"""OpenHarness-backed unified science condition planner."""

from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Dict

from src.agents.experiment_agent.agents.base.agent import BaseAgent
from src.agents.experiment_agent.agents.science.reviewer import (
    SCIENCE_REVIEWER,
    SCIENCE_REVIEWER_IDS,
    science_reviewer_prompt,
)
from src.agents.experiment_agent.agents.science.worker import (
    SCIENCE_WORKER,
    science_worker_prompt,
)
from src.agents.experiment_agent.config import get_agent_model
from src.agents.experiment_agent.runtime.contracts import (
    PHASE_VERDICT_FIELDS,
    SCIENCE_CONDITION_REVIEW_FIELDS,
    SCIENCE_CONDITION_STEP_FIELDS,
    format_field_bullets,
    format_named_paths,
)
from src.agents.experiment_agent.runtime.idea_components import (
    format_canonical_components_markdown,
    load_canonical_components,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, write_json_file, workspace_paths
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
    planner_output_schema,
    review_output_schema,
    worker_output_schema,
    with_phase_defaults,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.report_layout import artifact_rel, planner_rel
from src.agents.experiment_agent.telemetry import Colors, print_status


EXPERIMENT_SCIENCE_PLANNER = "experiment_science_planner"


def _science_step_schema() -> Dict[str, Any]:
    properties = {
        "condition_id": {"type": "string"},
        "goal": {"type": "string"},
        "enabled_components": {"type": "array", "items": {"type": "string"}},
        "disabled_components": {"type": "array", "items": {"type": "string"}},
        "reference_condition_id": {"type": ["string", "null"]},
        "train_dataset_binding": {"type": "object"},
        "evaluation_dataset_bindings": {"type": "array", "items": {"type": "object"}},
        "metric_bindings": {"type": "array", "items": {"type": "object"}},
        "component_set_description": {"type": "string"},
        "result_interpretation_rule": {"type": "string"},
        "run_level": {"type": "string", "enum": ["full"]},
        "setup_rationale": {"type": "string"},
        "source_basis": {"type": "array", "items": {"type": "object"}},
        "runtime_probe_summary": {"type": "string"},
        "training_protocol": {"type": "object"},
        "evaluation_protocol": {"type": "object"},
        "repo_source_paths": {"type": "array", "items": {"type": "string"}},
        "repo_copy_intent": {"type": "string"},
        "project_target_paths": {"type": "array", "items": {"type": "string"}},
        "input_paths": {"type": "object"},
        "repos_policy": {"type": "string"},
        "project_must_be_self_contained": {"type": "boolean"},
        "command": {"type": "string"},
        "output_dir": {"type": "string"},
        "raw_evidence": {"type": "array", "items": {"type": "string"}},
        "pass_condition": {"type": "string"},
        "artifact_ids": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }

class ScienceAgent(BaseAgent):
    planner_name = "Science"
    planner_role = EXPERIMENT_SCIENCE_PLANNER
    summary_key = "science_summary"
    plan_key = "science_plan"
    reviewer_key = "science_reviewer"
    worker_role = SCIENCE_WORKER
    reviewer_role = SCIENCE_REVIEWER
    scope = "science"

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
            model=model or get_agent_model("science_agent", "science"),
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
        self.workspace_paths = workspace_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.report_path = self.paths[self.summary_key]
        self.plan_path = self.paths[self.plan_key]
        self.review_report_path = self.paths[self.reviewer_key]
        self.canonical_components = load_canonical_components(workspace_root)

    def _input_paths(self) -> Dict[str, str]:
        return {
            "idea_path": self.idea_path,
            "idea_json_path": self.paths["idea_json"],
            "prepare_reviewer_path": self.paths["prepare_reviewer"],
            "code_reviewer_path": self.paths["code_reviewer"],
            "code_summary_path": self.paths["code_summary"],
            "code_usage_path": self.paths["code_usage"],
        }

    def _workspace_path_summary(self) -> Dict[str, str]:
        return {
            "workspace_dir": self.workspace_paths["workspace_dir"],
            "project_dir": self.workspace_paths["project_dir"],
            "dataset_dir": self.workspace_paths["dataset_dir"],
            "model_dir": self.workspace_paths["model_dir"],
            "results_dir": self.workspace_paths["results_dir"],
            "science_results_dir": self.workspace_paths["science_results_dir"],
            "agent_reports_dir": self.workspace_paths["agent_reports_dir"],
            "mcp_status_path": self.paths["mcp_status"],
        }

    def _build_planner_artifact_registry(self) -> ArtifactRegistry:
        registry = ArtifactRegistry(workspace_root=self.workspace_root)
        registry.add(
            ArtifactSpec(
                artifact_id="science.plan",
                stage="science_planner",
                path=planner_rel(self.scope, "latest.json"),
                kind="json",
                schema_name="science_plan",
                description="Validated science condition plan with all-components and component-disabled experiment conditions.",
            )
        )
        registry.add(
            ArtifactSpec(
                artifact_id="science.blueprint",
                stage="science_planner",
                path=planner_rel(self.scope, "blueprint.md"),
                kind="file",
                required=False,
                description="Planner notes and rationale for science condition workers.",
            )
        )
        registry.add(
            ArtifactSpec(
                artifact_id="runtime.science.planner_report",
                stage="science_planner",
                path=planner_rel(self.scope, "planner_report.json"),
                kind="json",
                required=False,
                writer="runtime",
                description="Runtime-owned science planner report.",
            )
        )
        return registry

    def _build_user_prompt(self) -> str:
        step_fields = format_field_bullets(SCIENCE_CONDITION_STEP_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        condition_review_fields = format_field_bullets(SCIENCE_CONDITION_REVIEW_FIELDS)
        planner_registry = self._build_planner_artifact_registry()
        artifact_text = artifact_prompt_context(planner_registry)
        blueprint_path = self.paths.get("science_blueprint", "")
        return "\n".join(
            [
                "## Task: Run Science Conditions",
                "",
                "Science has one unified experiment model: each condition declares `enabled_components` and `disabled_components`.",
                "An all-components condition is the reference case where `disabled_components` is empty. Each component-disabled condition removes exactly one canonical idea component and references an earlier all-components condition.",
                "",
                "### Master Plan",
                self.plan,
                "",
                "### Code Summary",
                self.code_summary or "(empty)",
                "",
                "### Code Usage",
                self.code_usage or "(empty)",
                "",
                "### Canonical Idea Components",
                format_canonical_components_markdown(self.canonical_components),
                "",
                "### Input Paths",
                format_named_paths(self._input_paths()),
                "",
                "### Workspace Paths",
                format_named_paths(self._workspace_path_summary()),
                "",
                "### Required Planner Output",
                "Write a JSON object with `stages`, `summary`, and `usage_notes`.",
                "You must write the managed artifact `science.plan` with Xcientist artifact tools until it is accepted by the hook.",
                "The managed `science.plan` artifact must be a JSON object with a top-level `stages` list; the hook validates that list as the executable science contract.",
                "Do not write alternate science plans, planner reports, or runtime reports with generic write_file/edit_file/bash; use `write_artifact` for `science.plan`.",
                "After `science.plan` and optional `science.blueprint` are accepted, stop using tools and return the required final JSON object directly.",
                "",
                artifact_text,
                "Each condition must include:",
                step_fields,
                "",
                "### Runtime Setup Planning",
                "You are responsible for determining formal science hyperparameters at runtime. Do not assume config defaults are correct.",
                "Use local evidence first: project runner help, code defaults, prepared dataset size, code smoke evidence, prior phase reports, and lightweight bounded probes if needed.",
                "Use external web/Tavily-style research only when local evidence is insufficient for a task-specific setup decision. If used, record the exact source in `source_basis` and `science.blueprint`.",
                "Before relying on external search, read `mcp_status_path`. If MCP/Tavily is unavailable, make the setup decision from local evidence or state the limitation in `source_basis`; do not invent external-source support.",
                "Small probes may calibrate shape, memory, runtime, or command syntax, but probe/smoke/debug outputs are not formal science evidence.",
                "Every final condition must set `run_level: \"full\"` and must not use smoke, dry-run, debug, probe-only, or quick commands.",
                "`training_protocol` must include `epochs`, `max_batches`, `batch_size`, `device`, `seed`, `expected_runtime_sec`, and `full_setup_basis`.",
                "`evaluation_protocol` must include `horizons`, `mask_rates`, `mask_patterns`, `metrics`, `reference_condition_id`, "
                "`perturbation_boundary`, `preprocessing_boundary`, and `ablation_isolation_assumptions`.",
                "`setup_rationale`, `source_basis`, and `runtime_probe_summary` must explain why the protocol is appropriate for this workspace and task.",
                "For robustness or stress tests, define where synthetic perturbations enter the pipeline and whether they must pass through the same preprocessing/component path as natural inputs. If a perturbation intentionally bypasses a component, the rationale and interpretation rule must say so explicitly.",
                "",
                "Planning rules enforced by hooks:",
                "- Use one `science` plan only.",
                "- Put all raw outputs under `results/science/<condition_id>/`.",
                "- Run commands from the workspace root; do not `cd project`.",
                "- Every condition's `enabled_components` plus `disabled_components` must cover `idea.json.components` exactly, with no extras.",
                "- Plan exactly `1 + len(idea.json.components)` conditions.",
                "- The first condition must be the only all-components reference: `disabled_components: []`, all canonical components enabled, and `reference_condition_id: null` or empty.",
                "- Every later condition must disable exactly one canonical component, must reference the first all-components condition, and must make that single disable/toggle visible in command or config arguments.",
                "- Across those component-disabled conditions, every canonical idea component must be disabled exactly once, with no repeats and no omissions.",
                "- `evaluation_protocol.reference_condition_id` must match the condition `reference_condition_id` exactly; all-components reference conditions use an empty string or null-equivalent there.",
                "- If a command includes epochs, batch size, seed, device, or reference condition options, those values must match the protocol fields exactly.",
                "- Do not write `ablation_results.json`; final materialization is runtime-owned after reviewer-approved science evidence.",
                "- Each condition has exactly one managed evidence artifact: `science.<condition_id>.evidence` under `agent_reports/science/evidence/<condition_id>.json`.",
                "- The condition `raw_evidence` entries must be files that the worker will also list in that managed evidence manifest.",
                "- The evidence manifest must record condition_id, enabled_components, disabled_components, reference_condition_id, run_level, exact command, returncode, output_dir, raw_outputs, logs, metrics_files, dataset_bindings, model_bindings, and duration_sec or started_at/ended_at.",
                "",
                f"- After exploring the workspace, write the managed artifact `science.blueprint` to `{blueprint_path}` capturing runners, command templates, component toggles, runtime setup rationale, local/external sources, probe results, final hyperparameters, expected outputs, and reference comparisons.",
                "",
                f"The runtime will execute each condition through `{self.worker_role}` and review it through `{self.reviewer_role}`.",
                "The final phase-level reviewer report must include:",
                verdict_fields,
                "",
                "Each condition reviewer report must also include:",
                condition_review_fields,
            ]
        )

    async def _plan(self) -> Dict[str, Any]:
        if self.resume and os.path.exists(self.plan_path):
            try:
                payload = self._load_required_plan_payload_from_artifact()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                print_status("planner", "SKIP", "reusing existing science plan", color=Colors.OKGREEN)
                materialize_executable_plan(
                    workspace_root=self.workspace_root,
                    scope=self.scope,
                    plan_payload=payload,
                    planner_report={
                        "scope": self.scope,
                        "summary": payload["summary"],
                        "usage_notes": payload["usage_notes"],
                        "plan_path": self.plan_path,
                        "blueprint_path": self.paths.get("science_blueprint", ""),
                        "resume_reused": True,
                    },
                )
                return payload

        registry = self._build_planner_artifact_registry()
        write_artifact_registry_snapshot(registry)
        self.set_artifact_context(registry.to_context(stage="science_planner", step_id="plan", attempt=1))
        output_schema = planner_output_schema(step_schema=_science_step_schema())
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            agent_name=self.planner_role,
            output_schema=output_schema,
            extra_tool_metadata={
                "xcientist_prefinish_gate": planner_artifact_prefinish_gate(
                    output_schema=output_schema,
                    registry=registry,
                    plan_artifact_id="science.plan",
                )
            },
            enable_mcp=True,
        )
        _ = result
        payload = self._load_required_plan_payload_from_artifact()
        materialize_executable_plan(
            workspace_root=self.workspace_root,
            scope=self.scope,
            plan_payload=payload,
            planner_report={
                "scope": self.scope,
                "summary": payload["summary"],
                "usage_notes": payload["usage_notes"],
                "plan_path": self.plan_path,
                "blueprint_path": self.paths.get("science_blueprint", ""),
            },
        )
        return payload

    def _load_required_plan_payload_from_artifact(self) -> Dict[str, Any]:
        if not os.path.exists(self.plan_path):
            raise RuntimeError(
                f"Missing managed planner artifact `science.plan` at `{self.plan_path}`. "
                "The science planner must write it with `write_artifact` before finishing."
            )
        try:
            with open(self.plan_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Could not read managed planner artifact `science.plan`: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Managed planner artifact `science.plan` must be a JSON object.")
        if not isinstance(payload.get("stages"), list):
            raise RuntimeError("Managed planner artifact `science.plan.stages` must be a list.")
        for field in ("summary", "usage_notes"):
            if not isinstance(payload.get(field), str):
                raise RuntimeError(f"Managed planner artifact `science.plan.{field}` must be a string.")
        return payload

    async def _call_worker(
        self,
        step: Dict[str, Any],
        previous_review: Dict[str, Any] | None,
        artifact_context: Dict[str, Any],
        prefinish_gate=None,
    ) -> Dict[str, Any]:
        self.set_artifact_context(artifact_context)
        artifact_text = artifact_prompt_context(ArtifactRegistry.from_context(artifact_context))
        blueprint_path = self.paths.get("science_blueprint", "")
        blueprint_hint = ""
        if blueprint_path and os.path.exists(blueprint_path):
            blueprint_hint = f"""
Before executing, read the planner's blueprint:
`{blueprint_path}`
"""
        condition_id = str(step.get("condition_id") or "").strip()
        evidence_rel = (
            artifact_rel(self.scope, condition_id, "evidence.json")
            if condition_id
            else "agent_reports/science/evidence/<condition_id>.json"
        )
        reference_paths = format_named_paths(
            {
                "idea_md": os.path.realpath(self.paths.get("idea") or self.idea_path),
                "idea_json": os.path.realpath(self.paths["idea_json"]),
                "code_phase_report": self.paths["code_reviewer"],
                "code_summary": self.paths["code_summary"],
                "code_usage": self.paths["code_usage"],
                "science_plan": self.paths["science_executable_plan"],
                "science_blueprint": blueprint_path,
                "science_evidence_manifest": evidence_rel,
            }
        )
        prompt = f"""Execute this science condition.
{blueprint_hint}
Reference paths:
{reference_paths}

Condition contract:
```json
{json.dumps(step, ensure_ascii=False, indent=2)}
```

{artifact_text}

Previous reviewer feedback:
```json
{json.dumps(previous_review or {}, ensure_ascii=False, indent=2)}
```

Return structured worker JSON only. Do not write report files; the STOP/prefinish hook writes structured worker attempts and latest files under `agent_reports/science/worker/<condition>/`.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=self.worker_role,
            system_prompt=science_worker_prompt(),
            output_schema=worker_output_schema(),
            extra_tool_metadata={"xcientist_prefinish_gate": prefinish_gate} if prefinish_gate else None,
        )
        return result["output"]

    async def _call_reviewer(self, step: Dict[str, Any], worker_payload: Dict[str, Any], artifact_context: Dict[str, Any]) -> list[Dict[str, Any]]:
        self.set_artifact_context(artifact_context)
        artifact_text = artifact_prompt_context(ArtifactRegistry.from_context(artifact_context))
        condition_id = str(step.get("condition_id") or "").strip()
        evidence_rel = (
            artifact_rel(self.scope, condition_id, "evidence.json")
            if condition_id
            else "agent_reports/science/evidence/<condition_id>.json"
        )
        reference_paths = format_named_paths(
            {
                "idea_md": os.path.realpath(self.paths.get("idea") or self.idea_path),
                "idea_json": os.path.realpath(self.paths["idea_json"]),
                "code_phase_report": self.paths["code_reviewer"],
                "code_summary": self.paths["code_summary"],
                "code_usage": self.paths["code_usage"],
                "science_plan": self.paths["science_executable_plan"],
                "science_blueprint": self.paths.get("science_blueprint", ""),
                "science_evidence_manifest": evidence_rel,
            }
        )
        async def _run_one(reviewer_id: str) -> Dict[str, Any]:
            prompt = f"""Review this science condition.

Reference paths:
{reference_paths}

Condition contract:
```json
{json.dumps(step, ensure_ascii=False, indent=2)}
```

{artifact_text}

Worker report:
```json
{json.dumps(worker_payload, ensure_ascii=False, indent=2)}
```

Return structured review JSON only. Do not write report files; the STOP/prefinish hook writes structured review attempts and latest files under `agent_reports/science/review/<condition>/<reviewer_id>/`.
"""
            result = await self.run(
                user_prompt=prompt,
                agent_name=reviewer_id,
                system_prompt=science_reviewer_prompt(reviewer_id),
                output_schema=review_output_schema(include_science_condition_fields=True),
                purpose="prefinish_review",
                extra_tool_metadata={"xcientist_expected_reviewer_id": reviewer_id},
            )
            review = result["output"]
            review.setdefault("reviewer_id", reviewer_id)
            return review

        return await asyncio.gather(*[_run_one(reviewer_id) for reviewer_id in SCIENCE_REVIEWER_IDS])

    def _phase_summary_payload(self, plan_payload: Dict[str, Any], step_result: Dict[str, Any]) -> Dict[str, Any]:
        ok = step_result["status"] == "PASS"
        failed = step_result.get("failed_review_report") or {}
        reports = step_result.get("step_reports", []) if ok else []
        component_map: Dict[str, Dict[str, Any]] = {}
        reference_conditions = []
        component_conditions = []
        for step, report in zip(plan_payload.get("stages") or [], reports):
            if not isinstance(step, dict) or not isinstance(report, dict):
                continue
            disabled = [str(item) for item in step.get("disabled_components") or [] if str(item).strip()]
            condition_id = str(step.get("condition_id") or "")
            if disabled:
                component_conditions.append(condition_id)
            else:
                reference_conditions.append(condition_id)
            for component in disabled:
                component_map.setdefault(
                    component,
                    {
                        "result": report.get("result", ""),
                        "metric": report.get("metric", ""),
                        "value": report.get("value", ""),
                        "confidence": report.get("confidence", 0.0),
                        "analysis": report.get("analysis", ""),
                        "method_context": report.get("method_context", ""),
                        "follow_up_required": report.get("follow_up_required", False),
                        "condition_id": condition_id,
                        "reference_condition_id": step.get("reference_condition_id") or "",
                    },
                )
        coverage_issues = []
        if ok:
            try:
                canonical_names = [
                    item["component"]
                    for item in load_canonical_components(self.workspace_root)
                ]
            except Exception as exc:
                canonical_names = []
                coverage_issues.append(f"Failed to load canonical idea components: {exc}")
            if canonical_names:
                missing = [name for name in canonical_names if name not in component_map]
                extra = sorted(set(component_map) - set(canonical_names))
                if missing:
                    coverage_issues.append(
                        "science_component_results missing component conclusions for: "
                        + ", ".join(missing)
                    )
                if extra:
                    coverage_issues.append(
                        "science_component_results contains non-canonical components: "
                        + ", ".join(extra)
                    )
            if coverage_issues:
                ok = False
                failed = {
                    "evidence_summary": "science phase aggregation did not cover every canonical idea component",
                    "blocking_issues": coverage_issues,
                    "required_followup": coverage_issues,
                    "terminal_blocker": False,
                    "next_worker_input": (
                        "Repair the science phase before finalization: every canonical idea component "
                        "must have one component-disabled full run and one reviewer-approved "
                        "structured_findings.component_result in agent_reports/science/phase.json. "
                        + " ".join(coverage_issues)
                    ),
                }
        payload = with_phase_defaults(
            {
                "status": "PASS" if ok else "FAIL",
                "scope": self.scope,
                "checked_artifacts": phase_checked_artifacts(
                    step_result,
                    self.plan_path,
                    self.paths["science_executable_plan"],
                    self.paths["science_planner_report"],
                    artifact_ledger_path(self.workspace_root),
                ),
                "findings": [] if ok else list(failed.get("findings") or []),
                "evidence_summary": plan_payload["summary"] if ok else str(failed.get("evidence_summary") or "science phase incomplete"),
                "phase_completion_status": "complete" if ok else "partial",
                "ready_for_next_phase": bool(ok),
                "blocking_issues": [] if ok else list(failed.get("blocking_issues") or [f"Failed condition: {step_result.get('failed_step', {}).get('condition_id', 'unknown')}"]),
                "required_followup": [] if ok else list(failed.get("required_followup") or []),
                "artifact_role": "phase_result",
                "run_level": "full",
                "self_contained_project": True,
                "self_contained_violations": [],
                "artifact_ledger_present": os.path.exists(artifact_ledger_path(self.workspace_root)),
                "artifact_ledger_path": artifact_ledger_path(self.workspace_root),
                "terminal_blocker": bool(failed.get("terminal_blocker")),
                "next_worker_input": str(failed.get("next_worker_input") or ""),
            },
            scope=self.scope,
        )
        if component_map:
            payload["science_component_results"] = component_map
        if ok:
            confidences = [float(entry.get("confidence") or 0.0) for entry in component_map.values()]
            payload["science_conditions"] = {
                "reference_conditions": reference_conditions,
                "component_disabled_conditions": component_conditions,
            }
            payload["summary"] = {
                "feasible": True,
                "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "key_findings": [str(entry.get("analysis") or "") for entry in component_map.values()],
            }
        return payload

    async def execute(self) -> Dict[str, Any]:
        plan_payload = await self._plan()
        steps = plan_payload["stages"]
        for step in steps:
            ensure_runtime_report_paths(self.scope, step, self.workspace_root)
        step_result = await execute_step_with_prefinish_review(
            steps=steps,
            scope=self.scope,
            workspace_root=self.workspace_root,
            call_worker=self._call_worker,
            call_reviewer=self._call_reviewer,
        )
        phase_report = self._phase_summary_payload(plan_payload, step_result)
        write_json_file(self.review_report_path, phase_report)
        summary_lines = [
            "# Science Summary",
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
    agent = ScienceAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        project_root=project_root,
        workspace_root=workspace_root,
        plan=plan,
        code_summary=code_summary,
        code_usage=code_usage,
        model=model or get_agent_model("science_agent", "science"),
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
