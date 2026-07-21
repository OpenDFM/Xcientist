from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.agents.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.agents.prepare.reviewer import (
    PREPARE_REVIEWERS,
    prepare_reviewer_prompt,
)
from src.agents.experiment_agent.agents.prepare.worker import (
    PREPARE_DATASET_WORKER,
    PREPARE_ENV_WORKER,
    PREPARE_MODEL_WORKER,
    PREPARE_REPO_WORKER,
    PREPARE_SYNTHESIS_WORKER,
    prepare_worker_prompt,
)
from src.agents.experiment_agent.config import (
    ensure_experiment_dirs,
    get_prepare_agent_model,
    normalize_workspace_path,
)
from src.agents.experiment_agent.runtime.contracts import (
    PHASE_VERDICT_FIELDS,
    PREPARE_STAGE_CONTRACT_FIELDS,
    format_field_bullets,
)
from src.agents.experiment_agent.runtime.idea_components import (
    format_canonical_components_markdown,
    load_canonical_components,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    write_json_file,
)
from src.agents.experiment_agent.runtime.artifacts import (
    ArtifactRegistry,
    ArtifactSpec,
    artifact_ledger_path,
    artifact_prompt_context,
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
from src.agents.experiment_agent.runtime.report_layout import planner_rel


@dataclass
class PrepareReport:
    experiment_id: str
    workspace_dir: str
    project_dir: str
    repos_dir: str
    dataset_dir: str
    model_dir: str
    results_dir: str
    reports_dir: str
    idea_md_path: str


def _prepare_step_schema() -> Dict[str, Any]:
    properties = {
        "stage_id": {"type": "string"},
        "goal": {"type": "string"},
        "input_paths": {"type": "object"},
        "repos_policy": {"type": "string"},
        "project_must_be_self_contained": {"type": "boolean"},
        "research_required": {"type": "boolean"},
        "acquisition_required": {"type": "boolean"},
        "existing_local_hints": {"type": "array", "items": {"type": "string"}},
        "done_condition": {"type": "string"},
        "artifact_ids": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }


class PrepareAgent(BaseAgent):
    STAGE_WORKER_ROLES = {
        "repos": PREPARE_REPO_WORKER,
        "dataset": PREPARE_DATASET_WORKER,
        "model": PREPARE_MODEL_WORKER,
        "env": PREPARE_ENV_WORKER,
        "synthesis": PREPARE_SYNTHESIS_WORKER,
    }

    def __init__(
        self,
        model: Optional[str] = None,
        verbose: bool = True,
        workspace_root: Optional[str] = None,
    ):
        super().__init__(
            agent_type="PrepareAgent",
            model=model or get_prepare_agent_model(),
            verbose=verbose,
            workspace_root=workspace_root,
        )

    def _build_planner_artifact_registry(self, workspace_dir: str) -> ArtifactRegistry:
        registry = ArtifactRegistry(workspace_root=workspace_dir)
        registry.add(
            ArtifactSpec(
                artifact_id="prepare.plan",
                stage="prepare_planner",
                path=planner_rel("prepare", "latest.json"),
                kind="json",
                schema_name="prepare_plan",
                description="Validated prepare phase plan ordered as repos, dataset, model, env, synthesis.",
            )
        )
        registry.add(
            ArtifactSpec(
                artifact_id="prepare.blueprint",
                stage="prepare_planner",
                path=planner_rel("prepare", "blueprint.md"),
                kind="file",
                required=False,
                description="Planner notes and rationale for prepare workers.",
            )
        )
        registry.add(
            ArtifactSpec(
                artifact_id="runtime.prepare.planner_report",
                stage="prepare_planner",
                path=planner_rel("prepare", "planner_report.json"),
                kind="json",
                required=False,
                writer="runtime",
                description="Runtime-owned prepare planner report.",
            )
        )
        return registry

    def _build_user_prompt(self, **kwargs) -> str:
        pb = PromptBuilder()
        stage_contract_fields = format_field_bullets(PREPARE_STAGE_CONTRACT_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        canonical_components = kwargs.get("canonical_components") or []
        pb.add_header("Prepare Workspace Task")
        for key in (
            "experiment_id",
            "idea_json_path",
            "workspace_dir",
            "project_dir",
            "repos_dir",
            "dataset_dir",
            "model_dir",
            "results_dir",
            "reports_dir",
            "mcp_status_path",
        ):
            pb.add_key_value(key, str(kwargs.get(key) or ""))
        pb.add_text("")
        pb.add_header("Canonical Idea Components", level=2)
        pb.add_text(format_canonical_components_markdown(canonical_components) if canonical_components else "- (none)")
        pb.add_header("Required Planner Output", level=2)
        pb.add_text("Write a JSON object with `stages`, `summary`, and `usage_notes`.")
        pb.add_text("You must write the managed artifact `prepare.plan` with Xcientist artifact tools until it is accepted by the hook.")
        pb.add_text("The managed `prepare.plan` artifact must be a JSON object with a top-level `stages` list; the hook validates that list as the executable prepare contract.")
        pb.add_text("Do not write `prepare_plan.json` with generic write_file/edit_file/bash; use `write_artifact`.")
        pb.add_text("Do not write `prepare_planner_report.json` or other planner/runtime reports with tools.")
        pb.add_text("After `prepare.plan` and optional `prepare.blueprint` are accepted, stop using tools and return the required final JSON object directly.")
        artifact_text = kwargs.get("artifact_text")
        if artifact_text:
            pb.add_text("")
            pb.add_text(str(artifact_text))
        pb.add_header("Prepare Acquisition Protocol", level=2)
        pb.add_text(
            "Assume no repository, dataset, model, benchmark, or virtualenv is pre-approved. "
            "The plan must make prepare discover candidates, acquire real resources, verify them locally, "
            "build the final environment after resource choices are fixed, and synthesize a stable handoff."
        )
        pb.add_text(
            "Tavily/MCP search may be used for discovery, but search summaries are not proof. "
            "Proof must be local evidence: cloned commit, downloaded/located files, checksums or sizes, "
            "loader/schema probes, API dry-runs or model load probes, and venv/import smoke logs."
        )
        pb.add_text(
            "Before relying on external search, read `mcp_status_path`. If MCP/Tavily is unavailable, "
            "use local evidence or produce a truthful BLOCKED artifact with concrete attempted queries and missing requirements."
        )
        pb.add_text(
            "If no suitable resource exists, the stage must produce a clear blocker with candidate rejection reasons. "
            "Do not plan toy data, mock datasets, placeholder models, or degraded proxy experiments as a success path."
        )
        pb.add_text(
            "`prepare.discovery` is the resource-decision contract, not a prose summary. "
            "The repos stage must produce a structured decision matrix with `task_signature`, "
            "`resource_requirements`, `mcp_status_snapshot`, `selection_criteria`, concrete `queries`, "
            "`candidate_table` for repos/datasets/models, `selected_candidate_ids`, `rejected_candidates`, "
            "`evidence_gaps`, `selected_resources`, and `selection_rationale`. The deterministic hook rejects "
            "READY discovery artifacts that do not expose this decision chain."
        )
        pb.add_text(
            "Each managed prepare JSON artifact must declare `status: \"READY\"` or `status: \"BLOCKED\"`. "
            "READY means the resource was acquired or verified with local evidence. BLOCKED means the artifact contains "
            "a `blocker` object with `reason`, `attempted_queries`, `rejected_candidates`, `missing_requirements`, "
            "`user_action_required`, and local `evidence_paths`. A credible BLOCKED stage stops prepare cleanly and is not "
            "a substitute for success."
        )
        pb.add_text("The stage list must be ordered exactly as: repos, dataset, model, env, synthesis.")
        pb.add_text(
            "Stage artifact obligations are fixed: "
            "repos -> `prepare.discovery` at `agent_reports/prepare/artifacts/discovery.json` and `prepare.repos` at `agent_reports/prepare/artifacts/repos.json`; "
            "dataset -> `prepare.dataset` at `agent_reports/prepare/artifacts/dataset.json`; "
            "model -> `prepare.model` at `agent_reports/prepare/artifacts/model.json`; "
            "env -> `prepare.env` at `agent_reports/prepare/artifacts/env.json`; "
            "synthesis -> `prepare.idea` at `agent_reports/prepare/artifacts/idea.md` and `prepare.target_inventory` at `agent_reports/prepare/artifacts/target_inventory.json`. "
            "Each stage's `artifact_ids` and `done_condition` must mention exactly its assigned managed artifacts. "
            "Each `done_condition` must require Xcientist artifact tools and must name the artifact ledger as proof; "
            "the runtime will reject invalid contracts instead of repairing them."
        )
        pb.add_text("Each stage must include:")
        pb.add_text(stage_contract_fields)
        pb.add_text(
            "The runtime will execute each stage through the matching worker role. "
            "Before the worker can finish, deterministic hooks validate formal artifacts, then multiple read-only prepare reviewers run in parallel."
        )
        pb.add_text("The final phase-level reviewer report must include:")
        pb.add_text(verdict_fields)
        pb.add_text("")
        pb.add_header("Synthesis Stage Requirement", level=2)
        pb.add_text(
            "The synthesis stage (`stage_id=synthesis`) must include in its `done_condition` that "
            "`agent_reports/prepare/artifacts/idea.md` must contain a section with the exact heading "
            "`## Canonical Idea Components`, listing all components from the Canonical Idea Components "
            "section above in canonical order (by index). The reviewer enforces this, so the worker "
            "must be instructed through the done_condition to produce it."
        )
        return pb.build()

    async def _call_worker(
        self,
        stage: Dict[str, Any],
        previous_review: Dict[str, Any] | None,
        artifact_context: Dict[str, Any],
        prefinish_gate=None,
    ) -> Dict[str, Any]:
        stage_id = str(stage.get("stage_id") or "")
        self.set_artifact_context(artifact_context)
        artifact_text = artifact_prompt_context(ArtifactRegistry.from_context(artifact_context))
        prompt = f"""Execute this prepare stage.

Stage contract:
```json
{json.dumps(stage, ensure_ascii=False, indent=2)}
```

{artifact_text}

Previous reviewer feedback:
```json
{json.dumps(previous_review or {}, ensure_ascii=False, indent=2)}
```

Return structured worker JSON only. Include `outcome` as `READY` or `BLOCKED`; it must match the managed artifact status. Do not write report files; the STOP/prefinish hook writes structured worker attempts and latest files under `agent_reports/prepare/worker/<stage>/`.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=self.STAGE_WORKER_ROLES.get(stage_id) or PREPARE_REPO_WORKER,
            system_prompt=prepare_worker_prompt(stage_id),
            output_schema=worker_output_schema(include_outcome=True, require_outcome=True),
            extra_tool_metadata={"xcientist_prefinish_gate": prefinish_gate} if prefinish_gate else None,
            enable_mcp=True,
        )
        return result["output"]

    async def _call_reviewer(self, stage: Dict[str, Any], worker_payload: Dict[str, Any], artifact_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.set_artifact_context(artifact_context)
        artifact_text = artifact_prompt_context(ArtifactRegistry.from_context(artifact_context))
        base_prompt = f"""Review this prepare stage.

Stage contract:
```json
{json.dumps(stage, ensure_ascii=False, indent=2)}
```

{artifact_text}

Worker report:
```json
{json.dumps(worker_payload, ensure_ascii=False, indent=2)}
```

Return structured review JSON only. Do not write report files; the STOP/prefinish hook writes structured review attempts and latest files under `agent_reports/prepare/review/<stage>/<reviewer_id>/`.
"""
        async def _run_one(reviewer_id: str) -> Dict[str, Any]:
            result = await self.run(
                user_prompt=base_prompt,
                agent_name=reviewer_id,
                system_prompt=prepare_reviewer_prompt(reviewer_id),
                output_schema=review_output_schema(),
                purpose="prefinish_review",
                extra_tool_metadata={"xcientist_expected_reviewer_id": reviewer_id},
            )
            payload = result["output"]
            if isinstance(payload, dict):
                payload.setdefault("reviewer_id", reviewer_id)
            return payload

        return await asyncio.gather(*[_run_one(reviewer_id) for reviewer_id in PREPARE_REVIEWERS])

    async def prepare_workspace(
        self,
        experiment_id: str,
        force: bool = False,
        clone_depth: int = 1,
        skip_repos: bool = False,
        skip_datasets: bool = False,
    ) -> PrepareReport:
        _ = force, clone_depth, skip_repos, skip_datasets
        paths = ensure_experiment_dirs(experiment_id)
        workspace_dir = normalize_workspace_path(str(paths.get("workspace_dir") or ""))
        project_dir = normalize_workspace_path(str(paths.get("project_dir") or ""))
        repos_dir = normalize_workspace_path(str(paths.get("repos_dir") or os.path.join(workspace_dir, "repos")))
        dataset_dir = normalize_workspace_path(str(paths.get("dataset_dir") or os.path.join(workspace_dir, "dataset_candidate")))
        model_dir = normalize_workspace_path(str(paths.get("model_dir") or os.path.join(workspace_dir, "model_candidate")))
        results_dir = normalize_workspace_path(str(paths.get("results_dir") or os.path.join(workspace_dir, "results")))
        reports_dir = normalize_workspace_path(str(paths.get("reports_dir") or os.path.join(workspace_dir, "agent_reports")))
        self._refresh_runtime_roots(workspace_dir)

        idea_json_path = os.path.join(workspace_dir, "idea.json")
        with open(idea_json_path, "r", encoding="utf-8") as f:
            _ = json.load(f)
        canonical_components = load_canonical_components(workspace_dir, idea_json_path=idea_json_path)
        planner_registry = self._build_planner_artifact_registry(workspace_dir)
        write_artifact_registry_snapshot(planner_registry)
        self.set_artifact_context(
            planner_registry.to_context(stage="prepare_planner", step_id="plan", attempt=1)
        )
        prompt = self._build_user_prompt(
            experiment_id=experiment_id,
            idea_json_path=idea_json_path,
            workspace_dir=workspace_dir,
            project_dir=project_dir,
            repos_dir=repos_dir,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            results_dir=results_dir,
            reports_dir=reports_dir,
            mcp_status_path=artifact_paths(workspace_dir, project_dir)["mcp_status"],
            canonical_components=canonical_components,
            artifact_text=artifact_prompt_context(planner_registry),
        )
        output_schema = planner_output_schema(step_schema=_prepare_step_schema())
        plan_result = await self.run(
            user_prompt=prompt,
            agent_name="prepare-planner",
            output_schema=output_schema,
            extra_tool_metadata={
                "xcientist_prefinish_gate": planner_artifact_prefinish_gate(
                    output_schema=output_schema,
                    registry=planner_registry,
                    plan_artifact_id="prepare.plan",
                )
            },
            enable_mcp=True,
        )
        _ = plan_result
        file_payload = load_json_file(artifact_paths(workspace_dir, project_dir)["prepare_plan"])
        if not isinstance(file_payload, dict):
            raise RuntimeError(
                "Missing managed planner artifact `prepare.plan`. "
                "The prepare planner must write `agent_reports/prepare/plan/latest.json` "
                "with `write_artifact` before finishing."
            )
        if not isinstance(file_payload.get("stages"), list):
            raise RuntimeError("Managed planner artifact `prepare.plan.stages` must be a list.")
        for field in ("summary", "usage_notes"):
            if not isinstance(file_payload.get(field), str):
                raise RuntimeError(f"Managed planner artifact `prepare.plan.{field}` must be a string.")
        plan_payload = file_payload
        materialize_executable_plan(
            workspace_root=workspace_dir,
            scope="prepare",
            plan_payload=plan_payload,
            planner_report={
                "scope": "prepare",
                "summary": plan_payload["summary"],
                "usage_notes": plan_payload["usage_notes"],
            },
        )

        for stage in plan_payload["stages"]:
            # Resolve relative paths against workspace_dir
            for path_key in ("worker_report_path", "review_report_path", "hook_report_path"):
                if path_key in stage and stage[path_key]:
                    p = stage[path_key]
                    if not os.path.isabs(p):
                        stage[path_key] = os.path.join(workspace_dir, p)
        step_result = await execute_step_with_prefinish_review(
            steps=plan_payload["stages"],
            scope="prepare",
            workspace_root=workspace_dir,
            call_worker=self._call_worker,
            call_reviewer=self._call_reviewer,
        )
        ok = step_result["status"] == "PASS"
        blocked = step_result["status"] == "BLOCKED"
        failed = (
            step_result.get("blocked_review_report")
            if blocked
            else step_result.get("failed_review_report")
        ) or {}
        blocking_issues = [] if ok else list(failed.get("blocking_issues") or [])
        if blocked:
            worker_report = load_json_file(str(failed.get("worker_report_path") or ""))
            if isinstance(worker_report, dict):
                blocking_issues.extend(str(item) for item in worker_report.get("remaining_blockers") or [] if str(item).strip())
            if not blocking_issues:
                blocked_stage = (step_result.get("blocked_step") or {}).get("stage_id") or "unknown"
                blocking_issues.append(f"Prepare blocked at stage `{blocked_stage}`.")
        final_payload = with_phase_defaults(
            {
                "status": "PASS" if ok else "FAIL",
                "scope": "prepare",
                "checked_artifacts": phase_checked_artifacts(
                    step_result,
                    artifact_paths(workspace_dir, project_dir)["prepare_plan"],
                    artifact_paths(workspace_dir, project_dir)["prepare_executable_plan"],
                    artifact_paths(workspace_dir, project_dir)["prepare_planner_report"],
                    artifact_ledger_path(workspace_dir),
                ),
                "findings": [] if ok else list(failed.get("findings") or []),
                "evidence_summary": plan_payload["summary"] if ok else str(failed.get("evidence_summary") or "prepare phase incomplete"),
                "phase_completion_status": "complete" if ok else ("blocked" if blocked else "partial"),
                "ready_for_next_phase": bool(ok),
                "blocking_issues": blocking_issues,
                "required_followup": [] if ok else list(failed.get("required_followup") or []),
                "artifact_role": "phase_result",
                "run_level": "full",
                "self_contained_project": True,
                "self_contained_violations": [],
                "artifact_ledger_present": os.path.exists(artifact_ledger_path(workspace_dir)),
                "artifact_ledger_path": artifact_ledger_path(workspace_dir),
                "terminal_blocker": bool(blocked or failed.get("terminal_blocker")),
                "next_worker_input": str(failed.get("next_worker_input") or ""),
                "blocked_stage": (step_result.get("blocked_step") or {}).get("stage_id") if blocked else "",
            },
            scope="prepare",
        )
        write_json_file(artifact_paths(workspace_dir, project_dir)["prepare_reviewer"], final_payload)
        idea_md_path = artifact_paths(workspace_dir, project_dir)["idea"]
        return PrepareReport(
            experiment_id=experiment_id,
            workspace_dir=os.path.realpath(workspace_dir),
            project_dir=os.path.realpath(project_dir),
            repos_dir=os.path.realpath(repos_dir),
            dataset_dir=os.path.realpath(dataset_dir),
            model_dir=os.path.realpath(model_dir),
            results_dir=os.path.realpath(results_dir),
            reports_dir=os.path.realpath(reports_dir),
            idea_md_path=os.path.realpath(idea_md_path),
        )


async def run_prepare(
    experiment_id: str,
    force: bool = False,
    clone_depth: int = 1,
    skip_repos: bool = False,
    skip_datasets: bool = False,
    model: Optional[str] = None,
    verbose: bool = True,
) -> PrepareReport:
    paths = ensure_experiment_dirs(experiment_id)
    workspace_root = normalize_workspace_path(str(paths.get("workspace_dir") or ""))
    agent = PrepareAgent(
        model=model or get_prepare_agent_model(),
        verbose=bool(verbose),
        workspace_root=workspace_root,
    )
    return await agent.prepare_workspace(
        experiment_id=experiment_id,
        force=bool(force),
        clone_depth=int(clone_depth),
        skip_repos=bool(skip_repos),
        skip_datasets=bool(skip_datasets),
    )
