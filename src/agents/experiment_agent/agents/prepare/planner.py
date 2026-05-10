from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.agents.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.agents.prepare.validator import PREPARE_VALIDATOR, prepare_validator_prompt
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
    coerce_plan_payload,
    load_json_file,
    resolve_prepare_idea_path,
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
        "stage_contract_path": {"type": "string"},
        "executor_report_path": {"type": "string"},
        "input_paths": {"type": "object"},
        "allowed_write_roots": {"type": "array", "items": {"type": "string"}},
        "required_output_roots": {"type": "array", "items": {"type": "string"}},
        "worker_report_path": {"type": "string"},
        "validator_report_path": {"type": "string"},
        "repos_policy": {"type": "string"},
        "project_must_be_self_contained": {"type": "boolean"},
        "provenance_manifest_path": {"type": "string"},
        "research_required": {"type": "boolean"},
        "acquisition_required": {"type": "boolean"},
        "existing_local_hints": {"type": "array", "items": {"type": "string"}},
        "max_repair_rounds": {"type": "integer"},
        "done_condition": {"type": "string"},
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
        "env": PREPARE_ENV_WORKER,
        "dataset": PREPARE_DATASET_WORKER,
        "model": PREPARE_MODEL_WORKER,
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
        ):
            pb.add_key_value(key, str(kwargs.get(key) or ""))
        pb.add_text("")
        pb.add_header("Canonical Idea Components", level=2)
        pb.add_text(format_canonical_components_markdown(canonical_components) if canonical_components else "- (none)")
        pb.add_header("Required Planner Output", level=2)
        pb.add_text("Write a JSON object with `stages`, `summary`, and `usage_notes`.")
        pb.add_text("The stage list must be ordered exactly as: repos, env, dataset, model, synthesis.")
        pb.add_text("Each stage must include:")
        pb.add_text(stage_contract_fields)
        pb.add_text("The runtime will execute each stage through the matching worker role and review it through `prepare_validator`.")
        pb.add_text("The final phase-level validator report must include:")
        pb.add_text(verdict_fields)
        pb.add_text("")
        pb.add_header("Synthesis Stage Requirement", level=2)
        pb.add_text(
            "The synthesis stage (`stage_id=synthesis`) must include in its `done_condition` that "
            "`prepare_idea.md` must contain a section with the exact heading "
            "`## Canonical Idea Components`, listing all components from the Canonical Idea Components "
            "section above in canonical order (by index). The validator enforces this, so the worker "
            "must be instructed through the done_condition to produce it."
        )
        return pb.build()

    def _synthesize_prepare_blueprint(self, blueprint_path: str, payload: Dict[str, Any]) -> None:
        """Generate a prepare_blueprint.md from the plan payload."""
        if not blueprint_path:
            return
        lines = [
            "# Prepare Phase Blueprint",
            "",
            "## Planner Summary",
            "",
            payload.get("summary", "(none)"),
            "",
            f"## Planned Stages ({len(payload.get('stages', []))} stages)",
            "",
        ]
        for stage in payload.get("stages", []):
            stage_id = stage.get("stage_id", "?")
            lines.append(f"### `{stage_id}`")
            lines.append("")
            lines.append(f"**Goal:** {stage.get('goal', '')}")
            lines.append("")
            if stage.get('done_condition'):
                lines.append(f"**Done when:** {stage['done_condition']}")
                lines.append("")
            if stage.get('research_required'):
                lines.append("**Requires research:** Yes")
                lines.append("")
            if stage.get('acquisition_required'):
                lines.append("**Requires acquisition:** Yes")
                lines.append("")
            if stage.get('existing_local_hints'):
                lines.append(f"**Existing hints:** {', '.join(stage['existing_local_hints'])}")
                lines.append("")
        lines.append(f"## Usage Notes\n\n{payload.get('usage_notes', '')}\n")
        os.makedirs(os.path.dirname(blueprint_path), exist_ok=True)
        with open(blueprint_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    async def _call_worker(self, stage: Dict[str, Any], previous_review: Dict[str, Any] | None) -> Dict[str, Any]:
        stage_id = str(stage.get("stage_id") or "")
        prompt = f"""Execute this prepare stage.

Stage contract:
```json
{json.dumps(stage, ensure_ascii=False, indent=2)}
```

Previous validator feedback:
```json
{json.dumps(previous_review or {}, ensure_ascii=False, indent=2)}
```

Return structured worker JSON only. The runtime will write the worker report file.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=self.STAGE_WORKER_ROLES.get(stage_id) or PREPARE_REPO_WORKER,
            system_prompt=prepare_worker_prompt(stage_id),
            output_schema=worker_output_schema(),
        )
        return result["output"]

    async def _call_validator(self, stage: Dict[str, Any], worker_payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Review this prepare stage.

Stage contract:
```json
{json.dumps(stage, ensure_ascii=False, indent=2)}
```

Worker report:
```json
{json.dumps(worker_payload, ensure_ascii=False, indent=2)}
```

Return structured review JSON only. The runtime will write the validator report file.
"""
        result = await self.run(
            user_prompt=prompt,
            agent_name=PREPARE_VALIDATOR,
            system_prompt=prepare_validator_prompt(),
            output_schema=validator_output_schema(),
        )
        return result["output"]

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
            canonical_components=canonical_components,
        )
        plan_result = await self.run(
            user_prompt=prompt,
            agent_name="prepare-planner",
            output_schema=planner_output_schema(step_schema=_prepare_step_schema()),
        )
        prepare_plan_path = artifact_paths(workspace_dir, project_dir)["prepare_plan"]
        plan_payload = coerce_plan_payload(
            plan_result["output"],
            prepare_plan_path,
            scope="prepare",
        )
        write_json_file(prepare_plan_path, {"stages": plan_payload["stages"]})
        write_json_file(
            artifact_paths(workspace_dir, project_dir)["prepare_planner_report"],
            {"scope": "prepare", "summary": plan_payload["summary"], "usage_notes": plan_payload["usage_notes"]},
        )
        # Generate prepare blueprint
        blueprint_path = artifact_paths(workspace_dir, project_dir).get("prepare_blueprint", "")
        if blueprint_path:
            self._synthesize_prepare_blueprint(blueprint_path, plan_payload)
        expected_order = ["repos", "env", "dataset", "model", "synthesis"]
        planned_order = [stage.get("stage_id") for stage in plan_payload["stages"]]
        if planned_order != expected_order:
            raise RuntimeError(f"Prepare planner must emit stages in order {expected_order}, got {planned_order}")

        # Enrich synthesis stage: ensure prepare_target_inventory.json is a required output
        # and the worker knows to generate it from idea.json components.
        for stage in plan_payload["stages"]:
            if stage.get("stage_id") == "synthesis":
                inv_rel = "agent_reports/prepare_target_inventory.json"
                if inv_rel not in stage.get("required_output_roots", []):
                    stage.setdefault("required_output_roots", []).append(inv_rel)
                if "prepare_target_inventory.json" not in stage.get("done_condition", ""):
                    stage["done_condition"] += (
                        " Also produces prepare_target_inventory.json mapping each idea.json component "
                        "to concrete implementation targets (Python modules, classes, functions, config keys)."
                    )
                # Add inventory path to worker report paths for runtime tracking
                if "agent_reports/prepare_target_inventory.json" not in stage.get("required_output_roots", []):
                    stage["required_output_roots"].append("agent_reports/prepare_target_inventory.json")

        for stage in plan_payload["stages"]:
            # Resolve relative paths against workspace_dir
            for path_key in ("stage_contract_path", "worker_report_path", "validator_report_path", "executor_report_path", "provenance_manifest_path"):
                if path_key in stage and stage[path_key]:
                    p = stage[path_key]
                    if not os.path.isabs(p):
                        stage[path_key] = os.path.join(workspace_dir, p)
            # Stage contract data is already in plan JSON; execute_step_loop passes
            # it in worker/validator prompts. File writes are for debug/audit only.
            # We keep them because prepare uses stage_contract_path for auditability.
            write_json_file(stage["stage_contract_path"], stage)
        step_result = await execute_step_loop(
            steps=plan_payload["stages"],
            scope="prepare",
            workspace_root=workspace_dir,
            call_worker=self._call_worker,
            call_validator=self._call_validator,
        )
        ok = step_result["status"] == "PASS"
        failed = step_result.get("failed_validator_report") or {}
        final_payload = with_phase_defaults(
            {
                "status": "PASS" if ok else "FAIL",
                "scope": "prepare",
                "checked_artifacts": [artifact_paths(workspace_dir, project_dir)["prepare_plan"]],
                "findings": [] if ok else list(failed.get("findings") or []),
                "required_fixes": [] if ok else list(failed.get("required_fixes") or []),
                "evidence_summary": plan_payload["summary"] if ok else str(failed.get("evidence_summary") or "prepare phase incomplete"),
                "phase_completion_status": "complete" if ok else "partial",
                "ready_for_next_phase": bool(ok),
                "blocking_issues": [] if ok else list(failed.get("blocking_issues") or []),
                "required_followup": [] if ok else list(failed.get("required_followup") or []),
                "artifact_role": "phase_result",
                "run_level": "full",
                "self_contained_project": True,
                "self_contained_violations": [],
                "provenance_manifest_present": os.path.exists(artifact_paths(workspace_dir, project_dir)["project_code_provenance"]),
                "provenance_manifest_path": artifact_paths(workspace_dir, project_dir)["project_code_provenance"],
                "terminal_blocker": bool(failed.get("terminal_blocker")),
                "next_worker_input": str(failed.get("next_worker_input") or ""),
            },
            scope="prepare",
        )
        write_json_file(artifact_paths(workspace_dir, project_dir)["prepare_validator"], final_payload)
        idea_md_path = resolve_prepare_idea_path(workspace_dir, project_dir)
        if not os.path.exists(idea_md_path):
            os.makedirs(os.path.dirname(idea_md_path), exist_ok=True)
            with open(idea_md_path, "w", encoding="utf-8") as f:
                f.write("# Prepare Idea\n\n")
        inventory_path = artifact_paths(workspace_dir, project_dir)["prepare_target_inventory"]
        if not os.path.exists(inventory_path) or os.path.getsize(inventory_path) == 0:
            # Generate proper inventory from idea.json components
            components = load_canonical_components(workspace_dir, idea_json_path=idea_json_path)
            inventory = {
                "prepared_targets": [
                    {
                        "component": c["component"],
                        "explanation": c.get("explanation", ""),
                        "index": c.get("index", ""),
                    }
                    for c in components
                ],
                "source": "idea.json",
                "generated_by": "prepare_planner",
            }
            write_json_file(inventory_path, inventory)
        elif ok:
            # Synthesis passed but inventory might still be a stub from a previous run.
            # If it has empty prepared_targets, regenerate.
            existing = load_json_file(inventory_path) or {}
            targets = existing.get("prepared_targets")
            if not targets:
                components = load_canonical_components(workspace_dir, idea_json_path=idea_json_path)
                inventory = {
                    "prepared_targets": [
                        {
                            "component": c["component"],
                            "explanation": c.get("explanation", ""),
                            "index": c.get("index", ""),
                        }
                        for c in components
                    ],
                    "source": "idea.json",
                    "generated_by": "prepare_planner",
                }
                write_json_file(inventory_path, inventory)
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
