from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openhands.sdk import Agent
from openhands.sdk.subagent import register_agent
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool

from src.agents.experiment_agent.agents.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.agents.prepare.step_executor import (
    PREPARE_STEP_EXECUTOR,
    create_prepare_step_executor_agent,
)
from src.agents.experiment_agent.agents.prepare.validator import PREPARE_VALIDATOR
from src.agents.experiment_agent.agents.prepare.worker import PREPARE_WORKER
from src.agents.experiment_agent.config import (
    PLANNER_MAX_TURNS,
    PREPARE_AGENT_MODEL,
    ensure_experiment_dirs,
    normalize_workspace_path,
)
from src.agents.experiment_agent.runtime.contracts import (
    PHASE_VERDICT_FIELDS,
    PREPARE_STAGE_CONTRACT_FIELDS,
    format_field_bullets,
)
from src.agents.experiment_agent.runtime.idea_components import (
    IDEA_COMPONENTS_HEADING,
    format_canonical_components_markdown,
    load_canonical_components,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, workspace_contract_paths
from src.agents.experiment_agent.skills import get_prepare_agent_context
from src.agents.experiment_agent.tools.openhands import SecurityContext


@dataclass
class PrepareReport:
    experiment_id: str
    workspace_dir: str
    project_dir: str
    repos_dir: str
    dataset_dir: str
    results_dir: str
    reports_dir: str
    idea_md_path: str


class PrepareAgent(BaseAgent):
    PREPARE_DEFAULT_MCP_SERVERS = ["filesystem", "fetch"]
    PREPARE_STAGE_SPECS = [
        {
            "stage_id": "repos",
            "executor_type": PREPARE_STEP_EXECUTOR,
            "worker_type": PREPARE_WORKER,
            "validator_type": PREPARE_VALIDATOR,
            "worker_report_filename": "prepare_repo_worker_report.json",
            "validator_report_filename": "prepare_repo_validator_report.json",
            "goal": "repository acquisition and benchmark surface discovery",
        },
        {
            "stage_id": "env",
            "executor_type": PREPARE_STEP_EXECUTOR,
            "worker_type": PREPARE_WORKER,
            "validator_type": PREPARE_VALIDATOR,
            "worker_report_filename": "prepare_env_worker_report.json",
            "validator_report_filename": "prepare_env_validator_report.json",
            "goal": "runnable environment setup at project/venv",
        },
        {
            "stage_id": "dataset",
            "executor_type": PREPARE_STEP_EXECUTOR,
            "worker_type": PREPARE_WORKER,
            "validator_type": PREPARE_VALIDATOR,
            "worker_report_filename": "prepare_dataset_worker_report.json",
            "validator_report_filename": "prepare_dataset_validator_report.json",
            "goal": "dataset staging under dataset_candidate and exact target declaration",
        },
        {
            "stage_id": "synthesis",
            "executor_type": PREPARE_STEP_EXECUTOR,
            "worker_type": PREPARE_WORKER,
            "validator_type": PREPARE_VALIDATOR,
            "worker_report_filename": "prepare_handoff_worker_report.json",
            "validator_report_filename": "prepare_validator_report.json",
            "goal": "validated prepare handoff synthesis and phase verdict",
        },
    ]
    _subagents_registered = False

    def __init__(
        self,
        model: str = PREPARE_AGENT_MODEL,
        verbose: bool = True,
        workspace_root: Optional[str] = None,
    ):
        super().__init__(
            agent_type="PrepareAgent",
            model=model,
            max_turns=PLANNER_MAX_TURNS,
            verbose=verbose,
            workspace_root=workspace_root,
            enable_condenser=True,
            condenser_max_size=150,
            condenser_keep_first=20,
        )
        self._ensure_subagents_registered()

    def _ensure_subagents_registered(self) -> None:
        if PrepareAgent._subagents_registered:
            return
        registrations = (
            (PREPARE_STEP_EXECUTOR, create_prepare_step_executor_agent, "Executes one prepare-stage worker/validator repair loop."),
        )
        for name, factory, description in registrations:
            try:
                register_agent(
                    name=name,
                    factory_func=factory,
                    description=description,
                )
            except ValueError:
                pass
        PrepareAgent._subagents_registered = True

    def _get_tools(self) -> List:
        return [
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
            Tool(name=TaskToolSet.name),
        ]

    def _get_agent_context(self):
        return get_prepare_agent_context()

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return """You are the prepare planner for an experiment workspace.

You own orchestration, not direct execution. Your job is to define precise stage contracts, dispatch them through `task`, and rely on the prepare validator for completion decisions.

Core responsibilities:
1. Read the idea input and current workspace state before planning.
2. Write a concise ordered prepare plan.
3. Run the stage sequence strictly as `repos -> env -> dataset -> validated handoff synthesis`.
4. For each stage, launch `prepare_step_executor`.
5. Do not advance a stage until the stage executor reports validator-backed PASS.
6. The stage executor, not you, owns the worker/validator repair loop for the current stage.
7. Synthesize human-readable handoff artifacts and the final prepare phase verdict only after the stage executors have completed successfully.

Hard rules:
- The worker must do the real work; do not allow placeholder completion.
- The validator is the authority for stage completion.
- Datasets are not prepared until they exist on the prepared handoff surface under `dataset_candidate/`.
- The final prepare output must describe the discovered real experiment targets from the actual prepared workspace state, not from hardcoded assumptions.
- `idea.json.components` is canonical. Preserve its component names and order exactly in all prepare handoff artifacts.
- Do not substitute synthetic benchmarks for real prepared targets unless the prepare contract explicitly allows it.
- Keep all prepare-generated plans, reports, and handoff documents under `agent_reports/`.
- Never print the completion token before re-reading the generated handoff artifacts and `prepare_idea.md`.
"""

    def _build_user_prompt(self, **kwargs) -> str:
        pb = PromptBuilder()
        stage_contract_fields = format_field_bullets(PREPARE_STAGE_CONTRACT_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        candidate_env_vars = kwargs.get("candidate_env_vars") or []
        env_var_display = (
            "\n".join(f"- {name}" for name in candidate_env_vars)
            if candidate_env_vars
            else "- (none detected)"
        )

        pb.add_header("Prepare Workspace Task", level=1)
        pb.add_key_value("experiment_id", str(kwargs.get("experiment_id") or ""))
        pb.add_key_value("idea_json_path", str(kwargs.get("idea_json_path") or ""))
        pb.add_key_value("workspace_dir", str(kwargs.get("workspace_dir") or ""))
        pb.add_key_value("project_dir", str(kwargs.get("project_dir") or ""))
        pb.add_key_value("repos_dir", str(kwargs.get("repos_dir") or ""))
        pb.add_key_value("dataset_dir", str(kwargs.get("dataset_dir") or ""))
        pb.add_key_value("results_dir", str(kwargs.get("results_dir") or ""))
        pb.add_key_value("agent_reports_dir", str(kwargs.get("reports_dir") or ""))
        pb.add_text("")

        pb.add_header("Inputs", level=2)
        summary = kwargs.get("idea_summary") or {}
        canonical_components = kwargs.get("canonical_components") or []
        pb.add_code(json.dumps(summary, ensure_ascii=False, indent=2), language="json")

        pb.add_header("Canonical Idea Components", level=2)
        if canonical_components:
            pb.add_text(format_canonical_components_markdown(canonical_components))
        else:
            pb.add_text("- (no components found)")

        pb.add_header("Mission", level=2)
        pb.add_list(
            [
                "Prepare a real execution surface for the experiment, not just a discovery summary.",
                "Ground all later phases in exact prepared targets: exact model ids, exact env vars, exact dataset files, and exact benchmark entrypoints.",
                "Carry the exact `idea.json.components` list forward into the handoff. Component names and order must not change.",
                "Use `task` with `prepare_step_executor` for every stage.",
                "Treat validator verdicts as binding.",
                "Keep code under `project_dir`, datasets under `dataset_dir`, and reserve `results_dir` for later experiment outputs.",
            ],
            ordered=False,
        )

        pb.add_header("Required Flow", level=2)
        pb.add_list(
            [
                "1. Inspect idea.json and current workspace state.",
                "2. Write `agent_reports/prepare_plan.json` with explicit stage contracts and a matching `prepare_planner_report.json`.",
                "3. Run repository stage via `prepare_step_executor`; it must manage the worker/validator loop internally.",
                "4. Run environment stage via `prepare_step_executor`; it must manage the worker/validator loop internally.",
                "5. Run dataset stage via `prepare_step_executor`; it must manage the worker/validator loop internally.",
                "6. Run a synthesis stage via `prepare_step_executor` to produce the phase-level prepare verdict.",
                "7. Re-open the generated outputs, self-check them against the validator evidence, then print the completion token.",
            ],
            ordered=False,
        )

        pb.add_header("Stage Contracts", level=2)
        pb.add_list(
            [
                "Repository stage must verify exact repos, benchmark code locations, and runnable entrypoints.",
                "Environment stage must create or validate `project/venv` and record actual imports and commands.",
                "Dataset stage must place final verified datasets under `dataset_candidate/` and reject repo-local-only handoff paths.",
                "Synthesis stage must only summarize validated facts and exact real experiment targets, then produce the phase-level prepare verdict.",
                "Synthesis stage must write `prepare_idea.md` with the exact canonical component list from `idea.json.components` in the same order.",
                "Each stage contract must define a flat `*_contract.json` path plus a flat `*_executor_report.json` path under `agent_reports/`.",
            ],
            ordered=False,
        )

        pb.add_header("Plan Contract Schema", level=2)
        pb.add_text("Each prepare stage contract must include the following fields:")
        pb.add_text(stage_contract_fields)
        pb.add_text(
            f"The synthesis stage must use `{kwargs.get('reports_dir')}/prepare_validator_report.json` as its phase-level validator verdict path."
        )

        pb.add_header("Path Contract", level=2)
        pb.add_list(
            [
                "All runnable project code must live under `project_dir`.",
                "Prepared datasets must live under `dataset_dir`.",
                "Experiment outputs are reserved for `results_dir`; prepare must not place benchmark outputs there.",
                "Planner, step executor, worker, validator, and handoff files must live under `agent_reports_dir`.",
                "Use flat filenames only. Do not create subdirectories under `agent_reports_dir`.",
            ],
            ordered=False,
        )

        pb.add_header("Worker And Validator Output Rules", level=2)
        pb.add_list(
            [
                "Each worker run must write the stage-specific worker report named in the stage contract.",
                "Each validator run must write the stage-specific validator report named in the stage contract.",
                "Each step executor run must write the stage-specific executor report named in the stage contract.",
                "The stage executor must keep the current stage active until validator-backed PASS, terminal blocker, or `max_repair_rounds` exhaustion.",
                "The final prepare validator report must represent the phase-level prepare verdict; later agents should not recompute prepare completeness from individual stage files.",
            ],
            ordered=False,
        )
        pb.add_text("Validator reports must use `status: PASS|PARTIAL|FAIL` and include the following shared verdict fields:")
        pb.add_text(verdict_fields)

        pb.add_header("Real Target Requirements", level=2)
        pb.add_list(
            [
                "Do not hardcode model, dataset, or benchmark names into the plan. Derive them from the workspace and repository evidence.",
                "When you describe targets in `prepare_idea.md`, use the exact names and paths you actually verified.",
                "If a target is still ambiguous, record the ambiguity explicitly instead of inventing a canonical name.",
                "When you describe components in `prepare_idea.md`, copy the exact component names and order from `idea.json.components`.",
            ],
            ordered=False,
        )

        pb.add_header("prepare_idea.md Requirements", level=2)
        pb.add_list(
            [
                "Use exact headings: `## Environment Variables Used`, `## Resource Acquisition Log`, `## Repository-to-Dataset Mapping`, `## Real Experiment Targets`.",
                f"Add `{IDEA_COMPONENTS_HEADING}` as a dedicated section after `## Real Experiment Targets`.",
                "Under that section, list every idea component exactly once, in the exact order from `idea.json.components`, including each explanation.",
                "Write this document to `agent_reports/prepare_idea.md`.",
                "Never print secret values; only print env var names and purposes.",
                "Do not describe any target as ready unless the matching validator evidence supports it.",
            ],
            ordered=False,
        )

        pb.add_header("Environment Variable Candidates", level=2)
        pb.add_text(env_var_display)

        pb.add_header("HuggingFace Rule", level=2)
        pb.add_list(
            [
                "Use the official HuggingFace domains for downloads.",
                "Inherit the current shell proxy environment instead of overriding it.",
                "Do not assume direct access or proxy access will always work; record the actual network mode that succeeded or failed.",
            ],
            ordered=False,
        )

        pb.add_header("Completion Token", level=2)
        pb.add_text("Finish by printing exactly: PREPARE COMPLETE")
        return pb.build()

    def _extract_urls_from_text(self, text: str) -> Dict[str, List[str]]:
        github_urls: List[str] = []
        dataset_urls: List[str] = []
        if not text:
            return {"repos": github_urls, "datasets": dataset_urls}
        github_pattern = r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
        hf_pattern = r"https?://huggingface\.co/datasets/[A-Za-z0-9_.\-/]+"
        for match in re.findall(github_pattern, text):
            clean = match.rstrip(").,;\"' ")
            if clean not in github_urls:
                github_urls.append(clean)
        for match in re.findall(hf_pattern, text):
            clean = match.rstrip(").,;\"' ")
            if clean not in dataset_urls:
                dataset_urls.append(clean)
        return {"repos": github_urls, "datasets": dataset_urls}

    def _detect_candidate_env_vars(self) -> List[str]:
        keywords = [
            "TOKEN",
            "KEY",
            "API",
            "ENDPOINT",
            "BASE_URL",
            "PROXY",
            "HF",
            "HUGGINGFACE",
            "GITHUB",
            "CACHE",
        ]
        names = [
            name
            for name in os.environ.keys()
            if any(keyword in name.upper() for keyword in keywords)
        ]
        return sorted(set(names))[:40]

    def _refresh_runtime_roots(self, workspace_dir: str) -> None:
        normalized = os.path.realpath(workspace_dir)
        if self.workspace_root == normalized:
            return
        self.workspace_root = normalized
        if "/workspace/" in normalized or normalized.endswith("/workspace"):
            self.persistence_dir = os.path.join(normalized, ".conversations")
        else:
            self.persistence_dir = None
        self.conversation_id = uuid4()
        self.resume = False

    def _write_runtime_debug_summary(self, workspace_dir: str, project_dir: str) -> None:
        mcp_config = self._build_mcp_config()
        servers = mcp_config.get("mcpServers") if isinstance(mcp_config, dict) else {}
        payload = {
            "workspace_root": os.path.realpath(workspace_dir),
            "project_root": os.path.realpath(project_dir),
            "persistence_dir": self.persistence_dir,
            "mcp_servers": {},
        }
        if isinstance(servers, dict):
            for name, server in servers.items():
                payload["mcp_servers"][name] = {
                    "command": str((server or {}).get("command") or ""),
                    "args": list((server or {}).get("args") or []),
                    "env_keys": sorted(str(key) for key in ((server or {}).get("env") or {}).keys()),
                }
        output_path = Path(artifact_paths(workspace_dir, project_dir)["prepare_runtime_debug"])
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_mcp_config(self) -> Dict[str, Any]:
        base_config = super()._build_mcp_config()
        servers = base_config.get("mcpServers") if isinstance(base_config, dict) else None
        if not isinstance(servers, dict):
            return {"mcpServers": {}}
        allowed_raw = os.environ.get("EXPERIMENT_AGENT_PREPARE_MCP_SERVERS", "")
        allowed = [item.strip() for item in allowed_raw.split(",") if item and item.strip()]
        if not allowed:
            allowed = list(self.PREPARE_DEFAULT_MCP_SERVERS)
        filtered_servers = {name: servers[name] for name in allowed if name in servers}
        if not filtered_servers:
            fallback = list(self.PREPARE_DEFAULT_MCP_SERVERS)
            filtered_servers = {name: servers[name] for name in fallback if name in servers}
        return {"mcpServers": filtered_servers}

    async def prepare_workspace(
        self,
        experiment_id: str,
        force: bool = False,
        clone_depth: int = 1,
        skip_repos: bool = False,
        skip_datasets: bool = False,
    ) -> PrepareReport:
        if not experiment_id:
            raise ValueError("experiment_id is required")

        paths = ensure_experiment_dirs(experiment_id)
        contract = workspace_contract_paths(
            normalize_workspace_path(str(paths.get("workspace_dir") or "")),
            normalize_workspace_path(str(paths.get("project_dir") or "")),
        )
        workspace_dir = contract["workspace_dir"]
        project_dir = contract["project_dir"]
        repos_dir = normalize_workspace_path(str(paths.get("repos_dir") or contract["repos_dir"]))
        dataset_dir = normalize_workspace_path(str(paths.get("dataset_dir") or contract["dataset_dir"]))
        results_dir = normalize_workspace_path(str(paths.get("results_dir") or contract["results_dir"]))
        reports_dir = normalize_workspace_path(str(paths.get("reports_dir") or contract["agent_reports_dir"]))
        self._refresh_runtime_roots(workspace_dir)

        idea_json_candidates = [
            os.path.join(workspace_dir, "idea.json"),
            os.path.join(workspace_dir, "idea_result.json"),
        ]
        idea_json_path = next((candidate for candidate in idea_json_candidates if os.path.exists(candidate)), None)
        if idea_json_path is None:
            raise FileNotFoundError(f"idea.json not found in {workspace_dir}. Tried: {idea_json_candidates}")
        idea_md_path = artifact_paths(workspace_dir, project_dir)["idea"]

        SecurityContext.set_roots(
            project_root=os.path.realpath(project_dir),
            workspace_root=os.path.realpath(workspace_dir),
        )
        self._write_runtime_debug_summary(workspace_dir=workspace_dir, project_dir=project_dir)

        with open(idea_json_path, "r", encoding="utf-8") as f:
            raw_json_text = f.read()
        try:
            data = json.loads(raw_json_text)
        except Exception:
            data = {"_raw_text": raw_json_text}
        canonical_components = load_canonical_components(
            workspace_dir, idea_json_path=idea_json_path
        )

        summary: Dict[str, Any] = {
            "idea_json_content": data,
            "raw_idea_json_text": raw_json_text,
            "auto_discovered_candidates": self._extract_urls_from_text(raw_json_text),
            "canonical_components": canonical_components,
        }

        await self._run_agent(
            user_prompt=self._build_user_prompt(
                experiment_id=experiment_id,
                idea_json_path=normalize_workspace_path(idea_json_path),
                workspace_dir=normalize_workspace_path(workspace_dir),
                project_dir=normalize_workspace_path(project_dir),
                repos_dir=normalize_workspace_path(repos_dir),
                dataset_dir=normalize_workspace_path(dataset_dir),
                results_dir=normalize_workspace_path(results_dir),
                reports_dir=normalize_workspace_path(reports_dir),
                idea_summary=summary,
                canonical_components=canonical_components,
                force=bool(force),
                clone_depth=int(clone_depth),
                skip_repos=bool(skip_repos),
                skip_datasets=bool(skip_datasets),
                candidate_env_vars=self._detect_candidate_env_vars(),
            ),
            system_prompt=self._build_system_prompt(),
            tools=self._get_tools(),
            project_root=os.path.realpath(project_dir),
            purpose="prepare_workspace",
        )

        return PrepareReport(
            experiment_id=experiment_id,
            workspace_dir=os.path.realpath(workspace_dir),
            project_dir=os.path.realpath(project_dir),
            repos_dir=os.path.realpath(repos_dir),
            dataset_dir=os.path.realpath(dataset_dir),
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
        model=model or PREPARE_AGENT_MODEL,
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
