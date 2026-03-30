"""
Science planners built on TaskToolSet.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openhands.sdk import Agent
from openhands.sdk.subagent import register_agent
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.agent import OpenHandsBaseAgent
from src.agents.experiment_agent.agents.base.subagents import (
    default_builtin_tool_names,
)
from src.agents.experiment_agent.agents.science.step_executor import (
    ABLATION_SCIENCE_STEP_EXECUTOR,
    STANDARD_SCIENCE_STEP_EXECUTOR,
    create_ablation_science_step_executor_agent,
    create_standard_science_step_executor_agent,
)
from src.agents.experiment_agent.config import (
    get_planner_max_turns,
    get_science_agent_model,
)
from src.agents.experiment_agent.runtime.contracts import (
    ABLATION_COMPONENT_RESULT_FIELDS,
    PHASE_VERDICT_FIELDS,
    SCIENCE_ABLATION_STEP_FIELDS,
    SCIENCE_STANDARD_STEP_FIELDS,
    format_field_bullets,
    format_named_paths,
)
from src.agents.experiment_agent.runtime.idea_components import (
    format_canonical_components_markdown,
    load_canonical_components,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, workspace_contract_paths
from src.agents.experiment_agent.skills import get_exp_agent_context


EXPERIMENT_STANDARD_SCIENCE_PLANNER = "experiment_standard_science_planner"
EXPERIMENT_ABLATION_SCIENCE_PLANNER = "experiment_ablation_science_planner"
STANDARD_SCIENCE_VALIDATOR_REPORT = "standard_science_validator_report.json"
ABLATION_SCIENCE_VALIDATOR_REPORT = "ablation_science_validator_report.json"
_SCIENCE_SUBAGENTS_REGISTERED = False
_SCIENCE_PLANNERS_REGISTERED = False


def _planner_tools() -> List[Tool]:
    return [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=TaskToolSet.name),
    ]


# Templates are now used directly via SYSTEM_PROMPT_TEMPLATE class attribute
# StandardScienceAgent uses "standard_science_planner_agent.j2"
# AblationScienceAgent uses "ablation_science_planner_agent.j2"


def _register_science_subagents() -> None:
    global _SCIENCE_SUBAGENTS_REGISTERED
    if _SCIENCE_SUBAGENTS_REGISTERED:
        return
    registrations = (
        (
            STANDARD_SCIENCE_STEP_EXECUTOR,
            create_standard_science_step_executor_agent,
            "Executes one standard-science step through the worker/validator repair loop.",
        ),
        (
            ABLATION_SCIENCE_STEP_EXECUTOR,
            create_ablation_science_step_executor_agent,
            "Executes one ablation-science step through the worker/validator repair loop.",
        ),
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
    _SCIENCE_SUBAGENTS_REGISTERED = True


def create_standard_science_planner_agent(llm) -> Agent:
    _register_science_subagents()
    from openhands.sdk.context import AgentContext
    exp_context = get_exp_agent_context()
    return Agent(
        llm=llm,
        tools=_planner_tools(),
        agent_context=AgentContext(
            skills=exp_context.skills,
            load_public_skills=False,
        ),
        include_default_tools=default_builtin_tool_names(),
    )


def create_ablation_science_planner_agent(llm) -> Agent:
    _register_science_subagents()
    from openhands.sdk.context import AgentContext
    exp_context = get_exp_agent_context()
    return Agent(
        llm=llm,
        tools=_planner_tools(),
        agent_context=AgentContext(
            skills=exp_context.skills,
            load_public_skills=False,
        ),
        include_default_tools=default_builtin_tool_names(),
    )


def register_science_planners() -> None:
    global _SCIENCE_PLANNERS_REGISTERED
    _register_science_subagents()
    if _SCIENCE_PLANNERS_REGISTERED:
        return
    registrations = (
        (
            EXPERIMENT_STANDARD_SCIENCE_PLANNER,
            create_standard_science_planner_agent,
            "Plans standard benchmark execution with science step executors.",
        ),
        (
            EXPERIMENT_ABLATION_SCIENCE_PLANNER,
            create_ablation_science_planner_agent,
            "Plans component-level ablation execution with science step executors.",
        ),
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
    _SCIENCE_PLANNERS_REGISTERED = True


class _BaseSciencePlanner(OpenHandsBaseAgent):
    SCIENCE_DEFAULT_MCP_SERVERS: List[str] = []
    # Templates are set in subclasses: StandardScienceAgent and AblationScienceAgent
    SYSTEM_PROMPT_TEMPLATE = None
    planner_type = "Science"
    plan_filename = "science_plan.json"
    completion_token = "SCIENCE COMPLETE"
    validator_report_filename = ""
    summary_key = ""

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
            agent_type=self.planner_type,
            model=model or get_science_agent_model(),
            max_turns=get_planner_max_turns(),
            verbose=verbose,
            workspace_root=workspace_root,
            enable_condenser=True,
            condenser_max_size=250,
            condenser_keep_first=40,
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
        self.idea_json_path = self.paths["idea_json"]
        self.canonical_components = load_canonical_components(workspace_root)
        self.report_path = self.paths[self.summary_key]
        if self.plan_filename == "standard_science_plan.json":
            self.plan_path = self.paths["standard_science_plan"]
            self.validator_report_path = self.paths["standard_science_validator"]
        else:
            self.plan_path = self.paths["ablation_science_plan"]
            self.validator_report_path = self.paths["ablation_science_validator"]
        register_science_planners()

    def _read_optional_text(self, *parts: str) -> str:
        path = os.path.join(self.workspace_root, *parts)
        return self._read_text_file(path).strip()

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return ""

    def _get_tools(self):
        return _planner_tools()

    def _get_agent_context(self):
        return get_exp_agent_context()

    def _build_mcp_config(self) -> Dict[str, Any]:
        base_config = super()._build_mcp_config()
        servers = base_config.get("mcpServers") if isinstance(base_config, dict) else None
        if not isinstance(servers, dict):
            return {"mcpServers": {}}
        allowed_raw = os.environ.get("EXPERIMENT_AGENT_SCIENCE_MCP_SERVERS", "")
        allowed = [item.strip() for item in allowed_raw.split(",") if item and item.strip()]
        if not allowed:
            allowed = list(self.SCIENCE_DEFAULT_MCP_SERVERS)
        filtered_servers = {name: servers[name] for name in allowed if name in servers}
        if not filtered_servers:
            fallback = list(self.SCIENCE_DEFAULT_MCP_SERVERS)
            filtered_servers = {name: servers[name] for name in fallback if name in servers}
        return {"mcpServers": filtered_servers}

    def _validator_passed(self) -> bool:
        if not os.path.exists(self.validator_report_path):
            return False
        try:
            with open(self.validator_report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False
        return str(payload.get("status") or "").strip().upper() == "PASS"

    def _required_artifacts_exist(self) -> bool:
        raise NotImplementedError

    def _build_user_prompt(self) -> str:
        raise NotImplementedError

    async def execute(self) -> Dict[str, Any]:
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            system_prompt=self._build_system_prompt(),
        )
        output = self._extract_output(result)
        report_content = self._read_text_file(self.report_path).strip() or output or ""
        status = "completed" if self._validator_passed() else "insufficient"
        return {
            "report": report_content,
            "report_path": self.report_path,
            "status": status,
        }


class StandardScienceAgent(_BaseSciencePlanner):
    SYSTEM_PROMPT_TEMPLATE = "standard_science_planner_agent.j2"
    planner_type = "StandardScience"
    plan_filename = "standard_science_plan.json"
    completion_token = "STANDARD SCIENCE COMPLETE"
    validator_report_filename = STANDARD_SCIENCE_VALIDATOR_REPORT
    summary_key = "standard_summary"

    def _build_user_prompt(self) -> str:
        input_paths = format_named_paths(
            {
                "idea_path": self.idea_path,
                "idea_json_path": self.idea_json_path,
                "code_summary_path": self.paths["code_summary"],
                "code_usage_path": self.paths["code_usage"],
                "prepare_plan_path": self.paths["prepare_plan"],
                "prepare_phase_validator_report_path": self.paths["prepare_validator"],
                "code_worker_report_path": self.paths["code_worker"],
                "code_validator_report_path": self.paths["code_validator"],
                "standard_summary_path": self.report_path,
                "standard_validator_report_path": self.validator_report_path,
            }
        )
        path_contract = format_named_paths(
            {
                "workspace_dir": self.contract["workspace_dir"],
                "project_dir": self.contract["project_dir"],
                "results_dir": self.contract["results_dir"],
                "standard_results_dir": self.contract["standard_results_dir"],
                "agent_reports_dir": self.contract["agent_reports_dir"],
            }
        )
        step_fields = format_field_bullets(SCIENCE_STANDARD_STEP_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        return f"""## Task: Run Standard Science

### Master Plan
{self.plan}

### Input Paths
{input_paths}

### Path Contract
{path_contract}

### Required Flow
1. Read the validated prepare artifacts and code handoff before planning.
2. Write `{self.plan_path}` as an ordered list of concrete standard experiment steps.
3. Every step must include:
{step_fields}
4. Derive target names and paths from the workspace artifacts above instead of hardcoding them.
5. Every step must use a unique flat `worker_report_path`, `validator_report_path`, `step_contract_path`, and `executor_report_path` under `agent_reports_dir`.
6. Use filenames that stay flat under `agent_reports_dir`, for example `standard_science_step_01_<slug>_contract.json`, `standard_science_step_01_<slug>_executor_report.json`, and `standard_science_step_01_<slug>_attempt_01_worker_report.json`.
7. For each step, call `task` with `subagent_type="{STANDARD_SCIENCE_STEP_EXECUTOR}"`.
8. The runtime uses the standard step executor to keep the `standard_science_worker` -> `standard_science_validator` loop active for the current step.
9. The validator must write `{self.validator_report_path}` as the phase-level standard verdict after the step-local reports exist.
10. Do not move to the next step until the step executor reports validator-backed PASS.
11. Update `{self.report_path}` only as a human-readable summary of validator-backed evidence.
12. Do not write downstream structured final result artifacts yourself. In particular, do not materialize `ablation_results.json`; that file belongs to the later final-artifact materialization step.
13. The final validator report must use `status: PASS|FAIL`, set a generic `phase_completion_status`, and include:
{verdict_fields}

### Hard Rules
- Final evidence must come from real execution on validated prepared targets.
- Every experiment command must write its raw outputs under `standard_results_dir`.
- Do not claim a run is `final/full` unless the assigned command chain actually ran.
- Do not use synthetic stress fallback as a silent replacement for formal prepared targets.
- Do not ask the runtime to infer coverage from your summaries.
- The validator is the authority for PASS/FAIL.

Finish by printing exactly: {self.completion_token}"""

    def _required_artifacts_exist(self) -> bool:
        return True


class AblationScienceAgent(_BaseSciencePlanner):
    SYSTEM_PROMPT_TEMPLATE = "ablation_science_planner_agent.j2"
    planner_type = "AblationScience"
    plan_filename = "ablation_science_plan.json"
    completion_token = "ABLATION SCIENCE COMPLETE"
    validator_report_filename = ABLATION_SCIENCE_VALIDATOR_REPORT
    summary_key = "ablation_summary"

    def _build_user_prompt(self) -> str:
        input_paths = format_named_paths(
            {
                "idea_path": self.idea_path,
                "idea_json_path": self.idea_json_path,
                "code_summary_path": self.paths["code_summary"],
                "code_usage_path": self.paths["code_usage"],
                "prepare_plan_path": self.paths["prepare_plan"],
                "prepare_phase_validator_report_path": self.paths["prepare_validator"],
                "code_worker_report_path": self.paths["code_worker"],
                "code_validator_report_path": self.paths["code_validator"],
                "standard_science_validator_report_path": self.paths["standard_science_validator"],
                "standard_summary_path": self.paths["standard_summary"],
                "ablation_summary_path": self.report_path,
                "ablation_validator_report_path": self.validator_report_path,
            }
        )
        path_contract = format_named_paths(
            {
                "workspace_dir": self.contract["workspace_dir"],
                "project_dir": self.contract["project_dir"],
                "results_dir": self.contract["results_dir"],
                "ablation_results_dir": self.contract["ablation_results_dir"],
                "agent_reports_dir": self.contract["agent_reports_dir"],
            }
        )
        step_fields = format_field_bullets(SCIENCE_ABLATION_STEP_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        ablation_result_fields = format_field_bullets(ABLATION_COMPONENT_RESULT_FIELDS)
        canonical_components = format_canonical_components_markdown(self.canonical_components)
        return f"""## Task: Run Ablation Science

### Master Plan
{self.plan}

### Input Paths
{input_paths}

### Path Contract
{path_contract}

### Canonical Idea Components
{canonical_components}

### Required Flow
1. Read the validated prepare artifacts, the code handoff, the standard science evidence, and `idea.json.components` before planning.
2. Write `{self.plan_path}` as a complete ordered list of component-level ablation steps covering the full canonical component set, not just the currently missing gap.
3. Each step must include:
{step_fields}
4. The number of ablation steps must equal the number of canonical idea components.
5. Each step's `component_or_condition` must equal the exact component name from `idea.json.components`, in the same order.
6. Each step's `component_explanation` must carry the matching explanation from `idea.json.components`.
7. Every step must use a unique flat `worker_report_path`, `validator_report_path`, `step_contract_path`, and `executor_report_path` under `agent_reports_dir`.
8. Use filenames that stay flat under `agent_reports_dir`, for example `ablation_science_step_01_<component>_contract.json`, `ablation_science_step_01_<component>_executor_report.json`, and `ablation_science_step_01_<component>_attempt_01_worker_report.json`.
9. For each step, call `task` with `subagent_type="{ABLATION_SCIENCE_STEP_EXECUTOR}"`.
10. The runtime uses the ablation step executor to keep the `ablation_science_worker` -> `ablation_science_validator` loop active for the current step.
11. The validator must write `{self.validator_report_path}` as the phase-level ablation verdict after the step-local reports exist.
12. Do not move to the next step until the step executor reports validator-backed PASS.
13. Every step-level ablation validator report must include:
{ablation_result_fields}
14. Update `{self.report_path}` only as a human-readable summary of validator-backed evidence.
15. Do not write `ablation_results.json` yourself. A later final-artifact materialization step owns that file.
16. The final phase-level validator report must use `status: PASS|FAIL`, set a generic `phase_completion_status`, and include:
{verdict_fields}

### Hard Rules
- Even if the master review highlights one missing area such as stress testing, you must still produce a full canonical ablation plan whose step list exactly matches all components from `idea.json.components`.
- Do not collapse the plan to only one missing experiment unless `idea.json.components` itself contains only one component.
- Do not mark an ablation complete without serious evidence and explicit method context.
- Do not invent ablation verdicts from expectation or narrative.
- Every experiment command must write its raw outputs under `ablation_results_dir`.
- Do not rename, merge, split, omit, or reorder canonical idea components.
- Do not hardcode alternative component names into the plan. Use the exact component names from `idea.json.components`.
- The validator is the authority for PASS/FAIL.
- Preserve enough step-level evidence for the later final-artifact materialization step to produce the final canonical `ablation_results.json`.

Finish by printing exactly: {self.completion_token}"""

    def _required_artifacts_exist(self) -> bool:
        return True

    async def execute(self) -> Dict[str, Any]:
        return await super().execute()


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
        model=model or get_science_agent_model(),
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
        model=model or get_science_agent_model(),
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
