"""
Code planner for experiment enablement.
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
from src.agents.experiment_agent.agents.code.step_executor import (
    CODE_STEP_EXECUTOR,
    create_code_step_executor_agent,
)
from src.agents.experiment_agent.config import CODE_AGENT_MODEL, PLANNER_MAX_TURNS
from src.agents.experiment_agent.runtime.contracts import (
    CODE_STEP_CONTRACT_FIELDS,
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
    format_named_paths,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, workspace_contract_paths
from src.agents.experiment_agent.skills import get_code_agent_context


EXPERIMENT_CODE_PLANNER = "experiment_code_planner"
_CODE_SUBAGENTS_REGISTERED = False
_CODE_PLANNER_REGISTERED = False


def _planner_tools() -> List[Tool]:
    return [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=TaskToolSet.name),
    ]


def _code_planner_system_prompt() -> str:
    return """You are the code planner for experiment enablement.

You own code orchestration, not benchmark validation. Define concrete implementation steps, dispatch them through `task`, and rely on `code_validator` for PASS/FAIL decisions.

Core behavior:
1. Read `prepare_idea.md`, `idea.json`, and the validated prepare handoff artifacts before planning.
2. Write a short ordered `code_plan.json`.
3. Every step must map to a real prepared target or experiment path discovered from prior phase artifacts.
4. For each step, run `code_step_executor`.
5. The step executor, not you, owns the worker/validator repair loop for that step.
6. The final required step must be `final_integration_smoke`.
7. Only after validator-backed PASS should you mark a step done.
8. Only after all steps pass should you write the final code handoff artifacts.

Hard rules:
- Do not treat import success as sufficient when the contract requires real integration.
- Do not write science-owned artifacts.
- Do not accept synthetic placeholders when the target is a real prepared benchmark path.
- Treat `idea.json.components` as canonical for any component-scoped enablement or blocker notes.
- The validator report is the source of truth for step completion.
"""


def create_experiment_code_planner_agent(llm) -> Agent:
    _register_code_subagents()
    from openhands.sdk.context import AgentContext
    code_context = get_code_agent_context()
    return Agent(
        llm=llm,
        tools=_planner_tools(),
        agent_context=AgentContext(
            skills=code_context.skills,
            system_message_suffix=_code_planner_system_prompt(),
            load_public_skills=False,
        ),
    )


def _register_code_subagents() -> None:
    global _CODE_SUBAGENTS_REGISTERED
    if _CODE_SUBAGENTS_REGISTERED:
        return
    registrations = (
        (CODE_STEP_EXECUTOR, create_code_step_executor_agent, "Executes one code step through the worker/validator repair loop."),
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
    _CODE_SUBAGENTS_REGISTERED = True


def register_experiment_code_planner() -> None:
    global _CODE_PLANNER_REGISTERED
    if _CODE_PLANNER_REGISTERED:
        return
    try:
        register_agent(
            name=EXPERIMENT_CODE_PLANNER,
            factory_func=create_experiment_code_planner_agent,
            description="Plans code implementation steps and coordinates code step executors.",
        )
    except ValueError:
        pass
    _CODE_PLANNER_REGISTERED = True


class CodeAgent(OpenHandsBaseAgent):
    CODE_DEFAULT_MCP_SERVERS = ["filesystem"]

    def __init__(
        self,
        experiment_id: str,
        idea_path: str,
        project_root: str,
        workspace_root: str,
        plan: str,
        model: str = CODE_AGENT_MODEL,
        verbose: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            agent_type="Code",
            model=model,
            max_turns=PLANNER_MAX_TURNS,
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
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.summary_path = self.paths["code_summary"]
        self.usage_path = self.paths["code_usage"]
        self.plan_path = self.paths["code_plan"]
        self.validator_report_path = self.paths["code_validator"]
        register_experiment_code_planner()

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return _code_planner_system_prompt()

    def _get_tools(self):
        return _planner_tools()

    def _get_agent_context(self):
        return get_code_agent_context()

    def _get_relevant_env_var_names(self) -> List[str]:
        keywords = ("KEY", "TOKEN", "SECRET", "API", "BASE_URL", "ENDPOINT")
        names = [name for name in os.environ if any(k in name.upper() for k in keywords)]
        return sorted(names)[:50]

    def _build_mcp_config(self) -> Dict[str, Any]:
        base_config = super()._build_mcp_config()
        servers = base_config.get("mcpServers") if isinstance(base_config, dict) else None
        if not isinstance(servers, dict):
            return {"mcpServers": {}}
        allowed_raw = os.environ.get("EXPERIMENT_AGENT_CODE_MCP_SERVERS", "")
        allowed = [item.strip() for item in allowed_raw.split(",") if item and item.strip()]
        if not allowed:
            allowed = list(self.CODE_DEFAULT_MCP_SERVERS)
        filtered_servers = {name: servers[name] for name in allowed if name in servers}
        if not filtered_servers:
            fallback = list(self.CODE_DEFAULT_MCP_SERVERS)
            filtered_servers = {name: servers[name] for name in fallback if name in servers}
        return {"mcpServers": filtered_servers}

    def _build_user_prompt(self, **kwargs) -> str:
        _ = kwargs
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
                "results_dir": self.contract["results_dir"],
                "agent_reports_dir": self.contract["agent_reports_dir"],
            }
        )
        step_fields = format_field_bullets(CODE_STEP_CONTRACT_FIELDS)
        verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
        return f"""## Task: Enable Experiment Code Paths

### Master Plan
{self.plan}

### Input Paths
{input_paths}

### Path Contract
{path_contract}

### Required Flow
1. Read the idea and validated prepare artifacts first.
2. Write `{self.plan_path}` as an ordered step list. Each step must include:
{step_fields}
3. The plan must end with a mandatory step whose `step_id` is exactly `final_integration_smoke`.
4. Every step must tie to a real prepared target or benchmark path discovered from the prepare artifacts above.
5. Every step must use a unique flat `worker_report_path`, `validator_report_path`, `step_contract_path`, and `executor_report_path` under `agent_reports_dir`; do not reuse a phase-global path for multiple steps.
6. Use filenames that stay flat under `agent_reports_dir`, for example `code_step_01_<slug>_contract.json`, `code_step_01_<slug>_executor_report.json`, and `code_step_01_<slug>_attempt_01_worker_report.json`.
7. The `final_integration_smoke` step must:
   - use the real prepared dataset path from `dataset_dir`
   - use the real API/model path if the method depends on one
   - run the actual integrated command path that science will later rely on
   - save bounded raw smoke artifacts under flat filenames in `agent_reports_dir`
8. For each step, use `task` with `subagent_type="{CODE_STEP_EXECUTOR}"`.
9. The step executor must manage the internal `code_worker` -> `code_validator` repair loop.
10. Do not move to the next step until the step executor reports validator-backed PASS.
11. Do not manually paraphrase validator fixes back to the worker at the planner level.
12. After all steps pass, write:
   - `{self.paths["code_planner_report"]}`
   - `{self.summary_path}`
   - optional `{self.usage_path}`
   - `{self.paths["code_integration_readiness"]}`
   - `{self.paths["code_worker"]}`
   - `{self.paths["code_validator"]}`
13. The top-level code worker/validator reports are phase summaries only; debugging details must remain in the per-step flat reports under `agent_reports_dir`.
14. The final validator report must use `status: PASS|FAIL` and include:
{verdict_fields}

### Hard Rules
- The validator decides whether a step is complete.
- Do not hardcode model, dataset, benchmark, or API names into the plan. Infer them from the workspace artifacts you read.
- When a step concerns ablation support, use the exact component names from the validated idea handoff. Do not rename, merge, split, omit, or reorder them.
- All code edits and runnable entrypoints must live under `project_dir`.
- Do not place experiment outputs under `results_dir` during the code phase.
- Do not write any formal science result artifact or lane summary under `results/`. In particular, the code phase must not materialize `ablation_results.json`.
- Do not collapse all steps into a single shared `code_worker` or `code_validator` path.
- Do not call a target enabled unless it is materially runnable from the prepared workspace.
- The code phase cannot pass without validator-backed success for `final_integration_smoke`.
- Candidate env vars: {", ".join(self._get_relevant_env_var_names()) if self._get_relevant_env_var_names() else "(none detected)"}.

Finish by printing exactly: CODE ENABLEMENT COMPLETE"""

    def _validator_passed(self) -> bool:
        if not os.path.exists(self.validator_report_path):
            return False
        try:
            with open(self.validator_report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False
        return str(payload.get("status") or "").strip().upper() == "PASS"

    async def execute(self) -> Dict[str, Any]:
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            system_prompt=self._build_system_prompt(),
        )
        output = self._extract_output(result)
        summary_content = self._read_text_file(self.summary_path).strip() or output or ""
        usage_content = self._read_text_file(self.usage_path).strip()
        status = "completed" if self._validator_passed() else "insufficient"
        return {
            "summary": summary_content,
            "usage": usage_content or summary_content,
            "summary_path": self.summary_path,
            "usage_path": self.usage_path if os.path.exists(self.usage_path) else None,
            "status": status,
        }


async def run_code_agent(
    experiment_id: str,
    idea_path: str,
    project_root: str,
    workspace_root: str,
    plan: str,
    model: str = CODE_AGENT_MODEL,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = CodeAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        project_root=project_root,
        workspace_root=workspace_root,
        plan=plan,
        model=model,
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
