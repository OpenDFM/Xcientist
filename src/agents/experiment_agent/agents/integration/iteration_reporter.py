"""
Iteration integration agent - summarizes experiment status after each master iteration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.agent import OpenHandsBaseAgent
from src.agents.experiment_agent.config import (
    get_master_agent_model,
    get_planner_max_turns,
)
from src.agents.experiment_agent.runtime.idea_components import find_idea_json_path
from src.agents.experiment_agent.runtime.manifests import artifact_paths, workspace_contract_paths
from src.agents.experiment_agent.skills import get_worker_agent_context


ITERATION_REPORTER = "experiment_iteration_reporter"


class IterationReporterAgent(OpenHandsBaseAgent):
    """Summarizes experiment status after each master iteration."""

    REPORTING_DEFAULT_MCP_SERVERS: list[str] = []
    SYSTEM_PROMPT_TEMPLATE = "iteration_reporter.j2"

    def __init__(
        self,
        workspace_root: str,
        project_root: str,
        model: str | None = None,
        verbose: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            agent_type="IterationReporter",
            model=model or get_master_agent_model(),
            max_turns=get_planner_max_turns(),
            verbose=verbose,
            workspace_root=workspace_root,
            enable_condenser=True,
            condenser_max_size=300,
            condenser_keep_first=50,
            resume=resume,
        )
        self.workspace_root = workspace_root
        self.project_root = project_root
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.idea_json_path = find_idea_json_path(workspace_root) or self.paths["idea_json"]
        self.iteration_summary_path = self.paths["iteration_summary"]
        self.iteration_status_path = self.paths["iteration_status"]

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return ""

    def _get_tools(self):
        return [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]

    def _get_agent_context(self):
        return get_worker_agent_context("iteration_reporter")

    def _build_mcp_config(self) -> Dict[str, Any]:
        base_config = super()._build_mcp_config()
        servers = base_config.get("mcpServers") if isinstance(base_config, dict) else None
        if not isinstance(servers, dict):
            return {"mcpServers": {}}
        filtered_servers = {
            name: servers[name]
            for name in self.REPORTING_DEFAULT_MCP_SERVERS
            if name in servers
        }
        return {"mcpServers": filtered_servers}

    def _build_user_prompt(self) -> str:
        return f"""Summarize the current experiment iteration status.

Input paths:
- idea_json_path: {self.idea_json_path}
- agent_reports_dir: {self.contract['agent_reports_dir']}
- model_dir: {self.contract['model_dir']}
- results_dir: {self.contract['results_dir']}
- standard_results_dir: {self.contract['standard_results_dir']}
- ablation_results_dir: {self.contract['ablation_results_dir']}
- master_report: {self.paths['master_report']}

Output paths (BOTH must be written to agent_reports/):
- iteration_summary_md: {self.iteration_summary_path}
- iteration_status_json: {self.iteration_status_path}

CRITICAL:
1. Read `iteration_status.json` when present, then validator JSON, then only the targeted raw result windows needed to support findings.
2. Write iteration_summary.md with human-readable summary
3. Write iteration_status.json with machine-readable status
4. The master agent will read iteration_status.json for the next iteration decision
5. Explicitly remind master to read iteration_status.json in your output
6. Do not suggest manual file moves/copies for `ablation_results.json`; the final ablation report agent owns that artifact after the master loop converges.
7. Do not invent extra workflow phases such as `FINAL_SYNTHESIS`; use only the current workflow phases and, when appropriate, recommend final ablation reporting after all code/science phases complete.

iteration_status.json MUST have this EXACT schema:
{{
  "iteration": <number>,
  "code_status": "complete|incomplete|not_started",
  "code_evidence": ["list of evidence files"],
  "standard_experiments": "none|partial|complete",
  "standard_evidence": ["list of evidence files"],
  "ablation_experiments": "none|partial|complete",
  "ablation_evidence": ["list of evidence files"],
  "validation_status": "pass|fail|partial|unknown",
  "phase_states": {{
    "code": {{
      "phase_completion_status": "not_started|partial|complete|blocked",
      "ready_for_next_phase": true|false,
      "artifact_role": "phase_result",
      "run_level": "smoke|full|mixed",
      "blocking_issues": ["..."]
    }},
    "standard_science": {{
      "phase_completion_status": "not_started|partial|complete|blocked",
      "ready_for_next_phase": true|false,
      "artifact_role": "phase_result",
      "run_level": "smoke|full|mixed",
      "blocking_issues": ["..."]
    }},
    "ablation_science": {{
      "phase_completion_status": "not_started|partial|complete|blocked",
      "ready_for_next_phase": true|false,
      "artifact_role": "phase_result|final_result",
      "run_level": "smoke|full|mixed",
      "blocking_issues": ["..."]
    }}
  }},
  "key_findings": ["finding1", "finding2"],
  "blockers": ["blocker1"] or [],
  "next_recommendations": ["recommendation1", "recommendation2"]
}}

Evidence rules:
- Treat only validator-backed artifacts with `artifact_role=phase_result|final_result` as formal phase evidence.
- Treat `smoke_check` artifacts as implementation-readiness evidence only; they must not upgrade a science phase to `complete`.
- When a phase validator is missing, do not infer completion from raw outputs alone.

After writing both files, explicitly state: "Master agent should read {self.iteration_status_path} for the next iteration decision."
"""

    def _artifact_valid(self) -> bool:
        if not os.path.exists(self.iteration_status_path):
            return False
        try:
            payload = json.load(open(self.iteration_status_path, "r", encoding="utf-8"))
        except Exception:
            return False
        required_keys = {
            "iteration", "code_status", "code_evidence",
            "standard_experiments", "standard_evidence",
            "ablation_experiments", "ablation_evidence",
            "validation_status", "key_findings", "blockers", "next_recommendations"
        }
        if not isinstance(payload, dict) or not all(k in payload for k in required_keys):
            return False
        phase_states = payload.get("phase_states")
        if phase_states is not None:
            if not isinstance(phase_states, dict):
                return False
            for phase_name in ("code", "standard_science", "ablation_science"):
                phase_payload = phase_states.get(phase_name)
                if not isinstance(phase_payload, dict):
                    return False
                required_phase_keys = {
                    "phase_completion_status",
                    "ready_for_next_phase",
                    "artifact_role",
                    "run_level",
                    "blocking_issues",
                }
                if not required_phase_keys.issubset(set(phase_payload.keys())):
                    return False
        return True

    async def execute(self) -> Dict[str, Any]:
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            system_prompt=self._build_system_prompt(),
        )
        return {
            "output": self._extract_output(result),
            "iteration_summary_path": self.iteration_summary_path,
            "iteration_status_path": self.iteration_status_path,
            "valid": self._artifact_valid(),
        }


async def run_iteration_reporter(
    workspace_root: str,
    project_root: str,
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = IterationReporterAgent(
        workspace_root=workspace_root,
        project_root=project_root,
        model=model or get_master_agent_model(),
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
