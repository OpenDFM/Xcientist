"""
Top-level ablation report integrator runner.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.agent import OpenHandsBaseAgent
from src.agents.experiment_agent.config import (
    get_master_agent_model,
    get_planner_max_turns,
)
from src.agents.experiment_agent.runtime.idea_components import (
    canonical_component_names,
    find_idea_json_path,
)
from src.agents.experiment_agent.runtime.ablation_results import (
    write_ablation_results_artifacts,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, workspace_contract_paths
from src.agents.experiment_agent.skills import get_worker_agent_context


class AblationReportIntegratorAgent(OpenHandsBaseAgent):
    REPORTING_DEFAULT_MCP_SERVERS: List[str] = []

    def __init__(
        self,
        workspace_root: str,
        project_root: str,
        model: str | None = None,
        verbose: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            agent_type="AblationReportIntegrator",
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
        self.ablation_json_path = self.paths["ablation_results"]
        self.integrator_report_path = self.paths["ablation_report_integrator_report"]
        self.canonical_component_names = canonical_component_names(workspace_root)

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return """You are the final ablation report integrator.

Your only job is to write the final `ablation_results.json` from `idea.json` and the completed experiment evidence.

Hard rules:
1. Ignore any pre-existing `ablation_results.json`. Treat it as stale and do not use it as evidence.
2. Treat `idea.json.components` as the only canonical source for component names, count, and order.
3. Prefer validator-backed JSON, structured evidence refs, and targeted result windows from `agent_reports/` and `results/ablation/`.
4. Do not rely on file existence alone. Use `read_json`, `search`, and bounded `view` calls instead of broad file dumps.
5. Write `ablation_results.json` with exactly two top-level keys and no extras:
   - `components`
   - `summary`
6. `components` must be keyed by the exact canonical component names from `idea.json.components`, in the same order, with no extras and no omissions.
7. Every component entry must contain exactly:
   - `result`
   - `metric`
   - `value`
   - `confidence`
   - `analysis`
   - `method_context` - copy verbatim from `idea.json.components[<index>].description` for that component (the idea's original description of what this component does)
8. `summary` must contain:
   - `feasible`
   - `confidence`
   - `key_findings`
9. If evidence is insufficient for any canonical component, write that blocker to the integrator report and do not fabricate unsupported results.
10. Also write a concise integrator report JSON recording source evidence files used, whether integration succeeded, and any blocker.
"""

    def _get_tools(self):
        return [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]

    def _get_agent_context(self):
        return get_worker_agent_context("ablation_report_integrator")

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
        return f"""Write the final canonical ablation report.

Input paths:
- idea_json_path: {self.idea_json_path}
- agent_reports_dir: {self.contract['agent_reports_dir']}
- model_dir: {self.contract['model_dir']}
- ablation_results_dir: {self.contract['ablation_results_dir']}

Output paths:
- ablation_results_json: {self.ablation_json_path}
- integrator_report_json: {self.integrator_report_path}

Canonical components:
- {', '.join(self.canonical_component_names)}

CRITICAL FORMAT - ablation_results.json must have EXACTLY this structure:
{{
  "components": {{
    "<component_name>": {{
      "result": "positive|negative|inconclusive",
      "metric": "...",
      "value": "...",
      "confidence": 0.0,
      "analysis": "...",
      "method_context": "COPY FROM idea.json.components[<index>].description"
    }}
  }},
  "summary": {{
    "feasible": true|false,
    "confidence": 0.0,
    "key_findings": ["...", "..."]
  }}
}}

Requirements:
- Ignore any existing file at {self.ablation_json_path}; do not read it and do not use it as evidence.
- Use only `idea.json`, `agent_reports/`, and `results/ablation/` as evidence sources.
- Prefer machine-readable status files and validator reports before raw logs.
- The top-level keys must be EXACTLY "components" and "summary" - no other top-level keys allowed.
- Each component must have ALL 6 fields: result, metric, value, confidence, analysis, method_context.
- method_context must be COPIED FROM idea.json.components[<index>].description (the idea's original description of this component).
- If evidence is insufficient for any canonical component, write that blocker to the integrator report and do not invent missing component results.
"""

    def _artifact_valid(self) -> bool:
        if not os.path.exists(self.ablation_json_path):
            return False
        try:
            payload = json.load(open(self.ablation_json_path, "r", encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, dict) or set(payload.keys()) != {"components", "summary"}:
            return False
        components = payload.get("components")
        summary = payload.get("summary")
        if not isinstance(components, dict) or not isinstance(summary, dict):
            return False
        if list(components.keys()) != list(self.canonical_component_names):
            return False
        required = {"result", "metric", "value", "confidence", "analysis", "method_context"}
        for name in self.canonical_component_names:
            entry = components.get(name)
            if not isinstance(entry, dict) or set(entry.keys()) != required:
                return False
            if any(entry.get(field) in (None, "") for field in required):
                return False
        if summary.get("feasible") is None or summary.get("confidence") is None:
            return False
        if not isinstance(summary.get("key_findings"), list):
            return False
        return True

    def _try_deterministic_materialization(self) -> Dict[str, Any]:
        return write_ablation_results_artifacts(
            workspace_root=self.workspace_root,
            project_root=self.project_root,
            generated_by="ablation_report_integrator_runtime",
            ablation_results_path=self.ablation_json_path,
            integrator_report_path=self.integrator_report_path,
        )

    async def execute(self) -> Dict[str, Any]:
        if self._artifact_valid() and os.path.exists(self.integrator_report_path):
            return {
                "output": "Existing ablation results artifact is already valid.",
                "ablation_results_path": self.ablation_json_path,
                "integrator_report_path": self.integrator_report_path,
                "final_artifact_contract_path": self.paths["final_artifact_contract"],
                "valid": True,
                "mode": "existing",
            }

        deterministic_result = self._try_deterministic_materialization()
        if deterministic_result.get("valid") and self._artifact_valid():
            return {
                "output": "Deterministic ablation-results materialization succeeded.",
                "ablation_results_path": self.ablation_json_path,
                "integrator_report_path": self.integrator_report_path,
                "final_artifact_contract_path": deterministic_result.get("final_artifact_contract_path"),
                "valid": True,
                "mode": "deterministic",
            }

        for stale_path in (self.ablation_json_path, self.integrator_report_path):
            if os.path.exists(stale_path):
                try:
                    os.remove(stale_path)
                except OSError:
                    pass
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            system_prompt=self._build_system_prompt(),
        )
        return {
            "output": self._extract_output(result),
            "ablation_results_path": self.ablation_json_path,
            "integrator_report_path": self.integrator_report_path,
            "final_artifact_contract_path": self.paths["final_artifact_contract"],
            "valid": self._artifact_valid(),
            "mode": "llm_fallback",
            "deterministic_blocker": deterministic_result.get("blocker"),
        }


async def run_ablation_report_integrator(
    workspace_root: str,
    project_root: str,
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = AblationReportIntegratorAgent(
        workspace_root=workspace_root,
        project_root=project_root,
        model=model or get_master_agent_model(),
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
