"""
Master agent outer-loop orchestrator.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openhands.sdk import get_logger
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.agent import OpenHandsBaseAgent
from src.agents.experiment_agent.agents.code import register_experiment_code_planner
from src.agents.experiment_agent.agents.science import register_science_planners
from src.agents.experiment_agent.config import (
    MASTER_AGENT_MODEL,
    PLANNER_MAX_TURNS,
    SCIENCE_MAX_ITERATIONS,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    workspace_contract_paths,
)
from src.agents.experiment_agent.skills import get_master_agent_context

logger = get_logger(__name__)


class Decision(str):
    CONVERGED = "CONVERGED"


@dataclass
class AgentState:
    iteration: int
    phase: str
    decision: str
    conclusion: str = ""


class MasterAgent(OpenHandsBaseAgent):
    MASTER_DEFAULT_MCP_SERVERS = ["filesystem"]

    def __init__(
        self,
        experiment_id: str,
        idea_path: str,
        workspace_root: str,
        project_root: str,
        model: str = MASTER_AGENT_MODEL,
        verbose: bool = True,
        max_iterations: int = SCIENCE_MAX_ITERATIONS,
        resume: bool = False,
    ):
        super().__init__(
            agent_type="Master",
            model=model,
            max_turns=PLANNER_MAX_TURNS,
            verbose=verbose,
            workspace_root=workspace_root,
            enable_condenser=True,
            condenser_max_size=300,
            condenser_keep_first=50,
            resume=resume,
        )
        self.experiment_id = experiment_id
        self.idea_path = idea_path
        self.workspace_root = workspace_root
        self.project_root = project_root
        self.max_iterations = max_iterations
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.agent_md_path = self.paths["master_report"]
        self.continue_flag_path = os.path.join(
            os.path.dirname(self.agent_md_path), "master_continue_flag.json"
        )
        self.current_iteration = 1
        self.state = AgentState(iteration=1, phase="delegating", decision="")
        register_experiment_code_planner()
        register_science_planners()

    def _build_user_prompt(self, **kwargs) -> str:
        """Build the user prompt for the agent. MasterAgent uses _build_iteration_prompt directly."""
        _ = kwargs
        return ""

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return """You are the master orchestrator for an experiment workspace.

Your job is to route work between the code and science planners until the idea has been sufficiently validated or falsified by real workspace evidence.

Your end goal is not merely to finish phases. Your end goal is to ensure the idea has:
1. enough real experimental evidence,
2. enough exact per-component ablation evidence,
3. code that is implemented correctly enough for those experiments to be meaningful.

Core behavior:
1. Treat the actual contents of `agent_reports/`, `results/`, and `idea.json` as the source of truth.
2. Launch exactly one planner per iteration through the `task` tool when more work is needed.
3. Let planners run their own worker/validator loops.
4. Never invent a third science lane. Stress, adversarial, drift, robustness, or other follow-up science must stay inside standard science or ablation science.
5. Write your continue_iteration decision to the flag file path given in the iteration prompt.

CRITICAL: What counts as VALID evidence vs INVALID evidence:

VALID completion means:
- Standard experiment: metrics show meaningful differences between baseline and treatment
- Ablation experiment: probe_count > 0 AND control deltas > 0 (mechanism actually triggered)
- Results show effect sizes > 0 with statistical significance

INVALID completion (MUST continue iterating):
- "probe_count=0" or "delta_count=0" - the mechanism did not trigger
- "no significant differences" or "effect_size=0" between conditions
- "COMPLETE (with technical issues)" - technical issues mean validation FAILED
- Experiment ran but metrics are identical across conditions
- Any condition showing "no_data" for primary metrics

If ablation shows probe_count=0 or no differentiation between conditions:
  - This is NOT valid completion
  - You MUST continue iterating to fix the probe triggering issue
  - Do NOT treat "ran but didn't work" as "validated"

Only set continue_iteration=false when:
  - probe_count > 0 AND control deltas > 0 (mechanism verified)
  - Effect sizes are non-zero with meaningful magnitude
  - Component-level ablation results exist for all 8 canonical components
"""

    def _build_iteration_prompt(self) -> str:
        return f"""Run one master orchestration iteration for this experiment workspace.

This iteration must be controlled only by you. The outer loop will only read `{self.continue_flag_path}` to determine whether to continue.

You must inspect the actual contents of:
- agent_reports: {self.contract["agent_reports_dir"]}
- results: {self.contract["results_dir"]}
- idea input: {self.paths["idea_json"]}
- previous master note: {self.agent_md_path}

Rules:
1. Read file contents, not just filenames.
2. Treat `master_report.md` as the previous iteration note only. It is useful for continuity but must never outweigh newer evidence.
3. **MANDATORY PHASE ORDER**: Code implementation MUST be completed before experiments run. If `agent_reports/code_validator_report.json` does not exist with status PASS, you MUST choose `experiment_code_planner`.
4. If more work is needed, choose exactly one next planner and call the `task` tool exactly once with one of:
   - `experiment_code_planner`
   - `experiment_standard_science_planner`
   - `experiment_ablation_science_planner`
5. Do not invent any planner or phase name outside code, standard science, and ablation science.
6. If no more work is needed, do not call any further agent. The ablation science planner will automatically call the ablation report integrator after ablation experiments complete.
7. Update `{self.agent_md_path}` so it records:
- current iteration number
- current phase
- current decision
- the concrete reasons based on evidence
- the evidence files you relied on
- the next planner task you selected, or the final conclusion if stopping
8. Your goal is to ensure the idea is sufficiently validated or falsified through substantial experiments, exact per-component ablations, and code that is correct enough for the scientific conclusion to be meaningful.
9. When done, write to `{self.continue_flag_path}` exactly one of:
   - `{{"continue_iteration": true}}` if more work is needed
   - `{{"continue_iteration": false}}` if the workflow should stop
"""

    def _build_mcp_config(self) -> Dict[str, Any]:
        base_config = super()._build_mcp_config()
        servers = base_config.get("mcpServers") if isinstance(base_config, dict) else None
        if not isinstance(servers, dict):
            return {"mcpServers": {}}
        filtered_servers = {
            name: servers[name]
            for name in self.MASTER_DEFAULT_MCP_SERVERS
            if name in servers
        }
        return {"mcpServers": filtered_servers}

    def _get_tools(self):
        return [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
            Tool(name=TaskToolSet.name),
        ]

    def _get_agent_context(self):
        return get_master_agent_context()

    def _load_agent_md(self) -> Optional[AgentState]:
        if not os.path.exists(self.agent_md_path):
            return None
        try:
            content = open(self.agent_md_path, "r", encoding="utf-8").read()
        except Exception as exc:
            logger.warning("Failed to read master report: %s", exc)
            return None
        iteration = 1
        phase = "delegating"
        decision = ""
        conclusion = ""
        match = re.search(r"# Agent State - Iteration (\d+)", content)
        if match:
            iteration = int(match.group(1))
        for line in content.splitlines():
            if line.startswith("**Phase:**"):
                phase = line.split("**Phase:**", 1)[1].strip()
            elif line.startswith("**Decision:**"):
                decision = line.split("**Decision:**", 1)[1].strip()
        conclusion_match = re.search(r"## Conclusion\s*\n(.+)", content, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        return AgentState(iteration=iteration, phase=phase, decision=decision, conclusion=conclusion)

    def _read_continue_flag(self) -> bool:
        """Read continue_iteration flag from file."""
        if not os.path.exists(self.continue_flag_path):
            return True  # Default to continue if no flag file
        try:
            with open(self.continue_flag_path, "r") as f:
                data = json.load(f)
            return bool(data.get("continue_iteration", True))
        except Exception:
            return True

    async def run_orchestration(self) -> Dict[str, Any]:
        logger.info("Starting Master Agent multi-round orchestration...")
        previous_state = self._load_agent_md()
        if previous_state:
            self.current_iteration = previous_state.iteration
            self.state = previous_state

        start_iteration = self.current_iteration if self.current_iteration >= 1 else 1
        last_continue_iteration = False
        for iteration in range(start_iteration, self.max_iterations + 1):
            self.current_iteration = iteration
            result = await self.run(
                user_prompt=self._build_iteration_prompt(),
                system_prompt=self._build_system_prompt(),
            )
            continue_iteration = self._read_continue_flag()
            last_continue_iteration = continue_iteration
            latest_state = self._load_agent_md()
            if latest_state:
                self.state = latest_state
            if not continue_iteration:
                return {
                    "iterations": self.current_iteration,
                    "final_path": self.agent_md_path,
                    "converged": self.state.decision == Decision.CONVERGED,
                    "decision": self.state.decision or "",
                    "stopped_due_to_iteration_limit": False,
                }

        return {
            "iterations": self.current_iteration,
            "final_path": self.agent_md_path,
            "converged": self.state.decision == Decision.CONVERGED,
            "decision": self.state.decision or "",
            "stopped_due_to_iteration_limit": bool(last_continue_iteration),
        }


async def run_master(
    experiment_id: str,
    idea_path: str,
    workspace_root: str,
    project_root: str,
    max_iterations: int = SCIENCE_MAX_ITERATIONS,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = MasterAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        workspace_root=workspace_root,
        project_root=project_root,
        max_iterations=max_iterations,
        verbose=verbose,
        resume=resume,
    )
    return await agent.run_orchestration()
