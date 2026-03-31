"""
Master agent outer-loop orchestrator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openhands.sdk import get_logger
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.agent import OpenHandsBaseAgent
from src.agents.experiment_agent.agents.code import (
    EXPERIMENT_CODE_PLANNER,
    register_experiment_code_planner,
    run_code_agent,
)
from src.agents.experiment_agent.agents.science import (
    EXPERIMENT_ABLATION_SCIENCE_PLANNER,
    EXPERIMENT_STANDARD_SCIENCE_PLANNER,
    register_science_planners,
    run_ablation_science_agent,
    run_standard_science_agent,
)
from src.agents.experiment_agent.config import (
    get_master_agent_model,
    get_planner_max_turns,
    get_science_max_iterations,
)
from src.agents.experiment_agent.runtime.ablation_results import (
    build_ablation_results_artifacts,
    write_ablation_results_artifacts,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    write_json_file,
    workspace_contract_paths,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.self_contained import scan_project_self_contained
from src.agents.experiment_agent.skills import get_master_agent_context

logger = get_logger(__name__)


class Decision(str):
    PREPARE_NEEDED = "PREPARE_NEEDED"
    CODE_NEEDED = "CODE_NEEDED"
    STANDARD_EXP_NEEDED = "STANDARD_EXP_NEEDED"
    ABLATION_NEEDED = "ABLATION_NEEDED"
    CONVERGED = "CONVERGED"


DECISION_TO_PHASE = {
    Decision.PREPARE_NEEDED: "prepare",
    Decision.CODE_NEEDED: "code",
    Decision.STANDARD_EXP_NEEDED: "standard_science",
    Decision.ABLATION_NEEDED: "ablation_science",
    Decision.CONVERGED: "complete",
}

DECISION_TO_PLANNER = {
    Decision.CODE_NEEDED: EXPERIMENT_CODE_PLANNER,
    Decision.STANDARD_EXP_NEEDED: EXPERIMENT_STANDARD_SCIENCE_PLANNER,
    Decision.ABLATION_NEEDED: EXPERIMENT_ABLATION_SCIENCE_PLANNER,
}


@dataclass
class AgentState:
    iteration: int
    phase: str
    decision: str
    conclusion: str = ""


class MasterAgent(OpenHandsBaseAgent):
    MASTER_DEFAULT_MCP_SERVERS: list[str] = []
    SYSTEM_PROMPT_TEMPLATE = "master_agent.j2"

    def __init__(
        self,
        experiment_id: str,
        idea_path: str,
        workspace_root: str,
        project_root: str,
        model: str | None = None,
        verbose: bool = True,
        max_iterations: int | None = None,
        resume: bool = False,
    ):
        super().__init__(
            agent_type="Master",
            model=model or get_master_agent_model(),
            max_turns=get_planner_max_turns(),
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
        self.max_iterations = (
            int(max_iterations)
            if max_iterations is not None
            else get_science_max_iterations()
        )
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.agent_md_path = self.paths["master_report"]
        self.current_iteration = 1
        self.state = AgentState(iteration=1, phase="delegating", decision="")
        register_experiment_code_planner()
        register_science_planners()

    def _build_user_prompt(self, **kwargs) -> str:
        _ = kwargs
        return ""

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        return ""

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

    def _load_runtime_state(self) -> Optional[AgentState]:
        payload = load_json_file(self.paths["runtime_phase_state"])
        if not isinstance(payload, dict):
            return None
        try:
            iteration = int(payload.get("iteration") or 1)
        except Exception:
            iteration = 1
        phase = str(payload.get("active_phase") or "delegating").strip() or "delegating"
        decision = str(payload.get("decision") or "").strip()
        conclusion = str(payload.get("conclusion") or "").strip()
        return AgentState(
            iteration=iteration,
            phase=phase,
            decision=decision,
            conclusion=conclusion,
        )

    def _load_report(self, path: str) -> Dict[str, Any]:
        payload = load_json_file(path)
        return payload if isinstance(payload, dict) else {}

    def _prepare_ready(self, payload: Dict[str, Any]) -> bool:
        phase_report = normalize_phase_report(payload)
        if phase_report["status"] == "PASS":
            return True
        if phase_report["status"] == "PARTIAL" and phase_report["ready_for_next_phase"]:
            return True
        return False

    def _write_self_contained_report(self) -> Dict[str, Any]:
        report = scan_project_self_contained(
            self.project_root,
            self.workspace_root,
        )
        write_json_file(self.paths["self_contained_report"], report)
        return report

    def _format_phase_prompt(self, decision: str, reasons: list[str]) -> str:
        lines = [
            f"Master runtime selected {DECISION_TO_PHASE.get(decision, 'unknown')} work.",
            "",
            "Reasons:",
        ]
        lines.extend(f"- {reason}" for reason in reasons)
        lines.extend(
            [
                "",
                "Workspace paths:",
                f"- idea_path: {self.idea_path}",
                f"- idea_json_path: {self.paths['idea_json']}",
                f"- project_root: {self.project_root}",
                f"- model_dir: {self.contract['model_dir']}",
                f"- results_dir: {self.contract['results_dir']}",
                f"- agent_reports_dir: {self.contract['agent_reports_dir']}",
            ]
        )
        return "\n".join(lines)

    def _compute_gate_payload(self) -> Dict[str, Any]:
        prepare_payload = self._load_report(self.paths["prepare_validator"])
        if not self._prepare_ready(prepare_payload):
            phase_report = normalize_phase_report(prepare_payload)
            return {
                "decision": Decision.PREPARE_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.PREPARE_NEEDED],
                "phase_completion_status": phase_report["phase_completion_status"],
                "ready_for_next_phase": phase_report["ready_for_next_phase"],
                "blocking_issues": phase_report["blocking_issues"],
                "reasons": phase_report["blocking_issues"]
                or ["Prepare validator has not produced a ready handoff."],
                "evidence_files": [self.paths["prepare_validator"]],
            }

        self_contained_report = self._write_self_contained_report()
        if not self_contained_report.get("self_contained_project"):
            violation_lines = [
                f"{item.get('path')}:{item.get('line')} [{item.get('rule')}] {item.get('snippet')}"
                for item in self_contained_report.get("self_contained_violations") or []
            ]
            return {
                "decision": Decision.CODE_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.CODE_NEEDED],
                "phase_completion_status": "partial",
                "ready_for_next_phase": False,
                "blocking_issues": violation_lines,
                "reasons": violation_lines
                or ["Project code is not self-contained; remove runtime dependency on repos/."],
                "evidence_files": [self.paths["self_contained_report"]],
                "self_contained_project": False,
                "self_contained_violations": violation_lines,
            }

        code_payload = self._load_report(self.paths["code_validator"])
        code_phase = normalize_phase_report(code_payload)
        if code_phase["phase_completion_status"] != "complete":
            return {
                "decision": Decision.CODE_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.CODE_NEEDED],
                "phase_completion_status": code_phase["phase_completion_status"],
                "ready_for_next_phase": code_phase["ready_for_next_phase"],
                "blocking_issues": code_phase["blocking_issues"],
                "reasons": code_phase["blocking_issues"]
                or ["Code phase is not complete."],
                "evidence_files": [self.paths["code_validator"], self.paths["self_contained_report"]],
                "self_contained_project": self_contained_report.get("self_contained_project"),
                "self_contained_violations": list(self_contained_report.get("self_contained_violations") or []),
            }

        standard_payload = self._load_report(self.paths["standard_science_validator"])
        standard_phase = normalize_phase_report(standard_payload)
        if standard_phase["phase_completion_status"] != "complete":
            return {
                "decision": Decision.STANDARD_EXP_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.STANDARD_EXP_NEEDED],
                "phase_completion_status": standard_phase["phase_completion_status"],
                "ready_for_next_phase": standard_phase["ready_for_next_phase"],
                "blocking_issues": standard_phase["blocking_issues"],
                "reasons": standard_phase["blocking_issues"]
                or ["Standard science phase is not complete."],
                "evidence_files": [self.paths["standard_science_validator"], self.paths["self_contained_report"]],
                "self_contained_project": self_contained_report.get("self_contained_project"),
                "self_contained_violations": list(self_contained_report.get("self_contained_violations") or []),
            }

        ablation_payload = self._load_report(self.paths["ablation_science_validator"])
        ablation_phase = normalize_phase_report(ablation_payload)
        if ablation_phase["phase_completion_status"] != "complete":
            return {
                "decision": Decision.ABLATION_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.ABLATION_NEEDED],
                "phase_completion_status": ablation_phase["phase_completion_status"],
                "ready_for_next_phase": ablation_phase["ready_for_next_phase"],
                "blocking_issues": ablation_phase["blocking_issues"],
                "reasons": ablation_phase["blocking_issues"]
                or ["Ablation science phase is not complete."],
                "evidence_files": [self.paths["ablation_science_validator"], self.paths["self_contained_report"]],
                "self_contained_project": self_contained_report.get("self_contained_project"),
                "self_contained_violations": list(self_contained_report.get("self_contained_violations") or []),
            }

        materialization_preview = build_ablation_results_artifacts(
            self.workspace_root,
            self.project_root,
            generated_by="master_runtime_preview",
        )
        if not materialization_preview.get("valid"):
            return {
                "decision": Decision.ABLATION_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.ABLATION_NEEDED],
                "phase_completion_status": "partial",
                "ready_for_next_phase": False,
                "blocking_issues": [materialization_preview.get("blocker") or ""],
                "reasons": [
                    materialization_preview.get("blocker")
                    or "Ablation results cannot be materialized from current validator evidence."
                ],
                "evidence_files": materialization_preview.get("source_evidence_files")
                or [self.paths["ablation_science_validator"], self.paths["self_contained_report"]],
                "self_contained_project": self_contained_report.get("self_contained_project"),
                "self_contained_violations": list(self_contained_report.get("self_contained_violations") or []),
            }

        return {
            "decision": Decision.CONVERGED,
            "phase": DECISION_TO_PHASE[Decision.CONVERGED],
            "phase_completion_status": "complete",
            "ready_for_next_phase": True,
            "blocking_issues": [],
            "reasons": ["All validator-backed gates passed."],
            "evidence_files": materialization_preview.get("source_evidence_files")
            or [
                self.paths["prepare_validator"],
                self.paths["code_validator"],
                self.paths["standard_science_validator"],
                self.paths["ablation_science_validator"],
                self.paths["self_contained_report"],
            ],
            "self_contained_project": self_contained_report.get("self_contained_project"),
            "self_contained_violations": list(self_contained_report.get("self_contained_violations") or []),
        }

    def _write_master_decision_artifact(self, payload: Dict[str, Any]) -> str:
        artifact_payload = {
            "iteration": self.current_iteration,
            "decision": payload.get("decision"),
            "phase": payload.get("phase"),
            "phase_completion_status": payload.get("phase_completion_status"),
            "ready_for_next_phase": bool(payload.get("ready_for_next_phase")),
            "blocking_issues": list(payload.get("blocking_issues") or []),
            "reasons": list(payload.get("reasons") or []),
            "evidence_files": list(payload.get("evidence_files") or []),
            "self_contained_project": payload.get("self_contained_project"),
            "self_contained_violations": list(payload.get("self_contained_violations") or []),
        }
        return write_json_file(self.paths["master_decision"], artifact_payload)

    def _write_runtime_phase_state(self, payload: Dict[str, Any]) -> str:
        state_payload = {
            "iteration": self.current_iteration,
            "active_phase": payload.get("phase"),
            "decision": payload.get("decision"),
            "phase_completion_status": payload.get("phase_completion_status"),
            "ready_for_next_phase": bool(payload.get("ready_for_next_phase")),
            "blocking_issues": list(payload.get("blocking_issues") or []),
            "self_contained_project": payload.get("self_contained_project"),
            "conclusion": "; ".join(str(item).strip() for item in payload.get("reasons") or [] if str(item).strip()),
        }
        return write_json_file(self.paths["runtime_phase_state"], state_payload)

    def _write_master_report(self, payload: Dict[str, Any]) -> str:
        decision = str(payload.get("decision") or "")
        phase = str(payload.get("phase") or "delegating")
        reasons = [str(item).strip() for item in payload.get("reasons") or [] if str(item).strip()]
        evidence_files = [
            str(item).strip() for item in payload.get("evidence_files") or [] if str(item).strip()
        ]

        lines = [
            f"# Agent State - Iteration {self.current_iteration}",
            "",
            f"**Phase:** {phase}",
            f"**Decision:** {decision}",
            "",
            "## Reasons",
        ]
        if reasons:
            lines.extend(f"- {reason}" for reason in reasons)
        else:
            lines.append("- (none)")
        lines.extend(["", "## Evidence"])
        if evidence_files:
            lines.extend(f"- {path}" for path in evidence_files)
        else:
            lines.append("- (none)")
        lines.extend(
            [
                "",
                "## Conclusion",
                "Converged." if decision == Decision.CONVERGED else f"Next required phase: {phase}.",
                "",
            ]
        )
        os.makedirs(os.path.dirname(self.agent_md_path), exist_ok=True)
        with open(self.agent_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return self.agent_md_path

    def _materialize_ablation_results(self) -> bool:
        result = write_ablation_results_artifacts(
            self.workspace_root,
            self.project_root,
            generated_by="master_runtime",
        )
        return bool(result.get("valid"))

    def _materialize_results_summary(self) -> str:
        payload = self._compute_gate_payload()
        lines = [
            "# Master Summary",
            "",
            f"- iteration: {self.current_iteration}",
            f"- decision: {payload.get('decision')}",
            f"- phase: {payload.get('phase')}",
            f"- phase_completion_status: {payload.get('phase_completion_status')}",
            f"- ready_for_next_phase: {bool(payload.get('ready_for_next_phase'))}",
            "- reasons:",
        ]
        lines.extend(f"  - {reason}" for reason in payload.get("reasons") or [])
        lines.extend(["", "- blocking_issues:"])
        lines.extend(f"  - {issue}" for issue in payload.get("blocking_issues") or [])
        lines.extend(["", "- evidence_files:"])
        lines.extend(f"  - {path}" for path in payload.get("evidence_files") or [])
        summary_path = self.paths["results_summary"]
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return summary_path

    async def _run_planner_task(
        self,
        subagent_type: str,
        description: str,
        planner_prompt: str,
    ) -> Dict[str, Any]:
        logger.info("Master routing to %s: %s", subagent_type, description)
        if subagent_type == EXPERIMENT_CODE_PLANNER:
            return await run_code_agent(
                experiment_id=self.experiment_id,
                idea_path=self.idea_path,
                project_root=self.project_root,
                workspace_root=self.workspace_root,
                plan=planner_prompt,
                verbose=self.verbose,
                resume=self.resume,
            )
        if subagent_type == EXPERIMENT_STANDARD_SCIENCE_PLANNER:
            return await run_standard_science_agent(
                experiment_id=self.experiment_id,
                idea_path=self.idea_path,
                project_root=self.project_root,
                workspace_root=self.workspace_root,
                plan=planner_prompt,
                code_summary=self._read_text_file(self.paths["code_summary"]),
                code_usage=self._read_text_file(self.paths["code_usage"]),
                verbose=self.verbose,
                resume=self.resume,
            )
        if subagent_type == EXPERIMENT_ABLATION_SCIENCE_PLANNER:
            return await run_ablation_science_agent(
                experiment_id=self.experiment_id,
                idea_path=self.idea_path,
                project_root=self.project_root,
                workspace_root=self.workspace_root,
                plan=planner_prompt,
                code_summary=self._read_text_file(self.paths["code_summary"]),
                code_usage=self._read_text_file(self.paths["code_usage"]),
                verbose=self.verbose,
                resume=self.resume,
            )
        raise ValueError(f"Unsupported planner task: {subagent_type}")

    async def run_orchestration(self) -> Dict[str, Any]:
        logger.info("Starting Master runtime-controlled orchestration...")
        previous_state = self._load_runtime_state()
        if previous_state:
            self.current_iteration = max(1, previous_state.iteration)
            self.state = previous_state

        start_iteration = self.current_iteration if self.current_iteration >= 1 else 1
        last_decision = ""
        for iteration in range(start_iteration, self.max_iterations + 1):
            self.current_iteration = iteration
            payload = self._compute_gate_payload()
            decision = str(payload.get("decision") or "")
            phase = str(payload.get("phase") or "delegating")
            reasons = list(payload.get("reasons") or [])

            self.state = AgentState(
                iteration=iteration,
                phase=phase,
                decision=decision,
                conclusion="; ".join(reasons),
            )
            self._write_master_decision_artifact(payload)
            self._write_runtime_phase_state(payload)
            self._write_master_report(payload)

            if decision == Decision.CONVERGED:
                self._materialize_ablation_results()
                self._materialize_results_summary()
                return {
                    "iterations": self.current_iteration,
                    "final_path": self.paths["ablation_results"] if os.path.exists(self.paths["ablation_results"]) else self.agent_md_path,
                    "converged": True,
                    "decision": decision,
                    "stopped_due_to_iteration_limit": False,
                }

            if decision == Decision.PREPARE_NEEDED:
                self._materialize_results_summary()
                return {
                    "iterations": self.current_iteration,
                    "final_path": self.agent_md_path,
                    "converged": False,
                    "decision": decision,
                    "stopped_due_to_iteration_limit": False,
                }

            if decision == last_decision:
                logger.info(
                    "Master decision %s is unchanged after the previous planner run; skipping duplicate dispatch.",
                    decision,
                )
                continue

            planner_name = DECISION_TO_PLANNER[decision]
            planner_prompt = self._format_phase_prompt(decision, reasons)
            await self._run_planner_task(
                planner_name,
                description=f"Run {phase} phase",
                planner_prompt=planner_prompt,
            )
            last_decision = decision

        payload = self._compute_gate_payload()
        self._write_master_decision_artifact(payload)
        self._write_runtime_phase_state(payload)
        self._write_master_report(payload)
        self._materialize_results_summary()
        return {
            "iterations": self.current_iteration,
            "final_path": self.agent_md_path,
            "converged": str(payload.get("decision") or "") == Decision.CONVERGED,
            "decision": str(payload.get("decision") or ""),
            "stopped_due_to_iteration_limit": True,
        }


async def run_master(
    experiment_id: str,
    idea_path: str,
    workspace_root: str,
    project_root: str,
    max_iterations: int | None = None,
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
