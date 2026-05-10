"""
Master orchestrator for experiment-agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agents.experiment_agent.agents.code import (
    EXPERIMENT_CODE_PLANNER,
    run_code_agent,
)
from src.agents.experiment_agent.agents.integration import run_iteration_reporter
from src.agents.experiment_agent.agents.science import (
    EXPERIMENT_ABLATION_SCIENCE_PLANNER,
    EXPERIMENT_STANDARD_SCIENCE_PLANNER,
    run_ablation_science_agent,
    run_standard_science_agent,
)
from src.agents.experiment_agent.config import get_science_max_iterations
from src.agents.experiment_agent.runtime.ablation_results import (
    build_ablation_final_artifact_contract,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    write_json_file,
    workspace_contract_paths,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.self_contained import scan_project_self_contained


class Decision(str):
    CODE_NEEDED = "CODE_NEEDED"
    STANDARD_EXP_NEEDED = "STANDARD_EXP_NEEDED"
    ABLATION_NEEDED = "ABLATION_NEEDED"
    CONVERGED = "CONVERGED"


DECISION_TO_PHASE = {
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


class MasterAgent:
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
        _ = model
        self.experiment_id = experiment_id
        self.idea_path = idea_path
        self.workspace_root = workspace_root
        self.project_root = project_root
        self.max_iterations = int(max_iterations) if max_iterations is not None else get_science_max_iterations()
        self.verbose = verbose
        self.resume = resume
        self.contract = workspace_contract_paths(workspace_root, project_root)
        self.paths = artifact_paths(workspace_root, project_root)
        self.agent_md_path = self.paths["master_report"]
        self.current_iteration = 1
        self.state = AgentState(iteration=1, phase="delegating", decision="")

    def _read_text_file(self, path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

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

    def _ensure_prepare_ready(self) -> None:
        prepare_payload = self._load_report(self.paths["prepare_validator"])
        if self._prepare_ready(prepare_payload):
            return
        raise ValueError(
            "Master requires a validator-backed prepare handoff before code/science orchestration. "
            f"Expected ready prepare artifact at {self.paths['prepare_validator']}."
        )

    def _write_self_contained_report(self) -> Dict[str, Any]:
        report = scan_project_self_contained(self.project_root, self.workspace_root)
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
                "reasons": violation_lines or ["Project code is not self-contained; remove runtime dependency on repos/."],
                "evidence_files": [self.paths["self_contained_report"]],
                "self_contained_project": False,
                "self_contained_violations": violation_lines,
            }

        code_report = normalize_phase_report(self._load_report(self.paths["code_validator"]))
        if code_report["phase_completion_status"] != "complete":
            return {
                "decision": Decision.CODE_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.CODE_NEEDED],
                "phase_completion_status": code_report["phase_completion_status"],
                "ready_for_next_phase": code_report["ready_for_next_phase"],
                "blocking_issues": code_report["blocking_issues"],
                "reasons": code_report["blocking_issues"] or ["Code phase incomplete."],
                "evidence_files": [self.paths["code_validator"]],
                "self_contained_project": True,
            }
        standard_report = normalize_phase_report(self._load_report(self.paths["standard_science_validator"]))
        if standard_report["phase_completion_status"] != "complete":
            return {
                "decision": Decision.STANDARD_EXP_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.STANDARD_EXP_NEEDED],
                "phase_completion_status": standard_report["phase_completion_status"],
                "ready_for_next_phase": standard_report["ready_for_next_phase"],
                "blocking_issues": standard_report["blocking_issues"],
                "reasons": standard_report["blocking_issues"] or ["Standard science incomplete."],
                "evidence_files": [self.paths["standard_science_validator"]],
                "self_contained_project": True,
            }
        ablation_report = normalize_phase_report(self._load_report(self.paths["ablation_science_validator"]))
        if ablation_report["phase_completion_status"] != "complete":
            return {
                "decision": Decision.ABLATION_NEEDED,
                "phase": DECISION_TO_PHASE[Decision.ABLATION_NEEDED],
                "phase_completion_status": ablation_report["phase_completion_status"],
                "ready_for_next_phase": ablation_report["ready_for_next_phase"],
                "blocking_issues": ablation_report["blocking_issues"],
                "reasons": ablation_report["blocking_issues"] or ["Ablation science incomplete."],
                "evidence_files": [self.paths["ablation_science_validator"]],
                "self_contained_project": True,
            }
        return {
            "decision": Decision.CONVERGED,
            "phase": DECISION_TO_PHASE[Decision.CONVERGED],
            "phase_completion_status": "complete",
            "ready_for_next_phase": True,
            "blocking_issues": [],
            "reasons": ["All phases completed with validator-backed evidence."],
            "evidence_files": [
                self.paths["prepare_validator"],
                self.paths["code_validator"],
                self.paths["standard_science_validator"],
                self.paths["ablation_science_validator"],
            ],
            "self_contained_project": True,
        }

    def _write_master_decision_artifact(self, payload: Dict[str, Any]) -> str:
        return write_json_file(self.paths["master_decision"], payload)

    def _write_runtime_phase_state(self, payload: Dict[str, Any]) -> str:
        state = {
            "iteration": self.current_iteration,
            "active_phase": payload.get("phase"),
            "decision": payload.get("decision"),
            "conclusion": "; ".join(payload.get("reasons") or []),
        }
        return write_json_file(self.paths["runtime_phase_state"], state)

    def _write_master_report(self, payload: Dict[str, Any]) -> str:
        lines = [
            "# Master Report",
            "",
            f"- iteration: {self.current_iteration}",
            f"- decision: {payload.get('decision')}",
            f"- phase: {payload.get('phase')}",
            "- reasons:",
        ]
        lines.extend(f"  - {reason}" for reason in payload.get("reasons") or [])
        with open(self.agent_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return self.agent_md_path

    def _materialize_ablation_results(self) -> bool:
        contract = build_ablation_final_artifact_contract(
            self.workspace_root,
            idea_json_path=self.paths["idea_json"],
        )
        write_json_file(self.paths["final_artifact_contract"], contract)
        return True

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
        ]
        with open(self.paths["results_summary"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return self.paths["results_summary"]

    async def _run_planner_task(
        self,
        subagent_type: str,
        description: str,
        planner_prompt: str,
    ) -> Dict[str, Any]:
        _ = description
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
        self._ensure_prepare_ready()
        last_decision = ""
        for iteration in range(1, self.max_iterations + 1):
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
                    "final_path": self.agent_md_path,
                    "converged": True,
                    "decision": decision,
                    "stopped_due_to_iteration_limit": False,
                }

            if decision == last_decision:
                continue

            planner_name = DECISION_TO_PLANNER[decision]
            planner_prompt = self._format_phase_prompt(decision, reasons)
            await self._run_planner_task(
                planner_name,
                description=f"Run {phase} phase",
                planner_prompt=planner_prompt,
            )
            await run_iteration_reporter(
                workspace_root=self.workspace_root,
                project_root=self.project_root,
                verbose=self.verbose,
                resume=self.resume,
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
