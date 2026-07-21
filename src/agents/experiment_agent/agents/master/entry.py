"""Sequential master orchestrator for experiment-agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agents.experiment_agent.agents.code import (
    EXPERIMENT_CODE_PLANNER,
    run_code_agent,
)
from src.agents.experiment_agent.agents.science import (
    EXPERIMENT_SCIENCE_PLANNER,
    run_science_agent,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    write_json_file,
    workspace_paths,
)
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report
from src.agents.experiment_agent.runtime.self_contained import scan_project_self_contained


class Decision(str):
    RUN_CODE = "RUN_CODE"
    RUN_SCIENCE = "RUN_SCIENCE"
    CONVERGED = "CONVERGED"


DECISION_TO_PHASE = {
    Decision.RUN_CODE: "code",
    Decision.RUN_SCIENCE: "science",
    Decision.CONVERGED: "complete",
}

DECISION_TO_PLANNER = {
    Decision.RUN_CODE: EXPERIMENT_CODE_PLANNER,
    Decision.RUN_SCIENCE: EXPERIMENT_SCIENCE_PLANNER,
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
        resume: bool = False,
    ):
        _ = model
        self.experiment_id = experiment_id
        self.idea_path = idea_path
        self.workspace_root = workspace_root
        self.project_root = project_root
        self.verbose = verbose
        self.resume = resume
        self.workspace_paths = workspace_paths(workspace_root, project_root)
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
        return (
            phase_report["status"] == "PASS"
            and phase_report["phase_completion_status"] == "complete"
            and phase_report["ready_for_next_phase"] is True
        )

    def _prepare_blocked_payload(self) -> Dict[str, Any] | None:
        prepare_payload = self._load_report(self.paths["prepare_reviewer"])
        if self._prepare_ready(prepare_payload):
            return None
        phase_report = normalize_phase_report(prepare_payload)
        issues = list(phase_report.get("blocking_issues") or [])
        if not issues:
            issues = ["Prepare phase has not produced a reviewer-approved handoff."]
        return {
            "decision": "PREPARE_BLOCKED",
            "phase": "prepare",
            "phase_completion_status": phase_report["phase_completion_status"],
            "ready_for_next_phase": False,
            "blocking_issues": issues,
            "reasons": issues,
            "evidence_files": [self.paths["prepare_reviewer"]],
            "self_contained_project": True,
        }

    def _write_self_contained_report(self) -> Dict[str, Any]:
        report = scan_project_self_contained(self.project_root, self.workspace_root)
        write_json_file(self.paths["self_contained_report"], report)
        return report

    def _format_phase_prompt(self, decision: str, reasons: list[str]) -> str:
        lines = [
            f"Master runtime is running the fixed {DECISION_TO_PHASE.get(decision, 'unknown')} phase.",
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
                f"- model_dir: {self.workspace_paths['model_dir']}",
                f"- results_dir: {self.workspace_paths['results_dir']}",
                f"- agent_reports_dir: {self.workspace_paths['agent_reports_dir']}",
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
                "decision": Decision.RUN_CODE,
                "phase": DECISION_TO_PHASE[Decision.RUN_CODE],
                "phase_completion_status": "partial",
                "ready_for_next_phase": False,
                "blocking_issues": violation_lines,
                "reasons": violation_lines or ["Project code is not self-contained; remove runtime dependency on repos/."],
                "evidence_files": [self.paths["self_contained_report"]],
                "self_contained_project": False,
                "self_contained_violations": violation_lines,
            }

        code_report = normalize_phase_report(self._load_report(self.paths["code_reviewer"]))
        if not (
            code_report["status"] == "PASS"
            and code_report["phase_completion_status"] == "complete"
            and code_report["ready_for_next_phase"] is True
        ):
            return {
                "decision": Decision.RUN_CODE,
                "phase": DECISION_TO_PHASE[Decision.RUN_CODE],
                "status": code_report["status"],
                "phase_completion_status": code_report["phase_completion_status"],
                "ready_for_next_phase": code_report["ready_for_next_phase"],
                "blocking_issues": code_report["blocking_issues"],
                "reasons": code_report["blocking_issues"] or [
                    "Code phase has not produced a PASS/complete/ready phase report."
                ],
                "evidence_files": [self.paths["code_reviewer"]],
                "self_contained_project": True,
            }
        science_report = normalize_phase_report(self._load_report(self.paths["science_reviewer"]))
        if not (
            science_report["status"] == "PASS"
            and science_report["phase_completion_status"] == "complete"
            and science_report["ready_for_next_phase"] is True
        ):
            return {
                "decision": Decision.RUN_SCIENCE,
                "phase": DECISION_TO_PHASE[Decision.RUN_SCIENCE],
                "status": science_report["status"],
                "phase_completion_status": science_report["phase_completion_status"],
                "ready_for_next_phase": science_report["ready_for_next_phase"],
                "blocking_issues": science_report["blocking_issues"],
                "reasons": science_report["blocking_issues"] or [
                    "Science phase has not produced a PASS/complete/ready phase report."
                ],
                "evidence_files": [self.paths["science_reviewer"]],
                "self_contained_project": True,
            }
        return {
            "decision": Decision.CONVERGED,
            "phase": DECISION_TO_PHASE[Decision.CONVERGED],
            "phase_completion_status": "complete",
            "ready_for_next_phase": True,
            "blocking_issues": [],
            "reasons": ["All phases completed with reviewer-approved evidence."],
            "evidence_files": [
                self.paths["prepare_reviewer"],
                self.paths["code_reviewer"],
                self.paths["science_reviewer"],
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

    def _materialize_results_summary(self, payload: Optional[Dict[str, Any]] = None) -> str:
        payload = payload or self._compute_gate_payload()
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

    def _phase_snapshot(self, key: str) -> Dict[str, object]:
        report = normalize_phase_report(self._load_report(self.paths[key]))
        return {
            "status": report["status"],
            "phase_completion_status": report["phase_completion_status"],
            "ready_for_next_phase": report["ready_for_next_phase"],
            "artifact_role": report["artifact_role"],
            "run_level": report["run_level"],
            "blocking_issues": report["blocking_issues"],
        }

    def _write_iteration_snapshot(self) -> None:
        phase_states = {
            "code": self._phase_snapshot("code_reviewer"),
            "science": self._phase_snapshot("science_reviewer"),
        }
        payload = {
            "iteration": self.current_iteration,
            "code_status": (
                "complete"
                if phase_states["code"]["phase_completion_status"] == "complete"
                else "incomplete"
            ),
            "code_evidence": [self.paths["code_reviewer"]],
            "science_experiments": (
                "complete"
                if phase_states["science"]["phase_completion_status"] == "complete"
                else "partial"
            ),
            "science_evidence": [self.paths["science_reviewer"]],
            "phase_states": phase_states,
            "blockers": phase_states["code"]["blocking_issues"] + phase_states["science"]["blocking_issues"],
        }
        write_json_file(self.paths["iteration_status"], payload)
        lines = ["# Iteration Summary", ""]
        lines.extend(f"- {name}: {state['phase_completion_status']}" for name, state in phase_states.items())
        with open(self.paths["iteration_summary"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _record_master_payload(self, payload: Dict[str, Any]) -> None:
        decision = str(payload.get("decision") or "")
        phase = str(payload.get("phase") or "delegating")
        reasons = list(payload.get("reasons") or [])
        self.state = AgentState(
            iteration=self.current_iteration,
            phase=phase,
            decision=decision,
            conclusion="; ".join(reasons),
        )
        self._write_master_decision_artifact(payload)
        self._write_runtime_phase_state(payload)
        self._write_master_report(payload)

    def _blocked_result(self, payload: Dict[str, Any], *, iterations: int) -> Dict[str, Any]:
        self._materialize_results_summary(payload)
        return {
            "iterations": iterations,
            "final_path": self.agent_md_path,
            "converged": False,
            "decision": str(payload.get("decision") or ""),
            "stopped_due_to_iteration_limit": False,
            "blocked": True,
            "blocking_issues": list(payload.get("blocking_issues") or payload.get("reasons") or []),
        }

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
        if subagent_type == EXPERIMENT_SCIENCE_PLANNER:
            return await run_science_agent(
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
        return {
            "status": "blocked",
            "blocking_issues": [f"Unsupported planner task: {subagent_type}"],
        }

    async def run_orchestration(self) -> Dict[str, Any]:
        blocked_payload = self._prepare_blocked_payload()
        if blocked_payload is not None:
            self._record_master_payload(blocked_payload)
            return self._blocked_result(blocked_payload, iterations=0)

        phase_runs = 0
        for decision in (Decision.RUN_CODE, Decision.RUN_SCIENCE):
            phase_runs += 1
            self.current_iteration = phase_runs
            payload = {
                "decision": decision,
                "phase": DECISION_TO_PHASE[decision],
                "status": "RUNNING",
                "phase_completion_status": "partial",
                "ready_for_next_phase": False,
                "blocking_issues": [],
                "reasons": [
                    "Sequential experiment control plane runs code then science; "
                    "all validation is enforced inside each phase by prefinish hooks."
                ],
                "evidence_files": [],
                "self_contained_project": True,
            }
            self._record_master_payload(payload)
            reasons = list(payload.get("reasons") or [])
            planner_prompt = self._format_phase_prompt(decision, reasons)
            await self._run_planner_task(
                DECISION_TO_PLANNER[decision],
                description=f"Run {payload.get('phase') or DECISION_TO_PHASE[decision]} phase",
                planner_prompt=planner_prompt,
            )
            self._write_iteration_snapshot()
            payload = self._compute_gate_payload()
            self._record_master_payload(payload)
            current_decision = str(payload.get("decision") or "")
            if decision == Decision.RUN_CODE and current_decision == Decision.RUN_CODE:
                return self._blocked_result(payload, iterations=phase_runs)
            if decision == Decision.RUN_SCIENCE and current_decision != Decision.CONVERGED:
                return self._blocked_result(payload, iterations=phase_runs)

        payload = self._compute_gate_payload()
        self._record_master_payload(payload)
        self._materialize_results_summary(payload)
        converged = str(payload.get("decision") or "") == Decision.CONVERGED
        return {
            "iterations": phase_runs,
            "final_path": self.agent_md_path,
            "converged": converged,
            "decision": str(payload.get("decision") or ""),
            "stopped_due_to_iteration_limit": False,
            "blocked": not converged,
            "blocking_issues": [] if converged else list(payload.get("blocking_issues") or []),
        }


async def run_master(
    experiment_id: str,
    idea_path: str,
    workspace_root: str,
    project_root: str,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    agent = MasterAgent(
        experiment_id=experiment_id,
        idea_path=idea_path,
        workspace_root=workspace_root,
        project_root=project_root,
        verbose=verbose,
        resume=resume,
    )
    return await agent.run_orchestration()
