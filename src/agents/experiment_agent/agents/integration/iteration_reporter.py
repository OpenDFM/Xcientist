"""
Deterministic iteration integration reporter.
"""

from __future__ import annotations

from typing import Dict

from src.agents.experiment_agent.runtime.manifests import artifact_paths, load_json_file, write_json_file
from src.agents.experiment_agent.runtime.phase_contracts import normalize_phase_report


ITERATION_REPORTER = "experiment_iteration_reporter"


def _phase_state(paths: Dict[str, str], key: str) -> Dict[str, object]:
    payload = load_json_file(paths[key])
    normalized = normalize_phase_report(payload)
    return {
        "phase_completion_status": normalized["phase_completion_status"],
        "ready_for_next_phase": normalized["ready_for_next_phase"],
        "artifact_role": normalized["artifact_role"],
        "run_level": normalized["run_level"],
        "blocking_issues": normalized["blocking_issues"],
    }


class IterationReporterAgent:
    def __init__(
        self,
        workspace_root: str,
        project_root: str,
        model: str | None = None,
        verbose: bool = True,
        resume: bool = False,
    ):
        _ = model, verbose, resume
        self.workspace_root = workspace_root
        self.project_root = project_root

    async def execute(self) -> Dict[str, object]:
        paths = artifact_paths(self.workspace_root, self.project_root)
        phase_states = {
            "code": _phase_state(paths, "code_validator"),
            "standard_science": _phase_state(paths, "standard_science_validator"),
            "ablation_science": _phase_state(paths, "ablation_science_validator"),
        }
        payload = {
            "iteration": 0,
            "code_status": "complete" if phase_states["code"]["phase_completion_status"] == "complete" else "incomplete",
            "code_evidence": [paths["code_validator"]],
            "standard_experiments": "complete" if phase_states["standard_science"]["phase_completion_status"] == "complete" else "partial",
            "standard_evidence": [paths["standard_science_validator"]],
            "ablation_experiments": "complete" if phase_states["ablation_science"]["phase_completion_status"] == "complete" else "partial",
            "ablation_evidence": [paths["ablation_science_validator"]],
            "validation_status": "pass",
            "phase_states": phase_states,
            "key_findings": [],
            "blockers": phase_states["code"]["blocking_issues"] + phase_states["standard_science"]["blocking_issues"] + phase_states["ablation_science"]["blocking_issues"],
            "next_recommendations": [],
        }
        write_json_file(paths["iteration_status"], payload)
        with open(paths["iteration_summary"], "w", encoding="utf-8") as f:
            f.write("# Iteration Summary\n\n")
            for name, state in phase_states.items():
                f.write(f"- {name}: {state['phase_completion_status']}\n")
        return {
            "iteration_summary_path": paths["iteration_summary"],
            "iteration_status_path": paths["iteration_status"],
            "valid": True,
            "output": "iteration status written",
        }


async def run_iteration_reporter(
    workspace_root: str,
    project_root: str,
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, object]:
    agent = IterationReporterAgent(
        workspace_root=workspace_root,
        project_root=project_root,
        model=model,
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
