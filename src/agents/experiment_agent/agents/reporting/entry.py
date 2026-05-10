"""
Claude Code-backed top-level ablation report integrator runner.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.agents.experiment_agent.agents.base.agent import BaseAgent
from src.agents.experiment_agent.agents.reporting.integrator import (
    EXPERIMENT_ABLATION_REPORT_INTEGRATOR,
    ablation_report_integrator_prompt,
)
from src.agents.experiment_agent.config import get_agent_model
from src.agents.experiment_agent.runtime.ablation_results import (
    build_ablation_final_artifact_contract,
    write_ablation_results_from_payload,
)
from src.agents.experiment_agent.runtime.idea_components import (
    canonical_component_names,
    find_idea_json_path,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, workspace_contract_paths


def _integrator_output_schema() -> Dict[str, Any]:
    component_fields = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "result": {"type": "string"},
            "metric": {"type": "string"},
            "value": {"type": "string"},
            "confidence": {"type": "number"},
            "analysis": {"type": "string"},
            "method_context": {"type": "string"},
        },
        "required": ["result", "metric", "value", "confidence", "analysis", "method_context"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "components": {
                        "type": "object",
                        "additionalProperties": component_fields,
                    },
                    "summary": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "feasible": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "key_findings": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["feasible", "confidence", "key_findings"],
                    },
                },
                "required": ["components", "summary"],
            },
            "source_evidence_files": {"type": "array", "items": {"type": "string"}},
            "integration_notes": {"type": "array", "items": {"type": "string"}},
            "blocker": {"type": "string"},
        },
        "required": ["payload", "source_evidence_files", "integration_notes", "blocker"],
    }


class AblationReportIntegratorAgent(BaseAgent):
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
            model=model or get_agent_model("ablation_report_integrator", "master"),
            verbose=verbose,
            workspace_root=workspace_root,
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

    def _build_user_prompt(self) -> str:
        return f"""Compose the final canonical ablation result payload.

Input paths:
- idea_json_path: {self.idea_json_path}
- agent_reports_dir: {self.contract['agent_reports_dir']}
- ablation_results_dir: {self.contract['ablation_results_dir']}

Canonical components:
- {', '.join(self.canonical_component_names)}

Requirements:
- Return `payload.components` keyed by the exact canonical component names in the same order.
- `method_context` must come from the original idea component description/explanation.
- If evidence is insufficient, set a non-empty `blocker` and still return your best structured payload candidate.
- The runtime will validate and decide whether to write `ablation_results.json`.
"""

    async def execute(self) -> Dict[str, object]:
        result = await self.run(
            user_prompt=self._build_user_prompt(),
            agent_name=EXPERIMENT_ABLATION_REPORT_INTEGRATOR,
            system_prompt=ablation_report_integrator_prompt(),
            output_schema=_integrator_output_schema(),
        )
        candidate = result["output"]
        materialized = write_ablation_results_from_payload(
            self.workspace_root,
            self.project_root,
            payload=candidate["payload"],
            generated_by="agent",
            source_evidence_files=candidate["source_evidence_files"],
            blocker=str(candidate.get("blocker") or "").strip() or None,
            ablation_results_path=self.ablation_json_path,
            integrator_report_path=self.integrator_report_path,
        )
        if materialized.get("report"):
            report_payload = dict(materialized["report"])
            report_payload["integration_notes"] = list(candidate.get("integration_notes") or [])
            with open(self.integrator_report_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, ensure_ascii=False, indent=2)
        return materialized


async def run_ablation_report_integrator(
    workspace_root: str,
    project_root: str,
    model: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, object]:
    agent = AblationReportIntegratorAgent(
        workspace_root=workspace_root,
        project_root=project_root,
        model=model,
        verbose=verbose,
        resume=resume,
    )
    return await agent.execute()
