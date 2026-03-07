"""Top-level LigAgent control flow and final idea persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.idea_agent.agent.prompts import PROMPTS
from src.agents.idea_agent.utils.workflow.idea_helpers import build_mcts_evolution, collect_reference_material
from src.agents.idea_agent.utils.workflow.stage_contract import StageContext
from src.agents.idea_agent.utils.workflow.workflow_runtime import (
    StageSpec,
    WorkflowEdge,
    WorkflowSpec,
)
from src.agents.idea_agent.utils.workflow.ligagent_helpers import (
    build_algorithm_spec,
    synthesize_reference_summaries,
)
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    collect_paper_context_entries,
    generate_idea_introduction,
)


def build_action_workflow(agent, action: str) -> WorkflowSpec:
    return WorkflowSpec(
        name=f"ligagent.action.{action}",
        entry_stage=action,
        stages=_build_stage_specs(agent),
    )


def build_main_workflow(agent, logger) -> WorkflowSpec:
    rag_hits = agent.artifact.get("rag_hits", [])
    has_rag = bool(rag_hits and any(rag_hits))

    if has_rag:
        flow = ["advanced_analysis", "re_analysis_replan", "idea_generation"]
        logger.info("📋 rag_hits present — using flow: %s", " -> ".join(flow))
        transitions = {
            "advanced_analysis": [WorkflowEdge("re_analysis_replan")],
            "re_analysis_replan": [WorkflowEdge("idea_generation")],
        }
        entry_stage = "advanced_analysis"
    else:
        flow = ["knowledge_aquisition", "advanced_analysis", "idea_generation"]
        logger.info("📋 rag_hits empty — using flow: %s", " -> ".join(flow))
        transitions = {
            "knowledge_aquisition": [WorkflowEdge("advanced_analysis")],
            "advanced_analysis": [WorkflowEdge("idea_generation")],
        }
        entry_stage = "knowledge_aquisition"

    # build main workflow spec with conditional transitions based on RAG hits
    return WorkflowSpec(
        name="ligagent.main",
        entry_stage=entry_stage,
        stages=_build_stage_specs(agent),
        transitions=transitions,
    )


def make_stage_context(agent, workflow_name: str, **inputs: Any) -> StageContext:
    return StageContext(
        agent=agent,
        artifact=agent.artifact,
        workflow_name=workflow_name,
        inputs=inputs,
        logger=getattr(agent, "logger", None),
    )


def _build_stage_specs(agent) -> Dict[str, StageSpec]:
    return {
        "knowledge_aquisition": StageSpec(
            name="knowledge_aquisition",
            handler=agent._execute_knowledge_acquisition_stage,
            description="Semantic Scholar seed -> RAG query -> OutcomeRAG -> citation expansion -> triage",
            record_step=True,
        ),
        "advanced_analysis": StageSpec(
            name="advanced_analysis",
            handler=agent._execute_advanced_analysis_stage,
            description="Summarize curated literature and derive analysis seeds",
            record_step=True,
        ),
        "idea_generation": StageSpec(
            name="idea_generation",
            handler=agent._execute_idea_generation_stage,
            description="Prepare context, run memory-guided MCTS, materialize and persist best idea",
            record_step=True,
        ),
        "re_analysis_replan": StageSpec(
            name="re_analysis_replan",
            handler=agent._execute_reanalysis_replan_stage,
            description="Revise mature idea and retrieval keywords using analysis/ablation evidence",
            record_step=True,
        ),
    }


def run_agent_loop(agent, logger) -> None:
    """Run the explicit top-level LigAgent workflow."""
    spec = build_main_workflow(agent, logger)
    agent.workflow_executor.run(
        spec,
        make_stage_context(agent, workflow_name=spec.name),
    )


def persist_final_idea(
    best_entry: Dict[str, Any],
    paper_entries: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    idea_result_path: Path,
    chat_fn,
    model: str,
    logger,
    prompts: Optional[Dict[str, str]] = None,
    persist_to_artifact: bool = True,
) -> Dict[str, Any]:
    prompts = prompts or PROMPTS
    topic = artifact["topic"][-1] if artifact.get("topic") else "unspecified topic"
    raw_refs = collect_reference_material(artifact.get("references", []))
    algorithm = build_algorithm_spec(
        best_entry,
        topic,
        raw_refs,
        artifact,
        prompts,
        chat_fn,
        model,
        logger,
    )
    references = synthesize_reference_summaries(
        topic,
        best_entry,
        algorithm,
        raw_refs,
        prompts,
        chat_fn,
        model,
        logger,
    )
    entries = paper_entries or collect_paper_context_entries(
        artifact, artifact.get("references", [])
    )
    introduction = generate_idea_introduction(
        chat_fn=chat_fn,
        prompt_template=prompts["idea_introduction"],
        model=model,
        topic=topic,
        best_entry=best_entry,
        paper_entries=entries,
        logger=logger,
    )
    component_entries = best_entry.get("components_with_explanations")
    if not isinstance(component_entries, list):
        raw_components = best_entry.get("components") or []
        raw_explanations = best_entry.get("component_explanations") or {}
        component_entries = []
        if isinstance(raw_components, list):
            for component in raw_components:
                name = str(component).strip()
                if not name:
                    continue
                explanation = ""
                if isinstance(raw_explanations, dict):
                    explanation = str(raw_explanations.get(name, "")).strip()
                component_entries.append(
                    {
                        "component": name,
                        "explanation": explanation,
                    }
                )
    payload = {
        "title": best_entry.get("title"),
        "abstract": best_entry.get("abstract"),
        "introduction": introduction,
        "components": component_entries,
        "algorithm": algorithm,
        "reference_papers": references,
        "mcts_evolution": build_mcts_evolution(best_entry),
    }
    if best_entry.get("idea_contract"):
        payload["idea_contract"] = best_entry.get("idea_contract")
    best_entry["introduction"] = introduction
    if persist_to_artifact:
        artifact["idea_result"] = payload
    try:
        idea_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(idea_result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved idea result to %s", idea_result_path)
    except OSError as exc:
        logger.error("⚠️ Failed to persist idea_result.json: %s", exc)
    return payload
