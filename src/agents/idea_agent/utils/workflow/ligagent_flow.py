"""Top-level LigAgent control flow and final idea persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.idea_agent.agent.artifacts import artifact_get, artifact_set
from src.agents.idea_agent.agent.prompts import PROMPTS
from src.agents.idea_agent.utils.core.json_utils import write_json_file
from src.agents.idea_agent.utils.workflow.idea_helpers import (
    build_fusion_evolution,
    build_mcts_evolution,
    collect_reference_material,
)
from src.agents.idea_agent.utils.workflow.stage_contract import StageContext
from src.agents.idea_agent.utils.workflow.workflow_runtime import (
    StageSpec,
    WorkflowEdge,
    WorkflowSpec,
)
from src.agents.idea_agent.utils.workflow.ligagent_helpers import (
    build_algorithm_spec,
    collect_rag_citations,
)
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    align_public_idea_entry,
    collect_paper_context_entries,
    generate_idea_introduction,
)


def build_main_workflow(agent, logger) -> WorkflowSpec:
    ablation_results = artifact_get(agent.artifact, "ablation_results", [])
    has_ablation = bool(ablation_results)

    if has_ablation:
        flow = ["advanced_analysis", "re_analysis_replan", "idea_generation"]
        logger.info("📋 ablation_results present — using flow: %s", " -> ".join(flow))
        transitions = {
            "advanced_analysis": [WorkflowEdge("re_analysis_replan")],
            "re_analysis_replan": [WorkflowEdge("idea_generation")],
        }
        entry_stage = "advanced_analysis"
    else:
        flow = ["knowledge_aquisition", "advanced_analysis", "idea_generation"]
        logger.info("📋 ablation_results empty — using flow: %s", " -> ".join(flow))
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
        session=getattr(agent, "session", None),
        runtime=getattr(agent, "runtime", None),
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
            allowed_artifact_namespaces={"retrieval"},
        ),
        "advanced_analysis": StageSpec(
            name="advanced_analysis",
            handler=agent._execute_advanced_analysis_stage,
            description="Diagnose survey gaps and derive a conservative 1.1 root idea",
            record_step=True,
            allowed_artifact_namespaces={"analysis", "run"},
        ),
        "idea_generation": StageSpec(
            name="idea_generation",
            handler=agent._execute_idea_generation_stage,
            description="Prepare context, run memory-guided MCTS, materialize and persist best idea",
            record_step=True,
            allowed_artifact_namespaces={"ideation", "persistence"},
        ),
        "re_analysis_replan": StageSpec(
            name="re_analysis_replan",
            handler=agent._execute_reanalysis_replan_stage,
            description="Apply minimal evidence-driven patches to the mature idea",
            record_step=True,
            allowed_artifact_namespaces={"run", "retrieval", "analysis"},
        ),
    }


def run_agent_loop(agent, logger) -> None:
    """Run the explicit top-level LigAgent workflow."""
    spec = build_main_workflow(agent, logger)
    agent.workflow_executor.run(
        spec,
        make_stage_context(agent, workflow_name=spec.name),
    )


def build_idea_result_payload(
    best_entry: Dict[str, Any],
    paper_entries: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    chat_fn,
    model: str,
    logger,
    prompts: Optional[Dict[str, str]] = None,
    mature_idea_override: Optional[str] = None,
    refinement_scope_override: Optional[str] = None,
) -> Dict[str, Any]:
    prompts = prompts or PROMPTS
    topic_history = artifact_get(artifact, "topic", [])
    topic = topic_history[-1] if topic_history else "unspecified topic"
    mature_idea = str(
        mature_idea_override
        if mature_idea_override is not None
        else artifact_get(artifact, "mature_idea", "")
    ).strip()
    refinement_scope = str(
        refinement_scope_override
        if refinement_scope_override is not None
        else artifact_get(artifact, "refinement_scope", "")
    ).strip()
    entries = paper_entries or collect_paper_context_entries(
        artifact, artifact_get(artifact, "references", [])
    )
    public_entry = align_public_idea_entry(
        chat_fn=chat_fn,
        prompt_template=prompts["idea_result_alignment"],
        model=model,
        topic=topic,
        best_entry=best_entry,
        mature_idea=mature_idea,
        refinement_scope=refinement_scope,
        paper_entries=entries,
        logger=logger,
    )
    algorithm = build_algorithm_spec(
        public_entry,
        topic,
        prompts,
        chat_fn,
        model,
        logger,
    )
    introduction = generate_idea_introduction(
        chat_fn=chat_fn,
        prompt_template=prompts["idea_introduction"],
        model=model,
        topic=topic,
        best_entry=public_entry,
        paper_entries=entries,
        mature_idea=mature_idea,
        logger=logger,
    )
    component_entries = public_entry.get("components_with_explanations")
    if not isinstance(component_entries, list):
        raw_components = public_entry.get("components") or []
        raw_explanations = public_entry.get("component_explanations") or {}
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
    reference_titles: List[str] = []
    seen_reference_titles = set()
    rag_entries = artifact_get(artifact, "rag_hits", [])
    if isinstance(rag_entries, list):
        for rag_entry in rag_entries:
            hits = []
            if isinstance(rag_entry, dict):
                hits = rag_entry.get("hits") or []
            elif isinstance(rag_entry, list):
                hits = rag_entry
            for title in collect_rag_citations(hits):
                if not title:
                    continue
                key = title.lower()
                if key in seen_reference_titles:
                    continue
                seen_reference_titles.add(key)
                reference_titles.append(title)
    for reference in collect_reference_material(artifact_get(artifact, "references", [])):
        if not isinstance(reference, dict):
            continue
        title = str(reference.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen_reference_titles:
            continue
        seen_reference_titles.add(key)
        reference_titles.append(title)
    for title in best_entry.get("retrieved_core_titles") or []:
        cleaned = str(title or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen_reference_titles:
            continue
        seen_reference_titles.add(key)
        reference_titles.append(cleaned)
    mcts_evolution = build_mcts_evolution(best_entry)
    payload = {
        "title": public_entry.get("title"),
        "abstract": public_entry.get("abstract"),
        "method": public_entry.get("method"),
        "introduction": introduction,
        "components": component_entries,
        "algorithm": algorithm,
        "reference_papers": reference_titles,
        "mcts_evolution": mcts_evolution,
    }
    if best_entry.get("idea_source"):
        payload["idea_source"] = best_entry.get("idea_source")
    if isinstance(best_entry.get("source_modes"), list) and best_entry.get("source_modes"):
        payload["source_modes"] = best_entry.get("source_modes")
    if isinstance(best_entry.get("fusion_metadata"), dict) and best_entry.get("fusion_metadata"):
        payload["fusion_metadata"] = best_entry.get("fusion_metadata")
    if best_entry.get("idea_source") == "fused":
        payload["fusion_evolution"] = build_fusion_evolution(best_entry)
    if best_entry.get("idea_contract"):
        payload["idea_contract"] = best_entry.get("idea_contract")
    for key in ("title", "abstract", "core_contribution", "method", "risks"):
        if public_entry.get(key):
            best_entry[key] = public_entry[key]
    best_entry["introduction"] = introduction
    return payload


def save_idea_result_payload(
    payload: Dict[str, Any],
    idea_result_path: Path,
    logger,
) -> None:
    try:
        idea_result_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(idea_result_path, payload)
        logger.info("💾 Saved idea result to %s", idea_result_path)
    except OSError as exc:
        logger.error("⚠️ Failed to persist idea_result.json: %s", exc)


def save_candidate_payload(
    payload: Dict[str, Any],
    candidate_path: Path,
    logger,
) -> None:
    try:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(candidate_path, payload)
        logger.info("💾 Saved idea candidate to %s", candidate_path)
    except OSError as exc:
        logger.error("⚠️ Failed to persist idea_candidate.json: %s", exc)


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
    mature_idea_override: Optional[str] = None,
    refinement_scope_override: Optional[str] = None,
) -> Dict[str, Any]:
    payload = build_idea_result_payload(
        best_entry,
        paper_entries=paper_entries,
        artifact=artifact,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        prompts=prompts,
        mature_idea_override=mature_idea_override,
        refinement_scope_override=refinement_scope_override,
    )
    if persist_to_artifact:
        artifact_set(artifact, "idea_result", payload)
    save_idea_result_payload(payload, idea_result_path, logger)
    return payload
