"""Top-level LigAgent control flow and final idea persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.prompts import PROMPTS
from src.agents.idea_agent.utils.workflow.idea_helpers import build_mcts_evolution, collect_reference_material
from src.agents.idea_agent.utils.workflow.ligagent_helpers import (
    build_algorithm_spec,
    synthesize_reference_summaries,
)
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    collect_paper_context_entries,
    generate_idea_introduction,
)


def run_agent_loop(agent, logger) -> None:
    """Run the main planner loop for a LigAgent instance.

    The execution flow is deterministic:
    - If rag_hits is empty (no prior literature retrieval):
        knowledge_aquisition -> advanced_analysis -> idea_generation
    - If rag_hits is non-empty (literature already available):
        advanced_analysis -> re_analysis_replan -> idea_generation
    """
    rag_hits = agent.artifact.get("rag_hits", [])
    has_rag = bool(rag_hits and any(rag_hits))

    if has_rag:
        flow = ["advanced_analysis", "re_analysis_replan", "idea_generation"]
        logger.info("📋 rag_hits present — using flow: %s", " -> ".join(flow))
    else:
        flow = ["knowledge_aquisition", "advanced_analysis", "idea_generation"]
        logger.info("📋 rag_hits empty — using flow: %s", " -> ".join(flow))

    for turn, action in enumerate(flow):
        logger.info("========================================")
        logger.info("Turn %d: %s", turn + 1, action)
        agent.perform_action(action)


def persist_final_idea(
    best_entry: Dict[str, Any],
    paper_entries: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    idea_result_path: Path,
    chat_fn,
    model: str,
    logger,
    prompts: Optional[Dict[str, str]] = None,
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
    artifact["idea_result"] = payload
    try:
        idea_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(idea_result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved idea result to %s", idea_result_path)
    except OSError as exc:
        logger.error("⚠️ Failed to persist idea_result.json: %s", exc)
    return payload
