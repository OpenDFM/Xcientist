"""Workflow helpers for idea traces and fallback specs."""

import re
from typing import Any, Dict, List

from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract

def build_mcts_evolution(best_entry: Dict[str, Any]) -> Dict[str, Any]:
    trace = best_entry.get("search_trace") or []
    iterations: List[Dict[str, Any]] = []
    for hop in trace:
        if not isinstance(hop, dict):
            continue
        entry = {
            "iteration": hop.get("iteration"),
            "node_id": hop.get("node_id"),
            "depth": hop.get("depth"),
            "title": hop.get("title"),
            "operator": hop.get("operator"),
            "defects": hop.get("defects"),
            "score": hop.get("score"),
            "visits": hop.get("visits"),
            "path": hop.get("path"),
            "action_summary": hop.get("action_summary"),
        }
        evaluation = hop.get("evaluation")
        if evaluation is not None:
            entry["evaluation"] = evaluation
        memory_refs = hop.get("memory_refs")
        if memory_refs:
            entry["memory_refs"] = memory_refs
        rationale = hop.get("rationale")
        if rationale:
            entry["rationale"] = rationale
        signature = hop.get("signature")
        if signature:
            entry["signature"] = signature
        iterations.append(entry)
    evolution = {
        "best_path": best_entry.get("search_path"),
        "best_operator": best_entry.get("operator"),
        "target_defects": best_entry.get("target_defects"),
        "iterations": iterations,
    }
    pareto = best_entry.get("pareto_candidates")
    if pareto:
        evolution["pareto_front"] = pareto
    return evolution


def build_fusion_evolution(best_entry: Dict[str, Any]) -> Dict[str, Any]:
    fusion_metadata = best_entry.get("fusion_metadata")
    if not isinstance(fusion_metadata, dict):
        fusion_metadata = {}

    return {
        "source_modes": best_entry.get("source_modes") or [],
        "host_idea_mode": fusion_metadata.get("host_idea_mode"),
        "selected_components": fusion_metadata.get("selected_components") or [],
        "rejected_components": fusion_metadata.get("rejected_components") or [],
        "conflicts_and_resolutions": fusion_metadata.get("conflicts_and_resolutions") or [],
        "fused_core_thesis": fusion_metadata.get("fused_core_thesis") or "",
        "why_stronger_than_each_input": fusion_metadata.get("why_stronger_than_each_input") or "",
        "minimal_validation_plan": fusion_metadata.get("minimal_validation_plan") or "",
        "post_fusion_evaluation": build_mcts_evolution(best_entry),
    }


def collect_reference_material(reference_batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    seen_titles = set()
    for batch in reference_batches or []:
        for paper in batch:
            if not isinstance(paper, dict):
                continue
            title = (paper.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            entry = {
                "title": title,
                "authors": paper.get("authors") or [],
                "abstract": paper.get("abstract"),
                "tldr": paper.get("tldr"),
                "url": paper.get("url"),
                "year": paper.get("year"),
                "paper_id": paper.get("paper_id"),
                "source_keywords": paper.get("source_keywords"),
            }
            references.append(entry)
    return references


def derive_pipeline_steps(idea: Dict[str, Any]) -> List[str]:
    idea = normalize_idea_contract(idea, keep_extra=True)
    sections = [
        idea.get("method"),
        idea.get("experiments"),
        idea.get("abstract"),
        idea.get("core_contribution"),
    ]
    sentences: List[str] = []
    for section in sections:
        if not section:
            continue
        chunks = re.split(r"(?<=[.;])\s+", section)
        for chunk in chunks:
            cleaned = chunk.strip(" .;\n")
            if cleaned:
                sentences.append(cleaned)
            if len(sentences) >= 6:
                break
        if len(sentences) >= 3:
            break
    if not sentences:
        sentences = ["Outline the proposed method using available context."]
    return [f"Step {idx + 1}: {sentence}" for idx, sentence in enumerate(sentences)]


def fallback_algorithm_spec(idea: Dict[str, Any], inputs: List[str], outputs: List[str]) -> List[Dict[str, Any]]:
    pipeline = derive_pipeline_steps(idea)
    algorithm_entry = {
        "name": idea.get("title") or "Research Algorithm",
        "input": inputs,
        "output": outputs,
        "pipeline": pipeline,
    }
    return [algorithm_entry]
