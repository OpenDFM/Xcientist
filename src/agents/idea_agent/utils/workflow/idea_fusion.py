from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.agents.idea_agent.agent.mcts import IdeaNode, IdeaState, OperatorApplication
from src.agents.idea_agent.utils.core.config_loader import get_config_value
from src.agents.idea_agent.utils.mcts.mcts_helpers import (
    clip_text,
    component_inventory_payload,
    format_analysis_blob,
)
from src.agents.idea_agent.utils.mcts.mcts_runtime import (
    build_root_state,
    new_node,
    reset_search_state,
)
from src.agents.idea_agent.utils.prompting.prompt_views import format_idea_pool_prompt_view
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract


PROMPT_CLIP_LIMIT = 12000


def should_run_fusion(
    config: Any,
    *,
    ligagent_pro: bool,
    candidate_count: int,
) -> bool:
    enabled = bool(get_config_value(config, "fusion.enabled", True))
    only_when_ligagent_pro = bool(get_config_value(config, "fusion.only_when_ligagent_pro", True))
    min_candidates = max(2, int(get_config_value(config, "fusion.min_candidates", 2)))
    if not enabled:
        return False
    if only_when_ligagent_pro and not ligagent_pro:
        return False
    return candidate_count >= min_candidates


def fuse_ligagent_pro_ideas(
    *,
    agent: Any,
    runtime: Any,
    session: Any,
    stage_name: str,
    topic: str,
    context: Dict[str, Any],
    mode_entries: List[Dict[str, Any]],
    prompt_template: str,
    logger: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    model = str(get_config_value(agent.config, "fusion.model", "gpt-5.4"))
    temperature = float(get_config_value(agent.config, "fusion.temperature", 0.2))
    max_tokens = int(get_config_value(agent.config, "fusion.max_tokens", 65536))

    candidate_pack = [_candidate_prompt_payload(entry) for entry in mode_entries]
    logger.info(
        "🧬 Fusion start:\n%s",
        json.dumps(
            {
                "topic": topic,
                "model": model,
                "candidate_count": len(mode_entries),
                "candidates": [_candidate_log_payload(entry) for entry in mode_entries],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    prompt = prompt_template.format(
        topic=topic,
        mature_idea=clip_text(context.get("mature_idea") or "None", PROMPT_CLIP_LIMIT),
        root_domains=json.dumps(context.get("root_domains") or [], ensure_ascii=False, indent=2),
        analysis=clip_text(
            format_analysis_blob(context.get("analysis", [])) or "No analysis available.",
            PROMPT_CLIP_LIMIT,
        ),
        paper_context=clip_text(
            context.get("paper_context") or "No curated papers available yet.",
            PROMPT_CLIP_LIMIT,
        ),
        mode_count=len(mode_entries),
        candidate_ideas_json=json.dumps(candidate_pack, ensure_ascii=False, indent=2),
    )

    payload = runtime.llm_json(
        session=session,
        stage=stage_name,
        op_name="idea_fusion",
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("Fusion agent did not return a JSON object.")

    fused_raw = payload.get("fused_idea")
    if not isinstance(fused_raw, dict):
        raise ValueError("Fusion agent response is missing fused_idea.")

    fused_entry = normalize_idea_contract(fused_raw, keep_extra=True)
    fused_entry["idea_taste_mode"] = "fusion_agent"
    fused_entry["idea_source"] = "fused"
    fused_entry["source_modes"] = [entry.get("idea_taste_mode") for entry in mode_entries if entry.get("idea_taste_mode")]
    fused_entry["fusion_metadata"] = {
        "host_idea_mode": payload.get("host_idea_mode", ""),
        "selected_components": payload.get("selected_components", []),
        "rejected_components": payload.get("rejected_components", []),
        "conflicts_and_resolutions": payload.get("conflicts_and_resolutions", []),
        "fused_core_thesis": payload.get("fused_core_thesis", ""),
        "why_stronger_than_each_input": payload.get("why_stronger_than_each_input", ""),
        "minimal_validation_plan": payload.get("minimal_validation_plan", ""),
    }
    if not fused_entry.get("root_domains"):
        fused_entry["root_domains"] = list(context.get("root_domains") or [])
    fused_entry["components_with_explanations"] = component_inventory_payload(
        fused_entry.get("components") or [],
        fused_entry.get("component_explanations") or {},
    )
    logger.info(
        "🧬 Fusion draft:\n%s",
        json.dumps(
            {
                "host_idea_mode": payload.get("host_idea_mode", ""),
                "fused_title": fused_entry.get("title"),
                "selected_components": payload.get("selected_components", []),
                "rejected_components": payload.get("rejected_components", []),
                "conflicts_and_resolutions": payload.get("conflicts_and_resolutions", []),
                "fused_core_thesis": payload.get("fused_core_thesis", ""),
                "why_stronger_than_each_input": payload.get("why_stronger_than_each_input", ""),
                "minimal_validation_plan": payload.get("minimal_validation_plan", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    referee_mode = getattr(getattr(agent.mcts, "idea_taste_preset", None), "mode", None)
    eval_mcts = agent.build_mcts_for_mode(referee_mode)
    evaluated_entry, experiences = evaluate_candidate_entry(
        mcts=eval_mcts,
        topic=topic,
        context=context,
        entry=fused_entry,
    )
    if evaluated_entry is None:
        raise ValueError("Fusion candidate evaluation failed.")

    fusion_result = {
        "host_idea_mode": payload.get("host_idea_mode", ""),
        "selected_components": payload.get("selected_components", []),
        "rejected_components": payload.get("rejected_components", []),
        "conflicts_and_resolutions": payload.get("conflicts_and_resolutions", []),
        "fused_core_thesis": payload.get("fused_core_thesis", ""),
        "why_stronger_than_each_input": payload.get("why_stronger_than_each_input", ""),
        "minimal_validation_plan": payload.get("minimal_validation_plan", ""),
        "fused_entry": evaluated_entry,
        "evaluation_mode": referee_mode or "default",
    }
    logger.info(
        "🧬 Fusion candidate evaluated with %s: title=%s score=%.2f",
        fusion_result["evaluation_mode"],
        evaluated_entry.get("title"),
        evaluated_entry["search_score"],
    )
    return evaluated_entry, fusion_result, experiences


def evaluate_candidate_entry(
    *,
    mcts: Any,
    topic: str,
    context: Dict[str, Any],
    entry: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    reset_search_state(mcts)
    prepared = mcts.prepare_root_context(topic, context)
    mcts.topic = topic
    mcts.analysis_blob = format_analysis_blob(prepared.get("analysis", []))
    mcts.idea_pool_context = format_idea_pool_prompt_view(prepared.get("idea_pool") or [])
    mcts.paper_context = prepared.get("paper_context") or "No curated papers available yet."
    mcts.mature_idea = (prepared.get("mature_idea") or "").strip()
    mcts._mature_idea_components = list(prepared.get("components") or [])
    mcts._mature_idea_component_explanations = dict(prepared.get("component_explanations") or {})

    root_state = build_root_state(topic, prepared, IdeaState)
    root = new_node(
        root_state,
        depth=0,
        parent=None,
        signature_nodes=mcts.signature_nodes,
        id_iter=mcts._id_iter,
        idea_node_cls=IdeaNode,
        operator_application_cls=OperatorApplication,
    )
    candidate_state = _entry_to_state(
        entry,
        fallback_root_domains=prepared.get("root_domains") or [],
        fallback_target_defects=root.state.target_defects,
    )
    candidate = new_node(
        candidate_state,
        depth=1,
        parent=root,
        signature_nodes=mcts.signature_nodes,
        id_iter=mcts._id_iter,
        idea_node_cls=IdeaNode,
        operator_application_cls=OperatorApplication,
    )
    experiences: List[Dict[str, Any]] = []
    evaluation = mcts._simulate(candidate, [root, candidate], experiences)
    if evaluation is None:
        return None, experiences

    evaluated_entry = dict(entry)
    evaluated_entry.update(candidate.state.to_payload())
    evaluated_entry["components_with_explanations"] = component_inventory_payload(
        evaluated_entry.get("components") or [],
        evaluated_entry.get("component_explanations") or {},
    )
    evaluated_entry["evaluation"] = evaluation.to_dict()
    evaluated_entry["search_score"] = evaluation.composite
    evaluated_entry["search_path"] = candidate.path_summary()
    evaluated_entry["search_trace"] = [
        {
            "iteration": 0,
            "node_id": candidate.node_id,
            "depth": candidate.depth,
            "title": candidate.state.title,
            "operator": candidate.transformation.operator,
            "defects": list(candidate.transformation.defects),
            "memory_refs": list(candidate.transformation.memory_refs),
            "rationale": candidate.transformation.rationale,
            "score": evaluation.composite,
            "visits": 1,
            "path": candidate.path_summary(),
            "action_summary": (
                f"{candidate.transformation.operator} -> "
                f"{', '.join(candidate.transformation.defects) if candidate.transformation.defects else 'unspecified_defect'}"
            ),
            "evaluation": {
                **evaluation.to_dict(),
                "composite": evaluation.composite,
            },
            "signature": candidate.state.signature,
            "edit_plan": candidate.state.edit_plan,
            "skill_metrics": candidate.state.skill_metrics,
        }
    ]
    evaluated_entry["pareto_candidates"] = {}
    return evaluated_entry, experiences


def _candidate_prompt_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    components = entry.get("components") or []
    component_explanations = entry.get("component_explanations") or {}
    return {
        "idea_taste_mode": entry.get("idea_taste_mode"),
        "title": entry.get("title"),
        "core_contribution": entry.get("core_contribution"),
        "method": entry.get("method"),
        "evaluation": {
            "novelty": _nested(entry, "evaluation", "novelty"),
            "impact": _nested(entry, "evaluation", "impact"),
            "feasibility": _nested(entry, "evaluation", "feasibility"),
            "alignment_score": _nested(entry, "evaluation", "alignment_score"),
            "complexity_penalty": _nested(entry, "evaluation", "complexity_penalty"),
            "protocol_score": _nested(entry, "evaluation", "protocol_score"),
            "search_score": entry.get("search_score"),
        },
        "components": components,
        "components_with_explanations": component_inventory_payload(components, component_explanations),
        "edit_plan": entry.get("edit_plan"),
        "target_defects": entry.get("target_defects"),
    }


def _candidate_log_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "idea_taste_mode": entry.get("idea_taste_mode"),
        "title": entry.get("title"),
        "search_score": entry.get("search_score"),
        "operator": entry.get("operator"),
        "components": list(entry.get("components") or []),
    }


def _entry_to_state(
    entry: Dict[str, Any],
    *,
    fallback_root_domains: List[str],
    fallback_target_defects: List[str],
) -> IdeaState:
    return IdeaState(
        title=str(entry.get("title") or "").strip(),
        abstract=str(entry.get("abstract") or "").strip(),
        core_contribution=str(entry.get("core_contribution") or "").strip(),
        method=str(entry.get("method") or "").strip(),
        experiments=str(entry.get("experiments") or "").strip(),
        risks=str(entry.get("risks") or "").strip(),
        tags=list(entry.get("tags") or []),
        operator=str(entry.get("operator") or "fusion_agent").strip(),
        target_defects=list(entry.get("target_defects") or fallback_target_defects),
        rationale=str(entry.get("rationale") or "Fused from multiple idea taste modes.").strip(),
        memory_refs=list(entry.get("memory_refs") or []),
        budget=dict(entry.get("budget") or {}),
        components=list(entry.get("components") or []),
        component_explanations=dict(entry.get("component_explanations") or {}),
        root_domains=list(entry.get("root_domains") or fallback_root_domains),
        paper_graph_context=str(entry.get("paper_graph_context") or "").strip(),
        edit_plan=entry.get("edit_plan"),
        skill_metrics=dict(entry.get("skill_metrics") or {}),
    )


def _nested(payload: Dict[str, Any], parent_key: str, child_key: str) -> Any:
    parent = payload.get(parent_key)
    if not isinstance(parent, dict):
        return None
    return parent.get(child_key)
