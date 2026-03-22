from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.agents.idea_agent.agent.mcts import (
    MIN_COMPONENTS,
    IdeaNode,
    IdeaState,
    OperatorApplication,
)
from src.agents.idea_agent.agent.prompts.idea_fusion import (
    FUSION_REPAIR_INSTANTIATION_PROMPT,
    FUSION_REPAIR_PROMPT,
)
from src.agents.idea_agent.utils.core.config_loader import get_config_value
from src.agents.idea_agent.utils.mcts.mcts_helpers import (
    clip_text,
    component_inventory_payload,
    format_analysis_blob,
)
from src.agents.idea_agent.utils.mcts.mcts_runtime import (
    AtomicEditOp,
    ComponentEdit,
    EditPlan,
    MemoryBundle,
    ValidationProtocol,
    _estimate_budget_delta,
    apply_instantiated_mapping_to_plan,
    apply_edit_plan_to_components,
    build_root_state,
    format_op_descriptions,
    instantiate_compiled_plan_for_node,
    new_node,
    reset_search_state,
)
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract


PROMPT_CLIP_LIMIT = 65536
FUSION_REPAIR_SKILL_NAME = "fusion_repair"
FUSION_REPAIR_ALLOWED_OPS = {
    AtomicEditOp.REMOVE_COMPONENT,
    AtomicEditOp.REPLACE_COMPONENT,
    AtomicEditOp.REWIRE,
}
FUSION_DRAFT_MAX_ATTEMPTS = 5
FUSION_MATURE_IDEA_LIMIT = 4000
FUSION_ANALYSIS_LIMIT = 4000
FUSION_MINIMAL_RESPONSE_EXAMPLE = """{
  "fused_idea": {
    "title": "string",
    "abstract": "string",
    "core_contribution": "string",
    "method": "string",
    "risks": "string",
    "tags": ["string"],
    "operator": "fusion_agent",
    "target_defects": ["string"],
    "rationale": "string",
    "budget": {},
    "components": ["string"],
    "component_explanations": {
      "component_name": "string"
    },
    "root_domains": ["string"],
    "paper_graph_context": "string",
    "edit_plan": null,
    "skill_metrics": {}
  },
  "fusion_metadata": {
    "host_idea_mode": "string",
    "selected_components": [],
    "rejected_components": [],
    "conflicts_and_resolutions": [],
    "fused_core_thesis": "string",
    "why_stronger_than_each_input": "string",
    "minimal_validation_plan": "string"
  }
}"""


def _fusion_retry_guidance(validator_warning: str, validator_top_level_type: str) -> str:
    guidance = [
        "",
        "",
        "== Validator warning from previous response ==",
        validator_warning,
    ]
    if validator_top_level_type:
        guidance.extend(
            [
                "",
                f"Actual top-level JSON type from previous response: {validator_top_level_type}",
            ]
        )
    guidance.extend(
        [
            "",
            "Return STRICT JSON with a top-level object containing exactly `fused_idea` and `fusion_metadata`.",
            "A valid minimal response looks like:",
            FUSION_MINIMAL_RESPONSE_EXAMPLE,
        ]
    )
    return "\n".join(guidance)


def _summarize_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "None"
    summary = ". ".join(part.strip() for part in text.split(". ")[:3] if part.strip()).strip()
    if summary and summary[-1] not in ".!?":
        summary += "."
    return clip_text(summary or text, limit)


def _fusion_analysis_summary(context: Dict[str, Any]) -> str:
    analysis = context.get("analysis") or []
    if not analysis:
        return "No analysis available."
    latest = analysis[-1]
    lines: List[str] = []
    tldr = str(latest.get("tldr") or "").strip()
    if tldr:
        lines.append(f"TLDR: {tldr}")
    field_consensus = [str(item).strip() for item in (latest.get("field_consensus") or []) if str(item).strip()][:3]
    if field_consensus:
        lines.append("Constraints: " + "; ".join(field_consensus))
    existing_problems = [str(item).strip() for item in (latest.get("existing_problems") or []) if str(item).strip()][:3]
    if existing_problems:
        lines.append("Existing problems: " + "; ".join(existing_problems))
    evaluation_gaps = [
        str(item.get("gap") or "").strip()
        for item in (latest.get("evaluation_gaps") or [])
        if isinstance(item, dict) and str(item.get("gap") or "").strip()
    ][:2]
    if evaluation_gaps:
        lines.append("Evaluation gaps: " + "; ".join(evaluation_gaps))
    return clip_text("\n".join(lines) or "No analysis available.", FUSION_ANALYSIS_LIMIT)


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
    repair_max_steps = int(get_config_value(agent.config, "fusion.repair_max_steps", 10))
    repair_patience = int(get_config_value(agent.config, "fusion.repair_patience", 3))
    repair_epsilon = float(get_config_value(agent.config, "fusion.repair_epsilon", 0.1))    

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
        mature_idea=_summarize_text(context.get("mature_idea"), FUSION_MATURE_IDEA_LIMIT),
        root_domains=json.dumps(context.get("root_domains") or [], ensure_ascii=False),
        analysis=_fusion_analysis_summary(context),
        mode_count=len(mode_entries),
        candidate_ideas_json=json.dumps(candidate_pack, ensure_ascii=False, separators=(",", ":")),
    )
    payload: Dict[str, Any] = {}
    fused_raw: Dict[str, Any] = {}
    fused_entry: Dict[str, Any] = {}
    fusion_metadata: Dict[str, Any] = {}
    validator_warning = ""
    validator_top_level_type = ""
    last_exc: Optional[Exception] = None
    for attempt in range(1, FUSION_DRAFT_MAX_ATTEMPTS + 1):
        payload = {}
        current_payload_type = ""
        prompt_with_warning = prompt
        if validator_warning:
            prompt_with_warning += _fusion_retry_guidance(
                validator_warning,
                validator_top_level_type,
            )
        try:
            payload = runtime.llm_json(
                session=session,
                stage="idea_fusion",
                op_name="idea_fusion",
                prompt=prompt_with_warning,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if isinstance(payload, list):
                current_payload_type = "list"
                validator_top_level_type = "list"
                payload = payload[0] if payload else {}
            else:
                current_payload_type = type(payload).__name__
                validator_top_level_type = current_payload_type
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Fusion agent did not return a JSON object (got {type(payload).__name__})."
                )

            fused_raw = payload.get("fused_idea")
            if not isinstance(fused_raw, dict):
                available_keys = ", ".join(sorted(payload.keys())) if payload else "<empty>"
                raise ValueError(
                    "Fusion agent response is missing fused_idea. "
                    f"Top-level keys: {available_keys}"
                )
            fusion_metadata = payload.get("fusion_metadata")
            if not isinstance(fusion_metadata, dict):
                raise ValueError("Fusion agent response is missing fusion_metadata.")
            fused_entry = normalize_idea_contract(fused_raw, keep_extra=True)
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= FUSION_DRAFT_MAX_ATTEMPTS:
                raise
            validator_warning = str(exc)
            if current_payload_type:
                validator_top_level_type = current_payload_type
            logger.warning(
                "⚠️ Fusion draft validation failed (%d/%d): %s. Retrying...",
                attempt,
                FUSION_DRAFT_MAX_ATTEMPTS,
                exc,
            )
    if last_exc is not None and not fused_entry:
        raise last_exc

    fused_entry["idea_taste_mode"] = "fusion_agent"
    fused_entry["idea_source"] = "fused"
    fused_entry["source_modes"] = [entry.get("idea_taste_mode") for entry in mode_entries if entry.get("idea_taste_mode")]
    fused_entry["fusion_metadata"] = {
        "host_idea_mode": fusion_metadata.get("host_idea_mode", ""),
        "selected_components": fusion_metadata.get("selected_components", []),
        "rejected_components": fusion_metadata.get("rejected_components", []),
        "conflicts_and_resolutions": fusion_metadata.get("conflicts_and_resolutions", []),
        "fused_core_thesis": fusion_metadata.get("fused_core_thesis", ""),
        "why_stronger_than_each_input": fusion_metadata.get("why_stronger_than_each_input", ""),
        "minimal_validation_plan": fusion_metadata.get("minimal_validation_plan", ""),
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
                "host_idea_mode": fusion_metadata.get("host_idea_mode", ""),
                "fused_title": fused_entry.get("title"),
                "selected_components": fusion_metadata.get("selected_components", []),
                "rejected_components": fusion_metadata.get("rejected_components", []),
                "conflicts_and_resolutions": fusion_metadata.get("conflicts_and_resolutions", []),
                "fused_core_thesis": fusion_metadata.get("fused_core_thesis", ""),
                "why_stronger_than_each_input": fusion_metadata.get("why_stronger_than_each_input", ""),
                "minimal_validation_plan": fusion_metadata.get("minimal_validation_plan", ""),
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
    evaluated_entry, repair_trace, repair_experiences = repair_fused_entry(
        runtime=runtime,
        session=session,
        stage_name=stage_name,
        topic=topic,
        context=context,
        mode_entries=mode_entries,
        mcts=eval_mcts,
        entry=evaluated_entry,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_steps=repair_max_steps,
        patience=repair_patience,
        score_epsilon=repair_epsilon,
        logger=logger,
    )
    experiences.extend(repair_experiences)

    fusion_result = {
        "host_idea_mode": fusion_metadata.get("host_idea_mode", ""),
        "selected_components": fusion_metadata.get("selected_components", []),
        "rejected_components": fusion_metadata.get("rejected_components", []),
        "conflicts_and_resolutions": fusion_metadata.get("conflicts_and_resolutions", []),
        "fused_core_thesis": fusion_metadata.get("fused_core_thesis", ""),
        "why_stronger_than_each_input": fusion_metadata.get("why_stronger_than_each_input", ""),
        "minimal_validation_plan": fusion_metadata.get("minimal_validation_plan", ""),
        "fused_entry": evaluated_entry,
        "evaluation_mode": referee_mode or "default",
    }
    if repair_trace:
        fusion_result["repair_trace"] = repair_trace
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
    prepared = _prepare_mcts_context(mcts, topic, context)

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
    evaluated_entry["retrieved_core_titles"] = list(getattr(mcts, "retrieved_core_titles", []) or [])
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


def repair_fused_entry(
    *,
    runtime: Any,
    session: Any,
    stage_name: str,
    topic: str,
    context: Dict[str, Any],
    mode_entries: List[Dict[str, Any]],
    mcts: Any,
    entry: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    patience: int,
    score_epsilon: float,
    logger: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    best_entry = dict(entry)
    best_score = float(best_entry.get("search_score") or 0.0)
    repair_trace: List[Dict[str, Any]] = []
    repair_experiences: List[Dict[str, Any]] = []
    no_improve_steps = 0
    prepared = _prepare_mcts_context(mcts, topic, context)
    source_candidates = [_candidate_repair_payload(item) for item in mode_entries]

    logger.info(
        "🧬 Fusion repair start: score=%.2f max_steps=%d patience=%d",
        best_score,
        max_steps,
        patience,
    )

    progress = tqdm(total=max_steps, desc="Idea fusion repair", dynamic_ncols=True)
    try:
        for step_idx in range(1, max_steps + 1):
            progress.set_description(f"Idea fusion repair {step_idx}/{max_steps}")
            if no_improve_steps >= patience:
                logger.info(
                    "🧬 Fusion repair stop: no improvement for %d consecutive steps",
                    no_improve_steps,
                )
                break

            target_defects = _repair_target_defects(best_entry)
            parent_state = _entry_to_state(
                best_entry,
                fallback_root_domains=prepared.get("root_domains") or [],
                fallback_target_defects=target_defects,
            )
            bundle = _repair_memory_bundle(mcts, parent_state)
            payload = runtime.llm_json(
                session=session,
                stage=stage_name,
                op_name="idea_fusion_repair",
                prompt=FUSION_REPAIR_PROMPT.format(
                    topic=topic,
                    root_domains=json.dumps(prepared.get("root_domains") or [], ensure_ascii=False, indent=2),
                    current_idea_json=json.dumps(_candidate_prompt_payload(best_entry), ensure_ascii=False, indent=2),
                    current_evaluation_json=json.dumps(best_entry.get("evaluation") or {}, ensure_ascii=False, indent=2),
                    candidate_ideas_json=json.dumps(source_candidates, ensure_ascii=False, indent=2),
                    atomic_op_reference=format_op_descriptions(),
                ),
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            plan, stop_reason = _repair_plan_from_payload(
                payload,
                parent_state=parent_state,
                target_defects=target_defects,
            )
            if plan is None:
                if stop_reason:
                    logger.info("🧬 Fusion repair step %d stop: %s", step_idx, stop_reason)
                    break
                no_improve_steps += 1
                logger.info("🧬 Fusion repair step %d skipped: invalid local repair plan", step_idx)
                progress.update(1)
                continue

            logger.info(
                "🧬 Fusion repair step %d plan:\n%s",
                step_idx,
                json.dumps(_repair_plan_log_payload(plan), ensure_ascii=False, indent=2),
            )
            instantiated = instantiate_compiled_plan_for_node(
                mcts,
                plan,
                parent_state,
                bundle,
                prompt_template=FUSION_REPAIR_INSTANTIATION_PROMPT,
                plan_name="fusion_repair",
                root_domains_text=", ".join(parent_state.root_domains) if parent_state.root_domains else "Unspecified",
                stage="idea_fusion",
            )
            if isinstance(instantiated, dict) and instantiated.get("_skip_child_creation"):
                no_improve_steps += 1
                logger.info(
                    "🧬 Fusion repair step %d skipped: %s",
                    step_idx,
                    instantiated.get("_skip_reason", "child creation skipped"),
                )
                progress.update(1)
                continue

            apply_instantiated_mapping_to_plan(plan, instantiated, rewrite_components=False)
            candidate_state = mcts._materialize_child_state(
                parent_state,
                plan,
                instantiated,
                selection_metadata={"idea_taste_mode": "fusion_agent"},
            )
            candidate_entry = _preserve_fusion_fields(
                candidate_state.to_payload(),
                best_entry,
                repair_trace,
            )
            evaluated_candidate, step_experiences = evaluate_candidate_entry(
                mcts=mcts,
                topic=topic,
                context=context,
                entry=candidate_entry,
            )
            if evaluated_candidate is None:
                no_improve_steps += 1
                logger.info("🧬 Fusion repair step %d skipped: evaluation failed", step_idx)
                progress.update(1)
                continue

            next_score = float(evaluated_candidate.get("search_score") or 0.0)
            improved = next_score > (best_score + score_epsilon)
            logger.info(
                "🧬 Fusion repair step %d result: score=%.2f improved=%s",
                step_idx,
                next_score,
                improved,
            )
            progress.update(1)
            if not improved:
                no_improve_steps += 1
                continue

            trace_item = {
                "step": step_idx,
                "score_before": round(best_score, 4),
                "score_after": round(next_score, 4),
                "component_edits": [edit.to_dict() for edit in plan.component_edits],
            }
            repair_trace.append(trace_item)
            repair_experiences.extend(step_experiences)
            best_score = next_score
            best_entry = _preserve_fusion_fields(evaluated_candidate, best_entry, repair_trace)
            no_improve_steps = 0
    finally:
        progress.close()

    return best_entry, repair_trace, repair_experiences


def _candidate_prompt_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    components = entry.get("components") or []
    component_explanations = entry.get("component_explanations") or {}
    return {
        "idea_taste_mode": entry.get("idea_taste_mode"),
        "title": entry.get("title"),
        "core_contribution": entry.get("core_contribution"),
        "method": entry.get("method"),
        "scores": {
            "search_score": entry.get("search_score"),
            "novelty": _nested(entry, "evaluation", "novelty"),
            "impact": _nested(entry, "evaluation", "impact"),
            "feasibility": _nested(entry, "evaluation", "feasibility"),
        },
        "components": components,
        "component_explanations": {
            component: component_explanations.get(component, "")
            for component in components
        },
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


def _candidate_repair_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    components = entry.get("components") or []
    component_explanations = entry.get("component_explanations") or {}
    return {
        "idea_taste_mode": entry.get("idea_taste_mode"),
        "title": entry.get("title"),
        "search_score": entry.get("search_score"),
        "target_defects": entry.get("target_defects") or [],
        "components_with_explanations": component_inventory_payload(components, component_explanations),
    }


def _prepare_mcts_context(mcts: Any, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
    reset_search_state(mcts)
    prepared = mcts.prepare_root_context(topic, context)
    mcts.topic = topic
    mcts.analysis_blob = format_analysis_blob(prepared.get("analysis", []))
    mcts.paper_context = prepared.get("paper_context") or "No curated papers available yet."
    mcts.mature_idea = (prepared.get("mature_idea") or "").strip()
    mcts._mature_idea_components = list(prepared.get("components") or [])
    mcts._mature_idea_component_explanations = dict(prepared.get("component_explanations") or {})
    return prepared


def _repair_target_defects(entry: Dict[str, Any]) -> List[str]:
    evaluation = entry.get("evaluation") if isinstance(entry.get("evaluation"), dict) else {}
    detected = evaluation.get("detected_defects") if isinstance(evaluation, dict) else []
    defects = [str(tag).strip() for tag in (detected or []) if str(tag).strip()]
    if defects:
        return defects
    fallback = [str(tag).strip() for tag in (entry.get("target_defects") or []) if str(tag).strip()]
    return fallback or ["unclear_mechanism"]


def _repair_memory_bundle(mcts: Any, state: IdeaState) -> MemoryBundle:
    if not getattr(mcts, "enable_vector_memory", True):
        return MemoryBundle()
    return mcts.memory_accessor.retrieve_bundle(
        query=(
            f"{mcts.topic}\n{state.title}\n"
            f"{state.core_contribution}\n"
            f"defects={','.join(state.target_defects)}"
        )
    )


def _repair_plan_from_payload(
    payload: Any,
    *,
    parent_state: IdeaState,
    target_defects: List[str],
) -> Tuple[Optional[EditPlan], str]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return None, "repair planner did not return a JSON object"
    if payload.get("stop"):
        return None, str(payload.get("stop_reason") or "planner requested stop").strip()

    guardrails = [str(item).strip() for item in (payload.get("guardrails") or []) if str(item).strip()]
    plan_defects = [
        str(item).strip()
        for item in (payload.get("target_defects") or target_defects)
        if str(item).strip()
    ] or list(target_defects)
    validation = ValidationProtocol()
    component_edits: List[ComponentEdit] = []

    for raw_edit in payload.get("component_edits") or []:
        if not isinstance(raw_edit, dict):
            continue
        op_name = str(raw_edit.get("op") or "").strip().upper()
        if op_name not in {op.value for op in FUSION_REPAIR_ALLOWED_OPS}:
            continue
        component = str(raw_edit.get("component") or "").strip()
        target = str(raw_edit.get("target") or "").strip()
        condition = str(raw_edit.get("condition") or "").strip()
        details = str(raw_edit.get("details") or "").strip()
        reason = str(raw_edit.get("reason") or "").strip()

        if op_name == AtomicEditOp.REMOVE_COMPONENT.value and not component:
            continue
        if op_name == AtomicEditOp.REPLACE_COMPONENT.value and (not component or not target):
            continue
        if op_name == AtomicEditOp.REWIRE.value and (not component or not target):
            continue

        component_edits.append(
            ComponentEdit(
                op=AtomicEditOp(op_name),
                component=component,
                target=target,
                condition=condition,
                details=details,
                reason=reason,
            )
        )

    if not any(edit.op in {AtomicEditOp.REMOVE_COMPONENT, AtomicEditOp.REPLACE_COMPONENT, AtomicEditOp.REWIRE} for edit in component_edits):
        return None, ""

    plan = EditPlan(
        skill_name=FUSION_REPAIR_SKILL_NAME,
        objective="Locally repair the fused idea",
        target_defects=plan_defects,
        component_edits=component_edits,
        validation=validation,
        guardrails=guardrails,
        memory_refs=[],
        estimated_budget_delta=_estimate_budget_delta(component_edits),
        compile_notes="Compiled from fusion repair planner using existing atomic edit operations.",
    )
    next_components = apply_edit_plan_to_components(parent_state.components, plan)
    if len(next_components) < MIN_COMPONENTS:
        return None, ""
    if len(next_components) > len(parent_state.components):
        return None, ""
    if next_components == list(parent_state.components) and not any(edit.op == AtomicEditOp.REWIRE for edit in component_edits):
        return None, ""
    return plan, ""


def _preserve_fusion_fields(
    entry: Dict[str, Any],
    source_entry: Dict[str, Any],
    repair_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(entry)
    merged["idea_taste_mode"] = "fusion_agent"
    merged["idea_source"] = "fused"
    merged["source_modes"] = list(source_entry.get("source_modes") or [])
    fusion_metadata = dict(source_entry.get("fusion_metadata") or {})
    if repair_trace:
        fusion_metadata["repair_trace"] = list(repair_trace)
    merged["fusion_metadata"] = fusion_metadata
    return merged


def _repair_plan_log_payload(plan: EditPlan) -> Dict[str, Any]:
    payload = plan.to_dict()
    payload.pop("skill_name", None)
    return payload


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
        risks=str(entry.get("risks") or "").strip(),
        tags=list(entry.get("tags") or []),
        operator=str(entry.get("operator") or "fusion_agent").strip(),
        target_defects=list(entry.get("target_defects") or fallback_target_defects),
        rationale=str(entry.get("rationale") or "Fused from multiple idea taste modes.").strip(),
        memory_refs=[],
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
