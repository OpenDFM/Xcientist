from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.agents.idea_agent.agent.artifacts import artifact_get
from src.agents.idea_agent.agent.prompts import PROMPTS
from src.agents.idea_agent.utils.core.config_loader import get_config_value
from src.agents.idea_agent.utils.core.logger import (
    get_or_create_mode_logger,
    suspend_console_handlers,
)
from src.agents.idea_agent.utils.mcts.idea_taste_presets import IDEA_TASTE_PRESETS
from src.agents.idea_agent.utils.prompting.prompt_views import format_paper_capsules_prompt_view
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract
from src.agents.idea_agent.utils.workflow.idea_fusion import (
    fuse_ligagent_pro_ideas,
    should_run_fusion,
)
from src.agents.idea_agent.utils.workflow.ligagent_flow import (
    make_stage_context,
    persist_final_idea,
)
from src.agents.idea_agent.utils.workflow.ligagent_helpers import (
    collect_analysis_background_lines,
    collect_rag_citations,
    collect_rag_contents,
    extract_experiment_findings_from_raw_ablation,
    extract_root_idea_from_analysis,
    filter_and_compress_papers,
    generate_rag_query,
    normalize_analysis_entry,
    normalize_search_papers,
    paper_context_with_rag,
    prepare_query_papers,
    retrieve_outcome_rag,
    safely_enrich_papers_with_content,
    search_papers_from_citations,
)
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    collect_paper_context_entries,
)
from src.agents.idea_agent.utils.workflow.stage_contract import (
    ArtifactPatch,
    StageContext,
    StageResult,
)


def _logger(agent: Any, ctx: StageContext) -> Any:
    return ctx.logger or getattr(agent, "logger", None)


def _runtime(agent: Any, ctx: StageContext) -> Any:
    return ctx.runtime or getattr(agent, "runtime", None)


def _session(agent: Any, ctx: StageContext) -> Any:
    return ctx.session or getattr(agent, "session", None)


def _chat(agent: Any, ctx: StageContext, op_name: str):
    runtime = _runtime(agent, ctx)
    session = _session(agent, ctx)
    stage = ctx.stage_name or ctx.workflow_name
    workflow_name = ctx.workflow_name

    def _invoke(prompt: str, **kwargs: Any) -> str:
        return runtime.llm_text(
            session=session,
            stage=stage,
            workflow_name=workflow_name,
            op_name=op_name,
            prompt=prompt,
            **kwargs,
        )

    return _invoke


def _result_to_best_entry(result: Any, idea_taste_mode: Optional[str]) -> Dict[str, Any]:
    best_payload = result.best.to_dict()
    best_entry = normalize_idea_contract(best_payload["idea"], keep_extra=True)
    best_entry["evaluation"] = best_payload["evaluation"]
    best_entry["search_score"] = best_payload["score"]
    best_entry["search_path"] = best_payload["path"]
    best_entry["pareto_candidates"] = {
        label: cand.to_dict() if cand else None for label, cand in result.pareto.items()
    }
    best_entry["search_trace"] = result.trace
    best_entry["idea_taste_mode"] = idea_taste_mode or "default"
    best_entry["idea_source"] = "raw_mode"
    best_entry["source_modes"] = [idea_taste_mode or "default"]
    return best_entry


def execute_knowledge_acquisition_stage(agent: Any, ctx: StageContext) -> StageResult:
    spec = agent._build_knowledge_acquisition_workflow()
    nested = agent.workflow_executor.run(
        spec,
        make_stage_context(agent, workflow_name=spec.name),
    )
    summary = nested.state.get("summary") or (
        "\nIn this knowledge_aquisition action, no explicit retrieval outcome was recorded."
    )
    return StageResult(
        status=nested.status,
        step_summary=summary,
        metrics={
            "mode": nested.state.get("mode"),
            "seed_papers": len(nested.state.get("initial_papers", [])),
            "rag_hits": len(nested.state.get("rag_hits", [])),
            "curated_papers": len(nested.state.get("curated_papers", [])),
        },
    )


def execute_advanced_analysis_stage(agent: Any, ctx: StageContext) -> StageResult:
    artifact = agent.artifact
    logger = _logger(agent, ctx)
    session = _session(agent, ctx)
    runtime = _runtime(agent, ctx)

    topic_history = artifact_get(artifact, "topic", [])
    topic = topic_history[-1] if topic_history else "unspecified topic"
    reference_batches = artifact_get(artifact, "references", [])
    references = reference_batches[-1] if reference_batches else []
    mature_idea = artifact_get(artifact, "mature_idea", "")
    rag_contents = artifact_get(artifact, "rag_contents", [])
    latest_rag_contents = rag_contents[-1] if rag_contents else []
    raw_ablation = (ctx.inputs or {}).get("raw_ablation")
    if raw_ablation is None:
        artifact_raw_ablation = artifact.get("ablation_results_raw")
        raw_ablation = artifact_raw_ablation if isinstance(artifact_raw_ablation, dict) else None

    experiment_findings = None
    if isinstance(raw_ablation, dict):
        extractor_cfg = get_config_value(
            agent.config,
            "agent.experiment_findings_extraction",
            {},
        )

        def _extractor_chat(prompt: str, **kwargs: Any) -> str:
            extractor_stage = str(kwargs.pop("stage", "") or "experiment_findings_extraction").strip()
            return runtime.llm_text(
                session=session,
                stage=extractor_stage,
                workflow_name=ctx.workflow_name,
                op_name="experiment_findings_extraction",
                prompt=prompt,
                **kwargs,
            )

        experiment_findings = extract_experiment_findings_from_raw_ablation(
            raw_ablation,
            chat_fn=_extractor_chat,
            logger=logger,
            extractor_config=extractor_cfg,
        )

    prompt = PROMPTS["advanced_analysis"].format(
        topic=topic,
        mature_idea=(mature_idea or "").strip(),
        survey_contents="\n".join(latest_rag_contents) if isinstance(latest_rag_contents, list) else "",
        papers=format_paper_capsules_prompt_view(references),
        experiment_findings=(
            json.dumps(experiment_findings, ensure_ascii=False, indent=2)
            if isinstance(experiment_findings, dict) and experiment_findings
            else "None"
        ),
    )
    response = normalize_analysis_entry(
        runtime.llm_json(
            session=session,
            stage=ctx.stage_name,
            workflow_name=ctx.workflow_name,
            op_name="advanced_analysis",
            prompt=prompt,
            model=agent.model,
            max_output_tokens=65536,
        )
    )
    if isinstance(response, (dict, list)):
        logger.info(
            "📝 Advanced Analysis Result:\n%s",
            json.dumps(response, ensure_ascii=False, indent=2),
        )
    else:
        logger.info("📝 Advanced Analysis Result:\n%s", response)

    if experiment_findings is not None:
        response["experiment_findings"] = experiment_findings
    root_idea = extract_root_idea_from_analysis(response, topic=topic)
    response["root_idea"] = root_idea
    existing_background = set(artifact_get(artifact, "background_knowledge", []))
    background_lines = [
        line
        for line in collect_analysis_background_lines(response)
        if line not in existing_background
    ]
    if session is not None:
        session.set_slot("analysis.latest", response)
    step = (
        "\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: "
        f"{response.get('tldr', 'No TL;DR provided.')}"
    )
    return StageResult(
        artifact_patch=ArtifactPatch(
            replace={"root_idea": root_idea},
            append={
                "analysis": [response],
                "background_knowledge": background_lines,
            }
        ),
        step_summary=step,
        metrics={
            "reference_count": len(references),
            "background_lines": len(background_lines),
            "root_idea_title": root_idea.get("title"),
            "experiment_findings_used": bool(experiment_findings),
        },
    )


def execute_idea_generation_stage(agent: Any, ctx: StageContext) -> StageResult:
    artifact = agent.artifact
    logger = _logger(agent, ctx)
    session = _session(agent, ctx)

    topic_history = artifact_get(artifact, "topic", [])
    topic = topic_history[-1]
    reference_batches = artifact_get(artifact, "references", [])
    latest_batch = reference_batches[-1] if reference_batches else []
    batch_list = [latest_batch] if latest_batch else reference_batches
    paper_entries = collect_paper_context_entries(artifact, batch_list)
    latest_candidate = artifact_get(artifact, "latest_candidate", {})
    latest_candidate_payload = (
        normalize_idea_contract(latest_candidate, allow_legacy=True, keep_extra=True)
        if isinstance(latest_candidate, dict) and latest_candidate
        else None
    )
    root_idea = artifact_get(artifact, "root_idea", {})
    root_idea_payload = (
        normalize_idea_contract(root_idea, allow_legacy=True, keep_extra=True)
        if isinstance(root_idea, dict) and root_idea
        else None
    )
    mature_idea = artifact_get(artifact, "mature_idea", "")
    context = {
        "analysis": artifact_get(artifact, "analysis", []),
        "latest_candidate": latest_candidate_payload,
        "root_idea": root_idea_payload,
        "background_knowledge": artifact_get(artifact, "background_knowledge", []),
        "paper_context": paper_context_with_rag(paper_entries, artifact),
    }
    if isinstance(mature_idea, str) and mature_idea.strip():
        context["mature_idea"] = mature_idea.strip()

    agent._inject_symbolic_priors(topic, context)
    ligagent_pro = bool(get_config_value(agent.config, "run.LigAgent-Pro", False))

    mode_results: List[tuple[str, Any]] = []
    shared_context = context
    if ligagent_pro:
        # Run MCTS searches for all idea taste modes in parallel, starting from the same prepared root context.
        agent.mcts.symbolic_memory_path.parent.mkdir(parents=True, exist_ok=True)
        agent.mcts.symbolic_memory.save(str(agent.mcts.symbolic_memory_path))
        shared_context = agent.mcts.prepare_root_context(topic, context)
        modes = list(IDEA_TASTE_PRESETS.keys())
        with ThreadPoolExecutor(max_workers=len(modes)) as executor:
            futures = {
                executor.submit(
                    agent.build_mcts_for_mode(mode).search,
                    topic,
                    deepcopy(shared_context),
                ): mode
                for mode in modes
            }
            for future in as_completed(futures):
                mode = futures[future]
                result = future.result()
                if result.best:
                    mode_results.append((mode, result))
    else:
        # Run a single MCTS search with the default idea taste preset.
        with suspend_console_handlers(logger):
            result = agent.mcts.search(topic=topic, context=context)
        if result.best:
            mode_label = getattr(getattr(agent.mcts, "idea_taste_preset", None), "mode", None) or "default"
            mode_results.append((mode_label, result))

    if not mode_results:
        logger.warning("⚠️ MCTS search returned no candidate; keeping latest candidate unchanged.")
        replace_patch = (
            {"latest_candidate": latest_candidate_payload}
            if latest_candidate_payload is not None
            else {}
        )
        return StageResult(
            status="degraded",
            artifact_patch=ArtifactPatch(replace=replace_patch),
            step_summary=(
                "\nIn this idea_generation action, MCTS returned no candidate "
                "and no fallback legacy path was used."
            ),
            metrics={"experience_count": 0},
        )

    mode_order = {mode: idx for idx, mode in enumerate(IDEA_TASTE_PRESETS.keys())}
    mode_entries = [
        _result_to_best_entry(result, mode)
        for mode, result in sorted(
            mode_results,
            key=lambda item: mode_order.get(item[0], len(mode_order)),
        )
    ]
    best_entry = mode_entries[0]
    fused_entry = None
    fusion_result: Dict[str, Any] = {}
    fusion_experiences: List[Any] = []
    fusion_logger = (
        get_or_create_mode_logger(agent._mode_loggers, logger, agent.run_dir, "fusion")
        if ligagent_pro
        else logger
    )
    if should_run_fusion(agent.config, ligagent_pro=ligagent_pro, candidate_count=len(mode_entries)):
        try:
            fused_entry, fusion_result, fusion_experiences = fuse_ligagent_pro_ideas(
                agent=agent,
                runtime=_runtime(agent, ctx),
                session=session,
                stage_name=ctx.stage_name,
                topic=topic,
                context=shared_context,
                mode_entries=mode_entries,
                prompt_template=PROMPTS["idea_fusion"],
                logger=fusion_logger,
            )
            if fused_entry:
                best_entry = fused_entry
        except Exception as exc:
            fusion_logger.warning("⚠️ Idea fusion failed: %s", exc)
            if fusion_logger is not logger:
                logger.warning("⚠️ Idea fusion failed; keeping raw best idea: %s", exc)
    if fusion_result:
        fusion_result["selected_entry_source"] = best_entry.get("idea_source")
        fusion_result["selected_title"] = best_entry.get("title")
    final_payload = persist_final_idea(
        best_entry=best_entry,
        paper_entries=paper_entries,
        artifact=artifact,
        idea_result_path=agent.idea_result_path,
        chat_fn=_chat(agent, ctx, "idea_materialization"),
        model=agent.model,
        logger=logger,
        prompts=PROMPTS,
        persist_to_artifact=False,
    )

    pareto_lines = []
    pareto_count = 0
    for mode, result in mode_results:
        for label, cand in result.pareto.items():
            if cand:
                pareto_count += 1
                pareto_lines.append(
                    f"{mode}/{label}: {cand.node.state.title} (score={cand.evaluation.composite:.2f})"
                )
    pareto_summary = "; ".join(pareto_lines) if pareto_lines else "no Pareto picks"
    all_experiences: List[Any] = []
    all_evaluations: List[Any] = []
    for _, result in mode_results:
        all_experiences.extend(result.experiences)
        all_evaluations.append(result.best.to_dict()["evaluation"])
    all_experiences.extend(fusion_experiences)
    if fused_entry:
        all_evaluations.append(fused_entry["evaluation"])
    if session is not None:
        session.set_slot("idea.latest", best_entry)
        session.set_slot("idea_result.latest", final_payload)
    step = (
        f"\nIn this idea_generation action, I ran memory-guided MCTS over '{topic}'. "
        f"{'All idea taste modes started from the same prepared root context. ' if ligagent_pro else ''}"
        f"{'A GPT-5.4 fusion pass was applied. ' if fused_entry else ''}"
        f"Best idea: {best_entry['title']} (score={best_entry['search_score']:.2f}). "
        f"Pareto set -> {pareto_summary}. Persisted {len(all_experiences)} defect->fix lifts to long-term memory."
    )
    return StageResult(
        artifact_patch=ArtifactPatch(
            replace={
                "latest_candidate": best_entry,
                "idea_result": final_payload,
                "ligagent_pro_candidates": mode_entries if ligagent_pro else [],
                "fusion_result": fusion_result,
            },
            append={
                "evaluations": all_evaluations,
                "ltm_experiences": all_experiences,
            },
        ),
        step_summary=step,
        metrics={
            "experience_count": len(all_experiences),
            "pareto_count": pareto_count,
            "search_score": best_entry["search_score"],
            "mode_count": len(mode_results),
            "fusion_used": bool(best_entry.get("idea_source") == "fused"),
        },
    )


def execute_reanalysis_replan_stage(agent: Any, ctx: StageContext) -> StageResult:
    artifact = agent.artifact
    session = _session(agent, ctx)
    runtime = _runtime(agent, ctx)

    analysis_entries = artifact_get(artifact, "analysis", [])
    analysis = analysis_entries[-1] if analysis_entries else {}
    ablation_results = artifact_get(artifact, "ablation_results", [])
    mature_idea = artifact_get(artifact, "mature_idea", "")
    topic = artifact_get(artifact, "topic", [])[-1]

    prompt = PROMPTS["re_analysis_replan"].format(
        topic=topic,
        mature_idea=mature_idea or "(no mature idea yet)",
        analysis=json.dumps(analysis, ensure_ascii=False, indent=2) if isinstance(analysis, dict) else str(analysis),
        ablation_results=json.dumps(ablation_results, ensure_ascii=False, indent=2) if ablation_results else "[]",
    )
    response = runtime.llm_json(
        session=session,
        stage=ctx.stage_name,
        workflow_name=ctx.workflow_name,
        op_name="re_analysis_replan",
        prompt=prompt,
        model=agent.model,
    )

    component_decisions = response.get("component_decisions", [])
    search_kw = response.get("search_keywords", "")

    replace_patch: Dict[str, Any] = {}
    if response.get("mature_idea"):
        replace_patch["mature_idea"] = response["mature_idea"]

    append_patch: Dict[str, List[Any]] = {}
    if component_decisions:
        append_patch["component_decisions"] = list(component_decisions)
    if search_kw:
        append_patch["retrieval_keywords"] = [search_kw]

    if session is not None and response.get("mature_idea"):
        session.set_slot("mature_idea.latest", response["mature_idea"])
    n_decisions = len(component_decisions)
    decision_summary = "; ".join(
        f"{d['component']}->{d['decision']}" for d in component_decisions if isinstance(d, dict)
    ) or "no component decisions"
    step = (
        f"\nIn this re_analysis_replan action, I made {n_decisions} component-level "
        f"modification(s) based on ablation evidence: [{decision_summary}]. "
        f"Updated mature idea for MCTS root node."
    )
    return StageResult(
        artifact_patch=ArtifactPatch(
            replace=replace_patch,
            append=append_patch,
        ),
        step_summary=step,
        metrics={
            "component_decisions": n_decisions,
            "updated_mature_idea": bool(response.get("mature_idea")),
            "updated_search_keywords": bool(search_kw),
        },
    )


def ka_route_stage(agent: Any, ctx: StageContext) -> StageResult:
    artifact = agent.artifact
    retrieval_keywords = artifact_get(artifact, "retrieval_keywords", [])
    search_keywords = retrieval_keywords[-1]
    topic_history = artifact_get(artifact, "topic", [])
    topic = topic_history[-1] if topic_history else search_keywords
    mature_idea = (artifact_get(artifact, "mature_idea", "") or "").strip()
    mode = "mature_idea" if mature_idea else "standard"
    return StageResult(
        state_patch={
            "mode": mode,
            "topic": topic,
            "search_keywords": search_keywords,
            "mature_idea": mature_idea,
            "initial_papers": [],
            "query_papers": [],
            "rag_hits": [],
            "survey_contents": [],
            "citation_titles": [],
            "rag_papers": [],
            "combined_papers": [],
            "curated_papers": [],
            "summary": "",
        },
        next_stage="ka_query_generation" if mode == "mature_idea" else "ka_seed_search",
        metrics={"mode": mode},
    )


def ka_seed_search_stage(agent: Any, ctx: StageContext) -> StageResult:
    logger = _logger(agent, ctx)
    runtime = _runtime(agent, ctx)
    session = _session(agent, ctx)
    search_keywords = ctx.state["search_keywords"]
    try:
        papers = runtime.tool_call(
            session=session,
            stage=ctx.stage_name,
            workflow_name=ctx.workflow_name,
            op_name="semantic_search",
            tool_name="semantic_search",
            query=search_keywords,
            limit=agent.semantic_search_limit,
        )
        logger.info("📄 Found Papers:")
        initial_papers = normalize_search_papers(papers, search_keywords, logger)
    except Exception as exc:
        logger.error("Error during paper search: %s", exc)
        return StageResult(
            status="retryable_failure",
            error=str(exc),
            metrics={"seed_papers": 0},
        )

    if not initial_papers:
        return StageResult(
            state_patch={
                "initial_papers": [],
                "summary": (
                    f"\nIn this knowledge_aquisition action, I searched for papers about "
                    f"'{search_keywords}' but found none."
                ),
            },
            metrics={"seed_papers": 0},
        )

    return StageResult(
        state_patch={"initial_papers": initial_papers},
        next_stage="ka_query_generation",
        metrics={"seed_papers": len(initial_papers)},
    )


def ka_seed_search_fallback_stage(agent: Any, ctx: StageContext) -> StageResult:
    search_keywords = ctx.state.get("search_keywords", "")
    return StageResult(
        status="degraded",
        state_patch={
            "summary": (
                f"\nIn this knowledge_aquisition action, I acquired several papers about "
                f"'{search_keywords}'."
            ),
            "fallback_used": True,
        },
        metrics={"fallback_used": True},
    )


def ka_query_generation_stage(agent: Any, ctx: StageContext) -> StageResult:
    logger = _logger(agent, ctx)

    mode = ctx.state["mode"]
    mature_idea = ctx.state.get("mature_idea", "")
    initial_papers = ctx.state.get("initial_papers", [])
    topic = ctx.state["topic"]
    search_keywords = ctx.state["search_keywords"]
    query_papers: List[Dict[str, Any]] = []
    query_topic = topic
    if mode != "mature_idea":
        query_papers = prepare_query_papers(
            initial_papers,
            agent.paper_repository,
            logger,
        )
        query_topic = search_keywords
    rag_query = generate_rag_query(
        query_topic,
        query_papers,
        PROMPTS,
        _chat(agent, ctx, "rag_query_generation"),
        agent.model,
        logger,
        mature_idea=mature_idea if mature_idea else None,
    )
    logger.info(
        "🔎 Generated RAG Query%s: %s",
        " (mature idea)" if mode == "mature_idea" else "",
        rag_query,
    )
    return StageResult(
        state_patch={
            "query_papers": query_papers,
            "rag_query": rag_query,
        },
        metrics={"query_papers": len(query_papers), "rag_query_length": len(rag_query)},
    )


def ka_outcome_rag_stage(agent: Any, ctx: StageContext) -> StageResult:
    session = _session(agent, ctx)
    rag_query = ctx.state["rag_query"]
    rag_hits = retrieve_outcome_rag(
        query=rag_query,
        top_k=5,
        paper_repository=agent.paper_repository,
        logger=_logger(agent, ctx),
    )
    survey_contents = collect_rag_contents(rag_hits)
    citation_titles = collect_rag_citations(rag_hits)
    if session is not None:
        session.set_slot("rag.latest", {"query": rag_query, "hits": rag_hits})
    return StageResult(
        artifact_patch=ArtifactPatch(
            append={
                "rag_query": [rag_query],
                "rag_hits": [{"query": rag_query, "hits": rag_hits}],
                "rag_contents": [survey_contents],
            }
        ),
        state_patch={
            "rag_hits": rag_hits,
            "survey_contents": survey_contents,
            "citation_titles": citation_titles,
        },
        metrics={
            "rag_hits": len(rag_hits),
            "citation_titles": len(citation_titles),
        },
    )


def ka_citation_expansion_stage(agent: Any, ctx: StageContext) -> StageResult:
    rag_query = ctx.state["rag_query"]
    rag_papers = search_papers_from_citations(
        ctx.state.get("citation_titles", []),
        rag_query,
        agent.paper_repository,
    )
    mode = ctx.state["mode"]
    initial_papers = list(ctx.state.get("initial_papers", []))
    rag_hits = ctx.state.get("rag_hits", [])

    if mode == "mature_idea" and not rag_papers:
        return StageResult(
            state_patch={
                "rag_papers": [],
                "combined_papers": [],
                "summary": (
                    f"\nIn this knowledge_aquisition action, I used the mature idea to generate "
                    f"a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
                    "but found no cited papers to fetch."
                ),
            },
            metrics={"rag_papers": 0},
        )

    combined_papers = list(rag_papers) if mode == "mature_idea" else initial_papers + list(rag_papers)
    return StageResult(
        state_patch={
            "rag_papers": rag_papers,
            "combined_papers": combined_papers,
        },
        metrics={
            "rag_papers": len(rag_papers),
            "combined_papers": len(combined_papers),
        },
    )


def ka_enrichment_stage(agent: Any, ctx: StageContext) -> StageResult:
    papers = ctx.state.get("combined_papers", [])
    if not papers:
        return StageResult(
            status="degraded",
            metrics={"combined_papers": 0},
        )

    temp_artifact = {
        "retrieval": {
            "paper_contents": deepcopy(artifact_get(agent.artifact, "paper_contents", {}))
        }
    }
    safely_enrich_papers_with_content(
        papers,
        agent.paper_enrichment_timeout,
        agent.paper_repository,
        temp_artifact,
        _logger(agent, ctx),
    )
    return StageResult(
        artifact_patch=ArtifactPatch(
            merge={"paper_contents": temp_artifact["retrieval"]["paper_contents"]}
        ),
        metrics={"combined_papers": len(papers)},
    )


def ka_paper_triage_stage(agent: Any, ctx: StageContext) -> StageResult:
    session = _session(agent, ctx)

    topic = ctx.state["topic"]
    mature_idea = ctx.state.get("mature_idea", "")
    combined_papers = ctx.state.get("combined_papers", [])
    mode = ctx.state["mode"]
    rag_query = ctx.state["rag_query"]
    rag_hits = ctx.state.get("rag_hits", [])
    initial_papers = ctx.state.get("initial_papers", [])

    temp_artifact = {
        "retrieval": {
            "paper_contents": deepcopy(artifact_get(agent.artifact, "paper_contents", {}))
        }
    }
    curated_papers = filter_and_compress_papers(
        topic=topic,
        mature_idea=mature_idea,
        papers=combined_papers,
        artifact=temp_artifact,
        prompts=PROMPTS,
        chat_fn=_chat(agent, ctx, "paper_triage"),
        model=agent.model,
        logger=_logger(agent, ctx),
        top_k=5,
    )

    if mode == "mature_idea":
        summary = (
            f"\nIn this knowledge_aquisition action, I used the mature idea to generate "
            f"a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
            f"and curated {len(curated_papers)} cited papers for memory."
        )
    elif ctx.state.get("rag_papers"):
        summary = (
            f"\nIn this knowledge_aquisition action, I read {len(initial_papers)} seed papers, "
            f"generated a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
            f"and curated {len(curated_papers)} papers for memory."
        )
    else:
        summary = (
            f"\nIn this knowledge_aquisition action, I searched for papers about "
            f"'{ctx.state['search_keywords']}' and curated {len(curated_papers)} relevant papers "
            "for my research."
        )

    if session is not None:
        session.set_slot("references.latest", curated_papers)
    return StageResult(
        artifact_patch=ArtifactPatch(
            append={"references": [curated_papers]},
            merge={"paper_contents": temp_artifact["retrieval"]["paper_contents"]},
        ),
        state_patch={
            "curated_papers": curated_papers,
            "summary": summary,
        },
        metrics={"curated_papers": len(curated_papers)},
    )
