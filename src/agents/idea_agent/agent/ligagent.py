from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.agent import get_logger

from typing import Any, Dict, Literal, List, Optional, Set
from pathlib import Path
from copy import deepcopy
import time
import json
from dataclasses import fields

from src.agents.idea_agent.agent.tools import TOOLS
from src.agents.idea_agent.agent.artifacts import artifact_init
from src.agents.idea_agent.agent.prompts import PROMPTS
from src.agents.idea_agent.agent.mcts import (
    MemoryGuidedMCTS,
    MCTSConfig,
    apply_idea_taste_preset,
)
from src.agents.idea_agent.utils.papers.paper_repository import PaperRepository
from src.agents.idea_agent.utils.workflow.ligagent_flow import (
    build_action_workflow,
    make_stage_context,
    persist_final_idea,
)
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    collect_paper_context_entries,
    generate_idea_introduction,
    parse_json_response,
)
from src.agents.idea_agent.utils.workflow.ligagent_helpers import (
    build_action_lookup,
    generate_background_brief,
    normalize_search_papers,
    prepare_query_papers,
    generate_rag_query,
    retrieve_outcome_rag,
    collect_rag_citations,
    collect_rag_contents,
    search_papers_from_citations,
    safely_enrich_papers_with_content,
    filter_and_compress_papers,
    paper_context_with_rag,
    normalize_analysis_entry,
    collect_analysis_background_lines,
    latest_analysis_seed_ideas,
    sanitize_action_token,
    get_paper_content as load_paper_content,
)
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract
from src.agents.idea_agent.utils.workflow.stage_contract import (
    ArtifactPatch,
    StageContext,
    StageResult,
)
from src.agents.idea_agent.utils.workflow.workflow_runtime import (
    StageSpec,
    WorkflowEdge,
    WorkflowExecutor,
    WorkflowSpec,
)
from src.agents.idea_agent.utils.core.config_loader import get_config_value
from src.agents.idea_agent.utils.prompting.prompt_views import format_paper_capsules_prompt_view
from memory.api.component_taxonomy import (
    ContextSignature,
    extract_component_families,
    extract_context_signature,
)

logger = get_logger()

class LigAgent(AgentBase):
    ACTION_ALIASES: Dict[str, Set[str]] = {
        "knowledge_aquisition": {
            "knowledge_aquisition",
            "knowledge aquisition",
            "knowledge acquisition",
            "knowledgeaquisition",
            "knowledgeacquisition",
            "knowledge-acquisition",
        },
        "advanced_analysis": {
            "advanced_analysis",
            "advanced analysis",
            "advancedanalysis",
            "advanced-analysis",
        },
        "idea_generation": {
            "idea_generation",
            "idea generation",
            "ideageneration",
            "idea-generation",
        },
        "re_analysis_replan": {
            "re_analysis_replan",
            "re analysis replan",
            "reanalysisreplan",
            "re-analysis-replan",
        },
    }

    def _canonical_action(self, action: Optional[str]) -> Optional[str]:
        if not action:
            return None
        sanitized = sanitize_action_token(action)
        return self._action_lookup.get(sanitized)

    def __init__(self, *args, **kwargs):
        # Configure from provided config or load from file
        config = kwargs.pop("config", None)
        self.config = config

        chat_max_retries = get_config_value(config, "agent.chat_max_retries", 3)
        chat_retry_backoff = get_config_value(config, "agent.chat_retry_backoff", 2.0)
        survey_config_path = kwargs.pop("survey_config_path", None)
        run_dir = kwargs.pop("run_dir", None)
        rag_config = kwargs.pop("rag_config", None)
        action_selection_attempts = kwargs.pop(
            "action_selection_attempts",
            get_config_value(config, "agent.action_selection_attempts", 2),
        )
        model = kwargs.pop("model", None)
        super().__init__(*args, **kwargs)
        self.action_space = [
            "knowledge_aquisition",
            "advanced_analysis",
            "idea_generation",
            "re_analysis_replan",
        ]
        self.action_selection_attempts = max(1, action_selection_attempts)
        if model is None:
            model = get_config_value(config, "agent.model", "gpt-4.1")
        self.model = model
        self.tools = TOOLS
        self.artifact = artifact_init()

        # Result persistence paths
        self.run_dir = Path(run_dir) if run_dir else Path(__file__).resolve().parent.parent
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.idea_result_path = self.run_dir / "idea_result.json"
        self.idea_result_path.parent.mkdir(parents=True, exist_ok=True)

        self.chat_max_retries = chat_max_retries
        self.chat_retry_backoff = chat_retry_backoff
        self._action_lookup = build_action_lookup(self.ACTION_ALIASES)
        self.semantic_search_limit = get_config_value(config, "agent.semantic_search_limit", 5)
        self.idea_context_limit = get_config_value(config, "agent.idea_context_limit", 10)
        self.logger = logger
        self.workflow_executor = WorkflowExecutor(logger=logger)

        # Initialize mature_idea in artifact from config (user-provided) if available
        initial_mature_idea = get_config_value(config, "run.mature_idea", "")
        if isinstance(initial_mature_idea, str) and initial_mature_idea.strip():
            self.artifact["mature_idea"] = initial_mature_idea.strip()

        mcts_config = MCTSConfig()
        for field in fields(MCTSConfig):
            override = get_config_value(config, f"mcts.{field.name}", None)
            if override is not None:
                setattr(mcts_config, field.name, override)
        idea_taste_preset = apply_idea_taste_preset(mcts_config)
        if idea_taste_preset:
            self.artifact["idea_taste_mode"] = idea_taste_preset.mode
            self.artifact["idea_taste_label"] = idea_taste_preset.label
            logger.info(
                "[LigAgent] Applied idea taste preset %s (%s).",
                idea_taste_preset.mode,
                idea_taste_preset.label,
            )
        self.mcts = MemoryGuidedMCTS(
            chat_fn=self.chat,
            evaluation_prompt=PROMPTS.get("mcts_evaluation"),
            config=mcts_config,
            logger=logger,
        )

        self.paper_enrichment_timeout = kwargs.pop(
            "paper_enrichment_timeout",
            get_config_value(config, "agent.paper_enrichment_timeout_sec", 7200),
        )
        self.paper_repository = PaperRepository(
            config_path=survey_config_path,
            logger=logger,
            rag_config=rag_config,
        )

    def chat(self, prompt: str, model: str = "gpt-5-mini", **kwargs) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.chat_max_retries + 1):
            try:
                # Special handling for GPT-5 models
                if "gpt-5-mini" in model:
                    # Idea Generator: GPT-5 mini
                    kwargs["temperature"] = 1.0
                    return super().chat(prompt, model=model, reasoning={"effort": "high"}, **kwargs)
                elif "gpt-5" in model:
                    # Idea Evaluator: GPT-5.4
                    kwargs["temperature"] = 1.0
                    return super().chat(prompt, model=model, reasoning={"effort": "low"}, **kwargs)
                else:
                    return super().chat(prompt, model=model, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = self.chat_retry_backoff ** (attempt - 1)
                logger.warning(
                    "⚠️ Chat attempt %d/%d failed (%s). Retrying in %.2fs...",
                    attempt,
                    self.chat_max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc if last_exc else RuntimeError("Chat failed without exception detail.")

    def perform_action(self, action: str, **kwargs) -> Any:
        logger.info(f"🚀 Performing action: {action}...")
        resolved_action = self._canonical_action(action)
        if not resolved_action:
            raise ValueError(f"Unknown action: {action}")
        if resolved_action == "knowledge_aquisition":
            logger.info("🔍 Due to API and web request rate limits, this process may take some time...")
        return self._run_action_workflow(resolved_action, **kwargs)

    def bootstrap_topic(self, topic: str, retrieval_keywords: Optional[str] = None) -> None:
        normalized_topic = (topic or "").strip()
        if not normalized_topic:
            raise ValueError("Topic must be a non-empty string.")
        keywords = (
            retrieval_keywords.strip()
            if isinstance(retrieval_keywords, str)
            else normalized_topic
        )
        if not self.artifact.get("run_topic"):
            self.artifact["run_topic"] = normalized_topic
        self.artifact["topic"].append(normalized_topic)
        self.artifact["retrieval_keywords"].append(keywords)
        background = generate_background_brief(
            normalized_topic,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        if background:
            self.artifact["background_knowledge"].append(background)

    def knowledge_aquisition(
        self, search_type: Literal["paper_search", "website"] = "paper_search"
    ) -> str:
        return self._run_action_workflow("knowledge_aquisition", search_type=search_type)

    def get_paper_content(self, paper_id: str, include_markdown: bool = True) -> Dict[str, Any]:
        return load_paper_content(
            paper_id,
            include_markdown,
            self.artifact,
            self.paper_repository,
            logger,
        )

    def advanced_analysis(self, **kwargs) -> str:
        return self._run_action_workflow("advanced_analysis", **kwargs)

    def idea_generation(self, **kwargs) -> str:
        return self._run_action_workflow("idea_generation", **kwargs)

    def re_analysis_replan(self, **kwargs) -> str:
        return self._run_action_workflow("re_analysis_replan", **kwargs)

    def _persist_final_idea(
        self, best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
    ) -> None:
        persist_final_idea(
            best_entry=best_entry,
            paper_entries=paper_entries,
            artifact=self.artifact,
            idea_result_path=self.idea_result_path,
            chat_fn=self.chat,
            model=self.model,
            logger=logger,
            prompts=PROMPTS,
        )

    def _run_action_workflow(self, action: str, **kwargs) -> str:
        spec = build_action_workflow(self, action)
        before_steps = len(self.artifact.get("steps", []))
        self.workflow_executor.run(
            spec,
            make_stage_context(self, workflow_name=spec.name, **kwargs),
        )
        new_steps = self.artifact.get("steps", [])[before_steps:]
        if new_steps:
            logger.info(new_steps[-1])
            return new_steps[-1]
        return ""

    def _execute_knowledge_acquisition_stage(self, ctx: StageContext) -> StageResult:
        search_type = ctx.inputs.get("search_type", "paper_search")
        if search_type != "paper_search":
            return StageResult(
                status="terminal_failure",
                error=f"Unsupported knowledge acquisition search_type '{search_type}'.",
            )

        spec = self._build_knowledge_acquisition_workflow()
        nested = self.workflow_executor.run(
            spec,
            make_stage_context(self, workflow_name=spec.name, search_type=search_type),
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

    def _execute_advanced_analysis_stage(self, ctx: StageContext) -> StageResult:
        topic = self.artifact["topic"][-1] if self.artifact["topic"] else "unspecified topic"
        references = self.artifact["references"][-1] if self.artifact["references"] else []
        mature_idea = self.artifact.get("mature_idea", "")
        prompt = PROMPTS["advanced_analysis"].format(
            topic=topic,
            mature_idea=(mature_idea or "").strip(),
            survey_contents="\n".join(self.artifact["rag_contents"][-1]) if self.artifact.get("rag_contents") else "",
            papers=format_paper_capsules_prompt_view(references),
        )
        raw_response = self._parse_json_response(
            self.chat(prompt, model=self.model, max_output_tokens=8192)
        )
        response = normalize_analysis_entry(raw_response)
        if isinstance(response, (dict, list)):
            logger.info(
                "📝 Advanced Analysis Result:\n%s",
                json.dumps(response, ensure_ascii=False, indent=2),
            )
        else:
            logger.info("📝 Advanced Analysis Result:\n%s", response)

        existing_background = set(self.artifact.get("background_knowledge", []))
        background_lines = [
            line
            for line in collect_analysis_background_lines(response)
            if line not in existing_background
        ]
        step = (
            "\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: "
            f"{response.get('tldr', 'No TL;DR provided.')}"
        )
        return StageResult(
            artifact_patch=ArtifactPatch(
                append={
                    "analysis": [response],
                    "background_knowledge": background_lines,
                }
            ),
            step_summary=step,
            metrics={
                "reference_count": len(references),
                "background_lines": len(background_lines),
            },
        )

    def _execute_idea_generation_stage(self, ctx: StageContext) -> StageResult:
        topic = self.artifact["topic"][-1]
        reference_batches = self.artifact.get("references", [])
        latest_batch = reference_batches[-1] if reference_batches else []
        batch_list = [latest_batch] if latest_batch else reference_batches
        paper_entries = collect_paper_context_entries(
            self.artifact,
            batch_list,
        )
        idea_history = [
            normalize_idea_contract(entry, allow_legacy=True, keep_extra=True)
            for entry in self.artifact.get("idea_pool", [])
        ]
        seed_ideas = latest_analysis_seed_ideas(self.artifact)
        idea_context = idea_history if idea_history else seed_ideas
        mature_idea = self.artifact.get("mature_idea", "")
        context = {
            "analysis": self.artifact.get("analysis", []),
            "idea_pool": idea_context,
            "background_knowledge": self.artifact.get("background_knowledge", []),
            "paper_context": paper_context_with_rag(paper_entries, self.artifact),
        }
        if isinstance(mature_idea, str) and mature_idea.strip():
            context["mature_idea"] = mature_idea.strip()

        self._inject_symbolic_priors(topic, context)
        result = self.mcts.search(topic=topic, context=context)

        if not result.best:
            logger.warning("⚠️ MCTS search returned no candidate; keeping current idea pool unchanged.")
            replace_patch = {"idea_pool": idea_history} if idea_history else {}
            return StageResult(
                status="degraded",
                artifact_patch=ArtifactPatch(replace=replace_patch),
                step_summary=(
                    "\nIn this idea_generation action, MCTS returned no candidate "
                    "and no fallback legacy path was used."
                ),
                metrics={"experience_count": len(result.experiences)},
            )

        best_payload = result.best.to_dict()
        best_entry = normalize_idea_contract(best_payload["idea"], keep_extra=True)
        best_entry["evaluation"] = best_payload["evaluation"]
        best_entry["search_score"] = best_payload["score"]
        best_entry["search_path"] = best_payload["path"]
        best_entry["pareto_candidates"] = {
            label: cand.to_dict() if cand else None for label, cand in result.pareto.items()
        }
        best_entry["search_trace"] = result.trace
        final_payload = persist_final_idea(
            best_entry=best_entry,
            paper_entries=paper_entries,
            artifact=self.artifact,
            idea_result_path=self.idea_result_path,
            chat_fn=self.chat,
            model=self.model,
            logger=logger,
            prompts=PROMPTS,
            persist_to_artifact=False,
        )

        canonical_pool = list(idea_history)
        canonical_pool.append(best_entry)
        pareto_lines = []
        for label, cand in result.pareto.items():
            if cand:
                pareto_lines.append(
                    f"{label}: {cand.node.state.title} (score={cand.evaluation.composite:.2f})"
                )
        pareto_summary = "; ".join(pareto_lines) if pareto_lines else "no Pareto picks"
        step = (
            f"\nIn this idea_generation action, I ran memory-guided MCTS over '{topic}'. "
            f"Best idea: {best_entry['title']} (score={best_entry['search_score']:.2f}). "
            f"Pareto set -> {pareto_summary}. Persisted {len(result.experiences)} defect→fix lifts to long-term memory."
        )
        return StageResult(
            artifact_patch=ArtifactPatch(
                replace={
                    "idea_pool": canonical_pool,
                    "idea_result": final_payload,
                },
                append={
                    "evaluations": [best_payload["evaluation"]],
                    "ltm_experiences": list(result.experiences),
                },
            ),
            step_summary=step,
            metrics={
                "experience_count": len(result.experiences),
                "pareto_count": sum(1 for cand in result.pareto.values() if cand),
                "search_score": best_entry["search_score"],
            },
        )

    def _execute_reanalysis_replan_stage(self, ctx: StageContext) -> StageResult:
        analysis = self.artifact["analysis"][-1] if self.artifact["analysis"] else {}
        ablation_results = self.artifact.get("ablation_results", [])
        mature_idea = self.artifact.get("mature_idea", "")
        topic = self.artifact["topic"][-1]

        prompt = PROMPTS["re_analysis_replan"].format(
            topic=topic,
            mature_idea=mature_idea or "(no mature idea yet)",
            analysis=json.dumps(analysis, ensure_ascii=False, indent=2) if isinstance(analysis, dict) else str(analysis),
            ablation_results=json.dumps(ablation_results, ensure_ascii=False, indent=2) if ablation_results else "[]",
        )
        response = self._parse_json_response(self.chat(prompt, model=self.model))

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

        n_decisions = len(component_decisions)
        decision_summary = "; ".join(
            f"{d['component']}\u2192{d['decision']}" for d in component_decisions if isinstance(d, dict)
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

    def _build_knowledge_acquisition_workflow(self) -> WorkflowSpec:
        return WorkflowSpec(
            name="ligagent.knowledge_acquisition",
            entry_stage="ka_route",
            stages={
                "ka_route": StageSpec(name="ka_route", handler=self._ka_route_stage),
                "ka_seed_search": StageSpec(
                    name="ka_seed_search",
                    handler=self._ka_seed_search_stage,
                    fallback_stage="ka_seed_search_fallback",
                ),
                "ka_seed_search_fallback": StageSpec(
                    name="ka_seed_search_fallback",
                    handler=self._ka_seed_search_fallback_stage,
                ),
                "ka_query_generation": StageSpec(
                    name="ka_query_generation",
                    handler=self._ka_query_generation_stage,
                ),
                "ka_outcome_rag": StageSpec(
                    name="ka_outcome_rag",
                    handler=self._ka_outcome_rag_stage,
                ),
                "ka_citation_expansion": StageSpec(
                    name="ka_citation_expansion",
                    handler=self._ka_citation_expansion_stage,
                ),
                "ka_enrichment": StageSpec(
                    name="ka_enrichment",
                    handler=self._ka_enrichment_stage,
                ),
                "ka_paper_triage": StageSpec(
                    name="ka_paper_triage",
                    handler=self._ka_paper_triage_stage,
                ),
            },
            transitions={
                "ka_query_generation": [WorkflowEdge("ka_outcome_rag")],
                "ka_outcome_rag": [WorkflowEdge("ka_citation_expansion")],
                "ka_citation_expansion": [
                    WorkflowEdge("ka_enrichment", when=lambda stage_ctx, _result: bool(stage_ctx.state.get("combined_papers"))),
                ],
                "ka_enrichment": [WorkflowEdge("ka_paper_triage")],
            },
        )

    def _ka_route_stage(self, ctx: StageContext) -> StageResult:
        search_keywords = self.artifact["retrieval_keywords"][-1]
        topic = self.artifact["topic"][-1] if self.artifact.get("topic") else search_keywords
        mature_idea = (self.artifact.get("mature_idea", "") or "").strip()
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

    def _ka_seed_search_stage(self, ctx: StageContext) -> StageResult:
        search_keywords = ctx.state["search_keywords"]
        try:
            papers = self.run_tool(
                name="semantic_search",
                query=search_keywords,
                limit=self.semantic_search_limit,
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

    def _ka_seed_search_fallback_stage(self, ctx: StageContext) -> StageResult:
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

    def _ka_query_generation_stage(self, ctx: StageContext) -> StageResult:
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
                self.paper_repository,
                logger,
            )
            query_topic = search_keywords
        rag_query = generate_rag_query(
            query_topic,
            query_papers,
            PROMPTS,
            self.chat,
            self.model,
            logger,
            mature_idea=mature_idea if mature_idea else None,
        )
        logger.info("🔎 Generated RAG Query%s: %s", " (mature idea)" if mode == "mature_idea" else "", rag_query)
        return StageResult(
            state_patch={
                "query_papers": query_papers,
                "rag_query": rag_query,
            },
            metrics={"query_papers": len(query_papers), "rag_query_length": len(rag_query)},
        )

    def _ka_outcome_rag_stage(self, ctx: StageContext) -> StageResult:
        rag_query = ctx.state["rag_query"]
        rag_hits = retrieve_outcome_rag(
            query=rag_query,
            top_k=5,
            paper_repository=self.paper_repository,
            logger=logger,
        )
        survey_contents = collect_rag_contents(rag_hits)
        citation_titles = collect_rag_citations(rag_hits)
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

    def _ka_citation_expansion_stage(self, ctx: StageContext) -> StageResult:
        rag_query = ctx.state["rag_query"]
        rag_papers = search_papers_from_citations(
            ctx.state.get("citation_titles", []),
            rag_query,
            self.paper_repository,
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

    def _ka_enrichment_stage(self, ctx: StageContext) -> StageResult:
        papers = ctx.state.get("combined_papers", [])
        if not papers:
            return StageResult(
                status="degraded",
                metrics={"combined_papers": 0},
            )

        temp_artifact = {
            "paper_contents": deepcopy(self.artifact.get("paper_contents", {}))
        }
        safely_enrich_papers_with_content(
            papers,
            self.paper_enrichment_timeout,
            self.paper_repository,
            temp_artifact,
            logger,
        )
        return StageResult(
            artifact_patch=ArtifactPatch(
                merge={"paper_contents": temp_artifact["paper_contents"]}
            ),
            metrics={"combined_papers": len(papers)},
        )

    def _ka_paper_triage_stage(self, ctx: StageContext) -> StageResult:
        topic = ctx.state["topic"]
        mature_idea = ctx.state.get("mature_idea", "")
        combined_papers = ctx.state.get("combined_papers", [])
        mode = ctx.state["mode"]
        rag_query = ctx.state["rag_query"]
        rag_hits = ctx.state.get("rag_hits", [])
        initial_papers = ctx.state.get("initial_papers", [])

        temp_artifact = {
            "paper_contents": deepcopy(self.artifact.get("paper_contents", {}))
        }
        curated_papers = filter_and_compress_papers(
            topic=topic,
            mature_idea=mature_idea,
            papers=combined_papers,
            artifact=temp_artifact,
            prompts=PROMPTS,
            chat_fn=self.chat,
            model=self.model,
            logger=logger,
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

        return StageResult(
            artifact_patch=ArtifactPatch(
                append={"references": [curated_papers]},
                merge={"paper_contents": temp_artifact["paper_contents"]},
            ),
            state_patch={
                "curated_papers": curated_papers,
                "summary": summary,
            },
            metrics={"curated_papers": len(curated_papers)},
        )

    def _generate_idea_introduction(
        self, best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
    ) -> str:
        entries = paper_entries or collect_paper_context_entries(
            self.artifact,
            self.artifact.get("references", []),
        )
        topic = self.artifact["topic"][-1] if self.artifact["topic"] else "unspecified topic"
        return generate_idea_introduction(
            chat_fn=self.chat,
            prompt_template=PROMPTS["idea_introduction"],
            model=self.model,
            topic=topic,
            best_entry=best_entry,
            paper_entries=entries,
            logger=logger,
        )

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        return parse_json_response(raw)

    def _inject_symbolic_priors(
        self,
        topic: str,
        context: Dict[str, Any],
    ) -> None:
        """Inject component×op priors into ``self.mcts.symbolic_memory``.

        This is the single entry-point where external signals are translated
        into :class:`SymbolicRecord` instances that ``compute_action_priors``
        can consume during MCTS expand.

        Two signal sources may exist in the artifact, but only
        **experiment-agent ablation results** are injected by the default
        implementation:

        1. **Experiment-agent ablation results** – stored under
           ``self.artifact["ablation_results"]``.  Each entry is a dict::

               {
                   "component": "uncertainty_weighted_loss",
                   "op": "remove",          # add | remove | replace | ...
                   "delta_score": -0.12,     # observed Δ metric
                   "method_context": "...",  # optional, for family extraction
                   "context_signature": {},   # optional ContextSignature dict
                   "confidence": 0.8,
               }

        2. **Paper-graph conclusions** – stored under
           ``self.artifact["paper_graph_priors"]``.  They are not injected
           into symbolic memory by the default implementation.

        Override or extend this method to plug in additional signal sources.
        """
        sym = self.mcts.symbolic_memory
        injected = 0

        entries = self.artifact.get("ablation_results", [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            component = str(entry.get("component", "")).strip()
            main_op = str(entry.get("op", "")).strip().lower()
            delta = entry.get("delta_score")
            if not component or not main_op or delta is None:
                continue

            method_text = str(entry.get("method_context", ""))
            families = extract_component_families([component], method_text)
            component_family = families[0].get("family", "") if families else ""

            ctx_sig_raw = entry.get("context_signature")
            ctx_sig_dict = dict(ctx_sig_raw) if isinstance(ctx_sig_raw, dict) else {}

            confidence = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))

            try:
                record = sym.instantiate_symbolic_record(
                    summary=f"{main_op} on {component} -> delta={float(delta):+.4f}",
                    pattern=f"component={component}; op={main_op}",
                    conditions=[],
                    actions=[f"{main_op}:{component}"],
                    rationale=str(entry.get("rationale", "")),
                    expected_outcomes=[],
                    anti_patterns=[],
                    tags=["experiment_ablation", main_op, component],
                    priority=max(0.0, min(1.0, abs(float(delta)))),
                    confidence=confidence,
                    source="experiment_ablation",
                    support_count=int(entry.get("support_count", 1)),
                    metadata={"topic": topic, "raw_entry": entry},
                    component_family=component_family,
                    main_op=main_op,
                    context_signature=ctx_sig_dict,
                    delta_score=float(delta),
                )
                sym.upsert_normal_records([record], agent_id="idea_agent")
                injected += 1
            except Exception as exc:
                logger.warning(
                    "⚠️  Failed to inject symbolic prior for %s/%s: %s",
                    component, main_op, exc,
                )

        if injected:
            logger.info(
                "[LigAgent] Injected %d symbolic prior(s) into MCTS symbolic memory.",
                injected,
            )


    def ingest_ablation_results(self, results: List[Dict[str, Any]]) -> None:
        """Write experiment-agent ablation results into the artifact so that
        ``_inject_symbolic_priors`` can consume them on the next
        ``idea_generation`` call.

        Expected schema for each entry in ``results``::

            {
                "component":        str,   # e.g. "uncertainty_weighted_loss"
                "op":               str,   # add | remove | replace | ...
                "delta_score":      float, # observed Δ metric
                "method_context":   str,   # optional free-text for family extraction
                "context_signature": dict, # optional ContextSignature fields
                "confidence":       float, # [0, 1]
                "rationale":        str,   # optional
                "support_count":    int,   # optional, default 1
            }
        """
        if not results:
            return
        valid: List[Dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                logger.warning("[LigAgent] ingest_ablation_results: skipping non-dict entry %r", entry)
                continue
            if not entry.get("component") or not entry.get("op") or entry.get("delta_score") is None:
                logger.warning(
                    "[LigAgent] ingest_ablation_results: skipping entry missing required fields "
                    "(component/op/delta_score): %r",
                    entry,
                )
                continue
            valid.append(dict(entry))
        existing: List[Dict[str, Any]] = self.artifact.get("ablation_results", [])
        self.artifact["ablation_results"] = existing + valid
        logger.info(
            "[LigAgent] ingest_ablation_results: added %d record(s), total=%d",
            len(valid),
            len(self.artifact["ablation_results"]),
        )
