from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.agent import get_logger

from typing import Any, Dict, Literal, List, Optional, Set
from pathlib import Path
import time
import json
import http.client
from dataclasses import fields

from src.agents.idea_agent.agent.tools import TOOLS
from src.agents.idea_agent.agent.artifacts import artifact_init
from src.agents.idea_agent.agent.prompts import PROMPTS
from src.agents.idea_agent.agent.mcts import MemoryGuidedMCTS, MCTSConfig
from src.agents.idea_agent.agent.paper_repository import PaperRepository
from src.agents.idea_agent.agent.ligagent_flow import persist_final_idea
from src.agents.idea_agent.utils.ligagent_utils import (
    collect_paper_context_entries,
    generate_idea_introduction,
    parse_json_response,
)
from src.agents.idea_agent.utils.ligagent_helpers import (
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
    ingest_analysis_background,
    latest_analysis_seed_ideas,
    sanitize_action_token,
    get_paper_content as load_paper_content,
)
from src.agents.idea_agent.utils.config_loader import get_config_value
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

        # Initialize mature_idea in artifact from config (user-provided) if available
        initial_mature_idea = get_config_value(config, "run.mature_idea", "")
        if isinstance(initial_mature_idea, str) and initial_mature_idea.strip():
            self.artifact["mature_idea"] = initial_mature_idea.strip()

        mcts_config = MCTSConfig()
        for field in fields(MCTSConfig):
            override = get_config_value(config, f"mcts.{field.name}", None)
            if override is not None:
                setattr(mcts_config, field.name, override)
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
                if "gpt-5" in model:
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
        action = resolved_action

        if action == "knowledge_aquisition":
            logger.info("🔍 Due to API and web request rate limits, this process may take some time...")
            step = self.knowledge_aquisition(**kwargs)
            self.artifact["steps"].append(step)
            logger.info(step)
            return step
        if action == "advanced_analysis":
            step = self.advanced_analysis(**kwargs)
            self.artifact["steps"].append(step)
            logger.info(step)
            return step
        if action == "idea_generation":
            step = self.idea_generation(**kwargs)
            self.artifact["steps"].append(step)
            logger.info(step)
            return step
        if action == "re_analysis_replan":
            step = self.re_analysis_replan(**kwargs)
            self.artifact["steps"].append(step)
            logger.info(step)
            return step

        raise ValueError(f"Unknown action: {action}")

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
        if search_type == "paper_search":
            search_keywords = self.artifact["retrieval_keywords"][-1]
            topic = self.artifact["topic"][-1] if self.artifact.get("topic") else search_keywords
            mature_idea = self.artifact.get("mature_idea", "")
            # If a mature idea is provided, use it to generate a focused RAG query directly
            if len(mature_idea) > 0 and mature_idea.strip():
                try:
                    rag_query = generate_rag_query(
                        topic,
                        [],
                        PROMPTS,
                        self.chat,
                        self.model,
                        logger,
                        mature_idea=mature_idea,
                    )
                    logger.info("🔎 Generated RAG Query (mature idea): %s", rag_query)
                    rag_hits = retrieve_outcome_rag(query=rag_query, top_k=5, paper_repository=self.paper_repository, logger=logger)
                    self.artifact.setdefault("rag_query", []).append(rag_query)
                    self.artifact.setdefault("rag_hits", []).append(
                        {"query": rag_query, "hits": rag_hits}
                    )
                    citation_titles = collect_rag_citations(rag_hits)
                    survey_contents = collect_rag_contents(rag_hits)
                    rag_papers = search_papers_from_citations(
                        citation_titles, rag_query, self.paper_repository
                    )
                    if rag_papers:
                        safely_enrich_papers_with_content(
                            rag_papers,
                            self.paper_enrichment_timeout,
                            self.paper_repository,
                            self.artifact,
                            logger,
                        )
                        rag_papers = filter_and_compress_papers(
                            topic=topic,
                            mature_idea=mature_idea,
                            papers=rag_papers,
                            artifact=self.artifact,
                            prompts=PROMPTS,
                            chat_fn=self.chat,
                            model=self.model,
                            logger=logger,
                            top_k=5,
                        )
                        self.artifact["references"].append(rag_papers)
                        self.artifact["rag_contents"].append(survey_contents)
                        step = (
                            f"\nIn this knowledge_aquisition action, I used the mature idea to generate "
                            f"a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
                            f"and curated {len(rag_papers)} cited papers for memory."
                        )
                    else:
                        self.artifact.setdefault("rag_contents", []).append(survey_contents)
                        step = (
                            f"\nIn this knowledge_aquisition action, I used the mature idea to generate "
                            f"a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
                            "but found no cited papers to fetch."
                        )
                except Exception as e:
                    logger.error(f"Error during mature-idea RAG retrieval: {e}")
                    step = (
                        "\nIn this knowledge_aquisition action, mature-idea RAG retrieval failed; "
                        "skipping paper search."
                    )
                return step
            # Otherwise, perform standard paper search flow
            try:
                papers = self.run_tool(
                    name="semantic_search",
                    query=search_keywords,
                    limit=self.semantic_search_limit,
                )
                logger.info("📄 Found Papers:")
                mature_idea = self.artifact.get("mature_idea", "")
                initial_papers = normalize_search_papers(papers, search_keywords, logger)
                if initial_papers:
                    query_papers = prepare_query_papers(
                        initial_papers, self.paper_repository, logger
                    )
                    rag_query = generate_rag_query(
                        search_keywords,
                        query_papers,
                        PROMPTS,
                        self.chat,
                        self.model,
                        logger,
                        mature_idea=mature_idea if len(mature_idea) > 0 else None,
                    )
                    logger.info("🔎 Generated RAG Query: %s", rag_query)
                    rag_hits = retrieve_outcome_rag(query=rag_query, top_k=5, paper_repository=self.paper_repository, logger=logger)
                    self.artifact.setdefault("rag_query", []).append(rag_query)
                    self.artifact.setdefault("rag_hits", []).append(
                        {"query": rag_query, "hits": rag_hits}
                    )
                    citation_titles = collect_rag_citations(rag_hits)
                    survey_contents = collect_rag_contents(rag_hits)
                    rag_papers = search_papers_from_citations(
                        citation_titles, rag_query, self.paper_repository
                    )
                    if rag_papers:
                        combined_papers = initial_papers + rag_papers
                        safely_enrich_papers_with_content(
                            combined_papers,
                            self.paper_enrichment_timeout,
                            self.paper_repository,
                            self.artifact,
                            logger,
                        )
                        curated_papers = filter_and_compress_papers(
                            topic=topic,
                            mature_idea=mature_idea,
                            papers=combined_papers,
                            artifact=self.artifact,
                            prompts=PROMPTS,
                            chat_fn=self.chat,
                            model=self.model,
                            logger=logger,
                            top_k=5,
                        )
                        self.artifact["references"].append(curated_papers)
                        self.artifact["rag_contents"].append(survey_contents)
                        step = (
                            f"\nIn this knowledge_aquisition action, I read {len(initial_papers)} seed papers, "
                            f"generated a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
                            f"and curated {len(curated_papers)} papers for memory."
                        )
                    else:
                        safely_enrich_papers_with_content(
                            initial_papers,
                            self.paper_enrichment_timeout,
                            self.paper_repository,
                            self.artifact,
                            logger,
                        )
                        curated_papers = filter_and_compress_papers(
                            topic=topic,
                            mature_idea=mature_idea,
                            papers=initial_papers,
                            artifact=self.artifact,
                            prompts=PROMPTS,
                            chat_fn=self.chat,
                            model=self.model,
                            logger=logger,
                            top_k=5,
                        )
                        self.artifact["references"].append(curated_papers)
                        step = (
                            f"\nIn this knowledge_aquisition action, I searched for papers about '{search_keywords}' "
                            f"and curated {len(curated_papers)} relevant papers for my research."
                        )
                else:
                    step = (
                        f"\nIn this knowledge_aquisition action, I searched for papers about '{search_keywords}' "
                        "but found none."
                    )
            except Exception as e:
                logger.error(f"Error during paper search: {e}")

                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({"q": self.artifact["retrieval_keywords"][-1]})
                headers = {
                    "X-API-KEY": "7854e42317727ecbf17d214f5a96c420dbcdd9cf",
                    "Content-Type": "application/json",
                }
                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                print(data)
                step = (
                    f"\nIn this knowledge_aquisition action, I acquired several papers about '{search_keywords}'."
                )
            return step

    def get_paper_content(self, paper_id: str, include_markdown: bool = True) -> Dict[str, Any]:
        return load_paper_content(
            paper_id,
            include_markdown,
            self.artifact,
            self.paper_repository,
            logger,
        )

    def advanced_analysis(self, **kwargs) -> None:
        topic = self.artifact["topic"][-1] if self.artifact["topic"] else "unspecified topic"
        references = self.artifact["references"][-1] if self.artifact["references"] else []
        mature_idea = self.artifact.get("mature_idea", "")
        prompt = PROMPTS["advanced_analysis"].format(
            topic=topic,
            mature_idea=(mature_idea or "").strip(),
            survey_contents="\n".join(self.artifact["rag_contents"][-1]) if self.artifact.get("rag_contents") else "",
            papers=json.dumps(references, ensure_ascii=False, indent=2)
            if references
            else "[]",
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
        self.artifact["analysis"].append(response)
        ingest_analysis_background(response, self.artifact)
        step = (
            "\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: "
            f"{response.get('tldr', 'No TL;DR provided.')}"
        )
        return step

    def idea_generation(self, **kwargs) -> None:
        topic = self.artifact["topic"][-1]
        reference_batches = self.artifact.get("references", [])
        latest_batch = reference_batches[-1] if reference_batches else []
        batch_list = [latest_batch] if latest_batch else reference_batches
        paper_entries = collect_paper_context_entries(
            self.artifact,
            batch_list,
        )
        idea_history = list(self.artifact.get("idea_pool", []))
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

        # Inject external component×op priors into symbolic memory before MCTS
        self._inject_symbolic_priors(topic, context)

        result = self.mcts.search(topic=topic, context=context)
        
        if not result.best:
            logger.warning("⚠️ MCTS search returned no candidate; keeping current idea pool unchanged.")
            return "\nIn this idea_generation action, MCTS returned no candidate and no fallback legacy path was used."

        best_payload = result.best.to_dict()
        best_entry = best_payload["idea"]
        best_entry["evaluation"] = best_payload["evaluation"]
        best_entry["search_score"] = best_payload["score"]
        best_entry["search_path"] = best_payload["path"]
        best_entry["pareto_candidates"] = {
            label: cand.to_dict() if cand else None for label, cand in result.pareto.items()
        }
        best_entry["search_trace"] = result.trace
        self.artifact["idea_pool"].append(best_entry)
        self.artifact.setdefault("evaluations", []).append(best_payload["evaluation"])
        self.artifact.setdefault("ltm_experiences", []).extend(result.experiences)
        pareto_lines = []
        for label, cand in result.pareto.items():
            if cand:
                pareto_lines.append(
                    f"{label}: {cand.node.state.title} (score={cand.evaluation.composite:.2f})"
                )
        pareto_summary = "; ".join(pareto_lines) if pareto_lines else "no Pareto picks"
        self._persist_final_idea(best_entry, paper_entries)
        step = (
            f"\nIn this idea_generation action, I ran memory-guided MCTS over '{topic}'. "
            f"Best idea: {best_entry['title']} (score={best_entry['search_score']:.2f}). "
            f"Pareto set -> {pareto_summary}. Persisted {len(result.experiences)} defect→fix lifts to long-term memory."
        )
        return step

    def re_analysis_replan(self, **kwargs) -> None:
        # Gather analysis and ablation evidence for component-level revision
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

        # Store component-level decisions for traceability
        component_decisions = response.get("component_decisions", [])
        self.artifact.setdefault("component_decisions", []).extend(component_decisions)

        # Update mature_idea in artifact so that MCTS uses it as root node
        if response.get("mature_idea"):
            self.artifact["mature_idea"] = response["mature_idea"]

        # Update search keywords if new mechanisms were introduced
        search_kw = response.get("search_keywords", "")
        if search_kw:
            self.artifact["retrieval_keywords"].append(search_kw)

        n_decisions = len(component_decisions)
        decision_summary = "; ".join(
            f"{d['component']}\u2192{d['decision']}" for d in component_decisions if isinstance(d, dict)
        ) or "no component decisions"
        step = (
            f"\nIn this re_analysis_replan action, I made {n_decisions} component-level "
            f"modification(s) based on ablation evidence: [{decision_summary}]. "
            f"Updated mature idea for MCTS root node."
        )
        return step

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

        Two signal sources are consumed:

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
           ``self.artifact["paper_graph_priors"]``.  Same schema, but typically
           lower confidence and coarser ``context_signature``.

        Override or extend this method to plug in additional signal sources.
        """
        sym = self.mcts.symbolic_memory
        injected = 0

        for source_key, source_label in [
            ("ablation_results", "experiment_ablation"),
            ("paper_graph_priors", "paper_graph"),
        ]:
            entries = self.artifact.get(source_key, [])
            if not entries:
                continue
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
                        tags=[source_label, main_op, component],
                        priority=max(0.0, min(1.0, abs(float(delta)))),
                        confidence=confidence,
                        source=source_label,
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

    # ──────────────────────────────────────────────────────────────────────
    #  External prior ingestion entry-points  (not yet implemented)
    # ──────────────────────────────────────────────────────────────────────

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

    def ingest_paper_graph_priors(self, priors: List[Dict[str, Any]]) -> None:
        """Write paper-graph derived priors into the artifact so that
        ``_inject_symbolic_priors`` can consume them on the next
        ``idea_generation`` call.

        Expected schema for each entry in ``priors``: identical to
        :meth:`ingest_ablation_results`, but typically with lower
        ``confidence`` and coarser ``context_signature``.
        """
        if not priors:
            return
        valid: List[Dict[str, Any]] = []
        for entry in priors:
            if not isinstance(entry, dict):
                logger.warning("[LigAgent] ingest_paper_graph_priors: skipping non-dict entry %r", entry)
                continue
            if not entry.get("component") or not entry.get("op") or entry.get("delta_score") is None:
                logger.warning(
                    "[LigAgent] ingest_paper_graph_priors: skipping entry missing required fields "
                    "(component/op/delta_score): %r",
                    entry,
                )
                continue
            valid.append(dict(entry))
        existing: List[Dict[str, Any]] = self.artifact.get("paper_graph_priors", [])
        self.artifact["paper_graph_priors"] = existing + valid
        logger.info(
            "[LigAgent] ingest_paper_graph_priors: added %d record(s), total=%d",
            len(valid),
            len(self.artifact["paper_graph_priors"]),
        )
