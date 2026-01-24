from agent.base import AgentBase
from agent import get_logger

logger = get_logger()
logger.info("🤖 Initializing LigAgent...")

from typing import Any, Dict, Literal, List, Optional, Set
from pathlib import Path
import time
import json
import http.client

from agent.tools import TOOLS
from agent.memory import memory_init
from agent.prompts import PROMPTS
from agent.mcts import MemoryGuidedMCTS, MCTSConfig
from agent.paper_repository import PaperRepository
from src.agents.idea_agent.utils.idea_helpers import (
    build_mcts_evolution,
    collect_reference_material,
)
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
    search_papers_from_citations,
    safely_enrich_papers_with_content,
    paper_context_with_rag,
    normalize_analysis_entry,
    ingest_analysis_background,
    latest_analysis_seed_ideas,
    build_algorithm_spec,
    synthesize_reference_summaries,
    suggest_datasets,
    sanitize_action_token,
    get_paper_content as load_paper_content,
)


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
        "idea_evaluation": {
            "idea_evaluation",
            "idea evaluation",
            "ideaevaluation",
            "idea-evaluation",
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
        chat_max_retries = kwargs.pop("chat_max_retries", 3)
        chat_retry_backoff = kwargs.pop("chat_retry_backoff", 2.0)
        survey_config_path = kwargs.pop("survey_config_path", None)
        run_dir = kwargs.pop("run_dir", None)
        idea_result_path = kwargs.pop("idea_result_path", None)
        rag_config = kwargs.pop("rag_config", None)
        self.paper_enrichment_timeout = kwargs.pop("paper_enrichment_timeout", 7200)
        action_selection_attempts = kwargs.pop("action_selection_attempts", 2)
        super().__init__(*args, **kwargs)
        self.action_space = [
            "knowledge_aquisition",
            "advanced_analysis",
            "idea_generation",
            "idea_evaluation",
            "re_analysis_replan",
        ]
        self.action_selection_attempts = max(1, action_selection_attempts)
        self.model = "mimo-v2-flash"
        self.tools = TOOLS
        self.memory = memory_init()
        self.run_dir = Path(run_dir) if run_dir else Path(__file__).resolve().parent.parent
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if idea_result_path:
            self.idea_result_path = Path(idea_result_path)
        else:
            self.idea_result_path = self.run_dir / "idea_result.json"
        self.idea_result_path.parent.mkdir(parents=True, exist_ok=True)
        self.chat_max_retries = chat_max_retries
        self.chat_retry_backoff = chat_retry_backoff
        self._action_lookup = build_action_lookup(self.ACTION_ALIASES)
        self.mcts = MemoryGuidedMCTS(
            chat_fn=self.chat,
            generation_prompt=PROMPTS["mcts_generation"],
            evaluation_prompt=PROMPTS["mcts_evaluation"],
            config=MCTSConfig(),
            logger=logger,
        )
        self.paper_repository = PaperRepository(
            config_path=survey_config_path,
            logger=logger,
            rag_config=rag_config,
        )

    def chat(self, prompt: str, model: str = "gpt-4.1", **kwargs) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.chat_max_retries + 1):
            try:
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

    def select_action(self, observation: Any) -> str:
        invalid_action: Optional[str] = None
        for attempt in range(1, self.action_selection_attempts + 1):
            template_key = "action_selection" if attempt == 1 else "action_retry"
            template = PROMPTS.get(template_key, PROMPTS["action_selection"])
            prompt = template.format(
                action_space=self.action_space,
                step=observation,
                invalid_action=invalid_action or "",
            )
            response = self.chat(prompt, model=self.model)
            resolved = self._canonical_action(response)
            if resolved:
                return resolved
            invalid_action = response
            logger.warning(
                "⚠️ Invalid action '%s' (attempt %d/%d). Reminding agent of allowed actions.",
                response,
                attempt,
                self.action_selection_attempts,
            )
        raise ValueError(
            f"Unable to obtain a valid action after {self.action_selection_attempts} attempts (last response: {invalid_action})"
        )

    def perform_action(self, action: str, **kwargs) -> Any:
        """
        Dispatch an action to the corresponding method, forwarding positional and keyword arguments.
        """
        logger.info(f"🚀 Performing action: {action}...")
        resolved_action = self._canonical_action(action)
        if not resolved_action:
            raise ValueError(f"Unknown action: {action}")
        action = resolved_action

        if action == "knowledge_aquisition":
            logger.info("🔍 Due to API and web request rate limits, this process may take some time...")
            step = self.knowledge_aquisition(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action == "advanced_analysis":
            step = self.advanced_analysis(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action == "idea_generation":
            step = self.idea_generation(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action == "idea_evaluation":
            step = self.idea_evaluation(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action == "re_analysis_replan":
            step = self.re_analysis_replan(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step

        raise ValueError(f"Unknown action: {action}")

    def select_memory(self, action: str = "", info: dict = {}) -> str:
        pass

    def bootstrap_topic(self, topic: str, retrieval_keywords: Optional[str] = None) -> None:
        normalized_topic = (topic or "").strip()
        if not normalized_topic:
            raise ValueError("Topic must be a non-empty string.")
        keywords = (
            retrieval_keywords.strip()
            if isinstance(retrieval_keywords, str)
            else normalized_topic
        )
        self.memory["topic"].append(normalized_topic)
        self.memory["retrieval_keywords"].append(keywords)
        background = generate_background_brief(
            normalized_topic,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        if background:
            self.memory["background_knowledge"].append(background)

    def knowledge_aquisition(
        self, search_type: Literal["paper_search", "website"] = "paper_search"
    ) -> str:
        if search_type == "paper_search":
            search_keywords = self.memory["retrieval_keywords"][-1]
            try:
                papers = self.run_tool(name="semantic_search", query=search_keywords, limit=10)
                logger.info("📄 Found Papers:")
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
                    )
                    logger.info("🔎 Generated RAG Query: %s", rag_query)
                    rag_hits = retrieve_outcome_rag(rag_query, self.paper_repository, logger)
                    self.memory.setdefault("rag_query", []).append(rag_query)
                    self.memory.setdefault("rag_hits", []).append(
                        {"query": rag_query, "hits": rag_hits}
                    )
                    citation_titles = collect_rag_citations(rag_hits)
                    rag_papers = search_papers_from_citations(
                        citation_titles, rag_query, self.paper_repository
                    )
                    if rag_papers:
                        safely_enrich_papers_with_content(
                            rag_papers,
                            self.paper_enrichment_timeout,
                            self.paper_repository,
                            self.memory,
                            logger,
                        )
                        self.memory["references"].append(rag_papers)
                    step = (
                        f"\nIn this knowledge_aquisition action, I read {len(initial_papers)} seed papers, "
                        f"generated a focused query '{rag_query}', retrieved {len(rag_hits)} RAG hits, "
                        f"and fetched {len(rag_papers)} cited papers for memory."
                    )
                else:
                    step = (
                        f"\nIn this knowledge_aquisition action, I searched for papers about '{search_keywords}' "
                        "but found none."
                    )
            except Exception as e:
                logger.error(f"Error during paper search: {e}")

                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({"q": self.memory["retrieval_keywords"][-1]})
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
            self.memory,
            self.paper_repository,
            logger,
        )

    def advanced_analysis(self, **kwargs) -> None:
        topic = self.memory["topic"][-1] if self.memory["topic"] else "unspecified topic"
        references = self.memory["references"][-1] if self.memory["references"] else []
        prompt = PROMPTS["advanced_analysis"].format(
            topic=topic,
            papers=json.dumps(references, ensure_ascii=False, indent=2)
            if references
            else "[]",
        )
        raw_response = self._parse_json_response(
            self.chat(prompt, model=self.model, max_tokens=4096)
        )
        response = normalize_analysis_entry(raw_response)
        logger.info(f"📝 Advanced Analysis Result:\n{response}")
        self.memory["analysis"].append(response)
        ingest_analysis_background(response, self.memory)
        step = (
            "\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: "
            f"{response.get('tldr', 'No TL;DR provided.')}"
        )
        return step

    def idea_generation(self, **kwargs) -> None:
        topic = self.memory["topic"][-1] if self.memory["topic"] else "unspecified topic"
        paper_entries = collect_paper_context_entries(
            self.memory, self.memory.get("references", []), limit=10
        )
        idea_history = list(self.memory.get("idea_pool", []))
        seed_ideas = latest_analysis_seed_ideas(self.memory)
        idea_context = idea_history if idea_history else seed_ideas
        context = {
            "analysis": self.memory.get("analysis", []),
            "idea_pool": idea_context,
            "background_knowledge": self.memory.get("background_knowledge", []),
            "paper_context": paper_context_with_rag(paper_entries, self.memory),
        }
        result = self.mcts.search(topic=topic, context=context)
        if not result.best:
            logger.warning("⚠️ MCTS search returned no candidate, falling back to legacy generator.")
            legacy_step = self._legacy_single_idea(topic, paper_entries)
            return legacy_step

        best_payload = result.best.to_dict()
        best_entry = best_payload["idea"]
        best_entry["evaluation"] = best_payload["evaluation"]
        best_entry["search_score"] = best_payload["score"]
        best_entry["search_path"] = best_payload["path"]
        best_entry["pareto_candidates"] = {
            label: cand.to_dict() if cand else None for label, cand in result.pareto.items()
        }
        best_entry["search_trace"] = result.trace
        self.memory["idea_pool"].append(best_entry)
        self.memory.setdefault("evaluations", []).append(best_payload["evaluation"])
        self.memory.setdefault("ltm_experiences", []).extend(result.experiences)
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

    def _legacy_single_idea(self, topic: str, paper_entries: List[Dict[str, Any]]) -> str:
        prompt = PROMPTS["idea_generation"].format(
            topic=topic,
            analysis=self.memory.get("analysis", []),
            ideas=self.memory.get("idea_pool", []),
            papers=json.dumps(paper_entries, ensure_ascii=False, indent=2),
        )
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        self.memory["idea_pool"].append(response)
        return (
            "\nIn this idea_generation action, I generated new research ideas via fallback prompt:\n💡 "
            f"{response}"
        )

    def idea_evaluation(self, **kwargs) -> None:
        prompt = PROMPTS["idea_evaluation"].format(
            topic=self.memory["topic"][-1], idea=self.memory["idea_pool"][-1]
        )
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        self.memory["idea_pool"][-1]["evaluation"] = response
        step = (
            "\nIn this idea_evaluation action, I evaluated the generated research ideas:\n✅ "
            f"{response}"
        )
        return step

    def re_analysis_replan(self, **kwargs) -> None:
        prompt = PROMPTS["re_analysis_replan"].format(
            topic=self.memory["topic"][-1],
            idea=self.memory["idea_pool"][-1],
            last_queries=self.memory["retrieval_keywords"],
            topics=self.memory["topic"],
        )
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        self.memory["topic"].append(response["new_topic"])
        self.memory["retrieval_keywords"].append(response["search_keywords"])
        step = (
            "\nIn this re_analysis_replan action, I replanned my research topic to "
            f"'{response['new_topic']}' and decided to search for new information using keywords "
            f"'{response['search_keywords']}'."
        )
        return step

    def _persist_final_idea(
        self, best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
    ) -> None:
        topic = self.memory["topic"][-1] if self.memory["topic"] else "unspecified topic"
        raw_refs = collect_reference_material(self.memory.get("references", []))
        algorithm = build_algorithm_spec(
            best_entry,
            topic,
            raw_refs,
            self.memory,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        references = synthesize_reference_summaries(
            topic,
            best_entry,
            algorithm,
            raw_refs,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        datasets = suggest_datasets(
            topic,
            best_entry,
            algorithm,
            references,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        introduction = self._generate_idea_introduction(best_entry, paper_entries)
        payload = {
            "title": best_entry.get("title"),
            "abstract": best_entry.get("abstract"),
            "introduction": introduction,
            "algorithm": algorithm,
            "reference_papers": references,
            "datasets": datasets,
            "mcts_evolution": build_mcts_evolution(best_entry),
        }
        best_entry["introduction"] = introduction
        self.memory["idea_result"] = payload
        try:
            with open(self.idea_result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved idea result to {self.idea_result_path}")
        except OSError as exc:
            logger.error(f"⚠️ Failed to persist idea_result.json: {exc}")

    def _generate_idea_introduction(
        self, best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
    ) -> str:
        entries = paper_entries or collect_paper_context_entries(
            self.memory, self.memory.get("references", []), limit=6
        )
        topic = self.memory["topic"][-1] if self.memory["topic"] else "unspecified topic"
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
