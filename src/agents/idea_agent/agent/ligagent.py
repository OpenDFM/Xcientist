from agent.base import AgentBase
from agent import get_logger
logger = get_logger()
logger.info("🤖 Initializing LigAgent...")
from typing import Any, Dict, Literal, List, Optional
from pathlib import Path
import time
from agent.tools import TOOLS
from agent.memory import MEMORY_FORMAT, memory_init
from agent.prompts import PROMPTS
from agent.mcts import MemoryGuidedMCTS, MCTSConfig
from agent.paper_repository import PaperRepository
from agents.idea_agent.utils.idea_helpers import (
    build_mcts_evolution,
    collect_reference_material,
    derive_pipeline_steps,
    fallback_algorithm_spec,
)
import json
import http.client

class LigAgent(AgentBase):
    def __init__(self, *args, **kwargs):
        chat_max_retries = kwargs.pop("chat_max_retries", 3)
        chat_retry_backoff = kwargs.pop("chat_retry_backoff", 2.0)
        survey_config_path = kwargs.pop("survey_config_path", None)
        super().__init__(*args, **kwargs)
        self.action_space = ["knowledge_aquisition", "advanced_analysis", "idea_generation", "idea_evaluation", "re_analysis_replan"]
        self.model = "mimo-v2-flash"
        self.tools = TOOLS
        self.memory = memory_init()
        self.idea_result_path = Path(__file__).resolve().parent.parent / "idea_result.json"
        self.chat_max_retries = chat_max_retries
        self.chat_retry_backoff = chat_retry_backoff
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
        prompt = PROMPTS["action_selection"].format(action_space=self.action_space, step=observation)
        response = self.chat(prompt, model=self.model)
        return response.lower().strip()

    def perform_action(self, action: str, **kwargs) -> Any:
        """
        Dispatch an action to the corresponding method, forwarding positional and keyword arguments.
        """
        logger.info(f"🚀 Performing action: {action}...")

        if action in ["knowledge_aquisition", "knowledge aquisition"]:
            logger.info("🔍 Due to API and web request rate limits, this process may take some time...")
            step = self.knowledge_aquisition(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action in ["advanced_analysis", "advanced analysis"]:
            step = self.advanced_analysis(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action in ["idea_generation", "idea generation"]:
            step = self.idea_generation(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action in ["idea_evaluation", "idea evaluation"]:
            step = self.idea_evaluation(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step
        if action in ["re_analysis_replan", "re analysis replan"]:
            step = self.re_analysis_replan(**kwargs)
            self.memory["steps"].append(step)
            logger.info(step)
            return step

        raise ValueError(f"Unknown action: {action}")
        

    def select_memory(self, action: str="", info: dict={}) -> str:
        pass

    def knowledge_aquisition(self, search_type: Literal["paper_search", "website"]="paper_search") -> str:
        if search_type == "paper_search":
            search_keywords = self.memory["retrieval_keywords"][-1]
            try:
                papers = self.run_tool(name="semantic_search", query=search_keywords, limit=5) 
                logger.info("📄 Found Papers:")
                new_papers = []
                if papers and len(papers) > 0:
                    for i, paper in enumerate(papers, 1):
                        title = paper.get('title') if isinstance(paper, dict) else str(paper)
                        logger.info(f"📄 {i}. {title}")
                        abstract = paper.get('abstract', 'No abstract available.') if isinstance(paper, dict) else 'No abstract available.'
                        authors_field = paper.get("authors", []) if isinstance(paper, dict) else []
                        if isinstance(authors_field, list):
                            authors = [a.get("name", str(a)) for a in authors_field]
                        elif authors_field:
                            authors = [str(authors_field)]
                        else:
                            authors = []
                        paper_entry = {
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": paper.get("year") if isinstance(paper, dict) else None,
                            "url": paper.get("url") if isinstance(paper, dict) else None,
                            "tldr": paper.get("tldr") if isinstance(paper, dict) else None,
                            "paper_id": paper.get("paperId") if isinstance(paper, dict) else None,
                            "source_keywords": search_keywords,
                        }
                        new_papers.append(paper_entry)
                    self._enrich_papers_with_content(new_papers)
                    step = (
                        f"\nIn this knowledge_aquisition action, I acquired {len(papers)} papers "
                        f"about '{search_keywords}' and fetched their parsed + summarized content."
                    )
                    self.memory["references"].append(new_papers)

                else:
                    step = f"\nIn this knowledge_aquisition action, I searched for papers about '{search_keywords}' but found none."
            except Exception as e:
                logger.error(f"Error during paper search: {e}")
                
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({
                "q": self.memory["retrieval_keywords"][-1]
                })
                headers = {
                'X-API-KEY': '7854e42317727ecbf17d214f5a96c420dbcdd9cf',
                'Content-Type': 'application/json'
                }
                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                # import pdb; pdb.set_trace()
                print(data)
                # step = f"\nIn this knowledge_aquisition action, I attempted to search on website about '{search_keywords}', and received the following response: {data}"
                step = f"\nIn this knowledge_aquisition action, I acquired several papers about '{search_keywords}'."
            return step
    def _enrich_papers_with_content(self, papers: List[Dict[str, Any]]) -> None:
        if not papers:
            return
        paper_ids = [paper.get("paper_id") for paper in papers if paper.get("paper_id")]
        if not paper_ids:
            return
        try:
            prepared = self.paper_repository.prepare_papers(paper_ids)
        except Exception as exc:
            logger.warning("Failed to prepare parsed papers: %s", exc)
            return
        storage = self.memory.setdefault("paper_contents", {})
        for paper in papers:
            pid = paper.get("paper_id")
            if not pid:
                continue
            content = prepared.get(pid)
            keynote_data = None
            if content:
                keynote_data = content.get("keynote") or content

            if not keynote_data:
                fallback_text = paper.get("abstract") or paper.get("title") or "No content available."
                keynote_data = {
                    "tldr": fallback_text,
                    "source": "title_abstract_fallback",
                }
                paper["has_parsed_markdown"] = False
            else:
                paper["has_parsed_markdown"] = True

            paper["keynote"] = keynote_data
            logger.info(
                "🗒️ Enriched paper '%s' with summary source=%s",
                paper.get("title"),
                keynote_data.get("source", "parsed"),
            )
            storage[pid] = {
                "keynote": keynote_data,
                "source_keywords": paper.get("source_keywords"),
            }

    def get_paper_content(self, paper_id: str, include_markdown: bool = True) -> Dict[str, Any]:
        if not paper_id:
            return {}
        stored = self.memory.setdefault("paper_contents", {}).get(paper_id, {}).copy()
        stored["paper_id"] = paper_id
        if include_markdown:
            try:
                stored["markdown"] = self.paper_repository.get_markdown(paper_id)
            except Exception as exc:
                logger.warning("Unable to load markdown for %s: %s", paper_id, exc)
        return stored

    def advanced_analysis(self, **kwargs) -> None:
        # import pdb; pdb.set_trace()
        prompt = PROMPTS["advanced_analysis"].format(topic=self.memory["topic"][-1], papers=self.memory["references"][-1])
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        logger.info(f"📝 Advanced Analysis Result:\n{response}")
        self.memory["analysis"].append(response)
        step = f"\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: {response['tldr']}"
        return step

    def idea_generation(self, **kwargs) -> None:
        topic = self.memory["topic"][-1] if self.memory["topic"] else "unspecified topic"
        context = {
            "analysis": self.memory.get("analysis", []),
            "idea_pool": self.memory.get("idea_pool", []),
            "background_knowledge": self.memory.get("background_knowledge", []),
        }
        result = self.mcts.search(topic=topic, context=context)
        if not result.best:
            logger.warning("⚠️ MCTS search returned no candidate, falling back to legacy generator.")
            legacy_step = self._legacy_single_idea(topic)
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
                pareto_lines.append(f"{label}: {cand.node.state.title} (score={cand.evaluation.composite:.2f})")
        pareto_summary = "; ".join(pareto_lines) if pareto_lines else "no Pareto picks"
        self._persist_final_idea(best_entry)
        step = (
            f"\nIn this idea_generation action, I ran memory-guided MCTS over '{topic}'. "
            f"Best idea: {best_entry['title']} (score={best_entry['search_score']:.2f}). "
            f"Pareto set -> {pareto_summary}. Persisted {len(result.experiences)} defect→fix lifts to long-term memory."
        )
        return step

    def _legacy_single_idea(self, topic: str) -> str:
        prompt = PROMPTS["idea_generation"].format(
            topic=topic,
            analysis=self.memory.get("analysis", []),
            ideas=self.memory.get("idea_pool", []),
        )
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        self.memory["idea_pool"].append(response)
        return f"\nIn this idea_generation action, I generated new research ideas via fallback prompt:\n💡 {response}"

    def idea_evaluation(self, **kwargs) -> None:
        prompt = PROMPTS["idea_evaluation"].format(topic=self.memory["topic"][-1], idea=self.memory["idea_pool"][-1])
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        self.memory["idea_pool"][-1]["evaluation"] = response
        step = f"\nIn this idea_evaluation action, I evaluated the generated research ideas:\n✅ {response}"
        return step
    
    def re_analysis_replan(self, **kwargs) -> None:
        prompt = PROMPTS["re_analysis_replan"].format(topic=self.memory["topic"][-1], idea=self.memory["idea_pool"][-1], last_queries=self.memory["retrieval_keywords"], topics=self.memory["topic"])
        response = self._parse_json_response(self.chat(prompt, model=self.model))
        self.memory["topic"].append(response['new_topic'])
        self.memory["retrieval_keywords"].append(response['search_keywords'])
        step = f"\nIn this re_analysis_replan action, I replanned my research topic to '{response['new_topic']}' and decided to search for new information using keywords '{response['search_keywords']}'."
        return step

    def _build_algorithm_spec(
        self,
        idea: Dict[str, Any],
        topic: str,
        raw_references: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        analysis_entries = self.memory.get("analysis", [])
        latest_analysis = analysis_entries[-1] if analysis_entries else {}
        base_inputs: List[str] = []
        if topic and topic != "unspecified topic":
            base_inputs.append(f"Topic focus: {topic}")
        retrieval_history = self.memory.get("retrieval_keywords", [])
        if retrieval_history:
            base_inputs.append(f"Latest retrieval keywords: {retrieval_history[-1]}")
        if isinstance(latest_analysis, dict):
            tldr = latest_analysis.get("tldr")
            if tldr:
                base_inputs.append(f"Analysis TL;DR: {tldr}")
            key_methods = latest_analysis.get("key_methods")
            if key_methods:
                base_inputs.append("Key methods referenced: " + "; ".join(key_methods[:3]))
        if raw_references:
            ref_titles = [r.get("title") for r in raw_references[:3] if r.get("title")]
            if ref_titles:
                base_inputs.append("Reference anchors: " + "; ".join(ref_titles))
        target_defects = idea.get("target_defects")
        if target_defects:
            base_inputs.append(f"Target defects: {', '.join(target_defects)}")

        base_outputs: List[str] = []
        abstract = idea.get("abstract")
        if abstract:
            base_outputs.append(f"Abstract focus: {abstract}")
        core = idea.get("core_contribute") or idea.get("core_contribution")
        if core:
            base_outputs.append(f"Core contribution: {core}")
        methodology = idea.get("methodology") or idea.get("method")
        if methodology:
            base_outputs.append(f"Methodology: {methodology}")
        experiments = idea.get("experiment_design") or idea.get("experiments")
        if experiments:
            base_outputs.append(f"Experiment design: {experiments}")
        score = idea.get("search_score")
        if isinstance(score, (int, float)):
            base_outputs.append(f"MCTS search score: {score:.2f}")

        prompt = PROMPTS["algorithm_structuring"].format(
            topic=topic,
            idea_title=idea.get("title", ""),
            idea_abstract=idea.get("abstract", ""),
            idea=json.dumps(idea, ensure_ascii=False, indent=2),
            base_inputs=json.dumps(base_inputs, ensure_ascii=False, indent=2),
            base_outputs=json.dumps(base_outputs, ensure_ascii=False, indent=2),
            analysis=json.dumps(latest_analysis, ensure_ascii=False, indent=2),
            references=json.dumps(raw_references[:5], ensure_ascii=False, indent=2),
        )
        prompt += "\n Directly output JSON."
        try:
            response = self.chat(prompt, temperature=0.01, max_tokens=4096, model=self.model)
            payload = self._parse_json_response(response)
            candidate = payload.get("algorithms", payload)
            if isinstance(candidate, list) and candidate:
                return self._align_algorithms_with_idea(idea, candidate)
        except Exception as exc:
            logger.warning("⚠️ Algorithm structuring failed: %s", exc)

        return fallback_algorithm_spec(idea, base_inputs, base_outputs)

    def _align_algorithms_with_idea(
        self, idea: Dict[str, Any], algorithms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        title = (idea.get("title") or "").strip()
        abstract = (idea.get("abstract") or "").strip()
        if not algorithms or (not title and not abstract):
            return algorithms

        prompt = PROMPTS["algorithm_alignment"].format(
            idea_title=title,
            idea_abstract=abstract or "No abstract provided.",
            algorithms=json.dumps(algorithms, ensure_ascii=False, indent=2),
        )
        prompt += "\nDirectly output JSON."
        try:
            response = self.chat(
                prompt, temperature=0.01, max_tokens=2048, model=self.model
            )
            payload = self._parse_json_response(response)
            candidate = payload.get("algorithms", payload)
            if isinstance(candidate, list) and candidate:
                return candidate
        except Exception as exc:
            logger.warning("⚠️ Algorithm alignment failed: %s", exc)
        return algorithms

    def _synthesize_reference_summaries(
        self,
        topic: str,
        best_entry: Dict[str, Any],
        algorithm: List[Dict[str, Any]],
        raw_references: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not raw_references:
            return []
        prompt = PROMPTS["reference_grounding"].format(
            topic=topic,
            idea_title=best_entry.get("title", ""),
            idea_abstract=best_entry.get("abstract", ""),
            algorithm=json.dumps(algorithm, ensure_ascii=False, indent=2),
            references=json.dumps(raw_references, ensure_ascii=False, indent=2),
        )
        prompt += "\n Directly output JSON."
        try:
            response = self.chat(prompt, temperature=0.01, max_tokens=4096, model=self.model)
            payload = self._parse_json_response(response)
            candidate = payload.get("reference_papers", payload)
            if isinstance(candidate, list):
                return candidate
        except Exception as exc:
            logger.warning("⚠️ Reference synthesis failed: %s", exc)
        return raw_references

    def _suggest_datasets(
        self,
        topic: str,
        best_entry: Dict[str, Any],
        algorithm: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not references:
            return []
        prompt = PROMPTS["dataset_grounding"].format(
            topic=topic,
            idea_title=best_entry.get("title", ""),
            idea_abstract=best_entry.get("abstract", ""),
            algorithm=json.dumps(algorithm, ensure_ascii=False, indent=2),
            references=json.dumps(references, ensure_ascii=False, indent=2),
        )
        try:
            response = self.chat(prompt, temperature=0.01, max_tokens=4096, model=self.model)
            payload = self._parse_json_response(response)
            candidate = payload.get("datasets", payload)
            if isinstance(candidate, list):
                return candidate
        except Exception as exc:
            logger.warning("⚠️ Dataset synthesis failed: %s", exc)
        return []

    def _persist_final_idea(self, best_entry: Dict[str, Any]) -> None:
        topic = self.memory["topic"][-1] if self.memory["topic"] else "unspecified topic"
        raw_refs = collect_reference_material(self.memory.get("references", []))
        algorithm = self._build_algorithm_spec(best_entry, topic, raw_refs)
        references = self._synthesize_reference_summaries(topic, best_entry, algorithm, raw_refs)
        datasets = self._suggest_datasets(topic, best_entry, algorithm, references)
        payload = {
            "title": best_entry.get("title"),
            "abstract": best_entry.get("abstract"),
            "algorithm": algorithm,
            "reference_papers": references,
            "datasets": datasets,
            "mcts_evolution": build_mcts_evolution(best_entry),
        }
        self.memory["idea_result"] = payload
        try:
            with open(self.idea_result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved idea result to {self.idea_result_path}")
        except OSError as exc:
            logger.error(f"⚠️ Failed to persist idea_result.json: {exc}")

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """
        LLM responses occasionally include code fences or extra commentary.
        This helper strips the noise and extracts the first JSON object/array well-formed enough for json.loads.
        """
        text = (raw or "").strip()
        if not text:
            raise ValueError("Empty response")
        if text.startswith("```"):
            fence_end = text.find("\n")
            if fence_end != -1:
                text = text[fence_end + 1 :]
            if text.endswith("```"):
                text = text[: -3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for idx, ch in enumerate(text):
                if ch in "{[":
                    try:
                        parsed, _ = decoder.raw_decode(text[idx:])
                        return parsed
                    except json.JSONDecodeError:
                        continue
        raise ValueError(f"Unable to parse JSON from response: {text[:200]}")
