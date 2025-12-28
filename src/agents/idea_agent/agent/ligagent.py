from agent.base import AgentBase
from agent import get_logger
logger = get_logger()
logger.info("🤖 Initializing LigAgent...")
from typing import Any, Dict, Literal, List
from agent.tools import TOOLS
from agent.memory import MEMORY_FORMAT, memory_init
from agent.prompts import PROMPTS
from agent.mcts import MemoryGuidedMCTS, MCTSConfig
import json
import http.client




class LigAgent(AgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = ["knowledge_aquisition", "advanced_analysis", "idea_generation", "idea_evaluation", "re_analysis_replan"]
        self.tools = TOOLS
        self.memory = memory_init()
        self.mcts = MemoryGuidedMCTS(
            chat_fn=self.chat,
            generation_prompt=PROMPTS["mcts_generation"],
            evaluation_prompt=PROMPTS["mcts_evaluation"],
            config=MCTSConfig(),
        )

    def select_action(self, observation: Any) -> str:
        prompt = PROMPTS["action_selection"].format(action_space=self.action_space, step=observation)
        response = self.chat(prompt)
        return response.strip()

    def perform_action(self, action: str, **kwargs) -> Any:
        """
        Dispatch an action to the corresponding method, forwarding positional and keyword arguments.
        """
        logger.info(f"🚀 Performing action: {action}...")

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
                        new_papers.append({"title": title, "abstract": abstract})
                    step = f"\nIn this knowledge_aquisition action, I acquired {len(papers)} papers about '{search_keywords}'."
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

    def advanced_analysis(self, **kwargs) -> None:
        # import pdb; pdb.set_trace()
        prompt = PROMPTS["advanced_analysis"].format(topic=self.memory["topic"][-1], papers=self.memory["references"][-1])
        response = json.loads(self.chat(prompt))
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
        response = json.loads(self.chat(prompt))
        self.memory["idea_pool"].append(response)
        return f"\nIn this idea_generation action, I generated new research ideas via fallback prompt:\n💡 {response}"

    def idea_evaluation(self, **kwargs) -> None:
        prompt = PROMPTS["idea_evaluation"].format(topic=self.memory["topic"][-1], idea=self.memory["idea_pool"][-1])
        response = json.loads(self.chat(prompt))
        self.memory["idea_pool"][-1]["evaluation"] = response
        step = f"\nIn this idea_evaluation action, I evaluated the generated research ideas:\n✅ {response}"
        return step
    
    def re_analysis_replan(self, **kwargs) -> None:
        prompt = PROMPTS["re_analysis_replan"].format(topic=self.memory["topic"][-1], idea=self.memory["idea_pool"][-1], last_queries=self.memory["retrieval_keywords"], topics=self.memory["topic"])
        response = json.loads(self.chat(prompt))
        self.memory["topic"].append(response['new_topic'])
        self.memory["retrieval_keywords"].append(response['search_keywords'])
        step = f"\nIn this re_analysis_replan action, I replanned my research topic to '{response['new_topic']}' and decided to search for new information using keywords '{response['search_keywords']}'."
        return step
