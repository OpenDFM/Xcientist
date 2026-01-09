from agent.base import AgentBase
from agent import get_logger

logger = get_logger()
logger.info("🤖 Initializing LigAgent...")
import os
from typing import Any, Dict, Literal, List, Optional
from agent.tools import TOOLS
from agent.memory import MEMORY_FORMAT, memory_init
from agent.prompts import PROMPTS
import json
import http.client

MODEL = os.getenv("MODEL")


class LigAgent(AgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = [
            "knowledge_aquisition",
            "advanced_analysis",
            "idea_generation",
            "idea_evaluation",
            "re_analysis_replan",
        ]
        self.tools = TOOLS
        self.memory = memory_init()

    def select_action(self, observation: Any) -> str:
        prompt = PROMPTS["action_selection"].format(
            action_space=self.action_space, step=observation
        )
        response = self.chat(prompt, model=MODEL)
        return response.strip()

    def perform_action(self, action: str, **kwargs) -> Any:
        """
        Dispatch an action to the corresponding method, forwarding positional and keyword arguments.
        """
        logger.info(f"🚀 Performing action: {action}...")

        if action == "knowledge_aquisition":
            logger.info(
                "🔍 Due to API and web request rate limits, this process may take some time..."
            )
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

    def _parse_json(self, text: str, context: str) -> Optional[Dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            cleaned = cleaned[start : end + 1]
        try:
            return json.loads(cleaned)
        except Exception as exc:
            logger.error(f"JSON parse failed in {context}: {exc}; raw: {text}")
            return None

    def knowledge_aquisition(
        self, search_type: Literal["paper_search", "website"] = "paper_search"
    ) -> str:
        if search_type == "paper_search":
            search_keywords = self.memory["retrieval_keywords"][-1]
            try:
                papers = self.run_tool(
                    name="semantic_search", query=search_keywords, limit=5
                )
                logger.info("📄 Found Papers:")
                new_papers = []
                if papers and len(papers) > 0:
                    for i, paper in enumerate(papers, 1):
                        title = (
                            paper.get("title")
                            if isinstance(paper, dict)
                            else str(paper)
                        )
                        logger.info(f"📄 {i}. {title}")
                        abstract = (
                            paper.get("abstract", "No abstract available.")
                            if isinstance(paper, dict)
                            else "No abstract available."
                        )
                        new_papers.append({"title": title, "abstract": abstract})
                    step = f"\nIn this knowledge_aquisition action, I acquired {len(papers)} papers about '{search_keywords}'."
                    self.memory["references"].append(new_papers)

                else:
                    step = f"\nIn this knowledge_aquisition action, I searched for papers about '{search_keywords}' but found none."
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
                # import pdb; pdb.set_trace()
                print(data)
                # step = f"\nIn this knowledge_aquisition action, I attempted to search on website about '{search_keywords}', and received the following response: {data}"
                step = f"\nIn this knowledge_aquisition action, I acquired several papers about '{search_keywords}'."
            return step

    def advanced_analysis(self, **kwargs) -> None:
        # import pdb; pdb.set_trace()
        prompt = PROMPTS["advanced_analysis"].format(
            topic=self.memory["topic"][-1], papers=self.memory["references"][-1]
        )
        raw = self.chat(prompt, model=MODEL)
        response = self._parse_json(raw, "advanced_analysis")
        if response is None:
            step = f"\nIn this advanced_analysis action, model returned non-JSON text:\n{raw}"
            return step
        logger.info(f"📝 Advanced Analysis Result:\n{response}")
        self.memory["analysis"].append(response)
        step = f"\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: {response['tldr']}"
        return step

    def idea_generation(self, **kwargs) -> None:
        prompt = PROMPTS["idea_generation"].format(
            topic=self.memory["topic"][-1],
            analysis=self.memory["analysis"],
            ideas=self.memory["idea_pool"],
        )
        raw = self.chat(prompt, model=MODEL)
        response = self._parse_json(raw, "idea_generation")
        if response is None:
            step = f"\nIn this idea_generation action, model returned non-JSON text:\n{raw}"
            return step
        self.memory["idea_pool"].append(response)
        step = f"\nIn this idea_generation action, I generated new research ideas:\n💡 {response}"
        return step

    def idea_evaluation(self, **kwargs) -> None:
        if len(self.memory["idea_pool"]) == 0:
            step = "\nIn this idea_evaluation action, no idea is available to evaluate. Skipping."
            return step
        prompt = PROMPTS["idea_evaluation"].format(
            topic=self.memory["topic"][-1], idea=self.memory["idea_pool"][-1]
        )
        raw = self.chat(prompt, model=MODEL)
        response = self._parse_json(raw, "idea_evaluation")
        if response is None:
            step = f"\nIn this idea_evaluation action, model returned non-JSON text:\n{raw}"
            return step
        self.memory["idea_pool"][-1]["evaluation"] = response
        step = f"\nIn this idea_evaluation action, I evaluated the generated research ideas:\n✅ {response}"
        return step

    def re_analysis_replan(self, **kwargs) -> None:
        prompt = PROMPTS["re_analysis_replan"].format(
            topic=self.memory["topic"][-1],
            idea=self.memory["idea_pool"][-1],
            last_queries=self.memory["retrieval_keywords"],
            topics=self.memory["topic"],
        )
        raw = self.chat(prompt, model=MODEL)
        response = self._parse_json(raw, "re_analysis_replan")
        if response is None:
            step = f"\nIn this re_analysis_replan action, model returned non-JSON text:\n{raw}"
            return step
        self.memory["topic"].append(response["new_topic"])
        self.memory["retrieval_keywords"].append(response["search_keywords"])
        step = f"\nIn this re_analysis_replan action, I replanned my research topic to '{response['new_topic']}' and decided to search for new information using keywords '{response['search_keywords']}'."
        return step
