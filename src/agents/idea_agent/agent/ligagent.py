from agent.base import AgentBase
from agent import get_logger
logger = get_logger()
logger.info("🤖 Initializing LigAgent...")
from typing import Any, Dict, Literal, List
from agent.tools import TOOLS
from agent.memory import MEMORY_FORMAT, memory_init
from agent.prompts import PROMPTS
import json


class LigAgent(AgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = ["knowledge_aquisition", "advanced_analysis", "idea_generation", "idea_evaluation", "re_analysis_replan"]
        self.tools = TOOLS
        self.memory = memory_init()

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
            return step

    def advanced_analysis(self, **kwargs) -> None:
        # import pdb; pdb.set_trace()
        prompt = PROMPTS["advanced_analysis"].format(topic=self.memory["topic"], papers=self.memory["references"][-1])
        response = json.loads(self.chat(prompt))
        logger.info(f"📝 Advanced Analysis Result:\n{response}")
        self.memory["analysis"].append(response)
        step = f"\nIn this advanced_analysis action, I analyzed the collected papers and summarized my findings: {response['tldr']}"
        return step

    def idea_generation(self, **kwargs) -> None:
        prompt = PROMPTS["idea_generation"].format(topic=self.memory["topic"], analysis=self.memory["analysis"])
        response = json.loads(self.chat(prompt))
        self.memory["idea_pool"].append(response)
        step = f"\nIn this idea_generation action, I generated new research ideas:\n💡 {response}"
        return step

    def idea_evaluation(self, **kwargs) -> None:
        prompt = PROMPTS["idea_evaluation"].format(topic=self.memory["topic"], idea=self.memory["idea_pool"][-1])
        response = json.loads(self.chat(prompt))
        self.memory["idea_pool"][-1]["evaluation"] = response
        step = f"\nIn this idea_evaluation action, I evaluated the generated research ideas:\n✅ {response}"
        return step

    def re_analysis_replan(self, **kwargs) -> None:
        prompt = PROMPTS["re_analysis_replan"].format(topic=self.memory["topic"], idea=self.memory["idea_pool"][-1], last_queries=self.memory["retrieval_keywords"])
        response = self.chat(prompt)
        self.memory["retrieval_keywords"].append(response)
        step = f"\nIn this re_analysis_replan action, I replanned my research and decided to search for new information using keywords: {response}."
        return step