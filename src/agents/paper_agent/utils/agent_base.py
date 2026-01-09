import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from agents import Agent, ModelSettings, Runner

from src.agents.paper_agent.utils.config import setup_openai_api
from src.agents.paper_agent.utils.hooks import create_hooks


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, agent_type: str, model: str, max_turns: int = 999, verbose: bool = True):
        self.agent_type = str(agent_type or "Agent")
        self.model = str(model or "")
        self.max_turns = int(max_turns)
        self.verbose = bool(verbose)
        self.hooks = create_hooks(agent_type=self.agent_type, show_llm_responses=self.verbose, show_tools=self.verbose, show_tool_args=self.verbose)

    @abstractmethod
    def _build_system_prompt(self, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def _build_user_prompt(self, **kwargs) -> str:
        raise NotImplementedError

    def _get_tools(self) -> List:
        return []

    async def run(self, user_prompt: str, system_prompt: Optional[str] = None, tools: Optional[List] = None, **kwargs) -> Any:
        if system_prompt is None:
            system_prompt = self._build_system_prompt(**kwargs)
        if tools is None:
            tools = self._get_tools()

        _ = setup_openai_api(model=self.model, verbose=False)

        agent = Agent(
            name=self.agent_type,
            model=self.model,
            instructions=system_prompt,
            tools=tools,
            model_settings=ModelSettings(),
        )
        return await Runner.run(agent, user_prompt, max_turns=self.max_turns, hooks=self.hooks)


