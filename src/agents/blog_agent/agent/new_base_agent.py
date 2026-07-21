"""
New BaseAgent using OpenHands SDK.

Features:
- OpenHands SDK for agent execution
- Config loaded from config.yaml
- Network error retry logic
- MiniMax model detection
- Async support with callbacks
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Callable
from openhands.sdk import LLMConvertibleEvent
from openhands.sdk.utils.async_utils import AsyncCallbackWrapper
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.context import AgentContext, Skill, KeywordTrigger
from openhands.sdk.context.skills import load_skills_from_dir, load_public_skills

logger = logging.getLogger(__name__)

# Load config
from src.config import load_config

_config = load_config().get("blog", {})

# =============================================================================
# Config - loaded from config.yaml
# =============================================================================
_minimax_config = _config.get("minimax", {})
OPENAI_CONFIG = _config.get("openai", {})
GEMINI_CONFIG = _config.get("gemini", {})
MINIMAX_API_KEY = _minimax_config.get("api_key", "")
MINIMAX_API_BASE = _minimax_config.get("base_url", "https://api.minimaxi.com/v1")
MINIMAX_MODELS = ["MiniMax-M2.1", "MiniMax-M2.5"]
OPENAI_API_KEY = OPENAI_CONFIG.get("api_key", "")
OPENAI_API_BASE: Optional[str] = OPENAI_CONFIG.get("base_url", "")
GEMINI_API_KEY = GEMINI_CONFIG.get("api_key", "")
GEMINI_API_BASE: Optional[str] = GEMINI_CONFIG.get("base_url", "")
GEMINI_MODELS = ["gemini"]
# Default values from config
DEFAULT_MODEL = _config.get("model", "MiniMax-M2.5")


# =============================================================================
# Model detection and config
# =============================================================================
def is_minimax_model(model_name: str) -> bool:
    """Check if model is a MiniMax model."""
    if not model_name:
        return False
    return any(m.lower() in model_name.lower() for m in MINIMAX_MODELS)


def is_gemini_model(model_name: str) -> bool:
    """Check if model is a Gemini model."""
    if not model_name:
        return False
    return any(m.lower() in model_name.lower() for m in GEMINI_MODELS)


def get_openhands_config(model: str) -> dict:
    '''Preferred: MiniMax model. Default: OpenAI model.'''
    if is_minimax_model(model):
        return {
            "api_key": MINIMAX_API_KEY,
            "model": 'minimax/'+model,
            "base_url": MINIMAX_API_BASE,
        }
    elif is_gemini_model(model):
        return {
            "api_key": GEMINI_API_KEY,
            "model": 'gemini/'+model,
            "base_url": GEMINI_API_BASE,
        }
    else:
        return {
            "api_key": OPENAI_API_KEY,
            "model": 'openai/'+model,
            "base_url": OPENAI_API_BASE,
        }
 


def create_llm(model: Optional[str] = None) -> LLM:
    """
    Create an OpenHands LLM instance based on model.

    Args:
        model: Model name

    Returns:
        LLM instance configured for the model
    """
    config = get_openhands_config(model)

    return LLM(**config)


# =============================================================================
# BaseAgent using OpenHands SDK
# =============================================================================
class BaseAgent(ABC):
    """
    BaseAgent using OpenHands SDK.

    Features:
    - Network error retry with exponential backoff
    - Tool registration via add_tool()
    - Async support with callbacks
    - MiniMax model detection
    - Config loaded from config.yaml
    """

    def __init__(
        self,
        agent_type: str,
        model: Optional[str] = None,
        verbose: bool = True,
        workspace: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            agent_type: Name of the agent (e.g., "Writer", "Editor")
            model: Model name (default from config)
            verbose: Whether to print verbose output
            workspace: Workspace directory
        """
        self.agent_type = agent_type
        self.model = model or DEFAULT_MODEL
        self.verbose = verbose
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        parent_dir = os.path.dirname(current_dir)
        default_workspace = os.path.join(parent_dir, "workspaces")
        self.workspace = workspace or default_workspace
        self._filter_tools_regexes: List[str] = []
        self._mcp_configs: List[dict] = []
        self._skills: List[Skill] = []  # Store skills
        self._load_public_skills: bool = False  # Whether to load public skills

        # Internal state
        self._llm: Optional[LLM] = None
        self._agent: Optional[Agent] = None
        self._conversation: Optional[Conversation] = None
        self._tools: List[Tool] = []
        self._llm_messages=[]

        # Check if MiniMax model
        self._is_minimax = is_minimax_model(self.model)
        if self._is_minimax:
            self._log_info(f"Using MiniMax model: {self.model}")

        self._log_info(f"{agent_type} initialized with model={self.model}")

    # =========================================================================
    # Abstract methods to be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _build_system_prompt(self, **kwargs) -> str:
        """
        Build the system prompt for the agent.

        Returns:
            System prompt string
        """
        pass

    # =========================================================================
    # Tool registration
    # =========================================================================

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool: OpenHands Tool instance
        """
        tool_name = getattr(tool, "name", None)
        if tool_name:
            self._tools.append(tool)
            self._log_success(f"Added tool: {tool_name}")

    def add_mcp(self, mcp_config: dict) -> None:
        """
        Add an MCP server configuration.

        Args:
            mcp_config: MCP server configuration dict
        """
        self._mcp_configs.append(mcp_config)
        server_names = list(mcp_config.get("mcpServers", {}).keys())
        self._log_success(f"Added MCP server: {server_names}")

    def add_filter_tools_regex(self, regex: str) -> None:
        """
        Add a regex pattern to filter tools.

        Args:
            regex: Regex pattern for filtering tools
        """
        self._filter_tools_regexes.append(regex)
        self._log_success(f"Added filter tools regex: {regex}")

    # =========================================================================
    # Skill registration
    # =========================================================================

    def add_skill(
        self,
        name: str,
        content: str,
        trigger: Optional[KeywordTrigger] = None,
    ) -> None:
        """
        Add a skill to the agent.

        Args:
            name: Skill name/identifier
            content: Skill content/instructions
            trigger: Optional KeywordTrigger for conditional activation
        """
        skill = Skill(name=name, content=content, trigger=trigger)
        self._skills.append(skill)
        self._log_success(f"Added skill: {name}")

    def load_skills_from_dir(self, skills_dir: str) -> None:
        """
        Load skills from a directory.

        Args:
            skills_dir: Path to skills directory
        """
        repo_skills, knowledge_skills, agent_skills = load_skills_from_dir(skills_dir)

        # Create Skill objects from loaded skill identifiers
        self._skills.extend(list(agent_skills.values()))

        self._log_success(f"Loaded {len(agent_skills)} skills from: {skills_dir}")
        

    def enable_public_skills(self) -> None:
        """Enable loading of public skills from the registry."""
        self._load_public_skills = True
        self._log_success("Public skills enabled")



    # =========================================================================
    # Core execution methods
    # =========================================================================
    async def run_async(
        self,
        user_prompt: str,
    ) -> Any:
        """
        Run the agent asynchronously.

        Args:
            user_prompt: User input prompt

        Returns:
            Agent execution result
        """
        return await self._run_impl_async(user_prompt=user_prompt)

    async def _run_impl_async(
        self,
        user_prompt: str,
    ) -> Any:
        """Internal implementation of agent execution (async)."""
        # Build prompts
        async def callback_coro(event):
            if isinstance(event, LLMConvertibleEvent):
                self._llm_messages.append(event.to_llm_message())

        self._log_info(f"{self.agent_type}: Running agent (async)...")

        # Merge MCP configs
        final_mcp_config = {}
        if self._mcp_configs:
            final_mcp_config = {"mcpServers": {}}
            for config in self._mcp_configs:
                if "mcpServers" in config:
                    final_mcp_config["mcpServers"].update(config["mcpServers"])

        # Merge filter tools regex
        final_filter_regex = "|".join(self._filter_tools_regexes) if self._filter_tools_regexes else None

        # Create LLM based on model detection
        self._llm = create_llm(model=self.model)

        # Build AgentContext with skills
        agent_context = None
        custom_skills = list(self._skills)
        public_skills_count = 0

        if self._load_public_skills:
            try:
                public_skills = load_public_skills()
                public_skills_count = len(public_skills)
            except Exception as e:
                self._log_warning(f"Failed to load public skills: {e}")

        if custom_skills or self._load_public_skills:
            agent_context = AgentContext(skills=custom_skills, load_public_skills=self._load_public_skills)
            self._log_info(f"AgentContext: {len(custom_skills)} custom skills + {public_skills_count} public skills")

        # Create agent with registered tools and skills
        self._agent = Agent(
            llm=self._llm,
            tools=self._tools,
            mcp_config=final_mcp_config,
            filter_tools_regex=final_filter_regex,
            agent_context=agent_context,
        )

        # Define blocking function to run in thread pool
        def run_conversation():
            self._conversation.send_message(user_prompt)
            self._conversation.run()

        # Run blocking conversation.run() in thread pool (non-blocking!)
        loop = asyncio.get_running_loop()
        callback = AsyncCallbackWrapper(callback_coro, loop)
        self._conversation = Conversation(
            agent=self._agent,
            workspace=self.workspace,
            callbacks=[callback],
        )
        # Get result
        await loop.run_in_executor(None, run_conversation)
        return self._extract_result()

    def _extract_result(self) -> Any:
        """从实时收集的消息列表中提取最后的结果"""
        if not self._llm_messages:
            return None
        last_msg = self._llm_messages[-1]
        content = getattr(last_msg, 'content', None)
        
        if not content:
            return str(last_msg)
        if isinstance(content, list):
            texts = []
            for item in content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
                elif isinstance(item, str):
                    texts.append(item)
            
            return "\n".join(texts) if texts else str(content)
        return content

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _log_info(self, message: str):
        if self.verbose:
            print(f"{self.agent_type}: {message}")
        logger.info(f"{self.agent_type}: {message}")

    def _log_error(self, message: str):
        print(f"{self.agent_type}: ❌ {message}")
        logger.error(f"{self.agent_type}: {message}")

    def _log_success(self, message: str):
        if self.verbose:
            print(f"{self.agent_type}: ✅ {message}")
        logger.info(f"{self.agent_type}: {message}")

    def _log_warning(self, message: str):
        if self.verbose:
            print(f"{self.agent_type}: ⚠️ {message}")
        logger.warning(f"{self.agent_type}: {message}")
