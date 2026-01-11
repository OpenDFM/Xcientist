import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import httpx
import httpcore
import openai

from agents import Agent, ModelSettings, Runner, RawResponsesStreamEvent

from src.agents.paper_agent.utils.config import setup_openai_api
from src.agents.paper_agent.utils.hooks import create_hooks, Colors


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
        
        # Retry logic for network/protocol errors
        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Use streaming to prevent timeouts on long responses
                stream_result = Runner.run_streamed(agent, user_prompt, max_turns=self.max_turns, hooks=self.hooks)
                
                # Streaming Output Handler
                # Unified "Thinking" visualization logic
                is_printing_block = False
                
                # Wrap stream iteration with heartbeat to detect hangs
                iterator = stream_result.stream_events().__aiter__()
                while True:
                    try:
                        # Wait up to 30s for the next event
                        event = await asyncio.wait_for(iterator.__anext__(), timeout=30.0)
                    except asyncio.TimeoutError:
                        print(f"{Colors.FAIL} ... (Waiting for LLM response) ... {Colors.ENDC}", flush=True)
                        continue
                    except StopAsyncIteration:
                        break
                    
                    is_text_event = False
                    delta = ""
                    
                    if isinstance(event, RawResponsesStreamEvent):
                        etype = getattr(event.data, "type", "")
                        if etype in ["response.reasoning_text.delta", "response.reasoning_summary_text.delta", "response.output_text.delta"]:
                            delta = getattr(event.data, "delta", "")
                            if delta:
                                is_text_event = True
                    
                    if is_text_event:
                        if not is_printing_block:
                            # Start of a new text block (Turn Start or Post-Tool)
                            print(f"\n{Colors.OKCYAN}🧠 Thinking:{Colors.ENDC}\n{Colors.WARNING}", end="", flush=True)
                            is_printing_block = True
                        
                        print(delta, end="", flush=True)
                    else:
                        # Non-text event (e.g., Tool call, Step end, etc.)
                        # Close the current block if it was open
                        if is_printing_block:
                            print(f"{Colors.ENDC}\n", flush=True)
                            is_printing_block = False
                
                # End of stream cleanup
                if is_printing_block:
                    print(f"{Colors.ENDC}\n", flush=True)
                
                return stream_result
            except (
                httpx.RemoteProtocolError, 
                httpcore.RemoteProtocolError, 
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.ConnectTimeout,
                httpx.PoolTimeout,
                openai.APIConnectionError, 
                openai.InternalServerError, 
                openai.RateLimitError,
                openai.APITimeoutError
            ) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to run agent after {max_retries} attempts. Last error: {e}")
                    raise
                
                wait_time = 2.0 * (1.5 ** attempt) 
                logger.warning(f"Network/Protocol error during agent run (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)


