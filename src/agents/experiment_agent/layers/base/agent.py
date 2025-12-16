"""
Base Agent - Common base class for all agents.

Provides:
- Unified initialization (hooks, model config)
- Common prompt building utilities
- Result extraction helpers
- Logging integration

All specific agents (Architect, Manager, Worker, Integrator) inherit from this.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from agents import Agent, Runner, ModelSettings

from src.agents.experiment_agent.shared.logger.hooks import create_hooks, Colors
from src.agents.experiment_agent.shared.logger.hooks import process_stream_events
from src.agents.experiment_agent.shared.tools.parsing import extract_json_from_llm_output, parse_to_model
from src.agents.experiment_agent.shared.utils.config import setup_openai_api


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all SuperAgent agents.

    Provides common functionality:
    - Model and hook initialization
    - Prompt building utilities
    - Result extraction
    - Logging
    """

    def __init__(
        self,
        agent_type: str,
        model: str,
        max_turns: int = 10000,
        verbose: bool = True,
    ):
        """
        Initialize the base agent.

        Args:
            agent_type: Type identifier for logging (e.g., "Architect", "Worker")
            model: Model name to use
            max_turns: Maximum tool-calling turns
            verbose: Enable verbose output
        """
        self.agent_type = agent_type
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose

        # Create hooks for verbose output
        self.hooks = create_hooks(
            agent_type=agent_type,
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=verbose,
        )

        logger.debug(f"{agent_type} initialized with model={model}")

    @abstractmethod
    def _build_system_prompt(self, **kwargs) -> str:
        """
        Build the system prompt for this agent.

        Must be implemented by subclasses.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def _build_user_prompt(self, **kwargs) -> str:
        """
        Build the user prompt for this agent.

        Must be implemented by subclasses.

        Returns:
            User prompt string
        """
        pass

    def _get_tools(self) -> List:
        """
        Get the tools available to this agent.

        Override in subclasses to provide specific tools.

        Returns:
            List of tool functions
        """
        return []

    async def _run_agent(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """
        Run the agent with given prompts.

        Args:
            user_prompt: The user/task prompt
            system_prompt: Optional system prompt (uses _build_system_prompt if not provided)
            tools: Optional tools list (uses _get_tools if not provided)

        Returns:
            Agent result object
        """
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

        if self.verbose:
            logger.info(f"{self.agent_type}: Running agent...")

        def _truncate_for_console(text: str, max_chars: int = 2000) -> str:
            if not isinstance(text, str):
                return str(text)
            if len(text) <= max_chars:
                return text
            return (
                text[:max_chars]
                + "\n... (truncated, total tokens: "
                + str(len(text))
                + ")"
            )

        def _print_streaming_input(text: str) -> None:
            # Pretty-print similar to VerboseRunHooks, but only once (streaming mode).
            header = f"📤 LLM Request (Turn 1) | {self.agent_type}"
            width = 78
            print(
                f"\n{Colors.BOLD}{Colors.MAGENTA}{'┌' + '─' * width + '┐'}{Colors.ENDC}"
            )
            pad = width - len(header) - 2
            print(
                f"{Colors.BOLD}{Colors.MAGENTA}│ {header}{' ' * max(0, pad)}│{Colors.ENDC}"
            )
            print(
                f"{Colors.BOLD}{Colors.MAGENTA}{'└' + '─' * width + '┘'}{Colors.ENDC}"
            )

            print(f"{Colors.OKCYAN}📥 Input:{Colors.ENDC}\n")
            content = _truncate_for_console(text).rstrip()
            if content:
                # Light indent for readability
                for line in content.splitlines():
                    print(f"{Colors.OKBLUE}{line}{Colors.ENDC}")
            print("")

        # Stream model output text in real time only for selected agents (debuggability).
        # Keep all other agents' behavior unchanged.
        # We disable hook-based LLM response printing in streaming mode to avoid duplicate output.
        stream_whitelist = {"CodeArchitect", "ExpArchitect"}
        should_stream = (self.agent_type or "") in stream_whitelist
        if should_stream and self.verbose:
            # Print the effective input once (hooks on_llm_start is disabled in streaming mode).
            _print_streaming_input(user_prompt)

            hooks = create_hooks(
                agent_type=self.agent_type,
                show_llm_responses=False,
                show_tools=True,
                show_tool_args=True,
            )
            stream = Runner.run_streamed(
                agent,
                user_prompt,
                max_turns=self.max_turns,
                hooks=hooks,
            )
            stats = await process_stream_events(stream, show_tool_args_delta=False)
            print("")  # newline after streaming

            # If the model mostly called tools first, it may not emit ResponseTextDelta until the end.
            # Make sure the final output is still visible.
            try:
                final_output = getattr(stream, "final_output", None)
            except Exception:
                final_output = None
            if (not stats) or int(stats.get("text_chars", 0) or 0) == 0:
                if isinstance(final_output, str) and final_output.strip():
                    print(f"{Colors.OKCYAN}📤 Output:{Colors.ENDC}")
                    print(
                        f"{Colors.OKGREEN}{_truncate_for_console(final_output, max_chars=8000)}{Colors.ENDC}\n"
                    )

            result = stream
        else:
            result = await Runner.run(
                agent,
                user_prompt,
                max_turns=self.max_turns,
                hooks=self.hooks if self.verbose else None,
            )

        return result

    def _extract_output(self, result: Any) -> str:
        """
        Extract text output from agent result.

        Args:
            result: Agent result object

        Returns:
            Output string
        """
        if hasattr(result, "final_output") and result.final_output:
            return result.final_output
        elif hasattr(result, "output") and result.output:
            return result.output
        else:
            return str(result)

    def _extract_json(self, result: Any) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from agent result.

        Args:
            result: Agent result object

        Returns:
            Parsed JSON dictionary, or None
        """
        output = self._extract_output(result)
        return extract_json_from_llm_output(output)

    def _log_info(self, message: str):
        """Log an info message with agent prefix."""
        if self.verbose:
            print(f"{self.agent_type}: {message}")
        logger.info(f"{self.agent_type}: {message}")

    def _log_error(self, message: str):
        """Log an error message with agent prefix."""
        print(f"{self.agent_type}: ❌ {message}")
        logger.error(f"{self.agent_type}: {message}")

    def _log_success(self, message: str):
        """Log a success message with agent prefix."""
        if self.verbose:
            print(f"{self.agent_type}: ✅ {message}")
        logger.info(f"{self.agent_type}: {message}")

    def _log_warning(self, message: str):
        """Log a warning message with agent prefix."""
        if self.verbose:
            print(f"{self.agent_type}: ⚠️ {message}")
        logger.warning(f"{self.agent_type}: {message}")


class PromptBuilder:
    """
    Utility class for building structured prompts.
    """

    def __init__(self):
        self.parts: List[str] = []

    def add_header(self, title: str, level: int = 1) -> "PromptBuilder":
        """Add a markdown header."""
        prefix = "#" * level
        self.parts.append(f"{prefix} {title}")
        self.parts.append("")
        return self

    def add_text(self, text: str) -> "PromptBuilder":
        """Add plain text."""
        self.parts.append(text)
        self.parts.append("")
        return self

    def add_list(self, items: List[str], ordered: bool = False) -> "PromptBuilder":
        """Add a list of items."""
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            self.parts.append(f"{prefix} {item}")
        self.parts.append("")
        return self

    def add_code(self, code: str, language: str = "") -> "PromptBuilder":
        """Add a code block."""
        self.parts.append(f"```{language}")
        self.parts.append(code)
        self.parts.append("```")
        self.parts.append("")
        return self

    def add_section(self, title: str, content: str) -> "PromptBuilder":
        """Add a section with title and content."""
        self.add_header(title, level=2)
        self.add_text(content)
        return self

    def add_key_value(self, key: str, value: str) -> "PromptBuilder":
        """Add a key-value pair."""
        self.parts.append(f"**{key}:** {value}")
        return self

    def add_separator(self) -> "PromptBuilder":
        """Add a horizontal separator."""
        self.parts.append("---")
        self.parts.append("")
        return self

    def build(self) -> str:
        """Build the final prompt string."""
        return "\n".join(self.parts)
