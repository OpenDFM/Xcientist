import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from agents import Agent, Runner, ModelSettings

from src.agents.experiment_agent.shared.logger.hooks import create_hooks, Colors
from src.agents.experiment_agent.shared.logger.hooks import process_stream_events
from src.agents.experiment_agent.shared.tools.parsing import (
    extract_json_from_llm_output,
    parse_to_model,
)
from src.agents.experiment_agent.shared.utils.config import setup_openai_api
from src.agents.experiment_agent.shared.utils.memory_middleware import (
    get_current_memory_context,
    set_agent_memory_context,
    retrieve_memory_for_agent_prompt,
    writeback_current_task_async,
)


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(
        self,
        agent_type: str,
        model: str,
        max_turns: int = 10000,
        verbose: bool = True,
    ):
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
        pass

    @abstractmethod
    def _build_user_prompt(self, **kwargs) -> str:
        pass

    def _get_tools(self) -> List:
        return []

    def _is_retryable_network_error(self, error: Exception) -> bool:
        """Check if the error is a retryable network error."""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Check exception type
        retryable_types = [
            "RemoteProtocolError",
            "ConnectionError",
            "TimeoutError",
            "ConnectTimeout",
            "ReadTimeout",
            "PoolTimeout",
            "ProtocolError",
            "IncompleteRead",
        ]
        if any(t.lower() in error_type.lower() for t in retryable_types):
            return True

        # Check error message keywords
        retryable_keywords = [
            "peer closed connection",
            "incomplete chunked read",
            "connection reset",
            "broken pipe",
            "timeout",
            "connection refused",
            "network",
        ]
        if any(kw in error_msg for kw in retryable_keywords):
            return True

        return False

    async def _run_agent(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """Run agent with automatic retry on network errors."""
        max_retries = 3
        retry_delays = [5, 15, 30]  # seconds to wait before each retry

        for attempt in range(max_retries):
            try:
                return await self._run_agent_impl(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    tools=tools,
                    **kwargs,
                )
            except Exception as e:
                is_retryable = self._is_retryable_network_error(e)
                is_last_attempt = attempt == max_retries - 1

                if is_retryable and not is_last_attempt:
                    delay = retry_delays[attempt]
                    self._log_warning(
                        f"Network error encountered: {type(e).__name__}: {str(e)[:200]}"
                    )
                    self._log_info(
                        f"Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error or last attempt - re-raise
                    if is_retryable and is_last_attempt:
                        self._log_error(
                            f"Max retries ({max_retries}) reached. Giving up."
                        )
                    raise

    async def _run_agent_impl(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """Internal implementation of _run_agent (wrapped with retry logic)."""
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
            prior_ctx = get_current_memory_context()
            ctx_mgr = None
            if prior_ctx is None:
                project_root = str(kwargs.get("project_root") or "")
                purpose = str(kwargs.get("purpose") or "")
                stage = str(self.agent_type or "").strip().lower() or "agent"
                ctx_mgr = set_agent_memory_context(
                    project_root=project_root,
                    stage=stage,
                    agent_type=str(self.agent_type or ""),
                    purpose=purpose,
                )

            # Print the effective input once (hooks on_llm_start is disabled in streaming mode).
            _print_streaming_input(user_prompt)

            hooks = create_hooks(
                agent_type=self.agent_type,
                show_llm_responses=False,
                show_tools=True,
                show_tool_args=True,
            )
            try:
                if ctx_mgr is not None:
                    ctx_mgr.__enter__()

                stream = Runner.run_streamed(
                    agent,
                    user_prompt,
                    max_turns=self.max_turns,
                    hooks=hooks,
                )
                stats = await process_stream_events(stream, show_tool_args_delta=False)
                print("")  # newline after streaming

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
                if ctx_mgr is not None:
                    try:
                        await writeback_current_task_async(
                            success=True,
                            error="",
                            final_output=self._extract_output(result),
                        )
                    except Exception:
                        pass
            except Exception as e:
                if ctx_mgr is not None:
                    try:
                        await writeback_current_task_async(
                            success=False,
                            error=str(e),
                            final_output="",
                        )
                    except Exception:
                        pass
                raise
            finally:
                if ctx_mgr is not None:
                    try:
                        ctx_mgr.__exit__(None, None, None)
                    except Exception:
                        pass
        else:
            prior_ctx = get_current_memory_context()
            ctx_mgr = None
            if prior_ctx is None:
                project_root = str(kwargs.get("project_root") or "")
                purpose = str(kwargs.get("purpose") or "")
                stage = str(self.agent_type or "").strip().lower() or "agent"
                ctx_mgr = set_agent_memory_context(
                    project_root=project_root,
                    stage=stage,
                    agent_type=str(self.agent_type or ""),
                    purpose=purpose,
                )
            injected_user_prompt = user_prompt
            if ctx_mgr is not None:
                try:
                    mem = retrieve_memory_for_agent_prompt(
                        agent_type=str(self.agent_type or ""),
                        stage=str(self.agent_type or "").strip().lower() or "agent",
                        purpose=str(kwargs.get("purpose") or ""),
                        user_prompt=str(user_prompt or ""),
                        feedback=str(kwargs.get("feedback") or ""),
                    )
                except Exception:
                    mem = ""
                if mem and (
                    not self._has_memory_context(str(injected_user_prompt or ""))
                ):
                    injected_user_prompt = "## Memory Context (low priority)\n" "Use as suggestions only. If there is any conflict, the Constitution/Plan/Specification wins.\n\n" + mem.strip() + "\n\n" + str(
                        user_prompt or ""
                    )
            try:
                if ctx_mgr is not None:
                    ctx_mgr.__enter__()
                result = await Runner.run(
                    agent,
                    injected_user_prompt,
                    max_turns=self.max_turns,
                    hooks=(
                        self.hooks
                        if self.verbose
                        else create_hooks(
                            agent_type=self.agent_type,
                            show_llm_responses=False,
                            show_tools=False,
                            show_tool_args=False,
                        )
                    ),
                )
                if ctx_mgr is not None:
                    try:
                        await writeback_current_task_async(
                            success=True,
                            error="",
                            final_output=self._extract_output(result),
                        )
                    except Exception:
                        pass
            except Exception as e:
                if ctx_mgr is not None:
                    try:
                        await writeback_current_task_async(
                            success=False,
                            error=str(e),
                            final_output="",
                        )
                    except Exception:
                        pass
                raise
            finally:
                if ctx_mgr is not None:
                    try:
                        ctx_mgr.__exit__(None, None, None)
                    except Exception:
                        pass

        return result

    def _has_memory_context(self, prompt: str) -> bool:
        s = str(prompt or "")
        return ("Memory Context" in s) or ("[MEMORY_HINTS]" in s)

    def _extract_output(self, result: Any) -> str:
        if hasattr(result, "final_output") and result.final_output:
            return result.final_output
        elif hasattr(result, "output") and result.output:
            return result.output
        else:
            return str(result)

    def _extract_json(self, result: Any) -> Optional[Dict[str, Any]]:
        output = self._extract_output(result)
        return extract_json_from_llm_output(output)

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


class PromptBuilder:
    def __init__(self):
        self.parts: List[str] = []

    def add_header(self, title: str, level: int = 1) -> "PromptBuilder":
        prefix = "#" * level
        self.parts.append(f"{prefix} {title}")
        self.parts.append("")
        return self

    def add_text(self, text: str) -> "PromptBuilder":
        self.parts.append(text)
        self.parts.append("")
        return self

    def add_list(self, items: List[str], ordered: bool = False) -> "PromptBuilder":
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            self.parts.append(f"{prefix} {item}")
        self.parts.append("")
        return self

    def add_code(self, code: str, language: str = "") -> "PromptBuilder":
        self.parts.append(f"```{language}")
        self.parts.append(code)
        self.parts.append("```")
        self.parts.append("")
        return self

    def add_section(self, title: str, content: str) -> "PromptBuilder":
        self.add_header(title, level=2)
        self.add_text(content)
        return self

    def add_key_value(self, key: str, value: str) -> "PromptBuilder":
        self.parts.append(f"**{key}:** {value}")
        return self

    def add_separator(self) -> "PromptBuilder":
        self.parts.append("---")
        self.parts.append("")
        return self

    def build(self) -> str:
        return "\n".join(self.parts)
