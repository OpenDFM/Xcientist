import os
from typing import Any, Callable, Dict, Optional, Union

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only in limited test environments
    OpenAI = None  # type: ignore[assignment]

from src.agents.idea_agent.utils.core.chat_transport import (
    extract_response_text,
    normalize_chat_completions_kwargs,
    normalize_responses_kwargs,
    resolve_chat_transport,
)

ToolType = Union[Callable[..., Any], object]

class AgentBase:
    """
    Minimal agent base with an action space, a tool space and a chat model.
    - action_space: list of allowed action names (strings)
    - tools: mapping from tool name to a callable or an object exposing a `run` method
    - chat_model: chat model to interact with the user
    """

    def __init__(self,
                    actions: Optional[Dict[str, str]] = None,
                    tools: Optional[Dict[str, ToolType]] = None,
                    chat_model: Optional[Any] = None) -> None:
        self.action_space: Dict[str, str] = actions or {}
        self.tools: Dict[str, ToolType] = tools or {}
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.api_style = os.getenv("OPENAI_API_STYLE", "auto")
        if chat_model is not None:
            self.chat_model = chat_model
        else:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required to construct the default AgentBase chat client."
                )
            self.chat_model = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=self.base_url,
            )
   
    def run_tool(self, name: str, *args, **kwargs) -> Any:
        """
        Run a registered tool. If the tool is callable, call it.
        If it is an object with `run`, call `tool.run(*args, **kwargs)`.
        Raises KeyError if no such tool.
        The result is automatically appended to the dialogue as an agent message (optional).
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        tool = self.tools[name]
        if callable(tool):
            result = tool(*args, **kwargs)
        else:
            run_fn = getattr(tool, "run", None)
            if callable(run_fn):
                result = run_fn(*args, **kwargs)
            else:
                raise TypeError(f"Tool '{name}' is not callable and has no 'run' method")
        return result

    def perform_action(self, action: str, **params) -> Any:
        """
        Perform a chosen action. 
        Should be implemented by subclasses.
        """
        if action not in self.action_space:
            raise ValueError(f"Action '{action}' not in action_space")
        raise NotImplementedError("perform_action must be implemented by subclasses")

    def chat(self, prompt: str, model: str="gpt-5-mini", **kwargs) -> str:
        """
        Interact with the chat model using the given prompt.
        Returns the response text.
        """
        transport = resolve_chat_transport(self._current_base_url(), self.api_style)
        if transport == "chat_completions":
            request_kwargs = normalize_chat_completions_kwargs(kwargs)
            response = self.chat_model.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs,
            )
            return extract_response_text(response)

        request_kwargs = normalize_responses_kwargs(kwargs)
        response = self.chat_model.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            **request_kwargs,
        )
        return extract_response_text(response)

    def _current_base_url(self) -> Optional[str]:
        client_base_url = getattr(self.chat_model, "base_url", None)
        return str(client_base_url or self.base_url or "").strip() or None
