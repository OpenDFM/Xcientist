import os
from typing import Callable, Dict, Any, Optional, List, Union
from openai import OpenAI

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
                    chat_model: Optional[OpenAI] = None) -> None:
        self.action_space: Dict[str, str] = actions or {}
        self.tools: Dict[str, ToolType] = tools or {}
        self.chat_model: OpenAI = chat_model or OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
   
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

    def select_action(self, observation: Any) -> str:
        """
        Choose an action given an observation.
        Default implementation is a stub; override in subclasses.
        """
        raise NotImplementedError("select_action must be implemented by subclasses")

    def select_memory(self, observation: Any) -> str:
        """
        Choose a memory given an observation.
        Default implementation is a stub; override in subclasses.
        """
        raise NotImplementedError("select_memory must be implemented by subclasses")

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
        response = self.chat_model.responses.create(
            model=model,
            input=prompt,
            **kwargs
        )
        return response.output_text