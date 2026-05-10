"""
Minimal Claude Code-backed base abstractions for experiment-agent runtime.
"""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional

from src.agents.experiment_agent.config import (
    get_external_tool_config,
    get_workspace_config,
)
from src.agents.experiment_agent.runtime.claude_cli import ClaudeCodeRunner


def build_experiment_mcp_config(
    *,
    workspace_root: str,
    allowed_servers: Iterable[str] | None = None,
) -> Dict[str, Any] | None:
    from src.agents.experiment_agent.runtime.claude_project import (
        build_experiment_mcp_config as build_workspace_mcp_config,
    )

    _ = workspace_root, allowed_servers
    return build_workspace_mcp_config(
        workspace_cfg=get_workspace_config(),
        external_cfg=get_external_tool_config(),
        global_client_cfg={},
    )


def create_oh_llm(model: str, usage_id: str = "agent", stream: bool = False) -> Dict[str, Any]:
    _ = usage_id, stream
    return {"model": model}


def get_default_tools() -> List[str]:
    return ["Read", "Edit", "Bash"]


def get_search_tools() -> List[str]:
    return []


def get_all_tools() -> List[str]:
    return get_default_tools() + get_search_tools()


class BaseAgent(ABC):
    """
    Base class backed by Claude Code CLI execution.
    """

    def __init__(
        self,
        agent_type: str,
        model: str,
        max_turns: int = 10000,
        verbose: bool = True,
        workspace_root: str | None = None,
        persistence_dir: str | None = None,
        enable_condenser: bool = True,
        condenser_max_size: int = 20,
        condenser_keep_first: int = 2,
        resume: bool = False,
    ):
        _ = (
            max_turns,
            persistence_dir,
            enable_condenser,
            condenser_max_size,
            condenser_keep_first,
        )
        self.agent_type = agent_type
        self.model = model
        self.verbose = verbose
        self.resume = resume
        self.workspace_root = os.path.realpath(workspace_root or os.getcwd())
        self.runner = ClaudeCodeRunner(
            model=model,
            workspace_root=self.workspace_root,
            verbose=verbose,
        )

    def _read_text_file(self, path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _refresh_runtime_roots(self, workspace_root: str) -> None:
        self.workspace_root = os.path.realpath(workspace_root)
        self.runner = ClaudeCodeRunner(
            model=self.model,
            workspace_root=self.workspace_root,
            verbose=self.verbose,
        )

    def _build_mcp_config(self) -> Dict[str, Any]:
        return {"mcpServers": {}}

    async def run(
        self,
        user_prompt: str,
        system_prompt: str = "",
        agent_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        output_format: str = "text",
        cwd: Optional[str] = None,
        purpose: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        _ = tools, purpose, kwargs
        if output_schema:
            payload = await self.runner.run_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_name=agent_name,
                output_schema=output_schema,
                cwd=cwd,
            )
            return {"output": payload, "content": payload}
        text = await self.runner.run_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            cwd=cwd,
        )
        return {"output": text, "content": text}

    async def _run_agent(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.run(*args, **kwargs)

    def _extract_output(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            output = result.get("output")
            if isinstance(output, str):
                return output
            if isinstance(output, dict):
                return json.dumps(output, ensure_ascii=False, indent=2)
            content = result.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                return json.dumps(content, ensure_ascii=False, indent=2)
        return ""


class PromptBuilder:
    """Helper class for building structured prompts."""

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
