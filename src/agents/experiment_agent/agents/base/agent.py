"""OpenHarness-backed base abstractions for experiment-agent runtime."""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.runtime.openharness_runner import OpenHarnessAgentRunner


class BaseAgent(ABC):
    """
    Base class backed by vendored OpenHarness execution.
    """

    def __init__(
        self,
        agent_type: str,
        model: str,
        max_turns: Optional[int] = None,
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
        self.artifact_context: Dict[str, Any] = {}
        self.runner = OpenHarnessAgentRunner(
            model=model,
            workspace_root=self.workspace_root,
            verbose=verbose,
            artifact_context=self.artifact_context,
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
        self.runner = OpenHarnessAgentRunner(
            model=self.model,
            workspace_root=self.workspace_root,
            verbose=self.verbose,
            artifact_context=self.artifact_context,
        )

    def set_artifact_context(self, artifact_context: Optional[Dict[str, Any]]) -> None:
        self.artifact_context = dict(artifact_context or {})
        self.runner = OpenHarnessAgentRunner(
            model=self.model,
            workspace_root=self.workspace_root,
            verbose=self.verbose,
            artifact_context=self.artifact_context,
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
        _ = tools
        extra_tool_metadata = kwargs.pop("extra_tool_metadata", None)
        enable_mcp = bool(kwargs.pop("enable_mcp", False))
        reviewer_mode = str(purpose or "").strip().lower() in {"review", "reviewer", "prefinish_review"}
        runner = (
            OpenHarnessAgentRunner(
                model=self.model,
                workspace_root=self.workspace_root,
                verbose=self.verbose,
                reviewer_mode=True,
                artifact_context=self.artifact_context,
                extra_tool_metadata=extra_tool_metadata if isinstance(extra_tool_metadata, dict) else None,
                enable_mcp=False,
            )
            if reviewer_mode
            else (
                OpenHarnessAgentRunner(
                    model=self.model,
                    workspace_root=self.workspace_root,
                    verbose=self.verbose,
                    artifact_context=self.artifact_context,
                    extra_tool_metadata=extra_tool_metadata if isinstance(extra_tool_metadata, dict) else None,
                    enable_mcp=enable_mcp,
                )
                if (isinstance(extra_tool_metadata, dict) and extra_tool_metadata) or enable_mcp
                else self.runner
            )
        )
        if output_schema:
            payload = await runner.run_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_name=agent_name,
                output_schema=output_schema,
                cwd=cwd,
            )
            return {"output": payload, "content": payload}
        text = await runner.run_text(
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
