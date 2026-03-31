"""
Shared helpers for phase-local worker and validator agents.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from openhands.sdk import Agent, AgentContext
from openhands.sdk.context import Skill
from openhands.sdk.tool import Tool

from src.agents.experiment_agent.skills import get_worker_agent_context

DEFAULT_INCLUDE_DEFAULT_TOOLS: tuple[str, ...] = ("FinishTool", "ThinkTool")


def default_builtin_tool_names() -> list[str]:
    return list(DEFAULT_INCLUDE_DEFAULT_TOOLS)


def build_tool_list(tool_names: Iterable[str]) -> List[Tool]:
    unique_names: List[str] = []
    for name in tool_names:
        if name not in unique_names:
            unique_names.append(name)
    return [Tool(name=name) for name in unique_names]


def create_phase_subagent(
    llm,
    *,
    role: str,
    tool_names: Iterable[str],
    system_prompt: str,
    extra_skills: Optional[List[Skill]] = None,
    mcp_servers: Optional[Iterable[str]] = None,
    workspace_root: Optional[str] = None,
) -> Agent:
    # __file__ is .../src/agents/experiment_agent/agents/base/subagents.py
    # 3 dirname calls gives .../src/agents/experiment_agent
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    template_path = os.path.join(base_dir, "prompts", f"{role}.j2")

    base_context = get_worker_agent_context(role)
    agent_context = AgentContext(
        skills=[*base_context.skills, *(extra_skills or [])],
        load_public_skills=False,
    )
    agent_kwargs = dict(
        llm=llm,
        tools=build_tool_list(tool_names),
        agent_context=agent_context,
        include_default_tools=default_builtin_tool_names(),
    )
    if os.path.isfile(template_path):
        agent_kwargs["system_prompt_filename"] = template_path
    else:
        agent_kwargs["system_prompt"] = system_prompt
    from src.agents.experiment_agent.agents.base.agent import build_experiment_mcp_config

    mcp_config = build_experiment_mcp_config(
        workspace_root=workspace_root or os.environ.get("EXPERIMENT_AGENT_WORKSPACE_DIR") or os.getcwd(),
        allowed_servers=list(mcp_servers or []),
    )
    if mcp_config:
        agent_kwargs["mcp_config"] = mcp_config
    return Agent(
        **agent_kwargs,
    )
