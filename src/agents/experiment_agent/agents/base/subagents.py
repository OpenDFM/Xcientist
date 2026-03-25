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
    return Agent(
        llm=llm,
        tools=build_tool_list(tool_names),
        agent_context=agent_context,
        system_prompt_filename=template_path,
    )
