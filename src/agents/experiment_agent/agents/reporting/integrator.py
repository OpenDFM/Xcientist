"""
Ablation report integrator agent.
"""

from __future__ import annotations

import os

from openhands.sdk import Agent
from openhands.sdk.subagent import register_agent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.subagents import (
    build_tool_list,
    default_builtin_tool_names,
)
from src.agents.experiment_agent.skills import get_worker_agent_context


EXPERIMENT_ABLATION_REPORT_INTEGRATOR = "experiment_ablation_report_integrator"
ABLATION_REPORT_INTEGRATOR_TEMPLATE = "ablation_report_integrator.j2"
_ABLATION_REPORT_INTEGRATOR_REGISTERED = False


def create_ablation_report_integrator_agent(llm) -> Agent:
    from openhands.sdk.context import AgentContext
    worker_context = get_worker_agent_context("ablation_report_integrator")
    # __file__ is .../src/agents/experiment_agent/agents/reporting/integrator.py
    # 3 dirname calls gives .../src/agents/experiment_agent
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    template_path = os.path.join(base_dir, "prompts", ABLATION_REPORT_INTEGRATOR_TEMPLATE)
    return Agent(
        llm=llm,
        tools=build_tool_list(
            [
                TerminalTool.name,
                FileEditorTool.name,
                TaskTrackerTool.name,
            ]
        ),
        agent_context=AgentContext(
            skills=worker_context.skills,
            load_public_skills=False,
        ),
        system_prompt_filename=template_path,
        include_default_tools=default_builtin_tool_names(),
    )


def register_ablation_report_integrator() -> None:
    global _ABLATION_REPORT_INTEGRATOR_REGISTERED
    if _ABLATION_REPORT_INTEGRATOR_REGISTERED:
        return
    try:
        register_agent(
            name=EXPERIMENT_ABLATION_REPORT_INTEGRATOR,
            factory_func=create_ablation_report_integrator_agent,
            description="Reads idea and ablation evidence, then writes the final ablation_results.json artifact.",
        )
    except ValueError:
        pass
    _ABLATION_REPORT_INTEGRATOR_REGISTERED = True
