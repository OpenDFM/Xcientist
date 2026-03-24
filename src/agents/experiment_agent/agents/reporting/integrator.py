"""
Ablation report integrator agent.
"""

from __future__ import annotations

from openhands.sdk import Agent
from openhands.sdk.subagent import register_agent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.subagents import build_tool_list
from src.agents.experiment_agent.skills import get_worker_agent_context


EXPERIMENT_ABLATION_REPORT_INTEGRATOR = "experiment_ablation_report_integrator"
_ABLATION_REPORT_INTEGRATOR_REGISTERED = False


def _ablation_report_integrator_prompt() -> str:
    return """You are the ablation report integrator.

Your only job is to write the final `ablation_results.json` from existing idea and experiment evidence.

Core rules:
1. Read `idea.json` carefully and treat `idea.json.components` as the only canonical source for component names, count, and order.
2. Read the actual ablation experiment records under `agent_reports/` and the raw evidence under `results/ablation/`.
3. Do not rely on file existence alone. Read the substantive content of plans, worker reports, validator reports, summaries, and raw evidence indices/logs.
4. Write `ablation_results.json` with exactly this top-level shape and no extra top-level keys:
   {
     "components": {
       "<canonical_component_name>": {
         "result": "positive|negative|inconclusive",
         "metric": "...",
         "value": "...",
         "confidence": 0.0,
         "analysis": "...",
         "method_context": "..."
       }
     },
     "summary": {
       "feasible": true,
       "confidence": 0.0,
       "key_findings": ["...", "..."]
     }
   }
5. `components` must be an object keyed by the exact canonical component names from `idea.json.components`, in the same order, with no extras and no omissions.
6. Every component entry must include:
   - `result`
   - `metric`
   - `value`
   - `confidence`
   - `analysis`
   - `method_context`
7. `method_context` must be copied verbatim from `idea.json.components[<index>].description` (or equivalent description field) for that canonical component - it is the idea's original description of what this component does.
8. Do not include legacy sections such as `idea_title`, `verdict`, `results`, `recommendations`, `experiment_metadata`, or any other extra keys.
9. Do not invent unsupported findings. If evidence is insufficient for any canonical component, fail rather than fabricating.
10. Also write a concise integrator report JSON to the requested report path. It must record the source evidence files used, whether integration succeeded, and any missing-data blocker.
11. Return a concise summary after writing the files.
"""


def create_ablation_report_integrator_agent(llm) -> Agent:
    from openhands.sdk.context import AgentContext
    worker_context = get_worker_agent_context("ablation_report_integrator")
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
            system_message_suffix=_ablation_report_integrator_prompt(),
            load_public_skills=False,
        ),
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
