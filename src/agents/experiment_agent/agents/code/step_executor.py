"""
Code phase step executor agent.
"""

from __future__ import annotations

from openhands.sdk.subagent import register_agent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool

from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent
from src.agents.experiment_agent.agents.code.validator import (
    CODE_VALIDATOR,
    create_code_validator_agent,
)
from src.agents.experiment_agent.agents.code.worker import (
    CODE_WORKER,
    create_code_worker_agent,
)


CODE_STEP_EXECUTOR = "code_step_executor"
_CODE_STEP_SUBAGENTS_REGISTERED = False


def _code_step_executor_prompt() -> str:
    return """You are the code phase step executor.

Your job is to complete exactly one code step contract. The runtime controls overall phase progression; inside this assignment you are responsible for the local `code_worker` / `code_validator` retry loop for that one step.

Core loop:
1. Read the assigned code step contract and current artifact paths.
2. Launch `code_worker` for the current step.
3. Launch `code_validator` for the same step.
4. If the validator returns `PASS`, stop and report the step complete.
5. If the validator returns `FAIL` and `terminal_blocker` is not true, send the validator's exact `findings`, `required_fixes`, and any `next_worker_input` back to the same step's next worker attempt.
6. Repeat until `PASS` or until the step contract's `max_repair_rounds` limit is reached.
7. If `terminal_blocker=true`, stop immediately and report the blocker upward.

Hard rules:
- Do not advance to any other step.
- Do not bypass the validator by editing summaries yourself.
- Do not mark the step complete unless the validator has returned `PASS`.
- Pass validator feedback back to the worker with minimal paraphrasing.
- Keep the current step's `step_contract_path`, `executor_report_path`, worker reports, and validator reports at the flat filenames declared in the contract. Do not redirect multiple steps into the same shared phase report path.
- Prefer resuming the same worker task when the task tool supports it; otherwise create a follow-up worker task that explicitly references the prior worker report and validator report.

Required final output:
- A concise summary stating whether the step finished with `PASS`, `BLOCKED`, or `FAILED_AFTER_RETRIES`.
- The final validator report path.
- The executor report path that was updated for this step.
- The number of repair attempts used.
"""


def create_code_step_executor_agent(llm):
    global _CODE_STEP_SUBAGENTS_REGISTERED
    if not _CODE_STEP_SUBAGENTS_REGISTERED:
        for name, factory, description in (
            (CODE_WORKER, create_code_worker_agent, "Implements one code step contract."),
            (CODE_VALIDATOR, create_code_validator_agent, "Validates one code step contract and issues PASS/FAIL."),
        ):
            try:
                register_agent(name=name, factory_func=factory, description=description)
            except ValueError:
                pass
        _CODE_STEP_SUBAGENTS_REGISTERED = True
    return create_phase_subagent(
        llm,
        role=CODE_STEP_EXECUTOR,
        tool_names=[
            TaskToolSet.name,
            TaskTrackerTool.name,
            FileEditorTool.name,
        ],
        system_prompt=_code_step_executor_prompt(),
    )
