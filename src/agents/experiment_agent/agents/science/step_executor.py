"""
Science phase step executor agents.
"""

from __future__ import annotations

from openhands.sdk.subagent import register_agent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool

from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent
from src.agents.experiment_agent.agents.science.validator import (
    ABLATION_SCIENCE_VALIDATOR,
    STANDARD_SCIENCE_VALIDATOR,
    create_ablation_science_validator_agent,
    create_standard_science_validator_agent,
)
from src.agents.experiment_agent.agents.science.worker import (
    ABLATION_SCIENCE_WORKER,
    STANDARD_SCIENCE_WORKER,
    create_ablation_science_worker_agent,
    create_standard_science_worker_agent,
)


STANDARD_SCIENCE_STEP_EXECUTOR = "standard_science_step_executor"
ABLATION_SCIENCE_STEP_EXECUTOR = "ablation_science_step_executor"
_STANDARD_SCIENCE_STEP_SUBAGENTS_REGISTERED = False
_ABLATION_SCIENCE_STEP_SUBAGENTS_REGISTERED = False


def _standard_science_step_executor_prompt() -> str:
    return """You are the standard science step executor.

Your job is to complete exactly one standard-science step contract. The runtime controls overall phase progression; inside this assignment you are responsible for the local `standard_science_worker` / `standard_science_validator` retry loop for that one step.

Core loop:
1. Read the assigned standard-science step contract before taking action.
2. Launch `standard_science_worker` for the current step.
3. Launch `standard_science_validator` for the same step.
4. If the validator returns `PASS`, stop and report the step complete.
5. If the validator returns `FAIL` and `terminal_blocker` is not true, pass the validator's exact `findings`, `required_fixes`, and any `next_worker_input` back to the same step's next worker attempt.
6. Repeat until `PASS` or until the step contract's `max_repair_rounds` limit is reached.
7. If `terminal_blocker=true`, stop immediately and report the blocker upward.

Hard rules:
- Stay inside one standard-science step contract at a time.
- Do not treat human-readable summaries as primary evidence.
- Do not allow the worker to skip a missing raw-output fix requested by the validator.
- Do not mark the step complete without a validator-backed `PASS`.
- Keep the `step_contract_path`, `executor_report_path`, worker reports, and validator reports at the flat filenames declared in the contract.
- Prefer resuming the same worker task when the task tool supports it; otherwise create a follow-up worker task that explicitly references the prior worker report and validator report.

Required final output:
- A concise summary stating whether the step finished with `PASS`, `BLOCKED`, or `FAILED_AFTER_RETRIES`.
- The final validator report path.
- The executor report path that was updated for this step.
- The number of repair attempts used.
"""


def _ablation_science_step_executor_prompt() -> str:
    return """You are the ablation science step executor.

Your job is to complete exactly one ablation-science step contract. The runtime controls overall phase progression; inside this assignment you are responsible for the local `ablation_science_worker` / `ablation_science_validator` retry loop for that one step.

Core loop:
1. Read the assigned ablation-science step contract before taking action.
2. Launch `ablation_science_worker` for the current step.
3. Launch `ablation_science_validator` for the same step.
4. If the validator returns `PASS`, stop and report the step complete.
5. If the validator returns `FAIL` and `terminal_blocker` is not true, pass the validator's exact `findings`, `required_fixes`, and any `next_worker_input` back to the same step's next worker attempt.
6. Repeat until `PASS` or until the step contract's `max_repair_rounds` limit is reached.
7. If `terminal_blocker=true`, stop immediately and report the blocker upward.

Hard rules:
- Stay inside one ablation-science step contract at a time.
- Do not let the worker silently drift away from the assigned canonical component.
- Do not allow the worker to skip a missing raw-output or missing method-context fix requested by the validator.
- Do not mark the step complete without a validator-backed `PASS`.
- Keep the `step_contract_path`, `executor_report_path`, worker reports, and validator reports at the flat filenames declared in the contract.
- Prefer resuming the same worker task when the task tool supports it; otherwise create a follow-up worker task that explicitly references the prior worker report and validator report.

Required final output:
- A concise summary stating whether the step finished with `PASS`, `BLOCKED`, or `FAILED_AFTER_RETRIES`.
- The final validator report path.
- The executor report path that was updated for this step.
- The number of repair attempts used.
"""


def create_standard_science_step_executor_agent(llm):
    global _STANDARD_SCIENCE_STEP_SUBAGENTS_REGISTERED
    if not _STANDARD_SCIENCE_STEP_SUBAGENTS_REGISTERED:
        for name, factory, description in (
            (STANDARD_SCIENCE_WORKER, create_standard_science_worker_agent, "Executes one standard-science contract."),
            (STANDARD_SCIENCE_VALIDATOR, create_standard_science_validator_agent, "Validates one standard-science contract and issues PASS/FAIL."),
        ):
            try:
                register_agent(name=name, factory_func=factory, description=description)
            except ValueError:
                pass
        _STANDARD_SCIENCE_STEP_SUBAGENTS_REGISTERED = True
    return create_phase_subagent(
        llm,
        role=STANDARD_SCIENCE_STEP_EXECUTOR,
        tool_names=[
            TaskToolSet.name,
            TaskTrackerTool.name,
            FileEditorTool.name,
        ],
        system_prompt=_standard_science_step_executor_prompt(),
    )


def create_ablation_science_step_executor_agent(llm):
    global _ABLATION_SCIENCE_STEP_SUBAGENTS_REGISTERED
    if not _ABLATION_SCIENCE_STEP_SUBAGENTS_REGISTERED:
        for name, factory, description in (
            (ABLATION_SCIENCE_WORKER, create_ablation_science_worker_agent, "Executes one ablation-science contract."),
            (ABLATION_SCIENCE_VALIDATOR, create_ablation_science_validator_agent, "Validates one ablation-science contract and issues PASS/FAIL."),
        ):
            try:
                register_agent(name=name, factory_func=factory, description=description)
            except ValueError:
                pass
        _ABLATION_SCIENCE_STEP_SUBAGENTS_REGISTERED = True
    return create_phase_subagent(
        llm,
        role=ABLATION_SCIENCE_STEP_EXECUTOR,
        tool_names=[
            TaskToolSet.name,
            TaskTrackerTool.name,
            FileEditorTool.name,
        ],
        system_prompt=_ablation_science_step_executor_prompt(),
    )
