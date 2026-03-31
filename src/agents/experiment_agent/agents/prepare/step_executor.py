"""
Prepare phase step executor agent.
"""

from __future__ import annotations

from openhands.sdk.subagent import register_agent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskToolSet
from openhands.tools.task_tracker import TaskTrackerTool

from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent
from src.agents.experiment_agent.agents.prepare.validator import (
    PREPARE_VALIDATOR,
    create_prepare_validator_agent,
)
from src.agents.experiment_agent.agents.prepare.worker import (
    PREPARE_DATASET_WORKER,
    PREPARE_ENV_WORKER,
    PREPARE_MODEL_WORKER,
    PREPARE_REPO_WORKER,
    PREPARE_SYNTHESIS_WORKER,
    create_prepare_dataset_worker_agent,
    create_prepare_env_worker_agent,
    create_prepare_model_worker_agent,
    create_prepare_repo_worker_agent,
    create_prepare_synthesis_worker_agent,
)


PREPARE_STEP_EXECUTOR = "prepare_step_executor"
_PREPARE_STEP_SUBAGENTS_REGISTERED = False


def _prepare_step_executor_prompt() -> str:
    return """You are the prepare phase step executor.

Your job is to complete exactly one prepare-stage contract. The runtime controls overall phase progression; inside this assignment you are responsible for the local stage worker / `prepare_validator` retry loop for that one stage.

Core loop:
1. Read the assigned stage contract before doing anything.
2. Route the stage to the correct worker: `prepare_repo_worker`, `prepare_env_worker`, `prepare_dataset_worker`, `prepare_model_worker`, or `prepare_synthesis_worker`.
3. Launch `prepare_validator` against the resulting worker report and produced artifacts.
4. If the validator returns `PASS`, stop and report the stage complete.
5. If the validator returns `FAIL` or `PARTIAL` and `terminal_blocker` is not true, send the validator's concrete `findings`, `required_fixes`, and any `next_worker_input` back to the same stage's next worker attempt.
6. Repeat until the validator passes or the stage contract's `max_repair_rounds` limit is reached.
7. If `terminal_blocker=true`, stop immediately and report the blocker upward.

Hard rules:
- You own only one stage contract at a time.
- Do not skip validator feedback.
- Do not declare completion without a validator-backed `PASS`.
- Do not ask the planner to manually summarize validator failures for the worker. Pass the validator report contents through directly.
- Keep the `stage_contract_path`, `executor_report_path`, worker reports, and validator reports at the contract paths updated on every attempt.
- Prefer resuming the same worker task when the task tool supports it; otherwise create a follow-up worker task that explicitly references the prior worker report and validator report.

Required final output:
- A concise summary stating whether the stage finished with `PASS`, `BLOCKED`, or `FAILED_AFTER_RETRIES`.
- The final validator report path.
- The executor report path that was updated for this stage.
- The number of repair attempts used.
"""


def create_prepare_step_executor_agent(llm):
    global _PREPARE_STEP_SUBAGENTS_REGISTERED
    if not _PREPARE_STEP_SUBAGENTS_REGISTERED:
        for name, factory, description in (
            (PREPARE_REPO_WORKER, create_prepare_repo_worker_agent, "Executes the prepare repos-stage contract."),
            (PREPARE_ENV_WORKER, create_prepare_env_worker_agent, "Executes the prepare env-stage contract."),
            (PREPARE_DATASET_WORKER, create_prepare_dataset_worker_agent, "Executes the prepare dataset-stage contract."),
            (PREPARE_MODEL_WORKER, create_prepare_model_worker_agent, "Executes the prepare model-stage contract."),
            (PREPARE_SYNTHESIS_WORKER, create_prepare_synthesis_worker_agent, "Executes the prepare synthesis-stage contract."),
            (PREPARE_VALIDATOR, create_prepare_validator_agent, "Validates one prepare-stage contract and issues PASS/FAIL."),
        ):
            try:
                register_agent(name=name, factory_func=factory, description=description)
            except ValueError:
                pass
        _PREPARE_STEP_SUBAGENTS_REGISTERED = True
    return create_phase_subagent(
        llm,
        role=PREPARE_STEP_EXECUTOR,
        tool_names=[
            TaskToolSet.name,
            TaskTrackerTool.name,
            FileEditorTool.name,
        ],
        system_prompt=_prepare_step_executor_prompt(),
        mcp_servers=["tavily"],
    )
