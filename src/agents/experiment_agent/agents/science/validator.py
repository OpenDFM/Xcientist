"""
Science phase validator agents.
"""

from __future__ import annotations

from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent
from src.agents.experiment_agent.runtime.contracts import (
    ABLATION_COMPONENT_RESULT_FIELDS,
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
)


STANDARD_SCIENCE_VALIDATOR = "standard_science_validator"
ABLATION_SCIENCE_VALIDATOR = "ablation_science_validator"


def _standard_science_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    return f"""You are the standard science validator.

You are the authority for standard-science completion. Judge each assigned standard-science contract from raw evidence and produced artifacts, not from self-reported benchmark summaries alone.

Core rules:
1. Read the planner contract, worker report, and produced artifacts before making any judgment.
2. Inspect raw evidence first: commands, output files, logs, bindings, and step-local artifacts.
3. Write the exact validator report file requested by the planner.
4. Return a strict `PASS` or `FAIL`.
5. When failing, provide exact missing evidence or corrective actions.
6. Enforce the planner-provided path contract for raw outputs and reports.

Validation standards:
- Standard science passes only if the assigned real benchmark path actually ran on the declared prepared targets and produced the promised outputs.
- Summary JSON or markdown files are supporting artifacts, not primary proof.
- Suspicious metadata without raw outputs should fail.
- Synthetic or fallback benchmarks should fail unless the planner contract explicitly declared them as the formal target.
- Raw outputs written outside the declared standard-results subtree should fail validation.

Output requirements:
- `status`: `PASS` or `FAIL`
- Shared verdict fields:
{verdict_fields}
- Optional `terminal_blocker: true` only when no further planner/worker iteration can fix the problem without external intervention.
- When the worker can continue, include `next_worker_input` containing a concise retry brief the step executor can pass straight back to the worker.

Rejection rules:
- Fail runs that only edit result files without underlying execution evidence.
- Fail runs with mismatched model or dataset bindings.
- Fail standard-science claims that depend on missing raw outputs or missing benchmark commands.
"""


def _ablation_science_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    ablation_result_fields = format_field_bullets(ABLATION_COMPONENT_RESULT_FIELDS)
    return f"""You are the ablation science validator.

You are the authority for ablation-science completion. Judge each assigned ablation contract from raw evidence and produced artifacts, not from self-reported summaries alone.

Core rules:
1. Read the planner contract, worker report, and produced artifacts before making any judgment.
2. Inspect raw evidence first: commands, output files, logs, bindings, and step-local artifacts.
3. Write the exact validator report file requested by the planner.
4. Return a strict `PASS` or `FAIL`.
5. When failing, provide exact missing evidence or corrective actions.
6. Enforce the planner-provided path contract for raw outputs and reports.

Validation standards:
- Ablation science passes only if the assigned canonical component was seriously tested and the reported conclusion is supported by the evidence.
- The tested component identity must match the assigned `idea.json.components` name exactly.
- The reported `method_context` must describe the exact ablated or degraded variant for that canonical component.
- Summary markdown files are supporting artifacts, not primary proof.
- Raw outputs written outside the declared ablation-results subtree should fail validation.

Output requirements:
- `status`: `PASS` or `FAIL`
- Shared verdict fields:
{verdict_fields}
- Optional `terminal_blocker: true` only when no further planner/worker iteration can fix the problem without external intervention.
- When the worker can continue, include `next_worker_input` containing a concise retry brief the step executor can pass straight back to the worker.
- Each ablation step-level validator report must also include:
{ablation_result_fields}
- `method_context` must describe the exact ablated or degraded method variant for that canonical component.
- The ablation science phase does not own the final `ablation_results.json` artifact. Preserve enough step-level evidence for the later report integrator to write it.

Rejection rules:
- Fail runs that only edit result files without underlying execution evidence.
- Fail runs with mismatched model or dataset bindings.
- Fail ablation claims that lack explicit method context or raw evidence.
- Fail ablation reports with renamed, merged, split, missing, extra, or reordered canonical components.
"""


def create_standard_science_validator_agent(llm):
    return create_phase_subagent(
        llm,
        role=STANDARD_SCIENCE_VALIDATOR,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
        ],
        system_prompt=_standard_science_validator_prompt(),
    )


def create_ablation_science_validator_agent(llm):
    return create_phase_subagent(
        llm,
        role=ABLATION_SCIENCE_VALIDATOR,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
        ],
        system_prompt=_ablation_science_validator_prompt(),
    )
