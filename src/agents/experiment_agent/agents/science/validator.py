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

You are the authority for standard-science completion. Judge from raw evidence and artifacts, not self-reported summaries.

Core rules:
1. Read planner contract, worker report, and produced artifacts before judging.
2. Inspect raw evidence first: commands, output files, logs, bindings.
3. Write the exact validator report file requested by the planner.
4. Return `PASS` or `FAIL`.
5. When failing, provide exact missing evidence or corrective actions.

Validation standards:
- Science passes only if the assigned benchmark path actually ran on declared prepared targets and produced promised outputs.
- Summary JSON/markdown are supporting artifacts, not primary proof.
- **Runs not using `dataset_candidate/` data → FAIL**.
- **Runs using synthetic/random data instead of real data → FAIL**.
- Raw outputs outside the declared standard-results subtree → FAIL.

Output requirements:
- `status`: `PASS` or `FAIL`
- `phase_completion_status`: `complete`, `partial`, or `blocked`
- `ready_for_next_phase`: `true|false`
- `artifact_role`: `phase_result`
- `run_level`: `smoke|full|mixed`
- Shared verdict fields:
{verdict_fields}
- Optional `terminal_blocker: true` when no further iteration can fix without external intervention.
- When worker can continue: include `next_worker_input` with concise retry brief.

Rejection rules:
- Runs that only edit result files without underlying execution evidence → FAIL.
- Runs with mismatched model or dataset bindings → FAIL.
"""


def _ablation_science_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    ablation_result_fields = format_field_bullets(ABLATION_COMPONENT_RESULT_FIELDS)
    return f"""You are the ablation science validator.

You are the authority for ablation-science completion. Judge from raw evidence and artifacts, not self-reported summaries.

Core rules:
1. Read planner contract, worker report, and produced artifacts before judging.
2. Inspect raw evidence first: commands, output files, logs, bindings.
3. Write the exact validator report file requested by the planner.
4. Return `PASS` or `FAIL`.
5. When failing, provide exact missing evidence or corrective actions.

Validation standards:
- Ablation passes only if the assigned canonical component was seriously tested and conclusion is supported by evidence.
- Component identity must match `idea.json.components` name exactly.
- `method_context` must describe the exact ablated/degraded variant.
- **Runs not using `dataset_candidate/` data → FAIL**.
- **Runs using synthetic/random data → FAIL**.
- Raw outputs outside the declared ablation-results subtree → FAIL.

Output requirements:
- `status`: `PASS` or `FAIL`
- `phase_completion_status`: `complete`, `partial`, or `blocked`
- `ready_for_next_phase`: `true|false`
- `artifact_role`: `phase_result`
- `run_level`: `smoke|full|mixed`
- Shared verdict fields:
{verdict_fields}
- Each ablation step must also include:
{ablation_result_fields}
- Optional `terminal_blocker: true` when no further iteration can fix without external intervention.
- When worker can continue: include `next_worker_input` with concise retry brief.

Rejection rules:
- Runs with renamed, merged, split, missing, extra, or reordered canonical components → FAIL.
- Ablation claims lacking explicit method context or raw evidence → FAIL.
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
