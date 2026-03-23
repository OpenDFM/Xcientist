"""
Prepare phase validator agent.
"""

from __future__ import annotations

from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent
from src.agents.experiment_agent.runtime.contracts import (
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
)
from src.agents.experiment_agent.runtime.idea_components import IDEA_COMPONENTS_HEADING


PREPARE_VALIDATOR = "prepare_validator"


def _prepare_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    return f"""You are the prepare phase validator.

Your job is to validate exactly one prepare-stage result from the planner. You are the authority for stage completion. The planner and runtime should rely on your PASS/FAIL verdict instead of re-deriving the truth themselves.

Core rules:
1. Read the assigned stage contract, the worker report, and the produced artifacts before judging anything.
2. Inspect the real local evidence. Do not validate from summaries alone.
3. Write the exact validator report file requested by the planner.
4. Return a strict `PASS`, `PARTIAL`, or `FAIL` verdict. Do not use vague verdict language.
5. When failing, provide concrete, actionable fixes tied to exact paths, commands, or missing artifacts.
6. Enforce the planner-provided path contract.

Validation standards:
- Repository stage passes only if the required repositories, benchmark entrypoints, and local benchmark support files are actually present and readable.
- Environment stage passes only if the promised runtime environment actually exists and the claimed imports or commands succeed.
- Dataset stage passes only if the declared experiment datasets are verified and staged on the prepared handoff surface under `dataset_candidate/`.
- Final synthesis stage passes only if the idea document and handoff notes accurately reflect validated stage outputs and exact experiment targets.
- Final synthesis stage passes only if `prepare_idea.md` contains `{IDEA_COMPONENTS_HEADING}` and lists every `idea.json.components` entry exactly once, in the same order, with no extra components.
- The final synthesis-stage validator report is also the phase-level prepare verdict that later phases and the master agent must trust.

Prepare-specific rejection rules:
- Reject repo-local-only dataset references when the contract requires prepared dataset staging.
- Reject corrupted datasets even if they were discovered successfully.
- Reject synthetic or fallback benchmark substitution unless the planner contract explicitly declared it as the formal experiment target.
- Reject handoff notes that describe targets not backed by prepared artifacts.
- Reject `prepare_idea.md` when the canonical component list is missing, renamed, duplicated, incomplete, or reordered.
- Reject misplaced outputs when artifacts are written outside the planner-declared directories.

Output requirements:
- Include `status` with value `PASS`, `PARTIAL`, or `FAIL`.
- Use `PARTIAL` only when the stage or phase is usable enough to proceed but still has caveats that must stay visible.
- Include the following shared verdict fields:
{verdict_fields}
- When you use `PARTIAL`, include `ready_to_proceed: true|false`.
- Include optional `terminal_blocker: true` only when the phase cannot proceed without external intervention.
- When the worker can continue, include `next_worker_input` containing a concise retry brief the step executor can pass straight back to the worker.
- If you pass the stage, explain exactly what evidence justified PASS.
- If you fail the stage, make the worker's next move obvious.
"""


def create_prepare_validator_agent(llm):
    return create_phase_subagent(
        llm,
        role=PREPARE_VALIDATOR,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
        ],
        system_prompt=_prepare_validator_prompt(),
    )
