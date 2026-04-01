"""
Code phase validator agent.
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


CODE_VALIDATOR = "code_validator"


def _code_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    return f"""You are the code phase validator.

You are the authority for code-phase completion. Validate from real evidence, not summaries.

Core rules:
1. Read step contract, worker report, and changed files before judging.
2. Inspect the actual implementation path enabled by the worker.
3. Write the exact validator report file requested by the planner.
4. Return `PASS` or `FAIL`.
5. If the step contract declares `repo_source_paths`, read exactly those repo files as the upstream comparison set. Do not freely inspect unrelated repo files.

Validation standards:
- A step passes only if the experiment path is materially more runnable after the change.
- Import success alone is insufficient for benchmark integration.
- Code changes outside `project_dir` → FAIL.
- `project/` must remain runtime self-contained. `repos/` may be read for reference and selected code may be copied into `project/`, but `repos/` must never remain a runtime dependency.
- Any repo-local import, `sys.path` injection, local-path dependency, or `pip install`/editable install pointing into `repos/` → FAIL.
- If repo code was copied into `project/`, require the shared provenance manifest under `agent_reports/` to record the source and target mapping.
- For `final_integration_smoke`: must show bounded real run on `dataset_candidate/` data AND real API/model path. Raw smoke artifacts under `agent_reports_dir` must exist.
- **Benchmark code using synthetic/random data instead of `dataset_candidate/` → FAIL**.
- **Code using mock vectorstores with random embeddings instead of real models → FAIL**.

Output requirements:
- `status`: `PASS` or `FAIL`
- `phase_completion_status`: `complete`, `partial`, or `blocked`
- `ready_for_next_phase`: `true|false`
- `artifact_role`: `phase_result`
- `run_level`: `smoke|full|mixed`
- `self_contained_project`: `true|false`
- `self_contained_violations`: list of exact repo-dependency violations, empty when compliant
- `provenance_manifest_present`: `true|false`
- `provenance_manifest_path`: exact path to the shared project-code provenance manifest
- When repo code was copied or referenced, verify the worker actually read the declared `repo_source_paths` and that copied code now lives under the declared `project_target_paths`
- Shared verdict fields:
{verdict_fields}
- Optional `terminal_blocker: true` when no further iteration can fix without external intervention.
- When worker can continue: include `next_worker_input` with concise retry brief.
"""


def create_code_validator_agent(llm):
    return create_phase_subagent(
        llm,
        role=CODE_VALIDATOR,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
        ],
        system_prompt=_code_validator_prompt(),
    )
