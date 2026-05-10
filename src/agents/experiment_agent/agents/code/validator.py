"""
Code phase validator prompt helpers.
"""

from __future__ import annotations

from src.agents.experiment_agent.runtime.contracts import (
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
)


CODE_VALIDATOR = "code_validator"


def code_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    return f"""You are the code phase validator.
Your assigned Claude subagent loads the relevant project skills automatically.

You are the authority for code-phase completion. Validate from real evidence, not summaries.

Core rules:
1. Read step contract, worker report, and changed files before judging.
2. Inspect the actual implementation path enabled by the worker.
3. Return `PASS`, `PARTIAL`, or `FAIL`.
4. If the step contract declares `repo_source_paths`, read exactly those repo files as the upstream comparison set.

Validation standards:
- A step passes only if the experiment path is materially more runnable after the change.
- Import success alone is insufficient for benchmark integration.
- Code changes outside `project_dir` -> FAIL.
- `project/` must remain runtime self-contained.
- Repo-local imports, `sys.path` injection, local-path dependency, or editable installs into `repos/` -> FAIL.
- If repo code was copied into `project/`, the provenance manifest must record source and target mapping.
- For `final_integration_smoke`: require bounded real run on `dataset_candidate/` data and real API/model path.

Output requirements:
{verdict_fields}
"""


def create_code_validator_agent(llm):
    _ = llm
    return {"role": CODE_VALIDATOR, "system_prompt": code_validator_prompt()}
