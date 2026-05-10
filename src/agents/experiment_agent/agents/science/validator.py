"""
Science phase validator prompt helpers.
"""

from __future__ import annotations

from src.agents.experiment_agent.runtime.contracts import (
    ABLATION_COMPONENT_RESULT_FIELDS,
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
)


STANDARD_SCIENCE_VALIDATOR = "standard_science_validator"
ABLATION_SCIENCE_VALIDATOR = "ablation_science_validator"


def standard_science_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    return f"""You are the standard science validator.
Your assigned Claude subagent loads the relevant project skills automatically.

Judge from raw evidence and artifacts, not self-reported summaries.

Requirements:
- Return `PASS`, `PARTIAL`, or `FAIL`.
- A run passes only if the assigned benchmark path actually ran on declared prepared targets and produced promised outputs.
- `project/` must remain runtime self-contained.
- Runs not using `dataset_candidate/` data -> FAIL.
- Raw outputs outside the declared standard-results subtree -> FAIL.

Output fields:
{verdict_fields}
"""


def ablation_science_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    ablation_fields = format_field_bullets(ABLATION_COMPONENT_RESULT_FIELDS)
    return f"""You are the ablation science validator.
Your assigned Claude subagent loads the relevant project skills automatically.

Judge from raw evidence and artifacts, not self-reported summaries.

Requirements:
- Return `PASS`, `PARTIAL`, or `FAIL`.
- The assigned component identity must match `idea.json.components` exactly.
- `method_context` must describe the exact ablated variant.
- Runs not using `dataset_candidate/` data -> FAIL.
- Raw outputs outside the declared ablation-results subtree -> FAIL.

Output fields:
{verdict_fields}

Per-step ablation fields:
{ablation_fields}
"""


def create_standard_science_validator_agent(llm):
    _ = llm
    return {"role": STANDARD_SCIENCE_VALIDATOR, "system_prompt": standard_science_validator_prompt()}


def create_ablation_science_validator_agent(llm):
    _ = llm
    return {"role": ABLATION_SCIENCE_VALIDATOR, "system_prompt": ablation_science_validator_prompt()}
