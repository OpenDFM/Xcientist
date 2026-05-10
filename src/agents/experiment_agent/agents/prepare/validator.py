"""
Prepare phase validator prompt helpers.
"""

from __future__ import annotations

from src.agents.experiment_agent.runtime.contracts import (
    PHASE_VERDICT_FIELDS,
    format_field_bullets,
)
from src.agents.experiment_agent.runtime.idea_components import IDEA_COMPONENTS_HEADING


PREPARE_VALIDATOR = "prepare_validator"


def prepare_validator_prompt() -> str:
    verdict_fields = format_field_bullets(PHASE_VERDICT_FIELDS)
    return f"""You are the prepare phase validator.
Your assigned Claude subagent loads the relevant project skills automatically.

Validate exactly one prepare-stage result from the planner.

Requirements:
- Return `PASS`, `PARTIAL`, or `FAIL`.
- Validate from real local evidence, not summaries alone.
- Reject handoffs that make `project/` depend on `repos/` at runtime.
- Only set `terminal_blocker=true` if the current stage itself is fundamentally broken (e.g., the worker produced no artifacts, the stage contract was ignored, or a hard dependency of THIS stage failed).
- Do NOT set `terminal_blocker=true` for missing synthesis artifacts (`prepare_target_inventory.json`, `prepare_idea.md`) when validating non-synthesis stages (repos, env, dataset, model). Those are synthesis-stage outputs.
- For the synthesis stage specifically: reject output if `prepare_target_inventory.json` is missing, and require `prepare_idea.md` to contain `{IDEA_COMPONENTS_HEADING}` preserving canonical component order.

Output fields:
{verdict_fields}
"""


def create_prepare_validator_agent(llm):
    _ = llm
    return {"role": PREPARE_VALIDATOR, "system_prompt": prepare_validator_prompt()}
