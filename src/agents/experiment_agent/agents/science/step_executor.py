"""
Backward-compatible science step executor identifiers.
"""

from __future__ import annotations


STANDARD_SCIENCE_STEP_EXECUTOR = "standard_science_step_executor"
ABLATION_SCIENCE_STEP_EXECUTOR = "ablation_science_step_executor"


def create_standard_science_step_executor_agent(llm):
    _ = llm
    return {"role": STANDARD_SCIENCE_STEP_EXECUTOR}


def create_ablation_science_step_executor_agent(llm):
    _ = llm
    return {"role": ABLATION_SCIENCE_STEP_EXECUTOR}
