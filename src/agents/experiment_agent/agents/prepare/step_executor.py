"""
Backward-compatible prepare step executor identifier.
"""

from __future__ import annotations


PREPARE_STEP_EXECUTOR = "prepare_step_executor"


def create_prepare_step_executor_agent(llm):
    _ = llm
    return {"role": PREPARE_STEP_EXECUTOR}
