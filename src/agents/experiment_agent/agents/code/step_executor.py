"""
Backward-compatible code step executor identifier.
"""

from __future__ import annotations


CODE_STEP_EXECUTOR = "code_step_executor"


def create_code_step_executor_agent(llm):
    _ = llm
    return {"role": CODE_STEP_EXECUTOR}
