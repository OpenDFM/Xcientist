"""Code judge agent module."""

from src.agents.experiment_agent.sub_agents.code_judge.code_judge_agent import (
    create_code_judge_agent,
)
from src.agents.experiment_agent.sub_agents.code_judge.output_schemas import (
    CodeJudgeOutput,
    CodeIssue,
    UnitTestSpec,
)


def get_recommended_tools():
    """
    Get recommended tools for code judge agent.

    Returns:
        List of tools for code review and unit test execution
    """
    from src.agents.experiment_agent.tools import (
        FILE_TOOLS,
        CODE_ANALYSIS_TOOLS,
        LOCAL_EXECUTION_TOOLS,
    )

    return FILE_TOOLS + CODE_ANALYSIS_TOOLS + LOCAL_EXECUTION_TOOLS


__all__ = [
    "create_code_judge_agent",
    "CodeJudgeOutput",
    "CodeIssue",
    "UnitTestSpec",
    "get_recommended_tools",
]
