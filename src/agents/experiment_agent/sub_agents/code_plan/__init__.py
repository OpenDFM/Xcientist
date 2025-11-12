"""Code planning agent module."""

from src.agents.experiment_agent.sub_agents.code_plan.code_plan_agent import (
    create_code_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def get_recommended_tools():
    """
    Get recommended tools for code plan agent.
    
    Note: Uses read-only file tools since code plan should only
    read reference code, not write files.

    Returns:
        Dictionary mapping scenario types to tool lists
    """
    from src.agents.experiment_agent.tools import (
        FILE_TOOLS_READONLY,
        CODE_ANALYSIS_TOOLS,
        REPOSITORY_TOOLS,
    )

    # Only read-only tools for planning - no write/create/delete
    tools = FILE_TOOLS_READONLY + CODE_ANALYSIS_TOOLS + REPOSITORY_TOOLS

    return {
        "initial": tools,
        "judge_feedback": tools,
        "error_feedback": tools,
        "analysis_feedback": tools,
    }


__all__ = ["create_code_plan_agent", "CodePlanOutput", "get_recommended_tools"]
