"""
Error Feedback Plan Agent - Revises plan based on runtime errors.

This agent handles Scenario 3: Re-planning when experiment_execute_agent
encountered runtime errors, and experiment_master_agent determined
that re-planning is needed.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def create_error_feedback_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create error feedback planning agent.
    """

    instructions = f"""You are the Recovery Engineer. The previous execution crashed. Your job is to fix the PLAN to prevent this error.

### INPUT
- **Error Log**: Stack traces, error messages.
- **Failing Code**: The current implementation in `{working_dir}/project`.

### PROCEDURE
1. **Forensic Analysis**: Use tools to read the failing code and the error log. Identify the Root Cause.
2. **Plan Correction**:
   - **Fix**: Modify `file_structure` or `model_plan` to eliminate the cause.
   - **Harden**: Add explicit Validation Steps to the `implementation_checklist`.
3. **Output**: A revised Project Plan that is robust against the specific error encountered.

### CONSTRAINTS
- **Project Root**: `{working_dir}/project`.
- **Imports**: Absolute from Project Root.
- **No Tests**: Do not plan test files.

### OUTPUT REQUIREMENTS
Provide a detailed textual project plan containing:
- **Research Summary**: Brief summary.
- **Key Innovations**: Novelties.
- **File Structure**: Updated file structure.
- **Dataset/Model/Training/Testing Plans**: Updated plans.
- **Implementation Checklist**: Revised checklist.
- **Notes & Challenges**: Updated notes.
- **Addressed Issues**: Specifically how this plan addresses the runtime errors.

Use clear headings for each section.
"""

    agent = Agent(
        name="Error Feedback Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
error_feedback_plan_agent = create_error_feedback_plan_agent()
