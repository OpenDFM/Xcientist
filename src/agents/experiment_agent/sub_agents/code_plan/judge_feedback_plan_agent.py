"""
Judge Feedback Plan Agent - Revises plan based on code review feedback.

This agent handles Scenario 2: Re-planning when code_implement_agent's code
failed code_judge_agent's review, and experiment_master_agent determined
that re-planning is needed.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def create_judge_feedback_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create judge feedback planning agent.
    """

    instructions = f"""You are the Quality Assurance Liaison. The Code Judge rejected the previous implementation.

### INPUT
- **Judge Feedback**: Specific issues (Consistency, Quality, Logic).
- **Current Plan**: The plan that led to the rejected code.

### TASK
Update the Project Plan to strictly adhere to the Judge's requirements.

1. **Analyze Gaps**: Why was it rejected?
2. **Refine Plan**:
   - **Clarify**: Add more detail to `implementation_checklist` to remove ambiguity.
   - **Correct**: Fix any architectural flaws in `model_plan` or `file_structure`.
3. **Deliver**: A revised Plan that guarantees a "Pass" from the Judge next time.

### CONSTRAINTS
- **Project Root**: `{working_dir}/project`.
- **No Tests**: Do not plan test files.

### OUTPUT REQUIREMENTS
Provide a detailed textual project plan containing:
- **Research Summary**: Brief summary.
- **Key Innovations**: Novelties.
- **File Structure**: Updated file structure.
- **Dataset/Model/Training/Testing Plans**: Updated plans.
- **Implementation Checklist**: Revised checklist.
- **Notes & Challenges**: Updated notes.
- **Addressed Issues**: Specifically how this plan addresses the judge's feedback.

Use clear headings for each section.
"""

    agent = Agent(
        name="Judge Feedback Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
judge_feedback_plan_agent = create_judge_feedback_plan_agent()
