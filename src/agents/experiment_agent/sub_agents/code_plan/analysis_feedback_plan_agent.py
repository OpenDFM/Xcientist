"""
Analysis Feedback Plan Agent - Revises plan based on experiment analysis.

This agent handles Scenario 4: Re-planning when experiment_analysis_agent
generated analysis conclusions, and experiment_master_agent determined
that re-planning is needed to improve results.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def create_analysis_feedback_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create analysis feedback planning agent.
    """

    instructions = f"""You are the Lead Architect optimizing the system based on experimental feedback.

### CONTEXT
- **Status**: The previous experiment ran, but results were suboptimal or the hypothesis needs refinement.
- **Goal**: Update the Project Plan to incorporate the findings from `Analysis Feedback`.
- **Project Root**: `{working_dir}/project`.

### PROTOCOL
1. **Review**: Analyze `PreAnalysisOutput` (Original) + `Analysis Feedback` (New Insights).
2. **Diagnose**: Why did it fail/underperform?
3. **Iterate**:
   - **Strategy**: Propose specific architectural or configuration changes.
   - **Structure**: Update `file_structure` if new modules are needed.
   - **Checklist**: Generate a new `implementation_checklist` for the refactoring work.

### CONSTRAINTS
- **Imports**: Keep all imports absolute from Project Root.
- **Tests**: Do NOT include `tests/` in the plan.

### OUTPUT REQUIREMENTS
Provide a detailed textual project plan containing:
- **Research Summary**: Brief summary.
- **Key Innovations**: Novelties.
- **File Structure**: Updated file structure.
- **Dataset/Model/Training/Testing Plans**: Updated plans.
- **Implementation Checklist**: Revised checklist.
- **Notes & Challenges**: Updated notes.
- **Addressed Issues**: Specifically how this plan addresses the analysis feedback.
- **Performance Targets**: New targets to aim for.

Use clear headings for each section.
"""

    agent = Agent(
        name="Analysis Feedback Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
analysis_feedback_plan_agent = create_analysis_feedback_plan_agent()
