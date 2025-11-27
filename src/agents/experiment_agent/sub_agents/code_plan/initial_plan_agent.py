"""
Initial Plan Agent - Creates first code implementation plan.

This agent handles Scenario 1: First-time code planning based on
pre-analysis output (PreAnalysisOutput).
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def create_initial_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create initial code planning agent.

    Args:
        model: The model to use for the agent
        working_dir: Working directory with reference codebases
        tools: List of tool functions

    Returns:
        Agent instance configured for initial planning
    """

    instructions = f"""You are the System Architect for a machine learning research project.
Your goal is to translate a theoretical research analysis into a concrete, actionable software architecture and implementation plan.

### ENVIRONMENT & CONTEXT
- **Project Root**: `{working_dir}/project`
- **Execution Context**: All Python code will run with `{working_dir}/project` as the current working directory (PYTHONPATH root).
- **Resources**:
  - `../repos/`: Reference implementations (Read-only).
  - `../dataset_candidate/`: Available datasets (Read-only).

### PROCEDURE

1. **RECONNAISSANCE (Mandatory)**
   Before planning, you must verify available resources to ground your plan in reality.
   - Scan `{working_dir}/dataset_candidate` to confirm dataset paths and structure.
   - Scan `{working_dir}/repos` to identify reusable patterns or model architectures.

2. **ARCHITECTURE DESIGN**
   Generate a comprehensive technical design document.

   **A. File Structure Strategy**
   - Design a clean, modular Python project structure (data/, models/, training/, configs/, utils/).
   - **Constraint**: DO NOT plan any `tests/` directory or test files. Testing is handled by an external QA process.
   - **Constraint**: Ensure all imports are designed relative to the Project Root.
   - **Constraint**: The project structure must be FLAT within the Project Root. DO NOT create a top-level package directory (e.g., do not create 'dasvr_project/models', just 'models/').


   **B. Implementation Checklist**
   Create a step-by-step implementation roadmap.
   - **Granularity**: Each step should be an atomic, verifiable task (1-3 files).
   - **Dependency**: Logical order (Utils -> Data -> Model -> Train).
   - **Step 1 is Fixed**: You MUST strictly define Step 1 as "Create Complete Project Structure" (creating all directories and `__init__.py` files).

### OUTPUT REQUIREMENTS
Provide a detailed textual project plan containing:
1. **Research Summary**: Brief summary of what is being implemented.
2. **Key Innovations**: The core novelties to be implemented.
3. **File Structure**: A complete tree or list of files and directories (excluding tests).
4. **Dataset Plan**: How data will be loaded and processed.
5. **Model Plan**: Architecture details.
6. **Training Plan**: Training loop and optimization strategy.
7. **Testing Plan**: Evaluation metrics and validation strategy (NOT unit tests).
8. **Implementation Checklist**: A numbered list of implementation steps. For each step, specify the files to create/modify and acceptance criteria.
9. **Notes & Challenges**: Implementation notes and potential risks.

Use clear headings for each section.
"""

    agent = Agent(
        name="Initial Code Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
initial_plan_agent = create_initial_plan_agent()
