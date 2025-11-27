"""
Initial Implementation Agent - Implements code from scratch based on plan.

This agent handles the first-time code implementation based on
CodePlanOutput from code_plan_agent.
"""

from agents import Agent


def create_initial_implement_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create initial implementation agent.

    Args:
        model: The model to use for the agent
        working_dir: Working directory for code generation
        tools: List of tool functions

    Returns:
        Agent instance configured for initial implementation
    """

    instructions = f"""You are a Senior Research Engineer responsible for implementing a specific step of a machine learning project.

### TASK SCOPE
You are assigned **ONE** specific step from the `implementation_checklist`.
- **Current Step ID**: {{current_step_id}}
- **Objective**: Implement the files listed in `files_to_create` and modify `files_to_modify` for this step.
- **Constraint**: DO NOT touch files or implement features belonging to future steps.

### ENVIRONMENT
- **Project Root**: `{working_dir}` (This is your current working directory).
- **Reference Repos**: `../repos/` (Use `list_directory` and `read_file` to study them for patterns).
- **Datasets**: `../dataset_candidate/` (Use relative paths via `os.path` logic to access them).

### ENGINEERING STANDARDS

1. **Import Resolution**:
   - All code runs from Project Root.
   - Use absolute imports relative to root (e.g., `from data.loader import Loader`).
   - NEVER use the prefix `project.` or relative imports (`..`) for cross-module dependencies.

2. **File System**:
   - Adhere STRICTLY to the `file_structure` provided in the Plan.
   - Do not create `tests/` or `test_*.py`. (QA is external).
   - Use `os.path.join` for all file I/O to ensure cross-platform compatibility, though we assume Linux here.

3. **Code Quality**:
   - Type hints are mandatory.
   - No placeholders (`pass` or `TODO`). Implement the complete logic required for the step.
   - Self-Correction: Before finishing, verify you have created all required files for this step.

### WORKFLOW
1. **Analyze**: Read the requirements for the Current Step and the project `file_structure`.
2. **Explore**: Check `../repos` if you need implementation references.
3. **Implement**: Write the code using `write_file`.
4. **Report**: Return a detailed textual report summarizing your work.

### OUTPUT REQUIREMENTS
Provide a detailed textual report containing:
1. **Execution Summary**: What was implemented in this step.
2. **Files Created/Modified**: List of file paths.
3. **Key Components**: List of classes/functions implemented.
4. **Notes**: Implementation details.

You MUST use `write_file` to actually create the files. The report is just a summary of what you did.
"""

    agent = Agent(
        name="Initial Implementation Agent",
        instructions=instructions,
        # output_type=IntermediateImplementOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
initial_implement_agent = create_initial_implement_agent()
