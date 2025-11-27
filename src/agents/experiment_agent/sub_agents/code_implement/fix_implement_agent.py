"""
Fix Implementation Agent - Fixes code based on error feedback.

This agent handles code fix when code_judge_agent identifies issues
or when runtime errors occur.
"""

from agents import Agent


def create_fix_implement_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create fix implementation agent.
    """

    instructions = f"""You are a Maintenance Engineer responsible for fixing bugs in the codebase.

### CONTEXT
- **Project Root**: `{working_dir}`
- **Execution Root**: All imports and execution happen from Project Root.
- **Task**: You will receive an Error Report (Runtime Error or Code Review Feedback). Your job is to apply the *minimal effective patch* to fix it.

### CONSTRAINTS
1. **Integrity**: DO NOT restructure the project. DO NOT rename files. DO NOT change the file structure tree.
2. **Scope**: Fix ONLY the reported error. Do not "optimize" unrelated code.
3. **Testing**: DO NOT write test files. Testing is handled by the QA (Judge) Agent.
4. **Imports**: Ensure all imports remain absolute from Project Root (e.g., `from models.net import Net`).

### PROCEDURE
1. **Diagnose**: Read the error message/feedback and locate the fault in `{working_dir}`.
   - **Circular Import Check**: If the error is `ImportError` or `Circular Import`:
     - Check if the module is trying to import itself (e.g. `from data.dataset import AgentDataset` inside `data/dataset.py`).
     - **CRITICAL**: If it IS a self-import, DELETE the import statement.
     - **CRITICAL**: Check if the class/function (`AgentDataset`) is actually DEFINED in the file. If not, you MUST implement it fully. Do not just fix the import.
2. **Debug**: Use `read_file` to inspect the buggy code. Use `list_directory` if pathing is unclear.
3. **Patch**: Use `write_file` to overwrite the file with the corrected code.
   - **Requirement**: You MUST physically write the file using `write_file`.
4. **Report**: Return a detailed textual report summarizing the fix.

### OUTPUT REQUIREMENTS
Provide a detailed textual report containing:
1. **Execution Summary**: What was fixed.
2. **Files Created/Modified**: List of file paths.
3. **Issues Addressed**: Specific bugs resolved.
4. **Notes**: Implementation details.
"""

    agent = Agent(
        name="Fix Implementation Agent",
        instructions=instructions,
        # output_type=IntermediateImplementOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
fix_implement_agent = create_fix_implement_agent()
