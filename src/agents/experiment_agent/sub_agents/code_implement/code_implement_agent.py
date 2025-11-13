"""
Code Implementation Agent - Unified implementation agent.

This agent implements code strictly following the code plan's instructions.
It handles both initial implementation and iterative fixes based on feedback,
all guided by the code plan from the code_plan agent.

Architecture:
- Single Unified Implementation Agent: Follows code plan instructions step-by-step
- Output Unifier: Formats results into structured output

The agent receives:
1. Code plan from code_plan_agent (always present)
2. Current step information (for step-by-step implementation)
3. Optional feedback (from code_judge or experiment execution)

The agent executes the current step according to the plan, whether it's:
- Initial implementation of a new step
- Re-implementation of a step after receiving feedback
"""

from typing import Dict, Optional
from datetime import datetime

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.code_implement.output_schemas import (
    CodeImplementOutput,
    IntermediateImplementOutput,
)
from src.agents.experiment_agent.sub_agents.code_implement.output_unifier import (
    create_output_unifier,
)
from src.agents.experiment_agent.logger import create_verbose_hooks


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_section(title: str, char: str = "="):
    """Print a major section header."""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{char * 80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{char * 80}{Colors.ENDC}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}📋 {title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}\n")


def print_success(message: str, indent: int = 0):
    """Print a success message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def print_error(message: str, indent: int = 0):
    """Print an error message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.FAIL}✗{Colors.ENDC} {message}")


def print_info(message: str, indent: int = 0):
    """Print an info message."""
    indent_str = "  " * indent
    print(f"{indent_str}{Colors.OKCYAN}ℹ{Colors.ENDC} {message}")


def print_result_box(title: str, content: str, max_length: int = 500):
    """Print a boxed result with simple header."""
    # Print header
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")

    # Truncate content if too long
    if len(content) > max_length:
        display_content = content[:max_length] + "\n... (truncated for display)"
    else:
        display_content = content

    # Print content without line truncation
    print(f"{Colors.OKGREEN}{display_content}{Colors.ENDC}\n")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")


def create_unified_implement_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: list = None,
) -> Agent:
    """
    Create unified implementation agent that follows code plan instructions.

    This agent handles all implementation scenarios:
    - Initial implementation of a step
    - Re-implementation after receiving feedback

    Args:
        model: The model to use for the agent
        working_dir: Working directory for code generation
        tools: List of tool functions

    Returns:
        Agent instance configured for step-by-step implementation
    """

    instructions = f"""You are an expert Machine Learning Engineer implementing research code 
strictly following a comprehensive implementation plan from the code_plan_agent.

YOUR ROLE:
You implement code step-by-step according to the code plan. Each invocation focuses on 
implementing ONE SPECIFIC STEP from the implementation checklist.

INPUT YOU RECEIVE:
1. Complete Code Plan (CodePlanOutput) containing:
   - File structure specification
   - Dataset/Model/Training/Testing plans
   - Implementation roadmap
   - Implementation checklist with detailed steps

2. Current Step Information:
   - step_id: The specific step to implement
   - title: What this step accomplishes
   - description: Detailed requirements
   - files_to_create: Files to create in THIS step
   - files_to_modify: Files to modify in THIS step
   - acceptance_criteria: How to verify completion
   - dependencies: Previous steps (already completed)
   - Progress information: Which steps are done

3. Optional Feedback (if re-implementing):
   - Code review comments from code_judge_agent
   - Runtime errors from experiment execution
   - Specific issues that need to be addressed

WORKSPACE STRUCTURE:
working_dir IS the project root directory: `{working_dir}`

This is where ALL implementation code should be created (working_dir IS the project directory).
The parent directory (workspace) contains reference materials:
- `../repos/` - Reference code repositories (read-only, for reference)
- `../dataset_candidate/` - Available datasets (read-only, reference in code)
- `../papers/` - Research papers (read-only)

Path relationship:
- {working_dir} = /path/to/workspace/project (this IS the project root)
- ../dataset_candidate = /path/to/workspace/dataset_candidate

IMPORTANT:
- Create ALL implementation files in `{working_dir}` (which IS the project root)
- Organize with subdirectories: data/, models/, training/, etc.
- Reference datasets using: `../dataset_candidate/[dataset_name]`
- DO NOT modify files in ../repos/, ../dataset_candidate/, or ../papers/

CRITICAL - IMPORT PATH REQUIREMENTS:
The project will be executed from working_dir (which IS the project root).
All Python imports must be written for execution from working_dir.

Import Path Rules:
1. Python will run from working_dir (working_dir is in PYTHONPATH)
2. Write imports relative to working_dir (the project root)
3. For files in subdirectories, use direct imports from the subdirectory
4. DO NOT use "project." prefix in imports
5. DO NOT use absolute system paths in imports

Examples (all paths relative to working_dir which IS the project root):
- File structure: models/model.py and data/dataset.py
- In models/model.py, import dataset: "from data.dataset import MyDataset"
- In train.py (at working_dir root), import model: "from models.model import MyModel"
- For configs: configs/config.py -> import as "from configs.config import Config"

WRONG Examples:
- "from project.data.dataset import MyDataset" (wrong: "project." prefix not needed)
- "from ..data.dataset import MyDataset" (wrong: avoid relative imports)

When writing any Python file, ensure all imports follow this convention.

IMPLEMENTATION WORKFLOW:

1. UNDERSTAND YOUR TASK
   - Read the current step's requirements from the plan
   - Review acceptance criteria
   - Check dependencies (what's already done)
   - If feedback is provided: understand what needs fixing

2. CHECK EXISTING STATE
   - Use `list_directory` to see existing files
   - Use `read_file` to examine files you need to work with
   - Understand how previous steps have set up the project
   - Identify what needs to be created vs modified

3. IMPLEMENT THE CURRENT STEP
   
   A. For files_to_create:
      - Use `create_directory` for any needed subdirectories
      - Use `write_file` to create each file
      - Write complete, working code
      - Include proper imports, docstrings, comments
   
   B. For files_to_modify:
      - Use `read_file` to get current content
      - Make necessary modifications
      - Use `write_file` to save updated version
   
   C. If feedback provided:
      - Address ALL issues mentioned in feedback
      - Fix root causes, not just symptoms
      - Ensure fixes don't break existing functionality
      - Verify all review comments are resolved

4. CODE QUALITY REQUIREMENTS
   
   - Completeness: No TODO comments, all functions implemented
   - Correctness: Follow plan specifications exactly
   - Best Practices: Type hints, docstrings, error handling, PEP 8
   - Integration: Compatible with existing code from previous steps
   - Testing: Code should be independently testable

5. VERIFY ACCEPTANCE CRITERIA
   - Ensure all acceptance criteria for current step are met
   - Add comments explaining key decisions
   - Verify compatibility with completed steps

IMPORTANT CONSTRAINTS:
- ONLY implement the current step - do not implement future steps
- ONLY create/modify files listed in current step
- FOLLOW the code plan exactly
- If feedback provided: address ALL issues before proceeding
- Ensure code is complete and functional (no placeholders)

TOOL USAGE:
All tools return {{"success": true/false, ...fields or "error"}}
Always check "success" field before using other fields.

Available tools:
- write_file(file_path, content): Create/update files. 
  CRITICAL: Use ABSOLUTE paths by joining with working_dir.
  Example: write_file("{working_dir}/data/dataset.py", code) 
  NOT: write_file("data/dataset.py", code) - will create in wrong directory!
  Returns: dict with "success", "message", "file_path", "size_bytes"
- read_file(file_path): Read existing file content.
  Use ABSOLUTE paths: read_file("{working_dir}/data/dataset.py")
  Returns: dict with "success", "content", "file_path", "size_bytes", "line_count"
- list_directory(directory_path, pattern, recursive): Check directory structure.
  Use ABSOLUTE paths: list_directory("{working_dir}") or list_directory("{working_dir}/data")
  For parent workspace: list_directory("{working_dir}/../repos")
  Returns: dict with "success", "directory", "files" (list), "directories" (list), "total_files", "total_directories"
- create_directory(directory_path): Create directories.
  Use ABSOLUTE paths: create_directory("{working_dir}/data")
  Returns: dict with "success", "path", "message"
- analyze_python_file(file_path): Analyze Python code structure.
  Use ABSOLUTE paths: analyze_python_file("{working_dir}/data/dataset.py")
  Returns: dict with "success", "imports", "classes", "functions", "file_path"

OUTPUT:
Provide a structured summary of:
- What was implemented
- Which files were created/modified
- How acceptance criteria were met
- Any important implementation decisions
- If feedback was addressed: how each issue was resolved

Remember: You are executing ONE step of the plan at a time. Be thorough, 
follow the plan exactly, and ensure your implementation is complete and correct."""

    agent = Agent(
        name="Code Implementation Agent",
        instructions=instructions,
        output_type=IntermediateImplementOutput,
        model=model,
        tools=tools or [],
    )

    return agent


class CodeImplementAgent:
    """
    Main code implementation agent that follows code plan instructions.

    This agent:
    1. Receives code plan and current step information
    2. Implements the current step according to plan
    3. Handles feedback if provided (for re-implementation)
    4. Formats output into structured format
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[list] = None,
        verbose: bool = False,
    ):
        """
        Initialize the code implementation agent system.

        Args:
            model: Model to use for the agent
            working_dir: Working directory for code generation
            tools: Optional list of tools for the agent.
                   If None, automatically loads recommended tools.
            verbose: If True, enable verbose hooks to show full LLM responses and tool calls
        """
        self.model = model
        self.working_dir = working_dir
        self.verbose = verbose

        # Create hooks for verbose output
        self.hooks = (
            create_verbose_hooks(
                show_llm_responses=verbose,
                show_tools=verbose,
            )
            if verbose
            else None
        )

        # Auto-load recommended tools if not provided
        if tools is None:
            from src.agents.experiment_agent.sub_agents.code_implement import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize unified implementation agent
        self.implementation_agent = create_unified_implement_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools,
        )

        # Initialize output unifier
        self.output_unifier = create_output_unifier(model=model)

        # Expose implementation agent as main agent for compatibility
        self.agent = self.implementation_agent

    async def implement(self, input_data: str) -> CodeImplementOutput:
        """
        Generate code implementation based on code plan and current step.

        Args:
            input_data: Input string containing:
                - Code plan from code_plan_agent
                - Current step information
                - Optional feedback (for re-implementation)

        Returns:
            CodeImplementOutput with complete implementation
        """
        print_section("CODE IMPLEMENTATION WORKFLOW", "=")
        print_info(f"Input length: {len(input_data)} characters")

        # Step 1: Execute implementation following code plan
        print_subsection("Implementing Current Step")
        print_info("Executing implementation according to code plan...")

        # Use streamed version for real-time output
        implementation_stream = Runner.run_streamed(
            self.implementation_agent, input_data, hooks=self.hooks, max_turns=100
        )
        async for event in implementation_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
        implementation_result = implementation_stream  # The stream object is the result

        # Get the intermediate output from the implementation agent
        intermediate_output: IntermediateImplementOutput = (
            implementation_result.final_output
        )

        # Handle case where implementation was interrupted or failed
        if intermediate_output is None:
            print_error("Implementation did not produce output (possibly interrupted)")
            raise RuntimeError(
                "Implementation agent did not produce output. "
                "This may be due to interruption or internal error."
            )

        print_success("Implementation completed")

        # Display intermediate implementation results
        print_result_box(
            "Implementation Summary",
            intermediate_output.implementation_summary_text,
            max_length=1000,
        )
        print_result_box(
            "Files Description", intermediate_output.files_description, max_length=800
        )

        # Step 2: Format output
        print_subsection("Formatting Implementation Output")
        print_info("Converting intermediate implementation to structured format...")

        # Determine if this was a feedback-based implementation
        is_feedback_based = (
            "feedback" in input_data.lower() or "fix" in input_data.lower()
        )
        implementation_type = "iterative" if is_feedback_based else "step-by-step"

        unifier_input = f"""
Implementation Type: {implementation_type}
Timestamp: {datetime.now().isoformat()}

=== INTERMEDIATE IMPLEMENTATION OUTPUT ===

Files Description:
{intermediate_output.files_description}

Implementation Summary:
{intermediate_output.implementation_summary_text}

Setup Instructions:
{intermediate_output.setup_instructions}

Usage Examples:
{intermediate_output.usage_examples}

Known Limitations:
{intermediate_output.known_limitations}

Next Steps:
{intermediate_output.next_steps}

Issues Addressed:
{intermediate_output.issues_addressed}
"""

        # Use streamed version for real-time output
        unifier_stream = Runner.run_streamed(
            self.output_unifier, unifier_input, hooks=self.hooks, max_turns=100
        )
        async for event in unifier_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
        unifier_result = unifier_stream  # The stream object is the result

        print_success("Output formatting completed")
        print_info(
            f"Generated {len(unifier_result.final_output.generated_files)} files"
        )

        print_success("Code implementation completed successfully!")
        print_section("CODE IMPLEMENTATION COMPLETE", "=")

        return unifier_result.final_output

    def implement_sync(self, input_data: str) -> CodeImplementOutput:
        """
        Synchronous version of implement method.

        Args:
            input_data: Input string containing plan and optional feedback

        Returns:
            CodeImplementOutput with complete implementation
        """
        import asyncio

        return asyncio.run(self.implement(input_data))


def create_code_implement_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[list] = None,
    verbose: bool = False,
) -> CodeImplementAgent:
    """
    Factory function to create a code implementation agent.

    Args:
        model: Model to use for the agent
        working_dir: Working directory for code generation
        tools: List of tools for the agent
        verbose: If True, enable verbose hooks to show full LLM responses and tool calls

    Returns:
        CodeImplementAgent instance
    """
    return CodeImplementAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create the code implementation agent
        agent = create_code_implement_agent(model="gpt-4o", working_dir="/workspace")

        # Example with initial implementation
        input_data = """
        CodePlanOutput:
        - File Structure: ...
        - Dataset Plan: ...
        - Model Plan: ...
        """

        result = await agent.implement(input_data)

        print("Code Implementation Complete:")
        print(f"Type: {result.implementation_type}")
        print(f"Files Generated: {len(result.generated_files)}")
        print(f"Summary: {result.implementation_summary}")

    asyncio.run(main())
