"""
Code Judge Agent - Reviews code implementation for consistency with plan and analysis.

This agent evaluates whether the implemented code matches the specifications
from the code plan and pre-analysis, identifying issues and providing feedback.

Architecture:
- Judge Agent: Reviews code and provides structured feedback
- Uses tools to read and analyze codebase
- Outputs structured evaluation with consistency scores and issues
"""

from typing import Optional

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.code_judge.output_schemas import (
    CodeJudgeOutput,
)
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
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


def print_warning(message: str, indent: int = 0):
    """Print a warning message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.WARNING}⚠{Colors.ENDC} {message}")


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


def create_judge_agent(model: str = "gpt-4o", tools: Optional[list] = None) -> Agent:
    """
    Create the code judge agent for reviewing implementation.

    Args:
        model: Model to use for the agent
        tools: List of tools for reading and analyzing code (to be implemented)

    Returns:
        Agent configured for code review
    """

    instructions = """You are an expert code reviewer responsible for evaluating whether 
the CURRENT IMPLEMENTATION STEP is consistent with the plan and correctly implemented.

EVALUATION MODE: SINGLE STEP REVIEW
You are evaluating ONE step from the implementation checklist, not the entire project.
Focus ONLY on the files created/modified in the current step.

CRITICAL - IMPORT PATH VERIFICATION:
When reviewing code, verify that imports follow the correct convention:
- working_dir IS the project root directory (not a parent of it)
- The project will be executed from working_dir (which is the project root)
- Imports must assume working_dir is in PYTHONPATH
- Check that imports DO NOT use "project." prefix

Examples of CORRECT imports (paths relative to working_dir which IS the project root):
- "from data.dataset import MyDataset" (for data/dataset.py)
- "from models.model import MyModel" (for models/model.py)
- "from configs.config import Config" (for configs/config.py)

Examples of INCORRECT imports to flag:
- "from project.data.dataset import MyDataset" (wrong: includes "project.")
- "from ..data.dataset import MyDataset" (verify if correct for execution context)

WORKSPACE STRUCTURE:
working_dir IS the project root directory where all implementation code resides.
The parent directory (workspace) contains additional resources:
- `../dataset_candidate/` - Available datasets for training/testing (read-only)
- `../repos/` - Reference code repositories (read-only)
- `../papers/` - Research papers (read-only)

Path relationship:
- working_dir = /path/to/workspace/project  (project root, passed as parameter)
- ../dataset_candidate = /path/to/workspace/dataset_candidate

When reviewing data loading code, verify that paths to datasets are constructed dynamically:
```python
# Correct example for accessing datasets
import os
dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset_candidate")
data_path = os.path.join(dataset_dir, "specific_dataset/subset")
```

YOUR RESPONSIBILITIES:

1. UNDERSTAND CURRENT STEP CONTEXT
   - Review the current step requirements from the checklist
   - Note which files should be created/modified in this step
   - Check acceptance criteria for this specific step
   - Understand dependencies (what previous steps provided)

2. CODE INSPECTION FOR CURRENT STEP
   - Use provided tools to read files created/modified in THIS step
   - Examine ONLY the code relevant to the current step
   - Check if all files mentioned in the step are present
   - Review implementation details for step-specific requirements

3. STEP-SPECIFIC CONSISTENCY EVALUATION
   
   A. Plan Consistency (for current step):
      - Files created/modified match the step's specification
      - Implementation follows step description
      - Code quality meets requirements
      - Interfaces with previous steps are correct
      - Dependencies are properly handled
   
   B. Acceptance Criteria Verification:
      - Check each acceptance criterion for the step
      - Verify completeness for THIS step only
      - Ensure no placeholders or TODOs in step files
      - Confirm code can be independently tested

4. STEP-SPECIFIC ISSUE IDENTIFICATION
   
   Focus ONLY on issues in the current step:
   - logic_error: Incorrect implementation in step files
   - missing_implementation: Step requirements not met
   - inconsistency: Implementation differs from step description
   - quality: Code quality issues in step files
   - integration_error: Incompatibility with previous steps
   
   Classify severity:
   - critical: Step cannot function, acceptance criteria not met
   - major: Important step requirements missing
   - minor: Code quality, optimization opportunities

5. STEP ACCEPTANCE DECISION
   
   Make a clear decision: is_consistent (True/False)
   - True: Step is correctly implemented and meets all acceptance criteria
   - False: Step has issues that must be fixed before proceeding
   
   Consider:
   - Are ALL acceptance criteria met?
   - Are ALL required files created/modified correctly?
   - Does the code integrate properly with previous steps?
   - Is the implementation complete (no TODOs/placeholders)?
   - Does the code follow quality standards?

6. UNIT TEST GENERATION (REQUIRED)
   
   For EVERY step (whether accepted or rejected), you MUST generate unit tests:
   
   A. Test Purpose:
      - Verify the current step implementation works correctly
      - Provide quick validation (not exhaustive testing)
      - Help catch obvious bugs and integration issues
   
   B. Test Requirements:
      - Create lightweight, fast unit tests
      - Each test MUST complete within 30 seconds (default time_limit_seconds)
      - For data processing: Use SMALL subsets (e.g., data_subset_size=10 or 100)
      - For model training: Use toy models or very few iterations
      - For data loading: Test with minimal samples
      - For full pipelines: Mock heavy components or use tiny configs
   
   C. Test Coverage (for current step):
      - Basic functionality: Can modules be imported? Do functions exist?
      - Core logic: Do key functions return expected types/shapes?
      - Integration: Does current step work with previous steps' outputs?
      - Edge cases: Handle None, empty inputs, basic error cases
   
   D. Test Writing Guidelines:
      - Use pytest framework (import pytest)
      - Write clear test names: test_<what>_<scenario>
      - Include docstrings explaining what is tested
      - Use assertions to validate behavior
      - Add timeout decorators if needed: @pytest.mark.timeout(30)
      - For dataset tests: Load ONLY a small subset
        Example: dataset = load_data(max_samples=10)
      - For training tests: Use minimal configs
        Example: train(epochs=1, batch_size=2)
   
   E. Example Unit Test Structures:
   
      For module imports and basic functionality:
      ```python
      import pytest
      from project.module import MyClass
      
      def test_module_import():
          '''Test that module can be imported successfully.'''
          assert MyClass is not None
      
      def test_basic_instantiation():
          '''Test that class can be instantiated.'''
          obj = MyClass()
          assert obj is not None
      ```
      
      For data processing with subset:
      ```python
      import pytest
      from project.data.dataset import load_dataset
      
      @pytest.mark.timeout(30)
      def test_dataset_loading_small_subset():
          '''Test dataset loading with small subset (10 samples).'''
          dataset = load_dataset(max_samples=10)
          assert len(dataset) == 10
          assert dataset[0] is not None
      ```
      
      For model forward pass:
      ```python
      import pytest
      import torch
      from project.models.model import Model
      
      @pytest.mark.timeout(30)
      def test_model_forward_pass():
          '''Test model forward pass with dummy input.'''
          model = Model(hidden_dim=8)  # Small model
          x = torch.randn(2, 10)  # Small batch
          output = model(x)
          assert output.shape[0] == 2
      ```
   
   F. Unit Test Generation Rules:
      - Generate 2-5 unit tests per step (focus on critical functionality)
      - Each test should be independent (no dependencies between tests)
      - IMPORTANT: Test files MUST be placed in "tests/" directory (relative to working_dir)
      - Specify test_file_path (e.g., "tests/test_step2_data.py" NOT "project/tests/...")
      - Provide complete, runnable test code (including all imports)
      - Set appropriate time_limit_seconds (default 30, max 60)
      - Specify data_subset_size if testing with datasets
      - List target_files that the test validates
   
   G. CRITICAL CONSTRAINTS:
      - Tests MUST be FAST (complete in seconds, not minutes)
      - Use MINIMAL data (10-100 samples, NOT full datasets)
      - Use TINY models (small hidden dims, few layers)
      - Use FEW iterations (1-5 epochs/steps, NOT full training)
      - MOCK expensive operations (file I/O, network calls)
      - SKIP integration with untested future components

7. FEEDBACK FOR STEP IMPROVEMENT
   
   If step is rejected (is_consistent=False), provide:
   - Clear description of what's wrong
   - Which acceptance criteria are not met
   - What was expected vs what was implemented
   - Specific suggestions for fixing the step
   - Focus on THIS step only, not future work

STEP EVALUATION PROCESS:

Step 1: Read the current step requirements and acceptance criteria
Step 2: Use tools to read files created/modified in THIS step
Step 3: Verify each acceptance criterion is met
Step 4: Check integration with previous steps
Step 5: Identify step-specific issues
Step 6: Generate unit tests for current step validation (2-5 tests)
Step 7: WRITE test files using write_file tool (REQUIRED)
Step 8: RUN tests using run_pytest_local tool (REQUIRED)
Step 9: Make acceptance decision (is_consistent=True/False)
Step 10: Provide step-focused feedback if rejected

CRITICAL: After generating unit_tests in your output, you MUST:
1. Use write_file tool to write EACH test file to the file system
2. Use run_pytest_local tool to execute the tests
3. Include test execution results in your evaluation

TOOL USAGE GUIDELINES:

All tools return a dictionary with the following structure:
- success (bool): Indicates if the operation succeeded
- If successful: Contains relevant data fields (content, imports, classes, functions, results, etc.)
- If failed: Contains an "error" field with error message

Example successful response:
{{
  "success": true,
  "content": "file content here",
  "file_path": "/path/to/file",
  "line_count": 150
}}

Example failed response:
{{
  "success": false,
  "error": "File not found: /path/to/file"
}}

Always check the "success" field before using other fields from tool results.
If a tool fails, report the error and try alternative approaches.

AVAILABLE TOOLS - CRITICAL PATH REQUIREMENT:
ALL tools require ABSOLUTE paths! Relative paths will be resolved from script directory, NOT working_dir.

- read_file(file_path): Read file content.
  MUST use: read_file("<working_dir>/data/dataset.py")
  NEVER use: read_file("data/dataset.py") - reads from wrong location!
  Returns: dict with "success", "content", "file_path", "size_bytes", "line_count"

- write_file(file_path, content): Write content to file (e.g., test files).
  MUST use: write_file("<working_dir>/tests/test_file.py", test_code)
  NEVER use: write_file("tests/test_file.py", test_code) - creates in wrong location!
  Returns: dict with "success", "message", "file_path", "size_bytes"

- list_directory(directory_path, pattern, recursive): List files in directory.
  Use: list_directory("<working_dir>") or list_directory("<working_dir>/data")
  Returns: dict with "success", "directory", "files" (list), "directories" (list), "total_files", "total_directories"

- analyze_python_file(file_path): Analyze Python code structure.
  Use: analyze_python_file("<working_dir>/data/dataset.py")
  Returns: dict with "success", "imports", "classes", "functions", "file_path"
- run_pytest_local(test_path, working_dir, timeout): Run pytest tests.
  CRITICAL: test_path relative to working_dir (e.g., "tests/"), working_dir should be project root absolute path.
  Returns: dict with "success", "exit_code", "stdout", "stderr", "execution_time"
- run_python_script_local(script_path, working_dir): Run Python script.
  Returns: dict with "success", "exit_code", "stdout", "stderr", "execution_time"

OUTPUT FORMAT:

You must output a structured CodeJudgeOutput with:
- is_consistent: boolean decision
- overall_assessment: summary paragraph
- plan_consistency_score: 0-1 score
- analysis_consistency_score: 0-1 score
- issues: list of CodeIssue objects
- strengths: list of positive aspects
- missing_components: list of missing features
- extra_components: list of unplanned additions
- unit_tests: list of UnitTestSpec objects (REQUIRED - generate 2-5 tests)
  Each UnitTestSpec must have:
  * test_file_path: where to save the test file
  * test_code: complete Python test code
  * test_description: what the test validates
  * target_files: which implementation files are tested
  * time_limit_seconds: max execution time (default 30)
  * data_subset_size: if using datasets, max samples to load
- priority_fixes: ordered list of high-priority actions
- implementation_suggestions: list of improvement suggestions
- next_steps: recommended actions

STEP EVALUATION GUIDELINES:

1. SCOPE: CURRENT STEP ONLY
   - ONLY evaluate files in current step's scope
   - DO NOT evaluate files from previous steps (assume they are correct)
   - DO NOT expect features from future steps
   - Focus on step-specific requirements

2. BE THOROUGH FOR THE STEP
   - Check all files mentioned in the step
   - Verify all acceptance criteria
   - Test integration with previous steps
   - Don't miss step-specific requirements

3. BE SPECIFIC IN FEEDBACK
   - Point to exact files and functions in the step
   - Reference specific acceptance criteria
   - Give actionable suggestions for THIS step
   - Explain how to meet unmet criteria

4. MAKE CLEAR DECISION
   - is_consistent=True: Step is complete and correct, ready for next step
   - is_consistent=False: Step needs revision, provide clear feedback
   - Don't approve incomplete or broken step implementation
   - Don't reject for issues outside step scope

5. BE CONSTRUCTIVE
   - Help the implementer fix THIS step
   - Don't ask for future features
   - Focus on what's needed NOW
   - Prioritize critical issues for the step

Remember: You are evaluating ONE step in an iterative process. The implementation 
will proceed to the next step ONLY if you approve this one. Be thorough but focused 
on the current step's requirements.

MANDATORY TEST EXECUTION WORKFLOW:
1. Generate unit tests (in unit_tests field of output)
2. For EACH test in unit_tests:
   a. Call write_file(test.test_file_path, test.test_code)
   b. Verify file was written successfully
3. After writing ALL test files:
   a. Call run_pytest_local("tests/", working_dir=<project_root>, timeout=300)
   b. Check if tests pass or fail
   c. Include test results in your overall_assessment
4. If tests fail:
   - Document failures in issues list
   - Set is_consistent=False if failures are critical
   - Provide specific suggestions for fixing

YOU MUST COMPLETE ALL THESE STEPS BEFORE RETURNING YOUR FINAL OUTPUT."""

    agent = Agent(
        name="Code Judge Agent",
        instructions=instructions,
        tools=tools or [],
        output_type=CodeJudgeOutput,
        model=model,
    )

    return agent


class CodeJudgeAgent:
    """
    Main code judge agent that reviews implementation consistency.

    This agent:
    1. Receives code plan and pre-analysis
    2. Reads and analyzes the implemented codebase
    3. Evaluates consistency and identifies issues
    4. Provides structured feedback for improvements
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[list] = None,
        verbose: bool = False,
    ):
        """
        Initialize the code judge agent.

        Args:
            model: Model to use for evaluation
            tools: Optional list of tools for code reading and analysis.
                   If None, automatically loads recommended tools.
            verbose: If True, enable verbose hooks to show full LLM responses and tool calls
        """
        self.model = model
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
            from src.agents.experiment_agent.sub_agents.code_judge import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize judge agent
        self.judge_agent = create_judge_agent(model=model, tools=self.tools)

        # Expose judge agent as main agent for handoff compatibility
        self.agent = self.judge_agent

    async def judge(
        self,
        input_data: str,
    ) -> CodeJudgeOutput:
        """
        Evaluate code implementation for consistency with current step.

        Args:
            input_data: String containing plan, analysis, current step info, and codebase path

        Returns:
            CodeJudgeOutput with evaluation results and feedback
        """
        print_section("CODE REVIEW WORKFLOW", "=")
        print_info(f"Input length: {len(input_data)} characters")

        # The input_data already contains all necessary information formatted
        # by the experiment master agent
        judge_input = f"""
{input_data}

=== EVALUATION INSTRUCTIONS ===

You are evaluating a SINGLE STEP in an iterative implementation process.

CRITICAL: The input above contains a "CODEBASE PATH" section that shows the project directory.
You MUST use the tools (list_directory, read_file, etc.) to check the files in that directory.
All file paths should be relative to that project directory or absolute paths starting from it.

The above input contains:
- The complete code plan for context
- The current step you need to evaluate (with step_id, files, acceptance criteria)
- The implementation output summary
- The path to the codebase (in "CODEBASE PATH" section)

YOUR TASK:
1. Extract the project directory path from the "CODEBASE PATH" section above
   Example: If it shows "/hpc_stor03/.../workspace/project", that's your working_dir
2. Use list_directory tool to check what files exist in the project directory
3. Use read_file tool to examine files that were created/modified in THIS step
4. Verify each acceptance criterion for the step is met
5. Generate 2-5 unit tests to validate the current step (define in unit_tests output field)
   IMPORTANT: Test paths should be relative (e.g., "tests/test_file.py", NOT "project/tests/test_file.py")
6. WRITE each test file using write_file tool
   Example: write_file("<working_dir>/tests/test_file.py", <test_code>)
   Or use absolute path: write_file("/full/path/to/project/tests/test_file.py", <test_code>)
7. RUN all tests using run_pytest_local tool WITH working_dir parameter
   CRITICAL: run_pytest_local("tests/", working_dir="<extracted_project_dir>", timeout=300)
   NOT: run_pytest_local("project/tests/", ...)
8. Analyze test results and incorporate into your evaluation
9. Make a clear decision: is_consistent (True/False)
10. If False, provide specific feedback with exact file paths

IMPORTANT FILE PATH HANDLING:
- The project directory shown above IS the working directory
- All file paths should be relative to that directory (NOT prefixed with "project/")
- Example: To check "data/file.py", NOT "project/data/file.py"
- When calling run_pytest_local, ALWAYS pass working_dir parameter
- Use list_directory to verify directory structure
- Use read_file to verify file contents
  
PATH EXAMPLES:
- Correct: read_file("data/dataset.py") - relative to project root
- Correct: run_pytest_local("tests/", working_dir="/abs/path/to/project")
- Wrong: read_file("project/data/dataset.py") - "project/" prefix not needed

TEST EXECUTION IS MANDATORY:
- You MUST write test files to the file system (use write_file)
- You MUST run the tests (use run_pytest_local)
- You MUST include test results in your overall_assessment
- Test failures should influence your is_consistent decision

DO NOT evaluate files from previous steps.
DO NOT expect features from future steps.
ONLY evaluate what should be done in the CURRENT step.
"""

        # Run judge agent
        print_subsection("Evaluating Current Step Implementation")
        print_info("Reviewing code against acceptance criteria...")

        # Use streamed version for real-time output
        result_stream = Runner.run_streamed(
            self.judge_agent, judge_input, hooks=self.hooks, max_turns=100
        )
        async for event in result_stream.stream_events():
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
        result = result_stream  # The stream object is the result

        evaluation: CodeJudgeOutput = result.final_output

        # Display evaluation results
        print_subsection("Evaluation Results")

        if evaluation.is_consistent:
            print_success(
                f"Code review passed! (Consistency: {evaluation.is_consistent})"
            )
            print_info(
                f"Plan consistency score: {evaluation.plan_consistency_score:.2f}"
            )
            print_info(
                f"Analysis consistency score: {evaluation.analysis_consistency_score:.2f}"
            )
        else:
            print_error(
                f"Code review failed! (Consistency: {evaluation.is_consistent})"
            )
            print_warning(
                f"Plan consistency score: {evaluation.plan_consistency_score:.2f}"
            )
            print_warning(
                f"Analysis consistency score: {evaluation.analysis_consistency_score:.2f}"
            )
            print_warning(f"Issues found: {len(evaluation.issues)}")

        # Display overall assessment
        print_result_box(
            "Overall Assessment",
            evaluation.overall_assessment,
            max_length=1000,
        )

        # Display issues if any
        if evaluation.issues:
            print_subsection("Issues Identified")
            for i, issue in enumerate(evaluation.issues, 1):
                severity_color = (
                    Colors.FAIL
                    if issue.severity == "critical"
                    else Colors.WARNING if issue.severity == "major" else Colors.OKBLUE
                )
                print(
                    f"{severity_color}Issue {i}: [{issue.severity.upper()}] {issue.issue_type}{Colors.ENDC}"
                )
                print(f"  File: {issue.file_path}")
                if hasattr(issue, "line_numbers") and issue.line_numbers:
                    print(f"  Lines: {issue.line_numbers}")
                print(f"  Description: {issue.description}")
                if hasattr(issue, "suggestion") and issue.suggestion:
                    print(f"  Suggestion: {issue.suggestion}")
                print()

        # Display strengths
        if evaluation.strengths:
            print_subsection("Strengths")
            for strength in evaluation.strengths:
                print_success(strength, indent=1)

        # Display unit tests
        if evaluation.unit_tests:
            print_subsection("Unit Tests Generated")
            print_info(
                f"Generated {len(evaluation.unit_tests)} unit test(s) for validation"
            )
            for i, test in enumerate(evaluation.unit_tests, 1):
                print(f"\n{Colors.OKCYAN}Test {i}: {test.test_file_path}{Colors.ENDC}")
                print(f"  Description: {test.test_description}")
                print(f"  Target files: {', '.join(test.target_files)}")
                print(f"  Time limit: {test.time_limit_seconds}s")
                if test.data_subset_size:
                    print(f"  Data subset size: {test.data_subset_size}")
                # Show first few lines of test code
                code_lines = test.test_code.split("\n")
                preview_lines = min(10, len(code_lines))
                print(f"  Code preview (first {preview_lines} lines):")
                for line in code_lines[:preview_lines]:
                    print(f"    {Colors.OKBLUE}{line}{Colors.ENDC}")
                if len(code_lines) > preview_lines:
                    print(
                        f"    {Colors.OKBLUE}... ({len(code_lines) - preview_lines} more lines){Colors.ENDC}"
                    )

        # Display priority fixes if any
        if evaluation.priority_fixes:
            print_subsection("Priority Fixes Required")
            for i, fix in enumerate(evaluation.priority_fixes, 1):
                print_warning(f"{i}. {fix}", indent=1)

        print_success("Code review completed!")
        print_section("CODE REVIEW COMPLETE", "=")

        return evaluation

    def judge_sync(
        self,
        input_data: str,
    ) -> CodeJudgeOutput:
        """
        Synchronous version of judge method.

        Args:
            input_data: String containing plan, analysis, current step info, and codebase path

        Returns:
            CodeJudgeOutput with evaluation results and feedback
        """
        import asyncio

        return asyncio.run(self.judge(input_data))

    def _format_file_structure(self, file_structure: dict) -> str:
        """Format file structure dict to readable string."""
        lines = []
        for path, description in file_structure.items():
            lines.append(f"  {path}:")
            lines.append(f"    {description}")
        return "\n".join(lines)

    def _format_roadmap(self, roadmap: dict) -> str:
        """Format implementation roadmap to readable string."""
        lines = []
        for phase, details in roadmap.items():
            lines.append(f"\n{phase}:")
            lines.append(f"  {details}")
        return "\n".join(lines)


def create_code_judge_agent(
    model: str = "gpt-4o",
    tools: Optional[list] = None,
    verbose: bool = False,
) -> CodeJudgeAgent:
    """
    Factory function to create a code judge agent.

    Args:
        model: Model to use for evaluation
        tools: List of tools for code reading and analysis
        verbose: If True, enable verbose hooks to show full LLM responses and tool calls

    Returns:
        CodeJudgeAgent instance
    """
    return CodeJudgeAgent(model=model, tools=tools, verbose=verbose)


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Example 1: Simple creation (tools auto-load)
        print("Example 1: Simple creation")
        print("=" * 60)

        # Create agent - tools automatically loaded!
        agent = create_code_judge_agent(model="gpt-4o")
        print("✓ Code judge agent created with all tools automatically loaded\n")

        # Example 2: Custom tools (if needed)
        print("\nExample 2: Custom tool selection")
        print("=" * 60)

        from src.agents.experiment_agent.tools import (
            read_file,
            analyze_python_file,
            search_in_codebase,
        )

        # Create with custom tools
        custom_agent = create_code_judge_agent(
            model="gpt-4o",
            tools=[read_file, analyze_python_file, search_in_codebase],
        )
        print("✓ Code judge agent created with custom tools\n")

        # Example code plan and pre-analysis would be loaded here
        # code_plan = CodePlanOutput(...)
        # pre_analysis = PreAnalysisOutput(...)

        # Evaluate implementation
        # result = await agent.judge(
        #     code_plan=code_plan,
        #     pre_analysis=pre_analysis,
        #     codebase_path="/path/to/implemented/code"
        # )

        # print("Evaluation Result:")
        # print(f"Consistent: {result.is_consistent}")
        # print(f"Plan Consistency Score: {result.plan_consistency_score}")
        # print(f"Analysis Consistency Score: {result.analysis_consistency_score}")
        # print(f"Issues Found: {len(result.issues)}")
        # print(f"Priority Fixes: {result.priority_fixes}")

    asyncio.run(main())
