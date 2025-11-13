"""
Code Plan Agent - Main orchestrator for code planning using handoff mechanism.

This agent uses OpenAI agents library's handoff mechanism to route inputs
to appropriate scenario-specific planners and produces unified YAML-format code plans.

Architecture:
- Triage Agent: Determines scenario and hands off to appropriate planner
- Scenario-specific Planners: Generate plans based on input type
- Output Unifier: Formats plans into YAML-compatible structure
"""

from typing import Dict, Optional
from datetime import datetime

from agents import Agent, Runner, handoff

# Import hooks
from src.agents.experiment_agent.logger import create_verbose_hooks


# =============================================================================
# Pretty Print Utilities
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

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
    """Print a section header."""
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
    prefix = "  " * indent
    print(f"{prefix}{Colors.OKBLUE}ℹ{Colors.ENDC} {message}")


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


from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
    IntermediatePlanOutput,
)
from src.agents.experiment_agent.sub_agents.code_plan.initial_plan_agent import (
    create_initial_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.judge_feedback_plan_agent import (
    create_judge_feedback_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.error_feedback_plan_agent import (
    create_error_feedback_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.analysis_feedback_plan_agent import (
    create_analysis_feedback_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.output_unifier import (
    create_output_unifier,
)


def create_triage_agent(
    initial_planner: Agent,
    judge_feedback_planner: Agent,
    error_feedback_planner: Agent,
    analysis_feedback_planner: Agent,
) -> Agent:
    """
    Create a triage agent that uses handoffs to route to appropriate planners.

    Args:
        initial_planner: Agent for initial planning
        judge_feedback_planner: Agent for judge feedback planning
        error_feedback_planner: Agent for error feedback planning
        analysis_feedback_planner: Agent for analysis feedback planning

    Returns:
        Triage agent with handoffs configured
    """

    instructions = """You are a planning triage agent responsible for determining which planning 
scenario applies and handing off to the appropriate specialized planner.

CRITICAL - IMPORT PATH REQUIREMENTS FOR CODE PLANNING:
When planning code structure and imports, remember:
- The final project will be executed from working_dir/project directory
- All Python imports must assume project/ is the execution root
- Imports should NOT include "project." prefix
- Plan file structure with this execution context in mind

Example: For project/models/model.py importing from project/data/dataset.py
Correct import: "from data.dataset import MyDataset"
Wrong import: "from project.data.dataset import MyDataset"

ANALYZE the input to determine which scenario it represents:

1. INITIAL PLANNING
   - Input: Only PreAnalysisOutput (research analysis)
   - No feedback or error information present
   - First-time code planning
   → Handoff to Initial Plan Agent

2. JUDGE FEEDBACK PLANNING
   - Input: PreAnalysisOutput + Code review feedback
   - Contains code quality issues, logic errors
   - References code_judge_agent output
   → Handoff to Judge Feedback Plan Agent

3. ERROR FEEDBACK PLANNING
   - Input: PreAnalysisOutput + Runtime error information
   - Contains error messages, stack traces
   - References experiment_execute_agent
   → Handoff to Error Feedback Plan Agent

4. ANALYSIS FEEDBACK PLANNING
   - Input: PreAnalysisOutput + Experiment analysis
   - Contains performance metrics, experimental results
   - References experiment_analysis_agent
   → Handoff to Analysis Feedback Plan Agent

IMPORTANT: After analyzing the input, you MUST handoff to the appropriate specialized 
planning agent. Do not generate plans yourself."""

    triage = Agent(
        name="Planning Triage Agent",
        instructions=instructions,
        handoffs=[
            handoff(
                initial_planner,
                tool_description_override="Handoff to Initial Plan Agent for first-time code planning based on research analysis.",
            ),
            handoff(
                judge_feedback_planner,
                tool_description_override="Handoff to Judge Feedback Plan Agent for re-planning after code review feedback.",
            ),
            handoff(
                error_feedback_planner,
                tool_description_override="Handoff to Error Feedback Plan Agent for re-planning after runtime errors.",
            ),
            handoff(
                analysis_feedback_planner,
                tool_description_override="Handoff to Analysis Feedback Plan Agent for re-planning after experimental analysis.",
            ),
        ],
    )

    return triage


class CodePlanAgent:
    """
    Main code planning agent that orchestrates the entire planning workflow using handoffs.

    This agent:
    1. Uses triage agent to determine scenario
    2. Hands off to appropriate scenario-specific planner via handoff mechanism
    3. Formats output into YAML-compatible structure
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[Dict[str, list]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the code planning agent system.

        Args:
            model: Model to use for all agents
            working_dir: Working directory with reference codebases
            tools: Optional dictionary mapping scenario types to their tools.
                   If None, automatically loads recommended tools.
                   e.g., {"initial": [...], "judge_feedback": [...], ...}
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
            from src.agents.experiment_agent.sub_agents.code_plan import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize scenario-specific planners
        self.initial_planner = create_initial_plan_agent(
            model=model, working_dir=working_dir, tools=self.tools.get("initial", [])
        )

        self.judge_feedback_planner = create_judge_feedback_plan_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools.get("judge_feedback", []),
        )

        self.error_feedback_planner = create_error_feedback_plan_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools.get("error_feedback", []),
        )

        self.analysis_feedback_planner = create_analysis_feedback_plan_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools.get("analysis_feedback", []),
        )

        # Initialize triage agent with handoffs
        self.triage_agent = create_triage_agent(
            self.initial_planner,
            self.judge_feedback_planner,
            self.error_feedback_planner,
            self.analysis_feedback_planner,
        )

        # Initialize output unifier
        self.output_unifier = create_output_unifier(model=model)

        # Expose triage agent as main agent for handoff compatibility
        self.agent = self.triage_agent

    async def plan(self, input_data: str) -> CodePlanOutput:
        """
        Generate code implementation plan based on input.

        Args:
            input_data: Input string containing research analysis and optional feedback

        Returns:
            CodePlanOutput with complete implementation plan
        """
        print_section("CODE PLANNING WORKFLOW", "=")
        print_info(f"Input length: {len(input_data)} characters")

        # Step 1: Run triage agent (will automatically handoff to appropriate planner)
        print_subsection("Planning Triage & Execution")
        print_info(
            "Determining planning scenario and routing to appropriate planner..."
        )

        # Use streamed version for real-time output
        planning_stream = Runner.run_streamed(
            self.triage_agent, input_data, hooks=self.hooks, max_turns=100
        )
        async for event in planning_stream.stream_events():
            # Only print text content, not tool call arguments
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                # Skip function call argument deltas (too noisy)
                if "FunctionCallArguments" not in event_type:
                    # Look for text/message content
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        # Check if delta has text content
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)
        planning_result = planning_stream  # The stream object is the result

        # The final_output will be from the planner that was handed off to
        intermediate_plan: IntermediatePlanOutput = planning_result.final_output

        # Get the agent that actually did the planning from RunResult
        active_agent_name = (
            planning_result.last_agent.name
            if planning_result.last_agent
            else "Unknown Agent"
        )

        # Determine scenario based on agent name
        if "Initial" in active_agent_name:
            scenario = "initial"
        elif "Judge" in active_agent_name:
            scenario = "judge_feedback"
        elif "Error" in active_agent_name:
            scenario = "error_feedback"
        elif "Analysis" in active_agent_name:
            scenario = "analysis_feedback"
        else:
            scenario = "initial"  # default

        print_success(f"Planning completed using: {active_agent_name}")
        print_info(f"Scenario: {scenario}")

        # Display intermediate plan results
        print_result_box(
            "Research Summary", intermediate_plan.research_summary, max_length=1000
        )
        print_result_box(
            "Implementation Steps",
            intermediate_plan.implementation_steps,
            max_length=2000,
        )

        # Step 2: Format output
        unifier_input = f"""
Plan Type: {scenario}
Timestamp: {datetime.now().isoformat()}

=== INTERMEDIATE PLAN ===

Research Summary:
{intermediate_plan.research_summary}

Key Innovations:
{intermediate_plan.key_innovations}

File Structure:
{intermediate_plan.file_structure_description}

Project Structure Tree:
{intermediate_plan.project_structure_tree}

Dataset Plan:
{intermediate_plan.dataset_plan}

Model Plan:
{intermediate_plan.model_plan}

Training Plan:
{intermediate_plan.training_plan}

Testing Plan:
{intermediate_plan.testing_plan}

Implementation Steps:
{intermediate_plan.implementation_steps}

Implementation Notes:
{intermediate_plan.implementation_notes}

Potential Challenges:
{intermediate_plan.potential_challenges}

Addressed Issues:
{intermediate_plan.addressed_issues}
"""

        print_subsection("Formatting Code Plan Output")
        print_info("Converting intermediate plan to YAML-compatible format...")

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
        print_info(f"Generated {len(unifier_result.final_output.file_structure)} files")
        print_info(
            f"Implementation roadmap: {len(unifier_result.final_output.implementation_roadmap)} steps"
        )

        print_success("Code planning completed successfully!")
        print_section("CODE PLANNING COMPLETE", "=")

        return unifier_result.final_output

    def plan_sync(self, input_data: str) -> CodePlanOutput:
        """
        Synchronous version of plan method.

        Args:
            input_data: Input string containing research analysis and optional feedback

        Returns:
            CodePlanOutput with complete implementation plan
        """
        import asyncio

        return asyncio.run(self.plan(input_data))


def create_code_plan_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[Dict[str, list]] = None,
    verbose: bool = False,
) -> CodePlanAgent:
    """
    Factory function to create a code planning agent system.

    Args:
        model: Model to use for all agents
        working_dir: Working directory with reference codebases
        tools: Dictionary mapping scenario types to their tools
        verbose: If True, enable verbose hooks to show full LLM responses and tool calls

    Returns:
        CodePlanAgent instance
    """
    return CodePlanAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create the code planning agent
        agent = create_code_plan_agent(model="gpt-4o", working_dir="/workspace")

        # Example with initial planning
        input_data = """
        PreAnalysisOutput:
        - System Architecture: ...
        - Algorithms: ...
        """

        result = await agent.plan(input_data)

        print("Code Plan Generated:")
        print(f"Plan Type: {result.plan_type}")
        print(f"File Structure: {len(result.file_structure)} items")
        print(f"Implementation Steps: {len(result.implementation_roadmap)} steps")

    asyncio.run(main())
