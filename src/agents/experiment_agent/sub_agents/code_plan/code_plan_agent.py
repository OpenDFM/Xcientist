"""
Code Plan Agent - Main orchestrator for code planning using deterministic routing.

This agent uses deterministic logic (based on pending_feedback_type) to route inputs
to appropriate scenario-specific planners and produces unified YAML-format code plans.
"""

from typing import Dict, Optional, Any

from agents import Agent, Runner
from src.agents.experiment_agent.logger import create_verbose_hooks


from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)
from src.agents.experiment_agent.sub_agents.code_plan.initial_plan_agent import (
    create_initial_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.error_feedback_plan_agent import (
    create_error_feedback_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan.analysis_feedback_plan_agent import (
    create_analysis_feedback_plan_agent,
)

from src.agents.experiment_agent.utils.print_utils import *
from src.agents.experiment_agent.utils.json_utils import (
    extract_and_parse_json,
    generate_json_schema_instruction,
    JSONParseError,
)


class CodePlanAgent:
    """
    Main code planning agent that orchestrates the entire planning workflow.
    
    Uses deterministic routing based on pending_feedback_type to select the correct planner:
    - None or "initial" -> Initial Plan Agent
    - "error_feedback" -> Error Feedback Plan Agent
    - "analysis_feedback" -> Analysis Feedback Plan Agent
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[Dict[str, list]] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.working_dir = working_dir
        self.verbose = verbose
        # Always create hooks to show tool arguments
        # verbose mode controls whether to show detailed responses and results
        self.hooks = create_verbose_hooks(
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=True,  # Always show tool arguments
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

        # Default agent (for compatibility)
        self.agent = self.initial_planner

    def _select_planner(self, feedback_type: Optional[str]) -> tuple[Agent, str]:
        """
        Select the appropriate planner based on feedback type.
        
        Args:
            feedback_type: One of None, "initial", "error_feedback", "analysis_feedback"
            
        Returns:
            Tuple of (planner_agent, plan_type_name)
        """
        if feedback_type == "error_feedback":
            return self.error_feedback_planner, "error_feedback"
        elif feedback_type == "analysis_feedback":
            return self.analysis_feedback_planner, "analysis_feedback"
        else:
            # Default to initial planner
            return self.initial_planner, "initial"

    def _extract_error_feedback(self, context: Any) -> str:
        output = context.get("experiment_execute_output", None)
        if not output:
            return ""
            
        error_message = output.get("error_message", "")
        execution_summary = output.get("execution_summary", "")
        stdout_preview = output.get("stdout_preview", "")
        stderr_preview = output.get("stderr_preview", "")
        
        retry_count = context.get("retry_count", 0)
        max_retries = context.get("max_retries", 10)
        
        return f"""## Runtime Error Feedback

### Error Summary
{error_message}

### Execution Summary
{execution_summary if execution_summary else "N/A"}

### Stdout Preview
{stdout_preview if stdout_preview else "N/A"}

### Stderr/Error Log
{stderr_preview if stderr_preview else "N/A"}

### Retry Information
Attempt: {retry_count}/{max_retries}
"""

    def _extract_analysis_feedback(self, context: Any) -> str:
        output = context.get("experiment_analysis_output", None)
        if not output:
            return ""
        
        # Unified access
        feedback = output.get("feedback", "")
            
        return f"## Analysis Feedback\n\n{feedback}"

    async def process(self, context: Any, **kwargs) -> CodePlanOutput:
        """
        Process the current step using context data.
        
        Uses deterministic routing based on context.pending_feedback_type.
        """
        # Determine feedback type from context
        feedback_type = context.get("pending_feedback_type", None)
        
        # Extract feedback
        feedback = ""
        if feedback_type == "error_feedback":
            feedback = self._extract_error_feedback(context)
        elif feedback_type == "analysis_feedback":
            feedback = self._extract_analysis_feedback(context)
            
        feedback_section = f"\nPRIORITY FEEDBACK:\n{feedback}\n" if feedback else ""

        summary = ""
        code_repos_info = ""
        if context.get("pre_analysis_output", None):
            if context.pre_analysis_output.get("summary", None):
                summary = context.pre_analysis_output.get("summary", "")
                summary = f"RESEARCH SUMMARY:\n{summary}\n\n"
            if context.pre_analysis_output.get("code_repos_info", None):
                code_repos_info = context.pre_analysis_output.get("code_repos_info", "")
                code_repos_info = f"CODE REPOSITORIES ANALYSIS:\n{code_repos_info}\n\n"
        
        # Select appropriate planner
        planner, plan_type = self._select_planner(feedback_type)
        
        print_info(f"Routing to: {planner.name} (feedback_type={feedback_type})")

        formatted_input = f"""
PLANNING REQUEST

{summary}
{code_repos_info}
{context.get("pre_analysis_output", "")}

{feedback_section}

Objective: Generate/Update the Code Plan based on the inputs above.
Use the code repositories analysis to identify reusable components and patterns.
"""

        return await self.plan_with_planner(formatted_input, planner, plan_type)

    async def plan_with_planner(self, input_data: str, planner: Agent, plan_type: str) -> CodePlanOutput:
        """
        Generate code implementation plan using specified planner.
        """
        print_section("CODE PLANNING WORKFLOW", "=")

        print_subsection(f"Executing {planner.name}")

        # Use streamed version for real-time output
        planning_stream = Runner.run_streamed(
            planner, input_data, hooks=self.hooks, max_turns=100
        )
        async for event in planning_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)

        planning_result = planning_stream

        # Get raw text output (duplex mode)
        final_text = ""
        if hasattr(planning_result, "final_output") and isinstance(
            planning_result.final_output, str
        ):
            final_text = planning_result.final_output
        elif hasattr(planning_result, "chat_history") and planning_result.chat_history:
            final_text = planning_result.chat_history[-1].content

        if not final_text:
            print_error("Planning failed to produce output")
            raise RuntimeError("Planning agent failed to produce output")

        print_success(f"Planning text generated using: {planner.name}")
        print_subsection("Parsing JSON Output")

        # Extract and parse JSON from the planner output using post-processing
        # Use raise_on_failure=True to trigger retry in master agent
        try:
            final_plan = extract_and_parse_json(final_text, CodePlanOutput, raise_on_failure=True)
        except JSONParseError as e:
            # Re-raise JSONParseError to trigger retry in master agent
            print_error(f"JSON parsing failed, will trigger retry: {e}")
            raise

        print_info(f"Plan type: {final_plan.plan_type}")
        print_section("CODE PLANNING COMPLETE", "=")

        return final_plan

    async def plan(self, input_data: str, feedback_type: Optional[str] = None) -> CodePlanOutput:
        """
        Generate code implementation plan based on input.
        
        Args:
            input_data: Input text for planning
            feedback_type: Optional feedback type to determine planner
        """
        planner, plan_type = self._select_planner(feedback_type)
        return await self.plan_with_planner(input_data, planner, plan_type)

    def plan_sync(self, input_data: str, feedback_type: Optional[str] = None) -> CodePlanOutput:
        import asyncio

        return asyncio.run(self.plan(input_data, feedback_type))


def create_code_plan_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[Dict[str, list]] = None,
    verbose: bool = False,
) -> CodePlanAgent:
    """
    Factory function to create a code planning agent system.
    """
    return CodePlanAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_code_plan_agent(model="gpt-4o", working_dir="/workspace")
        print("Agent created.")

    asyncio.run(main())
