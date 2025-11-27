"""
Code Plan Agent - Main orchestrator for code planning using handoff mechanism.

This agent uses OpenAI agents library's handoff mechanism to route inputs
to appropriate scenario-specific planners and produces unified YAML-format code plans.
"""

from typing import Dict, Optional, Any

from src.agents.experiment_agent.config import (
    OUTPUT_UNIFIER_MODEL,
)
from agents import Agent, Runner, handoff
from src.agents.experiment_agent.logger import create_verbose_hooks


from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
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

from src.agents.experiment_agent.utils.print_utils import *


def create_triage_agent(
    initial_planner: Agent,
    judge_feedback_planner: Agent,
    error_feedback_planner: Agent,
    analysis_feedback_planner: Agent,
) -> Agent:
    """
    Create a triage agent that uses handoffs to route to appropriate planners.
    """

    instructions = """You are the Routing Dispatcher. Your ONLY job is to analyze the input context and handoff to the correct specialist.

### ROUTING RULES

1. **INITIAL PLANNING**
   - Context: Research Analysis only. No previous failures.
   - Action: Handoff to `Initial Plan Agent`.

2. **CODE REVIEW (JUDGE) FEEDBACK**
   - Context: Code was rejected by the Judge/QA.
   - Keywords: "Issues Found", "Code Quality", "Logic Error".
   - Action: Handoff to `Judge Feedback Plan Agent`.

3. **RUNTIME ERROR FEEDBACK**
   - Context: Execution crashed.
   - Keywords: "Traceback", "Exception", "Runtime Error".
   - Action: Handoff to `Error Feedback Plan Agent`.

4. **ANALYSIS FEEDBACK**
   - Context: Experiment ran successfully but results were poor (Low accuracy, etc.).
   - Keywords: "Analysis Result", "Metrics", "Optimization".
   - Action: Handoff to `Analysis Feedback Plan Agent`.

### EXECUTION
- Do NOT generate a plan yourself.
- IMMEDIATELY call the handoff tool matching the scenario.
"""

    triage = Agent(
        name="Planning Triage Agent",
        instructions=instructions,
        handoffs=[
            handoff(
                initial_planner,
                tool_description_override="Handoff to Initial Plan Agent.",
            ),
            handoff(
                judge_feedback_planner,
                tool_description_override="Handoff to Judge Feedback Plan Agent.",
            ),
            handoff(
                error_feedback_planner,
                tool_description_override="Handoff to Error Feedback Plan Agent.",
            ),
            handoff(
                analysis_feedback_planner,
                tool_description_override="Handoff to Analysis Feedback Plan Agent.",
            ),
        ],
    )

    return triage


def create_code_plan_unifier_agent(model: str = "gpt-4o") -> Agent:
    return Agent(
        name="Code Plan Output Unifier",
        instructions="""You are an expert data structuring assistant.
Your task is to convert the raw textual code plan into a structured `CodePlanOutput` object.

Input text will contain sections for:
- Research Summary
- Key Innovations
- File Structure
- Dataset Plan
- Model Plan
- Training Plan
- Testing Plan
- Implementation Checklist
- Implementation Notes & Challenges

Map these sections to the corresponding fields in the output schema.
Preserve the detailed content of each section.
Ensure `file_structure` is a list of `FileStructureItem` (path, type, description).
Ensure `implementation_checklist` is a list of `ChecklistItem` (step_id, title, description, files_to_create, files_to_modify, acceptance_criteria, dependencies, estimated_complexity).
Set `plan_type` based on the context (initial, judge_feedback, error_feedback, analysis_feedback).
""",
        output_type=CodePlanOutput,
        model=OUTPUT_UNIFIER_MODEL,
    )


class CodePlanAgent:
    """
    Main code planning agent that orchestrates the entire planning workflow using handoffs.
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
        self.output_unifier = create_code_plan_unifier_agent(model=model)

        # Expose triage agent as main agent for handoff compatibility
        self.agent = self.triage_agent

    async def process(self, context: Any, **kwargs) -> CodePlanOutput:
        """
        Process the current step using context data.
        """
        data = kwargs
        feedback = data.get("feedback", "")
        feedback_section = f"\nPRIORITY FEEDBACK:\n{feedback}\n" if feedback else ""

        summary = ""
        if hasattr(context.pre_analysis_output, "summary"):
            summary = f"RESEARCH SUMMARY:\n{context.pre_analysis_output.summary}\n\n"

        formatted_input = f"""
PLANNING REQUEST

{summary}
{context.pre_analysis_output}

{feedback_section}

Objective: Generate/Update the Code Plan based on the inputs above.
"""

        return await self.plan(formatted_input)

    async def plan(self, input_data: str) -> CodePlanOutput:
        """
        Generate code implementation plan based on input.
        """
        print_section("CODE PLANNING WORKFLOW", "=")

        print_subsection("Planning Triage & Execution")

        # Use streamed version for real-time output
        planning_stream = Runner.run_streamed(
            self.triage_agent, input_data, hooks=self.hooks, max_turns=100
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

        active_agent_name = (
            planning_result.last_agent.name
            if planning_result.last_agent
            else "Unknown Agent"
        )

        print_success(f"Planning text generated using: {active_agent_name}")
        print_subsection("Unifying Output Format")

        # Step 2: Unify output
        unifier_input = f"""
Please convert the following code plan into the structured `CodePlanOutput` format.

=== CODE PLAN ===
{final_text}
"""
        unifier_stream = Runner.run_streamed(
            self.output_unifier, unifier_input, hooks=None
        )

        async for _ in unifier_stream.stream_events():
            pass

        final_plan = unifier_stream.final_output

        print_info(f"Plan type: {final_plan.plan_type}")
        print_section("CODE PLANNING COMPLETE", "=")

        return final_plan

    def plan_sync(self, input_data: str) -> CodePlanOutput:
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
