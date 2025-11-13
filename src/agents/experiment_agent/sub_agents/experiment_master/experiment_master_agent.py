"""
Experiment Master Agent - Manages the complete experiment workflow using a state machine.

This agent coordinates all sub-agents in the experiment workflow using a
rule-based state machine that determines transitions based on agent outputs.

Workflow:
1. Pre-analysis: Analyze research paper/idea
2. Code Plan: Generate implementation plan
3. Code Implement: Implement the code
4. Code Judge: Review code consistency
5. Experiment Execute: Run the experiment
6. Experiment Analysis: Analyze results

The state machine handles feedback loops automatically:
- If code judge fails -> back to code plan or implementation
- If execution errors -> back to code implement
- If analysis suggests improvements -> back to appropriate stage
"""

import os
from typing import Dict, Optional, Any
from datetime import datetime

from src.agents.experiment_agent.sub_agents.experiment_master.output_schemas import (
    ExperimentMasterOutput,
    WorkflowStep,
)
from src.agents.experiment_agent.sub_agents.experiment_master.workflow_state_machine import (
    WorkflowStateMachine,
    WorkflowState,
    WorkflowContext,
)
from src.agents.experiment_agent.sub_agents.experiment_master.cache_manager import (
    CacheManager,
)
from src.agents.experiment_agent.sub_agents.pre_analysis import (
    create_pre_analysis_agent,
)
from src.agents.experiment_agent.sub_agents.code_plan import (
    create_code_plan_agent,
)
from src.agents.experiment_agent.sub_agents.code_implement import (
    create_code_implement_agent,
)
from src.agents.experiment_agent.sub_agents.code_judge import (
    create_code_judge_agent,
)
from src.agents.experiment_agent.sub_agents.experiment_execute import (
    create_experiment_execute_agent,
)
from src.agents.experiment_agent.sub_agents.experiment_analysis import (
    create_experiment_analysis_agent,
)
from src.memory.api.slot_process_api import (
    SlotProcess,
)
from src.memory.memory_system.decorator import (
    short_term_slot_trace,
)
from src.memory.api.faiss_memory_system_api import (
    FAISSMemorySystem,
)


# Orchestrator removed - using rule-based state machine instead


class ExperimentMasterAgent:
    """
    Main experiment master agent that manages the complete workflow using a state machine.

    This agent:
    1. Manages all sub-agents
    2. Uses a rule-based state machine to determine workflow transitions
    3. Handles iterative improvement cycles automatically
    4. Tracks workflow history and context
    5. Produces final consolidated results
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 5,
        tools: Optional[Dict[str, list]] = None,
        working_dir: Optional[str] = None,
        log_dir: str = "./experiment_logs",
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the experiment master agent.

        Args:
            model: Model to use for all agents
            max_iterations: Maximum number of complete workflow iterations
            tools: Optional dictionary mapping agent types to their tools.
                   If None, automatically loads recommended tools for all agents.
            working_dir: Working directory for code operations
            log_dir: Directory for execution logs
            cache_dir: Cache directory for agent outputs (default: ./cached)
            verbose: If True, enable verbose hooks to show full LLM responses and tool calls
        """
        self.model = model
        self.max_iterations = max_iterations
        self.working_dir = working_dir
        self.log_dir = log_dir
        self.verbose = verbose
        self.slot_process = SlotProcess()
        self.semantic_memory_store = FAISSMemorySystem(memory_type="semantic")
        self.episodic_memory_store = FAISSMemorySystem(memory_type="episodic")
        self.procedural_memory_store = FAISSMemorySystem(memory_type="procedural")

        if tools is None:
            from src.agents.experiment_agent.sub_agents.experiment_master import (
                get_all_agent_tools,
            )

            self.tools = get_all_agent_tools()
        else:
            self.tools = tools

        # Initialize state machine
        self.state_machine = WorkflowStateMachine()

        # Initialize cache manager
        if cache_dir is None:
            cache_dir = "./cached"
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        print(f"[CACHE] Cache directory: {cache_dir}")

        # Initialize all sub-agents
        self.pre_analysis_agent = create_pre_analysis_agent(
            model=model, tools=self.tools.get("pre_analysis"), verbose=verbose
        )

        self.code_plan_agent = create_code_plan_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools.get("code_plan"),
        )

        self.code_implement_agent = create_code_implement_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools.get("code_implement"),
        )

        self.code_judge_agent = create_code_judge_agent(
            model=model, tools=self.tools.get("code_judge")
        )

        self.experiment_execute_agent = create_experiment_execute_agent(
            model=model,
            log_dir=log_dir,
            tools=self.tools.get("experiment_execute"),
        )

        self.experiment_analysis_agent = create_experiment_analysis_agent(
            model=model, tools=self.tools.get("experiment_analysis")
        )

        # Agent mapping for easy access
        self.agents = {
            "pre_analysis": self.pre_analysis_agent,
            "code_plan": self.code_plan_agent,
            "code_implement": self.code_implement_agent,
            "code_judge": self.code_judge_agent,
            "experiment_execute": self.experiment_execute_agent,
            "experiment_analysis": self.experiment_analysis_agent,
        }

    async def run_workflow(
        self, research_input: str, input_type: str = "paper"
    ) -> ExperimentMasterOutput:
        """
        Run the complete experiment workflow using state machine.

        Args:
            research_input: Research paper content or idea description
            input_type: Type of input ('paper' or 'idea')

        Returns:
            ExperimentMasterOutput with complete workflow results
        """
        print(f"\n{'='*80}")
        print("Starting workflow with state machine...")
        print(f"{'='*80}\n")

        # Initialize workflow context
        context = WorkflowContext(
            research_input=research_input,
            input_type=input_type,
            current_state=WorkflowState.INITIAL,
            iteration_count=0,
            max_iterations=self.max_iterations,
        )

        # Workflow loop
        step_count = 0
        max_steps = self.max_iterations * 10  # Safety limit to prevent infinite loops

        # Initial transition from INITIAL to first real state
        transition = self.state_machine.transition(context)
        print(f"\n[STATE TRANSITION]")
        print(f"From: {transition.from_state.value}")
        print(f"To: {transition.to_state.value}")
        print(f"Reason: {transition.reason}")

        while not self.state_machine.is_terminal_state(context.current_state):
            step_count += 1

            if step_count > max_steps:
                print(f"[ERROR] Workflow exceeded max steps ({max_steps})")
                context.current_state = WorkflowState.FAILED
                context.last_error = "Maximum steps exceeded"
                break

            # Execute agent for the current state (if applicable)
            agent_name = self.state_machine.get_required_agent(context.current_state)
            if agent_name:
                print(f"\n[EXECUTING] {agent_name} agent (step {step_count})...")

                try:
                    result = await self._execute_agent(agent_name, context, {})

                except Exception as e:
                    print(f"[ERROR] Agent execution failed: {str(e)}")
                    context.last_error = str(e)
                    context.current_state = WorkflowState.FAILED
                    break

            # Transition to next state based on current results
            transition = self.state_machine.transition(context)

            print(f"\n[STATE TRANSITION]")
            print(f"From: {transition.from_state.value}")
            print(f"To: {transition.to_state.value}")
            print(f"Reason: {transition.reason}")

        # Build final output
        print(f"\n{'='*80}")
        print(f"Workflow completed: {context.current_state.value}")
        print(f"{'='*80}\n")

        return self._build_final_output(context)

    @short_term_slot_trace(context=context)
    async def _execute_agent(
        self, agent_name: str, context: WorkflowContext, data: Dict[str, Any]
    ) -> Any:
        """
        Execute a specific agent.

        Args:
            agent_name: Name of agent to execute
            context: Current workflow context
            data: Additional data for agent execution

        Returns:
            Agent output
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Prepare input for agent based on state
        agent_input = self._prepare_agent_input(agent_name, context, data)

        print(f"[DEBUG] Agent input length: {len(agent_input)} characters")
        print(f"[DEBUG] Agent input preview:\n{agent_input[:300]}...")

        # Check cache first
        cached_result = self.cache_manager.load_cache(agent_name, agent_input)
        if cached_result is not None:
            print(f"[CACHE] Using cached result for {agent_name}")
            return cached_result

        # Execute agent based on its interface
        try:
            if hasattr(agent, "analyze"):
                # Pre-analysis agent uses analyze method
                print(f"[DEBUG] Calling agent.analyze()")
                result = await agent.analyze(agent_input)
            elif hasattr(agent, "plan"):
                # Code plan agent
                print(f"[DEBUG] Calling agent.plan()")
                result = await agent.plan(agent_input)
            elif hasattr(agent, "implement"):
                # Code implement agent
                print(f"[DEBUG] Calling agent.implement()")
                result = await agent.implement(agent_input)
            elif hasattr(agent, "judge"):
                # Code judge agent
                print(f"[DEBUG] Calling agent.judge()")
                result = await agent.judge(agent_input)
            elif hasattr(agent, "execute"):
                # Experiment execute agent
                print(f"[DEBUG] Calling agent.execute()")
                result = await agent.execute(agent_input)
            elif hasattr(agent, "run"):
                # Generic run method
                print(f"[DEBUG] Calling agent.run()")
                result = await agent.run(agent_input)
            else:
                raise ValueError(
                    f"Agent {agent_name} has no recognized execution method"
                )

            print(f"[DEBUG] Agent method completed, result type: {type(result)}")

            # Save result to cache
            self.cache_manager.save_cache(
                agent_name=agent_name,
                input_data=agent_input,
                output=result,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                },
            )

            return result

        except Exception as e:
            print(f"[ERROR] Agent {agent_name} execution failed: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _format_list(self, items: list) -> str:
        """Format a list into a readable string."""
        if not items:
            return "N/A"
        return "\n".join(f"- {item}" for item in items)

    def _format_dict(self, d: dict) -> str:
        """Format a dictionary into a readable string."""
        if not d:
            return "N/A"
        return "\n".join(f"- {key}: {value}" for key, value in d.items())

    def _prepare_agent_input(
        self, agent_name: str, context: WorkflowContext, data: Dict[str, Any]
    ) -> str:
        """Prepare input string for agent execution."""
        # Build input based on agent type
        if agent_name == "pre_analysis":
            if context.input_type == "idea":
                # Parse idea JSON and format with explicit fields
                import json

                try:
                    idea_data = json.loads(context.research_input)
                    idea_obj = idea_data.get("idea", {})

                    # Format the idea with explicit field labels
                    formatted_input = f"""Analyze the following research idea:

=== IDEA INFORMATION ===

**Title:** {idea_obj.get("title", "N/A")}

**Description:**
{idea_obj.get("description", "N/A")}

**Key Innovations:**
{self._format_list(idea_obj.get("key_innovations", []))}

**Methodology:**
{self._format_dict(idea_obj.get("methodology", {}))}

**Expected Outcomes:**
{self._format_list(idea_obj.get("expected_outcomes", []))}

**Reference Papers:**
{self._format_list(idea_data.get("reference_papers", []))}

=== TASK ===
Provide a comprehensive analysis covering:
1. System architecture and conceptual framework
2. Key innovations and theoretical basis
3. Algorithms and mathematical formulations (design the specific algorithms based on the methodology)
4. Technical details and computational methods
5. Implementation guidance"""

                    return formatted_input

                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return f"""Analyze the following research idea:

{context.research_input}

Provide a comprehensive analysis covering system architecture, algorithms, innovations, and technical details."""
            else:
                # For papers, use the original format
                return f"""Analyze the following research {context.input_type}:

{context.research_input}

Provide a comprehensive analysis covering system architecture, algorithms, innovations, and technical details."""

        elif agent_name == "code_plan":
            feedback = data.get("feedback", "")
            feedback_section = (
                f"\nFeedback from previous iteration:\n{feedback}\n" if feedback else ""
            )

            return f"""Create an implementation plan based on the following analysis:

{context.pre_analysis_output}
{feedback_section}
Provide a detailed implementation plan including file structure, model architecture, and training configuration."""

        elif agent_name == "code_implement":
            error = data.get("error", "")
            feedback = data.get("feedback", "")

            error_section = f"\nError to fix:\n{error}\n" if error else ""
            feedback_section = f"\nFeedback:\n{feedback}\n" if feedback else ""

            return f"""Implement the code based on the following plan:

{context.code_plan_output}
{error_section}{feedback_section}
Write the complete implementation code."""

        elif agent_name == "code_judge":
            return f"""Review the following code implementation:

Analysis:
{context.pre_analysis_output}

Plan:
{context.code_plan_output}

Implementation:
{context.code_implement_output}

Evaluate consistency and provide recommendations."""

        elif agent_name == "experiment_execute":
            return f"""Execute the following code:

{context.code_implement_output}

Run the experiment and capture results."""

        elif agent_name == "experiment_analysis":
            return f"""Analyze the experiment results:

Expected (from analysis):
{context.pre_analysis_output}

Execution Results:
{context.experiment_execute_output}

Provide analysis and recommendations."""

        return ""

    def _update_context_with_result(
        self, context: WorkflowContext, state: WorkflowState, result: Any
    ):
        """Update context with agent result."""
        if state == WorkflowState.PRE_ANALYSIS:
            context.pre_analysis_output = result
        elif state == WorkflowState.CODE_PLAN:
            context.code_plan_output = result
        elif state == WorkflowState.CODE_IMPLEMENT:
            context.code_implement_output = result
        elif state == WorkflowState.CODE_JUDGE:
            context.code_judge_output = result
        elif state == WorkflowState.EXPERIMENT_EXECUTE:
            context.experiment_execute_output = result
        elif state == WorkflowState.EXPERIMENT_ANALYSIS:
            context.experiment_analysis_output = result

    def _build_final_output(self, context: WorkflowContext) -> ExperimentMasterOutput:
        """Build final output from workflow context."""
        workflow_completed = context.current_state == WorkflowState.COMPLETED
        final_status = context.current_state.value

        # Build workflow history from state transitions
        workflow_history = []
        for i, transition in enumerate(context.state_history, 1):
            step = WorkflowStep(
                step_number=i,
                agent_name=self.state_machine.get_required_agent(transition.to_state)
                or transition.to_state.value,
                status=(
                    "completed"
                    if transition.to_state != WorkflowState.FAILED
                    else "failed"
                ),
                summary=transition.reason[:200],
            )
            workflow_history.append(step)

        # Extract summary from analysis output
        overall_summary = ""
        key_findings = []
        final_recommendations = ""

        if context.experiment_analysis_output:
            if hasattr(context.experiment_analysis_output, "summary"):
                overall_summary = context.experiment_analysis_output.summary
            if hasattr(context.experiment_analysis_output, "findings"):
                key_findings = context.experiment_analysis_output.findings
            if hasattr(context.experiment_analysis_output, "recommendations"):
                final_recommendations = (
                    context.experiment_analysis_output.recommendations
                )

        if not overall_summary:
            overall_summary = f"Workflow {final_status} after {context.iteration_count} iterations. Final state: {context.current_state.value}"

        return ExperimentMasterOutput(
            workflow_completed=workflow_completed,
            final_status=final_status,
            total_iterations=context.iteration_count,
            workflow_history=workflow_history,
            overall_summary=overall_summary,
            key_findings=key_findings,
            final_recommendations=final_recommendations,
        )

    def run_workflow_sync(
        self, research_input: str, input_type: str = "paper"
    ) -> ExperimentMasterOutput:
        """
        Synchronous version of run_workflow.

        Args:
            research_input: Research paper content or idea description
            input_type: Type of input ('paper' or 'idea')

        Returns:
            ExperimentMasterOutput with complete workflow results
        """
        import asyncio

        return asyncio.run(self.run_workflow(research_input, input_type))


def create_experiment_master_agent(
    model: str = "gpt-4o",
    max_iterations: int = 5,
    tools: Optional[Dict[str, list]] = None,
    working_dir: Optional[str] = None,
    log_dir: str = "./experiment_logs",
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> ExperimentMasterAgent:
    """
    Factory function to create an experiment master agent.

    Args:
        model: Model to use for all agents
        max_iterations: Maximum number of workflow iterations
        tools: Dictionary mapping agent types to their tools
        working_dir: Working directory for code operations
        log_dir: Directory for execution logs
        cache_dir: Cache directory for agent outputs
        verbose: If True, enable verbose hooks to show full LLM responses and tool calls

    Returns:
        ExperimentMasterAgent instance
    """
    return ExperimentMasterAgent(
        model=model,
        max_iterations=max_iterations,
        tools=tools,
        working_dir=working_dir,
        log_dir=log_dir,
        cache_dir=cache_dir,
        verbose=verbose,
    )
