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
import asyncio
import logging
import httpx
import httpcore
from typing import Dict, Optional, Any
from datetime import datetime

# Retry mechanism imports
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

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

from src.memory.decorator import (
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
        pre_analysis_model: Optional[str] = None,
        code_plan_model: Optional[str] = None,
        code_implement_model: Optional[str] = None,
        code_judge_model: Optional[str] = None,
        execute_experiment_model: Optional[str] = None,
        result_analysis_model: Optional[str] = None,
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
            model: Default model to use for all agents (fallback if specific model not provided)
            pre_analysis_model: Model for pre-analysis agent. If None, uses model.
            code_plan_model: Model for code plan agent. If None, uses model.
            code_implement_model: Model for code implement agent. If None, uses model.
            code_judge_model: Model for code judge agent. If None, uses model.
            execute_experiment_model: Model for execute experiment agent. If None, uses model.
            result_analysis_model: Model for result analysis agent. If None, uses model.
            max_iterations: Maximum number of complete workflow iterations
            tools: Optional dictionary mapping agent types to their tools.
                   If None, automatically loads recommended tools for all agents.
            working_dir: Working directory for code operations
            log_dir: Directory for execution logs
            cache_dir: Cache directory for agent outputs (default: ./cached)
            verbose: If True, enable verbose hooks to show full LLM responses and tool calls
        """
        self.model = model
        self.pre_analysis_model = pre_analysis_model or model
        self.code_plan_model = code_plan_model or model
        self.code_implement_model = code_implement_model or model
        self.code_judge_model = code_judge_model or model
        self.execute_experiment_model = execute_experiment_model or model
        self.result_analysis_model = result_analysis_model or model
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
        self.cache_dir = cache_dir  # Save as instance attribute
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        print(f"[CACHE] Cache directory: {cache_dir}")

        # Initialize all sub-agents with their specific models
        self.pre_analysis_agent = create_pre_analysis_agent(
            model=self.pre_analysis_model,
            tools=self.tools.get("pre_analysis"),
            verbose=verbose,
        )

        self.code_plan_agent = create_code_plan_agent(
            model=self.code_plan_model,
            working_dir=working_dir,
            tools=self.tools.get("code_plan"),
            verbose=verbose,
        )

        self.code_implement_agent = create_code_implement_agent(
            model=self.code_implement_model,
            working_dir=working_dir,
            tools=self.tools.get("code_implement"),
            verbose=verbose,
        )

        self.code_judge_agent = create_code_judge_agent(
            model=self.code_judge_model,
            working_dir=working_dir,
            tools=self.tools.get("code_judge"),
            verbose=verbose,
        )

        self.experiment_execute_agent = create_experiment_execute_agent(
            model=self.execute_experiment_model,
            working_dir=working_dir,
            log_dir=log_dir,
            tools=self.tools.get("experiment_execute"),
        )

        self.experiment_analysis_agent = create_experiment_analysis_agent(
            model=self.result_analysis_model,
            working_dir=working_dir,
            tools=self.tools.get("experiment_analysis"),
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
        self,
        research_input: str,
        input_type: str = "paper",
        experiment_id: Optional[str] = None,
    ) -> ExperimentMasterOutput:
        """
        Run the complete experiment workflow using state machine.

        Args:
            research_input: Research paper content or idea description
            input_type: Type of input ('paper' or 'idea')
            experiment_id: Unique identifier for this experiment (if None, generates timestamp-based ID)

        Returns:
            ExperimentMasterOutput with complete workflow results
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set domain for this workflow (use experiment_id as domain)
        self.domain = experiment_id

        # Start experiment in cache manager
        is_new_experiment = self.cache_manager.start_experiment(experiment_id)

        print(f"\n{'='*80}")
        print(f"Experiment ID: {experiment_id}")
        print(
            f"Status: {'New experiment' if is_new_experiment else 'Resuming experiment'}"
        )
        print(f"{'='*80}\n")

        # Prepare workspace information (scan codebases, papers, datasets)
        # Check if prepare info already exists, regardless of new/resumed status
        try:
            from src.agents.experiment_agent.sub_agents.experiment_master.prepare_helpers import (
                prepare_workspace_info,
                save_prepare_info_to_cache,
                load_prepare_info_from_cache,
            )

            # Check if step0_prepare.json exists
            existing_prepare = load_prepare_info_from_cache(self.cache_dir, self.domain)

            if existing_prepare is None:
                # Prepare info doesn't exist, create it
                print("\n[PREPARING WORKSPACE]")
                print("Scanning reference codebases, papers, and datasets...")

                # Get workspace directory (parent of project dir)
                import os

                workspace_dir = self.working_dir

                # Prepare workspace information
                prepare_info = prepare_workspace_info(workspace_dir)

                # Save to cache
                save_result = save_prepare_info_to_cache(
                    self.cache_dir, self.domain, prepare_info
                )

                if save_result:
                    print(f"✓ Workspace prepared:")
                    if prepare_info["reference_codebases"]["scan_result"].get(
                        "success"
                    ):
                        cb_count = prepare_info["reference_codebases"]["scan_result"][
                            "total_count"
                        ]
                        print(f"  - {cb_count} reference codebases found")
                    print(
                        f"  - {prepare_info['reference_papers']['count']} papers available"
                    )
                    print(f"  - {prepare_info['datasets']['count']} datasets available")
                else:
                    print(
                        "⚠ Warning: Could not save workspace preparation info to cache"
                    )
                print()
            else:
                # Prepare info already exists
                print("\n[WORKSPACE ALREADY PREPARED]")
                if (
                    existing_prepare.get("reference_codebases", {})
                    .get("scan_result", {})
                    .get("success")
                ):
                    cb_count = existing_prepare["reference_codebases"]["scan_result"][
                        "total_count"
                    ]
                    print(f"✓ {cb_count} reference codebases available")
                print()

        except Exception as e:
            print(f"⚠ Warning: Could not prepare workspace info: {e}")
            import traceback

            traceback.print_exc()
            print()

        # Initialize workflow context
        context = WorkflowContext(
            research_input=research_input,
            input_type=input_type,
            current_state=WorkflowState.INITIAL,
            iteration_count=0,
            max_iterations=self.max_iterations,
        )

        # Resume workflow context if this is a resumed experiment
        if not is_new_experiment:
            print("\n[RESUMING] Restoring workflow state from cache...")
            resume_success = self.cache_manager.resume_workflow_context(context)
            if not resume_success:
                print("⚠ Could not restore workflow state, starting fresh")
            print()

        # Workflow loop
        step_count = 0
        max_steps = self.max_iterations * 10  # Safety limit to prevent infinite loops

        # Initial transition (only for new experiments)
        if context.current_state == WorkflowState.INITIAL:
            # New experiment - do initial transition to pre_analysis
            transition = self.state_machine.transition(context)
            print(f"\n[STATE TRANSITION]")
            print(f"From: {transition.from_state.value}")
            print(f"To: {transition.to_state.value}")
            print(f"Reason: {transition.reason}")
            transition_data = transition.data or {}
        else:
            # Resumed experiment - check if current state already has output
            print(f"\n[RESUMED] Continuing from: {context.current_state.value}")

            # Check if the current state's output already exists
            state_output_exists = False
            if (
                context.current_state == WorkflowState.PRE_ANALYSIS
                and context.pre_analysis_output
            ):
                state_output_exists = True
            elif (
                context.current_state == WorkflowState.CODE_PLAN
                and context.code_plan_output
            ):
                state_output_exists = True
            elif (
                context.current_state == WorkflowState.CODE_IMPLEMENT
                and context.code_implement_output
            ):
                state_output_exists = True
            elif (
                context.current_state == WorkflowState.CODE_JUDGE
                and context.code_judge_output
            ):
                state_output_exists = True
            elif (
                context.current_state == WorkflowState.EXPERIMENT_EXECUTE
                and context.experiment_execute_output
            ):
                state_output_exists = True
            elif (
                context.current_state == WorkflowState.EXPERIMENT_ANALYSIS
                and context.experiment_analysis_output
            ):
                state_output_exists = True

            if state_output_exists:
                print(
                    f"[SKIP] Current state output already exists, transitioning to next state..."
                )
                # Perform state transition immediately
                transition = self.state_machine.transition(context)
                print(f"\n[STATE TRANSITION]")
                print(f"From: {transition.from_state.value}")
                print(f"To: {transition.to_state.value}")
                print(f"Reason: {transition.reason}")
                transition_data = transition.data or {}
            else:
                print(
                    f"[CONTINUE] Current state output not found, will execute agent..."
                )
                transition_data = {}

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

                # Debug: Show transition data being passed
                if transition_data:
                    print(
                        f"[DEBUG] Transition data keys: {list(transition_data.keys())}"
                    )
                    if "feedback" in transition_data:
                        print(f"[DEBUG] Has feedback from previous step")
                    if "judge_output" in transition_data:
                        print(f"[DEBUG] Has judge_output from previous step")

                try:
                    # ✅ FIX: Pass transition_data from previous state transition
                    # This includes feedback, judge_output, and other important data
                    result = await self._execute_agent(
                        agent_name, context, transition_data
                    )

                    # Debug output
                    print(f"[DEBUG] Agent result type: {type(result)}")
                    print(
                        f"[DEBUG] Agent result: {str(result)[:200] if result else 'None'}"
                    )

                    # Store result in context
                    self._update_context_with_result(
                        context, context.current_state, result
                    )

                    print(f"[COMPLETED] {agent_name} agent finished")

                    # Save workflow snapshot after each agent execution
                    self.cache_manager.save_workflow_snapshot(
                        context, agent_name=agent_name, agent_output=result
                    )

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

            # ✅ FIX: Save transition data for next agent execution
            # This ensures feedback, judge_output, etc. are passed to the next agent
            transition_data = transition.data or {}

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

        # Increment execution step counter for cache management
        context.execution_step_counter += 1
        step_id = context.execution_step_counter

        # Set step number in agent's hooks (if it has verbose hooks)
        if hasattr(agent, "hooks") and agent.hooks is not None:
            agent.hooks.current_step = step_id

        # Check if this is a retry with feedback (should not use cache)
        is_retry = (
            "feedback" in data
            or "judge_output" in data
            or context.checklist_step_retry_count > 0
        )

        # Check cache first (skip cache if retrying)
        cache_key = f"step{step_id}_{agent_name}"
        if not is_retry:
            cached_result = self.cache_manager.load_cache(agent_name, step_id)
            if cached_result is not None:
                print(f"[CACHE] Using cached result for {cache_key}")
                return cached_result
        else:
            print(
                f"[CACHE] Skipping cache for {cache_key} (retry attempt {context.checklist_step_retry_count})"
            )

        # Execute agent based on its interface
        try:
            if hasattr(agent, "process"):
                print(f"[DEBUG] Calling agent.process()")

                # Retry logic for network/protocol errors
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(5),  # Retry up to 5 times
                    wait=wait_exponential(
                        multiplier=1, min=2, max=30
                    ),  # Exponential backoff
                    retry=retry_if_exception_type(
                        (
                            httpx.RemoteProtocolError,
                            httpcore.RemoteProtocolError,
                            httpx.ReadTimeout,
                            httpx.ConnectTimeout,
                            httpx.PoolTimeout,
                            TimeoutError,
                            ConnectionError,
                        )
                    ),
                    reraise=True,
                ):
                    with attempt:
                        if attempt.retry_state.attempt_number > 1:
                            print(
                                f"[RETRY] Attempt {attempt.retry_state.attempt_number} for {agent_name} due to network error..."
                            )
                        result = await agent.process(context, **data)
            else:
                raise ValueError(
                    f"Agent {agent_name} has no recognized execution method (missing 'process')"
                )

            print(f"[DEBUG] Agent method completed, result type: {type(result)}")

            # Save result to cache
            self.cache_manager.save_cache(
                agent_name=agent_name,
                input_data=str(data),
                output=result,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                },
                step_id=step_id,
            )

            return result

        except Exception as e:
            print(f"[ERROR] Agent {agent_name} execution failed: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

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
            # Map overall_analysis to overall_summary
            if hasattr(context.experiment_analysis_output, "overall_analysis"):
                overall_summary = context.experiment_analysis_output.overall_analysis

            # Map unexpected_findings and potential_issues to key_findings
            findings_list = []
            if hasattr(context.experiment_analysis_output, "unexpected_findings"):
                findings_list.extend(
                    context.experiment_analysis_output.unexpected_findings
                )
            if hasattr(context.experiment_analysis_output, "potential_issues"):
                findings_list.extend(
                    context.experiment_analysis_output.potential_issues
                )
            key_findings = findings_list

            # Map next_steps and priority_actions to final_recommendations
            recommendation_parts = []
            if hasattr(context.experiment_analysis_output, "next_steps"):
                recommendation_parts.append(
                    context.experiment_analysis_output.next_steps
                )

            if hasattr(context.experiment_analysis_output, "priority_actions"):
                actions = context.experiment_analysis_output.priority_actions
                if actions:
                    recommendation_parts.append(
                        "Priority Actions:\n" + "\n".join(f"- {a}" for a in actions)
                    )

            if recommendation_parts:
                final_recommendations = "\n\n".join(recommendation_parts)

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
        self,
        research_input: str,
        input_type: str = "paper",
        experiment_id: Optional[str] = None,
    ) -> ExperimentMasterOutput:
        """
        Synchronous version of run_workflow.

        Args:
            research_input: Research paper content or idea description
            input_type: Type of input ('paper' or 'idea')
            experiment_id: Unique identifier for this experiment

        Returns:
            ExperimentMasterOutput with complete workflow results
        """
        import asyncio

        return asyncio.run(self.run_workflow(research_input, input_type, experiment_id))


def create_experiment_master_agent(
    model: str = "gpt-4o",
    pre_analysis_model: Optional[str] = None,
    code_plan_model: Optional[str] = None,
    code_implement_model: Optional[str] = None,
    code_judge_model: Optional[str] = None,
    execute_experiment_model: Optional[str] = None,
    result_analysis_model: Optional[str] = None,
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
        model: Default model to use for all agents (fallback if specific model not provided)
        pre_analysis_model: Model for pre-analysis agent. If None, uses model.
        code_plan_model: Model for code plan agent. If None, uses model.
        code_implement_model: Model for code implement agent. If None, uses model.
        code_judge_model: Model for code judge agent. If None, uses model.
        execute_experiment_model: Model for execute experiment agent. If None, uses model.
        result_analysis_model: Model for result analysis agent. If None, uses model.
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
        pre_analysis_model=pre_analysis_model,
        code_plan_model=code_plan_model,
        code_implement_model=code_implement_model,
        code_judge_model=code_judge_model,
        execute_experiment_model=execute_experiment_model,
        result_analysis_model=result_analysis_model,
        max_iterations=max_iterations,
        tools=tools,
        working_dir=working_dir,
        log_dir=log_dir,
        cache_dir=cache_dir,
        verbose=verbose,
    )
