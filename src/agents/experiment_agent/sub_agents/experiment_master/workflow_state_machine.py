"""
Rule-based State Machine for Experiment Workflow.

This module implements a state machine that manages the workflow transitions
between different stages of the experiment process without relying on an orchestrator.

States:
- INITIAL: Starting state
- PRE_ANALYSIS: Analyzing research input
- CODE_PLAN: Creating implementation plan
- CODE_IMPLEMENT: Implementing the code
- CODE_JUDGE: Reviewing code quality
- EXPERIMENT_EXECUTE: Running the experiment
- EXPERIMENT_ANALYSIS: Analyzing results
- COMPLETED: Terminal state (success)
- FAILED: Terminal state (failure)

Transitions are based on rules evaluated from each state's output.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


class WorkflowState(Enum):
    """Enumeration of all possible workflow states."""

    INITIAL = "initial"
    PRE_ANALYSIS = "pre_analysis"
    CODE_PLAN = "code_plan"
    CODE_IMPLEMENT = "code_implement"
    CODE_JUDGE = "code_judge"
    EXPERIMENT_EXECUTE = "experiment_execute"
    EXPERIMENT_ANALYSIS = "experiment_analysis"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StateTransition:
    """Represents a state transition with reason."""

    from_state: WorkflowState
    to_state: WorkflowState
    reason: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowContext:
    """
    Context object that holds all workflow data.

    This is passed between states and accumulates information
    as the workflow progresses.
    """

    # Input
    research_input: str
    input_type: str  # "paper" or "idea"

    # State tracking
    current_state: WorkflowState
    iteration_count: int
    max_iterations: int

    # Outputs from each stage
    pre_analysis_output: Optional[Any] = None
    code_plan_output: Optional[Any] = None
    code_implement_output: Optional[Any] = None
    code_judge_output: Optional[Any] = None
    experiment_execute_output: Optional[Any] = None
    experiment_analysis_output: Optional[Any] = None

    # Checklist tracking for iterative implementation
    current_checklist_step: int = 0  # Index of current step being implemented
    completed_checklist_steps: List[int] = None  # List of completed step IDs
    checklist_step_retry_count: int = 0  # Retry count for current step
    max_step_retries: int = 999  # Maximum retries per step (effectively unlimited)

    # Global execution tracking (for cache management)
    execution_step_counter: int = (
        0  # Global counter for all agent executions (increments with each agent call)
    )

    # Error tracking
    last_error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # History
    state_history: List[StateTransition] = None

    def __post_init__(self):
        if self.state_history is None:
            self.state_history = []
        if self.completed_checklist_steps is None:
            self.completed_checklist_steps = []


class WorkflowStateMachine:
    """
    Rule-based state machine for managing experiment workflow.

    This class implements a deterministic state machine that determines
    the next state based on the current state and the results of agent execution.
    """

    def __init__(self):
        """Initialize the state machine with transition rules."""
        # Define state transition rules
        # Format: (from_state, condition_func) -> to_state
        self._transition_rules = self._build_transition_rules()

    def _build_transition_rules(self) -> Dict:
        """Build the transition rule table."""
        return {
            WorkflowState.INITIAL: self._from_initial,
            WorkflowState.PRE_ANALYSIS: self._from_pre_analysis,
            WorkflowState.CODE_PLAN: self._from_code_plan,
            WorkflowState.CODE_IMPLEMENT: self._from_code_implement,
            WorkflowState.CODE_JUDGE: self._from_code_judge,
            WorkflowState.EXPERIMENT_EXECUTE: self._from_experiment_execute,
            WorkflowState.EXPERIMENT_ANALYSIS: self._from_experiment_analysis,
        }

    def _get_checklist(self, code_plan_output: Any) -> list:
        """
        Get implementation checklist from code plan output.
        Supports both object and dict formats (for cached data).
        Converts dict items to objects for consistent attribute access.

        Args:
            code_plan_output: Code plan output (object or dict)

        Returns:
            List of checklist items (as objects with attribute access)
        """
        checklist = []
        if isinstance(code_plan_output, dict):
            checklist = code_plan_output.get("implementation_checklist", [])
        elif hasattr(code_plan_output, "implementation_checklist"):
            checklist = code_plan_output.implementation_checklist

        # Convert dict items to objects for consistent attribute access
        if checklist and isinstance(checklist[0], dict):
            # Create simple objects from dicts
            class ChecklistStep:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)

            checklist = [ChecklistStep(item) for item in checklist]

        return checklist

    def get_next_state(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """
        Determine the next state based on current context.

        Args:
            context: Current workflow context

        Returns:
            Tuple of (next_state, reason, additional_data)
        """
        current_state = context.current_state

        # Check for terminal states
        if current_state in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            return current_state, "Already in terminal state", None

        # Check max iterations
        if context.iteration_count >= context.max_iterations:
            return (
                WorkflowState.FAILED,
                f"Maximum iterations ({context.max_iterations}) reached",
                None,
            )

        # Apply transition rules
        if current_state in self._transition_rules:
            rule_func = self._transition_rules[current_state]
            return rule_func(context)

        # Default: failed state if no rule found
        return (
            WorkflowState.FAILED,
            f"No transition rule for state {current_state}",
            None,
        )

    def _from_initial(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """Transition from INITIAL state."""
        # Always start with pre-analysis
        return (
            WorkflowState.PRE_ANALYSIS,
            "Starting workflow with pre-analysis",
            {"input": context.research_input, "input_type": context.input_type},
        )

    def _from_pre_analysis(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """Transition from PRE_ANALYSIS state."""
        if context.pre_analysis_output is None:
            return WorkflowState.FAILED, "Pre-analysis produced no output", None

        # Check if pre-analysis was successful
        if hasattr(context.pre_analysis_output, "status"):
            if context.pre_analysis_output.status == "failed":
                return (
                    WorkflowState.FAILED,
                    "Pre-analysis failed: " + str(context.pre_analysis_output.message),
                    None,
                )

        # Successful pre-analysis -> move to code planning
        return (
            WorkflowState.CODE_PLAN,
            "Pre-analysis completed successfully, moving to code planning",
            {"analysis": context.pre_analysis_output},
        )

    def _from_code_plan(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """Transition from CODE_PLAN state."""
        if context.code_plan_output is None:
            return WorkflowState.FAILED, "Code planning produced no output", None

        # Check if planning was successful
        if hasattr(context.code_plan_output, "status"):
            if context.code_plan_output.status == "failed":
                if context.retry_count < context.max_retries:
                    context.retry_count += 1
                    return (
                        WorkflowState.CODE_PLAN,
                        f"Code planning failed, retrying (attempt {context.retry_count}/{context.max_retries})",
                        {"feedback": str(context.code_plan_output.message)},
                    )
                else:
                    return (
                        WorkflowState.FAILED,
                        "Code planning failed after max retries",
                        None,
                    )

        # Initialize checklist-based implementation
        context.retry_count = 0  # Reset retry count
        context.current_checklist_step = 0  # Start from first step
        context.completed_checklist_steps = []
        context.checklist_step_retry_count = 0

        # Get checklist from plan (support both object and dict formats)
        checklist = self._get_checklist(context.code_plan_output)

        if not checklist:
            return (
                WorkflowState.FAILED,
                "Code plan does not contain implementation checklist",
                None,
            )

        # Start implementation with first checklist step
        current_step = checklist[0]
        return (
            WorkflowState.CODE_IMPLEMENT,
            f"Code planning completed, starting step 1/{len(checklist)}: {current_step.title}",
            {
                "analysis": context.pre_analysis_output,
                "plan": context.code_plan_output,
                "current_step": current_step,
                "checklist_progress": f"Step 1/{len(checklist)}",
            },
        )

    def _from_code_implement(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """
        Transition from CODE_IMPLEMENT state.

        After each step implementation, always go to CODE_JUDGE for evaluation.
        """
        if context.code_implement_output is None:
            return WorkflowState.FAILED, "Code implementation produced no output", None

        # Check if implementation was successful
        if hasattr(context.code_implement_output, "status"):
            if context.code_implement_output.status == "failed":
                # Implementation failed at system level (not review failure)
                if context.checklist_step_retry_count < context.max_step_retries:
                    context.checklist_step_retry_count += 1

                    checklist = self._get_checklist(context.code_plan_output)
                    current_step = checklist[context.current_checklist_step]

                    return (
                        WorkflowState.CODE_IMPLEMENT,
                        f"Step implementation failed, retrying (attempt {context.checklist_step_retry_count}/{context.max_step_retries})",
                        {
                            "plan": context.code_plan_output,
                            "current_step": current_step,
                            "feedback": str(context.code_implement_output.message),
                            "checklist_progress": f"Step {context.current_checklist_step + 1}/{len(checklist)}",
                        },
                    )
                else:
                    return (
                        WorkflowState.FAILED,
                        f"Step {context.current_checklist_step + 1} failed after max retries",
                        None,
                    )

        # Step implementation completed -> move to code judge for evaluation
        checklist = self._get_checklist(context.code_plan_output)
        current_step = checklist[context.current_checklist_step]

        return (
            WorkflowState.CODE_JUDGE,
            f"Step {context.current_checklist_step + 1}/{len(checklist)} implemented, moving to review",
            {
                "analysis": context.pre_analysis_output,
                "plan": context.code_plan_output,
                "implementation": context.code_implement_output,
                "current_step": current_step,
                "checklist_progress": f"Step {context.current_checklist_step + 1}/{len(checklist)}",
            },
        )

    def _from_code_judge(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """
        Transition from CODE_JUDGE state.

        Judge evaluates current step. If accepted, move to next step or complete.
        If rejected and retries available, return to CODE_IMPLEMENT with feedback.
        """
        if context.code_judge_output is None:
            return WorkflowState.FAILED, "Code review produced no output", None

        judge_output = context.code_judge_output
        checklist = self._get_checklist(context.code_plan_output)
        current_step_index = context.current_checklist_step
        current_step = checklist[current_step_index]

        # Check if current step is accepted
        step_accepted = False
        # Handle both dict and object formats (for cached vs fresh outputs)
        if isinstance(judge_output, dict):
            step_accepted = judge_output.get("is_consistent", False)
            if not step_accepted and "overall_score" in judge_output:
                step_accepted = judge_output.get("overall_score", 0) >= 0.7
        else:
            if hasattr(judge_output, "is_consistent"):
                step_accepted = judge_output.is_consistent
            elif hasattr(judge_output, "overall_score"):
                # Fallback to score-based evaluation
                step_accepted = judge_output.overall_score >= 0.7

        if step_accepted:
            # Current step passed review
            context.completed_checklist_steps.append(current_step.step_id)
            context.checklist_step_retry_count = 0  # Reset retry count for this step

            # Check if all steps are completed
            if current_step_index + 1 >= len(checklist):
                # All steps completed -> proceed to execution
                context.retry_count = 0
                return (
                    WorkflowState.EXPERIMENT_EXECUTE,
                    f"All {len(checklist)} implementation steps completed and reviewed, proceeding to execution",
                    {"code": context.code_implement_output},
                )

            # Move to next step
            else:
                context.current_checklist_step += 1
                next_step = checklist[context.current_checklist_step]
                return (
                    WorkflowState.CODE_IMPLEMENT,
                    f"Step {current_step_index + 1} approved, moving to step {context.current_checklist_step + 1}/{len(checklist)}: {next_step.title}",
                    {
                        "plan": context.code_plan_output,
                        "current_step": next_step,
                        "completed_steps": context.completed_checklist_steps,
                        "checklist_progress": f"Step {context.current_checklist_step + 1}/{len(checklist)}",
                    },
                )

        else:
            # Current step rejected
            if context.checklist_step_retry_count < context.max_step_retries:
                # Retry current step with feedback
                context.checklist_step_retry_count += 1

                feedback = ""
                # Handle both dict and object formats
                if isinstance(judge_output, dict):
                    feedback = judge_output.get("next_steps") or judge_output.get(
                        "overall_assessment", ""
                    )
                else:
                    if hasattr(judge_output, "next_steps"):
                        feedback = judge_output.next_steps
                    elif hasattr(judge_output, "overall_assessment"):
                        feedback = judge_output.overall_assessment

                return (
                    WorkflowState.CODE_IMPLEMENT,
                    f"Step {current_step_index + 1} needs improvement (attempt {context.checklist_step_retry_count}/{context.max_step_retries})",
                    {
                        "plan": context.code_plan_output,
                        "current_step": current_step,
                        "feedback": feedback,
                        "judge_output": judge_output,
                        "checklist_progress": f"Step {context.current_checklist_step + 1}/{len(checklist)}",
                    },
                )
            else:
                # Max retries exceeded for this step
                return (
                    WorkflowState.FAILED,
                    f"Step {current_step_index + 1} failed review after {context.max_step_retries} attempts",
                    None,
                )

    def _from_experiment_execute(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """Transition from EXPERIMENT_EXECUTE state."""
        if context.experiment_execute_output is None:
            return WorkflowState.FAILED, "Experiment execution produced no output", None

        execute_output = context.experiment_execute_output

        # Check execution status
        if hasattr(execute_output, "status"):
            if execute_output.status == "success":
                context.retry_count = 0
                return (
                    WorkflowState.EXPERIMENT_ANALYSIS,
                    "Experiment executed successfully, moving to analysis",
                    {
                        "execution_results": execute_output,
                        "expected_results": context.pre_analysis_output,
                    },
                )
            elif execute_output.status == "error":
                if context.retry_count < context.max_retries:
                    context.retry_count += 1
                    return (
                        WorkflowState.CODE_IMPLEMENT,
                        f"Experiment execution failed, fixing code (attempt {context.retry_count}/{context.max_retries})",
                        {
                            "plan": context.code_plan_output,
                            "error": getattr(execute_output, "error_message", ""),
                        },
                    )
                else:
                    return (
                        WorkflowState.FAILED,
                        "Experiment execution failed after max retries",
                        None,
                    )

        # Default: proceed to analysis
        context.retry_count = 0
        return (
            WorkflowState.EXPERIMENT_ANALYSIS,
            "Experiment execution completed, moving to analysis",
            {
                "execution_results": execute_output,
                "expected_results": context.pre_analysis_output,
            },
        )

    def _from_experiment_analysis(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """Transition from EXPERIMENT_ANALYSIS state."""
        if context.experiment_analysis_output is None:
            return WorkflowState.FAILED, "Experiment analysis produced no output", None

        analysis_output = context.experiment_analysis_output

        # Check if analysis recommends iteration
        if hasattr(analysis_output, "needs_iteration"):
            if (
                analysis_output.needs_iteration
                and context.iteration_count < context.max_iterations
            ):
                context.iteration_count += 1
                context.retry_count = 0

                # Determine where to iterate back to based on feedback
                if hasattr(analysis_output, "iteration_target"):
                    target = analysis_output.iteration_target
                    if target == "plan":
                        return (
                            WorkflowState.CODE_PLAN,
                            f"Analysis suggests revising plan (iteration {context.iteration_count})",
                            {
                                "analysis": context.pre_analysis_output,
                                "feedback": getattr(analysis_output, "feedback", ""),
                            },
                        )
                    elif target == "implementation":
                        return (
                            WorkflowState.CODE_IMPLEMENT,
                            f"Analysis suggests improving implementation (iteration {context.iteration_count})",
                            {
                                "plan": context.code_plan_output,
                                "feedback": getattr(analysis_output, "feedback", ""),
                            },
                        )

        # No iteration needed or max iterations reached -> completed
        return (
            WorkflowState.COMPLETED,
            "Experiment analysis completed successfully, workflow finished",
            {"final_analysis": analysis_output},
        )

    def transition(self, context: WorkflowContext) -> StateTransition:
        """
        Execute a state transition.

        Args:
            context: Current workflow context

        Returns:
            StateTransition object with details
        """
        from_state = context.current_state
        to_state, reason, data = self.get_next_state(context)

        # Update context
        context.current_state = to_state

        # Create transition record
        transition = StateTransition(
            from_state=from_state, to_state=to_state, reason=reason, data=data
        )

        # Add to history
        context.state_history.append(transition)

        return transition

    def is_terminal_state(self, state: WorkflowState) -> bool:
        """Check if a state is terminal (workflow should stop)."""
        return state in [WorkflowState.COMPLETED, WorkflowState.FAILED]

    def get_required_agent(self, state: WorkflowState) -> Optional[str]:
        """
        Get the agent required for a given state.

        Args:
            state: Workflow state

        Returns:
            Agent name or None if terminal state
        """
        agent_mapping = {
            WorkflowState.PRE_ANALYSIS: "pre_analysis",
            WorkflowState.CODE_PLAN: "code_plan",
            WorkflowState.CODE_IMPLEMENT: "code_implement",
            WorkflowState.CODE_JUDGE: "code_judge",
            WorkflowState.EXPERIMENT_EXECUTE: "experiment_execute",
            WorkflowState.EXPERIMENT_ANALYSIS: "experiment_analysis",
        }
        return agent_mapping.get(state)
