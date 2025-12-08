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
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from pydantic import Field

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel

if TYPE_CHECKING:
    from src.agents.experiment_agent.sub_agents.experiment_master.issue_tracker import IssueTracker


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


class StateTransition(BaseDictModel):
    """Represents a state transition with reason."""

    from_state: WorkflowState
    to_state: WorkflowState
    reason: str
    data: Optional[Dict[str, Any]] = None


class WorkflowContext(BaseDictModel):
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
    completed_checklist_steps: List[int] = Field(default_factory=list)  # List of completed step IDs
    checklist_step_retry_count: int = 0  # Retry count for current step
    max_step_retries: int = 999  # Maximum retries per step (effectively unlimited)

    # Global execution tracking (for cache management)
    execution_step_counter: int = (
        0  # Global counter for all agent executions (increments with each agent call)
    )

    # Error tracking
    last_error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 10

    # Pending feedback for code_plan re-planning scenarios
    # Type can be: "initial", "error_feedback", "analysis_feedback"
    pending_feedback_type: Optional[str] = None
    # Data contains the feedback content to pass to code_plan agent
    pending_feedback_data: Optional[Dict[str, Any]] = None

    # Issue tracking across judge iterations
    # Stored as dict for JSON serialization, converted to IssueTracker at runtime
    issue_tracker_data: Optional[Dict[str, Any]] = None
    
    # History
    state_history: List[StateTransition] = Field(default_factory=list)
    
    def get_issue_tracker(self) -> "IssueTracker":
        """Get or create IssueTracker instance from stored data."""
        from src.agents.experiment_agent.sub_agents.experiment_master.issue_tracker import IssueTracker
        
        if self.issue_tracker_data is None:
            return IssueTracker()
        return IssueTracker.from_dict(self.issue_tracker_data)
    
    def save_issue_tracker(self, tracker: "IssueTracker") -> None:
        """Save IssueTracker state to context for serialization."""
        self.issue_tracker_data = tracker.to_dict()


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
        if code_plan_output:
            # Unified access
            checklist = code_plan_output.get("implementation_checklist", [])

        # Convert dict items to objects for consistent attribute access
        if checklist and isinstance(checklist[0], dict):
            # Create simple objects from dicts
            class ChecklistStep:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)

            checklist = [ChecklistStep(item) for item in checklist]

        return checklist

    def _is_judge_accepted(self, judge_output: Any) -> bool:
        """
        Determine if judge output indicates acceptance.
        Uses is_consistent field.
        """
        return judge_output.get("is_consistent", False)
        
    def _extract_judge_feedback(self, judge_output: Any) -> str:
        """Extract feedback from judge output for retry."""
        # Extract issues as feedback
        issues = judge_output.get("issues", []) if hasattr(judge_output, "get") else getattr(judge_output, "issues", [])
        if issues:
            feedback_parts = []
            for issue in issues[:5]:  # Limit to first 5 issues
                desc = issue.get("description", "") if isinstance(issue, dict) else getattr(issue, "description", "")
                sugg = issue.get("suggestion", "") if isinstance(issue, dict) else getattr(issue, "suggestion", "")
                if desc:
                    feedback_parts.append(f"- {desc}" + (f" (Fix: {sugg})" if sugg else ""))
            return "\n".join(feedback_parts)
        return ""

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

        # Pre-analysis completed -> move to code planning
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

        # Initialize checklist-based implementation
        context.retry_count = 0
        context.current_checklist_step = 0
        context.completed_checklist_steps = []
        context.checklist_step_retry_count = 0
        
        # Clear pending feedback after successful code_plan completion
        context.pending_feedback_type = None
        context.pending_feedback_data = None

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
            None,
        )

    def _from_code_implement(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """
        Transition from CODE_IMPLEMENT state.
        After implementation, always go to CODE_JUDGE for evaluation.
        """
        if context.code_implement_output is None:
            return WorkflowState.FAILED, "Code implementation produced no output", None

        # Implementation completed -> move to code judge for evaluation
        checklist = self._get_checklist(context.code_plan_output)
        
        # Check index bounds
        if context.current_checklist_step >= len(checklist):
             return WorkflowState.FAILED, "Current step index out of bounds", None

        return (
            WorkflowState.CODE_JUDGE,
            f"Step {context.current_checklist_step + 1}/{len(checklist)} implemented, moving to review",
            None,
        )

    def _from_code_judge(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """
        Transition from CODE_JUDGE state.
        If accepted -> next step or execution. If rejected -> retry with feedback.
        """
        if context.code_judge_output is None:
            return WorkflowState.FAILED, "Code review produced no output", None

        judge_output = context.code_judge_output
        checklist = self._get_checklist(context.code_plan_output)
        
        # Check index bounds
        if context.current_checklist_step >= len(checklist):
             return WorkflowState.FAILED, "Current step index out of bounds", None
             
        current_step_index = context.current_checklist_step
        current_step = checklist[current_step_index]

        # Determine acceptance based on is_consistent field
        step_accepted = self._is_judge_accepted(judge_output)

        if step_accepted:
            # Current step passed review
            context.completed_checklist_steps.append(current_step.step_id)
            context.checklist_step_retry_count = 0

            if current_step_index + 1 >= len(checklist):
                # All steps completed -> proceed to execution
                context.retry_count = 0
                return (
                    WorkflowState.EXPERIMENT_EXECUTE,
                    f"All {len(checklist)} steps completed, proceeding to execution",
                    None,
                )
            else:
                # Move to next step
                context.current_checklist_step += 1
                return (
                    WorkflowState.CODE_IMPLEMENT,
                    f"Step {current_step_index + 1} approved, moving to step {context.current_checklist_step + 1}/{len(checklist)}",
                    None,
                )
        else:
            # Current step rejected - retry with feedback
            if context.checklist_step_retry_count < context.max_step_retries:
                context.checklist_step_retry_count += 1
                
                # Save pending feedback type
                context.pending_feedback_type = "judge_rejection"
                context.pending_feedback_data = None

                return (
                    WorkflowState.CODE_IMPLEMENT,
                    f"Step {current_step_index + 1} needs improvement (attempt {context.checklist_step_retry_count}/{context.max_step_retries})",
                    None,
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

        # Get execution status (handle both dict and object formats)
        if isinstance(execute_output, dict):
            execution_status = execute_output.get("execution_status", "unknown")
        else:
            execution_status = getattr(execute_output, "execution_status", "unknown")

        # Check execution status
        if execution_status == "success":
            context.retry_count = 0
            return (
                WorkflowState.EXPERIMENT_ANALYSIS,
                "Experiment executed successfully, moving to analysis",
                None,
            )
        elif execution_status == "skipped":
            # Execution was skipped (e.g., no entry script found)
            # Still proceed to analysis to evaluate the implementation
            context.retry_count = 0
            return (
                WorkflowState.EXPERIMENT_ANALYSIS,
                "Experiment execution skipped (library implementation), moving to analysis",
                None,
            )
        elif execution_status == "error":
            if context.retry_count < context.max_retries:
                context.retry_count += 1
                context.iteration_count += 1
                
                # Reset checklist progress for re-planning
                context.current_checklist_step = 0
                context.completed_checklist_steps = []
                context.checklist_step_retry_count = 0
                
                # Set pending feedback type
                context.pending_feedback_type = "error_feedback"
                context.pending_feedback_data = None
                
                return (
                    WorkflowState.CODE_PLAN,
                    f"Experiment execution failed, sending to code plan agent for re-planning (attempt {context.retry_count}/{context.max_retries})",
                    None,
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
            None,
        )

    def _from_experiment_analysis(
        self, context: WorkflowContext
    ) -> Tuple[WorkflowState, str, Optional[Dict]]:
        """Transition from EXPERIMENT_ANALYSIS state."""
        if context.experiment_analysis_output is None:
            return WorkflowState.FAILED, "Experiment analysis produced no output", None

        analysis_output = context.experiment_analysis_output

        # Check if analysis recommends iteration (unified .get() access)
        needs_iteration = analysis_output.get("needs_iteration", False) if hasattr(analysis_output, "get") else getattr(analysis_output, "needs_iteration", False)
        if needs_iteration and context.iteration_count < context.max_iterations:
            context.iteration_count += 1
            context.retry_count = 0

            # Determine where to iterate back to based on feedback
            target = analysis_output.get("iteration_target", None) if hasattr(analysis_output, "get") else getattr(analysis_output, "iteration_target", None)
            if target == "plan":
                # Set pending feedback type
                context.pending_feedback_type = "analysis_feedback"
                context.pending_feedback_data = None
                
                # Reset checklist progress for re-planning
                context.current_checklist_step = 0
                context.completed_checklist_steps = []
                context.checklist_step_retry_count = 0
                
                return (
                    WorkflowState.CODE_PLAN,
                    f"Analysis suggests revising plan (iteration {context.iteration_count})",
                    None,
                )
            elif target == "implementation":
                return (
                    WorkflowState.CODE_IMPLEMENT,
                    f"Analysis suggests improving implementation (iteration {context.iteration_count})",
                    None,
                )

        # No iteration needed or max iterations reached -> completed
        return (
            WorkflowState.COMPLETED,
            "Experiment analysis completed successfully, workflow finished",
            None,
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
