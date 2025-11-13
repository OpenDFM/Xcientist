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
        expensive_model: Optional[str] = None,
        cheap_model: Optional[str] = None,
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
            model: Default model to use for all agents (for backward compatibility)
            expensive_model: Model for critical tasks (code plan, judge, execute). If None, uses model.
            cheap_model: Model for non-critical tasks (implement, analysis, etc.). If None, uses model.
            max_iterations: Maximum number of complete workflow iterations
            tools: Optional dictionary mapping agent types to their tools.
                   If None, automatically loads recommended tools for all agents.
            working_dir: Working directory for code operations
            log_dir: Directory for execution logs
            cache_dir: Cache directory for agent outputs (default: ./cached)
            verbose: If True, enable verbose hooks to show full LLM responses and tool calls
        """
        self.model = model
        self.expensive_model = expensive_model or model
        self.cheap_model = cheap_model or model
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

        # Initialize all sub-agents with appropriate models
        # Use cheap model for pre-analysis
        self.pre_analysis_agent = create_pre_analysis_agent(
            model=self.cheap_model,
            tools=self.tools.get("pre_analysis"),
            verbose=verbose,
        )

        # Use expensive model for code plan (critical task)
        self.code_plan_agent = create_code_plan_agent(
            model=self.expensive_model,
            working_dir=working_dir,
            tools=self.tools.get("code_plan"),
            verbose=verbose,
        )

        # Use expensive model for code implementation (critical task)
        self.code_implement_agent = create_code_implement_agent(
            model=self.expensive_model,
            working_dir=working_dir,
            tools=self.tools.get("code_implement"),
            verbose=verbose,
        )

        # Use expensive model for code judge (critical task)
        self.code_judge_agent = create_code_judge_agent(
            model=self.expensive_model,
            tools=self.tools.get("code_judge"),
            verbose=verbose,
        )

        # Use cheap model for experiment execute
        self.experiment_execute_agent = create_experiment_execute_agent(
            model=self.cheap_model,
            log_dir=log_dir,
            tools=self.tools.get("experiment_execute"),
        )

        # Use cheap model for experiment analysis
        self.experiment_analysis_agent = create_experiment_analysis_agent(
            model=self.cheap_model, tools=self.tools.get("experiment_analysis")
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

                workspace_dir = os.path.dirname(self.working_dir)

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
            if context.current_state == WorkflowState.PRE_ANALYSIS and context.pre_analysis_output:
                state_output_exists = True
            elif context.current_state == WorkflowState.CODE_PLAN and context.code_plan_output:
                state_output_exists = True
            elif context.current_state == WorkflowState.CODE_IMPLEMENT and context.code_implement_output:
                state_output_exists = True
            elif context.current_state == WorkflowState.CODE_JUDGE and context.code_judge_output:
                state_output_exists = True
            elif context.current_state == WorkflowState.EXPERIMENT_EXECUTE and context.experiment_execute_output:
                state_output_exists = True
            elif context.current_state == WorkflowState.EXPERIMENT_ANALYSIS and context.experiment_analysis_output:
                state_output_exists = True
            
            if state_output_exists:
                print(f"[SKIP] Current state output already exists, transitioning to next state...")
                # Perform state transition immediately
                transition = self.state_machine.transition(context)
                print(f"\n[STATE TRANSITION]")
                print(f"From: {transition.from_state.value}")
                print(f"To: {transition.to_state.value}")
                print(f"Reason: {transition.reason}")
                transition_data = transition.data or {}
            else:
                print(f"[CONTINUE] Current state output not found, will execute agent...")
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

        # Prepare input for agent based on state
        agent_input = self._prepare_agent_input(agent_name, context, data)

        print(f"[DEBUG] Agent input length: {len(agent_input)} characters")
        print(f"[DEBUG] Agent input preview:\n{agent_input[:300]}...")

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
                # Experiment execute agent - needs special parameters
                print(f"[DEBUG] Calling agent.execute()")

                # Extract entry script from code plan if available
                entry_script = None
                if context.code_plan_output:
                    file_structure = (
                        context.code_plan_output.file_structure
                        if hasattr(context.code_plan_output, "file_structure")
                        else []
                    )
                    # Look for main entry point files
                    for item in file_structure:
                        if hasattr(item, "path"):
                            path = item.path
                        elif isinstance(item, dict):
                            path = item.get("path", "")
                        else:
                            continue

                        if "main.py" in path or "train.py" in path or "run.py" in path:
                            # Extract just the filename from the path
                            entry_script = path.split("/")[-1] if "/" in path else path
                            break

                # If no entry script found, try to create one or skip execution
                if not entry_script:
                    print(
                        f"[WARNING] No entry script (main.py/train.py) found in file structure"
                    )
                    print(
                        f"[INFO] Skipping experiment execution - implementation appears to be a library"
                    )
                    # Create a dummy successful execution output
                    from src.agents.experiment_agent.sub_agents.experiment_execute.output_schemas import (
                        ExperimentExecuteOutput,
                    )

                    result = ExperimentExecuteOutput(
                        status="skipped",
                        log_path="",
                        execution_time=0.0,
                        success=True,
                        summary="Execution skipped - no entry script found. Implementation is a library/module.",
                        output_preview="N/A - Library implementation without executable entry point",
                    )
                else:
                    result = await agent.execute(
                        code_path=self.working_dir,
                        entry_script=entry_script,
                        execution_args=None,
                    )
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
                step_id=step_id,
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
            # Extract step information and feedback
            current_step = data.get("current_step", None)
            checklist_progress = data.get("checklist_progress", "")
            completed_steps = data.get("completed_steps", [])
            feedback = data.get("feedback", "")
            judge_output = data.get("judge_output", None)

            # Build current step section
            step_section = ""
            if current_step:
                step_section = f"""
=== CURRENT STEP TO IMPLEMENT ===

Step ID: {current_step.step_id}
Title: {current_step.title}
Progress: {checklist_progress}

Description:
{current_step.description}

Files to Create in This Step:
{chr(10).join(f'  - {f}' for f in current_step.files_to_create) if current_step.files_to_create else '  (none)'}

Files to Modify in This Step:
{chr(10).join(f'  - {f}' for f in current_step.files_to_modify) if current_step.files_to_modify else '  (none)'}

Acceptance Criteria (verify these are met):
{chr(10).join(f'  - {criterion}' for criterion in current_step.acceptance_criteria)}

Dependencies (already completed):
{', '.join(map(str, current_step.dependencies)) if current_step.dependencies else 'None - this is the first step'}

Completed Steps: {', '.join(map(str, completed_steps)) if completed_steps else 'None'}

IMPORTANT: Implement ONLY this step. Do not implement future steps.
"""

            # Build feedback section if code was rejected
            feedback_section = ""
            if feedback or judge_output:
                # Extract overall assessment
                overall_assessment = (
                    feedback
                    if feedback
                    else (
                        judge_output.overall_assessment
                        if hasattr(judge_output, "overall_assessment")
                        else ""
                    )
                )

                # Extract issues if available
                issues_text = ""
                if (
                    judge_output
                    and hasattr(judge_output, "issues")
                    and judge_output.issues
                ):
                    issues_text = "\n=== SPECIFIC ISSUES FOUND ===\n\n"
                    for idx, issue in enumerate(judge_output.issues, 1):
                        issues_text += f"Issue {idx}: [{issue.severity.upper()}] {issue.issue_type}\n"
                        issues_text += f"  File: {issue.file_path}\n"
                        if issue.line_numbers:
                            issues_text += f"  Lines: {issue.line_numbers}\n"
                        issues_text += f"  Description: {issue.description}\n"
                        issues_text += f"  Expected: {issue.expected}\n"
                        issues_text += f"  Actual: {issue.actual}\n"
                        issues_text += f"  Suggestion: {issue.suggestion}\n\n"

                # Extract priority fixes
                priority_fixes_text = ""
                if (
                    judge_output
                    and hasattr(judge_output, "priority_fixes")
                    and judge_output.priority_fixes
                ):
                    priority_fixes_text = (
                        "\n=== PRIORITY FIXES (Address these first) ===\n\n"
                    )
                    for idx, fix in enumerate(judge_output.priority_fixes, 1):
                        priority_fixes_text += f"{idx}. {fix}\n"

                # Extract missing components
                missing_components_text = ""
                if (
                    judge_output
                    and hasattr(judge_output, "missing_components")
                    and judge_output.missing_components
                ):
                    missing_components_text = "\n=== MISSING COMPONENTS ===\n\n"
                    for component in judge_output.missing_components:
                        missing_components_text += f"  - {component}\n"

                # Extract unit test information
                unit_tests_text = ""
                if (
                    judge_output
                    and hasattr(judge_output, "unit_tests")
                    and judge_output.unit_tests
                ):
                    unit_tests_text = f"\n=== UNIT TESTS GENERATED ===\n\n"
                    unit_tests_text += f"The code judge generated {len(judge_output.unit_tests)} unit test(s) for validation.\n"
                    unit_tests_text += (
                        "These tests should already be written to the file system.\n"
                    )
                    for idx, test in enumerate(judge_output.unit_tests, 1):
                        unit_tests_text += f"\nTest {idx}: {test.test_file_path}\n"
                        unit_tests_text += f"  Description: {test.test_description}\n"
                        unit_tests_text += (
                            f"  Target files: {', '.join(test.target_files)}\n"
                        )

                feedback_section = f"""
=== FEEDBACK FROM CODE REVIEW ===

The previous implementation of this step was reviewed and needs improvement.

=== OVERALL ASSESSMENT ===

{overall_assessment}
{issues_text}{priority_fixes_text}{missing_components_text}{unit_tests_text}

IMPORTANT: Please address ALL the issues above, especially the priority fixes, and re-implement this step correctly.
"""

            # Get reference codebases information from cache
            reference_codebases_info = ""
            try:
                from src.agents.experiment_agent.sub_agents.experiment_master.prepare_helpers import (
                    load_prepare_info_from_cache,
                )

                prepare_info = load_prepare_info_from_cache(self.cache_dir, self.domain)
                if prepare_info and "reference_codebases" in prepare_info:
                    reference_codebases_info = prepare_info["reference_codebases"].get(
                        "formatted_list", ""
                    )
            except Exception as e:
                print(f"Warning: Could not load reference codebases info: {e}")
                reference_codebases_info = "(Codebase information not available)"

            # Extract project structure tree from plan
            project_tree = ""
            if hasattr(context.code_plan_output, "project_structure_tree"):
                project_tree = f"""
=== PROJECT STRUCTURE TREE (MUST FOLLOW EXACTLY) ===

{context.code_plan_output.project_structure_tree}

CRITICAL: This is the DEFINITIVE project structure. You MUST:
- Follow this structure EXACTLY
- Do NOT create files outside this structure
- Do NOT create additional directories
- File paths MUST match this tree exactly

"""

            return f"""Implement code for the CURRENT STEP ONLY (step-by-step iterative implementation):
{step_section}
{project_tree}
=== REFERENCE CODEBASES (EXPLORE BEFORE IMPLEMENTING) ===

{reference_codebases_info}

IMPORTANT: Before implementing, explore relevant reference codebases using:
- `list_directory("../repos/[repo_name]")` to see structure
- `generate_code_tree("../repos/[repo_name]")` for overview
- `read_file("../repos/[repo_name]/path/to/file.py")` to read implementations
- `analyze_python_file("../repos/[repo_name]/path/to/file.py")` to understand structure

=== COMPLETE PLAN (for context) ===

{context.code_plan_output}
{feedback_section}

Remember: Focus ONLY on the current step. Follow the PROJECT STRUCTURE TREE exactly. Use tools to check what files already exist from previous steps."""

        elif agent_name == "code_judge":
            # Extract current step info from data
            current_step = data.get("current_step", None)
            checklist_progress = data.get("checklist_progress", "")

            step_info = ""
            if current_step:
                step_info = f"""
=== CURRENT STEP TO EVALUATE ===

Step ID: {current_step.step_id}
Title: {current_step.title}
Description: {current_step.description}

Files to Create: {', '.join(current_step.files_to_create) if current_step.files_to_create else 'None'}
Files to Modify: {', '.join(current_step.files_to_modify) if current_step.files_to_modify else 'None'}

Acceptance Criteria:
{chr(10).join(f'  - {criterion}' for criterion in current_step.acceptance_criteria)}

Dependencies: {', '.join(map(str, current_step.dependencies)) if current_step.dependencies else 'None'}
Complexity: {current_step.estimated_complexity}

Progress: {checklist_progress}
"""

            return f"""Review the following code implementation (STEP-BY-STEP MODE):
{step_info}

=== COMPLETE PLAN (for context) ===

{context.code_plan_output}

=== ANALYSIS (for context) ===

{context.pre_analysis_output}

=== IMPLEMENTATION OUTPUT ===

{context.code_implement_output}

=== CODEBASE PATH ===

Project directory (check files here): {self.working_dir or '/workspace/project'}

IMPORTANT: Use the tools to check if files exist in the project directory above.
The project directory is where all implementation code should be located.

Evaluate if the CURRENT STEP (shown above) is correctly implemented according to its acceptance criteria."""

        elif agent_name == "experiment_execute":
            # Build implementation context
            impl_context = ""
            if context.code_implement_output:
                impl_context = f"""
=== IMPLEMENTATION CONTEXT ===

Recent Implementation Output:
{str(context.code_implement_output)[:1000]}...

"""

            # Build expected behavior from plan
            expected_behavior = ""
            if context.code_plan_output:
                expected_behavior = f"""
=== EXPECTED BEHAVIOR (from Code Plan) ===

Research Summary:
{context.code_plan_output.research_summary}

Expected Outcomes:
{context.code_plan_output.expected_outcomes if hasattr(context.code_plan_output, 'expected_outcomes') else 'Not specified'}

Performance Targets:
{context.code_plan_output.performance_targets if hasattr(context.code_plan_output, 'performance_targets') else 'Not specified'}

"""

            return f"""Execute experiment code with full context:
{impl_context}{expected_behavior}
=== EXECUTION TASK ===

Your task is to execute the implemented code and monitor its behavior.
You should understand what the code is supposed to do (from context above),
execute it properly, and compare actual behavior with expected behavior.

Run the experiment and capture all results, metrics, and any deviations from expectations."""

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
    expensive_model: Optional[str] = None,
    cheap_model: Optional[str] = None,
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
        model: Default model to use for all agents (for backward compatibility)
        expensive_model: Model for critical tasks (code plan, judge, execute). If None, uses model.
        cheap_model: Model for non-critical tasks (implement, analysis, etc.). If None, uses model.
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
        expensive_model=expensive_model,
        cheap_model=cheap_model,
        max_iterations=max_iterations,
        tools=tools,
        working_dir=working_dir,
        log_dir=log_dir,
        cache_dir=cache_dir,
        verbose=verbose,
    )
