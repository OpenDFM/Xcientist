"""
Code Judge Agent - Reviews code implementation for consistency with plan and analysis.

This agent evaluates whether the implemented code matches the specifications
from the code plan and pre-analysis, identifying issues and providing feedback.
"""

import os
from typing import Optional, Any

from src.agents.experiment_agent.config import (
    OUTPUT_UNIFIER_MODEL,
)
from agents import Agent, Runner
from src.agents.experiment_agent.sub_agents.code_judge.output_schemas import (
    CodeJudgeOutput,
)
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.utils.common_utils import (
    read_file_smart,
    extract_core_plan_context,
    extract_analysis_summary,
)

from src.agents.experiment_agent.utils.print_utils import *


def create_judge_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: Optional[list] = None
) -> Agent:
    """
    Create the code judge agent for reviewing implementation.
    """

    instructions = f"""You are the Lead QA Engineer. Your job is to certify that the Current Implementation Step is complete, correct, and bug-free.

### SCOPE
- **Target**: Files listed in `files_to_create` / `files_to_modify` for the Current Step.
- **Context**: Project Root is `{working_dir}/project`.

### WORKFLOW
You must strictly follow this verification pipeline:

1. **INSPECTION**
   - Use `read_file` to examine the code implemented in this step.
   - Verify: Are imports correct? (No `project.` prefix).
   - Verify: Are all required functions/classes defined?

2. **VERIFICATION (Dynamic Testing)**
   - You are the ONLY agent authorized to write tests.
   - Generate **MINIMAL, LIGHTWEIGHT** unit tests (2-5 tests) to verify the new code works.
   - **Action**:
     1. Write test file: `write_file("{working_dir}/project/tests/test_step_X.py", code)`
     2. Run test: `run_pytest_local("tests/", working_dir="{working_dir}/project")`
   - **Constraint**: Tests must be fast (< 30s). Mock heavy resources.
   - **Constraint**: Include `sys.path.append(os.getcwd())` in your test files to ensure `models`, `utils`, etc. are importable.
   - **CRITICAL**: Keep test code SHORT to avoid output truncation.

3. **ADJUDICATION**
   - Based on code inspection AND test results, determine `is_consistent`.
   - If tests fail -> `is_consistent: False`.
   - If logic is flawed -> `is_consistent: False`.
   - If strictly consistent with Plan and passes tests -> `is_consistent: True`.

### OUTPUT FORMAT
Provide your evaluation in clear, detailed text. Your response MUST include the following sections:

1. **Overall Assessment**: High-level summary of the code quality and consistency.
2. **Scores**:
   - Plan Consistency Score (0.0 - 1.0)
   - Analysis Consistency Score (0.0 - 1.0)
3. **Consistency Verdict**: Whether the code is consistent with the plan (True/False).
4. **Issues Identified**: List any issues found (Logic Errors, Missing Implementation, Quality Issues, etc.). For each issue, specify the file, severity, and a brief description.
5. **Strengths**: List aspects that are well implemented.
6. **Missing/Extra Components**: Components specified in the plan but missing, or extra components not in the plan.
7. **Unit Tests**: List the test files created and a brief description of what they verify.
8. **Recommendations**: Priority fixes and next steps.
"""

    agent = Agent(
        name="Code Judge Agent",
        instructions=instructions,
        tools=tools or [],
        # output_type=CodeJudgeOutput,  # Removed for duplex mode
        model=model,
    )

    return agent


def create_output_unifier_agent(model: str = "gpt-4o") -> Agent:
    """
    Create an agent to unify code judge output into structured format.
    """
    return Agent(
        name="Judge Output Unifier",
        instructions="""You are an expert data structuring assistant.
Your task is to convert the raw text evaluation from the Code Judge into a structured JSON format.

Input text will contain:
- Overall assessment
- Consistency scores
- Issues found
- Strengths and weaknesses
- Unit tests information

You must extract this information and map it to the `CodeJudgeOutput` structure.
Ensure all fields are correctly populated based on the text.
If specific details are missing, use reasonable defaults or empty lists.
""",
        output_type=CodeJudgeOutput,
        model=OUTPUT_UNIFIER_MODEL,
    )


class CodeJudgeAgent:
    """
    Main code judge agent that reviews implementation consistency.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[list] = None,
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
            from src.agents.experiment_agent.sub_agents.code_judge import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize judge agent
        self.judge_agent = create_judge_agent(
            model=model, working_dir=working_dir, tools=self.tools
        )

        # Initialize output unifier
        self.output_unifier = create_output_unifier_agent(model=model)

        # Expose judge agent as main agent for handoff compatibility
        self.agent = self.judge_agent

    async def process(self, context: Any, **kwargs) -> CodeJudgeOutput:
        """
        Process the current step using context data.
        """
        data = kwargs
        current_step = data.get("current_step", None)
        checklist_progress = data.get("checklist_progress", "")

        target_files_context = ""
        if current_step:
            files_to_check = []
            if current_step.files_to_create:
                files_to_check.extend(current_step.files_to_create)
            if current_step.files_to_modify:
                files_to_check.extend(current_step.files_to_modify)

            files_to_check = list(set(files_to_check))

            # Filter out directories - only keep actual files
            actual_files = []
            for f in files_to_check:
                if not f:
                    continue
                if f.endswith("/"):
                    continue
                full_path = os.path.join(self.working_dir, "project", f)
                if os.path.exists(full_path) and os.path.isdir(full_path):
                    continue
                actual_files.append(f)

            if actual_files:
                file_contents = []
                for f in actual_files:
                    # Ensure f is a clean relative path
                    clean_f = f.strip().lstrip("/").lstrip("\\")

                    # 1. Try standard location: working_dir/project/path
                    project_path = os.path.join(self.working_dir, "project", clean_f)

                    # 2. Fallback location: working_dir/path (if someone messed up structure)
                    root_path = os.path.join(self.working_dir, clean_f)

                    content = ""
                    final_path = ""

                    if os.path.exists(project_path) and os.path.isfile(project_path):
                        final_path = project_path
                    elif os.path.exists(root_path) and os.path.isfile(root_path):
                        final_path = root_path
                        content = f"[WARNING: File found at {root_path}, NOT in project/ subdirectory. Please move it.]\n"
                    else:
                        content = (
                            f"[File not found: {clean_f}]\n[Checked: {project_path}]"
                        )

                    if final_path:
                        try:
                            with open(
                                final_path, "r", encoding="utf-8", errors="replace"
                            ) as f_obj:
                                raw_lines = f_obj.readlines()
                                if len(raw_lines) > 300:
                                    content += (
                                        "".join(raw_lines[:50])
                                        + f"\n\n... ({len(raw_lines)-50} more lines) ..."
                                    )
                                else:
                                    content += "".join(raw_lines)
                        except Exception as e:
                            content = f"[Error reading file: {str(e)}]"

                    if not content.startswith("[Directory:") and not content.startswith(
                        "[Binary file:"
                    ):
                        file_contents.append(
                            f"--- FILE: {f} ---\n{content}\n----------------"
                        )

                target_files_context = "\n=== TARGET CODE CONTEXT ===\n" + "\n".join(
                    file_contents
                )

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

        core_plan_context = extract_core_plan_context(context.code_plan_output)

        input_data = f"""Review the following code implementation (STEP-BY-STEP MODE):
{step_info}

{target_files_context}

=== GLOBAL DESIGN CONTEXT ===
{core_plan_context}

=== ANALYSIS SUMMARY ===
{extract_analysis_summary(context.pre_analysis_output)}

=== CODEBASE PATH ===
Project directory: {self.working_dir}/project

INSTRUCTIONS:
1. Check files in the Project Directory using available tools.
2. Execute the VERIFICATION pipeline (write test -> run test -> judge).
"""

        return await self.judge(input_data)

    async def judge(
        self,
        input_data: str,
    ) -> CodeJudgeOutput:
        """
        Evaluate code implementation for consistency with current step.
        """
        print_section("CODE REVIEW WORKFLOW", "=")
        print_info(f"Input length: {len(input_data)} characters")

        judge_input = f"""
{input_data}

### EXECUTION REQUIRED
You must strictly follow the WORKFLOW defined in your system instructions.
1. INSPECT: `read_file` target files.
2. TEST: `write_file` (tests/test_step_x.py) -> `run_pytest_local`.
3. JUDGE: Return `CodeJudgeOutput` based on results.

Current Working Directory: `{self.working_dir}`
Project Root: `{self.working_dir}/project`
"""

        print_subsection("Evaluating Current Step Implementation")
        print_info("Reviewing code and running verification tests...")

        # Use streamed version for real-time output
        result_stream = Runner.run_streamed(
            self.judge_agent, judge_input, hooks=self.hooks, max_turns=100
        )
        final_text = ""
        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                        final_text += delta.content
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
                        final_text += delta.text

        # In duplex mode, the first agent returns text, not the structured object
        # We need to ensure we capture the full text output if not accumulated
        if hasattr(result_stream, "final_output") and isinstance(
            result_stream.final_output, str
        ):
            final_text = result_stream.final_output
        elif not final_text and hasattr(result_stream, "chat_history"):
            # Fallback: get last message
            final_text = result_stream.chat_history[-1].content

        print_success("Judge analysis completed. Formatting output...")

        # Step 2: Unify output
        unifier_input = f"""
Please convert the following code judge evaluation into the structured `CodeJudgeOutput` format.

=== JUDGE EVALUATION ===
{final_text}
"""
        unifier_stream = Runner.run_streamed(
            self.output_unifier, unifier_input, hooks=None
        )

        # We can just await the final result for the unifier
        result = unifier_stream

        # Ensure we wait for completion if needed (though run_streamed usually completes)
        # In the current Runner implementation, we iterate to drive it or access final_output after iteration
        async for _ in unifier_stream.stream_events():
            pass

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

        print_result_box(
            "Overall Assessment",
            evaluation.overall_assessment,
            max_length=1000,
        )

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

        if evaluation.strengths:
            print_subsection("Strengths")
            for strength in evaluation.strengths:
                print_success(strength, indent=1)

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
        import asyncio

        return asyncio.run(self.judge(input_data))

    def _format_file_structure(self, file_structure: dict) -> str:
        lines = []
        for path, description in file_structure.items():
            lines.append(f"  {path}:")
            lines.append(f"    {description}")
        return "\n".join(lines)

    def _format_roadmap(self, roadmap: dict) -> str:
        lines = []
        for phase, details in roadmap.items():
            lines.append(f"\n{phase}:")
            lines.append(f"  {details}")
        return "\n".join(lines)


def create_code_judge_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[list] = None,
    verbose: bool = False,
) -> CodeJudgeAgent:
    """
    Factory function to create a code judge agent.
    """
    return CodeJudgeAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        print("Example 1: Simple creation")
        print("=" * 60)

        agent = create_code_judge_agent(model="gpt-4o")
        print("✓ Code judge agent created with all tools automatically loaded\n")

    asyncio.run(main())
