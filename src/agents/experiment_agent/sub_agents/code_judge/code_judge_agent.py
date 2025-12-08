"""
Code Judge Agent - Reviews code implementation for consistency with plan and analysis.

This agent evaluates whether the implemented code matches the specifications
from the code plan and pre-analysis, identifying issues and providing feedback.
"""

import os
from typing import Optional, Any

from agents import Agent, Runner, RunConfig, ModelSettings
from src.agents.experiment_agent.sub_agents.code_judge.output_schemas import (
    CodeJudgeOutput,
)
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.utils.common_utils import (
    read_file_smart,
    extract_core_plan_context,
    extract_analysis_summary,
)
from src.agents.experiment_agent.utils.json_utils import (
    extract_and_parse_json,
    generate_json_schema_instruction,
    JSONParseError,
)

from src.agents.experiment_agent.utils.print_utils import *


# Generate JSON output instruction for CodeJudgeOutput
CODE_JUDGE_JSON_OUTPUT_INSTRUCTION = generate_json_schema_instruction(CodeJudgeOutput)


def create_judge_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: Optional[list] = None
) -> Agent:
    """
    Create the code judge agent for reviewing implementation.
    """

    instructions = f"""You are the QA Engineer certifying code quality. **ZERO TOLERANCE** for issues.

## WORKFLOW: 1️⃣ INSPECT → 2️⃣ TEST → 3️⃣ JUDGE

---

## 1️⃣ INSPECT (Interface Audit)

| Check | Method | Flag If |
|-------|--------|---------|
| Imports | `read_file` target files | Import doesn't resolve to actual export |
| Data contracts | Trace producer→consumer | Output format ≠ expected input format |
| Signatures | Compare definition vs calls | Mismatch in params/types |
| Breaking changes | Check callers of modified APIs | Callers not updated |

🚫 Flag as **CRITICAL** if Implement Agent didn't read related files before coding.

---

## 2️⃣ TEST (Dynamic Verification)

**Execute ALL tests:**
1. Create/modify test files:
   - **For NEW test files**: Use `write_file(file_path="{working_dir}/project/tests/test_step_X.py", contents=code)`
   - **For EXISTING test files**: Prefer `edit_file` to modify specific test cases instead of rewriting the entire file
   - ⚠️ **CRITICAL JSON SYNTAX RULES** (for both tools): 
     - The `contents`/`new_content` argument MUST be a SINGLE LINE string in the JSON payload.
     - Replace ALL actual newlines in the code with `\\n`.
     - Escape ALL double quotes (`"`) as `\\"`.
     - Escape ALL backslashes (`\\`) as `\\\\`.
     - DO NOT include actual line breaks inside the JSON string value.
2. `run_pytest_local("tests/", working_dir="{working_dir}/project")`

**Required Test Types:**

| Test | Purpose |
|------|---------|
| Interface Contract | Producer output matches consumer input |
| Integration | Real data flow between components |
| Smoke | Main entrypoint runs without crash |

**Test Constraints:**
- Include `sys.path.append(os.getcwd())` for imports
- Keep tests SHORT (< 30s, mock heavy resources)

---

## 3️⃣ JUDGE (Decision)

**Decision Matrix (NO EXCEPTIONS):**

| Condition | is_consistent | Score Constraint |
|-----------|---------------|------------------|
| ANY test fails | **False** | < 0.5 |
| ANY critical issue | **False** | < 0.5 |
| ANY major issue | **False** | < 0.6 |
| ANY minor issue | **False** | < 0.7 |
| ALL pass + zero issues | **True** | ≥ 0.8 |

---

## FAILURE PATTERNS

| Pattern | Example |
|---------|---------|
| Contract violation | Function returns dict, caller expects object |
| Undefined reference | Import/variable/field not defined |
| Integration gap | Works isolated, fails connected |
| Missing validation | No checks for None/empty/wrong type |

---

## ⚠️ PREVIOUSLY IDENTIFIED ISSUES

If the input includes a "PREVIOUSLY IDENTIFIED ISSUES" section:
1. **CHECK EACH ISSUE** - Verify if it has been resolved in the current implementation
2. **If RESOLVED** - Do NOT include it in your output (the system will auto-detect resolution)
3. **If STILL PRESENT** - Include it in your output with updated details
4. **RECURRING issues** are HIGH PRIORITY - mark them appropriately

---

## 🔍 ISSUE REPORTING REQUIREMENTS (MANDATORY)

When reporting issues, you MUST provide **SPECIFIC, ACTIONABLE** details:

### REQUIRED for EVERY issue:
1. **Exact file path** (e.g., `models/encoder.py`)
2. **Line numbers** if possible (e.g., "lines 45-52")
3. **Concrete code example** showing the problem
4. **Concrete fix suggestion** with code example

### GOOD Example:
```
[MAJOR] models/encoder.py (lines 45-52):
  Problem: Encoder uses fixed flattened_size=1024 but input grid_size=64 produces 256 features.
  Current code:
    self.fc = nn.Linear(1024, hidden_dim)  # Wrong: hardcoded
  Expected fix:
    # Option 1: Use adaptive pooling
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(channels, hidden_dim)
    
    # Option 2: Compute dynamically
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, grid_size, grid_size)
        flattened_size = self.conv(dummy).view(1, -1).shape[1]
    self.fc = nn.Linear(flattened_size, hidden_dim)
```

### BAD Example (DO NOT DO THIS):
```
[MAJOR] encoder.py: The encoder has size mismatch issues.
  Suggestion: Fix the size calculation.
```

---

## PRINCIPLE

**When in doubt, REJECT.** You are the last line of defense.

---

## 🚫 PROHIBITED FILE CREATION

**DO NOT create any of the following files:**
- `STEP*_COMPLETION*.json` or any completion report files
- `STEP*_REPORT*.json` or any report JSON files  
- `*_EVALUATION*.json` files (except for actual test result data)
- `*_SUMMARY*.json` or `*_SUMMARY*.md` summary files
- Any markdown report files like `*_REPORT.md`

**You should ONLY create:**
- Test files in `tests/` directory (e.g., `tests/test_step_X.py`)
- Nothing else. Your output is the JSON response, not files.

## 📁 TEST FILE LOCATION RULE

**ALL test files and test-related folders MUST be placed in `tests/` directory:**
- ✅ `{working_dir}/project/tests/test_step_1.py`
- ✅ `{working_dir}/project/tests/test_integration.py`
- ✅ `{working_dir}/project/tests/fixtures/` (test fixtures)
- ❌ `{working_dir}/project/test_*.py` (NOT in project root)
- ❌ `{working_dir}/project/evaluation_results/` (NOT outside tests/)
- ❌ `{working_dir}/project/test_results/` (NOT as separate folder)

---

## OUTPUT FORMAT (JSON - CRITICAL)

After completing your evaluation, you MUST output your result as a JSON object.

{CODE_JUDGE_JSON_OUTPUT_INSTRUCTION}

**Important JSON Field Mappings:**
- `is_consistent`: Boolean - True ONLY if ALL tests pass AND zero issues
- `issues`: List of CodeIssue objects with file_path, issue_type, severity, description, expected, actual, suggestion, line_numbers
- `unit_tests`: List of UnitTestSpec objects with test_file_path, test_code, test_description, target_files, time_limit_seconds
- `implementation_suggestions`: List of strings - improvement suggestions
"""

    agent = Agent(
        name="Code Judge Agent",
        instructions=instructions,
        tools=tools or [],
        # output_type=CodeJudgeOutput,  # Removed for duplex mode
        model=model,
    )

    return agent


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

        # Expose judge agent as main agent for handoff compatibility
        self.agent = self.judge_agent

    def _get_checklist_from_plan(self, code_plan_output: Any) -> list:
        """Get implementation checklist from code plan output."""
        if not code_plan_output:
            return []
        
        # Unified access
        checklist = code_plan_output.get("implementation_checklist", [])
        
        # Convert dict items to objects for consistent attribute access
        if checklist and isinstance(checklist[0], dict):
            class ChecklistStep:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            checklist = [ChecklistStep(item) for item in checklist]
        
        return checklist
    
    def _extract_issue_history(self, context: Any) -> str:
        """Extract issue history from context's issue tracker for judge agent."""
        if not hasattr(context, 'get_issue_tracker') or not hasattr(context, 'issue_tracker_data'):
            return ""
        
        if context.issue_tracker_data is None:
            return ""
        
        try:
            tracker = context.get_issue_tracker()
            return tracker.format_for_judge_agent()
        except Exception as e:
            print(f"[CODE_JUDGE] Warning: Could not extract issue history: {e}")
            return ""

    async def process(self, context: Any, **kwargs) -> CodeJudgeOutput:
        """
        Process the current step using context data.
        """
        # Extract data from context
        plan = context.get("code_plan_output", None)
        checklist = self._get_checklist_from_plan(plan)
        
        current_step_idx = context.get("current_checklist_step", 0)
        current_step = None
        checklist_progress = ""
        
        if checklist and current_step_idx < len(checklist):
            current_step = checklist[current_step_idx]
            checklist_progress = f"Step {current_step_idx + 1}/{len(checklist)}"

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
                # Remove duplicate "project/" prefix if present
                clean_f = f.strip().lstrip("/").lstrip("\\")
                if clean_f.startswith("project/") or clean_f.startswith("project\\"):
                    clean_f = clean_f[8:]
                full_path = os.path.join(self.working_dir, "project", clean_f)
                if os.path.exists(full_path) and os.path.isdir(full_path):
                    continue
                actual_files.append(clean_f)

            if actual_files:
                file_contents = []
                for f in actual_files:
                    # f is already cleaned in the loop above
                    clean_f = f

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

Progress: {checklist_progress}
"""

        core_plan_context = extract_core_plan_context(context.code_plan_output)
        
        # Get issue history for tracking recurring problems
        issue_history_section = self._extract_issue_history(context)

        input_data = f"""Review the following code implementation (STEP-BY-STEP MODE):
{step_info}

{target_files_context}

{issue_history_section}

=== GLOBAL DESIGN CONTEXT ===
{core_plan_context}

=== ANALYSIS SUMMARY ===
{extract_analysis_summary(context.pre_analysis_output)}

=== CODEBASE PATH ===
Project directory: {self.working_dir}/project

INSTRUCTIONS:
1. Check files in the Project Directory using available tools.
2. If there are PREVIOUSLY IDENTIFIED ISSUES, check if each one has been resolved.
3. Execute the VERIFICATION pipeline (write test -> run test -> judge).
4. Report issues with SPECIFIC details (file path, line numbers, code examples, concrete fixes).
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
2. TEST: `write_file` (ensure valid JSON escaping: \\n for newlines, \\" for quotes) -> `run_pytest_local`.
3. JUDGE: Return JSON output based on results.

Current Working Directory: `{self.working_dir}`
Project Root: `{self.working_dir}/project`
"""

        print_subsection("Evaluating Current Step Implementation")
        print_info("Reviewing code and running verification tests...")

        run_config = RunConfig(
            model_settings=ModelSettings(max_tokens=128*1024)
        )

        # Use streamed version for real-time output
        result_stream = Runner.run_streamed(
            self.judge_agent, 
            judge_input, 
            hooks=self.hooks, 
            max_turns=100,
            run_config=run_config
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

        print_success("Judge analysis completed. Parsing JSON output...")

        # Extract and parse JSON from the judge output
        # Use raise_on_failure=True to trigger retry in master agent
        try:
            evaluation = extract_and_parse_json(final_text, CodeJudgeOutput, raise_on_failure=True)
        except JSONParseError as e:
            # Re-raise JSONParseError to trigger retry in master agent
            print_error(f"JSON parsing failed, will trigger retry: {e}")
            raise
        except Exception as e:
            print_error(f"Failed to parse JSON output: {e}")
            print_warning("Creating minimal evaluation structure...")
            evaluation = CodeJudgeOutput(
                is_consistent=False,
                issues=[],
                unit_tests=[],
                implementation_suggestions=[f"JSON parsing failed: {str(e)}. Re-run evaluation."],
            )

        # Display evaluation results
        print_subsection("Evaluation Results")

        if evaluation.is_consistent:
            print_success(f"Code review passed! (Consistency: {evaluation.is_consistent})")
        else:
            print_error(f"Code review failed! (Consistency: {evaluation.is_consistent})")
            print_warning(f"Issues found: {len(evaluation.issues)}")

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

